"""
toc_engine.py — TOC-FIN klasifikācija, S apakšgrupas, harmonizētas krāsas un Excel eksports
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from toc_ingest import Leaf, leaf_amounts

FONT_NAME = "Calibri"
FONT_SIZE = 10
KPI_AVG_MODE = "mean_of_months"

# Grupu definīcijas: kods -> (Cilvēcīgais apakšgrupas nosaukums atskaitē, Lielā grupa)
GROUPS: dict[str, tuple[str, str]] = {
    "S_REV": ("Neto apgrozījums (pamatdarbība)", "S"),
    "S_OTHER": ("Granti un pārējie ieņēmumi", "S"),
    "TVC_MATERIALS": ("Izejvielas un materiāli", "TVC"),
    "TVC_DELIVERY": ("Preču piegādes izdevumi", "TVC"),
    "OE_ADMIN_WAGES": ("Administrācijas algas (+ sociālās iemaksas)", "OE"),
    "OE_WORKER_WAGES": ("Strādnieku algas (+ sociālās iemaksas)", "OE"),
    "OE_MARKETING": ("Mārketings un reklāma", "OE"),
    "OE_PLATFORMS": ("Pārdošanas platformas un citas pārdošanas izm.", "OE"),
    "OE_PREMISES": ("Ražošanas telpu noma un uzturēšana", "OE"),
    "OE_OFFICE_IT": ("Biroja, IT un sakaru izmaksas", "OE"),
    "OE_FIN_TAX": ("Procentu maksājumi un nodokļi (UIN)", "OE"),
    "OE_OTHER": ("Citi saimnieciskie uzturēšanas izdevumi", "OE"),
    "EXCLUDED": ("Izslēgtie (Amortizācija / Virtuālā nauda)", "EXCL"),
}

SUBGROUP_LABELS: dict[str, str] = {k: v[0] for k, v in GROUPS.items()}
LABEL_TO_KEY: dict[str, str] = {v[0]: k for k, v in GROUPS.items()}

MAIN_GROUP_TO_DEFAULT_SUB: dict[str, str] = {
    "🟡 S (Ieņēmumi)": "S_REV",
    "🟠 TVC (Mainīgās izmaksas)": "TVC_MATERIALS",
    "🟤 OE (Uzturēšanas izdevumi)": "OE_OTHER",
    "⚪ Izslēgtie (Virtuālā nauda)": "EXCLUDED",
}

S_KEYS = [k for k, v in GROUPS.items() if v[1] == "S"]
TVC_KEYS = [k for k, v in GROUPS.items() if v[1] == "TVC"]
OE_KEYS = [k for k, v in GROUPS.items() if v[1] == "OE"]

KNOWLEDGE_TEXT = ""
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    p = Path(fname)
    if p.exists():
        KNOWLEDGE_TEXT = f"\n\nTOC-FIN ZINĀŠANU BĀZE UN VADLĪNIJAS:\n{p.read_text(encoding='utf-8')}\n"
        break

CLASSIFY_SYSTEM = f"""Tu esi TOC-FIN (Throughput Accounting) vadības un finanšu eksperts.
Tavs uzdevums ir kritiski un pēc būtības izvērtēt katru PZA kontu (pēc tā nosaukuma un atrašanās sadaļas) un piešķirt tam atbilstošo TOC apakšgrupu.
{KNOWLEDGE_TEXT}
VADLĪNIJAS SITUATĪVAI IZVĒRTĒŠANAI:
1. Izproti ieņēmumu (S) sadalījumu:
   - Pamatdarbības preču un pakalpojumu pārdošana ('Neto apgrozījums') -> S_REV (is_income=true).
   - Saņemts granta finansējums, valsts atbalsts un 'Pārējie ieņēmumi' -> S_OTHER (is_income=true). Ja kādā situācijā ir šaubas par pārējiem ieņēmumiem, liec S_OTHER un vari uzdot jautājumu (confidence='low').
2. Stingri nodali TVC apakšgrupas:
   - 'Izejvielu un materiālu iepirkšanas un piegādes izd.' (vai izejvielas/materiāli ražošanā) -> OBLIGĀTI TVC_MATERIALS (confidence='high', neliec to pie TVC_DELIVERY!).
   - 'Preču piegādes izdevumi' (pārdošanas piegāde klientiem) -> TVC_DELIVERY (confidence='high').
3. Izproti algas un OE pēc konteksta (Sadaļas/Ceļa):
   - Ja 'Personāla izmaksas' un 'Sociālās iemaksas' atrodas zem ražošanas pašizmaksas sadaļas -> OE_WORKER_WAGES.
   - Ja tās atrodas zem administrācijas vai pārdošanas sadaļas -> OE_ADMIN_WAGES.
   - Reālie naudas izdevumi — procentu maksājumi un uzņēmumu ienākuma nodoklis (UIN) -> OE_FIN_TAX.
4. Izslēgtie (EXCLUDED):
   - Tikai virtuālie grāmatvedības ieraksti bez naudas plūsmas: pamatlīdzekļu nolietojums/amortizācija, krājumu pārvērtēšana, uzkrājumi."""

CLASSIFY_TOOL = {
    "name": "classify_accounts",
    "description": "Assign a TOC group to every account based on situational TOC logic. Every id must appear exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "group": {"enum": list(GROUPS)},
                        "is_income": {"type": "boolean"},
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "question": {"type": ["string", "null"]},
                    },
                    "required": ["id", "group", "is_income", "confidence"],
                },
            }
        },
        "required": ["items"],
    },
}


def leaves_for_llm(leaves: list[Leaf]) -> str:
    rows = [{"id": l.id, "path": l.path, "label": l.label} for l in leaves]
    return json.dumps(rows, ensure_ascii=False)


def validate_classification(items: list[dict], leaves: list[Leaf]) -> list[str]:
    errs = []
    ids = [i["id"] for i in items]
    want = {l.id for l in leaves}
    missing = sorted(want - set(ids))
    extra = sorted(set(ids) - want)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if missing:
        errs.append(f"Missing ids: {missing}")
    if extra:
        errs.append(f"Unknown ids: {extra}")
    if dup:
        errs.append(f"Duplicate ids: {dup}")
    bad = [i["id"] for i in items if i.get("group") not in GROUPS]
    if bad:
        errs.append(f"Invalid group for: {bad}")
    return errs


def classify_with_claude(client, model: str, leaves: list[Leaf], max_tries: int = 3) -> list[dict]:
    msgs = [{"role": "user", "content": "Izvērtē un klasificē šīs PZA lapas:\n" + leaves_for_llm(leaves)}]
    items: list[dict] = []
    for _ in range(max_tries):
        resp = client.messages.create(
            model=model, max_tokens=8000, system=CLASSIFY_SYSTEM,
            tools=[CLASSIFY_TOOL], tool_choice={"type": "tool", "name": CLASSIFY_TOOL["name"]},
            messages=msgs,
        )
        tu = next(b for b in resp.content if b.type == "tool_use")
        items = tu.input.get("items", [])
        errs = validate_classification(items, leaves)
        if not errs:
            return items
        msgs += [{"role": "assistant", "content": resp.content},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                               "content": "Fix and resend ALL items. " + " ".join(errs)}]}]
    raise ValueError("Klasifikācija nav pilnīga pēc vairākiem mēģinājumiem: " + " ".join(errs))


@dataclass
class TocResult:
    main: pd.DataFrame
    kpi: pd.DataFrame
    row_kinds: dict[str, str]
    periods: list[str]
    amounts: pd.DataFrame
    checks: list[dict]


def compute_toc(leaves: list[Leaf], mapping: dict[str, dict], period_cols: list[dict],
                reported_result: np.ndarray | None = None) -> TocResult:
    # Atbalsts arī vecajam 'S' kodam no profila, pārvēršot par 'S_REV'
    norm_map = {}
    for k, m in mapping.items():
        g = m["group"]
        if g == "S":
            g = "S_REV"
        norm_map[k] = {"group": g, "is_income": bool(m.get("is_income") or g in S_KEYS)}

    income_ids = {i for i, m in norm_map.items() if m["is_income"]}
    A, _ = leaf_amounts(leaves, income_ids)
    ids = [l.id for l in leaves]
    groups = np.array([norm_map[i]["group"] for i in ids])

    labels = [p["label"] for p in period_cols]
    month_idx = [j for j, p in enumerate(period_cols) if p["kind"] == "month"]

    def gsum(g: str) -> np.ndarray:
        m = groups == g
        return A[m].sum(axis=0) if m.any() else np.zeros(A.shape[1])

    rows: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}

    # 1. S un S apakšgrupas
    s_rows = {}
    for k in S_KEYS:
        if (groups == k).any():
            s_rows[k] = gsum(k)
    S_tot = sum(s_rows.values()) if s_rows else np.zeros(A.shape[1])
    rows["S"] = S_tot
    kinds["S"] = "S"
    for k, v in s_rows.items():
        rows[GROUPS[k][0]] = v
        kinds[GROUPS[k][0]] = "S_CHILD"

    # 2. TVC un TVC apakšgrupas
    tvc_rows = {}
    for k in TVC_KEYS:
        if (groups == k).any():
            tvc_rows[k] = -gsum(k)
    TVC = sum(tvc_rows.values()) if tvc_rows else np.zeros_like(S_tot)
    rows["TVC"] = TVC
    kinds["TVC"] = "TVC"
    for k, v in tvc_rows.items():
        rows[GROUPS[k][0]] = v
        kinds[GROUPS[k][0]] = "TVC_CHILD"

    # 3. T (Throughput)
    T = S_tot - TVC
    rows["T"] = T
    kinds["T"] = "T"

    # 4. OE un OE apakšgrupas
    oe_rows = {k: -gsum(k) for k in OE_KEYS if (groups == k).any()}
    OE = sum(oe_rows.values()) if oe_rows else np.zeros_like(S_tot)
    rows["OE"] = OE
    kinds["OE"] = "OE"
    for k, v in oe_rows.items():
        rows[GROUPS[k][0]] = v
        kinds[GROUPS[k][0]] = "OE_CHILD"

    # 5. NP (Net Profit)
    NP = T - OE
    rows["NP"] = NP
    kinds["NP"] = "NP"

    # 6. Izslēgtie (ja ir)
    EXCL = -gsum("EXCLUDED")
    if (groups == "EXCLUDED").any():
        rows[GROUPS["EXCLUDED"][0]] = EXCL
        kinds[GROUPS["EXCLUDED"][0]] = "EXCL"

    base = pd.DataFrame(rows, index=labels).T
    if len(month_idx) >= 2:
        mcols = [labels[j] for j in month_idx]
        year = _year_from_labels(mcols)
        base[f"{year} YTD"] = base[mcols].sum(axis=1)
        base[f"{year} AVG YTD"] = base[mcols].mean(axis=1)
    amounts = base.copy()

    main = {}
    for c in amounts.columns:
        main[c] = amounts[c]
        s = amounts.loc["S", c]
        main[f"%|{c}"] = amounts[c] / s if s else np.nan
    main = pd.DataFrame(main)

    def kpi_for(col_vals: pd.Series) -> dict[str, float]:
        S_, T_, OE_ = col_vals["S"], col_vals["T"], col_vals["OE"]
        aw = col_vals.get(GROUPS["OE_ADMIN_WAGES"][0], 0.0)
        ww = col_vals.get(GROUPS["OE_WORKER_WAGES"][0], 0.0)
        div = lambda a, b: a / b if b else np.nan
        return {
            "PR = T/OE": div(T_, OE_),
            "T% = T/S": div(T_, S_),
            "Eiro Sales / Eiro algas kopā": div(S_, aw + ww),
            "Eiro Sales / Eiro administrācijas algas": div(S_, aw),
            "Eiro Sales / Eiro strādnieku algas": div(S_, ww),
        }

    kpi = pd.DataFrame({c: kpi_for(amounts[c]) for c in amounts.columns})
    if len(month_idx) >= 2 and KPI_AVG_MODE == "mean_of_months":
        mcols = [labels[j] for j in month_idx]
        kpi[f"{_year_from_labels(mcols)} AVG YTD"] = kpi[mcols].mean(axis=1)

    checks = []
    tot_ok = np.allclose(NP - EXCL, A.sum(axis=0), atol=1.0)
    checks.append({"Pārbaude": "NP (TOC) − izslēgtās izmaksas = Σ visu lapu", "Statuss": "OK" if tot_ok else "KĻŪDA",
                   "Detaļas": "Katra lapa ir tieši vienā grupā."})
    if reported_result is not None:
        diff = np.abs(NP - EXCL - reported_result).max()
        checks.append({"Pārbaude": "NP (TOC) − izslēgtās izmaksas = faila gala rezultāts",
                       "Statuss": "OK" if diff <= 3.0 else "KĻŪDA",
                       "Detaļas": f"Lielākā starpība: {diff:,.2f}"})
    return TocResult(main, kpi, kinds, list(amounts.columns), amounts, checks)


def _year_from_labels(labels: list[str]) -> str:
    m = re.search(r"(20\d\d)", labels[0])
    return m.group(1) if m else ""


# Harmonizētā krāsu palete:
# S = Koši dzeltens (FFFF00), S_CHILD = Gaiši dzeltens (FFF2CC)
# TVC = Tumši sarkanbrūns/rūsas (C55A11), TVC_CHILD = Gaišs terakotas/oranžs (F8CBAD)
# T = Spilgti zils (00B0F0)
# OE = Silti brūnīgs smilšu/persiku (F4B084), OE_CHILD = Gaiši brūnīgs persiku (FCE4D6)
# NP = Spilgti zaļš (92D050)
FILLS = {
    "S": ("FFFF00", True, "000000"),
    "S_CHILD": ("FFF2CC", False, "000000"),
    "TVC": ("C55A11", True, "FFFFFF"),
    "TVC_CHILD": ("F8CBAD", False, "000000"),
    "T": ("00B0F0", True, "000000"),
    "OE": ("F4B084", True, "000000"),
    "OE_CHILD": ("FCE4D6", False, "000000"),
    "NP": ("92D050", True, "000000"),
    "EXCL": ("D9D9D9", False, "595959"),
}
THIN = Side(style="thin", color="BFBFBF")
MEDIUM = Side(style="medium", color="404040")


def build_excel(res: TocResult, path_or_buf, formulas: bool = True) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "TOC Diagnostika"
    fnt = lambda **kw: Font(name=FONT_NAME, size=FONT_SIZE, **kw)
    amounts = res.amounts
    periods = list(amounts.columns)
    row_names = list(amounts.index)
    kinds = [res.row_kinds.get(rn, "OE_CHILD") for rn in row_names]

    ws.cell(1, 1, "Rādītājs / Kontugrupa").font = fnt(bold=True)
    ws.cell(1, 1).border = Border(left=MEDIUM, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)
    col_of: dict[str, int] = {}
    c = 2
    for p in periods:
        col_of[p] = c
        c_val = ws.cell(1, c, p)
        c_pct = ws.cell(1, c + 1, "%")
        c_val.font = fnt(bold=True)
        c_pct.font = fnt(bold=True)
        c_val.border = Border(left=THIN, right=THIN, top=MEDIUM, bottom=MEDIUM)
        c_pct.border = Border(left=THIN, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)
        c += 2
    for cc in range(1, c):
        ws.cell(1, cc).alignment = Alignment(horizontal="center", vertical="center")

    r_of = {name: i + 2 for i, name in enumerate(row_names)}
    months = [p for p in periods if re.match(r"^\d{2}\.\d{4}", p)]
    ytd = next((p for p in periods if p.endswith(" YTD") and "AVG" not in p), None)
    avg = next((p for p in periods if "AVG" in p), None)
    L = get_column_letter
    child_s = [n for n in row_names if res.row_kinds[n] == "S_CHILD"]
    child_tvc = [n for n in row_names if res.row_kinds[n] == "TVC_CHILD"]
    child_oe = [n for n in row_names if res.row_kinds[n] == "OE_CHILD"]

    for i, name in enumerate(row_names):
        r = r_of[name]
        kind = kinds[i]
        next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
        fill, bold, color = FILLS.get(kind, FILLS["OE_CHILD"])

        thick_bottom = (
            kind in ("T", "NP")
            or (kind in ("S", "S_CHILD") and next_kind not in ("S", "S_CHILD"))
            or (kind in ("TVC", "TVC_CHILD") and next_kind not in ("TVC", "TVC_CHILD"))
            or (kind in ("OE", "OE_CHILD") and next_kind not in ("OE", "OE_CHILD"))
            or i == len(row_names) - 1
        )
        b_bot = MEDIUM if thick_bottom else THIN

        lbl_cell = ws.cell(r, 1, name if bold else f"   {name}")
        lbl_cell.fill = PatternFill("solid", start_color=fill, end_color=fill)
        lbl_cell.font = fnt(bold=bold, color=color)
        lbl_cell.border = Border(left=MEDIUM, right=MEDIUM, top=THIN, bottom=b_bot)

        for p in periods:
            cc = col_of[p]
            a, pc = ws.cell(r, cc), ws.cell(r, cc + 1)
            col = L(cc)
            is_ytd, is_avg = p == ytd, p == avg
            val = float(round(amounts.loc[name, p], 2))
            if is_ytd:
                f = "=" + "+".join(f"{L(col_of[m])}{r}" for m in months)
            elif is_avg:
                f = f"=AVERAGE({','.join(f'{L(col_of[m])}{r}' for m in months)})"
            elif kind == "S" and child_s:
                f = f"=SUM({col}{r_of[child_s[0]]}:{col}{r_of[child_s[-1]]})"
            elif kind == "TVC" and child_tvc:
                f = f"=SUM({col}{r_of[child_tvc[0]]}:{col}{r_of[child_tvc[-1]]})"
            elif kind == "T":
                f = f"={col}{r_of['S']}-{col}{r_of['TVC']}"
            elif kind == "OE" and child_oe:
                f = f"=SUM({col}{r_of[child_oe[0]]}:{col}{r_of[child_oe[-1]]})"
            elif kind == "NP":
                f = f"={col}{r_of['T']}-{col}{r_of['OE']}"
            else:
                f = val
            a.value = f if formulas else val
            pv = res.main.loc[name, f"%|{p}"]
            pc.value = (f"=IF({col}${r_of['S']}=0,\"\",{col}{r}/{col}${r_of['S']})" if formulas
                        else (None if pd.isna(pv) else float(pv)))
            a.number_format = '#,##0;-#,##0;"-"'
            pc.number_format = "0%"
            a.alignment = Alignment(horizontal="right")
            pc.alignment = Alignment(horizontal="right")

            for cell_item, right_b in ((a, THIN), (pc, MEDIUM)):
                cell_item.fill = PatternFill("solid", start_color=fill, end_color=fill)
                cell_item.font = fnt(bold=bold, color=color)
                cell_item.border = Border(left=THIN, right=right_b, top=THIN, bottom=b_bot)

    # KPI novietoti precīzi zem tām pašām mēnešu kolonnām (mēnesis pret mēnesi)
    k0 = len(row_names) + 3
    ws.cell(k0, 1, "Produktivitātes un atdeves rādītājs").font = fnt(bold=True)
    ws.cell(k0, 1).border = Border(left=MEDIUM, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)
    for p in periods:
        cc = col_of[p]
        ws.merge_cells(start_row=k0, start_column=cc, end_row=k0, end_column=cc + 1)
        kc = ws.cell(k0, cc, p)
        kc.font = fnt(bold=True)
        kc.alignment = Alignment(horizontal="center")
        ws.cell(k0, cc).border = Border(left=THIN, right=THIN, top=MEDIUM, bottom=MEDIUM)
        ws.cell(k0, cc + 1).border = Border(left=THIN, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)

    S_, T_, OE_ = r_of["S"], r_of["T"], r_of["OE"]
    aw = r_of.get(GROUPS["OE_ADMIN_WAGES"][0])
    ww = r_of.get(GROUPS["OE_WORKER_WAGES"][0])
    kpi_names = list(res.kpi.index)
    for i, kn in enumerate(kpi_names, start=1):
        r = k0 + i
        b_bot = MEDIUM if i == len(kpi_names) else THIN
        lc = ws.cell(r, 1, kn)
        lc.fill = PatternFill("solid", start_color="F2F4F4", end_color="F2F4F4")
        lc.font = fnt(bold=True)
        lc.border = Border(left=MEDIUM, right=MEDIUM, top=THIN, bottom=b_bot)

        for p in periods:
            cc = col_of[p]
            ws.merge_cells(start_row=r, start_column=cc, end_row=r, end_column=cc + 1)
            cell = ws.cell(r, cc)
            kv = res.kpi.loc[kn, p]
            avg_of_months = (p == avg and KPI_AVG_MODE == "mean_of_months")
            if avg_of_months:
                f = f"=AVERAGE({','.join(f'{L(col_of[m])}{r}' for m in months)})"
            else:
                a = L(cc)
                den_w = "+".join(f"{a}{x}" for x in (aw, ww) if x)
                f = {
                    "PR = T/OE": f"=IF({a}{OE_}=0,\"\",{a}{T_}/{a}{OE_})",
                    "T% = T/S": f"=IF({a}{S_}=0,\"\",{a}{T_}/{a}{S_})",
                    "Eiro Sales / Eiro algas kopā": f"=IF(({den_w or '0'})=0,\"\",{a}{S_}/({den_w or '1'}))" if den_w else "",
                    "Eiro Sales / Eiro administrācijas algas": f"=IF({a}{aw}=0,\"\",{a}{S_}/{a}{aw})" if aw else "",
                    "Eiro Sales / Eiro strādnieku algas": f"=IF({a}{ww}=0,\"\",{a}{S_}/{a}{ww})" if ww else "",
                }[kn]
            cell.value = (f or None) if formulas else (None if pd.isna(kv) else float(kv))
            cell.number_format = "0%" if kn.startswith("T%") else "0.00"
            cell.alignment = Alignment(horizontal="right")
            for sub_c, rb in ((cc, THIN), (cc + 1, MEDIUM)):
                sc = ws.cell(r, sub_c)
                sc.fill = PatternFill("solid", start_color="F2F4F4", end_color="F2F4F4")
                sc.font = fnt(bold=True)
                sc.border = Border(left=THIN, right=rb, top=THIN, bottom=b_bot)

    ws.column_dimensions["A"].width = 45
    for p in periods:
        cc = col_of[p]
        ws.column_dimensions[L(cc)].width = 11
        ws.column_dimensions[L(cc + 1)].width = 6
    ws.freeze_panes = "B2"
    ws.sheet_view.showGridLines = False
    wb.calculation.fullCalcOnLoad = True
    wb.save(path_or_buf)

"""
app.py — TOC-FIN Finanšu diagnostikas un vadības lēmumu sistēma (v17.0)
- Kompaktāks, čata veida asistentu logs ar īsām, konkrētām atbildēm
- Poga 'Ignorēt iepriekšējo atmiņu un klasificēt no jauna'
- Visi cipari tabulās stingri melnā krāsā
- Claude API tokenu un EUR izmaksu uzskaite katram klientam
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

from db import SupabaseProfileStore
from toc_engine import (
    GROUPS,
    SUBGROUP_LABELS,
    LABEL_TO_KEY,
    MAIN_GROUP_TO_DEFAULT_SUB,
    FILLS,
    build_excel,
    classify_with_claude,
    compute_toc,
)
from toc_ingest import (
    Structure,
    analyze,
    extract_lines,
    load_grids,
    pick_sheet,
    request_structure,
)

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="TOC-FIN Vadības Sistēma", layout="wide")


@dataclass
class LoadedReportResult:
    amounts: pd.DataFrame
    main: pd.DataFrame
    kpi: pd.DataFrame
    row_kinds: dict[str, str]
    reconciliation: pd.DataFrame
    by_group: dict


API_KEY = str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip().strip('"').strip("'")
if not API_KEY:
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit Secrets iestatījumos.")
    st.stop()


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=API_KEY)


def fmt_num(x) -> str:
    return "" if pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def fmt_pct_int(x) -> str:
    return "" if pd.isna(x) else f"{x * 100:.0f}%"


def _safe_line_stats(vals) -> tuple[float, float]:
    if vals is None:
        return 0.0, 0.0
    try:
        if isinstance(vals, dict):
            nums = [float(v) for v in vals.values() if v is not None and not pd.isna(v)]
        elif hasattr(vals, "values") and not isinstance(vals, (list, tuple, np.ndarray)):
            arr = np.asarray(vals.values, dtype=float)
            nums = [float(x) for x in arr if not np.isnan(x)]
        else:
            arr = np.asarray(vals, dtype=float).flatten()
            nums = [float(x) for x in arr if not np.isnan(x)]
        if not nums:
            return 0.0, 0.0
        return float(nums[0]), float(sum(nums))
    except Exception:
        return 0.0, 0.0


MONTH_NAMES_LV = {
    1: "Janvāris",
    2: "Februāris",
    3: "Marts",
    4: "Aprīlis",
    5: "Maijs",
    6: "Jūnijs",
    7: "Jūlijs",
    8: "Augusts",
    9: "Septembris",
    10: "Oktobris",
    11: "Novembris",
    12: "Decembris",
}

LV_MONTH_PARSE = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mai": 5, "may": 5,
    "jūn": 6, "jun": 6, "jūl": 7, "jul": 7, "aug": 8, "sep": 9,
    "okt": 10, "oct": 10, "nov": 11, "dec": 12,
}


def parse_period_month_year(col_name: str) -> tuple[int | None, int | None]:
    s = str(col_name or "").strip().lower()
    if "ytd" in s or "avg" in s or "kopā" in s or "izmaiņa" in s:
        return None, None

    m1 = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[\./\-](20\d{2})(?!\d)", s)
    if m1:
        return int(m1.group(1)), int(m1.group(2))

    m2 = re.search(r"(?<!\d)(20\d{2})[\./\-](0?[1-9]|1[0-2])(?!\d)", s)
    if m2:
        return int(m2.group(2)), int(m2.group(1))

    yr_m = re.search(r"(20\d{2})", s)
    yr = int(yr_m.group(1)) if yr_m else None
    for k, m_num in LV_MONTH_PARSE.items():
        if k in s:
            return m_num, yr

    return None, None


def compute_cumulative_toc_df(amounts_df: pd.DataFrame, fy_start_month: int = 1) -> tuple[pd.DataFrame, list[str], set[str]]:
    target_rows = [r for r in ["S", "TVC", "T", "NP"] if r in amounts_df.index]
    period_cols = [
        c for c in amounts_df.columns
        if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper() or "IZMAIŅA" in str(c).upper())
    ]
    if not period_cols:
        period_cols = list(amounts_df.columns)

    cum_data: dict[str, pd.Series] = {}
    running = pd.Series(0.0, index=target_rows)
    fy_boundary_cols: set[str] = set()

    for idx, col in enumerate(period_cols):
        m_num, _ = parse_period_month_year(str(col))
        if idx == 0:
            running = amounts_df.loc[target_rows, col].fillna(0.0).astype(float).copy()
            if m_num == fy_start_month:
                fy_boundary_cols.add(col)
        else:
            if m_num is not None and m_num == fy_start_month:
                running = amounts_df.loc[target_rows, col].fillna(0.0).astype(float).copy()
                fy_boundary_cols.add(col)
            else:
                running = running + amounts_df.loc[target_rows, col].fillna(0.0).astype(float)
        cum_data[col] = running.copy()

    cum_df = pd.DataFrame(cum_data, index=target_rows)
    row_labels = {
        "S": "S (uzkrājošais)",
        "TVC": "TVC (uzkrājošais)",
        "T": "T (uzkrājošais)",
        "NP": "NP (uzkrājošais)",
    }
    cum_df.index = [row_labels.get(r, r) for r in cum_df.index]
    return cum_df, period_cols, fy_boundary_cols


MAIN_GROUP_OPTIONS = list(MAIN_GROUP_TO_DEFAULT_SUB.keys())

RADIO_OPTIONS = [
    "S — Ieņēmumi (Sales)",
    "TVC — Patiesi mainīga izmaksa (1:1 ar pārdošanu)",
    "OE — Uzturēšanas / darbības izdevums",
    "Izslēgt — Virtuāla nauda (amortizācija)",
]


def normalize_group_key(group_key: str) -> str:
    g = str(group_key or "OE_OTHER").strip()
    if g == "S":
        return "S_REV"
    if g not in SUBGROUP_LABELS:
        return "OE_OTHER"
    return g


def get_main_code(group_key: str) -> str:
    g = normalize_group_key(group_key)
    return GROUPS.get(g, ("", "OE"))[1]


def get_main_cat(group_key: str) -> str:
    code = get_main_code(group_key)
    for opt in MAIN_GROUP_OPTIONS:
        if code == "S" and "S (" in opt:
            return opt
        if code == "TVC" and "TVC" in opt:
            return opt
        if code == "OE" and "OE" in opt:
            return opt
        if code == "EXCL" and "Izslēgt" in opt:
            return opt
    return MAIN_GROUP_OPTIONS[2] if len(MAIN_GROUP_OPTIONS) > 2 else ""


def get_radio_index_for_group(group_key: str) -> int:
    code = get_main_code(group_key)
    return {"S": 0, "TVC": 1, "OE": 2, "EXCL": 3}.get(code, 2)


def resolve_radio_choice(choice: str, original_group: str, konts_label: str) -> tuple[str, bool]:
    orig_grp = normalize_group_key(original_group)
    orig_code = get_main_code(orig_grp)
    lbl_low = str(konts_label or "").lower()

    if choice.startswith("S "):
        target_code = "S"
        fallback = "S_OTHER" if ("grant" in lbl_low or "citi" in lbl_low or "pārēj" in lbl_low) else "S_REV"
    elif choice.startswith("TVC"):
        target_code = "TVC"
        fallback = "TVC_DELIVERY" if ("piegād" in lbl_low or "transp" in lbl_low or "kurjer" in lbl_low) else "TVC_MATERIALS"
    elif choice.startswith("Izslēgt"):
        target_code = "EXCL"
        fallback = "EXCLUDED"
    else:
        target_code = "OE"
        fallback = "OE_OTHER"

    final_grp = orig_grp if orig_code == target_code else fallback
    return final_grp, final_grp.startswith("S_")


GROUP_SORT_ORDER = {
    "S": 1,
    "S_REV": 1,
    "S_OTHER": 2,
    "TVC_MATERIALS": 3,
    "TVC_DELIVERY": 4,
    "OE_ADMIN_WAGES": 5,
    "OE_WORKER_WAGES": 6,
    "OE_MARKETING": 7,
    "OE_PLATFORMS": 8,
    "OE_PREMISES": 9,
    "OE_OFFICE_IT": 10,
    "OE_FIN_TAX": 11,
    "OE_OTHER": 12,
    "EXCLUDED": 13,
}


def sort_cls_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Grupa"] = out["Grupa"].map(normalize_group_key)
    out["_sort"] = out["Grupa"].map(lambda g: GROUP_SORT_ORDER.get(g, 99))
    out = out.sort_values(by=["_sort", "Displeja_konts"], kind="mergesort").drop(columns=["_sort"]).reset_index(drop=True)
    return out


def build_display_labels(leaves) -> dict[str, str]:
    counts: dict[str, int] = {}
    for lf in leaves:
        lbl = str(lf.label).strip()
        counts[lbl] = counts.get(lbl, 0) + 1

    display_map: dict[str, str] = {}
    for lf in leaves:
        lbl = str(lf.label).strip()
        if counts.get(lbl, 0) > 1:
            path_parts = [p.strip() for p in str(getattr(lf, "path", "")).split(">") if p.strip()]
            parent = path_parts[-2] if len(path_parts) >= 2 else (path_parts[0] if path_parts else f"rinda {lf.row}")
            display_map[lf.id] = f"{lbl} [{parent}]"
        else:
            display_map[lf.id] = lbl
    return display_map


def render_unified_toc_tables_html(
    amounts_df: pd.DataFrame,
    main_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    row_kinds: dict[str, str],
    title_kpi: str = "Produktivitātes un atdeves rādītāji",
    grey_cols: set[str] | None = None,
    fy_start_month: int = 1,
    show_cumulative: bool = True,
) -> str:
    periods = list(amounts_df.columns)
    row_names = list(amounts_df.index)
    kinds = [row_kinds.get(rn, "OE_CHILD") for rn in row_names]
    grey_cols = grey_cols or set()

    html = ["""
    <style>
    .toc-table-wrap { width: 100%; overflow-x: auto; margin-bottom: 18px; }
    table.toc-unified { width: 100%; border-collapse: collapse; font-family: 'Calibri', 'Segoe UI', sans-serif; font-size: 12px; table-layout: auto; }
    table.toc-unified th { background-color: #EAECEE; color: #000000; font-weight: 700; text-align: center; padding: 6px 5px; border-top: 2px solid #404040; border-bottom: 2px solid #404040; white-space: nowrap; }
    table.toc-unified td { padding: 4px 6px; white-space: nowrap; color: #000000 !important; }
    table.toc-unified td.lbl-main { text-align: left; font-weight: 700; padding-left: 8px; padding-right: 10px; border-left: 2px solid #404040; border-right: 2px solid #404040; min-width: 210px; color: #000000 !important; }
    table.toc-unified td.lbl-sub { text-align: left; font-weight: 400; padding-left: 18px; padding-right: 10px; border-left: 2px solid #404040; border-right: 2px solid #404040; min-width: 210px; color: #000000 !important; }
    table.toc-unified td.val-col { text-align: right; border-left: 1px solid #BFBFBF; font-variant-numeric: tabular-nums; color: #000000 !important; }
    table.toc-unified td.pct-col { text-align: right; font-size: 11px; padding-left: 3px; padding-right: 5px; border-left: 1px dotted #D0D3D4; border-right: 2px solid #595959; color: #000000 !important; font-variant-numeric: tabular-nums; }
    table.toc-unified th.th-val { border-left: 1px solid #BFBFBF; }
    table.toc-unified th.th-pct { font-size: 11px; border-left: 1px dotted #BFBFBF; border-right: 2px solid #595959; }
    table.toc-unified th.th-lbl { text-align: left; padding-left: 8px; border-left: 2px solid #404040; border-right: 2px solid #404040; }
    table.toc-unified tr.spacer-row td { background: transparent !important; border: none !important; height: 28px; padding: 14px 0 6px 0; font-size: 15px; font-weight: 700; color: #1f2937 !important; }
    </style>
    <div class="toc-table-wrap"><table class="toc-unified">
    <thead><tr>
    """]
    html.append('<th class="th-lbl">Rādītājs / Kontugrupa</th>')
    for p in periods:
        th_bg = "background-color: #D5D8DC; color: #000000;" if p in grey_cols else ""
        html.append(f'<th class="th-val" style="{th_bg}">{p}</th><th class="th-pct" style="{th_bg}">%</th>')
    html.append("</tr></thead><tbody>")

    # --- 1. TABULA ---
    for i, rname in enumerate(row_names):
        kind = kinds[i]
        next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
        fill_hex, is_bold, _ = FILLS.get(kind, FILLS["OE_CHILD"])

        thick_bottom = (
            kind in ("T", "NP")
            or (kind in ("S", "S_CHILD") and next_kind not in ("S", "S_CHILD"))
            or (kind in ("TVC", "TVC_CHILD") and next_kind not in ("TVC", "TVC_CHILD"))
            or (kind in ("OE", "OE_CHILD") and next_kind not in ("OE", "OE_CHILD"))
            or i == len(row_names) - 1
        )
        border_bottom = "2.5px solid #404040" if thick_bottom else "1px solid #BFBFBF"
        font_w = "700" if is_bold else "400"
        lbl_cls = "lbl-main" if is_bold else "lbl-sub"

        # Visi teksti un cipari tabulās stingri melnā krāsā (#000000)
        row_style = (
            f"background-color: #{fill_hex}; color: #000000; font-weight: {font_w}; "
            f"border-top: 1px solid #BFBFBF; border-bottom: {border_bottom};"
        )
        html.append(f'<tr style="{row_style}"><td class="{lbl_cls}" title="{rname}" style="color: #000000;">{rname}</td>')

        for p in periods:
            val_str = fmt_num(main_df.loc[rname, p])
            pct_str = fmt_pct_int(main_df.loc[rname, f"%|{p}"])
            if p in grey_cols:
                g_bg = "#D5D8DC" if is_bold else "#EAEDED"
                cell_style = f"background-color: {g_bg}; color: #000000;"
            else:
                cell_style = "color: #000000;"
            html.append(f'<td class="val-col" style="{cell_style}">{val_str}</td><td class="pct-col" style="{cell_style}">{pct_str}</td>')
        html.append("</tr>")

    total_cols = 1 + 2 * len(periods)

    # --- 2. TABULA: KPI ---
    html.append(f'<tr class="spacer-row"><td colspan="{total_cols}">{title_kpi}</td></tr>')
    html.append("<tr>")
    html.append('<th class="th-lbl" style="background-color: #EAECEE; border-top: 2px solid #404040; border-bottom: 2px solid #404040;">Produktivitātes rādītājs</th>')
    for p in periods:
        th_bg = "#D5D8DC" if p in grey_cols else "#EAECEE"
        html.append(
            f'<th colspan="2" style="background-color: {th_bg}; color: #000000; border-top: 2px solid #404040; '
            f'border-bottom: 2px solid #404040; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959;">{p}</th>'
        )
    html.append("</tr>")

    for i, idx_name in enumerate(kpi_df.index):
        b_bot = "2.5px solid #404040" if i == len(kpi_df.index) - 1 else "1px solid #BFBFBF"
        html.append(f'<tr style="background-color: #F2F4F4; color: #000000; border-top: 1px solid #BFBFBF; border-bottom: {b_bot};">')
        html.append(f'<td class="lbl-main" style="font-weight: 600; color: #000000;" title="{idx_name}">{idx_name}</td>')
        for p in periods:
            v = kpi_df.loc[idx_name, p]
            if pd.isna(v):
                s = ""
            elif idx_name.startswith("T%"):
                s = f"{v * 100:.0f}%"
            else:
                s = f"{v:.2f}"
            kpi_cell_bg = "background-color: #E5E7E9; color: #000000;" if p in grey_cols else "color: #000000;"
            html.append(
                f'<td colspan="2" style="text-align: right; padding: 4px 6px; '
                f'border-left: 1px solid #BFBFBF; border-right: 2px solid #595959; font-weight: 600; {kpi_cell_bg}">{s}</td>'
            )
        html.append("</tr>")

    # --- 3. TABULA: UZKRĀJOŠIE ---
    if show_cumulative:
        cum_df, cum_periods, _ = compute_cumulative_toc_df(amounts_df, fy_start_month=fy_start_month)
        html.append(f'<tr class="spacer-row"><td colspan="{total_cols}">Uzkrājošie rādītāji</td></tr>')
        html.append("<tr>")
        html.append('<th class="th-lbl" style="background-color: #EAECEE; border-top: 2px solid #404040; border-bottom: 2px solid #404040;">Uzkrājošais rādītājs</th>')
        for p in periods:
            if p in cum_periods:
                html.append(f'<th class="th-val" style="background-color: #EAECEE; color: #000000;">{p}</th><th class="th-pct" style="background-color: #EAECEE; color: #000000;">%</th>')
            else:
                html.append(
                    f'<th colspan="2" style="background-color: #EAECEE; color: #000000; border-top: 2px solid #404040; '
                    f'border-bottom: 2px solid #404040; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959;">{p}</th>'
                )
        html.append("</tr>")

        s_row_label = cum_df.index[0] if len(cum_df.index) > 0 else None

        for idx_c, c_name in enumerate(cum_df.index):
            is_key_row = c_name.startswith("T ") or c_name.startswith("NP ")
            bg_c = "#E5E7E9" if is_key_row else "#F2F4F4"
            fw_c = "700" if is_key_row else "600"
            b_bot = "2.5px solid #404040" if (idx_c == len(cum_df.index) - 1 or is_key_row) else "1px solid #BFBFBF"

            html.append(
                f'<tr style="background-color: {bg_c}; color: #000000; font-weight: {fw_c}; '
                f'border-top: 1px solid #BFBFBF; border-bottom: {b_bot};">'
            )
            html.append(f'<td class="lbl-main" style="font-weight: {fw_c}; color: #000000;" title="{c_name}">{c_name}</td>')

            for p in periods:
                if p in cum_df.columns:
                    val_c = cum_df.loc[c_name, p]
                    s_c = cum_df.loc[s_row_label, p] if s_row_label else np.nan
                    pct_c = (val_c / s_c) if (s_c and not pd.isna(s_c) and s_c != 0) else np.nan
                    val_s = fmt_num(val_c)
                    pct_s = fmt_pct_int(pct_c)
                    val_color = "#922B21" if (not pd.isna(val_c) and val_c < 0 and c_name.startswith("NP")) else "#000000"
                    html.append(
                        f'<td class="val-col" style="color: {val_color}; font-weight: {fw_c};">{val_s}</td>'
                        f'<td class="pct-col" style="color: #000000; font-weight: {fw_c};">{pct_s}</td>'
                    )
                else:
                    last_p = cum_periods[-1] if cum_periods else None
                    if last_p and "YTD" in str(p).upper():
                        v_ytd = cum_df.loc[c_name, last_p]
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; color: #000000; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959; font-weight: {fw_c};">{fmt_num(v_ytd)}</td>'
                        )
                    elif last_p and "AVG" in str(p).upper() and len(cum_periods) > 0:
                        v_avg = cum_df.loc[c_name, last_p] / len(cum_periods)
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; color: #000000; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959; font-weight: {fw_c};">{fmt_num(v_avg)}</td>'
                        )
                    else:
                        html.append('<td colspan="2" style="border-left: 1px solid #BFBFBF; border-right: 2px solid #595959;"></td>')
            html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

S = st.session_state


# ------------------------------------------------------------------ ASISTENTS AR ĪSĀM, KONKRĒTĀM ATBILDĒM UN TOKENU UZSKAITI
def build_global_diagnostic_context() -> str:
    parts = [f"Aktīvais uzņēmuma profils: {S.get('active_profile', 'Nav')}"]
    parts.append(f"Finanšu gada sākuma mēnesis: {MONTH_NAMES_LV.get(S.get('fy_start_month', 1), 'Janvāris')}")

    if "structure" in S and S.get("structure") is not None:
        stc = S.get("structure")
        try:
            parts.append(f"\n--- 1. FAILA STRUKTŪRA ---\n{vars(stc)}")
        except Exception:
            pass

    if "analysis" in S and S.analysis is not None:
        an = S.analysis
        leaf_ids = {lf.id for lf in an.leaves}
        lines_info = []
        res_row = getattr(S.get("structure"), "result_row", None)
        for ln in an.ex.lines:
            status = "KONTS" if ln.id in leaf_ids else (
                "REZULTĀTA RINDA" if (res_row is not None and ln.row == res_row)
                else "IZLAISTS (STARPSUMMA / VIRSRAKSTS)"
            )
            first_val, tot_val = _safe_line_stats(getattr(ln, "vals", None))
            lines_info.append(
                f"Rinda #{ln.row} (id={ln.id}): '{ln.label}' | Sadaļa: '{getattr(ln, 'path', '')}' | "
                f"Statuss: {status} | 1. mēn.: {first_val:,.2f} | Kopā: {tot_val:,.2f}"
            )
        parts.append("\n--- 2. NOLASĪTĀS PZA RINDAS ---\n" + "\n".join(lines_info[:140]))

    if "cls_df" in S and S.cls_df is not None:
        df_c = S.cls_df[["id", "Konts", "Displeja_konts", "Grupa", "Ieņēmums"]]
        parts.append("\n--- 3. KONTU KLASIFIKĀCIJA ---\n" + df_c.to_markdown(index=False))

    if "result" in S and S.result is not None:
        res = S.result
        parts.append("\n--- 4. TOC-FIN TABULA (EUR) ---\n" + res.amounts.round(0).to_markdown())
        parts.append("\n--- 5. KPI ---\n" + res.kpi.round(3).to_markdown())

    return "\n\n".join(parts)


def ask_global_assistant(user_question: str) -> tuple[str, int, float]:
    """Atgriež (atbilde, izmantotie_tokeni, izmaksas_eur)."""
    ctx = build_global_diagnostic_context()
    know = f"\nTOC-FIN metodoloģija:\n{knowledge_file}\n" if knowledge_file else ""
    sys_prompt = (
        "Tu esi TOC-FIN sistēmas kodolīgs un profesionāls finanšu asistents. "
        "Atbildi ĻOTI ĪSI, KONKRĒTI un tieši uz lietotāja jautājumu. "
        "Ja tev nav precīzas atbildes vai dati nav skaidri, nesāc stāstīt garus vispārīgus tekstus — sniedz tieši konkrētus ieteikumus (piem., norādi uz kuru rindu vai sadaļu 'PZA rindu korekcija' vai 'Rediģēt kontus' jāskatās), vai pasaki, kas tieši jāpārbauda. "
        "Izvairies no liekiem ievadiem un stāstījumiem."
        f"{know}"
    )
    history = S.get("global_chat", [])
    messages = [
        {"role": "user", "content": f"Sistēmas konteksts:\n{ctx}"},
        {"role": "assistant", "content": "Sapratu."},
    ] + history + [{"role": "user", "content": user_question}]

    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=800,
        system=sys_prompt,
        messages=messages,
    )
    ans = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    
    # Rēķinām tokenus un izmaksas (Claude Sonnet 3.5/4 cena ~3$ / miljons input, ~15$ / miljons output -> vidēji 0.00001$ par tokenu)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 400) if usage else 400
    out_tok = getattr(usage, "output_tokens", 150) if usage else 150
    total_tok = in_tok + out_tok
    # Aptuvenā cena EUR (1 USD ~ 0.92 EUR)
    cost_eur = (in_tok * 0.000003 + out_tok * 0.000015) * 0.92
    return ans, total_tok, cost_eur


# ------------------------------------------------------------------ 1. LOGIN
if not S.get("authenticated", False):
    col_l1, col_l2, col_l3 = st.columns([1, 1.4, 1])
    with col_l2:
        st.title("TOC-FIN Vadības Sistēma")
        st.markdown("Ievadiet piešķirto uzņēmuma lietotājvārdu un paroli.")

        temp_store = SupabaseProfileStore()
        with st.form("login_form"):
            login_company = st.text_input("Lietotājvārds / Uzņēmums")
            login_password = st.text_input("Parole", type="password")
            login_submitted = st.form_submit_button("Ielogoties", type="primary", use_container_width=True)

            if login_submitted:
                ok, msg, is_adm = temp_store.authenticate(login_company, login_password)
                if ok:
                    S.authenticated = True
                    S.is_admin = is_adm
                    if is_adm:
                        S.active_profile = "admin"
                        S.view_mode = "admin_panel"
                    else:
                        S.active_profile = login_company.strip()
                        S.view_mode = "workspace"
                    st.rerun()
                else:
                    st.error(msg)
    st.stop()


# ------------------------------------------------------------------ 2. ADMIN PANELIS AR TOKENU UN EUR UZSKAITI
admin_store = SupabaseProfileStore()

if S.get("is_admin", False) and S.get("view_mode") == "admin_panel":
    top_a1, top_a2 = st.columns([5, 1])
    with top_a1:
        st.title("Administratora vadības panelis")
        st.caption("Klientu profili, paroles, limiti un Claude API tokenu / izmaksu uzskaite.")
    with top_a2:
        st.write("")
        if st.button("Iziet", use_container_width=True):
            S.clear()
            st.rerun()

    companies = admin_store.list_all_companies_admin()

    col_adm_left, col_adm_right = st.columns([1.8, 1.2])

    with col_adm_left:
        st.subheader(f"Reģistrētie klientu profili ({len(companies)})")
        if not companies:
            st.info("Vēl nav izveidots neviens klienta profils.")
        else:
            comp_df = pd.DataFrame(companies)[
                ["company_name", "is_active", "usage_count", "monthly_limit", "total_tokens", "total_cost_eur", "accounts_count", "reports_count", "notes"]
            ]
            comp_df.columns = ["Uzņēmums", "Aktīvs", "Izlietots", "Limits", "Tokeni kopā", "Izmaksas (€)", "Konti", "Atsk.", "Piezīmes"]
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("#### Piekļūt klienta profilam")
            c_names = [c["company_name"] for c in companies]
            sel_c1, sel_c2, sel_c3 = st.columns([2, 1, 1])
            with sel_c1:
                target_client = st.selectbox("Izvēlieties klientu", options=c_names, label_visibility="collapsed")
            with sel_c2:
                if st.button("Atvērt vidi", type="primary", use_container_width=True):
                    S.active_profile = target_client
                    S.view_mode = "workspace"
                    S.pop("result", None)
                    S.pop("cls_df", None)
                    S.pop("loaded_report_meta", None)
                    st.rerun()
            with sel_c3:
                if st.button("Dzēst", use_container_width=True):
                    S.confirm_del_client = target_client

            if S.get("confirm_del_client") == target_client:
                st.warning(f"Vai tiešām dzēst profilu **{target_client}**?")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Jā, dzēst", type="primary", use_container_width=True):
                        admin_store.delete_company_admin(target_client)
                        S.pop("confirm_del_client", None)
                        st.rerun()
                with dc2:
                    if st.button("Atcelt", use_container_width=True):
                        S.pop("confirm_del_client", None)
                        st.rerun()

    with col_adm_right:
        st.subheader("Izveidot vai rediģēt klientu")
        with st.form("admin_manage_client_form"):
            f_cname = st.text_input("Lietotājvārds / Uzņēmums", placeholder="piem., SIA Klients")
            f_pw = st.text_input("Parole (atstājiet tukšu, ja nemaināt)", type="password")
            f_limit = st.number_input("Mēneša limits", min_value=1, max_value=10000, value=50, step=5)
            f_active = st.checkbox("Profils ir aktīvs", value=True)
            f_reset = st.checkbox("Nullēt izlietoto skaitītāju un tokenus (0)", value=False)
            f_notes = st.text_input("Piezīmes")
            sub_adm = st.form_submit_button("Saglabāt profilu", type="primary", use_container_width=True)

            if sub_adm:
                ok_a, msg_a = admin_store.create_or_update_company_admin(
                    company_name=f_cname,
                    password=f_pw,
                    monthly_limit=int(f_limit),
                    is_active=f_active,
                    notes=f_notes,
                    reset_usage=f_reset,
                )
                if ok_a:
                    st.success(msg_a)
                    st.rerun()
                else:
                    st.error(msg_a)
    st.stop()


# ------------------------------------------------------------------ 3. SĀNJOSLA
profile = SupabaseProfileStore(company_name=S.active_profile)

if "fy_start_month" not in S:
    S.fy_start_month = 1

with st.sidebar:
    if S.get("is_admin", False):
        st.info(f"Admin skats: **{S.active_profile}**")
        if st.button("← Atpakaļ uz Admin paneli", use_container_width=True):
            S.view_mode = "admin_panel"
            S.pop("result", None)
            S.pop("cls_df", None)
            st.rerun()
    else:
        st.markdown(f"**Profils:** {S.active_profile}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Jauns aprēķins", use_container_width=True):
            S.view_mode = "workspace"
            S.pop("result", None)
            S.pop("cls_df", None)
            S.pop("loaded_report_meta", None)
            st.rerun()
    with col_s2:
        if st.button("Iziet", use_container_width=True):
            S.clear()
            st.rerun()

    saved_acc_count = profile.count_saved()
    if st.button(f"Rediģēt kontus ({saved_acc_count})", use_container_width=True):
        S.view_mode = "edit_accounts"
        st.rerun()

    saved_reports = profile.list_reports()
    with st.expander(f"Saglabātās atskaites ({len(saved_reports)})", expanded=False):
        if not saved_reports:
            st.caption("Nav saglabātu atskaišu.")
        else:
            rep_by_id = {r["id"]: r for r in saved_reports}
            chosen_rep_id = st.selectbox(
                "Atskaite",
                options=list(rep_by_id.keys()),
                format_func=lambda rid: f"{rep_by_id[rid].get('report_name')} ({str(rep_by_id[rid].get('created_at', ''))[:10]})",
                label_visibility="collapsed",
            )
            sel_rep = rep_by_id[chosen_rep_id]
            st.caption(
                f"**Nosaukums:** {sel_rep.get('report_name')}\n\n"
                f"**Periods:** {sel_rep.get('periods_label') or '—'}"
            )

            if st.button("Atvērt atskaiti", type="primary", use_container_width=True):
                loaded = profile.load_report(chosen_rep_id)
                if loaded:
                    S.result = LoadedReportResult(
                        amounts=loaded["amounts"],
                        main=loaded["main"],
                        kpi=loaded["kpi"],
                        row_kinds=loaded["row_kinds"],
                        reconciliation=loaded["reconciliation"],
                        by_group={},
                    )
                    S.loaded_report_meta = loaded
                    S.current_report_id = loaded["id"]
                    S.view_mode = "workspace"
                    if loaded.get("commentary"):
                        S.commentary = loaded["commentary"]
                    else:
                        S.pop("commentary", None)
                    st.rerun()

            if st.button("Dzēst atskaiti", use_container_width=True):
                S.confirm_delete_rep_id = chosen_rep_id

            if S.get("confirm_delete_rep_id") == chosen_rep_id:
                st.warning(f"Vai tiešām dzēst atskaiti **{sel_rep.get('report_name')}**?")
                cd1, cd2 = st.columns(2)
                with cd1:
                    if st.button("Jā, dzēst", type="primary", use_container_width=True):
                        if profile.delete_report(chosen_rep_id):
                            S.pop("confirm_delete_rep_id", None)
                            if S.get("current_report_id") == chosen_rep_id:
                                S.pop("result", None)
                                S.pop("loaded_report_meta", None)
                            st.rerun()
                with cd2:
                    if st.button("Atcelt", use_container_width=True):
                        S.pop("confirm_delete_rep_id", None)
                        st.rerun()

    with st.expander("Iestatījumi", expanded=False):
        selected_fy_month = st.selectbox(
            "Finanšu gada sākums",
            options=list(MONTH_NAMES_LV.keys()),
            format_func=lambda m: MONTH_NAMES_LV[m],
            index=list(MONTH_NAMES_LV.keys()).index(S.get("fy_start_month", 1)),
        )
        S.fy_start_month = selected_fy_month

        st.markdown("---")
        ok_q, q_info = profile.check_quota()
        if profile.is_connected:
            st.caption(f"Datubāze: pieslēgta ({saved_acc_count} konti)")
            if q_info:
                st.caption(f"Mēneša kvota: {q_info}")
        else:
            st.caption(f"Datubāze: nav savienojuma ({profile.init_error})")


# ------------------------------------------------------------------ VIRSRAKSTS UN ČATA VEIDA ASISTENTS
top_col1, top_col2 = st.columns([5, 1])
with top_col1:
    st.title("TOC-FIN: Finanšu diagnostika un vadības lēmumu sistēma")
with top_col2:
    st.write("")
    with st.popover("Asistents", use_container_width=True):
        st.markdown("**Asistents**")
        if "global_chat" not in S:
            S.global_chat = []

        # Kompaktāks čata izskats ar izceltām ziņām
        chat_box = st.container(height=320)
        with chat_box:
            if not S.global_chat:
                st.caption("Uzdodiet jautājumu...")
            for msg in S.global_chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        with st.form("side_agent_form", clear_on_submit=True):
            q_in = st.text_input("Jautājums", placeholder="Jautājums aģentam...", label_visibility="collapsed")
            f_c1, f_c2 = st.columns([3, 1])
            with f_c1:
                sub_q = st.form_submit_button("Sūtīt", type="primary", use_container_width=True)
            with f_c2:
                clr_q = st.form_submit_button("Notīrīt", use_container_width=True)

            if sub_q and q_in.strip():
                S.global_chat.append({"role": "user", "content": q_in.strip()})
                try:
                    ans, t_tok, t_eur = ask_global_assistant(q_in.strip())
                    profile.log_api_usage(t_tok, t_eur)
                except Exception as e:
                    ans = f"Kļūda: {e}"
                S.global_chat.append({"role": "assistant", "content": ans})
                st.rerun()
            if clr_q:
                S.global_chat = []
                st.rerun()

with st.expander("Metodoloģija un pamatjēdzieni"):
    st.markdown("""
    * **S (Sales / Ieņēmumi)** — Visa reālā nauda, kas ienāk uzņēmumā.
    * **TVC (Truly Variable Costs / Patiesi mainīgās izmaksas)** — Izdevumi, kas mainās 1:1 ar katru pārdoto vienību.
    * **T (Throughput / Caurlaidība)** — Jaunā naudas masa: $$T = S - TVC$$.
    * **OE (Operating Expenses / Uzturēšanas izdevumi)** — Kapacitātes un darbības nodrošināšanas izdevumi.
    * **NP (Net Profit / Neto peļņa)** — $$NP = T - OE$$.
    * **PR (Productivity Ratio / Produktivitātes rādītājs)** — $$PR = T / OE$$.
    """)


# ------------------------------------------------------------------ SKATS: TIEŠA KONTU REDIĢĒŠANA
if S.get("view_mode") == "edit_accounts":
    head_e1, head_e2 = st.columns([4, 1])
    with head_e1:
        st.header(f"Kontu klasifikācijas rediģēšana ({S.active_profile})")
    with head_e2:
        st.write("")
        if st.button("← Atpakaļ", use_container_width=True):
            S.view_mode = "workspace"
            st.rerun()

    if "cls_df" in S and S.cls_df is not None and "analysis" in S and S.analysis is not None:
        cls_df_edit: pd.DataFrame = S.cls_df.copy()
        cls_df_edit["Grupa"] = cls_df_edit["Grupa"].map(normalize_group_key)

        ed_view_df = pd.DataFrame(
            {
                "Konts": cls_df_edit["Displeja_konts"].tolist(),
                "Lielā grupa": [get_main_cat(g) for g in cls_df_edit["Grupa"]],
                "Apakšgrupa (Rinda atskaitē)": [SUBGROUP_LABELS[normalize_group_key(g)] for g in cls_df_edit["Grupa"]],
            }
        )
        ed_h = (len(ed_view_df) + 1) * 36 + 14
        edited_ses = st.data_editor(
            ed_view_df,
            hide_index=True,
            use_container_width=True,
            height=ed_h,
            disabled=["Konts"],
            key=f"direct_ses_editor_{S.get('editor_v', 0)}",
            column_config={
                "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
                "Lielā grupa": st.column_config.SelectboxColumn("Lielā grupa", options=MAIN_GROUP_OPTIONS, required=True, width="medium"),
                "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn("Apakšgrupa / Rinda atskaitē", options=list(SUBGROUP_LABELS.values()), required=True, width="large"),
            },
        )

        new_grps = []
        ch_flag = False
        for idx_p in range(len(cls_df_edit)):
            o_g = normalize_group_key(cls_df_edit.iloc[idx_p]["Grupa"])
            o_m = get_main_cat(o_g)
            o_s = SUBGROUP_LABELS[o_g]
            n_m = edited_ses.iloc[idx_p]["Lielā grupa"]
            n_s = edited_ses.iloc[idx_p]["Apakšgrupa (Rinda atskaitē)"]
            if n_s != o_s and n_s in LABEL_TO_KEY:
                new_grps.append(LABEL_TO_KEY[n_s])
                ch_flag = True
            elif n_m != o_m and n_m in MAIN_GROUP_TO_DEFAULT_SUB:
                new_grps.append(MAIN_GROUP_TO_DEFAULT_SUB[n_m])
                ch_flag = True
            else:
                new_grps.append(o_g)

        if ch_flag:
            cls_df_edit["Grupa"] = new_grps
            cls_df_edit["Ieņēmums"] = cls_df_edit["Grupa"].str.startswith("S_")
            S.cls_df = cls_df_edit
            S.editor_v = S.get("editor_v", 0) + 1
            st.rerun()

        if st.button("Saglabāt izmaiņas un pārrēķināt", type="primary"):
            an = S.analysis
            stc = S.structure
            cls_sorted = sort_cls_df(S.cls_df)
            S.cls_df = cls_sorted
            mapping = {
                r["id"]: {
                    "group": normalize_group_key(r["Grupa"]),
                    "is_income": bool(str(r["Grupa"]).startswith("S_") or r["Ieņēmums"]),
                }
                for _, r in cls_sorted.iterrows()
            }
            pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
            reported = an.ex.lines[pos].vals if pos is not None else None
            S.result = compute_toc(an.leaves, mapping, an.ex.period_cols, reported)
            profile.save(an.leaves, mapping)
            S.view_mode = "workspace"
            st.toast("Konti atjaunināti.")
            st.rerun()
    else:
        saved_maps = profile.list_saved_mappings()
        if not saved_maps:
            st.info("Nav saglabātu kontu.")
        else:
            db_df = pd.DataFrame(saved_maps)
            db_df["toc_group"] = db_df["toc_group"].map(normalize_group_key)
            db_view = pd.DataFrame(
                {
                    "id": db_df["id"],
                    "Sadaļa": db_df["path_label"],
                    "Konts": db_df["account_label"],
                    "Lielā grupa": [get_main_cat(g) for g in db_df["toc_group"]],
                    "Apakšgrupa (Rinda atskaitē)": [SUBGROUP_LABELS[normalize_group_key(g)] for g in db_df["toc_group"]],
                }
            )
            db_h = min((len(db_view) + 1) * 36 + 14, 750)
            edited_db = st.data_editor(
                db_view,
                hide_index=True,
                use_container_width=True,
                height=db_h,
                disabled=["id", "Sadaļa", "Konts"],
                key="db_mappings_editor",
                column_config={
                    "id": None,
                    "Sadaļa": st.column_config.TextColumn("Sadaļa", width="medium"),
                    "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
                    "Lielā grupa": st.column_config.SelectboxColumn("Lielā grupa", options=MAIN_GROUP_OPTIONS, required=True, width="medium"),
                    "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn("Apakšgrupa", options=list(SUBGROUP_LABELS.values()), required=True, width="large"),
                },
            )
            if st.button("Saglabāt izmaiņas", type="primary"):
                updates_to_save = []
                for idx_r in range(len(db_df)):
                    orig_g = normalize_group_key(db_df.iloc[idx_r]["toc_group"])
                    orig_m = get_main_cat(orig_g)
                    orig_s = SUBGROUP_LABELS[orig_g]
                    new_m = edited_db.iloc[idx_r]["Lielā grupa"]
                    new_s = edited_db.iloc[idx_r]["Apakšgrupa (Rinda atskaitē)"]
                    if new_s != orig_s and new_s in LABEL_TO_KEY:
                        final_g = LABEL_TO_KEY[new_s]
                    elif new_m != orig_m and new_m in MAIN_GROUP_TO_DEFAULT_SUB:
                        final_g = MAIN_GROUP_TO_DEFAULT_SUB[new_m]
                    else:
                        final_g = orig_g
                    updates_to_save.append({"id": int(db_df.iloc[idx_r]["id"]), "toc_group": final_g})
                if profile.update_saved_mappings_direct(updates_to_save):
                    st.success("Klasifikācija saglabāta.")
                    st.rerun()
                else:
                    st.error(f"Kļūda: {profile.last_error}")
    st.stop()


# ------------------------------------------------------------------ REZULTĀTU ATTELOŠANA
def render_results_and_simulation(res, report_title: str = ""):
    if report_title:
        st.caption(f"Atskaite: **{report_title}**")

    main_tab1, main_tab2 = st.tabs(["1. Finanšu diagnostika", "2. Scenāriju simulācija (%)"])

    with main_tab1:
        top_btn1, top_btn2, top_btn3 = st.columns([1.5, 1.5, 1])
        try:
            xl_f, xl_v = io.BytesIO(), io.BytesIO()
            build_excel(res, xl_f, formulas=True)
            build_excel(res, xl_v, formulas=False)
            top_btn1.download_button(
                "Lejupielādēt Excel (formulām)",
                xl_f.getvalue(),
                "TOC_Finansu_Diagnostika.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            top_btn2.download_button(
                "Lejupielādēt Excel (vērtībām)",
                xl_v.getvalue(),
                "TOC_Finansu_Diagnostika_vertibas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            xl_buf = io.BytesIO()
            with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                res.amounts.to_excel(writer, sheet_name="TOC_Summas")
                res.kpi.to_excel(writer, sheet_name="KPI")
            top_btn1.download_button(
                "Lejupielādēt Excel",
                xl_buf.getvalue(),
                "TOC_Saglabata_Atskaite.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with top_btn3:
            if st.button("Rediģēt kontus", use_container_width=True):
                S.view_mode = "edit_accounts"
                st.rerun()

        st.subheader("Finanšu pārskata tabula")
        st.markdown(
            render_unified_toc_tables_html(
                res.amounts,
                res.main,
                res.kpi,
                res.row_kinds,
                fy_start_month=S.get("fy_start_month", 1),
                show_cumulative=True,
            ),
            unsafe_allow_html=True,
        )

        if getattr(res, "reconciliation", None) is not None and not res.reconciliation.empty:
            with st.expander("Kontrole: Salīdzinājums ar PZA"):
                st.dataframe(res.reconciliation.style.format(fmt_num), use_container_width=True)

        st.markdown("---")
        st.subheader("Vadības kopsavilkums un secinājumi")
        cum_df_md, _, _ = compute_cumulative_toc_df(res.amounts, fy_start_month=S.get("fy_start_month", 1))
        tables_md = (
            res.amounts.round(0).to_markdown()
            + "\n\n"
            + res.kpi.round(3).to_markdown()
            + "\n\nUzkrājošie rādītāji:\n"
            + cum_df_md.round(0).to_markdown()
        )

        knowledge_context = f"TOC-FIN zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
        COMMENT_SYSTEM = (
            "Tu esi TOC-FIN finanšu konsultants. Raksti latviešu valodā, strukturēti, asi un uz lēmumiem vērsti."
            f"{knowledge_context}"
        )

        if "commentary" not in S or not S.commentary:
            if st.button("Ģenerēt analīzi", type="primary"):
                with st.spinner("Sagatavo analīzi..."):
                    r = get_client().messages.create(
                        model=MODEL,
                        max_tokens=2500,
                        system=COMMENT_SYSTEM,
                        messages=[{"role": "user", "content": f"Tabulas:\n\n{tables_md}\n\nUzraksti analīzi."}],
                    )
                    S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    if S.get("current_report_id"):
                        profile.update_report_commentary(S.current_report_id, S.commentary)
                    st.rerun()
        else:
            st.markdown(S.commentary)

    with main_tab2:
        st.subheader("Scenāriju simulācija")
        all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
        if not all_months:
            all_months = list(res.amounts.columns)

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            preset = st.selectbox(
                "Periods",
                options=[
                    f"Visi mēneši ({len(all_months)} mēn.)",
                    "Pirmie 6 mēneši" if len(all_months) >= 6 else "Pirmie 3",
                    "Manuāla izvēle",
                ],
            )
        if "6" in preset and len(all_months) >= 6:
            default_sel = all_months[:6]
        elif "3" in preset and len(all_months) >= 3:
            default_sel = all_months[:3]
        else:
            default_sel = all_months

        with p_col2:
            selected_months = st.multiselect("Mēneši", options=all_months, default=default_sel)

        if not selected_months:
            st.warning("Izvēlieties vismaz vienu mēnesi.")
            st.stop()

        st.markdown("---")
        st.markdown("#### Izmaiņas (%)")

        s_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "S_CHILD"]
        tvc_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "TVC_CHILD"]
        oe_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "OE_CHILD"]

        pct_changes: dict[str, float] = {}
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Ieņēmumi (S)**")
            for sc_name in s_children:
                pct_changes[sc_name] = st.number_input(f"{sc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{sc_name}")
        with sc2:
            st.markdown("**Mainīgās izmaksas (TVC)**")
            for tc_name in tvc_children:
                pct_changes[tc_name] = st.number_input(f"{tc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{tc_name}")
        with sc3:
            st.markdown("**Uzturēšanas izdevumi (OE)**")
            for oc_name in oe_children:
                def_v = 5.0 if "alga" in oc_name.lower() else 0.0
                pct_changes[oc_name] = st.number_input(f"{oc_name} (%)", value=def_v, step=1.0, format="%.1f", key=f"sim_{oc_name}")

        base_sub = res.amounts[selected_months].copy().astype(float)
        base_sum = base_sub.sum(axis=1)
        sim_sum = base_sum.copy()

        for rname, pct_val in pct_changes.items():
            if rname in sim_sum.index:
                sim_sum.loc[rname] = base_sum.loc[rname] * (1.0 + pct_val / 100.0)

        if s_children:
            sim_sum.loc["S"] = sim_sum.loc[s_children].sum()
        if tvc_children:
            sim_sum.loc["TVC"] = sim_sum.loc[tvc_children].sum()
        sim_sum.loc["T"] = sim_sum.loc["S"] - sim_sum.loc["TVC"]
        if oe_children:
            sim_sum.loc["OE"] = sim_sum.loc[oe_children].sum()
        sim_sum.loc["NP"] = sim_sum.loc["T"] - sim_sum.loc["OE"]

        diff_sum = sim_sum - base_sum

        summary_amounts = pd.DataFrame(
            {
                "Bāze KOPĀ": base_sum,
                "Simulācija KOPĀ": sim_sum,
                "Δ Izmaiņa (€)": diff_sum,
            },
            index=res.amounts.index,
        )

        sim_main = {}
        for c in summary_amounts.columns:
            sim_main[c] = summary_amounts[c]
            if c == "Δ Izmaiņa (€)":
                sim_main[f"%|{c}"] = np.where(base_sum != 0, diff_sum / np.abs(base_sum), np.nan)
            else:
                s_col = summary_amounts.loc["S", c]
                sim_main[f"%|{c}"] = summary_amounts[c] / s_col if s_col else np.nan
        sim_main_df = pd.DataFrame(sim_main, index=summary_amounts.index)

        def sim_kpi_for(col_vals: pd.Series) -> dict[str, float]:
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

        sim_kpi_df = pd.DataFrame(
            {
                "Bāze KOPĀ": sim_kpi_for(summary_amounts["Bāze KOPĀ"]),
                "Simulācija KOPĀ": sim_kpi_for(summary_amounts["Simulācija KOPĀ"]),
            }
        )
        sim_kpi_df["Δ Izmaiņa (€)"] = sim_kpi_df["Simulācija KOPĀ"] - sim_kpi_df["Bāze KOPĀ"]

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        ds_val, dt_val, doe_val, dnp_val = diff_sum["S"], diff_sum["T"], diff_sum["OE"], diff_sum["NP"]

        m1.metric("Simulētais S", f"{fmt_num(sim_sum['S'])} €", delta=f"{ds_val:+,.0f} €".replace(",", " "))
        m2.metric("Simulētais T", f"{fmt_num(sim_sum['T'])} €", delta=f"{dt_val:+,.0f} €".replace(",", " "))
        m3.metric("Simulētais OE", f"{fmt_num(sim_sum['OE'])} €", delta=f"{doe_val:+,.0f} €".replace(",", " "), delta_color="inverse")
        m4.metric("Simulētā Neto Peļņa (NP)", f"{fmt_num(sim_sum['NP'])} €", delta=f"{dnp_val:+,.0f} €".replace(",", " "))

        st.subheader(f"Simulētā tabula ({len(selected_months)} mēn.)")
        st.markdown(
            render_unified_toc_tables_html(
                summary_amounts,
                sim_main_df,
                sim_kpi_df,
                res.row_kinds,
                title_kpi="Simulētie produktivitātes rādītāji",
                grey_cols={"Bāze KOPĀ"},
                fy_start_month=S.get("fy_start_month", 1),
                show_cumulative=False,
            ),
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ DARBA PLŪSMA
uploaded = st.file_uploader("PZA fails (CSV vai Excel)", type=["csv", "xlsx", "xlsm", "xls"])

if uploaded is None:
    if "result" in S and S.get("loaded_report_meta"):
        meta = S.loaded_report_meta
        title_str = f"{meta.get('report_name')} ({meta.get('periods_label')})"
        render_results_and_simulation(S.result, report_title=title_str)
    else:
        st.info("Augšupielādējiet PZA failu vai izvēlieties saglabātu atskaiti.")
    st.stop()

ok_quota, quota_msg = profile.check_quota()
if not ok_quota and not S.get("is_admin", False):
    st.error(quota_msg)
    st.stop()

data = uploaded.getvalue()
fid = hashlib.md5(data).hexdigest()
if S.get("fid") != fid:
    S.fid = fid
    S.grids = load_grids(uploaded.name, data)
    S.pop("structure", None)
    S.pop("analysis", None)
    S.pop("cls_df", None)
    S.pop("result", None)
    S.pop("manual_leaf_overrides", None)

grids: dict[str, pd.DataFrame] = S.grids

if "structure" not in S:
    with st.spinner("Nolasa struktūru..."):
        try:
            sheet = pick_sheet(get_client(), MODEL, grids)
            S.structure = request_structure(get_client(), MODEL, sheet, grids[sheet])
        except Exception as e:
            st.error(f"Kļūda: {e}")
            st.stop()

stc: Structure = S.structure
try:
    ex = extract_lines(grids[stc.sheet], stc)
    base_analysis = analyze(ex, stc)
    overrides: dict[str, bool] = S.get("manual_leaf_overrides", {})
    if overrides:
        base_leaf_ids = {lf.id for lf in base_analysis.leaves}
        new_leaves = []
        for ln in base_analysis.ex.lines:
            if stc.result_row is not None and ln.row == stc.result_row:
                continue
            is_leaf = overrides.get(ln.id, ln.id in base_leaf_ids)
            if is_leaf:
                new_leaves.append(ln)
        base_analysis.leaves = new_leaves
    S.analysis = base_analysis
except Exception as e:
    st.error(f"Kļūda datu ielasīšanā: {e}")
    st.stop()

an = S.analysis
display_labels = build_display_labels(an.leaves)

with st.expander("PZA rindu korekcija"):
    leaf_id_set = {lf.id for lf in an.leaves}
    diag_rows = []
    for idx, ln in enumerate(an.ex.lines):
        ln_id = getattr(ln, "id", f"row_{idx}")
        ln_row = getattr(ln, "row", idx + 1)
        ln_label = getattr(ln, "label", str(ln))
        ln_path = getattr(ln, "path", "")
        is_res = stc.result_row is not None and ln_row == stc.result_row
        first_v, tot_v = _safe_line_stats(getattr(ln, "vals", None))
        diag_rows.append(
            {
                "id": ln_id,
                "Rinda": ln_row,
                "Iekļaut kā kontu": (ln_id in leaf_id_set) and not is_res,
                "Nosaukums": ln_label,
                "Sadaļa": ln_path,
                "1. periods": round(first_v, 2),
                "Kopā": round(tot_v, 2),
                "Statuss": "Rezultāta rinda" if is_res else ("Konts" if ln_id in leaf_id_set else "Izlaists"),
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    edited_diag = st.data_editor(
        diag_df,
        hide_index=True,
        use_container_width=True,
        disabled=["id", "Rinda", "Nosaukums", "Sadaļa", "1. periods", "Kopā", "Statuss"],
        key="pza_diag_editor",
    )
    if st.button("Saglabāt rindu izmaiņas"):
        new_overrides = {}
        for _, r_d in edited_diag.iterrows():
            new_overrides[r_d["id"]] = bool(r_d["Iekļaut kā kontu"])
        S.manual_leaf_overrides = new_overrides
        S.pop("cls_df", None)
        S.pop("result", None)
        st.rerun()

st.header("1. Kontu klasifikācija")

ignore_memory = st.checkbox("Ignorēt iepriekšējo profilu atmiņu un klasificēt kontus no jauna (AI)", value=False)

if "cls_df" not in S:
    if st.button("Klasificēt kontus", type="primary"):
        with st.spinner("Klasificē..."):
            try:
                known, unknown = ({}, an.leaves) if ignore_memory else profile.lookup(an.leaves)
                items = classify_with_claude(get_client(), MODEL, unknown) if unknown else []
                by_id = {i["id"]: i for i in items}
                rows = []
                for lf in an.leaves:
                    disp_lbl = display_labels.get(lf.id, lf.label)
                    if lf.id in known:
                        m = known[lf.id]
                        grp = normalize_group_key(m["group"])
                        rows.append(
                            dict(
                                id=lf.id,
                                Konts=lf.label,
                                Displeja_konts=disp_lbl,
                                Grupa=grp,
                                Ieņēmums=bool(grp.startswith("S_") or m.get("is_income", False)),
                                Jautājums="",
                            )
                        )
                    else:
                        i = by_id.get(lf.id, {"group": "OE_OTHER", "is_income": False, "question": ""})
                        grp = normalize_group_key(i.get("group", "OE_OTHER"))
                        rows.append(
                            dict(
                                id=lf.id,
                                Konts=lf.label,
                                Displeja_konts=disp_lbl,
                                Grupa=grp,
                                Ieņēmums=bool(grp.startswith("S_") or i.get("is_income", False)),
                                Jautājums=str(i.get("question") or "").strip(),
                            )
                        )
                S.cls_df = sort_cls_df(pd.DataFrame(rows))
                S.editor_v = 0
                S.known_count = len(known)
                S.unknown_count = len(unknown)
                st.rerun()
            except Exception as e:
                st.error(f"Kļūda klasifikācijā: {e}")
    if "cls_df" not in S:
        st.stop()

cls_df: pd.DataFrame = S.cls_df.copy()
cls_df["Grupa"] = cls_df["Grupa"].map(normalize_group_key)
if "Displeja_konts" not in cls_df.columns:
    cls_df["Displeja_konts"] = cls_df["id"].map(lambda x: display_labels.get(x, ""))

ambiguous_rows = cls_df[cls_df["Jautājums"].astype(str).str.strip() != ""]

if len(ambiguous_rows) > 0 and not S.get("clarifications_done", False):
    st.subheader("Precizēšana")
    clarifications: dict[str, tuple[str, bool]] = {}
    for _, r in ambiguous_rows.iterrows():
        row_id = r["id"]
        disp_name = r["Displeja_konts"]
        agent_grp = normalize_group_key(r["Grupa"])
        agent_sub_label = SUBGROUP_LABELS.get(agent_grp, agent_grp)
        default_idx = get_radio_index_for_group(agent_grp)

        st.markdown(f"**{disp_name}** *(Ieteikums: `{agent_sub_label}`)*")
        st.write(r["Jautājums"])

        choice = st.radio(
            f"Klasifikācija '{disp_name}'",
            options=RADIO_OPTIONS,
            index=default_idx,
            key=f"radio_{row_id}",
            label_visibility="collapsed",
        )
        clarifications[row_id] = resolve_radio_choice(choice, agent_grp, r["Konts"])
        st.markdown("---")

    if st.button("Apstiprināt", type="primary"):
        for row_id, (new_group, is_inc) in clarifications.items():
            mask = cls_df["id"] == row_id
            cls_df.loc[mask, "Grupa"] = new_group
            cls_df.loc[mask, "Ieņēmums"] = is_inc
            cls_df.loc[mask, "Jautājums"] = ""
        S.cls_df = sort_cls_df(cls_df)
        S.clarifications_done = True
        S.editor_v = S.get("editor_v", 0) + 1
        st.rerun()
    st.stop()

hdr_col1, hdr_col2 = st.columns([4, 1])
with hdr_col1:
    st.subheader("Kontu sadalījums")
with hdr_col2:
    if st.button("Sakārtot tabulu", use_container_width=True):
        S.cls_df = sort_cls_df(cls_df)
        S.editor_v = S.get("editor_v", 0) + 1
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
c1.metric("S (Ieņēmumi)", f"{cls_df['Grupa'].str.startswith('S_').sum()} konti")
c2.metric("TVC (Mainīgās)", f"{cls_df['Grupa'].str.startswith('TVC').sum()} konti")
c3.metric("OE (Uzturēšana)", f"{cls_df['Grupa'].str.startswith('OE_').sum()} konti")
c4.metric("Izslēgtie", f"{(cls_df['Grupa'] == 'EXCLUDED').sum()} konti")

view_df = pd.DataFrame(
    {
        "Konts": cls_df["Displeja_konts"].tolist(),
        "Lielā grupa": [get_main_cat(g) for g in cls_df["Grupa"]],
        "Apakšgrupa (Rinda atskaitē)": [SUBGROUP_LABELS[normalize_group_key(g)] for g in cls_df["Grupa"]],
    }
)

full_editor_height = (len(view_df) + 1) * 36 + 14
editor_key = f"clean_cls_editor_v_{S.get('editor_v', 0)}"

edited_view = st.data_editor(
    view_df,
    hide_index=True,
    use_container_width=True,
    height=full_editor_height,
    key=editor_key,
    disabled=["Konts"],
    column_config={
        "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
        "Lielā grupa": st.column_config.SelectboxColumn("Lielā grupa", options=MAIN_GROUP_OPTIONS, required=True, width="medium"),
        "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn("Apakšgrupa", options=list(SUBGROUP_LABELS.values()), required=True, width="large"),
    },
)

updated_groups: list[str] = []
changed = False

for pos_idx in range(len(cls_df)):
    old_grp = normalize_group_key(cls_df.iloc[pos_idx]["Grupa"])
    old_main = get_main_cat(old_grp)
    old_sub = SUBGROUP_LABELS[old_grp]

    new_main = edited_view.iloc[pos_idx]["Lielā grupa"]
    new_sub = edited_view.iloc[pos_idx]["Apakšgrupa (Rinda atskaitē)"]

    if new_sub != old_sub and new_sub in LABEL_TO_KEY:
        new_grp = LABEL_TO_KEY[new_sub]
        updated_groups.append(new_grp)
        changed = True
    elif new_main != old_main and new_main in MAIN_GROUP_TO_DEFAULT_SUB:
        new_grp = MAIN_GROUP_TO_DEFAULT_SUB[new_main]
        updated_groups.append(new_grp)
        changed = True
    else:
        updated_groups.append(old_grp)

if changed:
    cls_df["Grupa"] = updated_groups
    cls_df["Ieņēmums"] = cls_df["Grupa"].str.startswith("S_")
    S.cls_df = cls_df
    S.editor_v = S.get("editor_v", 0) + 1
    st.rerun()

st.header("2. Aprēķins un analīze")

default_rep_name = f"{Path(uploaded.name).stem} ({datetime.now().strftime('%d.%m.%Y')})"
col_rn, col_btn = st.columns([2, 1])
with col_rn:
    report_custom_name = st.text_input("Atskaites nosaukums", value=default_rep_name, label_visibility="collapsed")
with col_btn:
    run_calc = st.button("Apstiprināt un aprēķināt", type="primary", use_container_width=True)

if run_calc:
    cls_sorted = sort_cls_df(cls_df)
    S.cls_df = cls_sorted
    S.editor_v = S.get("editor_v", 0) + 1

    mapping = {
        r["id"]: {
            "group": normalize_group_key(r["Grupa"]),
            "is_income": bool(str(r["Grupa"]).startswith("S_") or r["Ieņēmums"]),
        }
        for _, r in cls_sorted.iterrows()
    }
    pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
    reported = an.ex.lines[pos].vals if pos is not None else None

    res = compute_toc(an.leaves, mapping, an.ex.period_cols, reported)
    S.result = res
    S.pop("loaded_report_meta", None)

    if profile.is_connected:
        ok_map, _ = profile.save(an.leaves, mapping)
        ok_rep, rep_id = profile.save_report(
            report_name=report_custom_name.strip() or default_rep_name,
            file_name=uploaded.name,
            res_obj=res,
            commentary="",
        )
        profile.increment_usage()
        S.current_report_id = rep_id
        if ok_map and ok_rep:
            st.toast("Atskaite saglabāta.")
        elif not ok_rep:
            st.warning(f"Kļūda saglabājot tabulu: {profile.last_error}")

    S.pop("commentary", None)
    st.rerun()

if "result" in S:
    render_results_and_simulation(S.result)

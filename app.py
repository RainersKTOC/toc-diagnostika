"""
app.py — TOC-FIN Vadības lēmumu un finanšu diagnostikas sistēma (v11.0)
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
import hashlib
import io
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
    reconcile,
    request_structure,
)

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="TOC-FIN Vadības Sistēma", layout="wide", page_icon="📊")

# --- IEVADS UN APRAKSTS ---
st.title("📊 TOC-FIN: Finanšu diagnostika un vadības lēmumu sistēma")
st.markdown("""
**TOC-FIN** pārvērš grāmatvedības datus vadības lēmumiem piemērotā ekonomiskajā informācijā. 
Sistēma atbrīvo vadības komandu no mākslīgām pašizmaksas alokācijām, nodalot patiesi mainīgās izmaksas ($TVC$) 
no kapacitātes uzturēšanas izmaksām ($OE$), lai novērtētu patieso caurlaidību ($T$), produktivitāti ($PR = T / OE$) un modelētu nākotnes lēmumus.
""")

with st.expander("ℹ️ Uzzināt vairāk: Kāpēc TOC-FIN nodala vadības lēmumus no grāmatvedības pašizmaksas?"):
    st.markdown("""
    **Organizatoriskais princips:** Biznesa lēmumi pieder vadības komandai (ieskaitot finanšu direktoru). Grāmatvedības uzdevums ir precīza darījumu uzskaite un nodokļu atbilstība, nevis biznesa lēmumu lietderības noteikšana.

    * **S (Sales / Ieņēmumi)** — Visa reālā nauda, kas ienāk uzņēmumā (pamatdarbības apgrozījums, granti, citi ieņēmumi).
    * **TVC (Truly Variable Costs / Patiesi mainīgās izmaksas)** — Tikai izdevumi, kas mainās **1:1 ar katru pārdoto vienību** (izejvielas, materiāli un tiešais transports klientam). Ja nepārdod nevienu vienību, TVC = 0.
    * **T (Throughput / Caurlaidība)** — Jaunā naudas masa, ko uzņēmums ģenerē: $$T = S - TVC$$.
    * **OE (Operating Expenses / Uzturēšanas izdevumi)** — Visa reālā nauda, ko uzņēmums tērē darbības nodrošināšanai (visas algas, telpas, IT, mārketings, kā arī procentu maksājumi un UIN).
    * **Izslēgtie (Virtuālā nauda)** — Tikai grāmatvedības ieraksti bez reālas naudas plūsmas šajā periodā (amortizācija/nolietojums, krājumu pārvērtēšana).
    * **NP (Net Profit / Neto peļņa)** — $$NP = T - OE$$.
    * **PR (Productivity Ratio / Produktivitātes rādītājs)** — $$PR = T / OE$$.
    """)

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


MAIN_CAT_BY_CODE = {
    "S": "🟡 S (Ieņēmumi)",
    "TVC": "🟠 TVC (Mainīgās izmaksas)",
    "OE": "🟤 OE (Uzturēšanas izdevumi)",
    "EXCL": "⚪ Izslēgtie (Virtuālā nauda)",
}

RADIO_OPTIONS = [
    "🟡 S — Ieņēmumi (Sales)",
    "🟠 TVC — Patiesi mainīga izmaksa (1:1 ar pārdošanu)",
    "🟤 OE — Uzturēšanas / darbības izdevums",
    "⚪ Izslēgt — Virtuāla nauda (amortizācija)",
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
    return MAIN_CAT_BY_CODE.get(code, MAIN_CAT_BY_CODE["OE"])


def get_radio_index_for_group(group_key: str) -> int:
    code = get_main_code(group_key)
    return {"S": 0, "TVC": 1, "OE": 2, "EXCL": 3}.get(code, 2)


def resolve_radio_choice(choice: str, original_group: str, konts_label: str) -> tuple[str, bool]:
    """
    Ja lietotājs apstiprina to pašu lielo kategoriju, ko aģents jau ieteica,
    saglabājam aģenta precīzo apakšgrupu. Ja maina uz citu lielo kategoriju —
    piešķiram atbilstošo noklusējuma apakšgrupu.
    """
    orig_grp = normalize_group_key(original_group)
    orig_code = get_main_code(orig_grp)
    lbl_low = str(konts_label or "").lower()

    if choice.startswith("🟡"):
        target_code = "S"
        fallback = "S_OTHER" if ("grant" in lbl_low or "citi" in lbl_low or "pārēj" in lbl_low) else "S_REV"
    elif choice.startswith("🟠"):
        target_code = "TVC"
        fallback = "TVC_DELIVERY" if ("piegād" in lbl_low or "transp" in lbl_low or "kurjer" in lbl_low) else "TVC_MATERIALS"
    elif choice.startswith("⚪"):
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
    """
    Ja PZA failā ir vairāki konti ar vienādu nosaukumu (piem., algas ražošanā un administrācijā),
    pievieno iekavās ceļu/sadaļu, lai lietotājs un tabula tos skaidri atšķirtu.
    """
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
            display_map[lf.id] = f"{lbl}  [{parent}]"
        else:
            display_map[lf.id] = lbl
    return display_map


def render_unified_toc_tables_html(
    amounts_df: pd.DataFrame,
    main_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    row_kinds: dict[str, str],
    title_kpi: str = "📈 Produktivitātes un atdeves rādītāji",
) -> str:
    periods = list(amounts_df.columns)
    row_names = list(amounts_df.index)
    kinds = [row_kinds.get(rn, "OE_CHILD") for rn in row_names]

    lbl_width_pct = 23.0
    rem_pct = 100.0 - lbl_width_pct
    per_period_pct = rem_pct / max(len(periods), 1)
    val_w = per_period_pct * 0.68
    pct_w = per_period_pct * 0.32

    html = [f"""
    <style>
    .toc-table-wrap {{ width: 100%; overflow-x: visible; margin-bottom: 14px; }}
    table.toc-unified {{ width: 100%; border-collapse: collapse; font-family: 'Calibri', sans-serif; font-size: 12px; table-layout: fixed; }}
    table.toc-unified th {{ background-color: #EAECEE; color: #111111; font-weight: 700; text-align: center; padding: 5px 2px; border-top: 2px solid #404040; border-bottom: 2px solid #404040; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    table.toc-unified td {{ padding: 4px 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    table.toc-unified td.lbl-main {{ text-align: left; font-weight: 700; padding-left: 6px; border-left: 2px solid #404040; border-right: 2px solid #404040; }}
    table.toc-unified td.lbl-sub {{ text-align: left; font-weight: 400; padding-left: 16px; border-left: 2px solid #404040; border-right: 2px solid #404040; }}
    table.toc-unified td.val-col {{ text-align: right; border-left: 1px solid #BFBFBF; }}
    table.toc-unified td.pct-col {{ text-align: right; font-size: 11px; padding-right: 3px; border-left: 1px dotted #D0D3D4; border-right: 2px solid #595959; }}
    table.toc-unified th.th-val {{ border-left: 1px solid #BFBFBF; }}
    table.toc-unified th.th-pct {{ font-size: 11px; border-left: 1px dotted #BFBFBF; border-right: 2px solid #595959; }}
    table.toc-unified th.th-lbl {{ text-align: left; padding-left: 6px; border-left: 2px solid #404040; border-right: 2px solid #404040; }}
    table.toc-unified tr.spacer-row td {{ background: transparent !important; border: none !important; height: 26px; padding: 10px 0 4px 0; font-size: 16px; font-weight: 700; color: #1f2937; }}
    </style>
    <div class="toc-table-wrap"><table class="toc-unified">
    <colgroup>
        <col style="width: {lbl_width_pct:.2f}%;">
    """]
    for _ in periods:
        html.append(f'<col style="width: {val_w:.3f}%;"><col style="width: {pct_w:.3f}%;">')
    html.append("</colgroup><thead><tr>")
    html.append('<th class="th-lbl">Rādītājs / Kontugrupa</th>')
    for p in periods:
        html.append(f'<th class="th-val">{p}</th><th class="th-pct">%</th>')
    html.append("</tr></thead><tbody>")

    for i, rname in enumerate(row_names):
        kind = kinds[i]
        next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
        fill_hex, is_bold, text_hex = FILLS.get(kind, FILLS["OE_CHILD"])

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

        row_style = (
            f"background-color: #{fill_hex}; color: #{text_hex}; font-weight: {font_w}; "
            f"border-top: 1px solid #BFBFBF; border-bottom: {border_bottom};"
        )
        html.append(f'<tr style="{row_style}"><td class="{lbl_cls}" title="{rname}">{rname}</td>')

        for p in periods:
            val_str = fmt_num(main_df.loc[rname, p])
            pct_str = fmt_pct_int(main_df.loc[rname, f"%|{p}"])
            html.append(f'<td class="val-col">{val_str}</td><td class="pct-col">{pct_str}</td>')
        html.append("</tr>")

    total_cols = 1 + 2 * len(periods)
    html.append(f'<tr class="spacer-row"><td colspan="{total_cols}">{title_kpi}</td></tr>')

    html.append("<tr>")
    html.append('<th class="th-lbl" style="background-color: #EAECEE; border-top: 2px solid #404040; border-bottom: 2px solid #404040;">Produktivitātes rādītājs</th>')
    for p in periods:
        html.append(
            f'<th colspan="2" style="background-color: #EAECEE; border-top: 2px solid #404040; '
            f'border-bottom: 2px solid #404040; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959;">{p}</th>'
        )
    html.append("</tr>")

    for i, idx_name in enumerate(kpi_df.index):
        b_bot = "2.5px solid #404040" if i == len(kpi_df.index) - 1 else "1px solid #BFBFBF"
        html.append(f'<tr style="background-color: #F2F4F4; color: #000000; border-top: 1px solid #BFBFBF; border-bottom: {b_bot};">')
        html.append(f'<td class="lbl-main" style="font-weight: 600;" title="{idx_name}">{idx_name}</td>')
        for p in periods:
            v = kpi_df.loc[idx_name, p]
            if pd.isna(v):
                s = ""
            elif idx_name.startswith("T%"):
                s = f"{v * 100:.0f}%"
            else:
                s = f"{v:.2f}"
            html.append(
                f'<td colspan="2" style="text-align: right; padding: 4px 6px; '
                f'border-left: 1px solid #BFBFBF; border-right: 2px solid #595959; font-weight: 600;">{s}</td>'
            )
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


# Zināšanu bāzes ielāde
knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

# Supabase profils un sānjoslas statuss
profile = SupabaseProfileStore(user_id="default_user")
with st.sidebar:
    st.subheader("🗄️ Datubāzes statuss")
    if profile.is_connected:
        saved_cnt = profile.count_saved()
        if profile.last_error:
            st.warning(f"Supabase pieslēgts, bet tabula nav gatava: {profile.last_error}")
        else:
            st.success(f"Supabase aktīvs ({saved_cnt} saglabāti konti)")
    else:
        st.warning("Supabase nav pieslēgts (pārbaudiet secrets.toml)")

    if st.button("🔄 Sākt pilnīgi no jauna", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------------------------------ 1. AUGŠUPIELĀDE
uploaded = st.file_uploader("Augšupielādējiet PZA failu (CSV vai Excel formātā)", type=["csv", "xlsx", "xlsm", "xls"])

if uploaded is None:
    st.info("Lai sāktu diagnostiku un lēmumu modelēšanu, augšupielādējiet PZA failu.")
    st.stop()

data = uploaded.getvalue()
fid = hashlib.md5(data).hexdigest()
if st.session_state.get("fid") != fid:
    st.session_state.clear()
    st.session_state.fid = fid
    st.session_state.grids = load_grids(uploaded.name, data)

S = st.session_state
grids: dict[str, pd.DataFrame] = S.grids

if "structure" not in S:
    with st.spinner("Nolasa faila struktūru un attīra datus..."):
        try:
            sheet = pick_sheet(get_client(), MODEL, grids)
            S.structure = request_structure(get_client(), MODEL, sheet, grids[sheet])
        except Exception as e:
            st.error(f"Neizdevās automātiski nolasīt struktūru: {e}")
            st.stop()

stc: Structure = S.structure
try:
    ex = extract_lines(grids[stc.sheet], stc)
    S.analysis = analyze(ex, stc)
except Exception as e:
    st.error(f"Kļūda datu ielasīšanā: {e}")
    st.stop()

an = S.analysis
display_labels = build_display_labels(an.leaves)

# ------------------------------------------------------------------ 2. KONTU KLASIFIKĀCIJA UN SUPABASE PROFILS
st.header("1. Kontu ekonomiskā klasifikācija")

if "cls_df" not in S:
    if st.button("🔍 Sākt kontu TOC-FIN klasifikāciju", type="primary"):
        with st.spinner("Analizē kontus un salīdzina ar saglabāto Supabase profilu..."):
            try:
                known, unknown = profile.lookup(an.leaves)
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
                                Avots="🗄️ Supabase profils",
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
                                Avots="🤖 AI klasifikācija",
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

if S.get("known_count", 0) > 0 or S.get("unknown_count", 0) > 0:
    st.caption(
        f"📌 No Supabase profila atpazīti **{S.get('known_count', 0)}** konti | "
        f"Ar AI no jauna klasificēti **{S.get('unknown_count', 0)}** konti."
    )

# --- 2.A NESKAIDRO POZĪCIJU JAUTĀJUMI (AR PAREIZU AĢENTA NOKLUSĒJUMU) ---
ambiguous_rows = cls_df[cls_df["Jautājums"].astype(str).str.strip() != ""]

if len(ambiguous_rows) > 0 and not S.get("clarifications_done", False):
    st.subheader("❓ Nepieciešams vadības lēmums par dažām pozīcijām")
    st.info("Katram kontam jau ir iezīmēts aģenta visticamākais ieteikums. Mainiet tikai tad, ja pēc būtības pozīcija atbilst citai grupai.")

    clarifications: dict[str, tuple[str, bool]] = {}
    for _, r in ambiguous_rows.iterrows():
        row_id = r["id"]
        disp_name = r["Displeja_konts"]
        agent_grp = normalize_group_key(r["Grupa"])
        agent_sub_label = SUBGROUP_LABELS.get(agent_grp, agent_grp)
        default_idx = get_radio_index_for_group(agent_grp)

        st.markdown(f"**📌 {disp_name}** *(Aģenta sākotnējā hipotēze: `{agent_sub_label}`)*")
        st.write(f"👉 **Aģenta jautājums:** {r['Jautājums']}")

        choice = st.radio(
            f"Kā klasificēt '{disp_name}'?",
            options=RADIO_OPTIONS,
            index=default_idx,
            key=f"radio_{row_id}",
        )
        clarifications[row_id] = resolve_radio_choice(choice, agent_grp, r["Konts"])
        st.markdown("---")

    if st.button("✅ Apstiprināt atbildes un pāriet uz kontu tabulu", type="primary"):
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

# --- 2.B KONTU SADALĪJUMA TABULA (BEZ RINDU PĀRRAKSTĪŠANAS BUG) ---
hdr_col1, hdr_col2 = st.columns([3, 1])
with hdr_col1:
    st.subheader("📋 Kontu sadalījums pa grupām un apakšgrupām")
with hdr_col2:
    if st.button("🔃 Sakārtot tabulu pa grupām", use_container_width=True):
        S.cls_df = sort_cls_df(cls_df)
        S.editor_v = S.get("editor_v", 0) + 1
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🟡 S (Ieņēmumi)", f"{cls_df['Grupa'].str.startswith('S_').sum()} konti")
c2.metric("🟠 TVC (Mainīgās)", f"{cls_df['Grupa'].str.startswith('TVC').sum()} konti")
c3.metric("🟤 OE (Uzturēšana)", f"{cls_df['Grupa'].str.startswith('OE_').sum()} konti")
c4.metric("⚪ Izslēgtie (Virtuālie)", f"{(cls_df['Grupa'] == 'EXCLUDED').sum()} konti")

# Veidojam redaktora skatu ar stabilu rindu secību
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
        "Lielā grupa": st.column_config.SelectboxColumn(
            "Lielā grupa",
            options=list(MAIN_GROUP_TO_DEFAULT_SUB.keys()),
            required=True,
            width="medium",
        ),
        "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn(
            "Apakšgrupa / Rinda atskaitē",
            options=list(SUBGROUP_LABELS.values()),
            required=True,
            width="large",
        ),
    },
)

# Pārbaudām izmaiņas pa tiešo pēc stabila rindas indeksa (bez pārkārtošanas!)
updated_groups: list[str] = []
changed = False

for pos_idx in range(len(cls_df)):
    old_grp = normalize_group_key(cls_df.iloc[pos_idx]["Grupa"])
    old_main = get_main_cat(old_grp)
    old_sub = SUBGROUP_LABELS[old_grp]

    new_main = edited_view.iloc[pos_idx]["Lielā grupa"]
    new_sub = edited_view.iloc[pos_idx]["Apakšgrupa (Rinda atskaitē)"]

    if new_sub != old_sub and new_sub in LABEL_TO_KEY:
        # Lietotājs nomainīja apakšgrupu -> Lielā grupa automātiski pielāgojas tai
        new_grp = LABEL_TO_KEY[new_sub]
        updated_groups.append(new_grp)
        changed = True
    elif new_main != old_main and new_main in MAIN_GROUP_TO_DEFAULT_SUB:
        # Lietotājs nomainīja Lielo grupu -> Apakšgrupa automātiski pārslēdzas uz šīs grupas noklusējumu
        new_grp = MAIN_GROUP_TO_DEFAULT_SUB[new_main]
        updated_groups.append(new_grp)
        changed = True
    else:
        updated_groups.append(old_grp)

if changed:
    cls_df["Grupa"] = updated_groups
    cls_df["Ieņēmums"] = cls_df["Grupa"].str.startswith("S_")
    S.cls_df = cls_df
    # Palielinām editor_v, lai st.data_editor pārzīmē abas kolonnas sinhroni ar jauno stāvokli
    S.editor_v = S.get("editor_v", 0) + 1
    st.rerun()

# ------------------------------------------------------------------ 3. GALA APRĒĶINS UN SAGLABĀŠANA SUPABASE
st.header("2. TOC-FIN Diagnostika un Lēmumu Atbalsts")
if st.button("🚀 Apstiprināt klasifikāciju un veikt TOC-FIN aprēķinu", type="primary"):
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

    if profile.is_connected:
        ok, cnt = profile.save(an.leaves, mapping)
        if ok:
            st.toast(f"✅ Supabase profilā saglabāti {cnt} kontu kartējumi!", icon="🗄️")
        else:
            st.warning(f"Neizdevās saglabāt Supabase datubāzē: {profile.last_error}")

    S.pop("commentary", None)
    S.chat = []

if "result" in S:
    res = S.result

    main_tab1, main_tab2 = st.tabs(
        [
            "📊 1. TOC-FIN Analytics (Vēsturiskā diagnostika un Vadības analīze)",
            "⚡ 2. TOC-FIN Decision Support (Procentu % Simulācija pa periodiem)",
        ]
    )

    with main_tab1:
        xl_f, xl_v = io.BytesIO(), io.BytesIO()
        build_excel(res, xl_f, formulas=True)
        build_excel(res, xl_v, formulas=False)
        d1, d2 = st.columns(2)
        d1.download_button(
            "📥 Lejupielādēt Excel ar formulām (.xlsx)",
            xl_f.getvalue(),
            "TOC_Finansu_Diagnostika.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        d2.download_button(
            "📥 Lejupielādēt Excel ar vērtībām (.xlsx)",
            xl_v.getvalue(),
            "TOC_Finansu_Diagnostika_vertibas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.subheader("Finanšu pārskata tabula (T* skats)")
        st.markdown(
            render_unified_toc_tables_html(res.amounts, res.main, res.kpi, res.row_kinds),
            unsafe_allow_html=True,
        )

        # Kontroles salīdzinājums ar grāmatvedības rezultātu (ja pieejams)
        if getattr(res, "reconciliation", None) is not None and not res.reconciliation.empty:
            with st.expander("🔍 Kontrole: Salīdzinājums ar grāmatvedības PZA rezultātu un izslēgtajām pozīcijām"):
                st.dataframe(res.reconciliation.style.format(fmt_num), use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Vadības kopsavilkums un stratēģiskie secinājumi")
        tables_md = res.amounts.round(0).to_markdown() + "\n\n" + res.kpi.round(3).to_markdown()

        knowledge_context = f"TOC-FIN zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
        COMMENT_SYSTEM = (
            "Tu esi pieredzējis TOC-FIN finanšu direktora padomnieks un vadības konsultants. "
            "Uzrunā lietotāju kā Vadības komandu. Nekad nelieto grāmatvedības pašizmaksas klišejas. "
            f"{knowledge_context}"
            "Raksti latviešu valodā, strukturēti, asi un uz lēmumiem vērsti."
        )

        if "commentary" not in S:
            if st.button("📝 Ģenerēt vadības analīzi un rīcības plānu", type="primary"):
                with st.spinner("Sagatavo TOC-FIN vadības kopsavilkumu..."):
                    r = get_client().messages.create(
                        model=MODEL,
                        max_tokens=2500,
                        system=COMMENT_SYSTEM,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Aprēķinātās tabulas (summas EUR):\n\n{tables_md}\n\nUzraksti vadības analīzi.",
                            }
                        ],
                    )
                    S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    st.rerun()
        else:
            st.markdown(S.commentary)

            st.markdown("---")
            st.subheader("💬 Diskusija ar TOC-FIN finanšu konsultantu")
            for msg in S.get("chat", []):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_q = st.chat_input("Uzdodiet jautājumu par šiem skaitļiem vai lēmumu scenārijiem...")
            if user_q:
                S.chat.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Domā..."):
                        chat_messages = [
                            {
                                "role": "user",
                                "content": f"Konteksts — uzņēmuma TOC-FIN tabulas:\n\n{tables_md}\n\nSākotnējā analīze:\n{S.commentary}",
                            },
                            {"role": "assistant", "content": "Sapratu datus. Esmu gatavs atbildēt uz vadības jautājumiem."},
                        ] + S.chat
                        resp = get_client().messages.create(
                            model=MODEL,
                            max_tokens=1500,
                            system=COMMENT_SYSTEM,
                            messages=chat_messages,
                        )
                        ans = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                        st.markdown(ans)
                        S.chat.append({"role": "assistant", "content": ans})

    with main_tab2:
        st.subheader("⚡ Interaktīvā scenāriju un mērķu simulācija (% izmaiņas pret izvēlēto periodu)")
        all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
        if not all_months:
            all_months = list(res.amounts.columns)

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            preset = st.selectbox(
                "Ātrā perioda izvēle:",
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
            selected_months = st.multiselect("Simulācijā iekļautie mēneši:", options=all_months, default=default_sel)

        if not selected_months:
            st.warning("Lūdzu, izvēlieties vismaz vienu mēnesi.")
            st.stop()

        st.markdown("---")
        st.markdown("#### Iestatiet procentuālās izmaiņas (%) katrai pozīcijai")

        s_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "S_CHILD"]
        tvc_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "TVC_CHILD"]
        oe_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "OE_CHILD"]

        pct_changes: dict[str, float] = {}
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("🟡 **Ieņēmumi (S apakšgrupas)**")
            for sc_name in s_children:
                pct_changes[sc_name] = st.number_input(
                    f"{sc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{sc_name}"
                )
        with sc2:
            st.markdown("🟠 **Mainīgās (TVC)**")
            for tc_name in tvc_children:
                pct_changes[tc_name] = st.number_input(
                    f"{tc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{tc_name}"
                )
        with sc3:
            st.markdown("🟤 **Darbības izdevumi (OE)**")
            for oc_name in oe_children:
                def_v = 5.0 if "alga" in oc_name.lower() else 0.0
                pct_changes[oc_name] = st.number_input(
                    f"{oc_name} (%)", value=def_v, step=1.0, format="%.1f", key=f"sim_{oc_name}"
                )

        sim_amounts = res.amounts[selected_months].copy().astype(float)
        base_sub = res.amounts[selected_months].copy().astype(float)

        for rname, pct_val in pct_changes.items():
            if rname in sim_amounts.index:
                sim_amounts.loc[rname] = base_sub.loc[rname] * (1.0 + pct_val / 100.0)

        if s_children:
            sim_amounts.loc["S"] = sim_amounts.loc[s_children].sum(axis=0)
        if tvc_children:
            sim_amounts.loc["TVC"] = sim_amounts.loc[tvc_children].sum(axis=0)
        sim_amounts.loc["T"] = sim_amounts.loc["S"] - sim_amounts.loc["TVC"]
        if oe_children:
            sim_amounts.loc["OE"] = sim_amounts.loc[oe_children].sum(axis=0)
        sim_amounts.loc["NP"] = sim_amounts.loc["T"] - sim_amounts.loc["OE"]

        base_sum = base_sub.sum(axis=1)
        sim_sum = sim_amounts[selected_months].sum(axis=1)
        diff_sum = sim_sum - base_sum

        sim_amounts["Bāze KOPĀ"] = base_sum
        sim_amounts["Simulācija KOPĀ"] = sim_sum
        sim_amounts["Δ Izmaiņa (€)"] = diff_sum
        sim_amounts["Simulācija AVG"] = sim_amounts[selected_months].mean(axis=1)

        sim_main = {}
        for c in sim_amounts.columns:
            sim_main[c] = sim_amounts[c]
            if c == "Δ Izmaiņa (€)":
                sim_main[f"%|{c}"] = np.where(base_sum != 0, diff_sum / np.abs(base_sum), np.nan)
            else:
                s_col = sim_amounts.loc["S", c]
                sim_main[f"%|{c}"] = sim_amounts[c] / s_col if s_col else np.nan
        sim_main_df = pd.DataFrame(sim_main, index=sim_amounts.index)

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

        sim_kpi_df = pd.DataFrame({c: sim_kpi_for(sim_amounts[c]) for c in sim_amounts.columns})
        sim_kpi_df["Δ Izmaiņa (€)"] = sim_kpi_df["Simulācija KOPĀ"] - sim_kpi_df["Bāze KOPĀ"]

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        ds_val, dt_val, doe_val, dnp_val = diff_sum["S"], diff_sum["T"], diff_sum["OE"], diff_sum["NP"]

        m1.metric("Simulētais S", f"{fmt_num(sim_sum['S'])} €", delta=f"{ds_val:+,.0f} €".replace(",", " "))
        m2.metric("Simulētais T", f"{fmt_num(sim_sum['T'])} €", delta=f"{dt_val:+,.0f} €".replace(",", " "))
        m3.metric(
            "Simulētais OE",
            f"{fmt_num(sim_sum['OE'])} €",
            delta=f"{doe_val:+,.0f} €".replace(",", " "),
            delta_color="inverse",
        )
        m4.metric("Simulētā Neto Peļņa (NP)", f"{fmt_num(sim_sum['NP'])} €", delta=f"{dnp_val:+,.0f} €".replace(",", " "))

        st.subheader("📊 Simulētā Finanšu un Produktivitātes tabula")
        st.markdown(
            render_unified_toc_tables_html(
                sim_amounts,
                sim_main_df,
                sim_kpi_df,
                res.row_kinds,
                title_kpi="📈 Simulētie Produktivitātes un atdeves rādītāji",
            ),
            unsafe_allow_html=True,
        )

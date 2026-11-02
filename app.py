"""
app.py — TOC-FIN Vadības lēmumu un finanšu diagnostikas sistēma (v10.3 ar Supabase un labojumiem)
Project ID: amprqueamdcxpjufigvm
"""
import hashlib
import io
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

from db import SupabaseProfileStore
from toc_engine import (GROUPS, SUBGROUP_LABELS, LABEL_TO_KEY, MAIN_GROUP_TO_DEFAULT_SUB,
                        FILLS, build_excel, classify_with_claude, compute_toc)
from toc_ingest import (Structure, analyze, extract_lines, load_grids, pick_sheet,
                        reconcile, request_structure)

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
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit noslēpumos.")
    st.stop()


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=API_KEY)


def fmt_num(x) -> str:
    return "" if pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def fmt_pct_int(x) -> str:
    return "" if pd.isna(x) else f"{x * 100:.0f}%"


def get_main_cat(group_key: str) -> str:
    if group_key == "S":
        group_key = "S_REV"
    code = GROUPS.get(group_key, ("", "OE"))[1]
    return {
        "S": "🟡 S (Ieņēmumi)",
        "TVC": "🟠 TVC (Mainīgās izmaksas)",
        "OE": "🟤 OE (Uzturēšanas izdevumi)",
        "EXCL": "⚪ Izslēgtie (Virtuālā nauda)"
    }.get(code, "🟤 OE (Uzturēšanas izdevumi)")


GROUP_SORT_ORDER = {
    "S": 1, "S_REV": 1, "S_OTHER": 2,
    "TVC_MATERIALS": 3, "TVC_DELIVERY": 4,
    "OE_ADMIN_WAGES": 5, "OE_WORKER_WAGES": 6, "OE_MARKETING": 7,
    "OE_PLATFORMS": 8, "OE_PREMISES": 9, "OE_OFFICE_IT": 10,
    "OE_FIN_TAX": 11, "OE_OTHER": 12, "EXCLUDED": 13,
}


def render_unified_toc_tables_html(amounts_df: pd.DataFrame, main_df: pd.DataFrame, kpi_df: pd.DataFrame, row_kinds: dict[str, str], title_kpi: str = "📈 Produktivitātes un atdeves rādītāji") -> str:
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

        row_style = f"background-color: #{fill_hex}; color: #{text_hex}; font-weight: {font_w}; border-top: 1px solid #BFBFBF; border-bottom: {border_bottom};"
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
        html.append(f'<th colspan="2" style="background-color: #EAECEE; border-top: 2px solid #404040; border-bottom: 2px solid #404040; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959;">{p}</th>')
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
            html.append(f'<td colspan="2" style="text-align: right; padding: 4px 6px; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959; font-weight: 600;">{s}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


# Zināšanu bāzes ielāde
knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

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

# ------------------------------------------------------------------ 2. KONTU KLASIFIKĀCIJA un SUPABASE PROFILE STORE
st.header("1. Kontu ekonomiskā klasifikācija")

profile = SupabaseProfileStore(user_id="default_user")

if "cls_df" not in S:
    if st.button("🔍 Sākt kontu TOC-FIN klasifikāciju", type="primary"):
        with st.spinner("Analizē un salīdzina ar saglabāto Supabase profilu..."):
            try:
                known, unknown = profile.lookup(an.leaves)
                items = classify_with_claude(get_client(), MODEL, unknown) if unknown else []
                by_id = {i["id"]: i for i in items}
                rows = []
                for lf in an.leaves:
                    if lf.id in known:
                        m = known[lf.id]
                        rows.append(dict(id=lf.id, Konts=lf.label, Grupa=m["group"], Ieņēmums=bool(m["is_income"]), Jautājums=""))
                    else:
                        i = by_id[lf.id]
                        rows.append(dict(id=lf.id, Konts=lf.label, Grupa=i["group"], Ieņēmums=bool(i["is_income"]), Jautājums=i.get("question") or ""))
                S.cls_df = pd.DataFrame(rows)
                st.rerun()
            except Exception as e:
                st.error(f"Kļūda klasifikācijā: {e}")
    if "cls_df" not in S:
        st.stop()

cls_df: pd.DataFrame = S.cls_df
cls_df["Grupa"] = cls_df["Grupa"].replace({"S": "S_REV"})

ambiguous_rows = cls_df[cls_df["Jautājums"].astype(str).str.strip() != ""]

if len(ambiguous_rows) > 0 and not S.get("clarifications_done", False):
    st.subheader("❓ Nepieciešams vadības lēmums par dažām pozīcijām")
    clarifications = {}
    for idx, r in ambiguous_rows.iterrows():
        st.markdown(f"**📌 {r['Konts']}**")
        st.write(f"👉 **Aģenta jautājums:** {r['Jautājums']}")
        
        # Nosakām aģenta ieteikto indeksu radio pogām
        agent_group = r.get("Grupa", "OE_OTHER")
        default_idx = 2
        if str(agent_group).startswith("S_"):
            default_idx = 0
        elif str(agent_group).startswith("TVC"):
            default_idx = 1
        elif str(agent_group) == "EXCLUDED":
            default_idx = 3

        choice = st.radio(
            f"Kā klasificēt '{r['Konts']}'?",
            options=[
                "🟡 S — Ieņēmumi (Sales)",
                "🟠 TVC — Patiesi mainīga izmaksa (1:1 ar pārdošanu)",
                "🟤 OE — Uzturēšanas / darbības izdevums",
                "⚪ Izslēgt — Virtuāla nauda (amortizācija)"
            ],
            index=default_idx,
            key=f"radio_{r['id']}"
        )
        if choice.startswith("🟡"):
            clarifications[r["id"]] = ("S_OTHER" if "grant" in r["Konts"].lower() else "S_REV", True)
        elif choice.startswith("🟠"):
            clarifications[r["id"]] = ("TVC_DELIVERY" if "piegād" in r["Konts"].lower() else "TVC_MATERIALS", False)
        elif choice.startswith("⚪"):
            clarifications[r["id"]] = ("EXCLUDED", False)
        else:
            clarifications[r["id"]] = ("OE_OTHER", False)
        st.markdown("---")

    if st.button("Apstiprināt atbildes", type="primary"):
        for row_id, (new_group, is_inc) in clarifications.items():
            cls_df.loc[cls_df["id"] == row_id, "Grupa"] = new_group
            cls_df.loc[cls_df["id"] == row_id, "Ieņēmums"] = is_inc
            cls_df.loc[cls_df["id"] == row_id, "Jautājums"] = ""
        S.cls_df = cls_df
        S.clarifications_done = True
        st.rerun()
    st.stop()

st.subheader("📋 Kontu sadalījums pa grupām un apakšgrupām")
cls_df["_sort"] = cls_df["Grupa"].map(lambda g: GROUP_SORT_ORDER.get(g, 99))
cls_df = cls_df.sort_values(by=["_sort", "Konts"], kind="mergesort").drop(columns=["_sort"]).reset_index(drop=True)
S.cls_df = cls_df

c1, c2, c3, c4 = st.columns(4)
c1.metric("🟡 S (Ieņēmumi)", f"{cls_df['Grupa'].str.startswith('S_').sum()} konti")
c2.metric("🟠 TVC (Mainīgās)", f"{cls_df['Grupa'].str.startswith('TVC').sum()} konti")
c3.metric("🟤 OE (Uzturēšana)", f"{cls_df['Grupa'].str.startswith('OE_').sum()} konti")
c4.metric("⚪ Izslēgtie (Virtuālie)", f"{(cls_df['Grupa'] == 'EXCLUDED').sum()} konti")

view_df = pd.DataFrame({
    "Konts": cls_df["Konts"],
    "Lielā grupa": cls_df["Grupa"].map(get_main_cat),
    "Apakšgrupa (Rinda atskaitē)": cls_df["Grupa"].map(SUBGROUP_LABELS),
})

full_editor_height = (len(view_df) + 1) * 36 + 12

edited_view = st.data_editor(
    view_df, hide_index=True, use_container_width=True, height=full_editor_height, key="clean_cls_editor", disabled=["Konts"],
    column_config={
        "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
        "Lielā grupa": st.column_config.SelectboxColumn("Lielā grupa", options=list(MAIN_GROUP_TO_DEFAULT_SUB.keys()), required=True, width="medium"),
        "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn("Apakšgrupa / Rinda atskaitē", options=list(SUBGROUP_LABELS.values()), required=True, width="large"),
    }
)

updated_groups = []
changed = False

for idx, row in cls_df.iterrows():
    old_grp = row["Grupa"]
    old_main = get_main_cat(old_grp)
    old_sub = SUBGROUP_LABELS[old_grp]

    matching_edited = edited_view[edited_view["Konts"] == row["Konts"]]
    if not matching_edited.empty:
        new_main = matching_edited.iloc[0]["Lielā grupa"]
        new_sub = matching_edited.iloc[0]["Apakšgrupa (Rinda atskaitē)"]

        if new_sub != old_sub:
            updated_groups.append(LABEL_TO_KEY[new_sub])
            changed = True
        elif new_main != old_main:
            updated_groups.append(MAIN_GROUP_TO_DEFAULT_SUB[new_main])
            changed = True
        else:
            updated_groups.append(old_grp)
    else:
        updated_groups.append(old_grp)

if changed:
    cls_df["Grupa"] = updated_groups
    cls_df["Ieņēmums"] = cls_df["Grupa"].str.startswith("S_")
    S.cls_df = cls_df
    st.rerun()

# ------------------------------------------------------------------ 3. GALA APRĒĶINS
st.header("2. TOC-FIN Diagnostika un Lēmumu Atbalsts")
if st.button("🚀 Apstiprināt klasifikāciju un veikt TOC-FIN aprēķinu", type="primary"):
    mapping = {r["id"]: {"group": r["Grupa"], "is_income": bool(str(r["Grupa"]).startswith("S_") or r["Ieņēmums"])} for _, r in cls_df.iterrows()}
    pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
    reported = an.ex.lines[pos].vals if pos is not None else None
    
    res = compute_toc(an.leaves, mapping, an.ex.period_cols, reported)
    S.result = res
    if profile:
        profile.save(an.leaves, mapping)
    S.pop("commentary", None)
    S.chat = []

if "result" in S:
    res = S.result

    main_tab1, main_tab2 = st.tabs([
        "📊 1. TOC-FIN Analytics (Vēsturiskā diagnostika un Trauksmes)", 
        "⚡ 2. TOC-FIN Decision Support (Procentu % Simulācija pa periodiem)"
    ])

    with main_tab1:
        xl_f, xl_v = io.BytesIO(), io.BytesIO()
        build_excel(res, xl_f, formulas=True)
        build_excel(res, xl_v, formulas=False)
        d1, d2 = st.columns(2)
        d1.download_button("📥 Lejupielādēt Excel ar formulām (.xlsx)", xl_f.getvalue(), "TOC_Finansu_Diagnostika.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d2.download_button("📥 Lejupielādēt Excel ar vērtībām (.xlsx)", xl_v.getvalue(), "TOC_Finansu_Diagnostika_vertibas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Finanšu pārskata tabula (T* skats)")
        st.markdown(render_unified_toc_tables_html(res.amounts, res.main, res.kpi, res.row_kinds), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📝 Vadības kopsavilkums un stratēģiskie secinājumi")
        tables_md = (res.amounts.round(0).to_markdown() + "\n\n" + res.kpi.round(3).to_markdown())
        
        knowledge_context = f"TOC-FIN zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
        COMMENT_SYSTEM = (
            "Tu esi pieredzējis TOC-FIN finanšu direktora padomnieks un vadības konsultants. "
            "Uzrunā lietotāju kā Vadības komandu. Nekad nelieto grāmatvedības pašizmaksas klišejas. "
            f"{knowledge_context}"
            "Raksti latviešu valodā, strukturēti, asi un uz lēmumiem vērsti.")

        if "commentary" not in S:
            if st.button("📝 Ģenerēt vadības analīzi un rīcības plānu", type="primary"):
                with st.spinner("Sagatavo TOC-FIN vadības kopsavilkumu..."):
                    r = get_client().messages.create(
                        model=MODEL, max_tokens=2500, system=COMMENT_SYSTEM,
                        messages=[{"role": "user", "content": f"Aprēķinātās tabulas (summas EUR):\n\n{tables_md}\n\nUzraksti vadības analīzi."}])
                    S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    st.rerun()
        else:
            st.markdown(S.commentary)

    with main_tab2:
        st.subheader("⚡ Interaktīvā scenāriju un mērķu simulācija (% izmaiņas pret izvēlēto periodu)")
        all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
        if not all_months:
            all_months = list(res.amounts.columns)

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            preset = st.selectbox("Ātrā perioda izvēle:", options=[f"Visi mēneši ({len(all_months)} mēn.)", "Pirmie 6 mēneši" if len(all_months) >= 6 else "Pirmie 3", "Manuāla izvēle"])
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

        pct_changes = {}
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("🟡 **Ieņēmumi (S apakšgrupas)**")
            for sc_name in s_children:
                pct_changes[sc_name] = st.number_input(f"{sc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{sc_name}")
        with sc2:
            st.markdown("🟠 **Mainīgās (TVC)**")
            for tc_name in tvc_children:
                pct_changes[tc_name] = st.number_input(f"{tc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{tc_name}")
        with sc3:
            st.markdown("🟤 **Darbības izdevumi (OE)**")
            for oc_name in oe_children:
                def_v = 5.0 if "alga" in oc_name.lower() else 0.0
                pct_changes[oc_name] = st.number_input(f"{oc_name} (%)", value=def_v, step=1.0, format="%.1f", key=f"sim_{oc_name}")

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

        def sim_kpi_for(col_name: str, col_vals: pd.Series) -> dict[str, float]:
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

        sim_kpi_df = pd.DataFrame({c: sim_kpi_for(c, sim_amounts[c]) for c in sim_amounts.columns})
        sim_kpi_df["Δ Izmaiņa (€)"] = sim_kpi_df["Simulācija KOPĀ"] - sim_kpi_df["Bāze KOPĀ"]

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        ds_val, dt_val, doe_val, dnp_val = diff_sum["S"], diff_sum["T"], diff_sum["OE"], diff_sum["NP"]

        m1.metric("Simulētais S", f"{fmt_num(sim_sum['S'])} €", delta=f"{ds_val:+,.0f} €".replace(",", " "))
        m2.metric("Simulētais T", f"{fmt_num(sim_sum['T'])} €", delta=f"{dt_val:+,.0f} €".replace(",", " "))
        m3.metric("Simulētais OE", f"{fmt_num(sim_sum['OE'])} €", delta=f"{doe_val:+,.0f} €".replace(",", " "), delta_color="inverse")
        m4.metric("Simulētā Neto Peļņa (NP)", f"{fmt_num(sim_sum['NP'])} €", delta=f"{dnp_val:+,.0f} €".replace(",", " "))

        st.subheader("📊 Simulētā Finanšu un Produktivitātes tabula")
        st.markdown(render_unified_toc_tables_html(sim_amounts, sim_main_df, sim_kpi_df, res.row_kinds, title_kpi="📈 Simulētie Produktivitātes un atdeves rādītāji"), unsafe_allow_html=True)

if st.button("🔄 Sākt no jauna"):
    st.session_state.clear()
    st.rerun()

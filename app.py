"""
app.py — TOC-FIN Vadības lēmumu un finanšu diagnostikas sistēma (v9)
Uzlabojumi:
- Kontu tabula automātiski sakārtota secībā: S -> TVC -> OE Algas -> OE Pārējais -> Izslēgtie
- Kontu tabula pilnā augstumā (bez vertikālās skrollēšanas)
- Finanšu pārskata un KPI tabulas pilnā izmērā bez horizontālās vai vertikālās skrollēšanas
- Kreisā kolonna iekrāsota rindas krāsā; Galvenās grupas Bold, apakšgrupas bez Bold
- Procentu ailēs virsraksts tikai '%', šauras kolonnas, veseli skaitļi bez komata (piem. 33%)
- Biezākas līnijas starp mēnešu blokiem (mēnesis + %) un starp galvenajām TOC grupām
"""
import hashlib
import io
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

from profile_store import ProfileStore
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
no kapacitātes uzturēšanas izmaksām ($OE$), lai novērtētu patieso caurlaidību ($T$), produktivitāti ($PR = T / OE$) un modelētu nākotnes lēmumus ($\Delta T - \Delta OE$).
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
    code = GROUPS.get(group_key, ("", "OE"))[1]
    return {
        "S": "🟡 S (Ieņēmumi)",
        "TVC": "🟠 TVC (Mainīgās izmaksas)",
        "OE": "🟤 OE (Uzturēšanas izdevumi)",
        "EXCL": "⚪ Izslēgtie (Virtuālā nauda)"
    }.get(code, "🟤 OE (Uzturēšanas izdevumi)")


# Stingra kārtošanas secība: S -> TVC -> OE Algas -> OE Pārējais -> Izslēgtie
GROUP_SORT_ORDER = {
    "S": 1,
    "TVC_MATERIALS": 2,
    "TVC_DELIVERY": 3,
    "OE_ADMIN_WAGES": 4,
    "OE_WORKER_WAGES": 5,
    "OE_MARKETING": 6,
    "OE_PLATFORMS": 7,
    "OE_PREMISES": 8,
    "OE_OFFICE_IT": 9,
    "OE_FIN_TAX": 10,
    "OE_OTHER": 11,
    "EXCLUDED": 12,
}


def render_toc_main_html(res) -> str:
    """Ģenerē pilna platuma HTML tabulu bez skrollēšanas, ar iekrāsotu kreiso aili un biezajām līnijām."""
    periods = list(res.amounts.columns)
    row_names = list(res.amounts.index)
    kinds = [res.row_kinds.get(rn, "OE_CHILD") for rn in row_names]

    html = ["""
    <style>
    .toc-table-wrap { width: 100%; overflow-x: visible; margin-bottom: 18px; }
    table.toc-fin { width: 100%; border-collapse: collapse; font-family: 'Calibri', sans-serif; font-size: 12px; table-layout: auto; }
    table.toc-fin th { background-color: #EAECEE; color: #111111; font-weight: 700; text-align: center; padding: 5px 3px; border-top: 2px solid #404040; border-bottom: 2px solid #404040; white-space: nowrap; }
    table.toc-fin td { padding: 4px 4px; white-space: nowrap; }
    table.toc-fin td.lbl-main { text-align: left; font-weight: 700; padding-left: 6px; border-left: 2px solid #404040; border-right: 2px solid #404040; }
    table.toc-fin td.lbl-sub { text-align: left; font-weight: 400; padding-left: 16px; border-left: 2px solid #404040; border-right: 2px solid #404040; }
    table.toc-fin td.val-col { text-align: right; border-left: 1px solid #BFBFBF; }
    table.toc-fin td.pct-col { text-align: right; width: 34px; font-size: 11px; padding-right: 4px; border-left: 1px dotted #D0D3D4; border-right: 2px solid #595959; }
    table.toc-fin th.th-val { border-left: 1px solid #BFBFBF; }
    table.toc-fin th.th-pct { width: 34px; font-size: 11px; border-left: 1px dotted #BFBFBF; border-right: 2px solid #595959; }
    table.toc-fin th.th-lbl { text-align: left; padding-left: 6px; border-left: 2px solid #404040; border-right: 2px solid #404040; }
    </style>
    <div class="toc-table-wrap"><table class="toc-fin">
    <thead><tr>
    <th class="th-lbl">Rādītājs / Kontugrupa</th>
    """]

    for p in periods:
        html.append(f'<th class="th-val">{p}</th><th class="th-pct">%</th>')
    html.append("</tr></thead><tbody>")

    for i, rname in enumerate(row_names):
        kind = kinds[i]
        next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
        fill_hex, is_bold, text_hex = FILLS.get(kind, FILLS["OE_CHILD"])

        # Noteicam, vai zem šīs rindas vajag biezo horizontālo līniju (grupas beigas)
        thick_bottom = False
        if kind == "S":
            thick_bottom = True
        elif kind in ("TVC", "TVC_CHILD") and next_kind not in ("TVC", "TVC_CHILD"):
            thick_bottom = True
        elif kind == "T":
            thick_bottom = True
        elif kind in ("OE", "OE_CHILD") and next_kind not in ("OE", "OE_CHILD"):
            thick_bottom = True
        elif kind == "NP":
            thick_bottom = True
        elif i == len(row_names) - 1:
            thick_bottom = True

        # Vai virs šīs rindas ir apakšgrupu atdalījums no galvenās rindas
        border_top = "1px solid #BFBFBF"
        border_bottom = "2.5px solid #404040" if thick_bottom else "1px solid #BFBFBF"
        font_w = "700" if is_bold else "400"
        lbl_cls = "lbl-main" if is_bold else "lbl-sub"

        row_style = f"background-color: #{fill_hex}; color: #{text_hex}; font-weight: {font_w}; border-top: {border_top}; border-bottom: {border_bottom};"
        html.append(f'<tr style="{row_style}">')
        html.append(f'<td class="{lbl_cls}">{rname}</td>')

        for p in periods:
            val_str = fmt_num(res.main.loc[rname, p])
            pct_str = fmt_pct_int(res.main.loc[rname, f"%|{p}"])
            html.append(f'<td class="val-col">{val_str}</td><td class="pct-col">{pct_str}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


def render_toc_kpi_html(res) -> str:
    """Ģenerē pilna platuma KPI tabulu bez skrollēšanas."""
    kp = res.kpi
    cols = list(kp.columns)

    html = ["""
    <div class="toc-table-wrap"><table class="toc-fin">
    <thead><tr>
    <th class="th-lbl">Produktivitātes rādītājs</th>
    """]
    for c in cols:
        html.append(f'<th style="border-right: 2px solid #595959;">{c}</th>')
    html.append("</tr></thead><tbody>")

    for i, idx_name in enumerate(kp.index):
        b_bot = "2px solid #404040" if i == len(kp.index) - 1 else "1px solid #BFBFBF"
        html.append(f'<tr style="background-color: #F2F4F4; color: #000000; border-bottom: {b_bot};">')
        html.append(f'<td class="lbl-main" style="font-weight: 600;">{idx_name}</td>')
        for c in cols:
            v = kp.loc[idx_name, c]
            if pd.isna(v):
                s = ""
            elif idx_name.startswith("T%"):
                s = f"{v * 100:.0f}%"
            else:
                s = f"{v:.2f}"
            html.append(f'<td style="text-align: right; padding: 4px 6px; border-right: 2px solid #595959; font-weight: 600;">{s}</td>')
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

# ------------------------------------------------------------------ FONĀ: STRUKTŪRA UN PĀRBAUDE
if "structure" not in S:
    with st.spinner("Nolasa faila struktūru un attīra datus no kopsummu dublikātiem..."):
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

# ------------------------------------------------------------------ 2. KONTU KLASIFIKĀCIJA
st.header("1. Kontu ekonomiskā klasifikācija")

profile = ProfileStore("default")
if "cls_df" not in S:
    if st.button("🔍 Sākt kontu TOC-FIN klasifikāciju", type="primary"):
        with st.spinner("Aģents izvērtē kontus pēc TOC-FIN principiem un situācijas konteksta..."):
            try:
                items = classify_with_claude(get_client(), MODEL, an.leaves)
                by_id = {i["id"]: i for i in items}
                rows = []
                for lf in an.leaves:
                    i = by_id[lf.id]
                    display_name = f"{lf.label} ({lf.path})" if lf.path and lf.label.lower().startswith(("personāla", "sociālās", "telpu", "pārējās")) else lf.label
                    rows.append(dict(
                        id=lf.id,
                        Konts=display_name,
                        Grupa=i["group"],
                        Ieņēmums=bool(i["is_income"]),
                        Jautājums=i.get("question") or ""
                    ))
                S.cls_df = pd.DataFrame(rows)
                st.rerun()
            except Exception as e:
                st.error(f"Kļūda klasifikācijā: {e}")
    if "cls_df" not in S:
        st.stop()

cls_df: pd.DataFrame = S.cls_df

# Interaktīva neskaidro kontu precizēšana
ambiguous_rows = cls_df[cls_df["Jautājums"].astype(str).str.strip() != ""]

if len(ambiguous_rows) > 0 and not S.get("clarifications_done", False):
    st.subheader("❓ Nepieciešams vadības lēmums par dažām pozīcijām")
    st.caption("Aģents ir identificējis pozīcijas, kuru klasifikācija ir atkarīga no jūsu biznesa situācijas:")

    clarifications = {}
    for idx, r in ambiguous_rows.iterrows():
        st.markdown(f"**📌 {r['Konts']}**")
        st.write(f"👉 **Aģenta jautājums:** {r['Jautājums']}")
        st.info("💡 **TOC-FIN orientieris:** **S** = reāli ienākošā nauda | **TVC** = mainās 1:1 ar katru pārdoto vienību | **OE** = perioda uzturēšanas izdevums | **Izslēgt** = virtuāls grāmatvedības ieraksts (amortizācija).")

        grp = str(r["Grupa"])
        default_idx = 0 if grp == "S" else (1 if "TVC" in grp else (3 if grp == "EXCLUDED" else 2))

        choice = st.radio(
            f"Kā klasificēt '{r['Konts']}' šajā situācijā?",
            options=[
                "🟡 S — Pievienot pie Ieņēmumiem (Sales)",
                "🟠 TVC — Patiesi mainīga izmaksa (1:1 ar pārdoto vienību)",
                "🟤 OE — Darbības un uzturēšanas izdevums (Operating Expense)",
                "⚪ Izslēgt — Virtuāla nauda (amortizācija / pārvērtēšana)"
            ],
            index=default_idx,
            key=f"radio_{r['id']}"
        )
        if choice.startswith("🟡"):
            clarifications[r["id"]] = ("S", True)
        elif choice.startswith("🟠"):
            clarifications[r["id"]] = ("TVC_DELIVERY" if "piegād" in r["Konts"].lower() else "TVC_MATERIALS", False)
        elif choice.startswith("⚪"):
            clarifications[r["id"]] = ("EXCLUDED", False)
        else:
            clarifications[r["id"]] = (grp if grp.startswith("OE_") else "OE_OTHER", bool(r["Ieņēmums"]))
        st.markdown("---")

    if st.button("Apstiprināt atbildes un atvērt kontu pārskatu", type="primary"):
        for row_id, (new_group, is_inc) in clarifications.items():
            cls_df.loc[cls_df["id"] == row_id, "Grupa"] = new_group
            cls_df.loc[cls_df["id"] == row_id, "Ieņēmums"] = is_inc
            cls_df.loc[cls_df["id"] == row_id, "Jautājums"] = ""
        S.cls_df = cls_df
        S.clarifications_done = True
        st.rerun()
    st.stop()

# --- SKAIDRS UN SAKĀRTOTS KONTU SADALĪJUMS PILNĀ AUGSTUMĀ ---
st.subheader("📋 Kontu sadalījums pa grupām un apakšgrupām")
st.caption("Konti ir automātiski sakārtoti secībā: **S → TVC → OE Algas → OE Pārējais → Izslēgtie**. Jūs varat brīvi mainīt gan Lielo grupu, gan Apakšgrupu:")

# Sakārtojam cls_df pēc stingrās TOC secības
cls_df["_sort"] = cls_df["Grupa"].map(lambda g: GROUP_SORT_ORDER.get(g, 99))
cls_df = cls_df.sort_values(by=["_sort", "Konts"], kind="mergesort").drop(columns=["_sort"]).reset_index(drop=True)
S.cls_df = cls_df

c1, c2, c3, c4 = st.columns(4)
c1.metric("🟡 S (Ieņēmumi)", f"{(cls_df['Grupa'] == 'S').sum()} konti")
c2.metric("🟠 TVC (Mainīgās)", f"{cls_df['Grupa'].str.startswith('TVC').sum()} konti")
c3.metric("🟤 OE (Uzturēšana)", f"{cls_df['Grupa'].str.startswith('OE_').sum()} konti")
c4.metric("⚪ Izslēgtie (Virtuālie)", f"{(cls_df['Grupa'] == 'EXCLUDED').sum()} konti")

view_df = pd.DataFrame({
    "Konts": cls_df["Konts"],
    "Lielā grupa": cls_df["Grupa"].map(get_main_cat),
    "Apakšgrupa (Rinda atskaitē)": cls_df["Grupa"].map(SUBGROUP_LABELS),
})

# Pilns augstums, lai nekas nav jāskrollē
full_editor_height = (len(view_df) + 1) * 36 + 12

edited_view = st.data_editor(
    view_df,
    hide_index=True,
    use_container_width=True,
    height=full_editor_height,
    key="clean_cls_editor",
    disabled=["Konts"],
    column_config={
        "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
        "Lielā grupa": st.column_config.SelectboxColumn(
            "Lielā grupa (S / TVC / OE / Izslēgtie)",
            options=list(MAIN_GROUP_TO_DEFAULT_SUB.keys()),
            required=True,
            width="medium",
        ),
        "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn(
            "Apakšgrupa / Rinda atskaitē (OE sadalījums u.c.)",
            options=list(SUBGROUP_LABELS.values()),
            required=True,
            width="large",
        ),
    }
)

updated_groups = []
changed = False
for idx in range(len(cls_df)):
    old_grp = cls_df.iloc[idx]["Grupa"]
    old_main = get_main_cat(old_grp)
    old_sub = SUBGROUP_LABELS[old_grp]

    new_main = edited_view.iloc[idx]["Lielā grupa"]
    new_sub = edited_view.iloc[idx]["Apakšgrupa (Rinda atskaitē)"]

    if new_sub != old_sub:
        g_code = LABEL_TO_KEY[new_sub]
        updated_groups.append(g_code)
        changed = True
    elif new_main != old_main:
        g_code = MAIN_GROUP_TO_DEFAULT_SUB[new_main]
        updated_groups.append(g_code)
        changed = True
    else:
        updated_groups.append(old_grp)

if changed:
    cls_df["Grupa"] = updated_groups
    cls_df["Ieņēmums"] = cls_df["Grupa"] == "S"
    S.cls_df = cls_df
    st.rerun()

# ------------------------------------------------------------------ 3. GALA APRĒĶINS
st.header("2. TOC-FIN Diagnostika un Lēmumu Atbalsts")
if st.button("🚀 Apstiprināt klasifikāciju un veikt TOC-FIN aprēķinu", type="primary"):
    mapping = {r["id"]: {"group": r["Grupa"], "is_income": bool(r["Grupa"] == "S" or r["Ieņēmums"])} for _, r in cls_df.iterrows()}
    pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
    reported = an.ex.lines[pos].vals if pos is not None else None
    S.result = compute_toc(an.leaves, mapping, an.ex.period_cols, reported)
    if profile:
        profile.save(an.leaves, mapping)
    S.pop("commentary", None)
    S.chat = []

if "result" in S:
    res = S.result

    main_tab1, main_tab2 = st.tabs([
        "📊 1. TOC-FIN Analytics (Vēsturiskā diagnostika un Trauksmes)", 
        "⚡ 2. TOC-FIN Decision Support (ΔT / ΔOE Lēmumu Simulators)"
    ])

    with main_tab1:
        xl_f, xl_v = io.BytesIO(), io.BytesIO()
        build_excel(res, xl_f, formulas=True)
        build_excel(res, xl_v, formulas=False)
        d1, d2 = st.columns(2)
        d1.download_button("📥 Lejupielādēt Excel ar formulām (.xlsx)", xl_f.getvalue(), "TOC_Finansu_Diagnostika.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d2.download_button("📥 Lejupielādēt Excel ar vērtībām (.xlsx)", xl_v.getvalue(), "TOC_Finansu_Diagnostika_vertibas.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("Finanšu pārskata tabula (T* skats)")
        st.markdown(render_toc_main_html(res), unsafe_allow_html=True)

        st.subheader("📈 Produktivitātes un atdeves rādītāji")
        st.markdown(render_toc_kpi_html(res), unsafe_allow_html=True)

        # --- VADĪBAS TRAUKSMES PANELIS (DECISION GUARDRAILS) ---
        st.markdown("---")
        st.subheader("🚦 Vadības Trauksmes Panelis (Decision Guardrails)")
        
        month_cols = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
        alerts_found = False

        if month_cols:
            pr_series = res.kpi.loc["PR = T/OE", month_cols]
            bad_pr_months = pr_series[pr_series < 1.0].index.tolist()
            if bad_pr_months:
                alerts_found = True
                st.error(f"🔴 **Kritiska trauksme (PR < 1.0):** Periodos **{', '.join(bad_pr_months)}** uzturēšanas izmaksas ($OE$) pārsniedza ģenerēto caurlaidību ($T$). Uzņēmums šajos mēnešos dedzināja naudas rezerves.")

            t_pct_series = res.kpi.loc["T% = T/S", month_cols]
            mean_t_pct = t_pct_series.mean()
            low_t_months = t_pct_series[t_pct_series < (mean_t_pct - 0.07)].index.tolist()
            if low_t_months:
                alerts_found = True
                st.warning(f"🟡 **Caurlaidības / Krājumu brīdinājums:** Periodos **{', '.join(low_t_months)}** caurlaidības marža ($T\\%$) nokritās vairāk nekā par 7 procentpunktiem zem vidējā ({mean_t_pct:.0%}). Pārbaudiet, vai šajā mēnesī netika iepirkta liela materiālu partija noliktavai (iesaldējot naudu $I$) vai piešķirtas pārmērīgas cenu atlaides.")

            oe_series = res.amounts.loc["OE", month_cols]
            mean_oe = oe_series.mean()
            high_oe_months = oe_series[oe_series > (mean_oe * 1.15)].index.tolist()
            if high_oe_months:
                alerts_found = True
                st.warning(f"🟠 **Kapacitātes izmaksu ($OE$) lēciens:** Periodos **{', '.join(high_oe_months)}** darbības izdevumi pārsniedza vidējo līmeni ({fmt_num(mean_oe)} €) par vairāk nekā 15%. Pārbaudiet, vai šis $OE$ pieaugums deva adekvātu $\\Delta T$ pieaugumu.")

            if not alerts_found:
                st.success("🟢 **Sistēmas darbība stabila:** Visos periodos $PR \\ge 1.0$ un nav konstatētas kritiskas $OE$ vai $T\\%$ novirzes.")

        # --- VADĪBAS KOPSAVILKUMS ---
        st.markdown("---")
        st.subheader("📝 Vadības kopsavilkums un stratēģiskie secinājumi")
        tables_md = (res.amounts.round(0).to_markdown() + "\n\n" + res.kpi.round(3).to_markdown())
        
        knowledge_context = f"TOC-FIN zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
        COMMENT_SYSTEM = (
            "Tu esi pieredzējis TOC-FIN finanšu direktora padomnieks un vadības konsultants. "
            "Uzrunā lietotāju kā Vadības komandu. Nekad nelieto grāmatvedības pašizmaksas klišejas. "
            "Tev ir doti jau precīzi aprēķināti skaitļi — citē tikai tos, kas ir tabulās. "
            f"{knowledge_context}"
            "Raksti latviešu valodā, strukturēti, asi un uz lēmumiem vērsti.")

        if "commentary" not in S:
            if st.button("📝 Ģenerēt vadības analīzi un rīcības plānu", type="primary"):
                with st.spinner("Sagatavo TOC-FIN vadības kopsavilkumu..."):
                    r = get_client().messages.create(
                        model=MODEL, max_tokens=2500, system=COMMENT_SYSTEM,
                        messages=[{"role": "user", "content":
                                   f"Aprēķinātās tabulas (summas EUR):\n\n{tables_md}\n\n"
                                   "Uzraksti vadības analīzi:\n"
                                   "### 1. Galvenās tendences un Caurlaidības (T*) dinamika\n"
                                   "### 2. Produktivitātes (PR) un darbaspēka atdeves vērtējums\n"
                                   "### 3. Vadības fokusēšanās soļi un rīcības plāns (3 konkrēti lēmumi)"}])
                    S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    st.rerun()
        else:
            st.markdown(S.commentary)

            st.subheader("💬 Diskusija ar TOC-FIN aģentu")
            S.setdefault("chat", [])
            for m in S.chat:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            q = st.chat_input("Uzdodiet jautājumu par datiem vai konkrētu biznesa lēmumu...")
            if q:
                S.chat.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Analizē..."):
                        r = get_client().messages.create(
                            model=MODEL, max_tokens=1500,
                            system=COMMENT_SYSTEM + f"\n\nAprēķinātās tabulas:\n{tables_md}\n\nKopsavilkums:\n{S.commentary}",
                            messages=S.chat)
                        reply = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                        st.markdown(reply)
                S.chat.append({"role": "assistant", "content": reply})

    # ==================================================================
    # 2. CILNE: TOC-FIN DECISION SUPPORT (DELTA SIMULATORS)
    # ==================================================================
    with main_tab2:
        st.subheader("⚡ Jauna lēmuma, pasūtījuma vai investīcijas simulācija (Δ skats)")
        st.markdown("""
        Šis rīks ļauj vadības komandai novērtēt konkrētu biznesa lēmumu (piemēram, papildu pasūtījumu ar atlaidi, jauna darbinieka pieņemšanu vai iekārtas iegādi) 
        pēc tā **faktiskajām papildu izmaiņām ($\Delta T$, $\Delta OE$, $\Delta I$)**, nevis vidējās grāmatvedības pašizmaksas.
        """)

        ref_col = next((c for c in res.amounts.columns if "AVG" in c), res.amounts.columns[-1])
        base_s = float(res.amounts.loc["S", ref_col])
        base_tvc = float(res.amounts.loc["TVC", ref_col])
        base_t = float(res.amounts.loc["T", ref_col])
        base_oe = float(res.amounts.loc["OE", ref_col])
        base_np = float(res.amounts.loc["NP", ref_col])
        base_t_pct = (base_t / base_s) if base_s else 0.5

        st.info(f"📌 **Salīdzinājuma bāze ({ref_col}):** Apgrozījums $S = {fmt_num(base_s)}$ € | Caurlaidība $T = {fmt_num(base_t)}$ € ({base_t_pct:.0%}) | Darbības izmaksas $OE = {fmt_num(base_oe)}$ € | Neto peļņa $NP = {fmt_num(base_np)}$ €")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("#### 1. Ietekme uz Caurlaidību ($\Delta T$)")
            delta_s = st.number_input("Papildu pārdošanas ieņēmumi (ΔS, EUR mēnesī)", value=20000, step=1000)
            default_dtvc = int(round(delta_s * (1.0 - base_t_pct)))
            delta_tvc = st.number_input("Papildu tiešās materiālu/piegādes izmaksas (ΔTVC, EUR)", value=default_dtvc, step=500)
            delta_t = delta_s - delta_tvc
            st.metric("Papildu Caurlaidība (ΔT = ΔS - ΔTVC)", f"{fmt_num(delta_t)} €")

        with col_s2:
            st.markdown("#### 2. Ietekme uz Kapacitāti ($\Delta OE$ un $\Delta I$)")
            delta_oe = st.number_input("Papildu uzturēšanas izmaksas (ΔOE, EUR mēnesī)", value=2000, step=500)
            delta_i = st.number_input("Nepieciešamā papildu investīcija / iesaldētā nauda (ΔI, EUR)", value=0, step=1000)
            ccr_impact = st.selectbox("Ietekme uz uzņēmuma ierobežojumu (CCR / vājāko posmu):", [
                "Nepārslogo CCR (tiek izmantota brīvā kapacitāte)",
                "Noslogo CCR, bet neizspiež citus pasūtījumus",
                "UZMANĪBU: Izspiež citus esošos pasūtījumus uz CCR!"
            ])

        with col_s3:
            st.markdown("#### 3. Lēmuma Rezultāts un Guardrails")
            delta_np = delta_t - delta_oe
            new_np = base_np + delta_np
            new_pr = (base_t + delta_t) / (base_oe + delta_oe) if (base_oe + delta_oe) else 0
            old_pr = base_t / base_oe if base_oe else 0

            st.metric("Papildu Neto Peļņa (ΔNP = ΔT - ΔOE)", f"{fmt_num(delta_np)} €",
                      delta=f"Jaunā mēneša NP: {fmt_num(new_np)} €")
            st.metric("Jaunais Produktivitātes Rādītājs (PR)", f"{new_pr:.2f}",
                      delta=f"{new_pr - old_pr:+.2f} pret bāzi ({old_pr:.2f})")

            if delta_i > 0:
                annual_roi = (delta_np * 12) / delta_i
                payback_months = delta_i / delta_np if delta_np > 0 else float("inf")
                st.metric("Investīcijas Gada Atdeve (ROI)", f"{annual_roi:.0%}",
                          delta=f"Atmaksāšanās: {payback_months:.1f} mēn." if delta_np > 0 else "Neatmaksājas")

        st.markdown("---")
        if delta_np > 0 and "Izspiež" not in ccr_impact:
            st.success(f"✅ **TOC-FIN Lēmuma slēdziens:** Lēmums ir finansiāli pamatots ($\Delta T - \Delta OE = +{fmt_num(delta_np)}$ €/mēn.). Pat ja tradicionālā grāmatvedības pilnā pašizmaksa šim darījumam izskatītos 'nerentabla', brīvās jaudas apstākļos tas tieši palielina uzņēmuma tīro peļņu un naudas plūsmu.")
        elif delta_np > 0 and "Izspiež" in ccr_impact:
            st.warning(f"⚠️ **Trauksme par CCR noslodzi:** Lai gan $\Delta NP$ ir pozitīvs ($+{fmt_num(delta_np)}$ €), šis lēmums izspiež citus pasūtījumus ierobežojumā! Pārliecinieties, ka jaunā pasūtījuma caurlaidība uz 1 CCR minūti ($T_u / CCR_{{min}}$) ir augstāka nekā izspiestajiem produktiem.")
        else:
            st.error(f"🛑 **Stop-krāns (Negatīvs $\Delta NP$):** Šis lēmums samazina uzņēmuma peļņu par **{fmt_num(abs(delta_np))} €** mēnesī, jo papildu uzturēšanas izmaksas ($\Delta OE$) pārsniedz iegūto caurlaidību ($\Delta T$). Nepieciešams stratēģisks pamatojums vai izmaksu pārskatīšana.")

if st.button("🔄 Sākt no jauna"):
    st.session_state.clear()
    st.rerun()

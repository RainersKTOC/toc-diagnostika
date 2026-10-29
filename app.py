"""
app.py — TOC-FIN Vadības lēmumu un finanšu diagnostikas sistēma (v6)
Ietver:
- Automātisku fona struktūras un kopsummu pārbaudi (bez tehniskiem logiem)
- Interaktīvus jautājumus šaubīgajām pozīcijām (TVC vs OE)
- Kontu sadalījumu pa cilnēm (S, TVC, OE, Izslēgtie grāmatvedības trokšņi)
- TOC-FIN Analytics: Vēsturisko T* tabulu, KPI un krāsainu Excel eksportu
- Vadības Trauksmes Mehānismu (Decision Guardrails)
- TOC-FIN Decision Support: Interaktīvu ΔT / ΔOE / ΔI lēmumu un pasūtījumu simulatoru
"""
import hashlib
import io
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

from profile_store import ProfileStore
from toc_engine import (GROUPS, FILLS, build_excel, classify_with_claude, compute_toc)
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

# --- INFORMĀCIJAS LOGS PAR TOC-FIN ---
with st.expander("ℹ️ Uzzināt vairāk: Kāpēc TOC-FIN nodala vadības lēmumus no grāmatvedības pašizmaksas?"):
    st.markdown("""
    **Organizatoriskais princips:** Biznesa lēmumi pieder vadības komandai (ieskaitot finanšu direktoru). Grāmatvedības uzdevums ir precīza darījumu uzskaite un nodokļu atbilstība, nevis biznesa lēmumu lietderības noteikšana.

    * **S (Sales / Neto apgrozījums)** — Reāli ienākošā nauda no pircējiem par pārdoto produkciju.
    * **TVC (Truly Variable Costs / Patiesi mainīgās izmaksas)** — Tikai izdevumi, kas mainās **1:1 ar katru pārdoto vienību** (izejvielas, materiāli un tiešais transports klientam). Ja nepārdod nevienu vienību, TVC = 0.
    * **T (Throughput / Caurlaidība)** — Jaunā naudas masa, ko uzņēmums ģenerē: $$T = S - TVC$$.
    * **OE (Operating Expenses / Uzturēšanas izdevumi)** — Visa nauda, ko uzņēmums tērē, lai uzturētu sistēmas jaudu (visas algas, telpas, IT, mārketings). Darbaspēks ir OE, nevis TVC!
    * **NP (Net Profit / Neto peļņa)** — $$NP = T - OE$$.
    * **PR (Productivity Ratio / Produktivitātes rādītājs)** — $$PR = T / OE$$ (cik eiro caurlaidības ģenerē katrs uzturēšanā ieguldītais eiro).
    * **Vēsturiskais $T^*$ pret Lēmumu $\Delta T$:** Vēsturiskie PZA dati rāda sistēmas tendences ($T^*$), savukārt katru jaunu pasūtījumu vai investīciju vadība vērtē pēc tīrās izmaiņas: $$\Delta NP = \Delta T - \Delta OE > 0$$.
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
        with st.spinner("Analizē kontus pēc ekonomiskās būtības (S, TVC, OE, Grāmatvedības troksnis)..."):
            try:
                known, unknown = profile.lookup(an.leaves) if profile else ({}, an.leaves)
                items = classify_with_claude(get_client(), MODEL, unknown) if unknown else []
                by_id = {i["id"]: i for i in items}
                rows = []
                for lf in an.leaves:
                    if lf.id in known:
                        m = known[lf.id]
                        rows.append(dict(id=lf.id, Sadaļa=lf.path, Konts=lf.label, Grupa=m["group"],
                                         Ieņēmums=bool(m["is_income"]), Drošība="profils", Jautājums=""))
                    else:
                        i = by_id[lf.id]
                        rows.append(dict(id=lf.id, Sadaļa=lf.path, Konts=lf.label, Grupa=i["group"],
                                         Ieņēmums=bool(i["is_income"]), Drošība=i["confidence"],
                                         Jautājums=i.get("question") or ""))
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
    st.subheader("❓ Nepieciešams vadības precizējums par robežpozīcijām")
    st.caption("Izvērtējiet šos posteņus pēc to reālās naudas plūsmas dabas (nevis grāmatvedības konta nosaukuma):")

    clarifications = {}
    for idx, r in ambiguous_rows.iterrows():
        st.markdown(f"**📌 {r['Konts']}** *(sadaļā: {r['Sadaļa']})*")
        st.write(f"👉 **Jautājums:** {r['Jautājums']}")
        
        tvc_hint = "Izvēlieties TVC TIKAI tad, ja šis izdevums pieaug tieši par 1 vienību ikreiz, kad pārdodat vēl 1 produktu klientam."
        oe_hint = "Izvēlieties OE, ja šo summu maksājat par perioda darbības vai jaudas uzturēšanu neatkarīgi no pārdotā skaita."
        st.info(f"💡 **TOC-FIN kritērijs:** {tvc_hint} | {oe_hint}")

        current_choice = "TVC" if "TVC" in str(r["Grupa"]) else "OE"
        choice = st.radio(
            f"Kā klasificēt '{r['Konts']}'?",
            options=["TVC (Patiesi mainīga izmaksa — 1:1 ar pārdošanu)", "OE (Uzturēšanas / kapacitātes izdevums)"],
            index=0 if current_choice == "TVC" else 1,
            key=f"radio_{r['id']}"
        )
        clarifications[r["id"]] = "TVC_DELIVERY" if "TVC" in choice else "OE_OTHER"
        st.markdown("---")

    if st.button("Saglabāt izvēles un parādīt kontu sadalījumu", type="primary"):
        for row_id, new_group in clarifications.items():
            cls_df.loc[cls_df["id"] == row_id, "Grupa"] = new_group
            cls_df.loc[cls_df["id"] == row_id, "Jautājums"] = ""
        S.cls_df = cls_df
        S.clarifications_done = True
        st.rerun()
    st.stop()

# --- PĀRSKATĀMS KONTU SADALĪJUMS PA GRUPĀM ---
st.subheader("📋 Kontu sadalījums pa TOC-FIN ekonomiskajām grupām")

tab_s, tab_tvc, tab_oe, tab_excl = st.tabs([
    "🟡 S — Pārdošana (Sales)", 
    "🟠 TVC — Patiesi mainīgās", 
    "🟤 OE — Uzturēšanas izdevumi", 
    "⚪ Izslēgtie (Grāmatvedības korekcijas / UIN / Amortizācija)"
])

s_keys = ["S"]
tvc_keys = ["TVC_MATERIALS", "TVC_DELIVERY"]
oe_keys = [k for k in GROUPS.keys() if k.startswith("OE_")]
excl_keys = ["EXCLUDED"]

with tab_s:
    st.dataframe(cls_df[cls_df["Grupa"].isin(s_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)
with tab_tvc:
    st.dataframe(cls_df[cls_df["Grupa"].isin(tvc_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)
with tab_oe:
    st.dataframe(cls_df[cls_df["Grupa"].isin(oe_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)
with tab_excl:
    st.caption("Šie posteņi tiek izslēgti no operatīvās TOC caurlaidības, lai novērstu virtuālo izmaksu un krājumu pārvērtēšanas radītos kropļojumus:")
    st.dataframe(cls_df[cls_df["Grupa"].isin(excl_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)

with st.expander("✏️ Atvērt pilno kontu tabulu (ja vēlaties manuāli pārcelt kādu posteni)"):
    edited = st.data_editor(
        cls_df, hide_index=True, use_container_width=True, key="cls_editor",
        disabled=["id", "Sadaļa", "Konts", "Drošība", "Jautājums"],
        column_config={"Grupa": st.column_config.SelectboxColumn(options=list(GROUPS), required=True),
                       "Ieņēmums": st.column_config.CheckboxColumn(help="Ieņēmuma konts")}
    )

# ------------------------------------------------------------------ 3. GALA APRĒĶINS
st.header("2. TOC-FIN Diagnostika un Lēmumu Atbalsts")
if st.button("🚀 Apstiprināt klasifikāciju un veikt TOC-FIN aprēķinu", type="primary"):
    mapping = {r["id"]: {"group": r["Grupa"], "is_income": bool(r["Ieņēmums"])} for _, r in edited.iterrows()}
    income_ids = {i for i, m in mapping.items() if m["is_income"] or m["group"] == "S"}
    rec2 = reconcile(an, stc, income_ids)
    if rec2["status"] == "fail":
        st.error(rec2["message"])
        st.dataframe(rec2["table"], hide_index=True)
    else:
        pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
        reported = an.ex.lines[pos].vals if pos is not None else None
        S.result = compute_toc(an.leaves, mapping, an.ex.period_cols, reported)
        if profile:
            profile.save(an.leaves, mapping)
        S.pop("commentary", None)
        S.chat = []

if "result" in S:
    res = S.result

    # Galvenās cilnes: Analytics (Vēsturiskais T*) vs Decision Support (Nākotnes Delta)
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
        disp = pd.DataFrame(index=res.main.index)
        for p in res.amounts.columns:
            disp[p] = res.main[p].map(fmt_num)
            disp[f"% {p}"] = res.main[f"%|{p}"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")

        def style_row(row):
            fill, bold, color = FILLS.get(res.row_kinds.get(row.name, "OE_CHILD"), FILLS["OE_CHILD"])
            return [f"background-color: #{fill}; color: #{color}; font-weight: {'bold' if bold else 'normal'};"] * len(row)

        st.dataframe(disp.style.apply(style_row, axis=1), use_container_width=True)

        st.subheader("📈 Produktivitātes un atdeves rādītāji")
        kp = res.kpi.copy()
        kp_disp = kp.copy().astype(object)
        for idx in kp.index:
            kp_disp.loc[idx] = kp.loc[idx].map(
                lambda x, i=idx: "" if pd.isna(x) else (f"{x:.1%}" if i.startswith("T%") else f"{x:.2f}"))
        st.dataframe(kp_disp, use_container_width=True)

        # --- VADĪBAS TRAUKSMES PANELIS (DECISION GUARDRAILS) ---
        st.markdown("---")
        st.subheader("🚦 Vadības Trauksmes Panelis (Decision Guardrails)")
        
        month_cols = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
        alerts_found = False

        if month_cols:
            # 1. PR < 1.0 pārbaude
            pr_series = res.kpi.loc["PR = T/OE", month_cols]
            bad_pr_months = pr_series[pr_series < 1.0].index.tolist()
            if bad_pr_months:
                alerts_found = True
                st.error(f"🔴 **Kritiska trauksme (PR < 1.0):** Periodos **{', '.join(bad_pr_months)}** uzturēšanas izmaksas ($OE$) pārsniedza ģenerēto caurlaidību ($T$). Uzņēmums šajos mēnešos dedzināja naudas rezerves.")

            # 2. TVC% anomālijas pārbaude (iesaldēti krājumi vai maržas kritums)
            t_pct_series = res.kpi.loc["T% = T/S", month_cols]
            mean_t_pct = t_pct_series.mean()
            low_t_months = t_pct_series[t_pct_series < (mean_t_pct - 0.07)].index.tolist()
            if low_t_months:
                alerts_found = True
                st.warning(f"🟡 **Caurlaidības / Krājumu brīdinājums:** Periodos **{', '.join(low_t_months)}** caurlaidības marža ($T\\%$) nokritās vairāk nekā par 7 procentpunktiem zem vidējā ({mean_t_pct:.1%}). Pārbaudiet, vai šajā mēnesī netika iepirkta liela materiālu partija noliktavai (iesaldējot naudu $I$) vai piešķirtas pārmērīgas cenu atlaides.")

            # 3. OE lēciena pārbaude
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

        # Bāzes rādītāji (no AVG YTD vai pēdējā mēneša)
        ref_col = next((c for c in res.amounts.columns if "AVG" in c), res.amounts.columns[-1])
        base_s = float(res.amounts.loc["S", ref_col])
        base_tvc = float(res.amounts.loc["TVC", ref_col])
        base_t = float(res.amounts.loc["T", ref_col])
        base_oe = float(res.amounts.loc["OE", ref_col])
        base_np = float(res.amounts.loc["NP", ref_col])
        base_t_pct = (base_t / base_s) if base_s else 0.5

        st.info(f"📌 **Salīdzinājuma bāze ({ref_col}):** Apgrozījums $S = {fmt_num(base_s)}$ € | Caurlaidība $T = {fmt_num(base_t)}$ € ({base_t_pct:.1%}) | Darbības izmaksas $OE = {fmt_num(base_oe)}$ € | Neto peļņa $NP = {fmt_num(base_np)}$ €")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("#### 1. Ietekme uz Caurlaidību ($\Delta T$)")
            delta_s = st.number_input("Papildu pārdošanas ieņēmumi (ΔS, EUR mēnesī)", value=20000, step=1000)
            default_dtvc = int(round(delta_s * (1.0 - base_t_pct)))
            delta_tvc = st.number_input("Papildu tiešās materiālu/piegādes izmaksas (ΔTVC, EUR)", value=default_dtvc, step=500,
                                        help="Tikai izejvielas un tiešās izmaksas šim konkrētajam pasūtījumam.")
            delta_t = delta_s - delta_tvc
            st.metric("Papildu Caurlaidība (ΔT = ΔS - ΔTVC)", f"{fmt_num(delta_t)} €")

        with col_s2:
            st.markdown("#### 2. Ietekme uz Kapacitāti ($\Delta OE$ un $\Delta I$)")
            delta_oe = st.number_input("Papildu uzturēšanas izmaksas (ΔOE, EUR mēnesī)", value=2000, step=500,
                                       help="Tikai reāli klāt nākošie tēriņi (piem., papildu darbinieka alga, virsstundas, kampaņa). Ja izmanto brīvo jaudu, ΔOE = 0.")
            delta_i = st.number_input("Nepieciešamā papildu investīcija / iesaldētā nauda (ΔI, EUR)", value=0, step=1000,
                                      help="Jaunas iekārtas iegāde vai papildu iesaldētie apgrozāmie līdzekļi krājumos.")
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
                st.metric("Investīcijas Gada Atdeve (ROI)", f"{annual_roi:.1%}",
                          delta=f"Atmaksāšanās: {payback_months:.1f} mēn." if delta_np > 0 else "Neatmaksājas")

        st.markdown("---")
        # Lēmuma trauksmes loģika
        if delta_np > 0 and "Izspiež" not in ccr_impact:
            st.success(f"✅ **TOC-FIN Lēmuma slēdziens:** Lēmums ir finansiāli pamatots ($\Delta T - \Delta OE = +{fmt_num(delta_np)}$ €/mēn.). Pat ja tradicionālā grāmatvedības pilnā pašizmaksa šim darījumam izskatītos 'nerentabla', brīvās jaudas apstākļos tas tieši palielina uzņēmuma tīro peļņu un naudas plūsmu.")
        elif delta_np > 0 and "Izspiež" in ccr_impact:
            st.warning(f"⚠️ **Trauksme par CCR noslodzi:** Lai gan $\Delta NP$ ir pozitīvs ($+{fmt_num(delta_np)}$ €), šis lēmums izspiež citus pasūtījumus ierobežojumā! Pārliecinieties, ka jaunā pasūtījuma caurlaidība uz 1 CCR minūti ($T_u / CCR_{{min}}$) ir augstāka nekā izspiestajiem produktiem.")
        else:
            st.error(f"🛑 **Stop-krāns (Negatīvs $\Delta NP$):** Šis lēmums samazina uzņēmuma peļņu par **{fmt_num(abs(delta_np))} €** mēnesī, jo papildu uzturēšanas izmaksas ($\Delta OE$) pārsniedz iegūto caurlaidību ($\Delta T$). Nepieciešams stratēģisks pamatojums vai izmaksu pārskatīšana.")

if st.button("🔄 Sākt no jauna"):
    st.session_state.clear()
    st.rerun()

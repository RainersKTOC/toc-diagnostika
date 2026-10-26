"""
app.py — TOC finanšu diagnostika (v5)
Lietotājam draudzīga plūsma:
- Fona datu ielasīšana un automātiska pārbaude (bez tehniskiem logiem)
- Interaktīvi TOC jautājumi par neskaidrajiem kontiem (TVC vs OE) ar padomiem
- Pārskatāms kontu sadalījums pa grupām (S, TVC, OE, Izslēgtie)
- Gala atskaite, Excel lejuplāde ar krāsām un vadības secinājumi
"""
import hashlib
import io
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

from profile_store import ProfileStore
from toc_engine import (GROUPS, FILLS, build_excel, classify_with_claude, compute_toc)
from toc_ingest import (PERIOD_KINDS, Structure, analyze, extract_lines, load_grids, pick_sheet,
                        reconcile, request_structure, verification_report)

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")

# --- IEVADS UN APRAKSTS ---
st.title("📊 Finanšu diagnostika pēc TOC principiem")
st.markdown("""
Šī sistēma pārveido standarta Peļņas un Zaudējumu pārskatu (PZA) **Ierobežojumu teorijas (TOC)** 
caurplūdes grāmatvedības formātā. Tā precīzi nodala patiesi mainīgās izmaksas ($TVC$) no uzturēšanas 
izmaksām ($OE$), aprēķinot patieso caurplūdi ($T$), biznesa sistēmas produktivitāti ($PR = T / OE$) un peļņu.
""")

# --- INFORMĀCIJAS LOGS PAR TOC ---
with st.expander("ℹ️ Uzzināt vairāk: Kas ir TOC caurplūdes grāmatvedība un kāpēc tā vajadzīga?"):
    st.markdown("""
    Tradicionālā izmaksu grāmatvedība bieži rada ilūziju par peļņu, sadalot netiešās un darba izmaksas pašizmaksā un krājumos. 
    **TOC (Throughput Accounting)** pieeja novērš kropļojumus, fokusējoties uz naudas plūsmas realitāti:

    * **S (Sales / Neto apgrozījums)** — Reāli ienākošie naudas līdzekļi no gala pircējiem par pārdoto produkciju.
    * **TVC (Truly Variable Costs / Patiesi mainīgās izmaksas)** — Tikai izdevumi, kas rodas **1:1 ar katru pārdoto vienību** (izejvielas, materiāli un preču tiešais transports klientam). Ja netiek pārdots neviens produkts, šīm izmaksām jābūt tieši 0.
    * **T (Throughput / Caurplūde)** — Uzņēmuma radītā jaunā naudas masa: $$T = S - TVC$$. Tā ir nauda, kas sedz darbības izdevumus un veido peļņu.
    * **OE (Operating Expenses / Darbības izdevumi)** — Visa nauda, ko uzņēmums tērē, lai uzturētu jaudu (visas algas, telpu noma, IT, mārketings utt.). Darbaspēks TOC izpratnē ir OE, jo algas nemainās par katru pārdoto produktu.
    * **NP (Net Profit / Neto peļņa)** — Gala peļņa: $$NP = T - OE$$.
    * **PR (Productivity Ratio / Produktivitātes rādītājs)** — $$PR = T / OE$$. 
      * Ja $PR > 1.0$ — uzņēmums strādā ar peļņu.
      * Ja $PR < 1.0$ — bīstama situācija, caurplūde nesedz uzņēmuma uzturēšanas izmaksas.
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


# Zināšanu bāzes ielāde (ja pievienota)
knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

# ------------------------------------------------------------------ 1. AUGŠUPIELĀDE
uploaded = st.file_uploader("Augšupielādējiet PZA failu (CSV vai Excel formātā)", type=["csv", "xlsx", "xlsm", "xls"])

if uploaded is None:
    st.info("Lai veiktu diagnostiku, lūdzu, augšupielādējiet PZA failu.")
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
    with st.spinner("Apstrādā faila struktūru un veic datu integritātes pārbaudi..."):
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
st.header("1. Kontu izvērtēšana un klasifikācija")

profile = ProfileStore("default")
if "cls_df" not in S:
    if st.button("🔍 Sākt kontu TOC analīzi", type="primary"):
        with st.spinner("Klasificē kontus atbilstoši TOC loģikai..."):
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
    st.subheader("❓ Nepieciešams Jūsu precizējums par dažām pozīcijām")
    st.caption("Lai aprēķins būtu 100% precīzs, izvēlieties, vai šie izdevumi mainās 1:1 ar katru pārdoto vienību (TVC) vai ir perioda fiksētās izmaksas (OE):")

    clarifications = {}
    for idx, r in ambiguous_rows.iterrows():
        st.markdown(f"**📌 {r['Konts']}** *(sadaļā: {r['Sadaļa']})*")
        st.write(f"👉 **Jautājums:** {r['Jautājums']}")
        
        tvc_hint = "Izvēlieties TVC, ja šīs izmaksas rodas tikai tad, kad produkts tiek saražots/pārdots, un pieaug tieši proporcionāli apjomam."
        oe_hint = "Izvēlieties OE, ja šī ir mēneša fiksēta maksa, bāzes uzturēšana vai administratīvs/vadības pakalpojums, kas jāmaksā neatkarīgi no apjoma."
        st.info(f"💡 **TOC padoms:** {tvc_hint} | {oe_hint}")

        current_choice = "TVC" if "TVC" in str(r["Grupa"]) else "OE"
        choice = st.radio(
            f"Kā klasificēt '{r['Konts']}'?",
            options=["TVC (Patiesi mainīga izmaksa)", "OE (Uzturēšanas / darbības izdevums)"],
            index=0 if current_choice == "TVC" else 1,
            key=f"radio_{r['id']}"
        )
        clarifications[r["id"]] = "TVC_DELIVERY" if "TVC" in choice else "OE_OTHER"
        st.markdown("---")

    if st.button("Saglabāt precizējumus un parādīt kontu sadalījumu", type="primary"):
        for row_id, new_group in clarifications.items():
            cls_df.loc[cls_df["id"] == row_id, "Grupa"] = new_group
            cls_df.loc[cls_df["id"] == row_id, "Jautājums"] = ""
        S.cls_df = cls_df
        S.clarifications_done = True
        st.rerun()
    st.stop()

# --- PĀRSKATĀMS KONTU SADALĪJUMS PA GRUPĀM ---
st.subheader("📋 Apstiprinātais kontu sadalījums pa TOC grupām")
st.caption("Pārbaudiet kontu piederību. Ja nepieciešams, jebkuram kontam varat nomainīt grupu:")

tab_s, tab_tvc, tab_oe, tab_excl = st.tabs(["🟡 S — Pārdošana (Sales)", "🟠 TVC — Patiesi mainīgās", "🟤 OE — Darbības izdevumi", "⚪ Izslēgtie posteņi"])

s_keys = ["S"]
tvc_keys = ["TVC_MATERIALS", "TVC_DELIVERY"]
oe_keys = [k for k in GROUPS.keys() if k.startswith("OE_")]
excl_keys = ["EXCLUDED"]

with tab_s:
    st.markdown("**Neto apgrozījums un pamatdarbības ieņēmumi:**")
    st.dataframe(cls_df[cls_df["Grupa"].isin(s_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)

with tab_tvc:
    st.markdown("**Patiesi mainīgās izmaksas (izejvielas, preču tiešā piegāde):**")
    st.dataframe(cls_df[cls_df["Grupa"].isin(tvc_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)

with tab_oe:
    st.markdown("**Darbības un uzturēšanas izdevumi (algas, telpas, mārketings, IT, birojs):**")
    st.dataframe(cls_df[cls_df["Grupa"].isin(oe_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)

with tab_excl:
    st.markdown("**Izslēgtie posteņi (amortizācija, procenti, UIN):**")
    st.dataframe(cls_df[cls_df["Grupa"].isin(excl_keys)][["Konts", "Sadaļa", "Grupa"]], hide_index=True, use_container_width=True)

with st.expander("✏️ Pilns kontu saraksts (noklikšķiniet, ja vēlaties veikt manuālas korekcijas)"):
    edited = st.data_editor(
        cls_df, hide_index=True, use_container_width=True, key="cls_editor",
        disabled=["id", "Sadaļa", "Konts", "Drošība", "Jautājums"],
        column_config={"Grupa": st.column_config.SelectboxColumn(options=list(GROUPS), required=True),
                       "Ieņēmums": st.column_config.CheckboxColumn(help="Ieņēmuma konts")}
    )

# ------------------------------------------------------------------ 3. GALA APRĒĶINS
st.header("2. TOC Diagnostikas aprēķins")
if st.button("🚀 Apstiprināt klasifikāciju un ģenerēt atskaiti", type="primary"):
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
    st.success("TOC diagnostikas atskaite ir gatava!")

    xl_f, xl_v = io.BytesIO(), io.BytesIO()
    build_excel(res, xl_f, formulas=True)
    build_excel(res, xl_v, formulas=False)
    d1, d2 = st.columns(2)
    d1.download_button("📥 Lejupielādēt noformētu Excel ar formulām (.xlsx)", xl_f.getvalue(), "TOC_Finansu_Diagnostika.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("📥 Lejupielādēt Excel ar vērtībām (.xlsx)", xl_v.getvalue(), "TOC_Finansu_Diagnostika_vertibas.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Finanšu pārskata tabula")
    disp = pd.DataFrame(index=res.main.index)
    for p in res.amounts.columns:
        disp[p] = res.main[p].map(fmt_num)
        disp[f"% {p}"] = res.main[f"%|{p}"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")

    def style_row(row):
        fill, bold, color = FILLS.get(res.row_kinds.get(row.name, "OE_CHILD"), FILLS["OE_CHILD"])
        return [f"background-color: #{fill}; color: #{color}; font-weight: {'bold' if bold else 'normal'};"] * len(row)

    st.dataframe(disp.style.apply(style_row, axis=1), use_container_width=True)

    st.subheader("📈 Produktivitātes rādītāji")
    kp = res.kpi.copy()
    kp_disp = kp.copy().astype(object)
    for idx in kp.index:
        kp_disp.loc[idx] = kp.loc[idx].map(
            lambda x, i=idx: "" if pd.isna(x) else (f"{x:.1%}" if i.startswith("T%") else f"{x:.2f}"))
    st.dataframe(kp_disp, use_container_width=True)
    st.caption("YTD = rādītājs no YTD summām; AVG YTD = mēnešu rādītāju vidējais aritmētiskais.")

    # -------------------------------------------------------------- 4. VADĪBAS ANALĪZE
    st.header("3. Vadības kopsavilkums un ieteikumi")
    tables_md = (res.amounts.round(0).to_markdown() + "\n\n" + res.kpi.round(3).to_markdown())
    
    knowledge_context = f"Papildu TOC vadlīnijas no grāmatām:\n{knowledge_file}\n\n" if knowledge_file else ""
    COMMENT_SYSTEM = (
        "Tu esi pieredzējis TOC (Theory of Constraints) finanšu analītiķis un vadības konsultants. "
        "Tev ir doti jau precīzi aprēķināti un pārbaudīti skaitļi. "
        "NEAPRĒĶINI neko no jauna un neizdomā skaitļus — citē tikai tos, kas ir tabulās. "
        f"{knowledge_context}"
        "PR < 1 nozīmē, ka T nesedz OE. Pievērs uzmanību T% stabilitātei, algu atdevei un fiksēto izmaksu lēcieniem. "
        "Raksti latviešu valodā, profesionāli, kodolīgi un vadībai uzreiz pielietojami.")

    if "commentary" not in S:
        if st.button("📝 Sagatavot vadības kopsavilkumu"):
            with st.spinner("Gatavo vadības kopsavilkumu un secinājumus..."):
                r = get_client().messages.create(
                    model=MODEL, max_tokens=2000, system=COMMENT_SYSTEM,
                    messages=[{"role": "user", "content":
                               f"Aprēķinātās tabulas (summas EUR, izmaksas pozitīvas):\n\n{tables_md}\n\n"
                               "Uzraksti strukturētu analīzi:\n"
                               "### Galvenās tendences un dinamika\n"
                               "### Produktivitātes vērtējums (PR un algu atdeve)\n"
                               "### Vadības rīcības plāns un prioritātes (3 konkrēti ieteikumi)"}])
                S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                st.rerun()
    else:
        st.markdown(S.commentary)

        st.subheader("💬 Jautājums vai diskusija par šo analīzi")
        S.setdefault("chat", [])
        for m in S.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        q = st.chat_input("Ieraksti jautājumu par pārskata datiem...")
        if q:
            S.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Analizē..."):
                    r = get_client().messages.create(
                        model=MODEL, max_tokens=1500,
                        system=COMMENT_SYSTEM + f"\n\nAprēķinātās tabulas:\n{tables_md}\n\nIepriekšējais kopsavilkums:\n{S.commentary}",
                        messages=S.chat)
                    reply = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    st.markdown(reply)
            S.chat.append({"role": "assistant", "content": reply})

if st.button("🔄 Sākt no jauna"):
    st.session_state.clear()
    st.rerun()

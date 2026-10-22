"""
app.py — TOC finanšu diagnostika (v3)
"""
import hashlib
import io

import anthropic
import pandas as pd
import streamlit as st

from profile_store import ProfileStore
from toc_engine import (GROUPS, FILLS, build_excel, classify_with_claude, compute_toc)
from toc_ingest import (PERIOD_KINDS, Structure, analyze, extract_lines, load_grids, pick_sheet,
                        reconcile, request_structure, verification_report)

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")
st.title("📊 Finanšu diagnostika pēc TOC principiem")
st.caption("Claude saprot faila struktūru un klasificē kontus; visus skaitļus aprēķina un pārbauda kods.")

API_KEY = str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip().strip('"').strip("'")
if not API_KEY:
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit noslēpumos.")
    st.stop()


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=API_KEY)


def fmt_num(x) -> str:
    return "" if pd.isna(x) else f"{x:,.0f}".replace(",", " ")


# ------------------------------------------------------------------ AUGŠUPIELĀDE
col_a, col_b = st.columns([3, 2])
with col_a:
    uploaded = st.file_uploader("PZA fails (CSV vai Excel, jebkurš formāts)", type=["csv", "xlsx", "xlsm", "xls"])
with col_b:
    company = st.text_input("Uzņēmums (profila nosaukums)", value="",
                            help="Apstiprinātā kontu klasifikācija tiek atcerēta šim uzņēmumam.")

if uploaded is None:
    st.info("Augšupielādējiet PZA failu, lai sāktu.")
    st.stop()

data = uploaded.getvalue()
fid = hashlib.md5(data).hexdigest()
if st.session_state.get("fid") != fid:
    st.session_state.clear()
    st.session_state.fid = fid
    st.session_state.grids = load_grids(uploaded.name, data)
S = st.session_state
grids: dict[str, pd.DataFrame] = S.grids

# ------------------------------------------------------------------ 1. STRUKTŪRA
st.header("1. Faila struktūra")
if "structure" not in S:
    if st.button("🔍 Nolasīt faila struktūru", type="primary"):
        with st.spinner("Claude nosaka, kur atrodas dati (tiek sūtīts tikai faila priekšskatījums)..."):
            try:
                sheet = pick_sheet(get_client(), MODEL, grids)
                S.structure = request_structure(get_client(), MODEL, sheet, grids[sheet])
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Neizdevās automātiski noteikt struktūru: {e}. Varat ievadīt to manuāli zemāk.")
                S.structure = Structure(sheet=next(iter(grids)), period_cols=[], first_row=0,
                                        last_row=len(next(iter(grids.values()))) - 1)
                S.manual = True
    if "structure" not in S:
        st.stop()

stc: Structure = S.structure
with st.expander("✏️ Atrastā struktūra (var labot)", expanded=bool(S.get("manual"))):
    with st.form("structure_form"):
        c1, c2, c3, c4 = st.columns(4)
        sheet = c1.selectbox("Lapa", list(grids), index=list(grids).index(stc.sheet) if stc.sheet in grids else 0)
        first_row = c2.number_input("Pirmā rinda (0-bāze)", 0, 100000, int(stc.first_row))
        last_row = c3.number_input("Pēdējā rinda", 0, 100000, int(stc.last_row))
        result_row = c4.number_input("Gala rezultāta rinda (-1 = nav)", -1, 100000,
                                     -1 if stc.result_row is None else int(stc.result_row))
        c5, c6 = st.columns(2)
        conv_opts = ["costs_negative", "costs_positive", "mixed"]
        sign = c5.selectbox("Izdevumu zīme detaļrindās", conv_opts,
                            index=conv_opts.index(stc.sign_convention) if stc.sign_convention in conv_opts else 2)
        pos_opts = ["auto", "after", "before"]
        tpos = c6.selectbox("Kopsummas atrodas", pos_opts,
                            index=pos_opts.index(stc.totals_position) if stc.totals_position in pos_opts else 0,
                            help="after = pēc komponentēm, before = pirms komponentēm")
        st.caption("Perioda kolonnas (kolonnas numurs 0-bāzē):")
        pdf = st.data_editor(pd.DataFrame(stc.period_cols or [{"col": 1, "label": "", "kind": "month"}]),
                             num_rows="dynamic", hide_index=True,
                             column_config={"kind": st.column_config.SelectboxColumn(options=list(PERIOD_KINDS))})
        if st.form_submit_button("Piemērot un pārbaudīt"):
            S.structure = Structure(
                sheet=sheet, period_cols=pdf.dropna(subset=["col"]).astype({"col": int}).to_dict("records"),
                first_row=int(first_row), last_row=int(last_row),
                result_row=None if result_row < 0 else int(result_row), sign_convention=sign,
                totals_position=tpos, pl_layout=stc.pl_layout, unit=stc.unit, company=stc.company, notes=stc.notes)
            for k in ("analysis", "cls_df", "result", "commentary", "chat"):
                S.pop(k, None)
            S.pop("manual", None)
            st.rerun()
    if stc.notes:
        st.caption(f"Claude piezīme: {stc.notes}")

if stc.pl_layout == "by_function":
    st.error("⚠️ Šis PZA ir sadalīts pēc FUNKCIJĀM (pārdošanas/administrācijas izmaksu kopsummas bez sadalījuma pa "
             "izmaksu veidiem). TOC nevar atdalīt TVC un algas. Nepieciešams kontu apgrozījuma pārskats "
             "vai PZA ar posteņu atšifrējumiem.")

try:
    ex = extract_lines(grids[stc.sheet], stc)
    S.analysis = analyze(ex, stc)
except Exception as e:  # noqa: BLE001
    st.error(f"Struktūra nav derīga: {e}")
    st.stop()
an = S.analysis

# ------------------------------------------------------------------ 2. PĀRBAUDE
st.header("2. Pārbaude")
report = pd.DataFrame(verification_report(grids[stc.sheet], stc, an))
st.dataframe(report, hide_index=True, use_container_width=True)
for n in an.notes:
    st.caption(n)
rec = reconcile(an, stc)
if rec["table"] is not None:
    with st.expander("Σ(lapas) pret faila gala rezultātu pa periodiem"):
        st.dataframe(rec["table"].style.format({"Σ lapas": fmt_num, "Faila rezultāts": fmt_num, "Starpība": "{:,.2f}"}),
                     hide_index=True)
blocking = (report["Statuss"] == "KĻŪDA").any()
if blocking:
    st.error("Datu nolasīšana neiztur pārbaudi. Pielabojiet struktūru augstāk (rindu diapazonu, kolonnas, zīmes) — "
             "tikai pēc tam turpiniet.")
    st.stop()

with st.expander(f"Atrastās lapas ({len(an.leaves)}) — kontu saraksts"):
    st.dataframe(pd.DataFrame({
        "Excel rinda": [l.row + 1 for l in an.leaves], "Sadaļa": [l.path for l in an.leaves],
        "Konts": [l.label for l in an.leaves],
        **{p["label"]: [l.vals[j] for l in an.leaves] for j, p in enumerate(an.ex.period_cols)}}).style.format(
            {p["label"]: fmt_num for p in an.ex.period_cols}), hide_index=True)

# ------------------------------------------------------------------ 3. KLASIFIKĀCIJA
st.header("3. Kontu klasifikācija pēc TOC")
profile = ProfileStore(company) if company.strip() else None
if "cls_df" not in S:
    if st.button("🏷️ Klasificēt kontus", type="primary"):
        with st.spinner("Claude klasificē kontus (skaitļi netiek sūtīti)..."):
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
                S.known_n = len(known)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Kļūda klasifikācijā: {e}")
    if "cls_df" not in S:
        st.stop()

cls: pd.DataFrame = S.cls_df
if S.get("known_n"):
    st.caption(f"{S.known_n} konti automātiski ņemti no uzņēmuma profila.")
low = cls[cls["Jautājums"].astype(str).str.strip() != ""]
if len(low):
    st.warning("Claude jautājumi par neskaidrajiem kontiem — atbildiet, labojot grupu tabulā:")
    for _, r in low.iterrows():
        st.markdown(f"- **{r['Konts']}** ({r['Sadaļa']}): {r['Jautājums']}")
edited = st.data_editor(
    cls, hide_index=True, use_container_width=True, key="cls_editor",
    disabled=["id", "Sadaļa", "Konts", "Drošība", "Jautājums"],
    column_config={"Grupa": st.column_config.SelectboxColumn(options=list(GROUPS), required=True),
                   "Ieņēmums": st.column_config.CheckboxColumn(help="Ieņēmuma konts (S, granti, citi ieņēmumi)")})

# ------------------------------------------------------------------ 4. APRĒĶINS
st.header("4. Aprēķins")
if st.button("🚀 Apstiprināt klasifikāciju un aprēķināt", type="primary"):
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
    bad = [c for c in res.checks if c["Statuss"] != "OK"]
    (st.error if bad else st.success)("Aprēķina pārbaudes: " + ("; ".join(f"{c['Pārbaude']} — {c['Detaļas']}" for c in bad)
                                                                 if bad else "visas kārtībā."))

    xl_f, xl_v = io.BytesIO(), io.BytesIO()
    build_excel(res, xl_f, formulas=True)
    build_excel(res, xl_v, formulas=False)
    d1, d2 = st.columns(2)
    d1.download_button("📥 Excel ar formulām (.xlsx)", xl_f.getvalue(), "TOC_Finansu_Diagnostika.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("📥 Excel tikai ar vērtībām (priekšskatījumiem)", xl_v.getvalue(), "TOC_Finansu_Diagnostika_vertibas.xlsx",
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

    # -------------------------------------------------------------- 5. KOMENTĀRS
    st.header("5. Vadības analīze")
    tables_md = (res.amounts.round(0).to_markdown() + "\n\n" + res.kpi.round(3).to_markdown())
    COMMENT_SYSTEM = (
        "Tu esi TOC finanšu analītiķis. Tev ir doti jau aprēķināti un pārbaudīti skaitļi. "
        "NEAPRĒĶINI neko no jauna un neizdomā skaitļus — citē tikai tos, kas ir tabulās. Ja vajadzīgs skaitlis, "
        "kura tur nav, pasaki to. PR < 1 nozīmē, ka T nesedz OE. Raksti latviski, profesionāli un īsi.")
    if "commentary" not in S:
        if st.button("📝 Sagatavot vadības kopsavilkumu"):
            with st.spinner("Raksta kopsavilkumu..."):
                r = get_client().messages.create(
                    model=MODEL, max_tokens=2000, system=COMMENT_SYSTEM,
                    messages=[{"role": "user", "content":
                               f"Aprēķinātās tabulas (summas EUR, izmaksas pozitīvas):\n\n{tables_md}\n\n"
                               "Uzraksti:\n### Galvenās tendences un dinamika\n### Produktivitātes vērtējums (PR un algu atdeve)\n"
                               "### Vadības rīcības plāns (3 konkrēti ieteikumi)"}])
                S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                st.rerun()
    else:
        st.markdown(S.commentary)

        st.subheader("💬 Jautājums par šo analīzi")
        S.setdefault("chat", [])
        for m in S.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        q = st.chat_input("Ieraksti jautājumu par datiem...")
        if q:
            S.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Domā..."):
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

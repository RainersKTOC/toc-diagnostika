import streamlit as st
import anthropic
import pandas as pd
import io
import re

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")

# Konfigurē API atslēgu (to ievieto Streamlit Secrets vai vides mainīgajā)
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

st.title("📊 Finanšu diagnostika pēc TOC principiem")
st.caption("Augšupielādējiet Peļņas un Zaudējumu pārskatu (CSV vai Excel), lai saņemtu noformētu Excel atskaiti un vadības kopsavilkumu.")

if not ANTHROPIC_API_KEY:
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit noslēpumos (.streamlit/secrets.toml vai Streamlit Cloud iestatījumos).")
    st.stop()

uploaded_file = st.file_uploader("Izvēlieties PZA failu", type=["xlsx", "xls", "csv"])

TOC_SYSTEM_PROMPT = """
Tu esi TOC (Throughput Accounting) finanšu diagnostikas dzinējs.
Tavs uzdevums ir nolasīt pievienoto PZA datu saturu un veikt precīzu aprēķinu:
1. S (Sales): Reālie pamatdarbības ieņēmumi (100%).
2. TVC: Tikai tiešie materiāli, izejvielas un tiešie ārpakalpojumi (1:1 mainīgie).
3. T = S - TVC.
4. OE: Dinamiskās grupas (7-12 rindas, būtiskuma princips). Algas jāsadala: Administrācija un Strādnieki.
5. NP = T - OE.
6. Izslēgt amortizāciju, krājumu izmaiņas un nodokļus.
7. Rādītāji: PR = T/OE, T% = T/S, Sales/Algas kopā, Sales/Administrācija, Sales/Strādnieki.

ATBILDES FORMĀTS:
1. Vispirms ievieto pilnu aprēķinu TSV tabulā iekš koda bloka:
```tsv
[Rādītājs / Kontu grupa]	01.XXXX	%	02.XXXX	%	...	YTD	%	AVG YTD	%
...
```
2. Tūlīt zem tabulas sniedz profesionālu, ļoti konkrētu vadības analīzi (3 daļās):
   - KAS IR LABS UN STABILS (T%, stabilākie periodi)
   - BĪSTAMIE SIGNĀLI (mēneši ar PR < 1.0, OE lēcieni, algu atdeves kritums)
   - VADĪBAS FOKUSS UN IETEIKUMI (kam nekavējoties pievērst uzmanību)
"""

if uploaded_file is not None:
    if st.button("Veikt diagnostiku", type="primary"):
        with st.spinner("Pārbauda modeļus un atslēgu..."):
            try:
                clean_key = ANTHROPIC_API_KEY.strip().strip('"').strip("'")
                st.write(f"Atslēgas sākums: `{clean_key[:14]}...` (Garums: {len(clean_key)})")
                
                client = anthropic.Anthropic(api_key=clean_key)
                models_resp = client.models.list()
                
                available = [m.id for m in models_resp.data]
                st.success("Savienojums veiksmīgs! Kontam pieejamie modeļi:")
                st.json(available)
            except Exception as e:
                st.error(f"Detalizēta kļūda: {e}")

import io
import re
import anthropic
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")

# Konfigurē API atslēgu no Streamlit Secrets
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
[Rādītājs / Kontu grupa]\t01.XXXX\t%\t02.XXXX\t%\t...\tYTD\t%\tAVG YTD\t%
...
Tūlīt zem tabulas sniedz profesionālu, ļoti konkrētu vadības analīzi (3 daļās):
KAS IR LABS UN STABILS (T%, stabilākie periodi)
BĪSTAMIE SIGNĀLI (mēneši ar PR < 1.0, OE lēcieni, algu atdeves kritums)
VADĪBAS FOKUSS UN IETEIKUMI (kam nekavējoties pievērst uzmanību)
"""
if uploaded_file is not None:
if st.button("Veikt diagnostiku", type="primary"):
with st.spinner("Apstrādā datus un ģenerē TOC analīzi..."):
try:
# Nolasām augšupielādētā faila teksta saturu
if uploaded_file.name.endswith(".csv"):
df_raw = pd.read_csv(uploaded_file)
file_text = df_raw.to_csv(index=False)
else:
df_raw = pd.read_excel(uploaded_file)
file_text = df_raw.to_csv(index=False)
            clean_key = str(ANTHROPIC_API_KEY).strip().strip('"').strip("'")
            client = anthropic.Anthropic(api_key=clean_key)
            
            response = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=4000,
                system=TOC_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Lūk, PZA dati analīzei:\n\n{file_text}"}
                ]
            )
            
            output_text = response.content[0].text
            
            # Izgūstam TSV tabulu un analīzes tekstu
            tsv_match = re.search(r"```tsv\n(.*?)```", output_text, re.DOTALL)
            
            st.success("Diagnostika veiksmīgi pabeigta!")
            
            if tsv_match:
                tsv_data = tsv_match.group(1).strip()
                df_res = pd.read_csv(io.StringIO(tsv_data), sep="\t")
                
                # Sagatavojam Excel failu lejupielādei
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    df_res.to_excel(writer, index=False, sheet_name="TOC Diagnostika")
                
                st.download_button(
                    label="📥 Lejupielādēt noformēto Excel failu (.xlsx)",
                    data=excel_buffer.getvalue(),
                    file_name="TOC_Finansu_Diagnostika.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.subheader("Finanšu pārskata tabula")
                st.dataframe(df_res, use_container_width=True)
            
            st.subheader("Vadības analīze un TOC secinājumi")
            clean_text = re.sub(r"```tsv.*?```", "", output_text, flags=re.DOTALL).strip()
            st.markdown(clean_text)

        except Exception as e:
            st.error(f"Kļūda apstrādē: {str(e)}")

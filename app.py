import io
import re
import anthropic
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

st.title("📊 Finanšu diagnostika pēc TOC principiem")
st.caption("Augšupielādējiet Peļņas un Zaudējumu pārskatu (CSV vai Excel), lai saņemtu noformētu Excel atskaiti un vadības kopsavilkumu.")

if not ANTHROPIC_API_KEY:
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit noslēpumos.")
    st.stop()

uploaded_file = st.file_uploader("Izvēlieties PZA failu", type=["xlsx", "xls", "csv"])

TOC_SYSTEM_PROMPT = """
Tu esi TOC (Throughput Accounting) finanšu diagnostikas dzinējs.
Veic precīzu aprēķinu atbilstoši šiem striktajiem noteikumiem:

1. S (Sales): Neto apgrozījums / pamatdarbības ieņēmumi (100%).
2. TVC (Truly Variable Costs): IEKĻAUT TIKAI UN VIENĪGI:
   - Izejvielas un materiāli
   - Preču piegādes izdevumi
   (NEKĀDAS telpu nomas, platformu maksas vai citus izdevumus TVC nelikt! Viss pārējais ir OE).
3. T (Throughput) = S - TVC.
4. OE (Operating Expenses): Dinamiskās grupas (7-12 rindas). Algas obligāti sadalīt: Administrācija un Strādnieki.
5. NP (Net Profit) = T - OE.
6. Izslēgt amortizāciju, krājumu izmaiņas, UIN un finanšu procentu maksājumus.
7. Galvenie rādītāji: PR = T/OE, T% = T/S, Sales/Algas kopā, Sales/Administrācija, Sales/Strādnieki.

STINGRI AIZLIEGTS:
- Nedrukā nekādus ievadtekstus, starpposmu skaidrojumus, rēķināšanas soļus vai metodoloģijas aprakstus.
- Sāc atbildi TIEŠI ar ```tsv koda bloku.

ATBILDES STRUKTŪRA:
1. Precīza TSV tabula koda blokā:
```tsv
Rādītājs / Kontu grupa\t01.XXXX\t%\t02.XXXX\t%\t...\tYTD\t%\tAVG YTD\t%
...
Īss un kodolīgs vadības kopsavilkums (tikai 3 īsas sadaļas):
Kas ir labs un stabils
(1-2 teikumi par T% un labākajiem periodiem)
Bīstamie signāli
(1-2 teikumi par mēnešiem ar PR < 1.0, OE pieaugumu vai algu atdevi)
Vadības fokuss un uzlabojumi
(Konkrēti, īsi soļi vadībai)
"""
if uploaded_file is not None:
if st.button("Veikt diagnostiku", type="primary"):
with st.spinner("Apstrādā datus un ģenerē TOC analīzi..."):
try:
if uploaded_file.name.endswith(".csv"):
df_raw = pd.read_csv(uploaded_file)
file_text = df_raw.to_csv(index=False)
else:
df_raw = pd.read_excel(uploaded_file)
file_text = df_raw.to_csv(index=False)
            clean_key = str(ANTHROPIC_API_KEY).strip().strip('"').strip("'")
            client = anthropic.Anthropic(api_key=clean_key)
            
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                system=TOC_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Lūk, PZA dati analīzei:\n\n{file_text}"}
                ]
            )
            
            output_text = ""
            for block in response.content:
                if getattr(block, "type", "") == "text":
                    output_text += block.text
                elif hasattr(block, "text"):
                    output_text += block.text
            
            # Izgūstam TSV tabulu
            tsv_match = re.search(r"```tsv\n(.*?)```", output_text, re.DOTALL)
            
            st.success("Diagnostika veiksmīgi pabeigta!")
            
            if tsv_match:
                tsv_data = tsv_match.group(1).strip()
                df_res = pd.read_csv(io.StringIO(tsv_data), sep="\t")
                
                # Lejupielādes poga Excel failam
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
            
            st.subheader("Vadības analīze un secinājumi")
            clean_text = re.sub(r"```tsv.*?```", "", output_text, flags=re.DOTALL).strip()
            st.markdown(clean_text)

        except Exception as e:
            st.error(f"Kļūda apstrādē: {str(e)}")

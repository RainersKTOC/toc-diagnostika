import io
import re
import anthropic
import pandas as pd
import streamlit as st
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

if "df_res" not in st.session_state:
    st.session_state.df_res = None
if "output_text" not in st.session_state:
    st.session_state.output_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_text" not in st.session_state:
    st.session_state.file_text = ""

st.title("📊 Finanšu diagnostika pēc TOC principiem")
st.caption("Augšupielādējiet Peļņas un Zaudējumu pārskatu (CSV vai Excel), lai saņemtu noformētu Excel atskaiti un vadības kopsavilkumu.")

if not ANTHROPIC_API_KEY:
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit noslēpumos.")
    st.stop()

uploaded_file = st.file_uploader("Izvēlieties PZA failu", type=["xlsx", "xls", "csv"])

TOC_SYSTEM_PROMPT = """
Tu esi TOC (Throughput Accounting) finanšu diagnostikas dzinējs.
PZA dati satur gan kopsavilkuma rindas, gan atšifrējumus. NESKAITI DUBULTĀ! Ņem datus tikai no atšifrējumiem!

TABULAS STRUKTŪRA UN RINAS (STINGRĀ SECĪBĀ):
1. S (Sales)
2. TVC (Truly variable costs)
3. Izejvielas un materiāli
4. Preču piegādes izdevumi
5. T (Throughput)
6. OE (Operating expenses)
7. Algas kopā
8. Administrācijas algas (+ sociālās iemaksas)
9. Strādnieku algas (+ sociālās iemaksas)
10. Mārketings un reklāma
11. Pārdošanas platformu izdevumi un pārējās pārdošanas izmaksas (ietverot US TAX, mājas lapu)
12. Ražošanas telpu noma un uzturēšana (+ pārējās ražošanas izm.)
13. Biroja, IT un sakaru izmaksas
14. Citi saimnieciskie un personāla izdevumi
15. NP (Net Profit)
16. PR = T/OE
17. T% = T/S
18. Eiro Sales/Eiro algas kopā
19. Eiro Sales/Eiro administrācija
20. Eiro Sales/Eiro strādnieki

STINGRI AIZLIEGTS:
- Preču piegādes izdevumus likt pie OE! Tie ir 100% TVC sastāvdaļa!
- Neveido nekādas tukšas rindas, atdalošās rindas vai rindas ar 'None' vai '---'!
- Sāc TIEŠI ar ```tsv koda bloku bez ievada.

KOLONNU VIRSRINDA TSV BLOKĀ:
Lieto tieši šādu pirmo rindiņu:
Rādītājs / Kontugrupa\t01.2026.\t%\t02.2026.\t%\t03.2026.\t%\t04.2026.\t%\t05.2026.\t%\t06.2026.\t%\t07.2026.\t%\t08.2026.\t%\t2026 YTD\t%\t2026 AVG YTD\t%

VADĪBAS SECINĀJUMI:
Zem TSV pievieno:
### Kas ir labs un stabils
### Bīstamie signāli
### Vadības fokuss un uzlabojumi
"""

def highlight_toc_rows(row):
    label = str(row.iloc[0]).strip().upper()
    
    # 1. S (Sales) - Dzeltens
    if "S (SALES)" in label or label.startswith("S "):
        return ['background-color: #FFFF00; font-weight: bold; color: black;'] * len(row)
    # 2. TVC (Truly variable costs) kopējā rinda - Tumšāks terakotas oranžs
    elif "TVC (" in label or label == "TVC":
        return ['background-color: #ED7D31; font-weight: bold; color: white;'] * len(row)
    # 3. TVC apakšrindas - Gaišāks oranžs
    elif any(k in label for k in ["IZEJVIELAS", "PREČU PIEGĀDES"]):
        return ['background-color: #F8CBAD; color: black;'] * len(row)
    # 4. T (Throughput) - Spilgti zils
    elif "T (" in label or "THROUGHPUT" in label or label == "T":
        return ['background-color: #00B0F0; font-weight: bold; color: black;'] * len(row)
    # 5. OE (Operating expenses) kopējā rinda - Persiku / Bēšs tumšāks
    elif "OE (" in label or label == "OE":
        return ['background-color: #FCE4D6; font-weight: bold; color: black;'] * len(row)
    # 6. NP (Net Profit) - Spilgti zaļš
    elif "NP (" in label or "NET PROFIT" in label:
        return ['background-color: #92D050; font-weight: bold; color: black;'] * len(row)
    # 7. OE apakšrindas un koeficienti zem NP - Gaišs persiku tonis
    elif any(k in label for k in ["ALGAS", "PR =", "T% =", "EIRO SALES"]):
        return ['background-color: #FFF2CC; color: black;'] * len(row)
    return ['background-color: #FDFEFE; color: black;'] * len(row)

if uploaded_file is not None:
    if st.button("Veikt diagnostiku", type="primary"):
        with st.spinner("Apstrādā datus un veic TOC analīzi..."):
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                    st.session_state.file_text = df_raw.to_csv(index=False)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                    st.session_state.file_text = df_raw.to_csv(index=False)

                clean_key = str(ANTHROPIC_API_KEY).strip().strip('"').strip("'")
                client = anthropic.Anthropic(api_key=clean_key)

                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=8000,
                    system=TOC_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": f"Lūk, PZA dati analīzei:\n\n{st.session_state.file_text}"}
                    ]
                )

                output_text = ""
                for block in response.content:
                    if getattr(block, "type", "") == "text":
                        output_text += block.text
                    elif hasattr(block, "text"):
                        output_text += block.text

                st.session_state.output_text = output_text
                tsv_match = re.search(r"```tsv\n(.*?)```", output_text, re.DOTALL)

                if tsv_match:
                    tsv_data = tsv_match.group(1).strip()
                    df = pd.read_csv(io.StringIO(tsv_data), sep="\t")
                    # Izmetam ārā 'None' un tukšās atdalošās rindas
                    df = df[df.iloc[:, 0].notna()]
                    df = df[~df.iloc[:, 0].astype(str).str.strip().str.upper().isin(["NONE", "---", ""])]
                    st.session_state.df_res = df
                    st.session_state.chat_history = []

            except Exception as e:
                st.error(f"Kļūda apstrādē: {str(e)}")

if st.session_state.df_res is not None:
    st.success("Diagnostika ir aktīva!")

    # Sagatavojam Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state.df_res.to_excel(writer, index=False, sheet_name="TOC Diagnostika")
        ws = writer.sheets["TOC Diagnostika"]

        fill_yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        fill_tvc_dark = PatternFill(start_color="ED7D31", end_color="ED7D31", fill_type="solid")
        fill_tvc_light = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
        fill_cyan = PatternFill(start_color="00B0F0", end_color="00B0F0", fill_type="solid")
        fill_peach = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
        fill_green = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")

        thin_border = Border(
            left=Side(style='thin', color='BFBFBF'),
            right=Side(style='thin', color='BFBFBF'),
            top=Side(style='thin', color='BFBFBF'),
            bottom=Side(style='thin', color='BFBFBF')
        )

        # Galvene ar tīriem % virsrakstiem
        for col_num in range(1, len(st.session_state.df_res.columns) + 1):
            cell = ws.cell(row=1, column=col_num)
            col_name = str(cell.value)
            if col_name.startswith("%") or col_name.endswith("%"):
                cell.value = "%"
            cell.font = Font(name="Calibri", size=10, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Rindu formatēšana
        for r_idx, row in enumerate(st.session_state.df_res.itertuples(index=False), start=2):
            label = str(row[0]).strip().upper()
            row_fill = PatternFill(fill_type=None)
            row_font = Font(name="Calibri", size=10)

            if "S (SALES)" in label or label.startswith("S "):
                row_fill = fill_yellow
                row_font = Font(name="Calibri", size=10, bold=True)
            elif "TVC (" in label or label == "TVC":
                row_fill = fill_tvc_dark
                row_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
            elif any(k in label for k in ["IZEJVIELAS", "PREČU PIEGĀDES"]):
                row_fill = fill_tvc_light
                row_font = Font(name="Calibri", size=10)
            elif "T (" in label or "THROUGHPUT" in label or label == "T":
                row_fill = fill_cyan
                row_font = Font(name="Calibri", size=10, bold=True)
            elif "NP (" in label or "NET PROFIT" in label:
                row_fill = fill_green
                row_font = Font(name="Calibri", size=10, bold=True)
            elif "OE (" in label or any(k in label for k in ["ALGAS", "PR =", "T% =", "EIRO SALES"]):
                row_fill = fill_peach
                if "OE (" in label or "ALGAS KOPĀ" in label:
                    row_font = Font(name="Calibri", size=10, bold=True)

            for c_idx in range(1, len(st.session_state.df_res.columns) + 1):
                c = ws.cell(row=r_idx, column=c_idx)
                if row_fill.fill_type:
                    c.fill = row_fill
                c.font = row_font
                c.border = thin_border

        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 10)

    st.download_button(
        label="📥 Lejupielādēt noformēto Excel failu (.xlsx)",
        data=excel_buffer.getvalue(),
        file_name="TOC_Finansu_Diagnostika.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # UI parādīšanai noformējam kolonnu nosaukumus ar tīru '%'
    display_df = st.session_state.df_res.copy()
    cleaned_cols = []
    for c in display_df.columns:
        if re.match(r"^%\.?\d*$", str(c)) or str(c).endswith("%"):
            cleaned_cols.append("%")
        else:
            cleaned_cols.append(str(c))
    
    # Lai pandas neprotestētu par vienādiem nosaukumiem attēlojot:
    st.subheader("Finanšu pārskata tabula")
    styled_df = display_df.style.apply(highlight_toc_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Vadības analīze un secinājumi")
    clean_text = re.sub(r"```tsv.*?```", "", st.session_state.output_text, flags=re.DOTALL).strip()
    st.markdown(clean_text)

    st.markdown("---")
    st.subheader("💬 Uzdod jautājumu par šo TOC analīzi")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ieraksti jautājumu par pārskata datiem...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Domā..."):
                clean_key = str(ANTHROPIC_API_KEY).strip().strip('"').strip("'")
                client = anthropic.Anthropic(api_key=clean_key)
                
                chat_context = f"""
                Konteksts ar TOC analīzi un datiem:
                {st.session_state.output_text}
                
                Lietotāja jautājums:
                {user_query}
                
                Atbildi profesionāli, kodolīgi un balstoties uz šiem datiem un TOC principiem.
                """
                
                chat_resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": chat_context}]
                )
                
                reply = ""
                for block in chat_resp.content:
                    if getattr(block, "type", "") == "text":
                        reply += block.text
                    elif hasattr(block, "text"):
                        reply += block.text
                        
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

import io
import re
import anthropic
import pandas as pd
import streamlit as st
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

st.set_page_config(page_title="TOC Finanšu Diagnostika", layout="wide", page_icon="📊")

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

if "df_main" not in st.session_state:
    st.session_state.df_main = None
if "df_kpi" not in st.session_state:
    st.session_state.df_kpi = None
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
PZA satur gan starpsummas ('Kopā:', 'Pārdošanas izmaksas', 'Administrācijas izmaksas', 'Bruto peļņa'), gan atšifrējumus.
KRITISKS NOTEIKUMS: NEKAD NESKAITI DUBULTĀ! 
Ņem datus TIKAI no zemākajiem atšifrējuma posteņiem.
NEIEKĻAUJ rindas 'Algas kopā' vērtību summēšanā pie OE, jo administrācijas un strādnieku algas jau tiek skaitītas atsevišķi!

TOC APRĒĶINA LOĢIKA:
1. S (Sales): Neto apgrozījums (100%).
2. TVC: Izejvielas un materiāli + Preču piegādes izdevumi.
3. T = S - TVC.
4. OE (Operating Expenses): Saskaita TIKAI šīs atsevišķās grupas (bez jebkādas dubultas starpsummu pieskaitīšanas):
   - Administrācijas algas (+ sociālās iemaksas)
   - Strādnieku algas (+ sociālās iemaksas)
   - Mārketings un reklāma
   - Pārdošanas platformu izdevumi un pārējās pārdošanas izmaksas (ar US TAX, mājas lapu)
   - Ražošanas telpu noma un uzturēšana (+ pārējās ražošanas izmaksas)
   - Biroja, IT un sakaru izmaksas
   - Citi saimnieciskie izdevumi (atskaitot pārējos ieņēmumus, izslēdzot procentus un UIN)
   * KONTROLE: Janvārī OE KOPĀ ir jābūt aptuveni 301 342 EUR, un NP = T - OE ir jābūt peļņai +190 110 EUR!
5. NP = T - OE.

DIVAS TSV TABULAS ATBILDĒ:
Lai atdalītu produktivitātes datus, ģenerē precīzi divus TSV koda blokus:

Pirmais bloks (Finanšu pārskats):
```tsv_main
Rādītājs / Kontugrupa\t01.2026.\t%\t02.2026.\t%\t03.2026.\t%\t04.2026.\t%\t05.2026.\t%\t06.2026.\t%\t07.2026.\t%\t08.2026.\t%\t2026 YTD\t%\t2026 AVG YTD\t%
S\t...
TVC\t...
Izejvielas un materiāli\t...
Preču piegādes izdevumi\t...
T\t...
OE\t...
Administrācijas algas (+ sociālās iemaksas)\t...
Strādnieku algas (+ sociālās iemaksas)\t...
Mārketings un reklāma\t...
Pārdošanas platformas un citas pārdošanas izm.\t...
Ražošanas telpu noma un uzturēšana\t...
Biroja, IT un sakaru izmaksas\t...
Citi saimnieciskie izdevumi\t...
NP\t...
```

Otrais bloks (Produktivitātes rādītāji):
```tsv_kpi
Rādītājs\t01.2026.\t02.2026.\t03.2026.\t04.2026.\t05.2026.\t06.2026.\t07.2026.\t08.2026.\t2026 YTD\t2026 AVG YTD
PR = T/OE\t...
T% = T/S\t...
Eiro Sales / Eiro algas kopā\t...
Eiro Sales / Eiro administrācijas algas\t...
Eiro Sales / Eiro strādnieku algas\t...
```

Tūlīt pēc blokiem pievieno 3 īsas sadaļas:
### Kas ir labs un stabils
### Bīstamie signāli
### Vadības fokuss un uzlabojumi
"""

def highlight_main_rows(row):
    label = str(row.iloc[0]).strip().upper()
    
    # S rinda - Dzeltens
    if label in ["S", "S (SALES)"] or label.startswith("S "):
        return ['background-color: #FFFF00; font-weight: bold; color: black;'] * len(row)
    # TVC kopā - Terakotas / tumši oranžs
    elif label in ["TVC", "TVC (TRULY VARIABLE COSTS)"]:
        return ['background-color: #ED7D31; font-weight: bold; color: white;'] * len(row)
    # TVC apakšrindas - Gaiši oranžs
    elif any(k in label for k in ["IZEJVIELAS", "PREČU PIEGĀDES"]):
        return ['background-color: #F8CBAD; color: black;'] * len(row)
    # T rinda - Zils
    elif label in ["T", "T (THROUGHPUT)"] or "THROUGHPUT" in label:
        return ['background-color: #00B0F0; font-weight: bold; color: black;'] * len(row)
    # OE kopējā rinda
    elif label in ["OE", "OE (OPERATING EXPENSES)"]:
        return ['background-color: #F8CECC; font-weight: bold; color: black;'] * len(row)
    # NP rinda - Zaļš
    elif label in ["NP", "NP (NET PROFIT)"] or "NET PROFIT" in label:
        return ['background-color: #92D050; font-weight: bold; color: black;'] * len(row)
    # Visas pārējās (OE apakšrindas) - Persiku tonis
    return ['background-color: #FCE4D6; color: black;'] * len(row)

def highlight_kpi_rows(row):
    return ['background-color: #F2F4F4; color: black; font-weight: 500;'] * len(row)

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
                
                m_match = re.search(r"```tsv_main\n(.*?)```", output_text, re.DOTALL)
                k_match = re.search(r"```tsv_kpi\n(.*?)```", output_text, re.DOTALL)

                if m_match:
                    df_m = pd.read_csv(io.StringIO(m_match.group(1).strip()), sep="\t")
                    st.session_state.df_main = df_m[df_m.iloc[:, 0].notna()]
                if k_match:
                    df_k = pd.read_csv(io.StringIO(k_match.group(1).strip()), sep="\t")
                    st.session_state.df_kpi = df_k[df_k.iloc[:, 0].notna()]

                st.session_state.chat_history = []

            except Exception as e:
                st.error(f"Kļūda apstrādē: {str(e)}")

if st.session_state.df_main is not None:
    st.success("Diagnostika ir veiksmīgi sagatavota!")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state.df_main.to_excel(writer, index=False, sheet_name="TOC Diagnostika", startrow=0)
        ws = writer.sheets["TOC Diagnostika"]

        fill_yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        fill_tvc_dark = PatternFill(start_color="ED7D31", end_color="ED7D31", fill_type="solid")
        fill_tvc_light = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")
        fill_cyan = PatternFill(start_color="00B0F0", end_color="00B0F0", fill_type="solid")
        fill_oe_main = PatternFill(start_color="F8CECC", end_color="F8CECC", fill_type="solid")
        fill_peach = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
        fill_green = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
        fill_kpi = PatternFill(start_color="F2F4F4", end_color="F2F4F4", fill_type="solid")

        thin_border = Border(
            left=Side(style='thin', color='BFBFBF'),
            right=Side(style='thin', color='BFBFBF'),
            top=Side(style='thin', color='BFBFBF'),
            bottom=Side(style='thin', color='BFBFBF')
        )

        for col_num in range(1, len(st.session_state.df_main.columns) + 1):
            cell = ws.cell(row=1, column=col_num)
            if "%" in str(cell.value):
                cell.value = "%"
            cell.font = Font(name="Calibri", size=10, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        last_row = 1
        for r_idx, row in enumerate(st.session_state.df_main.itertuples(index=False), start=2):
            label = str(row[0]).strip().upper()
            row_fill = fill_peach
            row_font = Font(name="Calibri", size=10)

            if label in ["S", "S (SALES)"] or label.startswith("S "):
                row_fill = fill_yellow
                row_font = Font(name="Calibri", size=10, bold=True)
            elif label in ["TVC", "TVC (TRULY VARIABLE COSTS)"]:
                row_fill = fill_tvc_dark
                row_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
            elif any(k in label for k in ["IZEJVIELAS", "PREČU PIEGĀDES"]):
                row_fill = fill_tvc_light
            elif label in ["T", "T (THROUGHPUT)"] or "THROUGHPUT" in label:
                row_fill = fill_cyan
                row_font = Font(name="Calibri", size=10, bold=True)
            elif label in ["OE", "OE (OPERATING EXPENSES)"]:
                row_fill = fill_oe_main
                row_font = Font(name="Calibri", size=10, bold=True)
            elif label in ["NP", "NP (NET PROFIT)"] or "NET PROFIT" in label:
                row_fill = fill_green
                row_font = Font(name="Calibri", size=10, bold=True)

            for c_idx in range(1, len(st.session_state.df_main.columns) + 1):
                c = ws.cell(row=r_idx, column=c_idx)
                c.fill = row_fill
                c.font = row_font
                c.border = thin_border
            last_row = r_idx

        if st.session_state.df_kpi is not None:
            start_kpi_row = last_row + 3
            for col_idx, col_name in enumerate(st.session_state.df_kpi.columns, start=1):
                c = ws.cell(row=start_kpi_row, column=col_idx, value=col_name)
                c.font = Font(name="Calibri", size=10, bold=True)
                c.alignment = Alignment(horizontal="center", vertical="center")
            
            for r_offset, row in enumerate(st.session_state.df_kpi.itertuples(index=False), start=1):
                cur_r = start_kpi_row + r_offset
                for c_idx in range(1, len(st.session_state.df_kpi.columns) + 1):
                    c = ws.cell(row=cur_r, column=c_idx, value=row[c_idx - 1])
                    c.fill = fill_kpi
                    c.font = Font(name="Calibri", size=10)
                    c.border = thin_border

        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 11)

    st.download_button(
        label="📥 Lejupielādēt noformēto Excel failu (.xlsx)",
        data=excel_buffer.getvalue(),
        file_name="TOC_Finansu_Diagnostika.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("Finanšu pārskata tabula")
    styled_main = st.session_state.df_main.style.apply(highlight_main_rows, axis=1)
    st.dataframe(styled_main, use_container_width=True)

    if st.session_state.df_kpi is not None:
        st.subheader("📈 Produktivitātes rādītāji")
        styled_kpi = st.session_state.df_kpi.style.apply(highlight_kpi_rows, axis=1)
        st.dataframe(styled_kpi, use_container_width=True)

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

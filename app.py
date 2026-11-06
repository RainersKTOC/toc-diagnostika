"""
app.py — TOC-FIN Vadības lēmumu un finanšu diagnostikas sistēma (v14.0)
- Visur pieejams AI diagnostikas un atbalsta čats (redz PZA struktūru, kontus un aprēķinus)
- Manuāla PZA rindu korekcija (ja kāds konts nolasīts kā starpsumma vai otrādi)
- Finanšu gada sākuma mēneša izvēle (Janvāris..Decembris, piem., no Septembra)
- Uzkrājošā (YTD) tabula: S, TVC, T un NP no finanšu gada sākuma pelēkā dizainā
- Salabota skaitļu apcirpšana (vairs nerāda '72...')
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import streamlit as st

from db import SupabaseProfileStore
from toc_engine import (
    GROUPS,
    SUBGROUP_LABELS,
    LABEL_TO_KEY,
    MAIN_GROUP_TO_DEFAULT_SUB,
    FILLS,
    build_excel,
    classify_with_claude,
    compute_toc,
)
from toc_ingest import (
    Structure,
    analyze,
    extract_lines,
    load_grids,
    pick_sheet,
    request_structure,
)

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="TOC-FIN Vadības Sistēma", layout="wide", page_icon="📊")


@dataclass
class LoadedReportResult:
    amounts: pd.DataFrame
    main: pd.DataFrame
    kpi: pd.DataFrame
    row_kinds: dict[str, str]
    reconciliation: pd.DataFrame
    by_group: dict


API_KEY = str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip().strip('"').strip("'")
if not API_KEY:
    st.error("Nav atrasta ANTHROPIC_API_KEY. Pievienojiet to Streamlit Secrets iestatījumos.")
    st.stop()


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=API_KEY)


def fmt_num(x) -> str:
    return "" if pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def fmt_pct_int(x) -> str:
    return "" if pd.isna(x) else f"{x * 100:.0f}%"


MONTH_NAMES_LV = {
    1: "1. Janvāris (Jan–Dec)",
    2: "2. Februāris (Feb–Jan)",
    3: "3. Marts (Mar–Feb)",
    4: "4. Aprīlis (Apr–Mar)",
    5: "5. Maijs (Mai–Apr)",
    6: "6. Jūnijs (Jūn–Mai)",
    7: "7. Jūlijs (Jūl–Jūn)",
    8: "8. Augusts (Aug–Jūl)",
    9: "9. Septembris (Sep–Aug)",
    10: "10. Oktobris (Okt–Sep)",
    11: "11. Novembris (Nov–Okt)",
    12: "12. Decembris (Dec–Nov)",
}

MONTH_SHORT_LV = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mai", 6: "Jūn",
    7: "Jūl", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec",
}

LV_MONTH_PARSE = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mai": 5, "may": 5,
    "jūn": 6, "jun": 6, "jūl": 7, "jul": 7, "aug": 8, "sep": 9,
    "okt": 10, "oct": 10, "nov": 11, "dec": 12,
}


def parse_period_month_year(col_name: str) -> tuple[int | None, int | None]:
    """Atpazīst mēnesi (1..12) un gadu no perioda kolonnas nosaukuma (piem., '01.2026.', '2026-09', 'Sep 2025')."""
    s = str(col_name or "").strip().lower()
    if "ytd" in s or "avg" in s or "kopā" in s or "izmaiņa" in s:
        return None, None

    # 1) MM.YYYY vai MM/YYYY vai MM-YYYY
    m1 = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[\./\-](20\d{2})(?!\d)", s)
    if m1:
        return int(m1.group(1)), int(m1.group(2))

    # 2) YYYY.MM vai YYYY-MM vai YYYY/MM
    m2 = re.search(r"(?<!\d)(20\d{2})[\./\-](0?[1-9]|1[0-2])(?!\d)", s)
    if m2:
        return int(m2.group(2)), int(m2.group(1))

    # 3) Teksta mēnesis
    yr_m = re.search(r"(20\d{2})", s)
    yr = int(yr_m.group(1)) if yr_m else None
    for k, m_num in LV_MONTH_PARSE.items():
        if k in s:
            return m_num, yr

    return None, None


def compute_cumulative_toc_df(amounts_df: pd.DataFrame, fy_start_month: int = 1) -> tuple[pd.DataFrame, list[str], set[str]]:
    """
    Aprēķina uzkrājošo (kumulatīvo) S, TVC, T un NP tabulu no izvēlētā finanšu gada sākuma mēneša.
    Ja tabulā ir vairāki finanšu gadi, uzkrājums automātiski sākas no nulles katrā jaunā finanšu gada sākumā.
    Atgriež: (cum_df, period_cols, fy_boundary_cols)
    """
    target_rows = [r for r in ["S", "TVC", "T", "NP"] if r in amounts_df.index]
    period_cols = [
        c for c in amounts_df.columns
        if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper() or "IZMAIŅA" in str(c).upper())
    ]
    if not period_cols:
        period_cols = list(amounts_df.columns)

    cum_data: dict[str, pd.Series] = {}
    running = pd.Series(0.0, index=target_rows)
    fy_boundary_cols: set[str] = set()

    for idx, col in enumerate(period_cols):
        m_num, _ = parse_period_month_year(str(col))
        if idx == 0:
            running = amounts_df.loc[target_rows, col].fillna(0.0).astype(float).copy()
            if m_num == fy_start_month:
                fy_boundary_cols.add(col)
        else:
            if m_num is not None and m_num == fy_start_month:
                # Jauns finanšu gads sākas šajā kolonnā — sākam uzkrāšanu no jauna
                running = amounts_df.loc[target_rows, col].fillna(0.0).astype(float).copy()
                fy_boundary_cols.add(col)
            else:
                running = running + amounts_df.loc[target_rows, col].fillna(0.0).astype(float)
        cum_data[col] = running.copy()

    cum_df = pd.DataFrame(cum_data, index=target_rows)
    row_labels = {
        "S": "∑ S — Uzkrātie ieņēmumi (Sales)",
        "TVC": "∑ TVC — Uzkrātās patiesi mainīgās izmaksas",
        "T": "∑ T — Uzkrātā caurlaidība (Throughput)",
        "NP": "∑ NP — Uzkrātā neto peļņa (Net Profit)",
    }
    cum_df.index = [row_labels.get(r, r) for r in cum_df.index]
    return cum_df, period_cols, fy_boundary_cols


MAIN_CAT_BY_CODE = {
    "S": "🟡 S (Ieņēmumi)",
    "TVC": "🟠 TVC (Mainīgās izmaksas)",
    "OE": "🟤 OE (Uzturēšanas izdevumi)",
    "EXCL": "⚪ Izslēgtie (Virtuālā nauda)",
}

RADIO_OPTIONS = [
    "🟡 S — Ieņēmumi (Sales)",
    "🟠 TVC — Patiesi mainīga izmaksa (1:1 ar pārdošanu)",
    "🟤 OE — Uzturēšanas / darbības izdevums",
    "⚪ Izslēgt — Virtuāla nauda (amortizācija)",
]


def normalize_group_key(group_key: str) -> str:
    g = str(group_key or "OE_OTHER").strip()
    if g == "S":
        return "S_REV"
    if g not in SUBGROUP_LABELS:
        return "OE_OTHER"
    return g


def get_main_code(group_key: str) -> str:
    g = normalize_group_key(group_key)
    return GROUPS.get(g, ("", "OE"))[1]


def get_main_cat(group_key: str) -> str:
    code = get_main_code(group_key)
    return MAIN_CAT_BY_CODE.get(code, MAIN_CAT_BY_CODE["OE"])


def get_radio_index_for_group(group_key: str) -> int:
    code = get_main_code(group_key)
    return {"S": 0, "TVC": 1, "OE": 2, "EXCL": 3}.get(code, 2)


def resolve_radio_choice(choice: str, original_group: str, konts_label: str) -> tuple[str, bool]:
    orig_grp = normalize_group_key(original_group)
    orig_code = get_main_code(orig_grp)
    lbl_low = str(konts_label or "").lower()

    if choice.startswith("🟡"):
        target_code = "S"
        fallback = "S_OTHER" if ("grant" in lbl_low or "citi" in lbl_low or "pārēj" in lbl_low) else "S_REV"
    elif choice.startswith("🟠"):
        target_code = "TVC"
        fallback = "TVC_DELIVERY" if ("piegād" in lbl_low or "transp" in lbl_low or "kurjer" in lbl_low) else "TVC_MATERIALS"
    elif choice.startswith("⚪"):
        target_code = "EXCL"
        fallback = "EXCLUDED"
    else:
        target_code = "OE"
        fallback = "OE_OTHER"

    final_grp = orig_grp if orig_code == target_code else fallback
    return final_grp, final_grp.startswith("S_")


GROUP_SORT_ORDER = {
    "S": 1,
    "S_REV": 1,
    "S_OTHER": 2,
    "TVC_MATERIALS": 3,
    "TVC_DELIVERY": 4,
    "OE_ADMIN_WAGES": 5,
    "OE_WORKER_WAGES": 6,
    "OE_MARKETING": 7,
    "OE_PLATFORMS": 8,
    "OE_PREMISES": 9,
    "OE_OFFICE_IT": 10,
    "OE_FIN_TAX": 11,
    "OE_OTHER": 12,
    "EXCLUDED": 13,
}


def sort_cls_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Grupa"] = out["Grupa"].map(normalize_group_key)
    out["_sort"] = out["Grupa"].map(lambda g: GROUP_SORT_ORDER.get(g, 99))
    out = out.sort_values(by=["_sort", "Displeja_konts"], kind="mergesort").drop(columns=["_sort"]).reset_index(drop=True)
    return out


def build_display_labels(leaves) -> dict[str, str]:
    counts: dict[str, int] = {}
    for lf in leaves:
        lbl = str(lf.label).strip()
        counts[lbl] = counts.get(lbl, 0) + 1

    display_map: dict[str, str] = {}
    for lf in leaves:
        lbl = str(lf.label).strip()
        if counts.get(lbl, 0) > 1:
            path_parts = [p.strip() for p in str(getattr(lf, "path", "")).split(">") if p.strip()]
            parent = path_parts[-2] if len(path_parts) >= 2 else (path_parts[0] if path_parts else f"rinda {lf.row}")
            display_map[lf.id] = f"{lbl}  [{parent}]"
        else:
            display_map[lf.id] = lbl
    return display_map


def render_unified_toc_tables_html(
    amounts_df: pd.DataFrame,
    main_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    row_kinds: dict[str, str],
    title_kpi: str = "📈 Produktivitātes un atdeves rādītāji",
    grey_cols: set[str] | None = None,
    fy_start_month: int = 1,
    show_cumulative: bool = True,
) -> str:
    periods = list(amounts_df.columns)
    row_names = list(amounts_df.index)
    kinds = [row_kinds.get(rn, "OE_CHILD") for rn in row_names]
    grey_cols = grey_cols or set()

    # Izmantojam table-layout: auto + horizontālo ritināšanu, lai skaitļi NEKAD netiktu apgriezti ar '72...'
    html = ["""
    <style>
    .toc-table-wrap { width: 100%; overflow-x: auto; margin-bottom: 18px; }
    table.toc-unified { width: 100%; border-collapse: collapse; font-family: 'Calibri', 'Segoe UI', sans-serif; font-size: 12px; table-layout: auto; }
    table.toc-unified th { background-color: #EAECEE; color: #111111; font-weight: 700; text-align: center; padding: 6px 5px; border-top: 2px solid #404040; border-bottom: 2px solid #404040; white-space: nowrap; }
    table.toc-unified td { padding: 4px 6px; white-space: nowrap; }
    table.toc-unified td.lbl-main { text-align: left; font-weight: 700; padding-left: 8px; padding-right: 10px; border-left: 2px solid #404040; border-right: 2px solid #404040; min-width: 210px; }
    table.toc-unified td.lbl-sub { text-align: left; font-weight: 400; padding-left: 18px; padding-right: 10px; border-left: 2px solid #404040; border-right: 2px solid #404040; min-width: 210px; }
    table.toc-unified td.val-col { text-align: right; border-left: 1px solid #BFBFBF; font-variant-numeric: tabular-nums; }
    table.toc-unified td.pct-col { text-align: right; font-size: 11px; padding-left: 3px; padding-right: 5px; border-left: 1px dotted #D0D3D4; border-right: 2px solid #595959; color: #333333; font-variant-numeric: tabular-nums; }
    table.toc-unified th.th-val { border-left: 1px solid #BFBFBF; }
    table.toc-unified th.th-pct { font-size: 11px; border-left: 1px dotted #BFBFBF; border-right: 2px solid #595959; }
    table.toc-unified th.th-lbl { text-align: left; padding-left: 8px; border-left: 2px solid #404040; border-right: 2px solid #404040; }
    table.toc-unified tr.spacer-row td { background: transparent !important; border: none !important; height: 28px; padding: 14px 0 6px 0; font-size: 15px; font-weight: 700; color: #1f2937; }
    </style>
    <div class="toc-table-wrap"><table class="toc-unified">
    <thead><tr>
    """]
    html.append('<th class="th-lbl">Rādītājs / Kontugrupa</th>')
    for p in periods:
        th_bg = "background-color: #D5D8DC; color: #333333;" if p in grey_cols else ""
        html.append(f'<th class="th-val" style="{th_bg}">{p}</th><th class="th-pct" style="{th_bg}">%</th>')
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

        row_style = (
            f"background-color: #{fill_hex}; color: #{text_hex}; font-weight: {font_w}; "
            f"border-top: 1px solid #BFBFBF; border-bottom: {border_bottom};"
        )
        html.append(f'<tr style="{row_style}"><td class="{lbl_cls}" title="{rname}">{rname}</td>')

        for p in periods:
            val_str = fmt_num(main_df.loc[rname, p])
            pct_str = fmt_pct_int(main_df.loc[rname, f"%|{p}"])
            if p in grey_cols:
                g_bg = "#D5D8DC" if is_bold else "#EAEDED"
                g_txt = "#2C3E50" if is_bold else "#4D5656"
                cell_style = f"background-color: {g_bg}; color: {g_txt};"
            else:
                cell_style = f"color: #{text_hex};"
            html.append(f'<td class="val-col" style="{cell_style}">{val_str}</td><td class="pct-col" style="{cell_style}">{pct_str}</td>')
        html.append("</tr>")

    total_cols = 1 + 2 * len(periods)

    # --- 2. TABULA: PRODUKTIVITĀTES UN ATDEVES RĀDĪTĀJI ---
    html.append(f'<tr class="spacer-row"><td colspan="{total_cols}">{title_kpi}</td></tr>')
    html.append("<tr>")
    html.append('<th class="th-lbl" style="background-color: #EAECEE; border-top: 2px solid #404040; border-bottom: 2px solid #404040;">Produktivitātes rādītājs</th>')
    for p in periods:
        th_bg = "#D5D8DC" if p in grey_cols else "#EAECEE"
        html.append(
            f'<th colspan="2" style="background-color: {th_bg}; border-top: 2px solid #404040; '
            f'border-bottom: 2px solid #404040; border-left: 1px solid #BFBFBF; border-right: 2px solid #595959;">{p}</th>'
        )
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
            kpi_cell_bg = "background-color: #E5E7E9; color: #4D5656;" if p in grey_cols else ""
            html.append(
                f'<td colspan="2" style="text-align: right; padding: 4px 6px; '
                f'border-left: 1px solid #BFBFBF; border-right: 2px solid #595959; font-weight: 600; {kpi_cell_bg}">{s}</td>'
            )
        html.append("</tr>")

    # --- 3. TABULA: UZKRĀJOŠIE RĀDĪTĀJI NO FINANŠU GADA SĀKUMA (ATŠĶIRĪGĀ PELĒKĀ NOFORMĒJUMĀ) ---
    if show_cumulative:
        cum_df, cum_periods, fy_boundaries = compute_cumulative_toc_df(amounts_df, fy_start_month=fy_start_month)
        fy_lbl = MONTH_NAMES_LV.get(fy_start_month, "1. Janvāris")
        html.append(
            f'<tr class="spacer-row"><td colspan="{total_cols}">'
            f'📊 Uzkrājošā finanšu dinamika no finanšu gada sākuma (Sākuma mēnesis: {fy_lbl})'
            f'</td></tr>'
        )
        # Tumšāka antracīta-pelēka galvene, lai vizuāli atšķirtos no KPI tabulas
        html.append("<tr>")
        html.append(
            '<th class="th-lbl" style="background-color: #34495E; color: #FFFFFF; '
            'border-top: 2px solid #1C2833; border-bottom: 2px solid #1C2833;">'
            'Uzkrājošais rādītājs (YTD dinamika)</th>'
        )
        for p in periods:
            is_fy_start = p in fy_boundaries and p != periods[0]
            left_b = "3px solid #1C2833" if is_fy_start else "1px solid #5D6D7E"
            badge = " ★" if (p in fy_boundaries) else ""
            if p in cum_periods:
                html.append(
                    f'<th class="th-val" style="background-color: #34495E; color: #FFFFFF; border-top: 2px solid #1C2833; border-bottom: 2px solid #1C2833; border-left: {left_b};">{p}{badge}</th>'
                    f'<th class="th-pct" style="background-color: #2C3E50; color: #D5D8DC; border-top: 2px solid #1C2833; border-bottom: 2px solid #1C2833; border-right: 2px solid #1C2833;">∑%</th>'
                )
            else:
                html.append(
                    f'<th colspan="2" style="background-color: #2C3E50; color: #BDC3C7; border-top: 2px solid #1C2833; border-bottom: 2px solid #1C2833; border-left: 1px solid #5D6D7E; border-right: 2px solid #1C2833;">{p}</th>'
                )
        html.append("</tr>")

        # 4 uzkrājošās rindas ar atšķirīgu pelēko gradāciju un kreiso akcenta joslu
        cum_styles = [
            # (fona_krāsa, teksta_krāsa, kreisais_akcents, ir_izcelts)
            ("#EBEDEF", "#1B2631", "#F1C40F", True),   # ∑ S
            ("#E5E7E9", "#2C3E50", "#D35400", False),  # ∑ TVC
            ("#D6DBDF", "#11171D", "#2980B9", True),   # ∑ T
            ("#CCD1D1", "#0B5345", "#27AE60", True),   # ∑ NP
        ]
        s_row_label = cum_df.index[0] if len(cum_df.index) > 0 else None

        for idx_c, c_name in enumerate(cum_df.index):
            bg_c, txt_c, accent_c, bold_c = cum_styles[idx_c % len(cum_styles)]
            b_bot = "2.5px solid #1C2833" if idx_c == len(cum_df.index) - 1 else "1px solid #ABB2B9"
            fw_c = "700" if bold_c else "600"
            html.append(
                f'<tr style="background-color: {bg_c}; color: {txt_c}; font-weight: {fw_c}; '
                f'border-top: 1px solid #ABB2B9; border-bottom: {b_bot};">'
            )
            html.append(
                f'<td class="lbl-main" style="border-left: 5px solid {accent_c}; font-weight: {fw_c}; color: {txt_c};" title="{c_name}">{c_name}</td>'
            )
            for p in periods:
                if p in cum_df.columns:
                    val_c = cum_df.loc[c_name, p]
                    s_c = cum_df.loc[s_row_label, p] if s_row_label else np.nan
                    pct_c = (val_c / s_c) if (s_c and not pd.isna(s_c) and s_c != 0) else np.nan
                    val_s = fmt_num(val_c)
                    pct_s = fmt_pct_int(pct_c)
                    # Negatīvu uzkrāto NP iezīmējam tumši sarkanā tonī uz pelēkā fona
                    val_color = "#922B21" if (not pd.isna(val_c) and val_c < 0 and "NP" in c_name) else txt_c
                    html.append(
                        f'<td class="val-col" style="color: {val_color}; font-weight: {fw_c};">{val_s}</td>'
                        f'<td class="pct-col" style="background-color: #DDE1E4; color: #2C3E50; font-weight: 600;">{pct_s}</td>'
                    )
                else:
                    # YTD vai AVG kolonnām rādām pēdējā perioda uzkrājumu vai mēneša vidējo
                    last_p = cum_periods[-1] if cum_periods else None
                    if last_p and "YTD" in str(p).upper():
                        v_ytd = cum_df.loc[c_name, last_p]
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; background-color: #D5D8DC; color: #1B2631; font-weight: 700; border-right: 2px solid #404040;">{fmt_num(v_ytd)}</td>'
                        )
                    elif last_p and "AVG" in str(p).upper() and len(cum_periods) > 0:
                        v_avg = cum_df.loc[c_name, last_p] / len(cum_periods)
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; background-color: #EAEDED; color: #2C3E50; font-weight: 600; border-right: 2px solid #404040;">{fmt_num(v_avg)}</td>'
                        )
                    else:
                        html.append('<td colspan="2" style="background-color: #EAEDED; border-right: 2px solid #404040;"></td>')
            html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

S = st.session_state


# ------------------------------------------------------------------ VISUR PIEEJAMĀ AĢENTA KONTEKSTA BŪVĒTĀJS
def build_global_diagnostic_context() -> str:
    """Sagatavo pilnu tehnisko un finanšu kontekstu par jebkuru posmu, kurā lietotājs šobrīd atrodas."""
    parts = [f"Aktīvais uzņēmuma profils: {S.get('active_profile', 'Nav')}"]
    parts.append(f"Izvēlētais finanšu gada sākuma mēnesis: {MONTH_NAMES_LV.get(S.get('fy_start_month', 1), 'Janvāris')}")

    if "stc" in S or "structure" in S:
        stc: Structure = S.get("structure")
        if stc:
            parts.append(
                f"\n--- 1. FAILA STRUKTŪRAS DIAGNOSTIKA (toc_ingest) ---\n"
                f"Izvēlētā lapa (sheet): {stc.sheet}\n"
                f"Kontu nosaukumu kolonnas: {stc.label_cols}\n"
                f"Periodu galvenes rinda (header_row): {stc.header_row}\n"
                f"PZA gala rezultāta rinda (result_row): {stc.result_row}\n"
                f"Periodu kolonnas: {stc.period_cols}"
            )

    if "analysis" in S and S.analysis is not None:
        an = S.analysis
        leaf_ids = {lf.id for lf in an.leaves}
        lines_info = []
        for ln in an.ex.lines:
            status = "IEKĻAUTS KĀ KONTS (LEAF)" if ln.id in leaf_ids else (
                "GALA REZULTĀTA RINDA" if (S.get("structure") and ln.row == S.structure.result_row)
                else "IZLAISTS (ATPAZĪTS KĀ VIRSRAKSTS VAI STARPSUMMA)"
            )
            tot_val = float(ln.vals.sum()) if hasattr(ln, "vals") else 0.0
            first_val = float(ln.vals.iloc[0]) if (hasattr(ln, "vals") and len(ln.vals) > 0) else 0.0
            lines_info.append(
                f"Rinda #{ln.row} (id={ln.id}): '{ln.label}' | Ceļš: '{ln.path}' | "
                f"Statuss: {status} | Pirmais mēn.: {first_val:,.2f} | Kopā pa periodiem: {tot_val:,.2f}"
            )
        parts.append(
            "\n--- 2. VISAS NO FAILA NOLASĪTĀS RINDAS UN TO STATUSS ---\n"
            + "\n".join(lines_info[:120])
        )

    if "cls_df" in S and S.cls_df is not None:
        df_c = S.cls_df[["id", "Konts", "Displeja_konts", "Grupa", "Ieņēmums", "Avots"]]
        parts.append("\n--- 3. PAŠREIZĒJĀ KONTU KLASIFIKĀCIJA ---\n" + df_c.to_markdown(index=False))

    if "result" in S and S.result is not None:
        res = S.result
        parts.append("\n--- 4. APRĒĶINĀTĀ TOC-FIN TABULA (EUR) ---\n" + res.amounts.round(0).to_markdown())
        parts.append("\n--- 5. PRODUKTIVITĀTES RĀDĪTĀJI (KPI) ---\n" + res.kpi.round(3).to_markdown())
        if getattr(res, "reconciliation", None) is not None and not res.reconciliation.empty:
            parts.append("\n--- 6. SALĪDZINĀJUMS AR GRĀMATVEDĪBAS PZA REZULTĀTU (RECONCILIATION) ---\n" + res.reconciliation.round(0).to_markdown())

    return "\n\n".join(parts)


def ask_global_assistant(user_question: str) -> str:
    ctx = build_global_diagnostic_context()
    know = f"\nTOC-FIN metodoloģija:\n{knowledge_file}\n" if knowledge_file else ""
    sys_prompt = (
        "Tu esi TOC-FIN sistēmas galvenais finanšu un tehniskās diagnostikas aģents. "
        "Lietotājs var tev uzdot jautājumus JEBKURĀ posmā — gan par to, kāpēc kāds PZA konts vai rinda no Excel/CSV faila "
        "ir nolasīta nepareizi (vai izlaista kā starpsumma, vai nepareizā kolonnā/zīmē), gan par kontu klasifikāciju, "
        "gan par vadības finanšu rādītājiem (S, TVC, T, OE, NP, PR). "
        "Ja lietotājs jautā par nepareizi nolasītu kontu: pārbaudi 'VISAS NO FAILA NOLASĪTĀS RINDAS UN TO STATUSS', "
        "pasaki precīzi, kurā rindā (#row) šis konts atrodas, kāds ir tā statuss (IEKĻAUTS vai IZLAISTS kā starpsumma) un kādas summas tam ir nolasītas, "
        "un paskaidro, kā lietotājs to var izlabot turpat sadaļā '🛠️ PZA struktūras un rindu korekcija' vai kontu tabulā. "
        f"{know}"
        "Atbildi latviešu valodā — skaidri, konkrēti un ar precīziem skaitļiem no konteksta."
    )
    history = S.get("global_chat", [])
    messages = [
        {"role": "user", "content": f"Šī brīža sistēmas un PZA faila stāvoklis:\n{ctx}"},
        {"role": "assistant", "content": "Esmu iepazinies ar pilno faila struktūru, rindām un aprēķiniem. Kā varu palīdzēt?"},
    ] + history + [{"role": "user", "content": user_question}]

    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=1800,
        system=sys_prompt,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


# ------------------------------------------------------------------ AUTENTIFIKĀCIJAS EKRĀNS (LOGIN / REGISTER)
if not S.get("authenticated", False):
    st.title("🔒 TOC-FIN Finanšu Vadības Sistēma — Autorizācija")
    st.markdown("Katram uzņēmumam ir sava droša, izolēta vide. Lūdzu, ielogojieties ar savu uzņēmuma profilu vai izveidojiet jaunu.")

    auth_tab1, auth_tab2 = st.tabs(["🔑 Ielagoties", "🏢 Izveidot jaunu uzņēmuma profilu"])

    temp_store = SupabaseProfileStore()

    with auth_tab1:
        st.subheader("Ielagošanās esošā uzņēmuma profilā")
        with st.form("login_form"):
            login_company = st.text_input("Uzņēmuma nosaukums:", placeholder="Piemēram, SIA MansUzņēmums")
            login_password = st.text_input("Parole:", type="password")
            login_submitted = st.form_submit_button("🚀 Ielogoties", type="primary")

            if login_submitted:
                ok, msg = temp_store.authenticate(login_company, login_password)
                if ok:
                    S.authenticated = True
                    S.active_profile = login_company.strip()
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with auth_tab2:
        st.subheader("Reģistrēt jaunu uzņēmumu")
        st.info("Šeit izveidotais profils nodrošinās, ka neviens cits uzņēmums nevarēs piekļūt jūsu datiem vai kontu klasifikācijai.")

        with st.form("register_form"):
            reg_company = st.text_input("Jauna uzņēmuma nosaukums:", placeholder="Piemēram, SIA Ražotājs")
            reg_password = st.text_input("Izveidojiet paroli (min. 4 simboli):", type="password")
            reg_submitted = st.form_submit_button("✨ Reģistrēt un ielogoties", type="primary")

            if reg_submitted:
                ok, msg = temp_store.register_company(reg_company, reg_password)
                if ok:
                    S.authenticated = True
                    S.active_profile = reg_company.strip()
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    st.stop()


# ------------------------------------------------------------------ GALVENĀ APLIKĀCIJA (PĒC AUTENTIFIKĀCIJAS)
profile = SupabaseProfileStore(company_name=S.active_profile)

if "fy_start_month" not in S:
    S.fy_start_month = 1

with st.sidebar:
    st.subheader(f"🏢 Uzņēmums: **{S.active_profile}**")
    if st.button("🚪 Izrakstīties (Logout)", use_container_width=True):
        S.authenticated = False
        S.pop("active_profile", None)
        S.pop("result", None)
        S.pop("cls_df", None)
        st.rerun()

    st.markdown("---")
    st.subheader("📅 Finanšu gada iestatījumi")
    selected_fy_month = st.selectbox(
        "Finanšu gada sākuma mēnesis:",
        options=list(MONTH_NAMES_LV.keys()),
        format_func=lambda m: MONTH_NAMES_LV[m],
        index=list(MONTH_NAMES_LV.keys()).index(S.get("fy_start_month", 1)),
        help="Nosaka, no kura mēneša tiek rēķināti uzkrājošie (YTD) rādītāji S, TVC, T un NP.",
    )
    S.fy_start_month = selected_fy_month

    st.markdown("---")
    st.subheader("🗄️ Datubāzes statuss")
    if profile.is_connected:
        saved_cnt = profile.count_saved()
        if profile.last_error:
            st.warning(f"SQL kļūda: {profile.last_error}")
        else:
            st.success(f"Dati izolēti un drošībā\nSaglabāti konti: {saved_cnt}")
    else:
        st.error(f"Supabase kļūda: {profile.init_error}")

    # --- SAGLABĀTĀS FINANŠU TABULAS ---
    st.markdown("---")
    st.subheader("📂 Saglabātās finanšu tabulas")
    saved_reports = profile.list_reports()

    if not saved_reports:
        st.caption("Šim uzņēmumam vēl nav saglabātu finanšu tabulu.")
    else:
        report_Labels = {}
        for rep in saved_reports:
            dt_str = str(rep.get("created_at", ""))[:16].replace("T", " ")
            lbl = f"{rep.get('report_name')} ({dt_str})"
            report_Labels[lbl] = rep["id"]

        chosen_rep_lbl = st.selectbox("Izvēlieties atskaiti:", options=list(report_Labels.keys()))
        chosen_rep_id = report_Labels[chosen_rep_lbl]

        col_open, col_del = st.columns([3, 1])
        with col_open:
            if st.button("📖 Atvērt tabulu", type="primary", use_container_width=True):
                loaded = profile.load_report(chosen_rep_id)
                if loaded:
                    S.result = LoadedReportResult(
                        amounts=loaded["amounts"],
                        main=loaded["main"],
                        kpi=loaded["kpi"],
                        row_kinds=loaded["row_kinds"],
                        reconciliation=loaded["reconciliation"],
                        by_group={},
                    )
                    S.loaded_report_meta = loaded
                    S.current_report_id = loaded["id"]
                    if loaded.get("commentary"):
                        S.commentary = loaded["commentary"]
                    else:
                        S.pop("commentary", None)
                    st.rerun()
        with col_del:
            if st.button("🗑️", help="Dzēst šo saglabāto tabulu", use_container_width=True):
                if profile.delete_report(chosen_rep_id):
                    st.rerun()

    st.markdown("---")
    if st.button("🔄 Sākt jaunu aprēķinu", use_container_width=True):
        S.pop("result", None)
        S.pop("cls_df", None)
        S.pop("loaded_report_meta", None)
        st.rerun()


# --- GALVENAIS VIRSRAKSTS ---
st.title("📊 TOC-FIN: Finanšu diagnostika un vadības lēmumu sistēma")
st.markdown("""
**TOC-FIN** pārvērš grāmatvedības datus vadības lēmumiem piemērotā ekonomiskajā informācijā. 
Sistēma atbrīvo vadības komandu no mākslīgām pašizmaksas alokācijām, nodalot patiesi mainīgās izmaksas ($TVC$) 
no kapacitātes uzturēšanas izmaksām ($OE$), lai novērtētu patieso caurlaidību ($T$), produktivitāti ($PR = T / OE$) un modelētu nākotnes lēmumus.
""")


# ------------------------------------------------------------------ VISUR PIEEJAMAIS AI AĢENTS (JEBKURĀ POSMĀ)
def render_always_available_agent():
    with st.expander("💬 Saruna ar TOC-FIN aģentu un PZA nolasīšanas diagnostika (pieejams jebkurā solī)", expanded=bool(S.get("global_chat"))):
        st.caption(
            "Uzdodiet jebkuru jautājumu par nolasītajiem PZA kontiem (piem., *'Kāpēc konts X nav sarakstā vai nolasījās nepareizi?'*), "
            "klasifikāciju vai finanšu rādītājiem. Aģents redz visas faila rindas un aprēķinus."
        )
        if "global_chat" not in S:
            S.global_chat = []

        for msg in S.global_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        col_q, col_clr = st.columns([5, 1])
        with col_clr:
            if S.global_chat and st.button("🧹 Notīrīt čatu", use_container_width=True):
                S.global_chat = []
                st.rerun()

        q_input = st.chat_input("Rakstiet jautājumu aģentam šeit (piem., Kāpēc nepareizi nolasīja PZA kontu...)...", key="global_agent_input")
        if q_input:
            S.global_chat.append({"role": "user", "content": q_input})
            with st.spinner("Aģents pārbauda PZA struktūru un datus..."):
                ans = ask_global_assistant(q_input)
                S.global_chat.append({"role": "assistant", "content": ans})
            st.rerun()


def render_results_and_simulation(res, report_title: str = ""):
    if report_title:
        st.info(f"📂 **Atvērta atskaite:** {report_title}")

    main_tab1, main_tab2 = st.tabs(
        [
            "📊 1. TOC-FIN Analytics (Vēsturiskā diagnostika un Uzkrājošie rādītāji)",
            "⚡ 2. TOC-FIN Decision Support (Procentu % Simulācija pa periodiem)",
        ]
    )

    with main_tab1:
        try:
            xl_f, xl_v = io.BytesIO(), io.BytesIO()
            build_excel(res, xl_f, formulas=True)
            build_excel(res, xl_v, formulas=False)
            d1, d2 = st.columns(2)
            d1.download_button(
                "📥 Lejupielādēt Excel ar formulām (.xlsx)",
                xl_f.getvalue(),
                "TOC_Finansu_Diagnostika.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            d2.download_button(
                "📥 Lejupielādēt Excel ar vērtībām (.xlsx)",
                xl_v.getvalue(),
                "TOC_Finansu_Diagnostika_vertibas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            xl_buf = io.BytesIO()
            with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                res.amounts.to_excel(writer, sheet_name="TOC_Summas")
                res.kpi.to_excel(writer, sheet_name="KPI")
            st.download_button(
                "📥 Lejupielādēt saglabāto tabulu Excel (.xlsx)",
                xl_buf.getvalue(),
                "TOC_Saglabata_Atskaite.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("Finanšu pārskata tabula (T* skats)")
        st.markdown(
            render_unified_toc_tables_html(
                res.amounts,
                res.main,
                res.kpi,
                res.row_kinds,
                fy_start_month=S.get("fy_start_month", 1),
                show_cumulative=True,
            ),
            unsafe_allow_html=True,
        )

        if getattr(res, "reconciliation", None) is not None and not res.reconciliation.empty:
            with st.expander("🔍 Kontrole: Salīdzinājums ar grāmatvedības PZA rezultātu un izslēgtajām pozīcijām"):
                st.dataframe(res.reconciliation.style.format(fmt_num), use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Vadības kopsavilkums un stratēģiskie secinājumi")
        cum_df_md, _, _ = compute_cumulative_toc_df(res.amounts, fy_start_month=S.get("fy_start_month", 1))
        tables_md = (
            res.amounts.round(0).to_markdown()
            + "\n\n"
            + res.kpi.round(3).to_markdown()
            + "\n\nUzkrājošie rādītāji (S, TVC, T, NP):\n"
            + cum_df_md.round(0).to_markdown()
        )

        knowledge_context = f"TOC-FIN zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
        COMMENT_SYSTEM = (
            "Tu esi pieredzējis TOC-FIN finanšu direktora padomnieks un vadības konsultants. "
            "Uzrunā lietotāju kā Vadības komandu. Nekad nelieto grāmatvedības pašizmaksas klišejas. "
            f"{knowledge_context}"
            "Raksti latviešu valodā, strukturēti, asi un uz lēmumiem vērsti."
        )

        if "commentary" not in S or not S.commentary:
            if st.button("📝 Ģenerēt vadības analīzi un rīcības plānu", type="primary"):
                with st.spinner("Sagatavo TOC-FIN vadības kopsavilkumu..."):
                    r = get_client().messages.create(
                        model=MODEL,
                        max_tokens=2500,
                        system=COMMENT_SYSTEM,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Aprēķinātās tabulas (summas EUR):\n\n{tables_md}\n\nUzraksti vadības analīzi.",
                            }
                        ],
                    )
                    S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    if S.get("current_report_id"):
                        profile.update_report_commentary(S.current_report_id, S.commentary)
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
            preset = st.selectbox(
                "Ātrā perioda izvēle:",
                options=[
                    f"Visi mēneši ({len(all_months)} mēn.)",
                    "Pirmie 6 mēneši" if len(all_months) >= 6 else "Pirmie 3",
                    "Manuāla izvēle",
                ],
            )
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

        pct_changes: dict[str, float] = {}
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("🟡 **Ieņēmumi (S apakšgrupas)**")
            for sc_name in s_children:
                pct_changes[sc_name] = st.number_input(
                    f"{sc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{sc_name}"
                )
        with sc2:
            st.markdown("🟠 **Mainīgās (TVC)**")
            for tc_name in tvc_children:
                pct_changes[tc_name] = st.number_input(
                    f"{tc_name} (%)", value=10.0, step=1.0, format="%.1f", key=f"sim_{tc_name}"
                )
        with sc3:
            st.markdown("🟤 **Darbības izdevumi (OE)**")
            for oc_name in oe_children:
                def_v = 5.0 if "alga" in oc_name.lower() else 0.0
                pct_changes[oc_name] = st.number_input(
                    f"{oc_name} (%)", value=def_v, step=1.0, format="%.1f", key=f"sim_{oc_name}"
                )

        base_sub = res.amounts[selected_months].copy().astype(float)
        base_sum = base_sub.sum(axis=1)
        sim_sum = base_sum.copy()

        for rname, pct_val in pct_changes.items():
            if rname in sim_sum.index:
                sim_sum.loc[rname] = base_sum.loc[rname] * (1.0 + pct_val / 100.0)

        if s_children:
            sim_sum.loc["S"] = sim_sum.loc[s_children].sum()
        if tvc_children:
            sim_sum.loc["TVC"] = sim_sum.loc[tvc_children].sum()
        sim_sum.loc["T"] = sim_sum.loc["S"] - sim_sum.loc["TVC"]
        if oe_children:
            sim_sum.loc["OE"] = sim_sum.loc[oe_children].sum()
        sim_sum.loc["NP"] = sim_sum.loc["T"] - sim_sum.loc["OE"]

        diff_sum = sim_sum - base_sum

        summary_amounts = pd.DataFrame(
            {
                "Bāze KOPĀ": base_sum,
                "Simulācija KOPĀ": sim_sum,
                "Δ Izmaiņa (€)": diff_sum,
            },
            index=res.amounts.index,
        )

        sim_main = {}
        for c in summary_amounts.columns:
            sim_main[c] = summary_amounts[c]
            if c == "Δ Izmaiņa (€)":
                sim_main[f"%|{c}"] = np.where(base_sum != 0, diff_sum / np.abs(base_sum), np.nan)
            else:
                s_col = summary_amounts.loc["S", c]
                sim_main[f"%|{c}"] = summary_amounts[c] / s_col if s_col else np.nan
        sim_main_df = pd.DataFrame(sim_main, index=summary_amounts.index)

        def sim_kpi_for(col_vals: pd.Series) -> dict[str, float]:
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

        sim_kpi_df = pd.DataFrame(
            {
                "Bāze KOPĀ": sim_kpi_for(summary_amounts["Bāze KOPĀ"]),
                "Simulācija KOPĀ": sim_kpi_for(summary_amounts["Simulācija KOPĀ"]),
            }
        )
        sim_kpi_df["Δ Izmaiņa (€)"] = sim_kpi_df["Simulācija KOPĀ"] - sim_kpi_df["Bāze KOPĀ"]

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        ds_val, dt_val, doe_val, dnp_val = diff_sum["S"], diff_sum["T"], diff_sum["OE"], diff_sum["NP"]

        m1.metric("Simulētais S", f"{fmt_num(sim_sum['S'])} €", delta=f"{ds_val:+,.0f} €".replace(",", " "))
        m2.metric("Simulētais T", f"{fmt_num(sim_sum['T'])} €", delta=f"{dt_val:+,.0f} €".replace(",", " "))
        m3.metric(
            "Simulētais OE",
            f"{fmt_num(sim_sum['OE'])} €",
            delta=f"{doe_val:+,.0f} €".replace(",", " "),
            delta_color="inverse",
        )
        m4.metric("Simulētā Neto Peļņa (NP)", f"{fmt_num(sim_sum['NP'])} €", delta=f"{dnp_val:+,.0f} €".replace(",", " "))

        st.subheader(f"📊 Simulētā Finanšu un Produktivitātes tabula ({len(selected_months)} mēn. kopsavilkums)")
        st.markdown(
            render_unified_toc_tables_html(
                summary_amounts,
                sim_main_df,
                sim_kpi_df,
                res.row_kinds,
                title_kpi="📈 Simulētie Produktivitātes un atdeves rādītāji",
                grey_cols={"Bāze KOPĀ"},
                fy_start_month=S.get("fy_start_month", 1),
                show_cumulative=False,
            ),
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ GALVENAIS DARBA EKRĀNS
uploaded = st.file_uploader("Augšupielādējiet PZA failu (CSV vai Excel formātā)", type=["csv", "xlsx", "xlsm", "xls"])

if uploaded is None:
    render_always_available_agent()
    if "result" in S and S.get("loaded_report_meta"):
        meta = S.loaded_report_meta
        title_str = f"{meta.get('report_name')} ({meta.get('periods_label')})"
        render_results_and_simulation(S.result, report_title=title_str)
    else:
        st.info(f"Sveicināti, **{S.active_profile}**! Augšupielādējiet PZA failu vai kreisajā sānjoslā izvēlieties iepriekš saglabātu atskaiti.")
    st.stop()

data = uploaded.getvalue()
fid = hashlib.md5(data).hexdigest()
if S.get("fid") != fid:
    S.fid = fid
    S.grids = load_grids(uploaded.name, data)
    S.pop("structure", None)
    S.pop("analysis", None)
    S.pop("cls_df", None)
    S.pop("result", None)
    S.pop("manual_leaf_overrides", None)

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
    base_analysis = analyze(ex, stc)
    # Pielietojam manuālos labojumus, ja lietotājs ir iekļāvis vai izslēdzis kādu PZA rindu
    overrides: dict[str, bool] = S.get("manual_leaf_overrides", {})
    if overrides:
        base_leaf_ids = {lf.id for lf in base_analysis.leaves}
        new_leaves = []
        for ln in base_analysis.ex.lines:
            if stc.result_row is not None and ln.row == stc.result_row:
                continue
            is_leaf = overrides.get(ln.id, ln.id in base_leaf_ids)
            if is_leaf:
                new_leaves.append(ln)
        base_analysis.leaves = new_leaves
    S.analysis = base_analysis
except Exception as e:
    st.error(f"Kļūda datu ielasīšanā: {e}")
    st.stop()

an = S.analysis
display_labels = build_display_labels(an.leaves)

# --- VISUR PIEEJAMAIS AI AĢENTS UN PZA STRUKTŪRAS DIAGNOSTIKA ---
render_always_available_agent()

with st.expander("🛠️ PZA struktūras diagnostika un manuāla rindu korekcija (ja kāds konts nav nolasīts pareizi)"):
    st.markdown(
        "Šeit redzamas **visas no faila nolasītās rindas**. Ja sistēma kādu kontu kļūdaini uzskatījusi par virsrakstu/kopsummu "
        "(vai otrādi — iekļāvusi starpsummu kā kontu), ieķeksējiet vai izņemiet ķeksi kolonnā **'Iekļaut kā kontu'** un nospiediet pogu zemāk."
    )
    leaf_id_set = {lf.id for lf in an.leaves}
    diag_rows = []
    for ln in an.ex.lines:
        is_res = stc.result_row is not None and ln.row == stc.result_row
        tot_v = float(ln.vals.sum()) if hasattr(ln, "vals") else 0.0
        first_v = float(ln.vals.iloc[0]) if (hasattr(ln, "vals") and len(ln.vals) > 0) else 0.0
        diag_rows.append(
            {
                "id": ln.id,
                "Rinda failā": ln.row,
                "Iekļaut kā kontu": (ln.id in leaf_id_set) and not is_res,
                "Nosaukums PZA failā": ln.label,
                "Pilns ceļš / Sadaļa": ln.path,
                "1. perioda summa": round(first_v, 2),
                "Summa visos periodos": round(tot_v, 2),
                "Statuss": "🏁 PZA Rezultāta rinda" if is_res else ("✅ Konts" if ln.id in leaf_id_set else "⏭️ Izlaists (Starpsumma / Virsraksts)"),
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    edited_diag = st.data_editor(
        diag_df,
        hide_index=True,
        use_container_width=True,
        disabled=["id", "Rinda failā", "Nosaukums PZA failā", "Pilns ceļš / Sadaļa", "1. perioda summa", "Summa visos periodos", "Statuss"],
        key="pza_diag_editor",
    )
    if st.button("✅ Saglabāt PZA rindu izmaiņas un pārlasīt kontus"):
        new_overrides = {}
        for _, r_d in edited_diag.iterrows():
            new_overrides[r_d["id"]] = bool(r_d["Iekļaut kā kontu"])
        S.manual_leaf_overrides = new_overrides
        S.pop("cls_df", None)
        S.pop("result", None)
        st.rerun()

st.header(f"1. Kontu ekonomiskā klasifikācija ({S.active_profile})")

if "cls_df" not in S:
    if st.button("🔍 Sākt kontu TOC-FIN klasifikāciju", type="primary"):
        with st.spinner(f"Salīdzina ar uzņēmuma '{S.active_profile}' iepriekšējo atmiņu un klasificē..."):
            try:
                known, unknown = profile.lookup(an.leaves)
                items = classify_with_claude(get_client(), MODEL, unknown) if unknown else []
                by_id = {i["id"]: i for i in items}
                rows = []
                for lf in an.leaves:
                    disp_lbl = display_labels.get(lf.id, lf.label)
                    if lf.id in known:
                        m = known[lf.id]
                        grp = normalize_group_key(m["group"])
                        rows.append(
                            dict(
                                id=lf.id,
                                Konts=lf.label,
                                Displeja_konts=disp_lbl,
                                Grupa=grp,
                                Ieņēmums=bool(grp.startswith("S_") or m.get("is_income", False)),
                                Jautājums="",
                                Avots="🗄️ Uzņēmuma atmiņa",
                            )
                        )
                    else:
                        i = by_id.get(lf.id, {"group": "OE_OTHER", "is_income": False, "question": ""})
                        grp = normalize_group_key(i.get("group", "OE_OTHER"))
                        rows.append(
                            dict(
                                id=lf.id,
                                Konts=lf.label,
                                Displeja_konts=disp_lbl,
                                Grupa=grp,
                                Ieņēmums=bool(grp.startswith("S_") or i.get("is_income", False)),
                                Jautājums=str(i.get("question") or "").strip(),
                                Avots="🤖 AI klasifikācija",
                            )
                        )
                S.cls_df = sort_cls_df(pd.DataFrame(rows))
                S.editor_v = 0
                S.known_count = len(known)
                S.unknown_count = len(unknown)
                st.rerun()
            except Exception as e:
                st.error(f"Kļūda klasifikācijā: {e}")
    if "cls_df" not in S:
        st.stop()

cls_df: pd.DataFrame = S.cls_df.copy()
cls_df["Grupa"] = cls_df["Grupa"].map(normalize_group_key)
if "Displeja_konts" not in cls_df.columns:
    cls_df["Displeja_konts"] = cls_df["id"].map(lambda x: display_labels.get(x, ""))

if S.get("known_count", 0) > 0 or S.get("unknown_count", 0) > 0:
    st.caption(
        f"📌 No uzņēmuma atmiņas atpazīti **{S.get('known_count', 0)}** konti | "
        f"Ar AI no jauna klasificēti **{S.get('unknown_count', 0)}** konti."
    )

ambiguous_rows = cls_df[cls_df["Jautājums"].astype(str).str.strip() != ""]

if len(ambiguous_rows) > 0 and not S.get("clarifications_done", False):
    st.subheader("❓ Nepieciešams vadības lēmums par dažām pozīcijām")
    clarifications: dict[str, tuple[str, bool]] = {}
    for _, r in ambiguous_rows.iterrows():
        row_id = r["id"]
        disp_name = r["Displeja_konts"]
        agent_grp = normalize_group_key(r["Grupa"])
        agent_sub_label = SUBGROUP_LABELS.get(agent_grp, agent_grp)
        default_idx = get_radio_index_for_group(agent_grp)

        st.markdown(f"**📌 {disp_name}** *(Hipotēze: `{agent_sub_label}`)*")
        st.write(f"👉 **Jautājums:** {r['Jautājums']}")

        choice = st.radio(
            f"Kā klasificēt '{disp_name}'?",
            options=RADIO_OPTIONS,
            index=default_idx,
            key=f"radio_{row_id}",
        )
        clarifications[row_id] = resolve_radio_choice(choice, agent_grp, r["Konts"])
        st.markdown("---")

    if st.button("✅ Apstiprināt atbildes un pāriet uz kontu tabulu", type="primary"):
        for row_id, (new_group, is_inc) in clarifications.items():
            mask = cls_df["id"] == row_id
            cls_df.loc[mask, "Grupa"] = new_group
            cls_df.loc[mask, "Ieņēmums"] = is_inc
            cls_df.loc[mask, "Jautājums"] = ""
        S.cls_df = sort_cls_df(cls_df)
        S.clarifications_done = True
        S.editor_v = S.get("editor_v", 0) + 1
        st.rerun()
    st.stop()

hdr_col1, hdr_col2 = st.columns([3, 1])
with hdr_col1:
    st.subheader("📋 Kontu sadalījums pa grupām un apakšgrupām")
with hdr_col2:
    if st.button("🔃 Sakārtot tabulu", use_container_width=True):
        S.cls_df = sort_cls_df(cls_df)
        S.editor_v = S.get("editor_v", 0) + 1
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🟡 S (Ieņēmumi)", f"{cls_df['Grupa'].str.startswith('S_').sum()} konti")
c2.metric("🟠 TVC (Mainīgās)", f"{cls_df['Grupa'].str.startswith('TVC').sum()} konti")
c3.metric("🟤 OE (Uzturēšana)", f"{cls_df['Grupa'].str.startswith('OE_').sum()} konti")
c4.metric("⚪ Izslēgtie", f"{(cls_df['Grupa'] == 'EXCLUDED').sum()} konti")

view_df = pd.DataFrame(
    {
        "Konts": cls_df["Displeja_konts"].tolist(),
        "Lielā grupa": [get_main_cat(g) for g in cls_df["Grupa"]],
        "Apakšgrupa (Rinda atskaitē)": [SUBGROUP_LABELS[normalize_group_key(g)] for g in cls_df["Grupa"]],
    }
)

full_editor_height = (len(view_df) + 1) * 36 + 14
editor_key = f"clean_cls_editor_v_{S.get('editor_v', 0)}"

edited_view = st.data_editor(
    view_df,
    hide_index=True,
    use_container_width=True,
    height=full_editor_height,
    key=editor_key,
    disabled=["Konts"],
    column_config={
        "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
        "Lielā grupa": st.column_config.SelectboxColumn(
            "Lielā grupa",
            options=list(MAIN_GROUP_TO_DEFAULT_SUB.keys()),
            required=True,
            width="medium",
        ),
        "Apakšgrupa (Rinda atskaitē)": st.column_config.SelectboxColumn(
            "Apakšgrupa / Rinda atskaitē",
            options=list(SUBGROUP_LABELS.values()),
            required=True,
            width="large",
        ),
    },
)

updated_groups: list[str] = []
changed = False

for pos_idx in range(len(cls_df)):
    old_grp = normalize_group_key(cls_df.iloc[pos_idx]["Grupa"])
    old_main = get_main_cat(old_grp)
    old_sub = SUBGROUP_LABELS[old_grp]

    new_main = edited_view.iloc[pos_idx]["Lielā grupa"]
    new_sub = edited_view.iloc[pos_idx]["Apakšgrupa (Rinda atskaitē)"]

    if new_sub != old_sub and new_sub in LABEL_TO_KEY:
        new_grp = LABEL_TO_KEY[new_sub]
        updated_groups.append(new_grp)
        changed = True
    elif new_main != old_main and new_main in MAIN_GROUP_TO_DEFAULT_SUB:
        new_grp = MAIN_GROUP_TO_DEFAULT_SUB[new_main]
        updated_groups.append(new_grp)
        changed = True
    else:
        updated_groups.append(old_grp)

if changed:
    cls_df["Grupa"] = updated_groups
    cls_df["Ieņēmums"] = cls_df["Grupa"].str.startswith("S_")
    S.cls_df = cls_df
    S.editor_v = S.get("editor_v", 0) + 1
    st.rerun()

st.header("2. TOC-FIN Diagnostika un Lēmumu Atbalsts")

default_rep_name = f"{Path(uploaded.name).stem} ({datetime.now().strftime('%d.%m.%Y')})"
report_custom_name = st.text_input("Atskaites nosaukums saglabāšanai arhīvā:", value=default_rep_name)

if st.button("🚀 Apstiprināt klasifikāciju, veikt aprēķinu un saglabāt", type="primary"):
    cls_sorted = sort_cls_df(cls_df)
    S.cls_df = cls_sorted
    S.editor_v = S.get("editor_v", 0) + 1

    mapping = {
        r["id"]: {
            "group": normalize_group_key(r["Grupa"]),
            "is_income": bool(str(r["Grupa"]).startswith("S_") or r["Ieņēmums"]),
        }
        for _, r in cls_sorted.iterrows()
    }
    pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
    reported = an.ex.lines[pos].vals if pos is not None else None

    res = compute_toc(an.leaves, mapping, an.ex.period_cols, reported)
    S.result = res
    S.pop("loaded_report_meta", None)

    if profile.is_connected:
        ok_map, cnt_map = profile.save(an.leaves, mapping)
        ok_rep, rep_id = profile.save_report(
            report_name=report_custom_name.strip() or default_rep_name,
            file_name=uploaded.name,
            res_obj=res,
            commentary="",
        )
        S.current_report_id = rep_id
        if ok_map and ok_rep:
            st.toast(f"✅ Dati saglabāti uzņēmuma '{S.active_profile}' drošajā arhīvā!", icon="🗄️")
        elif not ok_rep:
            st.warning(f"Kļūda saglabājot tabulu: {profile.last_error}")

    S.pop("commentary", None)
    st.rerun()

if "result" in S:
    render_results_and_simulation(S.result)

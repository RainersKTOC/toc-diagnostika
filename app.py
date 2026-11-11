"""
app.py — TOC-FIN Finanšu diagnostikas un vadības lēmumu sistēma (v25.5)
Simulācijas kopsummas virs apakškontiem | Interaktīvs TOC Eksperts ar darbību izpildi | Ērts čata logs
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import io
import json
import os
import re
import urllib.request
from pathlib import Path

import altair as alt
import anthropic
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

st.set_page_config(
    page_title="TOC-FIN Vadības Sistēma",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Pievienojam krāsu kodus T apakšrindām un pus-bold starpsummu rindām simulācijā
if "T_CHILD" not in FILLS:
    FILLS["T_CHILD"] = ("DAEEF3", False, "000000")
if "T_SUBTOTAL" not in FILLS:
    FILLS["T_SUBTOTAL"] = ("CBE4EE", False, "000000")
if "S_SUBTOTAL" not in FILLS:
    FILLS["S_SUBTOTAL"] = ("FFF9BF", False, "000000")
if "TVC_SUBTOTAL" not in FILLS:
    FILLS["TVC_SUBTOTAL"] = ("F7D6BF", False, "000000")

EXTRA_OE_SUBGROUPS = {
    "OE_ADMIN_OTHER": ("Pārējās administrācijas izmaksas", "OE"),
    "OE_SALES_OTHER": ("Pārējās pārdošanas izmaksas", "OE"),
    "OE_OPER_OTHER": ("Pārējās saimnieciskās darbības izmaksas", "OE"),
}
for k_ext, (lbl_ext, mc_ext) in EXTRA_OE_SUBGROUPS.items():
    if k_ext not in GROUPS:
        GROUPS[k_ext] = (lbl_ext, mc_ext)
    if k_ext not in SUBGROUP_LABELS:
        SUBGROUP_LABELS[k_ext] = lbl_ext
    LABEL_TO_KEY[lbl_ext] = k_ext


def register_custom_subgroup(key: str, label: str, main_code: str = "OE") -> None:
    GROUPS[key] = (label, main_code)
    SUBGROUP_LABELS[key] = label
    LABEL_TO_KEY[label] = key


def get_subgroups_for_main_code(main_code: str) -> list[str]:
    res = []
    for k, (lbl, mc) in GROUPS.items():
        if mc == main_code:
            res.append(lbl)
    return res


@dataclass
class LoadedReportResult:
    amounts: pd.DataFrame
    main: pd.DataFrame
    kpi: pd.DataFrame
    row_kinds: dict[str, str]
    reconciliation: pd.DataFrame
    by_group: dict


def _clean_api_key_str(raw_val: str) -> str:
    s = str(raw_val or "").strip().strip('"').strip("'")
    return re.sub(r"\s+", "", s)


def _load_anthropic_key() -> str:
    for k, v in os.environ.items():
        if isinstance(v, str):
            cl = _clean_api_key_str(v)
            if cl.startswith("sk-ant-"):
                return cl
            if k.upper() in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY") and cl:
                return cl

    def _scan_dict(d) -> str:
        try:
            items = d.items()
        except Exception:
            return ""
        for k, v in items:
            if isinstance(v, str):
                clean = _clean_api_key_str(v)
                if clean.startswith("sk-ant-"):
                    return clean
                if str(k).upper() in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "API_KEY") and not clean.startswith("eyJ") and not clean.startswith("http"):
                    return clean
            elif hasattr(v, "items"):
                found = _scan_dict(v)
                if found:
                    return found
        return ""

    try:
        res = _scan_dict(st.secrets)
        if res:
            return res
    except Exception:
        pass
    return ""


API_KEY = _load_anthropic_key()
if not API_KEY:
    st.error(
        "Nav atrasta ANTHROPIC_API_KEY. Pārbaudiet Streamlit Cloud -> Manage app -> Settings -> Secrets un pievienojiet:\n\n"
        'ANTHROPIC_API_KEY = "sk-ant-..."'
    )
    st.stop()


def _make_auth_token(company: str, is_admin: bool) -> str:
    adm_flag = "1" if is_admin else "0"
    payload = f"{company.strip().lower()}|{adm_flag}"
    sig = hmac.new(API_KEY.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:20]
    return f"{adm_flag}.{sig}"


def _verify_auth_token(company: str, token: str) -> tuple[bool, bool]:
    if not company or not token or "." not in token:
        return False, False
    adm_flag, _ = token.split(".", 1)
    is_adm = adm_flag == "1"
    expected = _make_auth_token(company, is_adm)
    if hmac.compare_digest(token, expected):
        return True, is_adm
    return False, False


def get_client() -> anthropic.Anthropic:
    active_key = _load_anthropic_key() or API_KEY
    return anthropic.Anthropic(api_key=active_key)


def transcribe_wav_bytes_lv(audio_bytes: bytes) -> tuple[str, str]:
    if not audio_bytes:
        return "", "Tukšs audio ieraksts."
    try:
        import speech_recognition as sr  # type: ignore
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        txt = str(recognizer.recognize_google(audio_data, language="lv-LV")).strip()
        if txt:
            return txt, ""
    except Exception as e:
        return "", f"Lai balss ieraksts darbotos uz servera, pievienojiet rindiņu `SpeechRecognition` savā GitHub failā `requirements.txt` ({e})."
    return "", "Neizdevās sadzirdēt skaidru tekstu ierakstā."


ACCOUNT_DETAILS_MARKER = "\n\n<!--TOC_ACC_DETAILS:"


def pack_commentary_with_account_details(commentary_text: str, acc_details: list[dict] | None) -> str:
    clean_comm = str(commentary_text or "").split(ACCOUNT_DETAILS_MARKER)[0].strip()
    if not acc_details:
        return clean_comm
    try:
        payload = json.dumps(acc_details, ensure_ascii=False)
        return f"{clean_comm}{ACCOUNT_DETAILS_MARKER}{payload}-->"
    except Exception:
        return clean_comm


def unpack_commentary_and_account_details(raw_commentary: str | None) -> tuple[str, list[dict]]:
    s = str(raw_commentary or "")
    if ACCOUNT_DETAILS_MARKER not in s:
        return s.strip(), []
    comm_part, rest = s.split(ACCOUNT_DETAILS_MARKER, 1)
    json_part = rest.rsplit("-->", 1)[0].strip()
    try:
        details = json.loads(json_part)
        if isinstance(details, list):
            return comm_part.strip(), details
    except Exception:
        pass
    return comm_part.strip(), []


def build_account_details_snapshot(leaves, cls_df: pd.DataFrame, period_cols: list[dict]) -> list[dict]:
    leaf_by_id = {lf.id: lf for lf in leaves}
    p_labels = [p["label"] for p in period_cols]
    snapshot = []
    for _, row in cls_df.iterrows():
        lf = leaf_by_id.get(row["id"])
        if lf is None:
            continue
        grp = normalize_group_key(row["Grupa"])
        m_vals = {p_labels[i]: round(float(lf.vals[i]), 2) for i in range(min(len(p_labels), len(lf.vals)))}
        snapshot.append(
            {
                "id": str(row["id"]),
                "row": int(lf.row + 1),
                "label": str(row["Konts"]),
                "display_label": str(row.get("Displeja_konts", row["Konts"])),
                "path": str(row.get("Sadaļa", "") or lf.path or ""),
                "group": grp,
                "months": m_vals,
                "total": round(float(np.sum(lf.vals)), 2),
            }
        )
    return snapshot


def fmt_num(x) -> str:
    return "" if pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def fmt_pct_int(x) -> str:
    return "" if pd.isna(x) else f"{x * 100:.0f}%"


def _safe_line_stats(vals) -> tuple[float, float]:
    if vals is None:
        return 0.0, 0.0
    try:
        if isinstance(vals, dict):
            nums = [float(v) for v in vals.values() if v is not None and not pd.isna(v)]
        elif hasattr(vals, "values") and not isinstance(vals, (list, tuple, np.ndarray)):
            arr = np.asarray(vals.values, dtype=float)
            nums = [float(x) for x in arr if not np.isnan(x)]
        else:
            arr = np.asarray(vals, dtype=float).flatten()
            nums = [float(x) for x in arr if not np.isnan(x)]
        if not nums:
            return 0.0, 0.0
        return float(nums[0]), float(sum(nums))
    except Exception:
        return 0.0, 0.0


MONTH_NAMES_LV = {
    1: "Janvāris",
    2: "Februāris",
    3: "Marts",
    4: "Aprīlis",
    5: "Maijs",
    6: "Jūnijs",
    7: "Jūlijs",
    8: "Augusts",
    9: "Septembris",
    10: "Oktobris",
    11: "Novembris",
    12: "Decembris",
}

LV_MONTH_PARSE = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mai": 5, "may": 5,
    "jūn": 6, "jun": 6, "jūl": 7, "jul": 7, "aug": 8, "sep": 9,
    "okt": 10, "oct": 10, "nov": 11, "dec": 12,
}


def parse_period_month_year(col_name: str) -> tuple[int | None, int | None]:
    s = str(col_name or "").strip().lower()
    if "ytd" in s or "avg" in s or "kopā" in s or "izmaiņa" in s:
        return None, None

    m1 = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[\./\-](20\d{2})(?!\d)", s)
    if m1:
        return int(m1.group(1)), int(m1.group(2))

    m2 = re.search(r"(?<!\d)(20\d{2})[\./\-](0?[1-9]|1[0-2])(?!\d)", s)
    if m2:
        return int(m2.group(2)), int(m2.group(1))

    yr_m = re.search(r"(20\d{2})", s)
    yr = int(yr_m.group(1)) if yr_m else None
    for k, m_num in LV_MONTH_PARSE.items():
        if k in s:
            return m_num, yr

    return None, None


def strip_excluded_rows(res):
    excl_label = GROUPS.get("EXCLUDED", ("Izslēgtie (Amortizācija / Virtuālā nauda)", "EXCL"))[0]
    drop_rows = [
        r for r in res.amounts.index
        if res.row_kinds.get(r) in ("EXCL", "EXCLUDED")
        or r == excl_label
        or str(r).lower().startswith("izslēgt")
    ]
    if drop_rows:
        res.amounts = res.amounts.drop(index=drop_rows, errors="ignore")
        res.main = res.main.drop(index=drop_rows, errors="ignore")
        for dr in drop_rows:
            res.row_kinds.pop(dr, None)
    return res


def filter_result_by_months(res, selected_months: list[str], all_months: list[str]):
    if not selected_months or len(selected_months) == len(all_months):
        return res

    sub_amounts = res.amounts[selected_months].copy().astype(float)
    years_found = []
    for m_col in selected_months:
        _, yr = parse_period_month_year(m_col)
        if yr and yr not in years_found:
            years_found.append(yr)
    yr_prefix = f"{years_found[-1]} " if len(years_found) == 1 else ""
    ytd_col = f"{yr_prefix}YTD".strip()
    avg_col = f"{yr_prefix}AVG YTD".strip()

    sub_amounts[ytd_col] = sub_amounts[selected_months].sum(axis=1)
    sub_amounts[avg_col] = sub_amounts[selected_months].mean(axis=1)

    sub_main_dict = {}
    for c in sub_amounts.columns:
        sub_main_dict[c] = sub_amounts[c]
        s_val = sub_amounts.loc["S", c] if "S" in sub_amounts.index else 0.0
        sub_main_dict[f"%|{c}"] = np.where(s_val != 0, sub_amounts[c] / s_val, np.nan)
    sub_main = pd.DataFrame(sub_main_dict, index=sub_amounts.index)

    def _calc_kpi_col(col_s: pd.Series) -> dict[str, float]:
        S_ = col_s.get("S", 0.0)
        T_ = col_s.get("T", 0.0)
        OE_ = col_s.get("OE", 0.0)
        aw = col_s.get(GROUPS["OE_ADMIN_WAGES"][0], 0.0)
        ww = col_s.get(GROUPS["OE_WORKER_WAGES"][0], 0.0)
        div = lambda a, b: a / b if b else np.nan
        return {
            "PR = T/OE": div(T_, OE_),
            "T% = T/S": div(T_, S_),
            "Eiro Sales / Eiro algas kopā": div(S_, aw + ww),
            "Eiro Sales / Eiro administrācijas algas": div(S_, aw),
            "Eiro Sales / Eiro strādnieku algas": div(S_, ww),
        }

    kpi_cols_dict = {c: _calc_kpi_col(sub_amounts[c]) for c in sub_amounts.columns}
    sub_kpi = pd.DataFrame(kpi_cols_dict)

    return LoadedReportResult(
        amounts=sub_amounts,
        main=sub_main,
        kpi=sub_kpi,
        row_kinds=dict(res.row_kinds),
        reconciliation=getattr(res, "reconciliation", pd.DataFrame()),
        by_group=getattr(res, "by_group", {}),
    )


def compute_cumulative_toc_df(amounts_df: pd.DataFrame, fy_start_month: int = 1) -> tuple[pd.DataFrame, list[str], set[str]]:
    target_rows = [r for r in ["S", "TVC", "T", "OE", "NP"] if r in amounts_df.index]
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
                running = amounts_df.loc[target_rows, col].fillna(0.0).astype(float).copy()
                fy_boundary_cols.add(col)
            else:
                running = running + amounts_df.loc[target_rows, col].fillna(0.0).astype(float)
        cum_data[col] = running.copy()

    cum_df = pd.DataFrame(cum_data, index=target_rows)
    table_rows = [r for r in ["S", "TVC", "T", "NP"] if r in cum_df.index]
    cum_table_df = cum_df.loc[table_rows].copy()
    row_labels = {
        "S": "S (uzkrājošais)",
        "TVC": "TVC (uzkrājošais)",
        "T": "T (uzkrājošais)",
        "NP": "NP (uzkrājošais)",
    }
    cum_table_df.index = [row_labels.get(r, r) for r in cum_table_df.index]
    return cum_table_df, period_cols, fy_boundary_cols


def _compute_linear_trend(y_vals: list[float]) -> list[float]:
    n = len(y_vals)
    if n < 2:
        return y_vals
    x = np.arange(n, dtype=float)
    y = np.array([0.0 if pd.isna(v) else float(v) for v in y_vals], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return list(slope * x + intercept)


def render_overview_charts(amounts_df: pd.DataFrame, kpi_df: pd.DataFrame, fy_start_month: int = 1) -> None:
    period_cols = [
        c for c in amounts_df.columns
        if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper() or "IZMAIŅA" in str(c).upper())
    ]
    if not period_cols:
        return

    st.markdown("---")
    st.subheader("Finanšu un produktivitātes dinamika")

    with st.container(border=True):
        ch_top1, ch_top2 = st.columns([2.2, 1], vertical_alignment="center")
        with ch_top1:
            st.markdown("##### Galveno rādītāju dinamika (S, TVC, T, Uzturēšanas izmaksas, NP)")
        with ch_top2:
            chart_mode = st.radio(
                "Grafika skats",
                options=["Ikmēneša rādītāji", "Uzkrājošie rādītāji"],
                horizontal=True,
                label_visibility="collapsed",
                key="toc_chart_mode_radio",
            )

        series_order = ["S", "TVC", "T", "Uzturēšanas izmaksas (OE)", "NP"]
        color_scale = alt.Scale(
            domain=series_order,
            range=["#EAB308", "#C55A11", "#00B0F0", "#F4B183", "#70AD47"],
        )

        raw_rows = [r for r in ["S", "TVC", "T", "OE", "NP"] if r in amounts_df.index]
        if chart_mode == "Ikmēneša rādītāji":
            plot_df = amounts_df.loc[raw_rows, period_cols].copy().astype(float)
        else:
            plot_df = pd.DataFrame(index=raw_rows, columns=period_cols, dtype=float)
            running = pd.Series(0.0, index=raw_rows)
            for idx, col in enumerate(period_cols):
                m_num, _ = parse_period_month_year(str(col))
                if idx == 0 or (m_num is not None and m_num == fy_start_month):
                    running = amounts_df.loc[raw_rows, col].fillna(0.0).astype(float).copy()
                else:
                    running = running + amounts_df.loc[raw_rows, col].fillna(0.0).astype(float)
                plot_df[col] = running

        rename_map = {"OE": "Uzturēšanas izmaksas (OE)"}
        plot_df.index = [rename_map.get(r, r) for r in plot_df.index]

        records = []
        for r_name in plot_df.index:
            for idx_p, p_col in enumerate(period_cols):
                val = float(plot_df.loc[r_name, p_col])
                records.append(
                    {
                        "Periods": str(p_col),
                        "Secība": idx_p,
                        "Rādītājs": r_name,
                        "Summa (€)": round(val, 0),
                    }
                )
        long_df = pd.DataFrame(records)

        base = alt.Chart(long_df).encode(
            x=alt.X("Periods:N", sort=period_cols, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Summa (€):Q", title="Summa (€)", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "Rādītājs:N",
                scale=color_scale,
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Periods:N", title="Periods"),
                alt.Tooltip("Rādītājs:N", title="Rādītājs"),
                alt.Tooltip("Summa (€):Q", title="Summa (€)", format=",.0f"),
            ],
        )

        lines = base.mark_line(strokeWidth=2.8)
        points = base.mark_circle(size=65)
        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="#888888").encode(y="y:Q")
        st.altair_chart((zero_rule + lines + points).properties(height=340).interactive(), use_container_width=True)

    kpi_m_cols = [c for c in period_cols if c in kpi_df.columns]
    if not kpi_m_cols:
        return

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        with st.container(border=True):
            st.markdown("##### Produktivitāte (`PR = T/OE` un `T% = T/S`)")
            pr_vals = [float(kpi_df.loc["PR = T/OE", c]) if "PR = T/OE" in kpi_df.index and not pd.isna(kpi_df.loc["PR = T/OE", c]) else 0.0 for c in kpi_m_cols]
            tp_vals = [float(kpi_df.loc["T% = T/S", c]) if "T% = T/S" in kpi_df.index and not pd.isna(kpi_df.loc["T% = T/S", c]) else 0.0 for c in kpi_m_cols]
            pr_trend = _compute_linear_trend(pr_vals)
            tp_trend = _compute_linear_trend(tp_vals)

            prod_rows = []
            for idx_m, m_lbl in enumerate(kpi_m_cols):
                prod_rows.append({"Periods": m_lbl, "Rādītājs": "PR = T/OE", "Vērtība": round(pr_vals[idx_m], 2), "Tips": "Fakts"})
                prod_rows.append({"Periods": m_lbl, "Rādītājs": "T% = T/S", "Vērtība": round(tp_vals[idx_m], 2), "Tips": "Fakts"})
                prod_rows.append({"Periods": m_lbl, "Rādītājs": "Linear (PR = T/OE)", "Vērtība": round(pr_trend[idx_m], 2), "Tips": "Trends"})
                prod_rows.append({"Periods": m_lbl, "Rādītājs": "Linear (T% = T/S)", "Vērtība": round(tp_trend[idx_m], 2), "Tips": "Trends"})

            prod_df = pd.DataFrame(prod_rows)
            prod_domain = ["PR = T/OE", "T% = T/S", "Linear (PR = T/OE)", "Linear (T% = T/S)"]
            prod_colors = ["#3B68B5", "#F5B000", "#3B68B5", "#F5B000"]

            fact_prod = alt.Chart(prod_df[prod_df["Tips"] == "Fakts"]).mark_line(strokeWidth=2.8, point=True).encode(
                x=alt.X("Periods:N", sort=kpi_m_cols, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Vērtība:Q", title="Koeficients", scale=alt.Scale(zero=False)),
                color=alt.Color("Rādītājs:N", scale=alt.Scale(domain=prod_domain, range=prod_colors), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["Periods:N", "Rādītājs:N", alt.Tooltip("Vērtība:Q", format=".2f")],
            )
            trend_prod = alt.Chart(prod_df[prod_df["Tips"] == "Trends"]).mark_line(strokeDash=[3, 4], strokeWidth=2).encode(
                x=alt.X("Periods:N", sort=kpi_m_cols),
                y=alt.Y("Vērtība:Q"),
                color=alt.Color("Rādītājs:N", scale=alt.Scale(domain=prod_domain, range=prod_colors)),
                tooltip=["Periods:N", "Rādītājs:N", alt.Tooltip("Vērtība:Q", format=".2f")],
            )
            ref_one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[2, 2], color="#999999").encode(y="y:Q")
            st.altair_chart((ref_one + fact_prod + trend_prod).properties(height=320).interactive(), use_container_width=True)

    with g_col2:
        with st.container(border=True):
            st.markdown("##### Eiro Sales / Eiro algas")
            w_row = "Eiro Sales / Eiro strādnieku algas"
            a_row = "Eiro Sales / Eiro administrācijas algas"
            w_vals = [float(kpi_df.loc[w_row, c]) if w_row in kpi_df.index and not pd.isna(kpi_df.loc[w_row, c]) else 0.0 for c in kpi_m_cols]
            a_vals = [float(kpi_df.loc[a_row, c]) if a_row in kpi_df.index and not pd.isna(kpi_df.loc[a_row, c]) else 0.0 for c in kpi_m_cols]
            w_trend = _compute_linear_trend(w_vals)
            a_trend = _compute_linear_trend(a_vals)

            w_rows = []
            for idx_m, m_lbl in enumerate(kpi_m_cols):
                w_rows.append({"Periods": m_lbl, "Rādītājs": "Eiro Sales / Eiro Algas strādnieki", "Vērtība": round(w_vals[idx_m], 2), "Tips": "Fakts"})
                w_rows.append({"Periods": m_lbl, "Rādītājs": "Eiro Sales / Eiro Algas Admin", "Vērtība": round(a_vals[idx_m], 2), "Tips": "Fakts"})
                w_rows.append({"Periods": m_lbl, "Rādītājs": "Linear (Algas strādnieki)", "Vērtība": round(w_trend[idx_m], 2), "Tips": "Trends"})
                w_rows.append({"Periods": m_lbl, "Rādītājs": "Linear (Algas Admin)", "Vērtība": round(a_trend[idx_m], 2), "Tips": "Trends"})

            w_df = pd.DataFrame(w_rows)
            w_domain = [
                "Eiro Sales / Eiro Algas strādnieki",
                "Eiro Sales / Eiro Algas Admin",
                "Linear (Algas strādnieki)",
                "Linear (Algas Admin)",
            ]
            w_colors = ["#7B9CE0", "#1F3864", "#7B9CE0", "#1F3864"]

            fact_w = alt.Chart(w_df[w_df["Tips"] == "Fakts"]).mark_line(strokeWidth=2.8, point=True).encode(
                x=alt.X("Periods:N", sort=kpi_m_cols, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Vērtība:Q", title="Eiro Sales / 1 Eiro alga", scale=alt.Scale(zero=False)),
                color=alt.Color("Rādītājs:N", scale=alt.Scale(domain=w_domain, range=w_colors), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["Periods:N", "Rādītājs:N", alt.Tooltip("Vērtība:Q", format=".2f")],
            )
            trend_w = alt.Chart(w_df[w_df["Tips"] == "Trends"]).mark_line(strokeDash=[3, 4], strokeWidth=2).encode(
                x=alt.X("Periods:N", sort=kpi_m_cols),
                y=alt.Y("Vērtība:Q"),
                color=alt.Color("Rādītājs:N", scale=alt.Scale(domain=w_domain, range=w_colors)),
                tooltip=["Periods:N", "Rādītājs:N", alt.Tooltip("Vērtība:Q", format=".2f")],
            )
            st.altair_chart((fact_w + trend_w).properties(height=320).interactive(), use_container_width=True)


MAIN_GROUP_OPTIONS = [
    "S (Ieņēmumi)",
    "TVC (Mainīgās izmaksas)",
    "Uzturēšanas izmaksas (OE)",
    "Izslēgt (Nolietojums / Virtuāls)",
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
    if code == "S":
        return MAIN_GROUP_OPTIONS[0]
    if code == "TVC":
        return MAIN_GROUP_OPTIONS[1]
    if code == "EXCL":
        return MAIN_GROUP_OPTIONS[3]
    return MAIN_GROUP_OPTIONS[2]


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
    "OE_ADMIN_OTHER": 12,
    "OE_SALES_OTHER": 13,
    "OE_OPER_OTHER": 14,
    "OE_OTHER": 15,
    "EXCLUDED": 99,
}


def sort_cls_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "Konts", "Displeja_konts", "Sadaļa", "Grupa", "Ieņēmums", "Jautājums"])
    out = df.copy()
    out["Grupa"] = out["Grupa"].map(normalize_group_key)
    out["_sort"] = out["Grupa"].map(lambda g: GROUP_SORT_ORDER.get(g, 50))
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
            path_parts = [p.strip() for p in re.split(r"[›>]", str(getattr(lf, "path", ""))) if p.strip()]
            parent = path_parts[-1] if path_parts else f"rinda {lf.row}"
            display_map[lf.id] = f"{lbl} [{parent}]"
        else:
            display_map[lf.id] = lbl
    return display_map


def compute_toc_with_dynamic_subgroups(leaves, mapping: dict[str, dict], period_cols: list[dict], reported=None):
    res = compute_toc(leaves, mapping, period_cols, reported)
    res = strip_excluded_rows(res)

    month_labels = [p["label"] for p in period_cols if p.get("kind") == "month"]
    all_cols = list(res.amounts.columns)

    used_groups = {m["group"] for m in mapping.values() if m.get("group") and m["group"] != "EXCLUDED"}
    missing_groups = [g for g in used_groups if GROUPS.get(g, ("", ""))[0] not in res.amounts.index]

    if not missing_groups:
        return res

    amounts = res.amounts.copy()
    row_kinds = dict(res.row_kinds)

    for g_key in missing_groups:
        lbl, m_code = GROUPS.get(g_key, (g_key, "OE"))
        if m_code == "EXCL":
            continue
        kind_tag = f"{m_code}_CHILD"
        g_leaves = [lf for lf in leaves if mapping.get(lf.id, {}).get("group") == g_key]
        if not g_leaves:
            continue

        row_series = pd.Series(0.0, index=all_cols)
        for j, p in enumerate(period_cols):
            p_lbl = p["label"]
            if p_lbl in row_series.index:
                tot_j = 0.0
                for lf in g_leaves:
                    v = float(lf.vals[j]) if j < len(lf.vals) else 0.0
                    tot_j += v if m_code == "S" else abs(v)
                row_series[p_lbl] = tot_j

        for c in all_cols:
            if "YTD" in c.upper() and "AVG" not in c.upper() and month_labels:
                row_series[c] = row_series[month_labels].sum()
            elif "AVG" in c.upper() and month_labels:
                row_series[c] = row_series[month_labels].mean()

        amounts.loc[lbl] = row_series
        row_kinds[lbl] = kind_tag

    s_ch = [r for r in amounts.index if row_kinds.get(r) == "S_CHILD"]
    tvc_ch = [r for r in amounts.index if row_kinds.get(r) == "TVC_CHILD"]
    oe_ch = [r for r in amounts.index if row_kinds.get(r) == "OE_CHILD"]

    if s_ch:
        amounts.loc["S"] = amounts.loc[s_ch].sum(axis=0)
    if tvc_ch:
        amounts.loc["TVC"] = amounts.loc[tvc_ch].sum(axis=0)
    amounts.loc["T"] = amounts.loc["S"] - amounts.loc["TVC"]
    if oe_ch:
        amounts.loc["OE"] = amounts.loc[oe_ch].sum(axis=0)
    amounts.loc["NP"] = amounts.loc["T"] - amounts.loc["OE"]

    ordered_idx = ["S"] + s_ch + ["TVC"] + tvc_ch + ["T", "OE"] + oe_ch + ["NP"]
    ordered_idx = [r for r in ordered_idx if r in amounts.index]
    amounts = amounts.loc[ordered_idx]

    main_dict = {}
    for c in amounts.columns:
        main_dict[c] = amounts[c]
        s_val = amounts.loc["S", c]
        main_dict[f"%|{c}"] = np.where(s_val != 0, amounts[c] / s_val, np.nan)
    res.amounts = amounts
    res.main = pd.DataFrame(main_dict, index=amounts.index)
    res.row_kinds = row_kinds
    return strip_excluded_rows(res)


def rebuild_report_from_account_snapshot(res, acc_details: list[dict]):
    """
    Pārrēķina finanšu tabulu no saglabātā kontu mēnešu snapshot (pat tad, ja skatāmies arhīva atskaiti),
    lai Asistents vai lietotājs var sadalīt apakšgrupas sīkāk jebkurā brīdī!
    """
    if not acc_details or res is None:
        return res

    all_cols = list(res.amounts.columns)
    month_cols = [c for c in all_cols if not ("YTD" in str(c).upper() or "AVG" in str(c).upper())]

    used_groups = []
    for it in acc_details:
        gk = normalize_group_key(it.get("group", "OE_OTHER"))
        if gk != "EXCLUDED" and gk not in used_groups:
            used_groups.append(gk)
    used_groups.sort(key=lambda g: GROUP_SORT_ORDER.get(g, 50))

    new_rows = {}
    new_kinds = {}

    for m_code in ("S", "TVC", "OE"):
        m_groups = [g for g in used_groups if get_main_code(g) == m_code]
        ch_labels = []
        for gk in m_groups:
            lbl = SUBGROUP_LABELS.get(gk, gk)
            g_items = [it for it in acc_details if normalize_group_key(it.get("group", "")) == gk]
            if not g_items:
                continue
            s_row = pd.Series(0.0, index=all_cols)
            for mc in month_cols:
                tot_m = 0.0
                for it in g_items:
                    v = float(it.get("months", {}).get(mc, 0.0) or 0.0)
                    tot_m += v if m_code == "S" else abs(v)
                s_row[mc] = tot_m
            for c in all_cols:
                if "YTD" in c.upper() and "AVG" not in c.upper() and month_cols:
                    s_row[c] = s_row[month_cols].sum()
                elif "AVG" in c.upper() and month_cols:
                    s_row[c] = s_row[month_cols].mean()
            new_rows[lbl] = s_row
            new_kinds[lbl] = f"{m_code}_CHILD"
            ch_labels.append(lbl)

        if ch_labels:
            new_rows[m_code] = pd.DataFrame({k: new_rows[k] for k in ch_labels}).sum(axis=1)
        else:
            new_rows[m_code] = res.amounts.loc[m_code].copy() if m_code in res.amounts.index else pd.Series(0.0, index=all_cols)
        new_kinds[m_code] = m_code

    new_rows["T"] = new_rows["S"] - new_rows["TVC"]
    new_kinds["T"] = "T"
    new_rows["NP"] = new_rows["T"] - new_rows["OE"]
    new_kinds["NP"] = "NP"

    s_ch = [k for k, v in new_kinds.items() if v == "S_CHILD"]
    tvc_ch = [k for k, v in new_kinds.items() if v == "TVC_CHILD"]
    oe_ch = [k for k, v in new_kinds.items() if v == "OE_CHILD"]
    ordered = ["S"] + s_ch + ["TVC"] + tvc_ch + ["T", "OE"] + oe_ch + ["NP"]

    new_amounts = pd.DataFrame({r: new_rows[r] for r in ordered}).T
    new_main_dict = {}
    for c in new_amounts.columns:
        new_main_dict[c] = new_amounts[c]
        sv = new_amounts.loc["S", c]
        new_main_dict[f"%|{c}"] = np.where(sv != 0, new_amounts[c] / sv, np.nan)
    res.amounts = new_amounts
    res.main = pd.DataFrame(new_main_dict, index=new_amounts.index)
    res.row_kinds = new_kinds
    return res


def _kpi_cond_color(idx_name: str, val: float, row_vals: list[float]) -> str:
    if pd.isna(val):
        return "#F8F9F9"
    if idx_name.startswith("PR"):
        if val >= 1.15:
            return "#D4EFDF"
        elif val >= 1.00:
            return "#FCF3CF"
        else:
            return "#FADBD8"

    clean = [float(x) for x in row_vals if not pd.isna(x)]
    if len(clean) < 2:
        return "#F8F9F9"
    q_low = float(np.percentile(clean, 33))
    q_high = float(np.percentile(clean, 67))
    if q_high - q_low < 1e-6:
        return "#F8F9F9"
    if val >= q_high:
        return "#D4EFDF"
    elif val <= q_low:
        return "#FADBD8"
    return "#FCF3CF"


def render_unified_toc_tables_html(
    amounts_df: pd.DataFrame,
    main_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    row_kinds: dict[str, str],
    title_main: str = "Finanšu pārskata tabula",
    title_kpi: str = "Produktivitātes un atdeves rādītāji",
    grey_cols: set[str] | None = None,
    fy_start_month: int = 1,
    show_cumulative: bool = True,
) -> str:
    valid_rows = [
        r for r in amounts_df.index
        if row_kinds.get(r) not in ("EXCL", "EXCLUDED") and not str(r).lower().startswith("izslēgt")
    ]
    amounts_df = amounts_df.loc[valid_rows]
    main_df = main_df.loc[valid_rows]

    periods = list(amounts_df.columns)
    n_per = max(len(periods), 1)
    row_names = list(amounts_df.index)
    kinds = [row_kinds.get(rn, "OE_CHILD") for rn in row_names]
    grey_cols = grey_cols or set()

    display_row_name = lambda rn: "Uzturēšanas izmaksas (OE)" if rn == "OE" else rn

    lbl_w_px = 235

    def _build_two_subcol_colgroup(num_periods: int, total_ref_periods: int) -> tuple[str, str]:
        if num_periods == total_ref_periods:
            table_style = "width: 100%; table-layout: fixed;"
        else:
            ratio_pct = (num_periods / total_ref_periods) * 100.0
            extra_px = lbl_w_px * (1.0 - num_periods / total_ref_periods)
            table_style = f"width: calc({ratio_pct:.4f}% + {extra_px:.1f}px); table-layout: fixed;"

        cols_html = [f'<col style="width: {lbl_w_px}px;">']
        for _ in range(num_periods):
            cols_html.append('<col style="width: 66%;">')
            cols_html.append('<col style="width: 34%;">')
        return "<colgroup>" + "".join(cols_html) + "</colgroup>", table_style

    def _build_single_col_kpi_colgroup(num_periods: int, total_ref_periods: int) -> tuple[str, str]:
        if num_periods == total_ref_periods:
            table_style = "width: 100%; table-layout: fixed;"
        else:
            ratio_pct = (num_periods / total_ref_periods) * 100.0
            extra_px = lbl_w_px * (1.0 - num_periods / total_ref_periods)
            table_style = f"width: calc({ratio_pct:.4f}% + {extra_px:.1f}px); table-layout: fixed;"

        cols_html = [f'<col style="width: {lbl_w_px}px;">']
        for _ in range(num_periods):
            cols_html.append('<col style="width: 100%;">')
        return "<colgroup>" + "".join(cols_html) + "</colgroup>", table_style

    main_colgroup, main_tbl_style = _build_two_subcol_colgroup(n_per, n_per)

    html = ["""
    <style>
    .toc-table-wrap { width: 100%; overflow-x: auto; margin-bottom: 14px; }
    .toc-sec-heading {
        font-family: 'Inter', 'Segoe UI', 'Calibri', sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin: 14px 0 8px 0;
        padding: 6px 14px;
        background-color: #1E293B;
        color: #FFFFFF !important;
        border-radius: 6px;
        display: inline-block;
    }
    table.toc-unified {
        border-collapse: collapse !important;
        font-family: 'Inter', 'Calibri', 'Segoe UI', sans-serif;
        font-size: 12px;
        border: 2.5px solid #000000 !important;
    }
    table.toc-unified th {
        background-color: #E2E8F0;
        color: #000000 !important;
        font-weight: 700;
        text-align: center;
        padding: 6px 5px;
        border-top: 2.5px solid #000000 !important;
        border-bottom: 2.5px solid #000000 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    table.toc-unified td {
        padding: 4px 6px;
        white-space: nowrap;
        color: #000000 !important;
        overflow: hidden;
        text-overflow: ellipsis;
        border-top: 1px solid #222222 !important;
        border-bottom: 1px solid #222222 !important;
    }
    table.toc-unified tr.thick-bottom td {
        border-bottom: 2.5px solid #000000 !important;
    }
    table.toc-unified td.lbl-main {
        text-align: left;
        font-weight: 700;
        padding-left: 8px;
        padding-right: 8px;
        border-left: 2.5px solid #000000 !important;
        border-right: 2.5px solid #000000 !important;
        color: #000000 !important;
    }
    table.toc-unified td.lbl-sub {
        text-align: left;
        font-weight: 400;
        padding-left: 16px;
        padding-right: 8px;
        border-left: 2.5px solid #000000 !important;
        border-right: 2.5px solid #000000 !important;
        color: #000000 !important;
    }
    /* Esošās darbības kopā stils: pus-bold (600), ne tik spēcīgs kā galvenās grupas (700) */
    table.toc-unified td.lbl-sub-medium {
        text-align: left;
        font-weight: 600;
        padding-left: 12px;
        padding-right: 8px;
        border-left: 2.5px solid #000000 !important;
        border-right: 2.5px solid #000000 !important;
        color: #1E293B !important;
    }
    table.toc-unified td.val-col {
        text-align: right;
        border-left: 1px solid #222222 !important;
        border-right: 1px solid #222222 !important;
        font-variant-numeric: tabular-nums;
        color: #000000 !important;
    }
    table.toc-unified td.pct-col {
        text-align: right;
        font-size: 11px;
        padding-left: 2px;
        padding-right: 5px;
        border-left: 1px solid #222222 !important;
        border-right: 2.5px solid #000000 !important;
        color: #000000 !important;
        font-variant-numeric: tabular-nums;
    }
    table.toc-unified td.kpi-col {
        text-align: right;
        padding: 4px 8px;
        border-left: 2.5px solid #000000 !important;
        border-right: 2.5px solid #000000 !important;
        font-weight: 400;
        font-variant-numeric: tabular-nums;
        color: #000000 !important;
    }
    table.toc-unified th.th-kpi {
        border-top: 2.5px solid #000000 !important;
        border-bottom: 2.5px solid #000000 !important;
        border-left: 2.5px solid #000000 !important;
        border-right: 2.5px solid #000000 !important;
    }
    table.toc-unified th.th-val { border-left: 1px solid #222222 !important; border-right: 1px solid #222222 !important; }
    table.toc-unified th.th-pct { font-size: 11px; border-left: 1px solid #222222 !important; border-right: 2.5px solid #000000 !important; }
    table.toc-unified th.th-lbl { text-align: left; padding-left: 8px; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; }
    </style>
    """]

    if title_main:
        html.append(f'<div class="toc-sec-heading">{title_main}</div>')
    html.append(f'<div class="toc-table-wrap"><table class="toc-unified" style="{main_tbl_style}">{main_colgroup}<thead><tr>')
    html.append('<th class="th-lbl">Rādītājs / Kontugrupa</th>')
    for p in periods:
        th_bg = "background-color: #CBD5E1; color: #000000;" if p in grey_cols else ""
        html.append(f'<th class="th-val" style="{th_bg}">{p}</th><th class="th-pct" style="{th_bg}">%</th>')
    html.append("</tr></thead><tbody>")

    s_grp_kinds = ("S", "S_CHILD", "S_SUBTOTAL")
    tvc_grp_kinds = ("TVC", "TVC_CHILD", "TVC_SUBTOTAL")
    t_grp_kinds = ("T", "T_CHILD", "T_SUBTOTAL")
    oe_grp_kinds = ("OE", "OE_CHILD")
    subtotal_kinds = ("S_SUBTOTAL", "TVC_SUBTOTAL", "T_SUBTOTAL")

    for i, rname in enumerate(row_names):
        kind = kinds[i]
        next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
        fill_hex, is_bold, _ = FILLS.get(kind, FILLS["OE_CHILD"])

        thick_bottom = (
            kind == "NP"
            or (kind in t_grp_kinds and next_kind not in t_grp_kinds)
            or (kind in s_grp_kinds and next_kind not in s_grp_kinds)
            or (kind in tvc_grp_kinds and next_kind not in tvc_grp_kinds)
            or (kind in oe_grp_kinds and next_kind not in oe_grp_kinds)
            or i == len(row_names) - 1
        )
        tr_cls = ' class="thick-bottom"' if thick_bottom else ""

        if kind in subtotal_kinds:
            font_w = "600"
            lbl_cls = "lbl-sub-medium"
        elif is_bold:
            font_w = "700"
            lbl_cls = "lbl-main"
        else:
            font_w = "400"
            lbl_cls = "lbl-sub"

        disp_rn = display_row_name(rname)

        row_style = f"background-color: #{fill_hex}; color: #000000; font-weight: {font_w};"
        html.append(f'<tr{tr_cls} style="{row_style}"><td class="{lbl_cls}" title="{disp_rn}">{disp_rn}</td>')

        for p in periods:
            val_str = fmt_num(main_df.loc[rname, p])
            pct_str = fmt_pct_int(main_df.loc[rname, f"%|{p}"])
            if p in grey_cols:
                g_bg = "#CBD5E1" if is_bold else ("#DCE4EE" if kind in subtotal_kinds else "#E2E8F0")
                cell_style = f"background-color: {g_bg}; color: #000000; font-weight: {font_w};"
            else:
                cell_style = f"color: #000000; font-weight: {font_w};"
            html.append(f'<td class="val-col" style="{cell_style}">{val_str}</td><td class="pct-col" style="{cell_style}">{pct_str}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")

    kpi_periods = [p for p in periods if p in kpi_df.columns]
    kpi_colgroup, kpi_tbl_style = _build_single_col_kpi_colgroup(len(kpi_periods), n_per)
    month_only_kpi_cols = [
        p for p in kpi_periods
        if not ("YTD" in str(p).upper() or "AVG" in str(p).upper() or "BĀZE" in str(p).upper())
    ]

    html.append(f'<div class="toc-sec-heading">{title_kpi}</div>')
    html.append(f'<div class="toc-table-wrap"><table class="toc-unified" style="{kpi_tbl_style}">{kpi_colgroup}<thead><tr>')
    html.append('<th class="th-lbl">Produktivitātes rādītājs</th>')
    for p in kpi_periods:
        th_bg = "background-color: #CBD5E1;" if p in grey_cols else "background-color: #E2E8F0;"
        html.append(f'<th class="th-kpi" style="{th_bg}">{p}</th>')
    html.append("</tr></thead><tbody>")

    for i, idx_name in enumerate(kpi_df.index):
        tr_cls = ' class="thick-bottom"' if i == len(kpi_df.index) - 1 else ""
        html.append(f'<tr{tr_cls} style="background-color: #FFFFFF; color: #000000;">')
        html.append(f'<td class="lbl-main" style="background-color: #F1F5F9; font-weight: 700; color: #000000;" title="{idx_name}">{idx_name}</td>')

        row_month_vals = [kpi_df.loc[idx_name, mp] for mp in month_only_kpi_cols] if month_only_kpi_cols else list(kpi_df.loc[idx_name].values)

        for p in kpi_periods:
            v = kpi_df.loc[idx_name, p]
            if pd.isna(v):
                s = ""
            elif idx_name.startswith("T%"):
                s = f"{v * 100:.0f}%"
            else:
                s = f"{v:.2f}"

            if p in grey_cols:
                bg_cell = "#E2E8F0"
            else:
                bg_cell = _kpi_cond_color(idx_name, v, row_month_vals)

            html.append(f'<td class="kpi-col" style="background-color: {bg_cell};">{s}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")

    if show_cumulative:
        cum_df, cum_periods, _ = compute_cumulative_toc_df(amounts_df, fy_start_month=fy_start_month)
        html.append('<div class="toc-sec-heading">Uzkrājošie rādītāji</div>')
        html.append(f'<div class="toc-table-wrap"><table class="toc-unified" style="{main_tbl_style}">{main_colgroup}<thead><tr>')
        html.append('<th class="th-lbl">Uzkrājošais rādītājs</th>')
        for p in periods:
            if p in cum_periods:
                html.append(f'<th class="th-val">{p}</th><th class="th-pct">%</th>')
            else:
                html.append(
                    f'<th colspan="2" style="background-color: #E2E8F0; color: #000000; border-top: 2.5px solid #000000 !important; '
                    f'border-bottom: 2.5px solid #000000 !important; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important;">{p}</th>'
                )
        html.append("</tr></thead><tbody>")

        s_row_label = cum_df.index[0] if len(cum_df.index) > 0 else None

        cum_label_fills = {
            "S (uzkrājošais)": FILLS["S"][0],
            "TVC (uzkrājošais)": FILLS["TVC"][0],
            "T (uzkrājošais)": FILLS["T"][0],
            "NP (uzkrājošais)": FILLS["NP"][0],
        }

        for idx_c, c_name in enumerate(cum_df.index):
            tr_cls = ' class="thick-bottom"' if idx_c == len(cum_df.index) - 1 else ""
            lbl_bg_hex = cum_label_fills.get(c_name, "F1F5F9")
            is_np_cum = c_name.startswith("NP")

            html.append(f'<tr{tr_cls} style="background-color: #FFFFFF; color: #000000; font-weight: 400;">')
            html.append(
                f'<td class="lbl-main" style="background-color: #{lbl_bg_hex}; font-weight: 700; color: #000000;" title="{c_name}">{c_name}</td>'
            )

            for p in periods:
                if p in cum_df.columns:
                    val_c = cum_df.loc[c_name, p]
                    s_c = cum_df.loc[s_row_label, p] if s_row_label else np.nan
                    pct_c = (val_c / s_c) if (s_c and not pd.isna(s_c) and s_c != 0) else np.nan
                    val_s = fmt_num(val_c)
                    pct_s = fmt_pct_int(pct_c)

                    if is_np_cum:
                        month_np_val = float(amounts_df.loc["NP", p]) if "NP" in amounts_df.index and p in amounts_df.columns else val_c
                        np_bg = "#D4EFDF" if (month_np_val >= 0 and val_c >= 0) else "#FADBD8"
                    else:
                        np_bg = "#FFFFFF"

                    html.append(
                        f'<td class="val-col" style="background-color: {np_bg}; color: #000000; font-weight: 400;">{val_s}</td>'
                        f'<td class="pct-col" style="background-color: {np_bg}; color: #000000; font-weight: 400;">{pct_s}</td>'
                    )
                else:
                    last_p = cum_periods[-1] if cum_periods else None
                    if last_p and "YTD" in str(p).upper() and "AVG" not in str(p).upper():
                        v_ytd = cum_df.loc[c_name, last_p]
                        ytd_bg = ("#D4EFDF" if v_ytd >= 0 else "#FADBD8") if is_np_cum else "#F8F9F9"
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; background-color: {ytd_bg}; color: #000000; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; font-weight: 400;">{fmt_num(v_ytd)}</td>'
                        )
                    elif last_p and "AVG" in str(p).upper() and len(cum_periods) > 0:
                        v_avg = cum_df.loc[c_name, last_p] / len(cum_periods)
                        avg_bg = ("#D4EFDF" if v_avg >= 0 else "#FADBD8") if is_np_cum else "#F8F9F9"
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; background-color: {avg_bg}; color: #000000; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; font-weight: 400;">{fmt_num(v_avg)}</td>'
                        )
                    else:
                        html.append('<td colspan="2" style="background-color: #FFFFFF; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important;"></td>')
            html.append("</tr>")

        html.append("</tbody></table></div>")

    return "".join(html)


knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

S = st.session_state

for c_key, (c_lbl, c_mc) in S.get("custom_subgroups", {}).items():
    register_custom_subgroup(c_key, c_lbl, c_mc)


# ------------------------------------------------------------------ PELDOŠA RĪKJOSLA UN DINAMISKS ASISTENTA LOGA IZMĒRS
is_expanded_asist = bool(S.get("assistant_expanded", False))
pop_min_w = "600px" if is_expanded_asist else "390px"
pop_max_w = "660px" if is_expanded_asist else "430px"

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    .block-container {{
        padding-top: 3.0rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 97% !important;
    }}
    div[data-testid="InputInstructions"] {{
        display: none !important;
    }}
    div[data-testid="stPopoverBody"] {{
        min-width: {pop_min_w} !important;
        max-width: {pop_max_w} !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
    }}
    div.sticky-nav-marker {{
        display: none !important;
    }}
    div[data-testid="stElementContainer"]:has(div.sticky-nav-marker),
    div.element-container:has(div.sticky-nav-marker) {{
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 0 !important;
    }}
    div[data-testid="stVerticalBlock"] > div:has(div.sticky-nav-marker) {{
        position: sticky !important;
        top: 2.8rem !important;
        z-index: 999 !important;
        background-color: var(--secondary-background-color, #F4F3EE) !important;
        border: 1px solid rgba(120, 120, 120, 0.28) !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07) !important;
        margin-bottom: 16px !important;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        letter-spacing: -0.3px;
    }}
    div[data-testid="stPopoverBody"] p,
    div[data-testid="stPopoverBody"] li,
    div[data-testid="stPopoverBody"] span {{
        font-size: 13px !important;
        line-height: 1.45 !important;
    }}
    .sim-col-header-s {{
        background-color: #FFF200; color: #000000 !important; font-weight: 700;
        padding: 6px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 5px solid #CA8A04; font-size: 13px;
    }}
    .sim-col-header-tvc {{
        background-color: #C55A11; color: #FFFFFF !important; font-weight: 700;
        padding: 6px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 5px solid #7C2D12; font-size: 13px;
    }}
    .sim-col-header-oe {{
        background-color: #F4B183; color: #000000 !important; font-weight: 700;
        padding: 6px 12px; border-radius: 6px; margin-bottom: 6px; border-left: 5px solid #C2410C; font-size: 13px;
    }}
    .top-brand-bar {{
        display: flex; align-items: baseline; gap: 12px;
    }}
    .top-brand-title {{
        font-size: 20px; font-weight: 800; letter-spacing: -0.5px; margin: 0;
    }}
    .top-brand-sub {{
        font-size: 13px; opacity: 0.8;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def build_global_diagnostic_context(profile_store: SupabaseProfileStore) -> str:
    parts = [f"Aktīvais uzņēmuma profils: {S.get('active_profile', 'Nav')}"]
    parts.append(f"Pašreizējā atvērtā sadaļa aplikācijā: {S.get('nav_page', 'overview')}")
    parts.append(
        f"Finanšu gada sākums: {S.get('fy_start_day', 1)}. {MONTH_NAMES_LV.get(S.get('fy_start_month', 1), 'Janvāris')}"
    )

    # 1. Pilns apakšgrupu un tajās ietilpstošo PZA pamatkontu sadalījums (lai Asistents vienmēr zina, kas slēpjas zem katras grupas!)
    acc_snap: list[dict] = []
    if "cls_df" in S and S.cls_df is not None and not S.cls_df.empty and "analysis" in S and S.analysis is not None:
        acc_snap = build_account_details_snapshot(S.analysis.leaves, S.cls_df, S.analysis.ex.period_cols)
    elif S.get("loaded_account_details"):
        acc_snap = S.get("loaded_account_details") or []

    if acc_snap:
        by_sub: dict[str, list[str]] = {}
        for it in acc_snap:
            gk = normalize_group_key(it.get("group", "OE_OTHER"))
            sub_lbl = SUBGROUP_LABELS.get(gk, gk)
            m_code = get_main_code(gk)
            tot_v = float(it.get("total", 0.0))
            disp_v = tot_v if m_code == "S" else abs(tot_v)
            by_sub.setdefault(f"[{m_code}] {sub_lbl}", []).append(
                f"  • '{it.get('label')}' (PZA rinda #{it.get('row')}, ceļš: '{it.get('path') or 'Pamatlīmenis'}') -> Kopā periodā: {disp_v:,.0f} EUR"
            )
        sub_lines = []
        for sg_title, acc_list in by_sub.items():
            sub_lines.append(f"Apakšgrupa {sg_title} ({len(acc_list)} konti):\n" + "\n".join(acc_list))
        parts.append("\n--- KAS SLĒPJAS ZEM KATRAS APAKŠGRUPAS (PZA PAMATKONTI UN TO SUMMAS) ---\n" + "\n\n".join(sub_lines))
    else:
        saved_m = profile_store.list_saved_mappings()
        if saved_m:
            by_sub_db: dict[str, list[str]] = {}
            for m_row in saved_m:
                gk = normalize_group_key(m_row.get("toc_group", "OE_OTHER"))
                sub_lbl = SUBGROUP_LABELS.get(gk, gk)
                m_code = get_main_code(gk)
                by_sub_db.setdefault(f"[{m_code}] {sub_lbl}", []).append(
                    f"  • '{m_row.get('account_label')}' (PZA sadaļa: '{m_row.get('path_label') or 'Pamatlīmenis'}')"
                )
            sub_lines_db = [f"Apakšgrupa {k} ({len(v)} konti):\n" + "\n".join(v) for k, v in by_sub_db.items()]
            parts.append("\n--- KAS SLĒPJAS ZEM KATRAS APAKŠGRUPAS (NO PROFILA DATUBĀZES) ---\n" + "\n\n".join(sub_lines_db))

    if "result" in S and S.result is not None:
        res = S.result
        parts.append("\n--- AKTĪVĀ TOC-FIN TABULA (EUR) ---\n" + res.amounts.round(0).to_markdown())
        parts.append("\n--- PRODUKTIVITĀTES UN ATDEVES RĀDĪTĀJI (KPI) ---\n" + res.kpi.round(3).to_markdown())

    return "\n\n".join(parts)


TOC_CORE_EXPERT_KNOWLEDGE = """
=== TOC (IEROBEŽOJUMU TEORIJAS UN CAURLAIDĪBAS GRĀMATVEDĪBAS) ZINĀŠANU BĀZE ===
1. Mērķis: Pelnīt naudu šobrīd un vienmēr. Formulas: T = S - TVC; NP = T - OE; PR = T / OE; ROI = (T - OE) / I.
2. TVC (Patiesi mainīgās izmaksas): Tikai izmaksas, kas mainās 1:1 ar katru pārdoto vienību (izejvielas, materiāli, tiešā piegāde klientam, pārdošanas komisijas). Darba algas, elektrība, apkure, noma un remonti NEKAD nav TVC — tās ir OE (Uzturēšanas izmaksas).
3. OE daba un Sviras efekts: OE nemainās proporcionāli apjomam (tām ir "pakāpienu" raksturs). Tiklīdz uzkrājošais T pārsniedz OE (lūzuma punkts T = OE), katrs nākamais T eiro pilnībā pārvēršas tīrajā peļņā (NP)!
4. Lēmumu prioritāte: 1) T ↑ (palielināt Caurlaidību — tam nav griestu!), 2) I ↓ (samazināt krājumos/WIP iesaldēto naudu), 3) OE ↓ (samazināt tikai tad, ja tas nekaitē T). Operatīvā formula: ΔT - ΔOE > 0.
5. Produktu un klientu prioritāte: Ja ierobežo jauda (CCR 100%), prioritāti nosaka Tu / CCR minūtē. Ja ierobežo tirgus (ir brīva jauda), jebkurš pasūtījums vai produkts ar pozitīvu T (kur ΔT - ΔOE > 0) palielina peļņu (atteikšanās no produkta ar pozitīvu T brīvas jaudas apstākļos samazina peļņu, jo OE paliek!).

=== KĀ ORIENTĒTIES UN STRĀDĀT TOC-FIN APLIKĀCIJĀ ===
Augšējā rīkjoslā ir 6 galvenās sadaļas:
- 'Pārskats' (nav_page='overview'): Finanšu tabula, Produktivitātes KPI, Uzkrājošie rādītāji, Grafiki un Vadības kopsavilkums.
- 'Scenāriju simulācija' (nav_page='simulation'): Kur var mainīt S, TVC un OE par % vai EUR, kā arī pievienot 'Jaunu produktu / projektu' un redzēt Esošo darbību kopā, Jauno produktu un Kopējo rezultātu.
- 'Kontu struktūra' (nav_page='accounts_explorer'): 3 līmeņu izklājums (1. Lielā grupa -> 2. Apakšgrupa -> 3. Konkrētie PZA pamatkonti ar mēnešu skaitļiem). Šeit var arī izveidot jaunu apakšgrupu un pārvietot jebkuru kontu uz jauno apakšgrupu!
- 'Atskaites un Simulācijas' (nav_page='archive'): Saglabāto finanšu atskaišu un simulāciju arhīvs.
- 'Jauns PZA / Klasifikācija' (nav_page='ingest'): Jauna PZA faila ielasīšana un kontu klasifikācija.
- 'Profils un Uzstādījumi' (nav_page='settings'): Finanšu gada sākuma datums, paroles maiņa un apakšgrupu pievienošana.

=== KĀ ASISTENTAM PIEDĀVĀT AUTOMĀTISKU DARBĪBU (ACTION) ===
Ja lietotājs vēlas:
a) doties uz kādu aplikācijas sadaļu (piem. 'Kontu struktūra' vai 'Scenāriju simulācija'), VAI
b) izveidot jaunu apakšgrupu un/vai sadalīt kādu lielo apakšgrupu (pārvietojot konkrētus PZA kontus uz jauno apakšgrupu),
tad savas atbildes tekstā laipni izstāsti, ko iesaki darīt, un pašās atbildes beigās pievieno JSON bloku tagos <ACTION>...</ACTION> (lietotājam parādīsies apstiprinājuma poga, lai viņš pirms izpildes redz un apstiprina izmaiņas!):

Piemērs 1 (Sadalīt grupu / izveidot jaunu apakšgrupu un pārvietot kontus):
<ACTION>{"type": "split_or_move_accounts", "summary": "Izveidot jaunu OE apakšgrupu 'Personāla labbūtības un apmācību izdevumi' un pārvietot uz to 2 kontus", "main_code": "OE", "new_subgroup": "Personāla labbūtības un apmācību izdevumi", "accounts": ["Darbinieku apdrošināšanas maksājumi", "Personāla apmācības izdevumi"]}</ACTION>

Piemērs 2 (Aizvest lietotāju uz konkrētu aplikācijas sadaļu):
<ACTION>{"type": "navigate", "summary": "Atvērt sadaļu 'Kontu struktūra'", "target_page": "accounts_explorer"}</ACTION>
"""


def execute_assistant_action(act: dict, profile_store: SupabaseProfileStore) -> tuple[bool, str]:
    """Izpilda lietotāja apstiprināto Asistenta darbību."""
    act_type = act.get("type", "")

    if act_type == "navigate":
        target = act.get("target_page", "accounts_explorer")
        S.nav_page = target
        st.query_params["p"] = target
        return True, f"Atvērta sadaļa **{target}**."

    if act_type in ("split_or_move_accounts", "create_subgroup"):
        main_code = str(act.get("main_code") or "OE").upper()
        if main_code not in ("S", "TVC", "OE"):
            main_code = "OE"
        new_sub_lbl = str(act.get("new_subgroup") or "").strip()
        acc_names: list[str] = [str(x).strip() for x in (act.get("accounts") or []) if str(x).strip()]

        target_g_key = None
        if new_sub_lbl:
            if new_sub_lbl in LABEL_TO_KEY:
                target_g_key = LABEL_TO_KEY[new_sub_lbl]
            else:
                target_g_key = f"{main_code}_CUSTOM_{hashlib.md5(new_sub_lbl.encode()).hexdigest()[:6].upper()}"
                S.setdefault("custom_subgroups", {})[target_g_key] = (new_sub_lbl, main_code)
                register_custom_subgroup(target_g_key, new_sub_lbl, main_code)

        moved_cnt = 0
        # 1) Ja ir aktīvs cls_df sesijā
        if target_g_key and acc_names and "cls_df" in S and S.cls_df is not None and not S.cls_df.empty:
            for acc_nm in acc_names:
                mask = (S.cls_df["Konts"].str.lower() == acc_nm.lower()) | (S.cls_df["Displeja_konts"].str.lower() == acc_nm.lower())
                if mask.any():
                    S.cls_df.loc[mask, "Grupa"] = target_g_key
                    S.cls_df.loc[mask, "Ieņēmums"] = target_g_key.startswith("S_")
                    moved_cnt += int(mask.sum())
            S.cls_df = sort_cls_df(S.cls_df)
            if "analysis" in S and S.analysis is not None:
                an = S.analysis
                stc = S.structure
                mapping_upd = {
                    r_u["id"]: {
                        "group": normalize_group_key(r_u["Grupa"]),
                        "is_income": bool(str(r_u["Grupa"]).startswith("S_")),
                    }
                    for _, r_u in S.cls_df.iterrows()
                }
                pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
                reported = an.ex.lines[pos].vals if pos is not None else None
                S.result = compute_toc_with_dynamic_subgroups(an.leaves, mapping_upd, an.ex.period_cols, reported)
                S.loaded_account_details = build_account_details_snapshot(an.leaves, S.cls_df, an.ex.period_cols)
                profile_store.save(an.leaves, mapping_upd)

        # 2) Ja skatāmies ielādētu atskaiti ar loaded_account_details
        elif target_g_key and acc_names and S.get("loaded_account_details"):
            snap_list = S.get("loaded_account_details") or []
            for it in snap_list:
                if it.get("label", "").strip().lower() in [a.lower() for a in acc_names] or it.get("display_label", "").strip().lower() in [a.lower() for a in acc_names]:
                    it["group"] = target_g_key
                    moved_cnt += 1
            S.loaded_account_details = snap_list
            if S.get("result") is not None:
                S.result = rebuild_report_from_account_snapshot(S.result, snap_list)
                if S.get("current_report_id"):
                    packed_c = pack_commentary_with_account_details(S.get("commentary", ""), snap_list)
                    profile_store.update_report_commentary(S.current_report_id, packed_c)

        # 3) Atjauninām arī datubāzes saglabātos kontus (saved_mappings)
        if target_g_key and acc_names:
            saved_m = profile_store.list_saved_mappings()
            db_updates = []
            for m_row in saved_m:
                if str(m_row.get("account_label", "")).strip().lower() in [a.lower() for a in acc_names]:
                    db_updates.append({"id": int(m_row["id"]), "toc_group": target_g_key})
            if db_updates:
                profile_store.update_saved_mappings_direct(db_updates)
                if moved_cnt == 0:
                    moved_cnt = len(db_updates)

        return True, f"Izveidota apakšgrupa **'{new_sub_lbl}'** un pārvietoti **{moved_cnt}** konti! Tabula ir pārrēķināta."

    return False, "Nezināma darbība."


def ask_global_assistant(user_question: str, profile_store: SupabaseProfileStore) -> tuple[str, dict | None, int, float]:
    ctx = build_global_diagnostic_context(profile_store)
    know = f"\nPapildu lietotāja faila zināšanu bāze:\n{knowledge_file}\n" if knowledge_file else ""
    sys_prompt = (
        "Tu esi laipns, cilvēcīgs un augsti profesionāls TOC-FIN eksperts un aplikācijas gids.\n\n"
        f"{TOC_CORE_EXPERT_KNOWLEDGE}\n"
        f"{know}\n"
        "SVARĪGI NOTEIKUMI ČATA STILAM:\n"
        "1. Raksti viegli lasāmā ČATA FORMĀTĀ: īsas rindkopas (1–2 teikumi), ar atstarpēm un aizzīmēm (•), lai lietotājam nav jālasa garš, blīvs teksta klucis!\n"
        "2. Ja lietotājs jautā, kas slēpjas zem kādas apakšgrupas vai vai kāda OE/TVC grupa nav par lielu — nosauc konkrētos PZA kontus un to EUR summas no konteksta un iesaki, kā tos vislabāk sadalīt.\n"
        "3. Paskaidro, kur aplikācijā to var izdarīt (sadaļā **Kontu struktūra**), UN piedāvā to izdarīt automātiski, pašās beigās pievienojot `<ACTION>...</ACTION>` JSON bloku (lietotājam parādīsies apstiprinājuma poga)."
    )
    history = S.get("global_chat", [])
    clean_history = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
    messages = [
        {"role": "user", "content": f"Sistēmas un uzņēmuma finanšu konteksts:\n{ctx}"},
        {"role": "assistant", "content": "Sapratu! Esmu gatavs palīdzēt gan ar TOC finanšu analīzi, gan ar kontu sadalīšanu un navigāciju aplikācijā."},
    ] + clean_history + [{"role": "user", "content": user_question}]

    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=850,
        system=sys_prompt,
        messages=messages,
    )
    raw_ans = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    action_obj = None
    m_act = re.search(r"<ACTION>\s*(\{.*?\})\s*</ACTION>", raw_ans, re.S)
    if m_act:
        try:
            action_obj = json.loads(m_act.group(1))
        except Exception:
            action_obj = None
        raw_ans = (raw_ans[: m_act.start()] + raw_ans[m_act.end() :]).strip()

    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 500) if usage else 500
    out_tok = getattr(usage, "output_tokens", 250) if usage else 250
    cost_eur = (in_tok * 0.000003 + out_tok * 0.000015) * 0.92
    return raw_ans, action_obj, in_tok + out_tok, cost_eur


def parse_simulation_dialog_with_claude(user_text: str, s_rows: list[str], tvc_rows: list[str], oe_rows: list[str]) -> dict:
    prompt = (
        f"Pieejamās pozīcijas:\n"
        f"- Ieņēmumi (S): {s_rows}\n"
        f"- Mainīgās izmaksas (TVC): {tvc_rows}\n"
        f"- Uzturēšanas izmaksas (OE): {oe_rows}\n\n"
        f"Lietotāja situācijas apraksts: \"{user_text}\"\n\n"
        "Atgriez TIKAI tīru JSON objektu šādā formātā:\n"
        "{\n"
        '  "pct_changes": {"pozīcijas_nosaukums": procents_skaitlis},\n'
        '  "eur_changes": {"pozīcijas_nosaukums": eiro_izmaiņa_kopā_skaitlis},\n'
        '  "new_product": {"enabled": true/false, "name": "nosaukums", "s_eur": 0, "tvc_mat_eur": 0, "tvc_deliv_eur": 0},\n'
        '  "summary": "1 laipns un skaidrs teikums latviski par to, kādas izmaiņas tika piemērotas simulācijā."\n'
        "}"
    )
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    m = re.search(r"\{.*\}", raw_txt, re.S)
    if m:
        return json.loads(m.group(0))
    return {}


# ------------------------------------------------------------------ AUTOMĀTISKA SESIJAS ATJAUNOŠANA NO URL MARĶIERA
if not S.get("authenticated", False):
    qp_user = st.query_params.get("u", "")
    qp_tok = st.query_params.get("t", "")
    if qp_user and qp_tok:
        valid_tok, is_adm_tok = _verify_auth_token(qp_user, qp_tok)
        if valid_tok:
            S.authenticated = True
            S.is_admin = is_adm_tok
            S.active_profile = "admin" if is_adm_tok else qp_user.strip()
            S.view_mode = "admin_panel" if is_adm_tok else "workspace"
            S.nav_page = st.query_params.get("p", "overview")
            S.auto_loaded_latest = False


# ------------------------------------------------------------------ 1. LOGIN
if not S.get("authenticated", False):
    col_l1, col_l2, col_l3 = st.columns([1.2, 1.2, 1.2])
    with col_l2:
        st.write("")
        st.write("")
        with st.container(border=True):
            st.subheader("TOC-FIN Vadības Sistēma")
            st.caption("Finanšu diagnostika un vadības lēmumu pieņemšana")
            temp_store = SupabaseProfileStore()
            with st.form("login_form"):
                login_company = st.text_input("Lietotājvārds / Uzņēmums")
                login_password = st.text_input("Parole", type="password")
                remember_me = st.checkbox("Atcerēties mani", value=True)
                login_submitted = st.form_submit_button("Pieslēgties", type="primary", use_container_width=True)

                if login_submitted:
                    ok, msg, is_adm = temp_store.authenticate(login_company, login_password)
                    if ok:
                        S.authenticated = True
                        S.is_admin = is_adm
                        prof_name = "admin" if is_adm else login_company.strip()
                        S.active_profile = prof_name
                        S.view_mode = "admin_panel" if is_adm else "workspace"
                        S.nav_page = "overview"
                        S.auto_loaded_latest = False
                        if remember_me:
                            st.query_params["u"] = prof_name
                            st.query_params["t"] = _make_auth_token(prof_name, is_adm)
                        st.rerun()
                    else:
                        st.error(msg)
    st.stop()


# ------------------------------------------------------------------ 2. ADMIN PANELIS
admin_store = SupabaseProfileStore()

if S.get("is_admin", False) and S.get("view_mode") == "admin_panel":
    top_a1, top_a2 = st.columns([5, 1], vertical_alignment="center")
    with top_a1:
        st.title("Administratora vadības panelis")
        st.caption("Klientu profili, paroles, limiti un API resursu uzskaite.")
    with top_a2:
        if st.button("Iziet", use_container_width=True):
            st.query_params.clear()
            S.clear()
            st.rerun()

    companies = admin_store.list_all_companies_admin()
    col_adm_left, col_adm_right = st.columns([1.8, 1.2])

    with col_adm_left:
        st.subheader(f"Reģistrētie klientu profili ({len(companies)})")
        if not companies:
            st.info("Vēl nav izveidots neviens klienta profils.")
        else:
            comp_df = pd.DataFrame(companies)[
                ["company_name", "is_active", "usage_count", "monthly_limit", "total_tokens", "total_cost_eur", "accounts_count", "reports_count", "notes"]
            ]
            comp_df.columns = ["Uzņēmums", "Aktīvs", "Izlietots", "Limits", "Tokeni kopā", "Izmaksas (€)", "Konti", "Atsk.", "Piezīmes"]
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("#### Piekļūt klienta profilam")
            c_names = [c["company_name"] for c in companies]
            sel_c1, sel_c2, sel_c3 = st.columns([2, 1, 1])
            with sel_c1:
                target_client = st.selectbox("Izvēlieties klientu", options=c_names, label_visibility="collapsed")
            with sel_c2:
                if st.button("Atvērt vidi", type="primary", use_container_width=True):
                    S.active_profile = target_client
                    S.view_mode = "workspace"
                    S.nav_page = "overview"
                    S.auto_loaded_latest = False
                    S.pop("result", None)
                    S.pop("cls_df", None)
                    S.pop("loaded_report_meta", None)
                    st.rerun()
            with sel_c3:
                if st.button("Dzēst", use_container_width=True):
                    S.confirm_del_client = target_client

            if S.get("confirm_del_client") == target_client:
                st.warning(f"Vai tiešām dzēst profilu **{target_client}**?")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Jā, dzēst", type="primary", use_container_width=True):
                        admin_store.delete_company_admin(target_client)
                        S.pop("confirm_del_client", None)
                        st.rerun()
                with dc2:
                    if st.button("Atcelt", use_container_width=True):
                        S.pop("confirm_del_client", None)
                        st.rerun()

    with col_adm_right:
        st.subheader("Izveidot vai rediģēt klientu")
        with st.form("admin_manage_client_form"):
            f_cname = st.text_input("Lietotājvārds / Uzņēmums", placeholder="piem., SIA Klients")
            f_pw = st.text_input("Parole (atstājiet tukšu, ja nemaināt)", type="password")
            f_limit = st.number_input("Mēneša limits", min_value=1, max_value=10000, value=50, step=5)
            f_active = st.checkbox("Profils ir aktīvs", value=True)
            f_reset = st.checkbox("Nullēt izlietoto skaitītāju un tokenus (0)", value=False)
            f_notes = st.text_input("Piezīmes")
            sub_adm = st.form_submit_button("Saglabāt profilu", type="primary", use_container_width=True)

            if sub_adm:
                ok_a, msg_a = admin_store.create_or_update_company_admin(
                    company_name=f_cname,
                    password=f_pw,
                    monthly_limit=int(f_limit),
                    is_active=f_active,
                    notes=f_notes,
                    reset_usage=f_reset,
                )
                if ok_a:
                    st.success(msg_a)
                    st.rerun()
                else:
                    st.error(msg_a)
    st.stop()


# ------------------------------------------------------------------ 3. PROFILA IELĀDE UN AUTOMĀTISKAIS OVERVIEW
profile = SupabaseProfileStore(company_name=S.active_profile)

if "fy_start_month" not in S:
    S.fy_start_month = 1
if "fy_start_day" not in S:
    S.fy_start_day = 1
if "nav_page" not in S:
    S.nav_page = "overview"
if "hide_top_toolbar" not in S:
    S.hide_top_toolbar = False

all_saved_items = profile.list_reports()
fin_reports = [r for r in all_saved_items if not str(r.get("file_name", "")).startswith("SIM::")]
sim_reports = [r for r in all_saved_items if str(r.get("file_name", "")).startswith("SIM::")]
saved_acc_count = profile.count_saved()

if not S.get("auto_loaded_latest", False):
    S.auto_loaded_latest = True
    if "result" not in S and fin_reports:
        latest_rep_id = fin_reports[0]["id"]
        loaded_init = profile.load_report(latest_rep_id)
        if loaded_init:
            init_res = LoadedReportResult(
                amounts=loaded_init["amounts"],
                main=loaded_init["main"],
                kpi=loaded_init["kpi"],
                row_kinds=loaded_init["row_kinds"],
                reconciliation=loaded_init["reconciliation"],
                by_group={},
            )
            S.result = strip_excluded_rows(init_res)
            S.loaded_report_meta = loaded_init
            S.current_report_id = loaded_init["id"]
            comm_clean, acc_snap = unpack_commentary_and_account_details(loaded_init.get("commentary"))
            S.loaded_account_details = acc_snap
            if comm_clean:
                S.commentary = comm_clean
            else:
                S.pop("commentary", None)


# ------------------------------------------------------------------ ASISTENTA IEKŠĒJAIS LOGS (AR IZMĒRA SLĒDZI, RITINĀŠANU UZ ATBILDES SĀKUMU UN DARBĪBU IZPILDI)
def _render_assistant_popover_content(form_key: str) -> None:
    top_c1, top_c2 = st.columns([1.8, 1.2], vertical_alignment="center")
    with top_c1:
        st.markdown("**TOC-FIN Eksperts**")
    with top_c2:
        st.toggle("Paplašināt logu", key=f"exp_tog_{form_key}", value=bool(S.get("assistant_expanded", False)), on_change=lambda: S.update(assistant_expanded=not S.get("assistant_expanded", False)))

    if "global_chat" not in S:
        S.global_chat = []

    box_h = 430 if S.get("assistant_expanded", False) else 265
    chat_box = st.container(height=box_h)
    with chat_box:
        if not S.global_chat:
            st.caption(
                "Sveiki! Pajautājiet man par uzņēmuma skaitļiem, kas slēpjas zem katras izmaksu grupas, "
                "vai palūdziet sadalīt kādu apakšgrupu sīkāk!"
            )
        else:
            # Ja ir vairākas ziņas, vecākās saliekam kompaktā izvelkamā blokā virs pašreizējās atbildes,
            # lai jaunākā atbilde vienmēr sākas pašā augšā un nav jātin uz augšu un leju!
            if len(S.global_chat) > 2:
                with st.expander(f"Iepriekšējā sarakste ({len(S.global_chat) - 2} ziņas)", expanded=False):
                    for old_m in S.global_chat[:-2]:
                        role_lbl = "Jūs" if old_m["role"] == "user" else "Eksperts"
                        st.markdown(f"**{role_lbl}:** {old_m['content']}")
                        st.markdown("---")

            recent_msgs = S.global_chat[-2:] if len(S.global_chat) >= 2 else S.global_chat
            for idx_m, msg in enumerate(recent_msgs):
                if msg["role"] == "assistant" and idx_m == len(recent_msgs) - 1:
                    st.markdown('<div id="toc-latest-answer-top"></div>', unsafe_allow_html=True)
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Automātiski uzritinām čata konteineru uz jaunākās atbildes SĀKUMU (nevis pašām beigām)
            components.html(
                """
                <script>
                setTimeout(function() {
                    const anchor = window.parent.document.getElementById('toc-latest-answer-top');
                    if (anchor) {
                        anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 80);
                </script>
                """,
                height=0,
            )

    # Ja Asistents ir sagatavojis darbību (piem., grupas sadalīšanu vai navigāciju), parādām apstiprinājuma kartīti
    pending_act = S.get("pending_assistant_action")
    if pending_act:
        with st.container(border=True):
            st.markdown(f"**Apstipriniet darbību:**  \n{pending_act.get('summary', 'Veikt izmaiņas')}")
            if pending_act.get("accounts"):
                st.caption("Konti: " + ", ".join(pending_act["accounts"]))
            ac_b1, ac_b2 = st.columns(2)
            with ac_b1:
                if st.button("Apstiprināt un izpildīt", key=f"confirm_act_{form_key}", type="primary", use_container_width=True):
                    ok_ex, msg_ex = execute_assistant_action(pending_act, profile)
                    S.pop("pending_assistant_action", None)
                    S.global_chat.append({"role": "assistant", "content": f"✅ {msg_ex}"})
                    st.rerun()
            with ac_b2:
                if st.button("Atcelt", key=f"cancel_act_{form_key}", use_container_width=True):
                    S.pop("pending_assistant_action", None)
                    st.rerun()

    with st.form(form_key, clear_on_submit=True):
        q_in = st.text_input("Jautājums", placeholder="Rakstiet savu jautājumu šeit...", label_visibility="collapsed")
        f_c1, f_c2 = st.columns([2.2, 1.2])
        sub_q = f_c1.form_submit_button("Sūtīt", type="primary", use_container_width=True)
        clr_q = f_c2.form_submit_button("Notīrīt", use_container_width=True)
        if sub_q and q_in.strip():
            S.global_chat.append({"role": "user", "content": q_in.strip()})
            try:
                ans, act_obj, t_tok, t_eur = ask_global_assistant(q_in.strip(), profile)
                profile.log_api_usage(t_tok, t_eur)
                if act_obj:
                    S.pending_assistant_action = act_obj
            except Exception as e:
                ans = f"Kļūda: {e}"
            S.global_chat.append({"role": "assistant", "content": ans})
            st.rerun()
        if clr_q:
            S.global_chat = []
            S.pop("pending_assistant_action", None)
            st.rerun()


# ------------------------------------------------------------------ 4. PELDOŠA (STICKY) UN IZCELTA AUGŠĒJĀ RĪKJOSLA
NAV_ITEMS = {
    "overview": "Pārskats",
    "simulation": "Scenāriju simulācija",
    "accounts_explorer": "Kontu struktūra",
    "archive": "Atskaites un Simulācijas",
    "ingest": "Jauns PZA / Klasifikācija",
    "settings": "Profils un Uzstādījumi",
}

active_rep_label = (
    S.loaded_report_meta.get("report_name")
    if S.get("loaded_report_meta")
    else ("Jauns aprēķins" if "result" in S else "Nav izvēlēta")
)

with st.container():
    st.markdown('<div class="sticky-nav-marker"></div>', unsafe_allow_html=True)

    if S.hide_top_toolbar:
        _, col_asist, col_arr = st.columns([6.2, 1.0, 0.35], vertical_alignment="center")
        with col_asist:
            with st.popover("Asistents", use_container_width=True):
                _render_assistant_popover_content("side_agent_form_min")
        with col_arr:
            if st.button("▼", help="Rādīt rīkjoslu", use_container_width=True):
                S.hide_top_toolbar = False
                st.rerun()
    else:
        hdr_left, hdr_asist, hdr_arr = st.columns([5.8, 1.0, 0.35], vertical_alignment="center")
        with hdr_left:
            st.markdown(
                f'<div class="top-brand-bar">'
                f'<span class="top-brand-title">TOC-FIN</span>'
                f'<span class="top-brand-sub">Profils: <b>{S.active_profile}</b> &nbsp;|&nbsp; Atskaite: <b>{active_rep_label}</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with hdr_asist:
            with st.popover("Asistents", use_container_width=True):
                _render_assistant_popover_content("side_agent_form")
        with hdr_arr:
            if st.button("▲", help="Paslēpt rīkjoslu", use_container_width=True):
                S.hide_top_toolbar = True
                st.rerun()

        nav_cols = st.columns([0.95, 1.25, 1.15, 1.35, 1.35, 1.25, 0.65] if not S.get("is_admin") else [0.9, 1.2, 1.1, 1.3, 1.3, 1.2, 0.55, 0.55])
        for idx_n, (n_key, n_label) in enumerate(NAV_ITEMS.items()):
            btn_type = "primary" if S.nav_page == n_key else "secondary"
            if nav_cols[idx_n].button(n_label, key=f"top_nav_{n_key}", type=btn_type, use_container_width=True):
                S.nav_page = n_key
                st.query_params["p"] = n_key
                st.rerun()

        if S.get("is_admin", False):
            if nav_cols[6].button("Admin", use_container_width=True):
                S.view_mode = "admin_panel"
                st.rerun()
            if nav_cols[7].button("Iziet", use_container_width=True):
                st.query_params.clear()
                S.clear()
                st.rerun()
        else:
            if nav_cols[6].button("Iziet", use_container_width=True):
                st.query_params.clear()
                S.clear()
                st.rerun()


# ==============================================================================
# POSMS 1: PĀRSKATS (OVERVIEW)
# ==============================================================================
if S.nav_page == "overview":
    if "result" not in S or S.result is None:
        st.info("Šajā profilā vēl nav nevienas saglabātas atskaites. Pārejiet uz sadaļu **Jauns PZA / Klasifikācija**, lai augšupielādētu pirmo PZA failu.")
        if st.button("Doties uz PZA faila augšupielādi →", type="primary"):
            S.nav_page = "ingest"
            st.rerun()
        st.stop()

    res = strip_excluded_rows(S.result)
    all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
    if not all_months:
        all_months = list(res.amounts.columns)

    months_by_year: dict[int, list[str]] = {}
    for m_col in all_months:
        _, yr = parse_period_month_year(m_col)
        if yr:
            months_by_year.setdefault(yr, []).append(m_col)

    fin_preset_opts = [f"Visi mēneši ({len(all_months)} mēn.)"]
    for yr in sorted(months_by_year.keys()):
        fin_preset_opts.append(f"{yr}. gads ({len(months_by_year[yr])} mēn.)")
    if len(all_months) > 12:
        fin_preset_opts.append("Pēdējie 12 mēneši")
    if len(all_months) >= 6:
        fin_preset_opts.append("Pirmie 6 mēneši")
        fin_preset_opts.append("Pēdējie 6 mēneši")
    fin_preset_opts.append("Manuāla mēnešu izlase")

    if "fin_period_preset" not in S or S.fin_period_preset not in fin_preset_opts:
        S.fin_period_preset = fin_preset_opts[0]
    if "fin_months_active" not in S or not set(S.fin_months_active).issubset(set(all_months)):
        S.fin_months_active = list(all_months)

    def _on_fin_preset_change():
        p = S.fin_period_preset
        if p.startswith("Visi mēneši"):
            S.fin_months_active = list(all_months)
        elif p == "Pēdējie 12 mēneši":
            S.fin_months_active = list(all_months[-12:])
        elif p == "Pirmie 6 mēneši":
            S.fin_months_active = list(all_months[:6])
        elif p == "Pēdējie 6 mēneši":
            S.fin_months_active = list(all_months[-6:])
        else:
            for yr_k, m_list in months_by_year.items():
                if p.startswith(f"{yr_k}. gads"):
                    S.fin_months_active = list(m_list)
                    break

    def _on_fin_months_change():
        cur = S.fin_months_active
        if cur == list(all_months):
            S.fin_period_preset = fin_preset_opts[0]
        else:
            S.fin_period_preset = "Manuāla mēnešu izlase"

    bar_c1, bar_c2, bar_c3, _ = st.columns([1.4, 1.2, 1.1, 2.3], vertical_alignment="bottom")
    with bar_c1:
        st.selectbox(
            "Periods pārskatam",
            options=fin_preset_opts,
            key="fin_period_preset",
            on_change=_on_fin_preset_change,
        )

    cur_fin_months = S.get("fin_months_active", all_months)
    with bar_c2:
        with st.popover(f"Mēnešu izlase ({len(cur_fin_months)}/{len(all_months)})", use_container_width=True):
            st.caption("Atzīmējiet mēnešus, kurus iekļaut tabulā:")
            st.multiselect(
                "Izvēlētie mēneši",
                options=all_months,
                key="fin_months_active",
                on_change=_on_fin_months_change,
                label_visibility="collapsed",
            )

    fin_selected_months = [m for m in all_months if m in S.get("fin_months_active", all_months)]
    if not fin_selected_months:
        st.warning("Izvēlieties vismaz vienu mēnesi finanšu pārskata tabulai.")
        st.stop()

    view_res = filter_result_by_months(res, fin_selected_months, all_months)

    with bar_c3:
        with st.popover("Lejupielādēt Excel ▾", use_container_width=True):
            st.caption("Izvēlieties Excel faila veidu:")
            try:
                xl_f, xl_v = io.BytesIO(), io.BytesIO()
                build_excel(view_res, xl_f, formulas=True)
                build_excel(view_res, xl_v, formulas=False)
                st.download_button(
                    "Ar Excel formulām (.xlsx)",
                    xl_f.getvalue(),
                    "TOC_Finansu_Diagnostika_formulas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )
                st.download_button(
                    "Tikai vērtības (.xlsx)",
                    xl_v.getvalue(),
                    "TOC_Finansu_Diagnostika_vertibas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                xl_buf = io.BytesIO()
                with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                    view_res.amounts.to_excel(writer, sheet_name="TOC_Summas")
                    view_res.kpi.to_excel(writer, sheet_name="KPI")
                st.download_button(
                    "Lejupielādēt (.xlsx)",
                    xl_buf.getvalue(),
                    "TOC_Saglabata_Atskaite.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )

    st.markdown(
        render_unified_toc_tables_html(
            view_res.amounts,
            view_res.main,
            view_res.kpi,
            view_res.row_kinds,
            title_main="Finanšu pārskata tabula",
            title_kpi="Produktivitātes un atdeves rādītāji",
            fy_start_month=S.get("fy_start_month", 1),
            show_cumulative=True,
        ),
        unsafe_allow_html=True,
    )

    render_overview_charts(view_res.amounts, view_res.kpi, fy_start_month=S.get("fy_start_month", 1))

    if getattr(res, "reconciliation", None) is not None and not res.reconciliation.empty:
        with st.expander("Kontrole: Salīdzinājums ar PZA"):
            st.dataframe(res.reconciliation.style.format(fmt_num), use_container_width=True)

    st.markdown("---")
    rec_c1, rec_c2 = st.columns([3, 1], vertical_alignment="center")
    with rec_c1:
        st.subheader(f"Vadības kopsavilkums un ieteikumi ({len(fin_selected_months)} mēn. periodam)")
    with rec_c2:
        regen_btn = st.button("Pieprasīt ieteikumus par periodu", type="primary", use_container_width=True)

    cum_df_md, _, _ = compute_cumulative_toc_df(view_res.amounts, fy_start_month=S.get("fy_start_month", 1))
    tables_md = (
        f"Izvēlētais periods: {', '.join(fin_selected_months)}\n\n"
        + view_res.amounts.round(0).to_markdown()
        + "\n\n"
        + view_res.kpi.round(3).to_markdown()
        + "\n\nUzkrājošie rādītāji:\n"
        + cum_df_md.round(0).to_markdown()
    )

    knowledge_context = f"Papildu TOC-FIN zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
    COMMENT_SYSTEM = (
        "Tu esi laipns, atbalstošs un augsta līmeņa TOC-FIN (Ierobežojumu teorijas un Caurlaidības grāmatvedības) finanšu eksperts.\n"
        f"{TOC_CORE_EXPERT_KNOWLEDGE}\n"
        f"{knowledge_context}"
        "Sagatavo strukturētu, dziļu un praktisku vadības kopsavilkumu par izvēlēto periodu latviešu valodā:\n"
        "1) **Kopējā TOC diagnostika**: Caurlaidība (T un T%), Uzturēšanas izmaksas (OE — vai tām ir pakāpienu raksturs vai nepamatots pieaugums), Produktivitāte (PR = T/OE) un uzkrājošais lūzuma punkts (kad uzkrājošais T pārsniedz OE un sākas sviras efekts).\n"
        "2) **Mēnešu dinamika un cēloņsakarības**: kuri mēneši ģenerēja augstāko caurlaidību un kur radās novirzes.\n"
        "3) **Konkrēti TOC vadības ieteikumi (pēc prioritātes T ↑ -> I ↓ -> OE ↓ un ΔT - ΔOE > 0)**: kā palielināt plūsmu un caurlaidību, un kādus scenārijus vērts pārbaudīt Scenāriju simulācijā."
    )

    if regen_btn:
        with st.spinner("Sagatavo vadības ieteikumus izvēlētajam periodam..."):
            try:
                r = get_client().messages.create(
                    model=MODEL,
                    max_tokens=2500,
                    system=COMMENT_SYSTEM,
                    messages=[{"role": "user", "content": f"Tabulas:\n\n{tables_md}\n\nUzraksti analīzi un konkrētus ieteikumus šim periodam."}],
                )
                S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                u_c = getattr(r, "usage", None)
                tok_c = (getattr(u_c, "input_tokens", 800) + getattr(u_c, "output_tokens", 800)) if u_c else 1600
                profile.log_api_usage(tok_c, tok_c * 0.000009)
                if S.get("current_report_id"):
                    packed_c = pack_commentary_with_account_details(S.commentary, S.get("loaded_account_details"))
                    profile.update_report_commentary(S.current_report_id, packed_c)
                    st.toast("Vadības kopsavilkums saglabāts pie atskaites!")
                st.rerun()
            except Exception as e:
                st.error(f"Neizdevās uzģenerēt ieteikumus: {e}")

    if S.get("commentary"):
        with st.container(border=True):
            st.markdown(S.commentary)
            with st.expander("Rediģēt vai papildināt vadības kopsavilkumu un saglabāt", expanded=False):
                edited_comm = st.text_area("Kopsavilkuma teksts", value=S.commentary, height=240)
                if st.button("Saglabāt kopsavilkumu atskaitē", type="primary"):
                    S.commentary = edited_comm
                    if S.get("current_report_id"):
                        packed_c = pack_commentary_with_account_details(edited_comm, S.get("loaded_account_details"))
                        profile.update_report_commentary(S.current_report_id, packed_c)
                        if S.get("loaded_report_meta"):
                            S.loaded_report_meta["commentary"] = packed_c
                        st.success("Vadības kopsavilkums saglabāts datubāzē!")
                    else:
                        st.toast("Kopsavilkums atjaunināts sesijā.")
                    st.rerun()
    else:
        st.caption("Nospiediet pogu **Pieprasīt ieteikumus par periodu**, lai saņemtu un saglabātu AI diagnostiku.")


# ==============================================================================
# POSMS 2: SCENĀRIJU SIMULĀCIJA (ESOŠĀ DARBĪBA KOPĀ VIRS APAKŠKONTIEM + JAUNS PRODUKTS APAKŠĀ)
# ==============================================================================
elif S.nav_page == "simulation":
    st.subheader("Scenāriju simulācija un Jauns projekts")
    if "result" not in S or S.result is None:
        st.info("Vispirms ielādējiet vai izveidojiet finanšu atskaiti, lai veiktu simulāciju.")
        st.stop()

    res = strip_excluded_rows(S.result)
    all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c)]
    if not all_months:
        all_months = list(res.amounts.columns)

    months_by_year: dict[int, list[str]] = {}
    for m_col in all_months:
        _, yr = parse_period_month_year(m_col)
        if yr:
            months_by_year.setdefault(yr, []).append(m_col)

    s_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "S_CHILD"]
    tvc_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "TVC_CHILD"]
    oe_children = [r for r in res.amounts.index if res.row_kinds.get(r) == "OE_CHILD"]

    sim_preset_opts = [f"Visi mēneši ({len(all_months)} mēn.)"]
    for yr in sorted(months_by_year.keys()):
        sim_preset_opts.append(f"{yr}. gads ({len(months_by_year[yr])} mēn.)")
    if len(all_months) >= 6:
        sim_preset_opts.append("Pirmie 6 mēneši")
        sim_preset_opts.append("Pēdējie 6 mēneši")
    sim_preset_opts.append("Manuāla mēnešu izlase")

    if "sim_preset_sel" not in S or S.sim_preset_sel not in sim_preset_opts:
        S.sim_preset_sel = sim_preset_opts[0]
    if "sim_months_active" not in S or not set(S.sim_months_active).issubset(set(all_months)):
        S.sim_months_active = list(all_months)
    if "sim_situation_text" not in S:
        S.sim_situation_text = ""

    def _on_sim_preset_change():
        p = S.sim_preset_sel
        if p.startswith("Visi mēneši"):
            S.sim_months_active = list(all_months)
        elif p == "Pirmie 6 mēneši":
            S.sim_months_active = list(all_months[:6])
        elif p == "Pēdējie 6 mēneši":
            S.sim_months_active = list(all_months[-6:])
        else:
            for yr_k, m_list in months_by_year.items():
                if p.startswith(f"{yr_k}. gads"):
                    S.sim_months_active = list(m_list)
                    break

    def _on_sim_months_change():
        cur = S.sim_months_active
        if cur == list(all_months):
            S.sim_preset_sel = sim_preset_opts[0]
        else:
            S.sim_preset_sel = "Manuāla mēnešu izlase"

    p_col1, p_col2, p_col3, _ = st.columns([1.3, 1.3, 1.2, 2.2], vertical_alignment="bottom")
    with p_col1:
        st.selectbox(
            "Periods simulācijai",
            options=sim_preset_opts,
            key="sim_preset_sel",
            on_change=_on_sim_preset_change,
        )

    cur_sim_months = S.get("sim_months_active", all_months)
    with p_col2:
        with st.popover(f"Mēnešu izlase ({len(cur_sim_months)}/{len(all_months)})", use_container_width=True):
            st.multiselect(
                "Mēneši simulācijā",
                options=all_months,
                key="sim_months_active",
                on_change=_on_sim_months_change,
                label_visibility="collapsed",
            )
    with p_col3:
        if st.button("Notīrīt visas izmaiņas (uz 0)", use_container_width=True):
            for r_nm in s_children + tvc_children + oe_children:
                S[f"sim_pct_{r_nm}"] = 0.0
                S[f"sim_eur_{r_nm}"] = 0.0
            S["np_enabled"] = False
            S["np_s_eur"] = 0.0
            S["np_mat_eur"] = 0.0
            S["np_deliv_eur"] = 0.0
            S["sim_situation_text"] = ""
            S["last_audio_hash"] = ""
            S["sim_ed_ver"] = S.get("sim_ed_ver", 0) + 1
            S["sim_audio_ver"] = S.get("sim_audio_ver", 0) + 1
            st.rerun()

    selected_months = [m for m in all_months if m in S.get("sim_months_active", all_months)]
    if not selected_months:
        st.warning("Izvēlieties vismaz vienu mēnesi.")
        st.stop()

    base_sub = res.amounts[selected_months].copy().astype(float)
    base_sum = base_sub.sum(axis=1)

    with st.expander("Aprakstīt vai ierunāt situāciju (AI automātiski saliks simulācijas skaitļus)", expanded=True):
        d_col1, d_col2 = st.columns([2.2, 1], vertical_alignment="top")
        with d_col2:
            audio_val = None
            if hasattr(st, "audio_input"):
                audio_val = st.audio_input(
                    "Balss ieraksts (pārlūka mikrofons)",
                    key=f"sim_audio_in_{S.get('sim_audio_ver', 0)}",
                )
            else:
                st.text_input("Balss ieraksts (pārlūka mikrofons)", value="Izmantojiet diktofonu", disabled=True)

            if audio_val is not None:
                a_bytes = audio_val.getvalue()
                a_hash = hashlib.md5(a_bytes).hexdigest()
                if S.get("last_audio_hash") != a_hash:
                    S["last_audio_hash"] = a_hash
                    with st.spinner("Atpazīst balss ierakstu un piemēro simulācijai..."):
                        transcribed, err_msg = transcribe_wav_bytes_lv(a_bytes)
                        if transcribed:
                            cur_t = S.get("sim_situation_text", "").strip()
                            full_t = (cur_t + " " + transcribed).strip()
                            S["sim_situation_text"] = full_t
                            try:
                                parsed = parse_simulation_dialog_with_claude(
                                    full_t, s_children, tvc_children, oe_children
                                )
                                for k_p, v_p in parsed.get("pct_changes", {}).items():
                                    if k_p in s_children + tvc_children + oe_children:
                                        S[f"sim_pct_{k_p}"] = float(v_p)
                                for k_e, v_e in parsed.get("eur_changes", {}).items():
                                    if k_e in s_children + tvc_children + oe_children:
                                        S[f"sim_eur_{k_e}"] = float(v_e)
                                np_d = parsed.get("new_product", {})
                                if np_d.get("enabled"):
                                    S["np_enabled"] = True
                                    S["np_name"] = str(np_d.get("name") or "Jauns produkts / projekts")
                                    S["np_s_eur"] = float(np_d.get("s_eur") or 0.0)
                                    S["np_mat_eur"] = float(np_d.get("tvc_mat_eur") or 0.0)
                                    S["np_deliv_eur"] = float(np_d.get("tvc_deliv_eur") or 0.0)
                                S["sim_ed_ver"] = S.get("sim_ed_ver", 0) + 1
                            except Exception:
                                pass
                            st.rerun()
                        elif err_msg:
                            st.caption(f"ℹ️ {err_msg}")

        with d_col1:
            sim_text_prompt = st.text_area(
                "Situācijas apraksts",
                key="sim_situation_text",
                placeholder="Piemēram: Paceļam pārdošanu par 10%, mārketings +15 000 EUR, un palaižam jaunu produktu ar 50 000 EUR ieņēmumiem un 20 000 EUR materiālu izmaksām...",
                height=88,
            )

        if st.button("Piemērot situāciju simulācijai", type="primary"):
            active_prompt = str(S.get("sim_situation_text") or "").strip()
            if not active_prompt and audio_val is not None:
                with st.spinner("Atpazīst balss ierakstu..."):
                    transcribed, err_msg = transcribe_wav_bytes_lv(audio_val.getvalue())
                    if transcribed:
                        active_prompt = transcribed
                        S["sim_situation_text"] = transcribed
                    elif err_msg:
                        st.warning(err_msg)

            if active_prompt:
                with st.spinner("Analizē situāciju un aizpilda simulācijas tabulu..."):
                    try:
                        parsed = parse_simulation_dialog_with_claude(
                            active_prompt, s_children, tvc_children, oe_children
                        )
                        for k_p, v_p in parsed.get("pct_changes", {}).items():
                            if k_p in s_children + tvc_children + oe_children:
                                S[f"sim_pct_{k_p}"] = float(v_p)
                        for k_e, v_e in parsed.get("eur_changes", {}).items():
                            if k_e in s_children + tvc_children + oe_children:
                                S[f"sim_eur_{k_e}"] = float(v_e)
                        np_d = parsed.get("new_product", {})
                        if np_d.get("enabled"):
                            S["np_enabled"] = True
                            S["np_name"] = str(np_d.get("name") or "Jauns produkts / projekts")
                            S["np_s_eur"] = float(np_d.get("s_eur") or 0.0)
                            S["np_mat_eur"] = float(np_d.get("tvc_mat_eur") or 0.0)
                            S["np_deliv_eur"] = float(np_d.get("tvc_deliv_eur") or 0.0)
                        S["sim_ed_ver"] = S.get("sim_ed_ver", 0) + 1
                        if parsed.get("summary"):
                            st.success(parsed["summary"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Neizdevās apstrādāt aprakstu: {e}")

    st.markdown("---")
    st.markdown("#### 1. Esošās darbības izmaiņas (% un/vai € kopā izvēlētajā periodā)")
    st.caption("Dubultklikšķis uz šūnas kolonnā **Izmaiņa %** vai **Izmaiņa +/- €**, lai ievadītu izmaiņu.")

    pct_changes: dict[str, float] = {}
    eur_changes: dict[str, float] = {}
    sim_ver = S.get("sim_ed_ver", 0)

    def _render_compact_sim_editor(group_rows: list[str], key_prefix: str) -> None:
        for r_nm in group_rows:
            if f"sim_pct_{r_nm}" not in S:
                S[f"sim_pct_{r_nm}"] = 0.0
            if f"sim_eur_{r_nm}" not in S:
                S[f"sim_eur_{r_nm}"] = 0.0

        df_in = pd.DataFrame(
            {
                "Pozīcija": group_rows,
                "Bāze (€)": [round(float(base_sum.get(r, 0.0)), 0) for r in group_rows],
                "Izmaiņa %": [float(S.get(f"sim_pct_{r}", 0.0)) for r in group_rows],
                "Izmaiņa +/- €": [float(S.get(f"sim_eur_{r}", 0.0)) for r in group_rows],
            }
        )
        tbl_h = (len(df_in) + 1) * 35 + 10
        edited_df = st.data_editor(
            df_in,
            hide_index=True,
            use_container_width=True,
            height=tbl_h,
            disabled=["Pozīcija", "Bāze (€)"],
            key=f"compact_sim_{key_prefix}_{sim_ver}",
            column_config={
                "Pozīcija": st.column_config.TextColumn("Pozīcija", width="medium"),
                "Bāze (€)": st.column_config.NumberColumn("Bāze (€)", format="%.0f €", width="small"),
                "Izmaiņa %": st.column_config.NumberColumn("Izmaiņa %", format="%.1f %%", step=1.0, width="small"),
                "Izmaiņa +/- €": st.column_config.NumberColumn("Izmaiņa +/- €", format="%.0f €", step=500.0, width="small"),
            },
        )
        for _, row_ed in edited_df.iterrows():
            r_nm = row_ed["Pozīcija"]
            p_v = 0.0 if pd.isna(row_ed["Izmaiņa %"]) else float(row_ed["Izmaiņa %"])
            e_v = 0.0 if pd.isna(row_ed["Izmaiņa +/- €"]) else float(row_ed["Izmaiņa +/- €"])
            S[f"sim_pct_{r_nm}"] = p_v
            S[f"sim_eur_{r_nm}"] = e_v
            pct_changes[r_nm] = p_v
            eur_changes[r_nm] = e_v

    sim_left_col, sim_right_col = st.columns([1, 1.15])
    with sim_left_col:
        st.markdown('<div class="sim-col-header-s">S — Ieņēmumi (Sales)</div>', unsafe_allow_html=True)
        _render_compact_sim_editor(s_children, "S")
        st.write("")
        st.markdown('<div class="sim-col-header-tvc">TVC — Mainīgās izmaksas</div>', unsafe_allow_html=True)
        _render_compact_sim_editor(tvc_children, "TVC")
    with sim_right_col:
        st.markdown('<div class="sim-col-header-oe">Uzturēšanas izmaksas (OE)</div>', unsafe_allow_html=True)
        _render_compact_sim_editor(oe_children, "OE")

    st.markdown("---")
    st.markdown("#### 2. Jauns produkts / projekts (tiek pieskaitīts kopējai simulācijai kā atsevišķas apakšrindas)")
    if "np_enabled" not in S:
        S["np_enabled"] = False
    if "np_name" not in S:
        S["np_name"] = "Jauns produkts / projekts"
    if "np_s_eur" not in S:
        S["np_s_eur"] = 0.0
    if "np_mat_eur" not in S:
        S["np_mat_eur"] = 0.0
    if "np_deliv_eur" not in S:
        S["np_deliv_eur"] = 0.0

    np_enable = st.checkbox("Iekļaut jaunu produktu / projektu kopējā simulācijā", key="np_enabled")
    np_s_val, np_mat_val, np_deliv_val = 0.0, 0.0, 0.0
    np_clean_title = str(S.get("np_name") or "Jauns produkts / projekts").strip()

    if np_enable:
        with st.container(border=True):
            np_c1, np_c2, np_c3, np_c4 = st.columns([1.4, 1, 1, 1])
            with np_c1:
                np_title = st.text_input("Produkta / projekta nosaukums", key="np_name")
                np_clean_title = np_title.strip() or "Jauns produkts / projekts"
                np_input_mode = st.radio(
                    "TVC ievades veids",
                    options=["Summas (€)", "% no produkta S"],
                    horizontal=True,
                    key="np_mode_radio",
                )
            with np_c2:
                np_s_val = st.number_input("Produkta S (Ieņēmumi, €)", min_value=0.0, step=1000.0, format="%.0f", key="np_s_eur")
            with np_c3:
                if np_input_mode == "Summas (€)":
                    np_mat_val = st.number_input("Izejvielas un materiāli (€)", min_value=0.0, step=500.0, format="%.0f", key="np_mat_eur")
                else:
                    mat_pct = st.number_input("Izejvielas un materiāli (% no S)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
                    np_mat_val = np_s_val * (mat_pct / 100.0)
            with np_c4:
                if np_input_mode == "Summas (€)":
                    np_deliv_val = st.number_input("Preču piegādes izdevumi (€)", min_value=0.0, step=500.0, format="%.0f", key="np_deliv_eur")
                else:
                    deliv_pct = st.number_input("Preču piegādes izdevumi (% no S)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
                    np_deliv_val = np_s_val * (deliv_pct / 100.0)

            np_tvc_tot = np_mat_val + np_deliv_val
            np_t_tot = np_s_val - np_tvc_tot
            np_t_pct = (np_t_tot / np_s_val * 100.0) if np_s_val > 0 else 0.0
            st.info(
                f"**{np_clean_title} struktūra:** S = **{fmt_num(np_s_val)} €** | "
                f"TVC kopā = **{fmt_num(np_tvc_tot)} €** (Materiāli: {fmt_num(np_mat_val)} €, Piegāde: {fmt_num(np_deliv_val)} €) | "
                f"Produkta Caurlaidība **T = {fmt_num(np_t_tot)} € ({np_t_pct:.0f}%)**"
            )

    sim_sum = base_sum.copy()
    sim_row_kinds = dict(res.row_kinds)

    for rname in s_children + tvc_children + oe_children:
        if rname in sim_sum.index:
            pct_v = pct_changes.get(rname, 0.0)
            eur_v = eur_changes.get(rname, 0.0)
            sim_sum.loc[rname] = base_sum.loc[rname] * (1.0 + pct_v / 100.0) + eur_v

    old_s_base = float(base_sum.loc[s_children].sum()) if s_children else float(base_sum.get("S", 0.0))
    old_s_sim = float(sim_sum.loc[s_children].sum()) if s_children else float(sim_sum.get("S", 0.0))

    old_tvc_base = float(base_sum.loc[tvc_children].sum()) if tvc_children else float(base_sum.get("TVC", 0.0))
    old_tvc_sim = float(sim_sum.loc[tvc_children].sum()) if tvc_children else float(sim_sum.get("TVC", 0.0))

    old_t_base = old_s_base - old_tvc_base
    old_t_sim = old_s_sim - old_tvc_sim

    np_tvc_total = np_mat_val + np_deliv_val
    np_t_total = np_s_val - np_tvc_total
    has_active_new_product = bool(np_enable and (np_s_val != 0 or np_tvc_total != 0))

    # Nosaukumi bez enter (↳) zīmēm!
    old_s_subtotal_lbl = "Esošā darbība kopā (S)"
    np_s_row_lbl = f"{np_clean_title} (S)"

    old_tvc_subtotal_lbl = "Esošā darbība kopā (TVC)"
    np_tvc_row_lbl = f"{np_clean_title} (TVC)"

    old_t_row_lbl = "Esošā darbība kopā (T)"
    np_t_row_lbl = f"{np_clean_title} (T)"

    s_display_rows: list[str] = []
    tvc_display_rows: list[str] = []
    t_display_rows: list[str] = []

    if has_active_new_product:
        # 1) S bloks: Ja ir vairāki vecie S apakškonti, VIRS tiem liekam "Esošā darbība kopā (S)", tad vecos apakškontus, un apakšā Jauno produktu
        if len(s_children) > 1:
            base_sum.loc[old_s_subtotal_lbl] = old_s_base
            sim_sum.loc[old_s_subtotal_lbl] = old_s_sim
            sim_row_kinds[old_s_subtotal_lbl] = "S_SUBTOTAL"
            s_display_rows.append(old_s_subtotal_lbl)
        s_display_rows.extend(s_children)

        base_sum.loc[np_s_row_lbl] = 0.0
        sim_sum.loc[np_s_row_lbl] = np_s_val
        sim_row_kinds[np_s_row_lbl] = "S_CHILD"
        s_display_rows.append(np_s_row_lbl)

        # 2) TVC bloks: VIRS apakškontiem (Izejvielas, Piegāde) liekam "Esošā darbība kopā (TVC)", tad abus apakškontus, un apakšā Jauno produktu!
        if len(tvc_children) > 1:
            base_sum.loc[old_tvc_subtotal_lbl] = old_tvc_base
            sim_sum.loc[old_tvc_subtotal_lbl] = old_tvc_sim
            sim_row_kinds[old_tvc_subtotal_lbl] = "TVC_SUBTOTAL"
            tvc_display_rows.append(old_tvc_subtotal_lbl)
        tvc_display_rows.extend(tvc_children)

        base_sum.loc[np_tvc_row_lbl] = 0.0
        sim_sum.loc[np_tvc_row_lbl] = np_tvc_total
        sim_row_kinds[np_tvc_row_lbl] = "TVC_CHILD"
        tvc_display_rows.append(np_tvc_row_lbl)

        # 3) T bloks: zem kopējā T vispirms "Esošā darbība kopā (T)" (pus-bold) un zem tās "Jauns produkts / projekts (T)"
        base_sum.loc[old_t_row_lbl] = old_t_base
        sim_sum.loc[old_t_row_lbl] = old_t_sim
        sim_row_kinds[old_t_row_lbl] = "T_SUBTOTAL"
        t_display_rows.append(old_t_row_lbl)

        base_sum.loc[np_t_row_lbl] = 0.0
        sim_sum.loc[np_t_row_lbl] = np_t_total
        sim_row_kinds[np_t_row_lbl] = "T_CHILD"
        t_display_rows.append(np_t_row_lbl)
    else:
        s_display_rows = list(s_children)
        tvc_display_rows = list(tvc_children)

    sim_sum.loc["S"] = old_s_sim + (np_s_val if has_active_new_product else 0.0)
    sim_sum.loc["TVC"] = old_tvc_sim + (np_tvc_total if has_active_new_product else 0.0)
    sim_sum.loc["T"] = sim_sum.loc["S"] - sim_sum.loc["TVC"]
    if oe_children:
        sim_sum.loc["OE"] = sim_sum.loc[oe_children].sum()
    sim_sum.loc["NP"] = sim_sum.loc["T"] - sim_sum.loc["OE"]

    ordered_sim_rows = (
        ["S"]
        + s_display_rows
        + ["TVC"]
        + tvc_display_rows
        + ["T"]
        + t_display_rows
        + ["OE"]
        + oe_children
        + ["NP"]
    )
    ordered_sim_rows = [r for r in ordered_sim_rows if r in sim_sum.index]
    base_sum = base_sum.loc[ordered_sim_rows]
    sim_sum = sim_sum.loc[ordered_sim_rows]
    diff_sum = sim_sum - base_sum

    summary_amounts = pd.DataFrame(
        {
            "Bāze KOPĀ": base_sum,
            "Simulācija KOPĀ": sim_sum,
            "Δ Izmaiņa (€)": diff_sum,
        },
        index=ordered_sim_rows,
    )

    sim_main = {}
    for c in summary_amounts.columns:
        sim_main[c] = summary_amounts[c]
        pct_col_series = pd.Series(np.nan, index=ordered_sim_rows, dtype=float)
        if c == "Δ Izmaiņa (€)":
            for r_k in ordered_sim_rows:
                b_v = base_sum.loc[r_k]
                d_v = diff_sum.loc[r_k]
                if b_v != 0:
                    pct_col_series.loc[r_k] = d_v / abs(b_v)
                elif has_active_new_product and r_k in (np_tvc_row_lbl, np_t_row_lbl) and np_s_val > 0:
                    pct_col_series.loc[r_k] = d_v / np_s_val
                elif has_active_new_product and r_k == np_s_row_lbl and np_s_val > 0:
                    pct_col_series.loc[r_k] = 1.0
        else:
            s_col_total = summary_amounts.loc["S", c]
            s_col_old = old_s_base if c == "Bāze KOPĀ" else old_s_sim
            for r_k in ordered_sim_rows:
                val_k = summary_amounts.loc[r_k, c]
                if has_active_new_product and c == "Simulācija KOPĀ" and r_k in (np_tvc_row_lbl, np_t_row_lbl) and np_s_val > 0:
                    pct_col_series.loc[r_k] = val_k / np_s_val
                elif has_active_new_product and r_k in (old_tvc_subtotal_lbl, old_t_row_lbl) and s_col_old > 0:
                    pct_col_series.loc[r_k] = val_k / s_col_old
                elif s_col_total:
                    pct_col_series.loc[r_k] = val_k / s_col_total
        sim_main[f"%|{c}"] = pct_col_series

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

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    ds_val, dt_val, doe_val, dnp_val = diff_sum["S"], diff_sum["T"], diff_sum["OE"], diff_sum["NP"]

    m1.metric("Simulētais S", f"{fmt_num(sim_sum['S'])} €", delta=f"{ds_val:+,.0f} €".replace(",", " "))
    m2.metric("Simulētais T", f"{fmt_num(sim_sum['T'])} €", delta=f"{dt_val:+,.0f} €".replace(",", " "))
    m3.metric("Simulētās Uzturēšanas izmaksas", f"{fmt_num(sim_sum['OE'])} €", delta=f"{doe_val:+,.0f} €".replace(",", " "), delta_color="inverse")
    m4.metric("Simulētā Neto Peļņa (NP)", f"{fmt_num(sim_sum['NP'])} €", delta=f"{dnp_val:+,.0f} €".replace(",", " "))

    st.markdown(
        render_unified_toc_tables_html(
            summary_amounts,
            sim_main_df,
            sim_kpi_df,
            sim_row_kinds,
            title_main=f"Simulētā tabula ({len(selected_months)} mēn.)",
            title_kpi="Simulētie produktivitātes rādītāji",
            grey_cols={"Bāze KOPĀ"},
            fy_start_month=S.get("fy_start_month", 1),
            show_cumulative=False,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    sv_c1, sv_c2 = st.columns([2, 1], vertical_alignment="bottom")
    with sv_c1:
        sim_save_name = st.text_input(
            "Simulācijas nosaukums saglabāšanai",
            value=f"Simulācija ({datetime.now().strftime('%d.%m.%Y %H:%M')})",
        )
    with sv_c2:
        if st.button("Saglabāt simulāciju arhīvā", type="primary", use_container_width=True):
            sim_payload_json = json.dumps(
                {
                    "pct_changes": pct_changes,
                    "eur_changes": eur_changes,
                    "situation_description": str(S.get("sim_situation_text") or "").strip(),
                    "selected_months": selected_months,
                    "new_product": {
                        "enabled": np_enable,
                        "name": S.get("np_name", ""),
                        "s_eur": np_s_val,
                        "tvc_mat_eur": np_mat_val,
                        "tvc_deliv_eur": np_deliv_val,
                    },
                },
                ensure_ascii=False,
            )
            ok_s, _ = profile.save_report(
                report_name=sim_save_name.strip() or "Simulācija",
                file_name="SIM::SCENARIO",
                res_obj=res,
                commentary=sim_payload_json,
            )
            if ok_s:
                st.toast("Simulācija un situācijas apraksts saglabāts arhīvā!")
                st.rerun()
            else:
                st.error(f"Neizdevās saglabāt simulāciju: {profile.last_error}")


# ==============================================================================
# POSMS 3: KONTU STRUKTŪRA
# ==============================================================================
elif S.nav_page == "accounts_explorer":
    st.subheader("Kontu struktūra un PZA mēnešu skaitļu kontrole")
    st.caption(
        "Pārskatāms 3 līmeņu izklājums: **1. Lielā TOC grupa** → **2. Apakšgrupa (rinda finanšu tabulā)** → "
        "**3. Konkrētā PZA pozīcija ar nolasītajiem mēnešu skaitļiem**."
    )

    with st.expander("Pievienot jaunu apakšgrupu (sadalīt izmaksas vai ieņēmumus sīkāk)", expanded=False):
        ns1, ns2, ns3 = st.columns([2, 1.5, 1], vertical_alignment="bottom")
        with ns1:
            custom_sub_lbl = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Pārējās administrācijas izmaksas", key="exp_new_sub_lbl")
        with ns2:
            custom_sub_main = st.selectbox("Lielā grupa", options=["Uzturēšanas izmaksas (OE)", "TVC (Mainīgās izmaksas)", "S (Ieņēmumi)"], key="exp_new_sub_main")
        with ns3:
            if st.button("Pievienot apakšgrupu", use_container_width=True, key="exp_btn_add_sub"):
                if custom_sub_lbl.strip():
                    clean_lbl = custom_sub_lbl.strip()
                    m_c = "S" if custom_sub_main.startswith("S") else ("TVC" if custom_sub_main.startswith("TVC") else "OE")
                    new_k = f"{m_c}_CUSTOM_{hashlib.md5(clean_lbl.encode()).hexdigest()[:6].upper()}"
                    S.setdefault("custom_subgroups", {})[new_k] = (clean_lbl, m_c)
                    register_custom_subgroup(new_k, clean_lbl, m_c)
                    st.toast(f"Apakšgrupa '{clean_lbl}' pievienota.")
                    st.rerun()

    has_live_pza = ("cls_df" in S and S.cls_df is not None and not S.cls_df.empty and "analysis" in S and S.analysis is not None)
    snap_details: list[dict] = S.get("loaded_account_details") or []
    saved_maps = profile.list_saved_mappings()
    res_for_exp = S.get("result")
    res_month_cols = (
        [c for c in res_for_exp.amounts.columns if not ("YTD" in str(c).upper() or "AVG" in str(c).upper())]
        if res_for_exp is not None
        else []
    )

    if not has_live_pza and not snap_details and not saved_maps:
        st.info("Šajā profilā vēl nav saglabātu kontu. Augšupielādējiet PZA failu sadaļā **Jauns PZA / Klasifikācija**.")
        st.stop()

    main_sections = [
        ("S", "S — Ieņēmumi (Sales)", "#FFF200", "#000000"),
        ("TVC", "TVC — Patiesi mainīgās izmaksas", "#C55A11", "#FFFFFF"),
        ("OE", "Uzturēšanas izmaksas (OE)", "#F4B183", "#000000"),
        ("EXCL", "Izslēgtie konti (Amortizācija / Virtuālā nauda)", "#E2E8F0", "#0F172A"),
    ]

    if has_live_pza or snap_details:
        if has_live_pza:
            an = S.analysis
            items_list = build_account_details_snapshot(an.leaves, S.cls_df, an.ex.period_cols)
        else:
            items_list = snap_details

        for m_code, m_title, bg_hex, fg_hex in main_sections:
            grp_keys_for_main = [k for k, (_, mc) in GROUPS.items() if mc == m_code]
            main_items = [it for it in items_list if normalize_group_key(it.get("group", "")) in grp_keys_for_main]
            main_tot = sum((it["total"] if m_code == "S" else abs(it["total"])) for it in main_items)

            st.markdown(
                f'<div style="background-color:{bg_hex}; color:{fg_hex}; padding:8px 14px; border-radius:8px; '
                f'font-weight:700; font-size:15px; margin-top:16px; margin-bottom:8px; border:1.5px solid #1E293B; '
                f'display:flex; justify-content:space-between;">'
                f'<span>1. LĪMENIS: {m_title} ({len(main_items)} konti)</span>'
                f'<span>Kopā periodā: {fmt_num(main_tot)} €</span></div>',
                unsafe_allow_html=True,
            )

            if not main_items:
                st.caption("Šajā grupā nav neviena konta.")
                continue

            for g_key in grp_keys_for_main:
                g_items = [it for it in main_items if normalize_group_key(it.get("group", "")) == g_key]
                if not g_items:
                    continue
                sub_lbl = SUBGROUP_LABELS.get(g_key, g_key)
                sub_tot = sum((it["total"] if m_code == "S" else abs(it["total"])) for it in g_items)

                with st.expander(f"📂 {sub_lbl}  —  {len(g_items)} konti  |  Kopā: {fmt_num(sub_tot)} €", expanded=False):
                    if res_for_exp is not None and sub_lbl in res_for_exp.amounts.index and res_month_cols:
                        sub_m_row = {mc: fmt_num(res_for_exp.amounts.loc[sub_lbl, mc]) for mc in res_month_cols}
                        sub_m_row["KOPĀ"] = fmt_num(res_for_exp.amounts.loc[sub_lbl, res_month_cols].sum())
                        st.caption("Apakšgrupas kopsumma pa mēnešiem finanšu tabulā (€):")
                        st.dataframe(pd.DataFrame([sub_m_row]), use_container_width=True, hide_index=True)

                    for it in g_items:
                        acc_id = it["id"]
                        orig_name = it["label"]
                        orig_path = it.get("path") or "Pamatlīmenis"
                        orig_row_num = it.get("row", "—")
                        acc_tot = float(it.get("total", 0.0))
                        m_dict_raw: dict = it.get("months", {})
                        non_zero_cnt = sum(1 for v in m_dict_raw.values() if abs(float(v or 0)) > 1e-4)

                        with st.expander(
                            f"📄 {orig_name}   (PZA rinda #{orig_row_num}  |  Kopā: {fmt_num(acc_tot)} €  |  Aktīvi mēneši: {non_zero_cnt}/{len(m_dict_raw)})",
                            expanded=False,
                        ):
                            inf_c1, inf_c2 = st.columns([2.2, 1.3], vertical_alignment="bottom")
                            with inf_c1:
                                st.markdown(
                                    f"**Oriģinālais nosaukums PZA:** `{orig_name}`  \n"
                                    f"**Oriģinālā vieta / sadaļa PZA:** `{orig_path}` (rinda **#{orig_row_num}**)  \n"
                                    f"**Piešķirtā TOC apakšgrupa:** **{sub_lbl}**"
                                )
                            with inf_c2:
                                allowed_sub_labels = get_subgroups_for_main_code(m_code)
                                if sub_lbl not in allowed_sub_labels:
                                    allowed_sub_labels.append(sub_lbl)

                                new_sub_choice = st.selectbox(
                                    "Pārvietot uz citu apakšgrupu",
                                    options=allowed_sub_labels,
                                    index=allowed_sub_labels.index(sub_lbl) if sub_lbl in allowed_sub_labels else 0,
                                    key=f"move_live_{acc_id}",
                                )
                                if new_sub_choice != sub_lbl and new_sub_choice in LABEL_TO_KEY:
                                    if st.button("Saglabāt jauno grupu un pārrēķināt", key=f"btn_mv_{acc_id}", type="primary"):
                                        new_g_k = LABEL_TO_KEY[new_sub_choice]
                                        if has_live_pza:
                                            mask = S.cls_df["id"] == acc_id
                                            S.cls_df.loc[mask, "Grupa"] = new_g_k
                                            S.cls_df.loc[mask, "Ieņēmums"] = new_g_k.startswith("S_")
                                            S.cls_df = sort_cls_df(S.cls_df)
                                            mapping_upd = {
                                                r_u["id"]: {
                                                    "group": normalize_group_key(r_u["Grupa"]),
                                                    "is_income": bool(str(r_u["Grupa"]).startswith("S_")),
                                                }
                                                for _, r_u in S.cls_df.iterrows()
                                            }
                                            stc = S.structure
                                            pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
                                            reported = an.ex.lines[pos].vals if pos is not None else None
                                            S.result = compute_toc_with_dynamic_subgroups(an.leaves, mapping_upd, an.ex.period_cols, reported)
                                            S.loaded_account_details = build_account_details_snapshot(an.leaves, S.cls_df, an.ex.period_cols)
                                            profile.save(an.leaves, mapping_upd)
                                        else:
                                            for s_it in snap_details:
                                                if s_it["id"] == acc_id:
                                                    s_it["group"] = new_g_k
                                            S.loaded_account_details = snap_details
                                            if S.get("result") is not None:
                                                S.result = rebuild_report_from_account_snapshot(S.result, snap_details)
                                                if S.get("current_report_id"):
                                                    packed_c = pack_commentary_with_account_details(S.get("commentary", ""), snap_details)
                                                    profile.update_report_commentary(S.current_report_id, packed_c)
                                        st.toast(f"Konts '{orig_name}' pārvietots uz '{new_sub_choice}'!")
                                        st.rerun()

                            if m_dict_raw:
                                disp_m = {k: fmt_num(float(v)) for k, v in m_dict_raw.items()}
                                disp_m["KOPĀ"] = fmt_num(acc_tot)
                                st.markdown("**Konta nolasītie skaitļi pa mēnešiem PZA failā (€):**")
                                st.dataframe(pd.DataFrame([disp_m]), use_container_width=True, hide_index=True)
    else:
        db_df = pd.DataFrame(saved_maps)
        db_df["toc_group"] = db_df["toc_group"].map(normalize_group_key)

        for m_code, m_title, bg_hex, fg_hex in main_sections:
            grp_keys_for_main = [k for k, (_, mc) in GROUPS.items() if mc == m_code]
            sub_df = db_df[db_df["toc_group"].isin(grp_keys_for_main)]

            main_tot_arch = 0.0
            if res_for_exp is not None and m_code in res_for_exp.amounts.index and res_month_cols:
                main_tot_arch = float(res_for_exp.amounts.loc[m_code, res_month_cols].sum())

            st.markdown(
                f'<div style="background-color:{bg_hex}; color:{fg_hex}; padding:8px 14px; border-radius:8px; '
                f'font-weight:700; font-size:15px; margin-top:16px; margin-bottom:8px; border:1.5px solid #1E293B; '
                f'display:flex; justify-content:space-between;">'
                f'<span>1. LĪMENIS: {m_title} ({len(sub_df)} konti)</span>'
                f'<span>Kopā atskaitē: {fmt_num(main_tot_arch)} €</span></div>',
                unsafe_allow_html=True,
            )

            if sub_df.empty:
                st.caption("Šajā grupā nav saglabātu kontu.")
                continue

            for g_key in grp_keys_for_main:
                g_rows = sub_df[db_df["toc_group"] == g_key]
                if g_rows.empty:
                    continue
                sub_lbl = SUBGROUP_LABELS.get(g_key, g_key)

                with st.expander(f"📂 {sub_lbl}  —  {len(g_rows)} konti", expanded=False):
                    for _, acc_row in g_rows.iterrows():
                        db_id = int(acc_row["id"])
                        orig_name = acc_row.get("account_label", "")
                        orig_path = acc_row.get("path_label", "") or "Pamatlīmenis"

                        with st.expander(f"📄 {orig_name}", expanded=False):
                            c_info, c_act = st.columns([2.2, 1.3], vertical_alignment="bottom")
                            with c_info:
                                st.markdown(
                                    f"**Oriģinālais nosaukums PZA:** `{orig_name}`  \n"
                                    f"**Oriģinālā sadaļa / ceļš PZA:** `{orig_path}`  \n"
                                    f"**Piešķirtā apakšgrupa:** **{sub_lbl}**"
                                )
                            with c_act:
                                allowed_sub_labels = get_subgroups_for_main_code(m_code)
                                if sub_lbl not in allowed_sub_labels:
                                    allowed_sub_labels.append(sub_lbl)

                                new_sub_db = st.selectbox(
                                    "Mainīt apakšgrupu",
                                    options=allowed_sub_labels,
                                    index=allowed_sub_labels.index(sub_lbl) if sub_lbl in allowed_sub_labels else 0,
                                    key=f"db_mv_{db_id}",
                                )
                                if new_sub_db != sub_lbl and new_sub_db in LABEL_TO_KEY:
                                    if st.button("Saglabāt izmaiņu", key=f"btn_db_mv_{db_id}", type="primary"):
                                        profile.update_saved_mappings_direct([{"id": db_id, "toc_group": LABEL_TO_KEY[new_sub_db]}])
                                        st.success("Konta grupa atjaunināta!")
                                        st.rerun()


# ==============================================================================
# POSMS 4: ATSKAITES UN SIMULĀCIJAS (ARHĪVS)
# ==============================================================================
elif S.nav_page == "archive":
    st.subheader("Saglabātās atskaites un simulācijas")
    st.caption("Šeit varat pārskatīt, atvērt vai dzēst iepriekš saglabātās finanšu atskaites un scenāriju simulācijas.")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("#### Finanšu atskaites")
        if not fin_reports:
            st.info("Nav saglabātu finanšu atskaišu.")
        else:
            for rep in fin_reports:
                rid = rep["id"]
                is_current = S.get("current_report_id") == rid
                with st.container(border=True):
                    rc1, rc2, rc3 = st.columns([2.2, 1, 0.8], vertical_alignment="center")
                    with rc1:
                        badge = " *(Aktīvā)*" if is_current else ""
                        st.markdown(f"**{rep.get('report_name')}**{badge}")
                        st.caption(
                            f"Periods: {rep.get('periods_label') or '—'} | "
                            f"Saglabāts: {str(rep.get('created_at', ''))[:16].replace('T', ' ')}"
                        )
                    with rc2:
                        if st.button("Atvērt Pārskatā", key=f"arch_open_f_{rid}", type="primary", use_container_width=True):
                            loaded = profile.load_report(rid)
                            if loaded:
                                loaded_res = LoadedReportResult(
                                    amounts=loaded["amounts"],
                                    main=loaded["main"],
                                    kpi=loaded["kpi"],
                                    row_kinds=loaded["row_kinds"],
                                    reconciliation=loaded["reconciliation"],
                                    by_group={},
                                )
                                S.result = strip_excluded_rows(loaded_res)
                                S.loaded_report_meta = loaded
                                S.current_report_id = loaded["id"]
                                S.pop("fin_months_active", None)
                                S.pop("fin_period_preset", None)
                                comm_clean, acc_snap = unpack_commentary_and_account_details(loaded.get("commentary"))
                                S.loaded_account_details = acc_snap
                                if comm_clean:
                                    S.commentary = comm_clean
                                else:
                                    S.pop("commentary", None)
                                S.nav_page = "overview"
                                st.rerun()
                    with rc3:
                        if st.button("Dzēst", key=f"arch_del_f_{rid}", use_container_width=True):
                            S.confirm_delete_rep_id = rid

                    if S.get("confirm_delete_rep_id") == rid:
                        st.warning(f"Dzēst atskaiti **{rep.get('report_name')}**?")
                        cd1, cd2 = st.columns(2)
                        with cd1:
                            if st.button("Jā, dzēst", key=f"yes_f_{rid}", type="primary", use_container_width=True):
                                if profile.delete_report(rid):
                                    S.pop("confirm_delete_rep_id", None)
                                    if S.get("current_report_id") == rid:
                                        S.pop("result", None)
                                        S.pop("loaded_report_meta", None)
                                        S.auto_loaded_latest = False
                                    st.rerun()
                        with cd2:
                            if st.button("Atcelt", key=f"no_f_{rid}", use_container_width=True):
                                S.pop("confirm_delete_rep_id", None)
                                st.rerun()

    with arch_col2:
        st.markdown("#### Saglabātās simulācijas")
        if not sim_reports:
            st.info("Nav saglabātu simulāciju.")
        else:
            for srep in sim_reports:
                sid = srep["id"]
                with st.container(border=True):
                    sc1, sc2, sc3 = st.columns([2.2, 1, 0.8], vertical_alignment="center")
                    with sc1:
                        st.markdown(f"**{srep.get('report_name')}**")
                        st.caption(
                            f"Bāzes periods: {srep.get('periods_label') or '—'} | "
                            f"Saglabāts: {str(srep.get('created_at', ''))[:16].replace('T', ' ')}"
                        )
                    with sc2:
                        if st.button("Atvērt Simulācijā", key=f"arch_open_s_{sid}", type="primary", use_container_width=True):
                            loaded_s = profile.load_report(sid)
                            if loaded_s:
                                loaded_sim_res = LoadedReportResult(
                                    amounts=loaded_s["amounts"],
                                    main=loaded_s["main"],
                                    kpi=loaded_s["kpi"],
                                    row_kinds=loaded_s["row_kinds"],
                                    reconciliation=loaded_s["reconciliation"],
                                    by_group={},
                                )
                                S.result = strip_excluded_rows(loaded_sim_res)
                                S.loaded_report_meta = loaded_s
                                try:
                                    sim_state = json.loads(loaded_s.get("commentary") or "{}")
                                    for r_k in loaded_sim_res.amounts.index:
                                        S[f"sim_pct_{r_k}"] = 0.0
                                        S[f"sim_eur_{r_k}"] = 0.0
                                    for k_s, v_s in sim_state.get("pct_changes", {}).items():
                                        S[f"sim_pct_{k_s}"] = float(v_s)
                                    for k_s, v_s in sim_state.get("eur_changes", {}).items():
                                        S[f"sim_eur_{k_s}"] = float(v_s)
                                    S["sim_situation_text"] = str(sim_state.get("situation_description") or "")
                                    if sim_state.get("selected_months"):
                                        S["sim_months_active"] = list(sim_state["selected_months"])
                                    np_cfg = sim_state.get("new_product", {})
                                    S["np_enabled"] = bool(np_cfg.get("enabled", False))
                                    S["np_name"] = str(np_cfg.get("name", "Jauns produkts / projekts"))
                                    S["np_s_eur"] = float(np_cfg.get("s_eur", 0.0))
                                    S["np_mat_eur"] = float(np_cfg.get("tvc_mat_eur", 0.0))
                                    S["np_deliv_eur"] = float(np_cfg.get("tvc_deliv_eur", 0.0))
                                    S["sim_ed_ver"] = S.get("sim_ed_ver", 0) + 1
                                except Exception:
                                    pass
                                S.nav_page = "simulation"
                                st.rerun()
                    with sc3:
                        if st.button("Dzēst", key=f"arch_del_s_{sid}", use_container_width=True):
                            S.confirm_delete_sim_id = sid

                    if S.get("confirm_delete_sim_id") == sid:
                        st.warning(f"Dzēst simulāciju **{srep.get('report_name')}**?")
                        cs1, cs2 = st.columns(2)
                        with cs1:
                            if st.button("Jā, dzēst", key=f"yes_s_{sid}", type="primary", use_container_width=True):
                                if profile.delete_report(sid):
                                    S.pop("confirm_delete_sim_id", None)
                                    st.rerun()
                        with cs2:
                            if st.button("Atcelt", key=f"no_s_{sid}", use_container_width=True):
                                S.pop("confirm_delete_sim_id", None)
                                st.rerun()


# ==============================================================================
# POSMS 5: JAUNS PZA / KONTU KLASIFIKĀCIJA
# ==============================================================================
elif S.nav_page == "ingest":
    st.subheader("Jauna PZA faila ielasīšana un kontu klasifikācija")

    ignore_memory = st.checkbox(
        "Ignorēt iepriekšējo profilu atmiņu un izvērtēt visus kontus pilnīgi no jauna (AI)",
        value=S.get("ignore_mem_flag", False),
        key="ignore_mem_checkbox",
    )
    S.ignore_mem_flag = ignore_memory

    uploaded = st.file_uploader("Augšupielādējiet PZA failu (CSV vai Excel)", type=["csv", "xlsx", "xlsm", "xls"])

    if uploaded is None:
        st.info("Augšupielādējiet PZA failu, lai sāktu strukturēšanu un kontu klasifikāciju.")
        st.stop()

    ok_quota, quota_msg = profile.check_quota()
    if not ok_quota and not S.get("is_admin", False):
        st.error(quota_msg)
        st.stop()

    data = uploaded.getvalue()
    fid = hashlib.md5(data).hexdigest()
    if S.get("fid") != fid:
        S.fid = fid
        S.grids = load_grids(uploaded.name, data)
        S.pop("structure", None)
        S.pop("analysis", None)
        S.pop("cls_df", None)
        S.pop("manual_leaf_overrides", None)

    grids: dict[str, pd.DataFrame] = S.grids

    if "structure" not in S:
        with st.spinner("Nolasa PZA struktūru un atdala pamatkontus no starpsummām..."):
            try:
                sheet = pick_sheet(get_client(), MODEL, grids)
                S.structure = request_structure(get_client(), MODEL, sheet, grids[sheet])
                profile.log_api_usage(1800, 0.012)
            except Exception as e:
                st.error(f"Kļūda: {e}")
                st.stop()

    stc: Structure = S.structure
    try:
        ex = extract_lines(grids[stc.sheet], stc)
        base_analysis = analyze(ex, stc)
        overrides: dict[str, bool] = S.get("manual_leaf_overrides", {})
        if overrides:
            base_leaf_ids = {lf.id for lf in base_analysis.leaves}
            new_leaves = []
            for idx_l, ln in enumerate(base_analysis.ex.lines):
                ln_id = getattr(ln, "id", f"r{ln.row}")
                if stc.result_row is not None and ln.row == stc.result_row:
                    continue
                is_leaf = overrides.get(ln_id, ln_id in base_leaf_ids)
                if is_leaf:
                    new_leaves.append(ln)
            base_analysis.leaves = new_leaves
        S.analysis = base_analysis
    except Exception as e:
        st.error(f"Kļūda datu ielasīšanā: {e}")
        st.stop()

    an = S.analysis
    display_labels = build_display_labels(an.leaves)

    with st.expander(f"PZA rindu kontrole (Atrasti {len(an.leaves)} pamatkonti, izlaistas {len(getattr(an, 'subtotals', []))} starpsummas un {len(getattr(an, 'empty_rows', []))} tukšas rindas)"):
        leaf_id_set = {lf.id for lf in an.leaves}
        sub_row_set = set(getattr(an, "subtotals", []))
        empty_row_set = set(getattr(an, "empty_rows", []))
        diag_rows = []
        for idx_l, ln in enumerate(an.ex.lines):
            ln_id = getattr(ln, "id", f"r{getattr(ln, 'row', idx_l)}")
            ln_row = getattr(ln, "row", idx_l)
            is_res = stc.result_row is not None and ln_row == stc.result_row
            first_v, tot_v = _safe_line_stats(getattr(ln, "vals", None))
            if is_res:
                st_txt = "Rezultāta rinda (NP)"
            elif ln_id in leaf_id_set:
                st_txt = "✅ Pamatkonts (iekļauts)"
            elif ln_row in empty_row_set:
                st_txt = "⚪ Tukša rinda / virsraksts (izlaists)"
            elif ln_row in sub_row_set:
                st_txt = "➕ Starpsumma / Kopā (izlaists)"
            else:
                st_txt = "Izlaists"
            diag_rows.append(
                {
                    "id": ln_id,
                    "Rinda": ln_row + 1,
                    "Iekļaut kā pamatkontu": (ln_id in leaf_id_set) and not is_res,
                    "Nosaukums": getattr(ln, "label", ""),
                    "Sadaļa": getattr(ln, "path", ""),
                    "1. periods": round(first_v, 2),
                    "Kopā": round(tot_v, 2),
                    "Noteiktais tips": st_txt,
                }
            )
        diag_df = pd.DataFrame(diag_rows)
        edited_diag = st.data_editor(
            diag_df,
            hide_index=True,
            use_container_width=True,
            disabled=["id", "Rinda", "Nosaukums", "Sadaļa", "1. periods", "Kopā", "Noteiktais tips"],
            key="pza_diag_editor",
        )
        if st.button("Saglabāt rindu izmaiņas"):
            new_overrides = {}
            for _, r_d in edited_diag.iterrows():
                new_overrides[r_d["id"]] = bool(r_d["Iekļaut kā pamatkontu"])
            S.manual_leaf_overrides = new_overrides
            S.pop("cls_df", None)
            st.rerun()

    st.markdown("#### 1. Kontu klasifikācija")

    if "cls_df" in S:
        if st.button("Pārlasīt un izvērtēt kontus no jauna"):
            S.pop("cls_df", None)
            S.pop("clarifications_done", None)
            st.rerun()

    if "cls_df" not in S:
        if st.button("Klasificēt kontus", type="primary"):
            if not an.leaves:
                st.warning("Netika atrasts neviens pamatkonts. Atveriet 'PZA rindu kontrole' augstāk un atzīmējiet vajadzīgos kontus.")
                st.stop()
            with st.spinner("Klasificē pamatkontus..."):
                try:
                    known, unknown = ({}, an.leaves) if ignore_memory else profile.lookup(an.leaves)
                    items = classify_with_claude(get_client(), MODEL, unknown) if unknown else []
                    if unknown:
                        profile.log_api_usage(1500, 0.01)
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
                                    Sadaļa=getattr(lf, "path", ""),
                                    Grupa=grp,
                                    Ieņēmums=bool(grp.startswith("S_")),
                                    Jautājums="",
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
                                    Sadaļa=getattr(lf, "path", ""),
                                    Grupa=grp,
                                    Ieņēmums=bool(grp.startswith("S_")),
                                    Jautājums=str(i.get("question") or "").strip(),
                                )
                            )
                    S.cls_df = sort_cls_df(pd.DataFrame(rows))
                    S.editor_v = 0
                    st.rerun()
                except Exception as e:
                    st.error(f"Kļūda klasifikācijā: {e}")
        st.stop()

    cls_df: pd.DataFrame = S.cls_df.copy()
    cls_df["Grupa"] = cls_df["Grupa"].map(normalize_group_key)
    if "Displeja_konts" not in cls_df.columns:
        cls_df["Displeja_konts"] = cls_df["id"].map(lambda x: display_labels.get(x, ""))

    ambiguous_rows = cls_df[cls_df["Jautājums"].astype(str).str.strip() != ""]

    if len(ambiguous_rows) > 0 and not S.get("clarifications_done", False):
        st.subheader("Pozīciju precizēšana (Apakšgrupas izvēle)")
        st.caption("Izvēlieties, kurā apakšgrupā iekļaut šos kontus, vai izveidojiet jaunu apakšgrupu.")

        with st.expander("➕ Izveidot jaunu apakšgrupu, ja sarakstā nav atbilstošas", expanded=False):
            ca1, ca2, ca3 = st.columns([2, 1.4, 1], vertical_alignment="bottom")
            with ca1:
                cl_new_sub = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Personāla ilgtspējas un pasākumu izdevumi", key="clar_new_sub_in")
            with ca2:
                cl_new_main = st.selectbox("Lielā grupa", options=["Uzturēšanas izmaksas (OE)", "TVC (Mainīgās izmaksas)", "S (Ieņēmumi)"], key="clar_new_main_in")
            with ca3:
                if st.button("Pievienot sarakstam", use_container_width=True, key="clar_add_btn"):
                    if cl_new_sub.strip():
                        clean_l = cl_new_sub.strip()
                        mc_new = "S" if cl_new_main.startswith("S") else ("TVC" if cl_new_main.startswith("TVC") else "OE")
                        nk = f"{mc_new}_CUSTOM_{hashlib.md5(clean_l.encode()).hexdigest()[:6].upper()}"
                        S.setdefault("custom_subgroups", {})[nk] = (clean_l, mc_new)
                        register_custom_subgroup(nk, clean_l, mc_new)
                        st.toast(f"Apakšgrupa '{clean_l}' pievienota!")
                        st.rerun()

        clarifications: dict[str, tuple[str, bool]] = {}
        for _, r in ambiguous_rows.iterrows():
            row_id = r["id"]
            disp_name = r["Displeja_konts"]
            agent_grp = normalize_group_key(r["Grupa"])
            agent_main_code = get_main_code(agent_grp)
            agent_sub_label = SUBGROUP_LABELS.get(agent_grp, agent_grp)

            with st.container(border=True):
                st.markdown(f"**{disp_name}** *(AI sākotnējais ieteikums: `{agent_sub_label}`)*")
                st.caption(r["Jautājums"])

                c_m, c_s = st.columns([1.2, 1.8])
                with c_m:
                    main_opts_map = {
                        "OE": "Uzturēšanas izmaksas (OE)",
                        "TVC": "TVC (Mainīgās izmaksas)",
                        "S": "S (Ieņēmumi)",
                        "EXCL": "Izslēgt (Nolietojums / Virtuāls)",
                    }
                    rev_main_map = {v: k for k, v in main_opts_map.items()}
                    m_list = list(main_opts_map.values())
                    sel_m_lbl = st.selectbox(
                        "1. Lielā grupa",
                        options=m_list,
                        index=m_list.index(main_opts_map.get(agent_main_code, "Uzturēšanas izmaksas (OE)")),
                        key=f"clar_m_{row_id}",
                    )
                    sel_m_code = rev_main_map[sel_m_lbl]
                with c_s:
                    sub_opts = get_subgroups_for_main_code(sel_m_code)
                    def_sub_idx = sub_opts.index(agent_sub_label) if agent_sub_label in sub_opts else 0
                    sel_sub_lbl = st.selectbox(
                        "2. Apakšgrupa (tikai no izvēlētās Lielās grupas)",
                        options=sub_opts,
                        index=def_sub_idx,
                        key=f"clar_s_{row_id}",
                    )
                    chosen_g_key = LABEL_TO_KEY.get(sel_sub_lbl, MAIN_GROUP_TO_DEFAULT_SUB.get(sel_m_code, "OE_OTHER"))
                    clarifications[row_id] = (chosen_g_key, chosen_g_key.startswith("S_"))

        if st.button("Apstiprināt izvēlētās apakšgrupas", type="primary"):
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

    hdr_col1, hdr_col2 = st.columns([4, 1], vertical_alignment="bottom")
    with hdr_col1:
        st.markdown("#### Kontu sadalījums pa Lielajām grupām un Apakšgrupām")
    with hdr_col2:
        if st.button("Sakārtot tabulu", use_container_width=True):
            S.cls_df = sort_cls_df(cls_df)
            S.editor_v = S.get("editor_v", 0) + 1
            st.rerun()

    with st.expander("➕ Pievienot jaunu apakšgrupu (sadalīt izmaksas vai ieņēmumus sīkāk)", expanded=False):
        ac1, ac2, ac3 = st.columns([2, 1.5, 1], vertical_alignment="bottom")
        with ac1:
            new_sub_title = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Personāla ilgtspējas pasākumu izdevumi")
        with ac2:
            new_sub_parent = st.selectbox("Lielā grupa", options=["Uzturēšanas izmaksas (OE)", "TVC (Mainīgās izmaksas)", "S (Ieņēmumi)"])
        with ac3:
            if st.button("Pievienot", use_container_width=True):
                if new_sub_title.strip():
                    clean_lbl = new_sub_title.strip()
                    m_code = "S" if new_sub_parent.startswith("S") else ("TVC" if new_sub_parent.startswith("TVC") else "OE")
                    new_k = f"{m_code}_CUSTOM_{hashlib.md5(clean_lbl.encode()).hexdigest()[:6].upper()}"
                    S.setdefault("custom_subgroups", {})[new_k] = (clean_lbl, m_code)
                    register_custom_subgroup(new_k, clean_lbl, m_code)
                    st.toast(f"Apakšgrupa '{clean_lbl}' pievienota.")
                    st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S (Ieņēmumi)", f"{cls_df['Grupa'].str.startswith('S_').sum()} konti")
    c2.metric("TVC (Mainīgās)", f"{cls_df['Grupa'].str.startswith('TVC').sum()} konti")
    c3.metric("Uzturēšanas izmaksas", f"{cls_df['Grupa'].str.startswith('OE_').sum()} konti")
    c4.metric("Izslēgtie", f"{(cls_df['Grupa'] == 'EXCLUDED').sum()} konti")

    cat_sections = [
        ("S", "S (Ieņēmumi)", MAIN_GROUP_OPTIONS[0]),
        ("TVC", "TVC (Mainīgās izmaksas)", MAIN_GROUP_OPTIONS[1]),
        ("OE", "Uzturēšanas izmaksas (OE)", MAIN_GROUP_OPTIONS[2]),
        ("EXCL", "Izslēgt (Nolietojums / Virtuāls)", MAIN_GROUP_OPTIONS[3]),
    ]

    changed_any = False
    new_groups_map: dict[str, str] = {}

    for m_code_sec, sec_title, m_cat_str in cat_sections:
        sec_mask = cls_df["Grupa"].map(get_main_code) == m_code_sec
        sec_df = cls_df[sec_mask].copy()
        if sec_df.empty:
            continue

        allowed_subs_sec = get_subgroups_for_main_code(m_code_sec)
        st.markdown(f"**{sec_title}** ({len(sec_df)} konti)")

        sec_view = pd.DataFrame(
            {
                "id": sec_df["id"].tolist(),
                "Konts": sec_df["Displeja_konts"].tolist(),
                "Lielā grupa": [m_cat_str] * len(sec_df),
                "Apakšgrupa": [
                    SUBGROUP_LABELS.get(normalize_group_key(g), allowed_subs_sec[0])
                    if SUBGROUP_LABELS.get(normalize_group_key(g)) in allowed_subs_sec
                    else allowed_subs_sec[0]
                    for g in sec_df["Grupa"]
                ],
            }
        )
        sec_h = (len(sec_view) + 1) * 35 + 10
        ed_sec = st.data_editor(
            sec_view,
            hide_index=True,
            use_container_width=True,
            height=sec_h,
            disabled=["id", "Konts"],
            key=f"sec_cls_ed_{m_code_sec}_{S.get('editor_v', 0)}",
            column_config={
                "id": None,
                "Konts": st.column_config.TextColumn("Konta nosaukums", width="large"),
                "Lielā grupa": st.column_config.SelectboxColumn("Lielā grupa", options=MAIN_GROUP_OPTIONS, required=True, width="medium"),
                "Apakšgrupa": st.column_config.SelectboxColumn(
                    f"Apakšgrupa ({sec_title})",
                    options=allowed_subs_sec,
                    required=True,
                    width="large",
                ),
            },
        )

        for idx_s in range(len(sec_df)):
            rid_s = sec_df.iloc[idx_s]["id"]
            old_g_s = normalize_group_key(sec_df.iloc[idx_s]["Grupa"])
            old_sub_s = SUBGROUP_LABELS.get(old_g_s, "")

            new_m_s = ed_sec.iloc[idx_s]["Lielā grupa"]
            new_sub_s = ed_sec.iloc[idx_s]["Apakšgrupa"]

            if new_m_s != m_cat_str:
                target_mc = "S" if new_m_s.startswith("S") else ("TVC" if new_m_s.startswith("TVC") else ("EXCL" if new_m_s.startswith("Izslēgt") else "OE"))
                new_groups_map[rid_s] = MAIN_GROUP_TO_DEFAULT_SUB.get(target_mc, "OE_OTHER")
                changed_any = True
            elif new_sub_s != old_sub_s and new_sub_s in LABEL_TO_KEY:
                new_groups_map[rid_s] = LABEL_TO_KEY[new_sub_s]
                changed_any = True
            else:
                new_groups_map[rid_s] = old_g_s

    if changed_any:
        cls_df["Grupa"] = cls_df["id"].map(lambda x: new_groups_map.get(x, "OE_OTHER"))
        cls_df["Ieņēmums"] = cls_df["Grupa"].str.startswith("S_")
        S.cls_df = sort_cls_df(cls_df)
        S.editor_v = S.get("editor_v", 0) + 1
        st.rerun()

    st.markdown("#### 2. Aprēķins un saglabāšana")

    default_rep_name = f"{Path(uploaded.name).stem} ({datetime.now().strftime('%d.%m.%Y')})"
    col_rn, col_btn = st.columns([2, 1], vertical_alignment="bottom")
    with col_rn:
        report_custom_name = st.text_input("Atskaites nosaukums", value=default_rep_name)
    with col_btn:
        run_calc = st.button("Apstiprināt, saglabāt un atvērt Pārskatā", type="primary", use_container_width=True)

    if run_calc:
        cls_sorted = sort_cls_df(cls_df)
        S.cls_df = cls_sorted
        S.editor_v = S.get("editor_v", 0) + 1

        mapping = {
            r["id"]: {
                "group": normalize_group_key(r["Grupa"]),
                "is_income": bool(str(r["Grupa"]).startswith("S_")),
            }
            for _, r in cls_sorted.iterrows()
        }
        pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
        reported = an.ex.lines[pos].vals if pos is not None else None

        res = compute_toc_with_dynamic_subgroups(an.leaves, mapping, an.ex.period_cols, reported)
        S.result = res
        S.pop("fin_months_active", None)
        S.pop("fin_period_preset", None)

        acc_snapshot = build_account_details_snapshot(an.leaves, cls_sorted, an.ex.period_cols)
        S.loaded_account_details = acc_snapshot

        if profile.is_connected:
            ok_map, _ = profile.save(an.leaves, mapping)
            packed_init_comm = pack_commentary_with_account_details("", acc_snapshot)
            ok_rep, rep_id = profile.save_report(
                report_name=report_custom_name.strip() or default_rep_name,
                file_name=uploaded.name,
                res_obj=res,
                commentary=packed_init_comm,
            )
            profile.log_api_usage(0, 0.0)
            S.current_report_id = rep_id
            S.loaded_report_meta = {
                "id": rep_id,
                "report_name": report_custom_name.strip() or default_rep_name,
                "periods_label": f"{len(an.ex.period_cols)} periodi",
            }
            if ok_map and ok_rep:
                st.toast("Atskaite un kontu mēnešu skaitļi saglabāti!")

        S.pop("commentary", None)
        S.nav_page = "overview"
        st.rerun()


# ==============================================================================
# POSMS 6: PROFILS UN UZSTĀDĪJUMI
# ==============================================================================
elif S.nav_page == "settings":
    st.subheader(f"Profils un Uzstādījumi ({S.active_profile})")

    with st.container(border=True):
        st.markdown("#### 1. Profila informācija")
        ok_q, q_info = profile.check_quota()
        pi1, pi2, pi3, pi4, pi5 = st.columns(5)
        pi1.metric("Uzņēmums / Profils", S.active_profile)
        pi2.metric("Saglabātie konti", saved_acc_count)
        pi3.metric("Finanšu atskaites", len(fin_reports))
        pi4.metric("Saglabātās simulācijas", len(sim_reports))
        pi5.metric("Mēneša kvota", q_info or "—")

    with st.container(border=True):
        st.markdown("#### 2. Finanšu gada sākums")
        st.caption("Nosaka, kurā datumā uzkrājošie rādītāji sāk jaunu finanšu gadu.")
        fy_c1, fy_c2, _ = st.columns([1, 1.5, 2.5])
        with fy_c1:
            selected_fy_day = st.selectbox(
                "Diena",
                options=list(range(1, 32)),
                index=int(S.get("fy_start_day", 1)) - 1,
            )
        with fy_c2:
            selected_fy_month = st.selectbox(
                "Mēnesis",
                options=list(MONTH_NAMES_LV.keys()),
                format_func=lambda m: MONTH_NAMES_LV[m],
                index=list(MONTH_NAMES_LV.keys()).index(S.get("fy_start_month", 1)),
            )
        S.fy_start_day = selected_fy_day
        S.fy_start_month = selected_fy_month

    with st.container(border=True):
        st.markdown("#### 3. Paroles maiņa")
        st.caption("Nomainiet piekļuves paroli šim uzņēmuma profilam.")
        pw_c1, pw_c2, pw_c3 = st.columns([1.5, 1.5, 1], vertical_alignment="bottom")
        with pw_c1:
            new_pw1 = st.text_input("Jaunā parole", type="password", key="set_new_pw1")
        with pw_c2:
            new_pw2 = st.text_input("Jaunā parole atkārtoti", type="password", key="set_new_pw2")
        with pw_c3:
            if st.button("Saglabāt jaunu paroli", use_container_width=True):
                if not new_pw1.strip():
                    st.warning("Ievadiet jauno paroli.")
                elif new_pw1 != new_pw2:
                    st.error("Paroles nesakrīt.")
                else:
                    all_c = admin_store.list_all_companies_admin()
                    cur_c = next((c for c in all_c if c["company_name"] == S.active_profile), {})
                    cur_lim = int(cur_c.get("monthly_limit", 50))
                    cur_notes = str(cur_c.get("notes", ""))
                    ok_pw, msg_pw = admin_store.create_or_update_company_admin(
                        company_name=S.active_profile,
                        password=new_pw1.strip(),
                        monthly_limit=cur_lim,
                        is_active=True,
                        notes=cur_notes,
                        reset_usage=False,
                    )
                    if ok_pw:
                        st.success("Parole veiksmīgi nomainīta!")
                    else:
                        st.error(msg_pw)

    with st.container(border=True):
        st.markdown("#### 4. Pievienot jaunu apakšgrupu")
        st.caption("Izveidojiet papildu izmaksu vai ieņēmumu rindu (visu kontu detalizētu izklājumu skatiet rīkjoslas sadaļā **Kontu struktūra**).")
        ns1, ns2, ns3 = st.columns([2, 1.5, 1], vertical_alignment="bottom")
        with ns1:
            custom_sub_lbl = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Pārējās administrācijas izmaksas", key="ed_new_sub_lbl")
        with ns2:
            custom_sub_main = st.selectbox("Lielā grupa", options=["Uzturēšanas izmaksas (OE)", "TVC (Mainīgās izmaksas)", "S (Ieņēmumi)"], key="ed_new_sub_main")
        with ns3:
            if st.button("Pievienot apakšgrupu", use_container_width=True, key="ed_btn_add_sub"):
                if custom_sub_lbl.strip():
                    clean_lbl = custom_sub_lbl.strip()
                    m_c = "S" if custom_sub_main.startswith("S") else ("TVC" if custom_sub_main.startswith("TVC") else "OE")
                    new_k = f"{m_c}_CUSTOM_{hashlib.md5(clean_lbl.encode()).hexdigest()[:6].upper()}"
                    S.setdefault("custom_subgroups", {})[new_k] = (clean_lbl, m_c)
                    register_custom_subgroup(new_k, clean_lbl, m_c)
                    st.toast(f"Apakšgrupa '{clean_lbl}' pievienota.")
                    st.rerun()

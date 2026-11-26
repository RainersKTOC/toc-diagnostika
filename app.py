"""
app.py — THRUVIA: MI instruments uzņēmuma finanšu analīzei un vadības lēmumu modelēšanai (v26.5)
8 soļu fokusēts Demo | 13.3" + monitora tabulas | 1 klikšķa Admin PZA pārbaude & e-pasts | Gudrāka klasifikācija
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
import base64
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from email.mime.text import MIMEText
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import smtplib
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

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
    page_title="THRUVIA — Finanšu analīze un vadības lēmumu modelēšana",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Pakalpojuma sniedzēja juridiskie rekvizīti (aizpildāmi ar precīziem SIA datiem pirms publiskas palaišanas)
LEGAL_PROVIDER_DETAILS = {
    "company": "SIA B4B (THRUVIA pakalpojuma uzturētājs)",
    "reg_no": "[Ievietot reģistrācijas numuru]",
    "address": "[Ievietot juridisko adresi, Rīga, Latvija]",
    "email": "info@b4b.lv",
}

LEGAL_DISCLAIMER_SHORT = (
    "Atruna: THRUVIA ir informatīvs finanšu datu vizualizācijas un vadības scenāriju modelēšanas instruments. "
    "Sistēma, tās MI funkcijas un pakalpojuma sniedzējs nesniedz licencētas finanšu, investīciju, nodokļu vai grāmatvedības konsultācijas "
    "un neuzņemas atbildību par lēmumiem, kas pieņemti, balstoties uz sistēmas aprēķiniem."
)

# Pārsaucam EXCLUDED grupu par plašāku, korektu nosaukumu
EXCLUDED_LABEL = "Neiekļautie konti (ārpus TOC aprēķina)"
_old_excl_lbl = SUBGROUP_LABELS.get("EXCLUDED")
if _old_excl_lbl and _old_excl_lbl != EXCLUDED_LABEL:
    LABEL_TO_KEY.pop(_old_excl_lbl, None)
GROUPS["EXCLUDED"] = (EXCLUDED_LABEL, "EXCL")
SUBGROUP_LABELS["EXCLUDED"] = EXCLUDED_LABEL
LABEL_TO_KEY[EXCLUDED_LABEL] = "EXCLUDED"

if "T_CHILD" not in FILLS:
    FILLS["T_CHILD"] = ("DAEEF3", False, "000000")
if "T_SUBTOTAL" not in FILLS:
    FILLS["T_SUBTOTAL"] = ("DAEEF3", False, "000000")
if "S_SUBTOTAL" not in FILLS:
    FILLS["S_SUBTOTAL"] = (FILLS["S_CHILD"][0], False, "000000")
if "TVC_SUBTOTAL" not in FILLS:
    FILLS["TVC_SUBTOTAL"] = (FILLS["TVC_CHILD"][0], False, "000000")
if "S_DETAIL" not in FILLS:
    FILLS["S_DETAIL"] = (FILLS["S_CHILD"][0], False, "000000")
if "TVC_DETAIL" not in FILLS:
    FILLS["TVC_DETAIL"] = (FILLS["TVC_CHILD"][0], False, "000000")
if "S_NEW_PROD" not in FILLS:
    FILLS["S_NEW_PROD"] = (FILLS["S_CHILD"][0], False, "000000")
if "TVC_NEW_PROD" not in FILLS:
    FILLS["TVC_NEW_PROD"] = (FILLS["TVC_CHILD"][0], False, "000000")
if "T_NEW_PROD" not in FILLS:
    FILLS["T_NEW_PROD"] = ("DAEEF3", False, "000000")

# Tikai divas papildu skaidras OE apakšgrupas (noņemta liekā OE_OPER_OTHER, kas dublēja OE_OTHER!)
EXTRA_OE_SUBGROUPS = {
    "OE_ADMIN_OTHER": ("Pārējās administrācijas izmaksas", "OE"),
    "OE_SALES_OTHER": ("Pārējās pārdošanas izmaksas", "OE"),
}
for k_ext, (lbl_ext, mc_ext) in EXTRA_OE_SUBGROUPS.items():
    if k_ext not in GROUPS:
        GROUPS[k_ext] = (lbl_ext, mc_ext)
    if k_ext not in SUBGROUP_LABELS:
        SUBGROUP_LABELS[k_ext] = lbl_ext
    LABEL_TO_KEY[lbl_ext] = k_ext

# Drošībai izņemam dublējošo OE_OPER_OTHER, ja tas būtu saglabājies atmiņā
if "OE_OPER_OTHER" in GROUPS:
    _dup_lbl = SUBGROUP_LABELS.pop("OE_OPER_OTHER", None)
    GROUPS.pop("OE_OPER_OTHER", None)
    if _dup_lbl:
        LABEL_TO_KEY.pop(_dup_lbl, None)


def now_riga() -> datetime:
    try:
        return datetime.now(ZoneInfo("Europe/Riga"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=2)))


def register_custom_subgroup(key: str, label: str, main_code: str = "OE") -> None:
    old_lbl = SUBGROUP_LABELS.get(key)
    if old_lbl and old_lbl != label:
        LABEL_TO_KEY.pop(old_lbl, None)
    GROUPS[key] = (label, main_code)
    SUBGROUP_LABELS[key] = label
    LABEL_TO_KEY[label] = key


def get_subgroups_for_main_code(main_code: str) -> list[str]:
    res = []
    for k, (lbl, mc) in GROUPS.items():
        if mc == main_code and lbl not in res:
            res.append(lbl)
    return res


def is_wage_row_label(label: str) -> bool:
    """Nosaka, vai apakšgrupa ir algu pozīcija (vienmēr jātur OE augšpusē un smalki jāizceļ)."""
    k = LABEL_TO_KEY.get(str(label).strip(), "")
    if k in ("OE_ADMIN_WAGES", "OE_WORKER_WAGES"):
        return True
    low = str(label or "").lower()
    return (
        (("alga" in low or "atalgojum" in low) and "sociāl" in low)
        or low.startswith("administrācijas algas")
        or low.startswith("strādnieku algas")
    )


def fmt_acc_count_lv(n: int) -> str:
    return f"{n} konts" if (n % 10 == 1 and n % 100 != 11) else f"{n} konti"


MAIN_GROUP_OPTIONS = [
    "S (Ieņēmumi)",
    "TVC (Mainīgās izmaksas)",
    "Uzturēšanas izmaksas (OE)",
    "Neiekļaut (ārpus TOC aprēķina)",
]

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
    "OE_OTHER": 14,
    "EXCLUDED": 99,
}


def normalize_group_key(group_key: str) -> str:
    g = str(group_key or "OE_OTHER").strip()
    if g == "S":
        return "S_REV"
    if g == "OE_OPER_OTHER":
        return "OE_OTHER"
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


def _get_secret_or_env(key_name: str, default: str = "") -> str:
    val = os.environ.get(key_name, "")
    if val:
        return str(val).strip()
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return default


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
                if (
                    str(k).upper() in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "API_KEY")
                    and not clean.startswith("eyJ")
                    and not clean.startswith("http")
                ):
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


def get_billing_config() -> dict:
    trial_days_str = _get_secret_or_env("TRIAL_DAYS", "5")
    try:
        trial_days = max(1, int(trial_days_str))
    except ValueError:
        trial_days = 5
    return {
        "trial_days": trial_days,
        "monthly_price_eur": _get_secret_or_env("SUBSCRIPTION_PRICE_EUR", "49"),
        "stripe_checkout_url": _get_secret_or_env("STRIPE_CHECKOUT_URL", ""),
        "stripe_portal_url": _get_secret_or_env("STRIPE_PORTAL_URL", ""),
        "support_email": _get_secret_or_env("SUPPORT_EMAIL", LEGAL_PROVIDER_DETAILS["email"]),
        "app_url": _get_secret_or_env("APP_PUBLIC_URL", ""),
    }


def extract_note_field(notes: str | None, prefix: str) -> str:
    s = str(notes or "")
    m = re.search(rf"{re.escape(prefix)}:([^|\n]+)", s)
    return m.group(1).strip() if m else ""


def extract_free_notes(notes: str | None) -> str:
    s = str(notes or "").strip()
    if not s:
        return ""
    if "NOTE:" in s:
        return s.split("NOTE:", 1)[1].strip()
    parts = [
        p.strip()
        for p in s.split("|")
        if not re.match(
            r"^(PLAN|ONBOARD|TRIAL_START|TRIAL_START_DT|EMAIL|PAID_SINCE|LEGAL_OK|CONTACT|WELCOME_SEEN):",
            p.strip(),
            re.I,
        )
    ]
    return " | ".join(p for p in parts if p)


def build_structured_notes(
    plan_code: str,
    email: str = "",
    trial_start: date | None = None,
    trial_start_dt: str = "",
    free_note: str = "",
    onboard_status: str = "APPROVED",
    contact_name: str = "",
    legal_ok_date: str = "",
    welcome_seen: str = "1",
) -> str:
    parts = [f"PLAN:{plan_code}", f"ONBOARD:{onboard_status}"]
    if plan_code == "TRIAL":
        if trial_start_dt:
            parts.append(f"TRIAL_START_DT:{trial_start_dt}")
            parts.append(f"TRIAL_START:{trial_start_dt[:10]}")
        elif trial_start:
            parts.append(f"TRIAL_START:{trial_start.isoformat()}")
    if plan_code == "ACTIVE":
        parts.append(f"PAID_SINCE:{date.today().isoformat()}")
    if contact_name.strip():
        parts.append(f"CONTACT:{contact_name.strip()}")
    if email.strip():
        parts.append(f"EMAIL:{email.strip()}")
    if legal_ok_date.strip():
        parts.append(f"LEGAL_OK:{legal_ok_date.strip()}")
    if welcome_seen in ("0", "1"):
        parts.append(f"WELCOME_SEEN:{welcome_seen}")
    if free_note.strip():
        parts.append(f"NOTE:{free_note.strip()}")
    return " | ".join(parts)


def parse_onboarding_status(notes: str | None) -> str:
    val = extract_note_field(notes, "ONBOARD").upper()
    if val in ("TUTORIAL", "SUBMIT_PZA", "PENDING_WAIT", "PENDING_IMMEDIATE", "APPROVED"):
        return val
    return "APPROVED"


ONBOARD_LABELS_LV = {
    "TUTORIAL": "Apmācība (Demo)",
    "SUBMIT_PZA": "Gaida PZA iesūtīšanu",
    "PENDING_WAIT": "⏳ Gaida PZA pārbaudi (Demo)",
    "PENDING_IMMEDIATE": "⚠️ Gaida pārbaudi (Strādā)",
    "APPROVED": "✅ Apstiprināts",
}


def update_company_onboarding_status(
    admin_store: SupabaseProfileStore,
    company_name: str,
    new_onboard_status: str,
    reset_trial_now: bool = False,
    welcome_seen: str | None = None,
) -> tuple[bool, datetime | None]:
    """Atjaunina klienta ONBOARD statusu, precīzu 5 dienu izmēģinājuma sākuma laiku un sagaidīšanas karogu."""
    all_c = admin_store.list_all_companies_admin()
    cur_c = next((c for c in all_c if c["company_name"] == company_name), None)
    if not cur_c:
        return False, None
    old_notes = str(cur_c.get("notes") or "")
    plan_code = extract_note_field(old_notes, "PLAN").upper() or "MANAGED"
    email = extract_note_field(old_notes, "EMAIL")
    contact = extract_note_field(old_notes, "CONTACT")
    legal_ok = extract_note_field(old_notes, "LEGAL_OK")
    free_n = extract_free_notes(old_notes)
    cur_welcome = extract_note_field(old_notes, "WELCOME_SEEN") or "1"
    final_welcome = welcome_seen if welcome_seen in ("0", "1") else cur_welcome

    tr_dt_str = extract_note_field(old_notes, "TRIAL_START_DT")
    tr_d_str = extract_note_field(old_notes, "TRIAL_START")
    tr_date = None
    trial_end_dt = None

    if reset_trial_now and plan_code == "TRIAL":
        now_dt = now_riga()
        tr_dt_str = now_dt.strftime("%Y-%m-%dT%H:%M")
        tr_date = now_dt.date()
        trial_end_dt = now_dt + timedelta(days=get_billing_config()["trial_days"])
    elif tr_dt_str:
        try:
            dt_parsed = datetime.strptime(tr_dt_str, "%Y-%m-%dT%H:%M")
            tr_date = dt_parsed.date()
            trial_end_dt = dt_parsed + timedelta(days=get_billing_config()["trial_days"])
        except ValueError:
            tr_dt_str = ""
    elif tr_d_str:
        try:
            tr_date = date.fromisoformat(tr_d_str)
            trial_end_dt = datetime.combine(tr_date, datetime.min.time()) + timedelta(days=get_billing_config()["trial_days"])
        except ValueError:
            tr_date = date.today()

    new_notes = build_structured_notes(
        plan_code=plan_code,
        email=email,
        trial_start=tr_date,
        trial_start_dt=tr_dt_str,
        free_note=free_n,
        onboard_status=new_onboard_status,
        contact_name=contact,
        legal_ok_date=legal_ok,
        welcome_seen=final_welcome,
    )
    ok_u, _ = admin_store.create_or_update_company_admin(
        company_name=company_name,
        password="",
        monthly_limit=int(cur_c.get("monthly_limit", 10000)),
        is_active=bool(cur_c.get("is_active", True)),
        notes=new_notes,
        reset_usage=False,
    )
    return ok_u, trial_end_dt


def parse_subscription_status(notes: str | None, trial_days: int = 5) -> dict:
    """Aprēķina abonementa statusu, atlikušās dienas/stundas un precīzu pirmās iekasēšanas datumu + laiku."""
    s = str(notes or "")
    if "PLAN:ACTIVE" in s.upper():
        return {
            "code": "ACTIVE",
            "label": "Aktīvs mēneša abonements",
            "days_left": None,
            "time_left_str": "",
            "first_charge_str": "",
            "expired": False,
        }
    if "PLAN:CANCELLED" in s.upper():
        return {
            "code": "CANCELLED",
            "label": "Abonements atcelts",
            "days_left": 0,
            "time_left_str": "0 d.",
            "first_charge_str": "",
            "expired": True,
        }

    onb = parse_onboarding_status(s)
    if onb in ("TUTORIAL", "SUBMIT_PZA", "PENDING_WAIT"):
        return {
            "code": "TRIAL",
            "label": f"Ievadposms ({trial_days} d. izmēģinājums sāksies pēc PZA apstiprināšanas)",
            "days_left": trial_days,
            "time_left_str": f"{trial_days} dienas",
            "first_charge_str": "",
            "expired": False,
        }

    now_local = now_riga().replace(tzinfo=None)
    tr_dt_str = extract_note_field(s, "TRIAL_START_DT")
    start_dt: datetime | None = None
    if tr_dt_str:
        try:
            start_dt = datetime.strptime(tr_dt_str, "%Y-%m-%dT%H:%M")
        except ValueError:
            start_dt = None

    if start_dt is None:
        m = re.search(r"TRIAL_START:(\d{4}-\d{2}-\d{2})", s)
        if m:
            try:
                start_dt = datetime.combine(date.fromisoformat(m.group(1)), datetime.strptime("09:00", "%H:%M").time())
            except ValueError:
                start_dt = None

    if start_dt is not None:
        end_dt = start_dt + timedelta(days=trial_days)
        charge_str = end_dt.strftime("%d.%m.%Y. plkst. %H:%M")
        rem_sec = (end_dt - now_local).total_seconds()
        if rem_sec > 0:
            rem_days = int(rem_sec // 86400)
            rem_hours = int((rem_sec % 86400) // 3600)
            d_word = "diena" if (rem_days % 10 == 1 and rem_days % 100 != 11) else "dienas"
            h_word = "stunda" if (rem_hours % 10 == 1 and rem_hours % 100 != 11) else "stundas"
            time_left = f"{rem_days} {d_word} un {rem_hours} {h_word}" if rem_days > 0 else f"{max(rem_hours, 1)} {h_word}"
            return {
                "code": "TRIAL",
                "label": f"Bezmaksas izmēģinājums (atlikušas {time_left}, līdz {charge_str})",
                "days_left": rem_days if rem_days > 0 else 1,
                "time_left_str": time_left,
                "first_charge_str": charge_str,
                "expired": False,
            }
        return {
            "code": "TRIAL_EXPIRED",
            "label": f"Izmēģinājuma periods beidzies ({charge_str})",
            "days_left": 0,
            "time_left_str": "0 dienas",
            "first_charge_str": charge_str,
            "expired": True,
        }

    return {
        "code": "MANAGED",
        "label": "Aktīvs B4B klienta profils",
        "days_left": None,
        "time_left_str": "",
        "first_charge_str": "",
        "expired": False,
    }


def send_approval_email_to_client(
    to_email: str,
    company_name: str,
    contact_name: str,
    first_charge_str: str,
    billing_cfg: dict,
) -> tuple[bool, str, str, str]:
    """
    Nosūta klientam apstiprinājuma e-pastu (caur Resend API vai SMTP, ja iestatīts Secrets).
    Vienmēr atgriež (sent_ok, status_msg, subject, body) ērtai kontrolei.
    """
    greet = f"Labdien, {contact_name}!" if contact_name else "Labdien!"
    app_link = billing_cfg.get("app_url") or "THRUVIA sistēmā"
    subject = f"THRUVIA: Jūsu uzņēmuma ({company_name}) PZA ir apstiprināts!"
    body = (
        f"{greet}\n\n"
        f"Jūsu uzņēmuma ({company_name}) iesūtītais PZA fails ir pārbaudīts, sakārtots un apstiprināts!\n"
        f"Tagad varat pieslēgties savam profilam ({app_link}) un pilnvērtīgi darboties ar sava uzņēmuma finanšu datiem un scenāriju simulācijām.\n\n"
        f"Jūsu {billing_cfg['trial_days']} dienu bezmaksas izmēģinājuma periods ir sācies šodien.\n"
        f"Pirmā automātiskā abonementa maksa ({billing_cfg['monthly_price_eur']} €/mēn.) tiks iekasēta: {first_charge_str} (Latvijas laiks), "
        f"ja abonements iepriekš netiks atcelts sadaļā Profils → Uzstādījumi un Abonements.\n\n"
        f"Veiksmīgu darbu!\n"
        f"THRUVIA / {LEGAL_PROVIDER_DETAILS['company']}\n"
        f"{billing_cfg['support_email']}"
    )

    if not to_email or "@" not in to_email:
        return False, "Klienta profilā nav norādīts e-pasts — e-pasts netika nosūtīts.", subject, body

    resend_key = _get_secret_or_env("RESEND_API_KEY", "")
    email_from = _get_secret_or_env("EMAIL_FROM", f"THRUVIA <{billing_cfg['support_email']}>")
    if resend_key:
        try:
            req_data = json.dumps(
                {"from": email_from, "to": [to_email], "subject": subject, "text": body}
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://api.resend.com/emails",
                data=req_data,
                headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return True, f"Apstiprinājuma e-pasts automātiski nosūtīts uz {to_email}!", subject, body
        except Exception as e:
            return False, f"Resend e-pasta kļūda ({e}).", subject, body

    smtp_host = _get_secret_or_env("SMTP_HOST", "")
    smtp_user = _get_secret_or_env("SMTP_USER", "")
    smtp_pass = _get_secret_or_env("SMTP_PASS", "")
    smtp_port = int(_get_secret_or_env("SMTP_PORT", "587") or "587")
    if smtp_host and smtp_user and smtp_pass:
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = email_from
            msg["To"] = to_email
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as srv:
                srv.starttls()
                srv.login(smtp_user, smtp_pass)
                srv.sendmail(smtp_user, [to_email], msg.as_string())
            return True, f"Apstiprinājuma e-pasts automātiski nosūtīts uz {to_email}!", subject, body
        except Exception as e:
            return False, f"SMTP e-pasta kļūda ({e}).", subject, body

    return (
        False,
        f"Profils apstiprināts! Lai e-pasti sūtītos pilnībā automātiski, pievienojiet `RESEND_API_KEY` vai `SMTP_HOST/SMTP_USER/SMTP_PASS` Streamlit Secrets. Līdz tam variet nospiest pogu zemāk, lai atvērtu gatavu e-pastu uz {to_email}.",
        subject,
        body,
    )


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


def enforce_obvious_classification_rules(label: str, path: str, grp: str, question: str) -> tuple[str, str]:
    """
    Deterministiski izlabo acīmredzamās kļūdas kontu klasifikācijā:
    - Zem algām (OE_ADMIN_WAGES / OE_WORKER_WAGES) TIKAI algas un algu nodokļi!
    - Personāla pasākumi, ilgtspēja, reprezentācija, komandējumi, apmācības -> OE_ADMIN_OTHER (NE algas!).
    - Pārējās pārdošanas izmaksas -> OE_SALES_OTHER (bez liekiem jautājumiem).
    - Pārējās administrācijas izmaksas -> OE_ADMIN_OTHER (bez liekiem jautājumiem).
    - Pārējās saimnieciskās darbības izmaksas -> OE_OTHER (bez liekiem jautājumiem).
    """
    low_lbl = str(label or "").strip().lower()
    low_path = str(path or "").strip().lower()
    grp = normalize_group_key(grp)
    q = str(question or "").strip()

    # 1. Nekad nelikt blakus-izdevumus zem algām
    non_wage_person_kw = (
        "reprezentāc", "ilgtspēj", "pasākum", "apmācīb", "kursi", "komandējum",
        "apdrošināš", "veselīb", "dāvan", "saliedēš", "darba apģērb", "aizsardzīb",
        "personāla atlas", "rekruit", "ēdināš", "kafij",
    )
    if any(kw in low_lbl for kw in non_wage_person_kw):
        if "pārdoš" in low_path:
            return "OE_SALES_OTHER", ""
        return "OE_ADMIN_OTHER", ""

    # 2. Ja klasifikators ielicis pie algām, bet nosaukumā nav algu/atalgojuma/VSAOI vārdu -> pārliekam uz administrācijas/citiem OE
    wage_kw = ("alga", "atalgoj", "darba samaks", "sociāl", "vsaoi", "prēmij", "atvaļinājum", "personāla izmaksas-atlīdzība", "atalgojum")
    if grp in ("OE_ADMIN_WAGES", "OE_WORKER_WAGES") and not any(wk in low_lbl for wk in wage_kw):
        if "pārdoš" in low_path:
            return "OE_SALES_OTHER", ""
        return "OE_ADMIN_OTHER", ""

    # 3. Acīmredzami "Pārējās pārdošanas / administrācijas / saimnieciskās" konti
    if "pārējās pārdošanas" in low_lbl or ("citas pārdošanas" in low_lbl and "platform" not in low_lbl):
        return "OE_SALES_OTHER", ""
    if "pārējās administrācijas" in low_lbl or "citas administrācijas" in low_lbl:
        return "OE_ADMIN_OTHER", ""
    if "pārējās saimnieciskās" in low_lbl or "citas saimnieciskās" in low_lbl:
        return "OE_OTHER", ""

    # 4. Transports administrācijas sadaļā -> Pārējās administrācijas izmaksas (nav TVC piegāde!)
    if "transport" in low_lbl and "administrāc" in low_path:
        return "OE_ADMIN_OTHER", ""

    # 5. Ja jautājums ir tikai par OE apakšgrupas niansi (nevis TVC pret OE vai EXCLUDED), noņemam lieko jautājumu
    if q and grp.startswith("OE_") and not any(w in q.lower() for w in ("tvc", "mainīg", "nolietoj", "amortiz", "virtuāl", "izslēg", "ieņēmum")):
        q = ""

    return grp, q


def smart_classify_accounts(client: anthropic.Anthropic, model: str, leaves: list) -> list[dict]:
    """
    Precīzs THRUVIA kontu klasifikators, kas izmanto aktuālās apakšgrupas (ieskaitot OE_ADMIN_OTHER un OE_SALES_OTHER),
    stingri atdala algas no personāla pasākumiem/reprezentācijas un neuzdod acīmredzamus jautājumus.
    """
    if not leaves:
        return []

    sub_lines = "\n".join(
        f"- `{k}` ({mc}): {lbl}"
        for k, (lbl, mc) in GROUPS.items()
    )
    acc_payload = [
        {
            "id": lf.id,
            "label": str(lf.label).strip(),
            "path": str(getattr(lf, "path", "") or "").strip(),
            "total": round(float(np.sum(lf.vals)), 2) if getattr(lf, "vals", None) is not None else 0.0,
        }
        for lf in leaves
    ]

    sys_msg = (
        "Tu esi precīzs Latvijas grāmatvedības un TOC Caurlaidības grāmatvedības (Throughput Accounting) kontu klasifikators.\n"
        f"Pieejamās grupu atslēgas:\n{sub_lines}\n\n"
        "STINGRI NOTEIKUMI:\n"
        "1. ALGAS (`OE_ADMIN_WAGES`, `OE_WORKER_WAGES`): Šeit liek TIKAI tiešās darba algas, atalgojumu, prēmijas, atvaļinājuma naudas un darba devēja sociālās iemaksas (VSAOI / algu nodokļus).\n"
        "   NEKAD neliec pie algām: personāla ilgtspējas pasākumus, darbinieku pasākumus, reprezentācijas izdevumus, komandējumus, veselības apdrošināšanu, apmācības vai darba apģērbu — tos visus liec pie `OE_ADMIN_OTHER`!\n"
        "2. PĀRĒJĀS ADMINISTRĀCIJAS IZMAKSAS (`OE_ADMIN_OTHER`): reprezentācija, personāla pasākumi/ilgtspēja/apmācības, komandējumi, transporta izdevumi administrācijā, grāmatvedības/juridiskie pakalpojumi, konti ar nosaukumu 'Pārējās administrācijas izmaksas'.\n"
        "3. PĀRĒJĀS PĀRDOŠANAS IZMAKSAS (`OE_SALES_OTHER`): konti ar nosaukumu 'Pārējās pārdošanas izmaksas' un citi pārdošanas sadaļas izdevumi, kas nav reklāma (`OE_MARKETING`), platformas (`OE_PLATFORMS`) vai preču piegāde (`TVC_DELIVERY`).\n"
        "4. CITI SAIMNIECISKIE UZTURĒŠANAS IZDEVUMI (`OE_OTHER`): 'Pārējās saimnieciskās darbības izmaksas' un pārējie vispārējie OE konti.\n"
        "5. TVC (`TVC_MATERIALS`, `TVC_DELIVERY`): TIKAI 1:1 mainīgās izmaksas (izejvielas, preces pārdošanai, tiešā piegāde klientam). Elektrība, noma, remonti, algas ir OE.\n"
        "6. NEIEKĻAUTIE (`EXCLUDED`): pamatlīdzekļu amortizācija, nolietojums, uzkrājumi, pārvērtēšana.\n"
        "7. JAUTĀJUMS (`question`): Atstāj `\"\"` (tukšu) visiem acīmredzamiem kontiem! Uzdod īsu jautājumu latviski TIKAI tad, ja konts ir patiesi neskaidrs starp lielajām grupām (piem., nav skaidrs, vai tas ir TVC vai OE, vai tajā nav nolietojums).\n\n"
        "Atgriez TIKAI derīgu JSON masīvu: "
        '[{"id": "...", "group": "OE_ADMIN_OTHER", "is_income": false, "question": ""}]'
    )

    resp = client.messages.create(
        model=model,
        max_tokens=3000,
        system=sys_msg,
        messages=[{"role": "user", "content": json.dumps(acc_payload, ensure_ascii=False)}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    m = re.search(r"\[.*\]", raw, re.S)
    parsed_list = json.loads(m.group(0)) if m else []
    by_id_raw = {str(it.get("id")): it for it in parsed_list if isinstance(it, dict)}

    final_items = []
    for lf in leaves:
        raw_it = by_id_raw.get(lf.id, {"group": "OE_OTHER", "is_income": False, "question": ""})
        g_clean, q_clean = enforce_obvious_classification_rules(
            lf.label,
            getattr(lf, "path", ""),
            raw_it.get("group", "OE_OTHER"),
            raw_it.get("question", ""),
        )
        final_items.append(
            {
                "id": lf.id,
                "group": g_clean,
                "is_income": g_clean.startswith("S_"),
                "question": q_clean,
            }
        )
    return final_items


def transcribe_wav_bytes_lv(audio_bytes: bytes) -> tuple[str, str]:
    if not audio_bytes or len(audio_bytes) < 100:
        return "", "Audio ieraksts ir tukšs (ja šobrīd runājat MS Teams vai Zoom zvanā, pārlūka mikrofons var būt aizņemts)."
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        return "", "Lai balss atpazīšana strādātu uz servera, pievienojiet rindiņu `SpeechRecognition` savā GitHub failā `requirements.txt`."

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        txt = str(recognizer.recognize_google(audio_data, language="lv-LV")).strip()
        if txt:
            return txt, ""
        return "", "Neizdevās sadzirdēt skaidru tekstu (pārliecinieties, ka mikrofonu neaizņem MS Teams vai cita programma)."
    except sr.UnknownValueError:
        return "", "Ierakstā netika sadzirdēta runa (ja vienlaikus izmantojat MS Teams, mikrofona skaņa var tikt bloķēta pārlūkā)."
    except Exception as e:
        return "", f"Neizdevās atpazīt balsi ({e})."


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


def build_demo_company_data() -> tuple[LoadedReportResult, list[dict]]:
    """
    12 mēnešu Demo uzņēmums ar APAĻIEM, viegli saprotamiem skaitļiem:
    Vidējais S = 120 000 €, TVC = 48 000 € (40%), T = 72 000 € (60%), OE = 60 000 € (50%), NP = 12 000 € (10%), PR = 1.20.
    """
    months = [f"{m:02d}.2026." for m in range(1, 13)]
    s_rev =   [95000, 95000, 105000, 115000, 105000, 115000, 110000, 115000, 125000, 130000, 135000, 145000]
    s_oth =   [5000,  5000,  5000,   5000,   5000,   5000,   5000,   5000,   5000,   5000,   5000,   5000]
    tvc_mat = [35000, 35000, 38000,  42000,  38000,  42000,  40000,  42000,  45000,  47000,  48000,  52000]
    tvc_del = [5000,  5000,  6000,   6000,   6000,   6000,   6000,   6000,   7000,   7000,   8000,   8000]
    oe_adm =  [15000, 15000, 15000,  15000,  15000,  15000,  15000,  15000,  15000,  15000,  15000,  15000]
    oe_wrk =  [20000, 20000, 20000,  20000,  20000,  20000,  20000,  20000,  20000,  20000,  20000,  20000]
    oe_mkt =  [8000,  8000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  12000,  12000,  12000]
    oe_prem = [7000,  7000,  7000,   7000,   7000,   7000,   7000,   7000,   7000,   7000,   7000,   7000]
    oe_it =   [4000,  4000,  4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000]
    oe_adm_o = [4000, 4000,  4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000,   4000]

    lbl_s_rev = SUBGROUP_LABELS["S_REV"]
    lbl_s_oth = SUBGROUP_LABELS["S_OTHER"]
    lbl_mat = SUBGROUP_LABELS["TVC_MATERIALS"]
    lbl_del = SUBGROUP_LABELS["TVC_DELIVERY"]
    lbl_adm = SUBGROUP_LABELS["OE_ADMIN_WAGES"]
    lbl_wrk = SUBGROUP_LABELS["OE_WORKER_WAGES"]
    lbl_mkt = SUBGROUP_LABELS["OE_MARKETING"]
    lbl_prem = SUBGROUP_LABELS["OE_PREMISES"]
    lbl_it = SUBGROUP_LABELS["OE_OFFICE_IT"]
    lbl_adm_o = SUBGROUP_LABELS["OE_ADMIN_OTHER"]

    s_tot = [a + b for a, b in zip(s_rev, s_oth)]
    tvc_tot = [a + b for a, b in zip(tvc_mat, tvc_del)]
    t_tot = [a - b for a, b in zip(s_tot, tvc_tot)]
    oe_tot = [sum(x) for x in zip(oe_adm, oe_wrk, oe_mkt, oe_prem, oe_it, oe_adm_o)]
    np_tot = [a - b for a, b in zip(t_tot, oe_tot)]

    data_rows = {
        "S": s_tot,
        lbl_s_rev: s_rev,
        lbl_s_oth: s_oth,
        "TVC": tvc_tot,
        lbl_mat: tvc_mat,
        lbl_del: tvc_del,
        "T": t_tot,
        "OE": oe_tot,
        lbl_adm: oe_adm,
        lbl_wrk: oe_wrk,
        lbl_mkt: oe_mkt,
        lbl_prem: oe_prem,
        lbl_it: oe_it,
        lbl_adm_o: oe_adm_o,
        "NP": np_tot,
    }
    row_kinds = {
        "S": "S",
        lbl_s_rev: "S_CHILD",
        lbl_s_oth: "S_CHILD",
        "TVC": "TVC",
        lbl_mat: "TVC_CHILD",
        lbl_del: "TVC_CHILD",
        "T": "T",
        "OE": "OE",
        lbl_adm: "OE_CHILD",
        lbl_wrk: "OE_CHILD",
        lbl_mkt: "OE_CHILD",
        lbl_prem: "OE_CHILD",
        lbl_it: "OE_CHILD",
        lbl_adm_o: "OE_CHILD",
        "NP": "NP",
    }

    amounts = pd.DataFrame(data_rows, index=months).T.astype(float)
    amounts["2026 KOPĀ"] = amounts[months].sum(axis=1)
    amounts["2026 AVG"] = amounts[months].mean(axis=1)

    main_dict = {}
    for c in amounts.columns:
        main_dict[c] = amounts[c]
        sv = amounts.loc["S", c]
        main_dict[f"%|{c}"] = np.where(sv != 0, amounts[c] / sv, np.nan)
    main_df = pd.DataFrame(main_dict, index=amounts.index)
    kpi_df = pd.DataFrame({c: _calc_kpi_for_series(amounts[c]) for c in amounts.columns})

    res_demo = LoadedReportResult(
        amounts=amounts,
        main=main_df,
        kpi=kpi_df,
        row_kinds=row_kinds,
        reconciliation=pd.DataFrame(),
        by_group={},
    )

    demo_accounts = [
        ("demo_1", 4, "Produkcijas pārdošanas ieņēmumi", "Ieņēmumi", "S_REV", s_rev),
        ("demo_2", 5, "Pārējie saimnieciskie ieņēmumi", "Ieņēmumi", "S_OTHER", s_oth),
        ("demo_3", 9, "Izejvielu un materiālu iepirkumi", "Ražošanas izmaksas", "TVC_MATERIALS", [-v for v in tvc_mat]),
        ("demo_4", 10, "Preču piegādes pakalpojumi klientiem", "Pārdošanas izmaksas", "TVC_DELIVERY", [-v for v in tvc_del]),
        ("demo_5", 14, "Administrācijas darba algas un VSAOI", "Administrācijas izmaksas", "OE_ADMIN_WAGES", [-v for v in oe_adm]),
        ("demo_6", 15, "Ražošanas strādnieku algas un VSAOI", "Ražošanas izmaksas", "OE_WORKER_WAGES", [-v for v in oe_wrk]),
        ("demo_7", 18, "Digitālais mārketings un reklāma", "Pārdošanas izmaksas", "OE_MARKETING", [-v for v in oe_mkt]),
        ("demo_8", 21, "Telpu noma un komunālie pakalpojumi", "Telpu uzturēšana", "OE_PREMISES", [-v for v in oe_prem]),
        ("demo_9", 24, "IT programmatūra un sakari", "Administrācijas izmaksas", "OE_OFFICE_IT", [-v for v in oe_it]),
        ("demo_10", 27, "Grāmatvedība, reprezentācija un pārējās admin. izmaksas", "Administrācijas izmaksas", "OE_ADMIN_OTHER", [-v for v in oe_adm_o]),
        ("demo_11", 31, "Pamatlīdzekļu nolietojums (amortizācija)", "Nolietojums", "EXCLUDED", [-5000.0] * 12),
    ]
    acc_snap = []
    for aid, rnum, albl, apath, agrp, avals in demo_accounts:
        acc_snap.append(
            {
                "id": aid,
                "row": rnum,
                "label": albl,
                "display_label": albl,
                "path": apath,
                "group": agrp,
                "months": {months[i]: float(avals[i]) for i in range(12)},
                "total": float(sum(avals)),
            }
        )

    return res_demo, acc_snap


def submit_pza_for_admin_review(
    profile_store: SupabaseProfileStore,
    filename: str,
    file_bytes: bytes,
    mode: str,
    user_note: str = "",
) -> tuple[bool, str]:
    if len(file_bytes) > 6 * 1024 * 1024:
        return False, "Fails ir lielāks par 6 MB. Lūdzu saglabājiet PZA bez liekām darblapām un mēģiniet vēlreiz."
    try:
        b64_str = base64.b64encode(file_bytes).decode("ascii")
        payload = json.dumps(
            {
                "filename": filename,
                "b64": b64_str,
                "mode": mode,
                "user_note": user_note.strip(),
                "submitted_at": now_riga().strftime("%d.%m.%Y %H:%M"),
                "size_kb": round(len(file_bytes) / 1024.0, 1),
            },
            ensure_ascii=False,
        )
        dummy_am = pd.DataFrame({"Pārbaude": [0.0]}, index=["S"])
        dummy_main = pd.DataFrame({"Pārbaude": [0.0], "%|Pārbaude": [0.0]}, index=["S"])
        dummy_kpi = pd.DataFrame({"Pārbaude": [0.0]}, index=["PR = T/OE"])
        dummy_res = LoadedReportResult(
            amounts=dummy_am,
            main=dummy_main,
            kpi=dummy_kpi,
            row_kinds={"S": "S"},
            reconciliation=pd.DataFrame(),
            by_group={},
        )
        ok_s, _ = profile_store.save_report(
            report_name=f"[PZA PĀRBAUDEI] {filename}",
            file_name=f"PENDING_PZA::{filename}",
            res_obj=dummy_res,
            commentary=payload,
        )
        if not ok_s:
            return False, f"Neizdevās saglabāt failu pārbaudei: {profile_store.last_error}"
        return True, "PZA fails veiksmīgi nosūtīts pārbaudei!"
    except Exception as e:
        return False, f"Kļūda faila nosūtīšanā: {e}"


def render_legal_documents_tabs(compact: bool = False) -> None:
    t_disc, t_terms, t_priv = st.tabs(
        [
            "Atbildības ierobežojums",
            "Lietošanas noteikumi",
            "Privātuma politika (VDAR)",
        ]
    )
    prov = LEGAL_PROVIDER_DETAILS

    with t_disc:
        st.markdown(
            f"""
            **Atbildības ierobežojums un finanšu padomu atruna**

            1. **Tikai informatīvs un modelēšanas nolūks:** Sistēma **THRUVIA**, tās aprēķini, grafiki, scenāriju simulācijas un mākslīgā intelekta (MI) komentāri ir paredzēti **tikai iekšējai informatīvai analīzei un vadības lēmumu modelēšanai**, balstoties uz Caurlaidības grāmatvedības (*Throughput Accounting*) metodoloģiju.
            2. **Nav finanšu, nodokļu vai grāmatvedības konsultācija:** Ne sistēma THRUVIA, ne **{prov['company']}** nesniedz licencētas finanšu, investīciju, nodokļu, juridiskās vai zvērinātu revidentu konsultācijas. Sistēma neaizstāj oficiālo grāmatvedības uzskaiti un normatīvos pārskatus.
            3. **Datu precizitāte un MI ierobežojumi:** Aprēķinu precizitāte ir tieši atkarīga no Lietotāja augšupielādēto datu pareizības un kontu klasifikācijas. Lietotājs apņemas pārbaudīt skaitļus pirms biznesa lēmumu pieņemšanas.
            4. **Atbildības neuzņemšanās:** Pakalpojuma sniedzējs neuzņemas nekādu tiesisku vai finansiālu atbildību par tiešiem vai netiešiem zaudējumiem, negūtu peļņu vai citiem rezultātiem, kas radušies, Lietotājam pieņemot lēmumus, balstoties uz THRUVIA sniegto informāciju.
            """
        )

    with t_terms:
        st.markdown(
            f"""
            **THRUVIA Lietošanas noteikumi**

            * **Pakalpojuma sniedzējs:** {prov['company']}, reģ. nr. {prov['reg_no']}, adrese: {prov['address']}, e-pasts: `{prov['email']}`.
            * **Pakalpojuma būtība:** Piekļuve tīmekļa lietotnei THRUVIA uz abonementa pamata (SaaS). Pirmreizējiem lietotājiem tiek nodrošināts 5 dienu bezmaksas izmēģinājums pēc pirmā PZA apstiprināšanas.
            * **Abonēšana un atteikšanās:** Pēc izmēģinājuma perioda piekļuve turpinās par spēkā esošo mēneša maksu (automātiska karšu apmaksa caur Stripe). Atcelt abonementu var jebkurā brīdī sadaļā *Profils → Uzstādījumi un Abonements*.
            * **Datu īpašumtiesības:** Lietotāja augšupielādētie finanšu dati pilnībā pieder Lietotājam.
            """
        )

    with t_priv:
        st.markdown(
            f"""
            **Privātuma politika (VDAR / GDPR)**

            * **Datu pārzinis:** {prov['company']}, reģ. nr. {prov['reg_no']}, adrese: {prov['address']}, e-pasts: `{prov['email']}`.
            * **Apstrādātie dati:** Konta dati (uzņēmums, kontaktpersona, e-pasts, šifrēta parole), Lietotāja augšupielādētie PZA dati un simulācijas. Karšu datus apstrādā tikai **Stripe** (THRUVIA tos neglabā).
            * **MI un datu drošība:** Datubāze tiek glabāta **Supabase**, MI funkcijas nodrošina **Anthropic Claude API** — saskaņā ar API komerciālajiem noteikumiem **Lietotāja finanšu dati netiek izmantoti MI modeļu apmācībai**.
            * **Lietotāja tiesības:** Lietotājs jebkurā brīdī var pieprasīt savu datu eksportu vai pilnīgu dzēšanu, rakstot uz `{prov['email']}`.
            """
        )


def render_app_footer_disclaimer() -> None:
    st.markdown(
        f'<div style="margin-top:32px; padding-top:10px; border-top:1px solid rgba(120,120,120,0.22); '
        f'font-size:11px; opacity:0.72; text-align:center; line-height:1.45;">'
        f'{LEGAL_DISCLAIMER_SHORT} &nbsp;|&nbsp; © {datetime.now().year} THRUVIA ({LEGAL_PROVIDER_DETAILS["company"]})'
        f'</div>',
        unsafe_allow_html=True,
    )


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
    1: "Janvāris", 2: "Februāris", 3: "Marts", 4: "Aprīlis",
    5: "Maijs", 6: "Jūnijs", 7: "Jūlijs", 8: "Augusts",
    9: "Septembris", 10: "Oktobris", 11: "Novembris", 12: "Decembris",
}

LV_MONTH_PARSE = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mai": 5, "may": 5,
    "jūn": 6, "jun": 6, "jūl": 7, "jul": 7, "aug": 8, "sep": 9,
    "okt": 10, "oct": 10, "nov": 11, "dec": 12,
}


def parse_period_month_year(col_name: str) -> tuple[int | None, int | None]:
    s = str(col_name or "").strip().lower()
    if "ytd" in s or "avg" in s or "kopā" in s or "izmaiņa" in s or "total" in s:
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
    drop_rows = [
        r for r in res.amounts.index
        if res.row_kinds.get(r) in ("EXCL", "EXCLUDED")
        or r == EXCLUDED_LABEL
        or str(r).lower().startswith("izslēgt")
        or str(r).lower().startswith("neiekļaut")
    ]
    if drop_rows:
        res.amounts = res.amounts.drop(index=drop_rows, errors="ignore")
        res.main = res.main.drop(index=drop_rows, errors="ignore")
        for dr in drop_rows:
            res.row_kinds.pop(dr, None)
    return res


def apply_custom_row_order_to_result(res):
    if res is None or res.amounts is None or res.amounts.empty:
        return res
    custom_ord: dict[str, list[str]] = st.session_state.get("custom_row_order", {})

    idx_list = list(res.amounts.index)
    rk = res.row_kinds

    s_ch = [r for r in idx_list if rk.get(r) == "S_CHILD"]
    tvc_ch = [r for r in idx_list if rk.get(r) == "TVC_CHILD"]
    oe_ch = [r for r in idx_list if rk.get(r) == "OE_CHILD"]

    def _sort_sub(curr: list[str], pref: list[str], pin_wages_top: bool = False) -> list[str]:
        if pin_wages_top:
            wages = [x for x in curr if is_wage_row_label(x)]
            non_wages = [x for x in curr if x not in wages]
            wages.sort(key=lambda w: 0 if "admin" in w.lower() else 1)
            if pref:
                ordered_nw = [x for x in pref if x in non_wages]
                rest_nw = [x for x in non_wages if x not in ordered_nw]
                return wages + ordered_nw + rest_nw
            return wages + non_wages
        else:
            if not pref:
                return curr
            ordered = [x for x in pref if x in curr]
            rest = [x for x in curr if x not in ordered]
            return ordered + rest

    s_ch = _sort_sub(s_ch, custom_ord.get("S", []), pin_wages_top=False)
    tvc_ch = _sort_sub(tvc_ch, custom_ord.get("TVC", []), pin_wages_top=False)
    oe_ch = _sort_sub(oe_ch, custom_ord.get("OE", []), pin_wages_top=True)

    new_idx = ["S"] + s_ch + ["TVC"] + tvc_ch + ["T", "OE"] + oe_ch + ["NP"]
    new_idx = [r for r in new_idx if r in res.amounts.index]
    res.amounts = res.amounts.loc[new_idx]
    res.main = res.main.loc[new_idx]
    return res


def reorder_table_subgroups(
    res,
    main_code: str = "OE",
    mode: str = "desc",
    explicit_order: list[str] | None = None,
    keep_other_last: bool = True,
) -> list[str]:
    if res is None or res.amounts is None:
        return []
    mc = main_code.upper()
    if mc not in ("S", "TVC", "OE"):
        mc = "OE"

    kind_tag = f"{mc}_CHILD"
    children = [r for r in res.amounts.index if res.row_kinds.get(r) == kind_tag]
    if not children:
        return []

    wages = [c for c in children if mc == "OE" and is_wage_row_label(c)]
    wages.sort(key=lambda w: 0 if "admin" in w.lower() else 1)
    rem_children = [c for c in children if c not in wages]

    month_cols = [c for c in res.amounts.columns if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper())]
    if not month_cols:
        month_cols = list(res.amounts.columns)

    if mode == "custom" and explicit_order:
        matched = []
        for req in explicit_order:
            req_l = req.strip().lower()
            for ch in rem_children:
                if ch not in matched and (ch.lower() == req_l or req_l in ch.lower() or ch.lower() in req_l):
                    matched.append(ch)
        rest = [c for c in rem_children if c not in matched]
        sorted_ch = wages + matched + rest
    else:
        totals_map = {ch: float(res.amounts.loc[ch, month_cols].sum()) for ch in rem_children}
        rev = mode != "asc"
        if keep_other_last:
            regular = [c for c in rem_children if not (c.lower().startswith("citi ") or c.lower().startswith("pārēj"))]
            others = [c for c in rem_children if c not in regular]
            regular.sort(key=lambda x: totals_map.get(x, 0.0), reverse=rev)
            others.sort(key=lambda x: totals_map.get(x, 0.0), reverse=rev)
            sorted_ch = wages + regular + others
        else:
            sorted_ch = wages + sorted(rem_children, key=lambda x: totals_map.get(x, 0.0), reverse=rev)

    st.session_state.setdefault("custom_row_order", {})[mc] = sorted_ch
    apply_custom_row_order_to_result(res)
    st.session_state["has_unsaved_changes"] = True
    return sorted_ch


def save_report_as_new_or_update(
    profile_store: SupabaseProfileStore,
    res_to_save,
    new_report_name: str,
    acc_details: list[dict] | None = None,
) -> tuple[bool, int | None]:
    if st.session_state.get("is_demo_mode", False):
        return False, None
    rep_meta = st.session_state.get("loaded_report_meta") or {}
    packed_c = pack_commentary_with_account_details(
        st.session_state.get("commentary", ""),
        acc_details if acc_details is not None else st.session_state.get("loaded_account_details"),
    )
    ok_r, rep_id = profile_store.save_report(
        report_name=new_report_name.strip() or rep_meta.get("report_name", "Atskaite"),
        file_name=rep_meta.get("file_name", "PZA.xlsx"),
        res_obj=res_to_save,
        commentary=packed_c,
    )
    if ok_r and rep_id:
        st.session_state["current_report_id"] = rep_id
        st.session_state["loaded_report_meta"] = {
            "id": rep_id,
            "report_name": new_report_name.strip(),
            "file_name": rep_meta.get("file_name", "PZA.xlsx"),
        }
        st.session_state["has_unsaved_changes"] = False
    return ok_r, rep_id


def rename_subgroup_everywhere(old_name: str, new_name: str, profile_store: SupabaseProfileStore) -> tuple[bool, str]:
    old_clean = str(old_name or "").strip()
    new_clean = str(new_name or "").strip()
    if not old_clean or not new_clean:
        return False, "Norādiet gan veco, gan jauno apakšgrupas nosaukumu."

    target_key = LABEL_TO_KEY.get(old_clean)
    if not target_key:
        for k, lbl in SUBGROUP_LABELS.items():
            if lbl.lower() == old_clean.lower():
                target_key = k
                old_clean = lbl
                break
    if not target_key:
        for k, lbl in SUBGROUP_LABELS.items():
            if old_clean.lower() in lbl.lower() or lbl.lower() in old_clean.lower():
                target_key = k
                old_clean = lbl
                break

    res = st.session_state.get("result")
    if not target_key and res is not None:
        for r_idx in res.amounts.index:
            if r_idx.lower() == old_clean.lower() or old_clean.lower() in r_idx.lower():
                old_clean = r_idx
                target_key = LABEL_TO_KEY.get(r_idx)
                break

    if target_key:
        mc = get_main_code(target_key)
        register_custom_subgroup(target_key, new_clean, mc)
        st.session_state.setdefault("custom_subgroups", {})[target_key] = (new_clean, mc)

    if res is not None and old_clean in res.amounts.index:
        res.amounts = res.amounts.rename(index={old_clean: new_clean})
        res.main = res.main.rename(index={old_clean: new_clean})
        if old_clean in res.row_kinds:
            kind_val = res.row_kinds.pop(old_clean)
            res.row_kinds[new_clean] = kind_val

    for mc_k, ord_list in st.session_state.get("custom_row_order", {}).items():
        st.session_state["custom_row_order"][mc_k] = [new_clean if x == old_clean else x for x in ord_list]

    st.session_state["has_unsaved_changes"] = True
    return True, f"Apakšgrupa **'{old_clean}'** pārdēvēta par **'{new_clean}'**!"


def merge_subgroups_everywhere(source_sub_label: str, target_sub_label: str, profile_store: SupabaseProfileStore) -> tuple[bool, str]:
    src_lbl = str(source_sub_label or "").strip()
    tgt_lbl = str(target_sub_label or "").strip()
    if not src_lbl or not tgt_lbl or src_lbl.lower() == tgt_lbl.lower():
        return False, "Izvēlieties divas dažādas apakšgrupas, ko apvienot."

    src_key = LABEL_TO_KEY.get(src_lbl)
    tgt_key = LABEL_TO_KEY.get(tgt_lbl)
    if not src_key:
        for k, v in SUBGROUP_LABELS.items():
            if src_lbl.lower() in v.lower():
                src_key, src_lbl = k, v
                break
    if not tgt_key:
        for k, v in SUBGROUP_LABELS.items():
            if tgt_lbl.lower() in v.lower():
                tgt_key, tgt_lbl = k, v
                break

    if not tgt_key:
        return False, f"Mērķa apakšgrupa '{tgt_lbl}' netika atrasta."

    S_loc = st.session_state
    moved_cnt = 0

    if "cls_df" in S_loc and S_loc.cls_df is not None and not S_loc.cls_df.empty and src_key:
        mask = S_loc.cls_df["Grupa"] == src_key
        moved_cnt = int(mask.sum())
        S_loc.cls_df.loc[mask, "Grupa"] = tgt_key
        S_loc.cls_df.loc[mask, "Ieņēmums"] = tgt_key.startswith("S_")
        S_loc.cls_df = sort_cls_df(S_loc.cls_df)
        if "analysis" in S_loc and S_loc.analysis is not None:
            an = S_loc.analysis
            stc = S_loc.structure
            mapping_upd = {
                r_u["id"]: {
                    "group": normalize_group_key(r_u["Grupa"]),
                    "is_income": bool(str(r_u["Grupa"]).startswith("S_")),
                }
                for _, r_u in S_loc.cls_df.iterrows()
            }
            pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
            reported = an.ex.lines[pos].vals if pos is not None else None
            S_loc.result = compute_toc_with_dynamic_subgroups(an.leaves, mapping_upd, an.ex.period_cols, reported)
            S_loc.loaded_account_details = build_account_details_snapshot(an.leaves, S_loc.cls_df, an.ex.period_cols)
            if not S_loc.get("is_demo_mode", False):
                profile_store.save(an.leaves, mapping_upd)

    elif S_loc.get("loaded_account_details") and src_key:
        snap_list = S_loc.get("loaded_account_details") or []
        for it in snap_list:
            if normalize_group_key(it.get("group", "")) == src_key:
                it["group"] = tgt_key
                moved_cnt += 1
        S_loc.loaded_account_details = snap_list
        if S_loc.get("result") is not None:
            S_loc.result = rebuild_report_from_account_snapshot(S_loc.result, snap_list)

    elif S_loc.get("result") is not None:
        res = S_loc.result
        if src_lbl in res.amounts.index and tgt_lbl in res.amounts.index:
            res.amounts.loc[tgt_lbl] = res.amounts.loc[tgt_lbl] + res.amounts.loc[src_lbl]
            res.amounts = res.amounts.drop(index=[src_lbl])
            res.row_kinds.pop(src_lbl, None)
            new_main_d = {}
            for c in res.amounts.columns:
                new_main_d[c] = res.amounts[c]
                sv = res.amounts.loc["S", c]
                new_main_d[f"%|{c}"] = np.where(sv != 0, res.amounts[c] / sv, np.nan)
            res.main = pd.DataFrame(new_main_d, index=res.amounts.index)

    if src_key and not S_loc.get("is_demo_mode", False):
        saved_m = profile_store.list_saved_mappings()
        db_upd = [
            {"id": int(m["id"]), "toc_group": tgt_key}
            for m in saved_m
            if normalize_group_key(m.get("toc_group", "")) == src_key
        ]
        if db_upd:
            profile_store.update_saved_mappings_direct(db_upd)
            if moved_cnt == 0:
                moved_cnt = len(db_upd)

    S_loc["has_unsaved_changes"] = True
    return True, f"Apakšgrupa **'{src_lbl}'** ({fmt_acc_count_lv(moved_cnt)}) ir pievienota pie **'{tgt_lbl}'**!"


def move_specific_accounts_to_subgroup(
    account_names_or_ids: list[str],
    target_sub_label: str,
    main_code: str,
    profile_store: SupabaseProfileStore,
) -> tuple[bool, str]:
    clean_target = str(target_sub_label or "").strip()
    if not clean_target:
        return False, "Norādiet mērķa apakšgrupas nosaukumu."
    mc = str(main_code or "OE").upper()
    if mc not in ("S", "TVC", "OE", "EXCL"):
        mc = "OE"

    if clean_target in LABEL_TO_KEY:
        target_g_key = LABEL_TO_KEY[clean_target]
    else:
        target_g_key = f"{mc}_CUSTOM_{hashlib.md5(clean_target.encode()).hexdigest()[:6].upper()}"
        st.session_state.setdefault("custom_subgroups", {})[target_g_key] = (clean_target, mc)
        register_custom_subgroup(target_g_key, clean_target, mc)

    if not account_names_or_ids:
        return True, f"Izveidota jauna apakšgrupa **'{clean_target}'**!"

    S_loc = st.session_state
    targets_low = {str(x).strip().lower() for x in account_names_or_ids if str(x).strip()}
    moved_cnt = 0

    if "cls_df" in S_loc and S_loc.cls_df is not None and not S_loc.cls_df.empty:
        mask = (
            S_loc.cls_df["id"].astype(str).str.lower().isin(targets_low)
            | S_loc.cls_df["Konts"].astype(str).str.lower().isin(targets_low)
            | S_loc.cls_df["Displeja_konts"].astype(str).str.lower().isin(targets_low)
        )
        if mask.any():
            S_loc.cls_df.loc[mask, "Grupa"] = target_g_key
            S_loc.cls_df.loc[mask, "Ieņēmums"] = target_g_key.startswith("S_")
            moved_cnt += int(mask.sum())
        S_loc.cls_df = sort_cls_df(S_loc.cls_df)
        if "analysis" in S_loc and S_loc.analysis is not None:
            an = S_loc.analysis
            stc = S_loc.structure
            mapping_upd = {
                r_u["id"]: {
                    "group": normalize_group_key(r_u["Grupa"]),
                    "is_income": bool(str(r_u["Grupa"]).startswith("S_")),
                }
                for _, r_u in S_loc.cls_df.iterrows()
            }
            pos = next((l.pos for l in an.ex.lines if l.row == stc.result_row), None)
            reported = an.ex.lines[pos].vals if pos is not None else None
            S_loc.result = compute_toc_with_dynamic_subgroups(an.leaves, mapping_upd, an.ex.period_cols, reported)
            S_loc.loaded_account_details = build_account_details_snapshot(an.leaves, S_loc.cls_df, an.ex.period_cols)
            if not S_loc.get("is_demo_mode", False):
                profile_store.save(an.leaves, mapping_upd)

    elif S_loc.get("loaded_account_details"):
        snap_list = S_loc.get("loaded_account_details") or []
        for it in snap_list:
            if (
                str(it.get("id", "")).strip().lower() in targets_low
                or str(it.get("label", "")).strip().lower() in targets_low
                or str(it.get("display_label", "")).strip().lower() in targets_low
            ):
                it["group"] = target_g_key
                moved_cnt += 1
        S_loc.loaded_account_details = snap_list
        if S_loc.get("result") is not None:
            S_loc.result = rebuild_report_from_account_snapshot(S_loc.result, snap_list)

    if not S_loc.get("is_demo_mode", False):
        saved_m = profile_store.list_saved_mappings()
        db_updates = []
        for m_row in saved_m:
            if (
                str(m_row.get("account_label", "")).strip().lower() in targets_low
                or str(m_row.get("id", "")).strip().lower() in targets_low
            ):
                db_updates.append({"id": int(m_row["id"]), "toc_group": target_g_key})
        if db_updates:
            profile_store.update_saved_mappings_direct(db_updates)
            if moved_cnt == 0:
                moved_cnt = len(db_updates)

    S_loc["has_unsaved_changes"] = True
    return True, f"Pārvietoti **{fmt_acc_count_lv(moved_cnt)}** uz apakšgrupu **'{clean_target}'**!"


def compute_toc_with_dynamic_subgroups(leaves, mapping: dict[str, dict], period_cols: list[dict], reported=None):
    res = compute_toc(leaves, mapping, period_cols, reported)
    res = strip_excluded_rows(res)

    month_labels = [p["label"] for p in period_cols if p.get("kind") == "month"]
    all_cols = list(res.amounts.columns)

    used_groups = {m["group"] for m in mapping.values() if m.get("group") and m["group"] != "EXCLUDED"}
    missing_groups = [g for g in used_groups if GROUPS.get(g, ("", ""))[0] not in res.amounts.index]

    if not missing_groups:
        return apply_custom_row_order_to_result(res)

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
    return apply_custom_row_order_to_result(strip_excluded_rows(res))


def _calc_kpi_for_series(col_s: pd.Series) -> dict[str, float]:
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


def filter_result_by_months(
    res,
    selected_months: list[str],
    all_months: list[str],
    summary_source_months: list[str] | None = None,
    summary_label_prefix: str | None = None,
):
    if not selected_months:
        return res

    sub_amounts = res.amounts[selected_months].copy().astype(float)
    sum_months = summary_source_months if summary_source_months else selected_months
    sum_df = res.amounts[sum_months].copy().astype(float)

    if summary_label_prefix is not None:
        yr_prefix = f"{summary_label_prefix} " if summary_label_prefix else ""
    else:
        years_found = []
        for m_col in sum_months:
            _, yr = parse_period_month_year(m_col)
            if yr and yr not in years_found:
                years_found.append(yr)
        yr_prefix = f"{years_found[-1]} " if len(years_found) == 1 else "Perioda "

    ytd_col = f"{yr_prefix}KOPĀ".strip()
    avg_col = f"{yr_prefix}AVG".strip()

    sub_amounts[ytd_col] = sum_df.sum(axis=1)
    sub_amounts[avg_col] = sum_df.mean(axis=1)

    sub_main_dict = {}
    for c in sub_amounts.columns:
        sub_main_dict[c] = sub_amounts[c]
        s_val = sub_amounts.loc["S", c] if "S" in sub_amounts.index else 0.0
        sub_main_dict[f"%|{c}"] = np.where(s_val != 0, sub_amounts[c] / s_val, np.nan)
    sub_main = pd.DataFrame(sub_main_dict, index=sub_amounts.index)

    kpi_cols_dict = {c: _calc_kpi_for_series(sub_amounts[c]) for c in sub_amounts.columns}
    sub_kpi = pd.DataFrame(kpi_cols_dict)

    return LoadedReportResult(
        amounts=sub_amounts,
        main=sub_main,
        kpi=sub_kpi,
        row_kinds=dict(res.row_kinds),
        reconciliation=getattr(res, "reconciliation", pd.DataFrame()),
        by_group=getattr(res, "by_group", {}),
    )


def aggregate_result_by_years(res, selected_months: list[str]):
    by_yr: dict[str, list[str]] = {}
    for m_col in selected_months:
        _, yr = parse_period_month_year(m_col)
        yr_lbl = f"{yr}. gads" if yr else "Periods"
        by_yr.setdefault(yr_lbl, []).append(m_col)

    yr_cols_data = {}
    for yr_lbl, m_list in by_yr.items():
        col_title = f"{yr_lbl} ({len(m_list)}m)" if len(m_list) < 12 else yr_lbl
        yr_cols_data[col_title] = res.amounts[m_list].astype(float).sum(axis=1)

    agg_amounts = pd.DataFrame(yr_cols_data, index=res.amounts.index)
    full_sel_df = res.amounts[selected_months].astype(float)
    agg_amounts["Visi KOPĀ"] = full_sel_df.sum(axis=1)
    agg_amounts["AVG / mēn."] = full_sel_df.mean(axis=1)

    agg_main_dict = {}
    for c in agg_amounts.columns:
        agg_main_dict[c] = agg_amounts[c]
        s_val = agg_amounts.loc["S", c] if "S" in agg_amounts.index else 0.0
        agg_main_dict[f"%|{c}"] = np.where(s_val != 0, agg_amounts[c] / s_val, np.nan)
    agg_main = pd.DataFrame(agg_main_dict, index=agg_amounts.index)
    agg_kpi = pd.DataFrame({c: _calc_kpi_for_series(agg_amounts[c]) for c in agg_amounts.columns})

    return LoadedReportResult(
        amounts=agg_amounts,
        main=agg_main,
        kpi=agg_kpi,
        row_kinds=dict(res.row_kinds),
        reconciliation=getattr(res, "reconciliation", pd.DataFrame()),
        by_group=getattr(res, "by_group", {}),
    )


def compute_cumulative_toc_df(amounts_df: pd.DataFrame, fy_start_month: int = 1) -> tuple[pd.DataFrame, list[str], set[str]]:
    target_rows = [r for r in ["S", "TVC", "T", "OE", "NP"] if r in amounts_df.index]
    period_cols = [
        c for c in amounts_df.columns
        if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper() or "IZMAIŅA" in str(c).upper() or "TOTAL" in str(c).upper())
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


# === BEIDZAS 1/3 DAĻA (zem šīs rindas līmēsi 2/3 daļu) ===
# === SĀKAS 2/3 DAĻA ===

def render_overview_charts(
    amounts_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    fy_start_month: int = 1,
    highlight_charts: bool = False,
) -> None:
    """
    Centrēts un izteiksmīgs (height=560) Galveno rādītāju dinamikas grafiks + divi KPI grafiki apakšā.
    """
    period_cols = [
        c for c in amounts_df.columns
        if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper() or "IZMAIŅA" in str(c).upper() or "TOTAL" in str(c).upper())
    ]
    if not period_cols:
        return

    st.markdown("---")
    if highlight_charts:
        st.markdown(
            '<div id="tut-focus-anchor" class="tut-section-tag">👉 5. SOLIS — FINANŠU UN PRODUKTIVITĀTES GRAFIKI</div>',
            unsafe_allow_html=True,
        )
    st.subheader("Finanšu un produktivitātes dinamika")

    _, ch_wrap_c, _ = st.columns([0.55, 3.9, 0.55])
    with ch_wrap_c:
        with st.container(border=True):
            ch_top1, ch_top2 = st.columns([2.1, 1.1], vertical_alignment="center")
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

            y_min = float(long_df["Summa (€)"].min()) if not long_df.empty else 0.0
            y_max = float(long_df["Summa (€)"].max()) if not long_df.empty else 1000.0
            pad = max((y_max - y_min) * 0.04, 500.0)
            x_angle = -35 if len(period_cols) > 14 else 0

            base = alt.Chart(long_df).encode(
                x=alt.X(
                    "Periods:N",
                    sort=period_cols,
                    title=None,
                    axis=alt.Axis(labelAngle=x_angle, labelFontWeight="bold", labelFontSize=11),
                    scale=alt.Scale(padding=0.15),
                ),
                y=alt.Y(
                    "Summa (€):Q",
                    title="Summa (€)",
                    scale=alt.Scale(domain=[y_min - pad, y_max + pad], zero=False, nice=True),
                    axis=alt.Axis(format=",.0f", tickCount=14, grid=True),
                ),
                color=alt.Color(
                    "Rādītājs:N",
                    scale=color_scale,
                    legend=alt.Legend(title=None, orient="top", labelFontSize=12),
                ),
                tooltip=[
                    alt.Tooltip("Periods:N", title="Periods"),
                    alt.Tooltip("Rādītājs:N", title="Rādītājs"),
                    alt.Tooltip("Summa (€):Q", title="Summa (€)", format=",.0f"),
                ],
            )

            lines = base.mark_line(strokeWidth=3.4)
            points = base.mark_circle(size=90)
            zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="#475569", strokeWidth=1.5).encode(y="y:Q")
            st.altair_chart((zero_rule + lines + points).properties(height=560).interactive(), use_container_width=True)

    kpi_m_cols = [c for c in period_cols if c in kpi_df.columns]
    if not kpi_m_cols:
        return

    kpi_angle = -35 if len(kpi_m_cols) > 12 else 0
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
                x=alt.X("Periods:N", sort=kpi_m_cols, title=None, axis=alt.Axis(labelAngle=kpi_angle)),
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
            st.altair_chart((ref_one + fact_prod + trend_prod).properties(height=360).interactive(), use_container_width=True)

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
                x=alt.X("Periods:N", sort=kpi_m_cols, title=None, axis=alt.Axis(labelAngle=kpi_angle)),
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
            st.altair_chart((fact_w + trend_w).properties(height=360).interactive(), use_container_width=True)


def rebuild_report_from_account_snapshot(res, acc_details: list[dict]):
    if not acc_details or res is None:
        return res

    all_cols = list(res.amounts.columns)
    month_cols = [c for c in all_cols if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper())]

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
                if ("YTD" in c.upper() or "KOPĀ" in c.upper()) and "AVG" not in c.upper() and month_cols:
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
    return apply_custom_row_order_to_result(res)


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


MUTED_DIFF_BG = {
    "S": "#FEF9C3",
    "S_SUBTOTAL": "#FFFDF0",
    "S_CHILD": "#FFFDF5",
    "S_DETAIL": "#FFFDF5",
    "S_NEW_PROD": "#FFFDF0",
    "TVC": "#F5CBA7",
    "TVC_SUBTOTAL": "#FDF2E9",
    "TVC_CHILD": "#FEF9F5",
    "TVC_DETAIL": "#FEF9F5",
    "TVC_NEW_PROD": "#FDF2E9",
    "T": "#AED6F1",
    "T_SUBTOTAL": "#EBF5FB",
    "T_CHILD": "#F4F9FD",
    "T_NEW_PROD": "#EBF5FB",
    "OE": "#F8D7BA",
    "OE_CHILD": "#FEF8F2",
    "NP": "#CBE8BA",
}


def render_unified_toc_tables_html(
    amounts_df: pd.DataFrame,
    main_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    row_kinds: dict[str, str],
    title_main: str = "Finanšu pārskata tabula",
    title_kpi: str = "Produktivitātes un atdeves rādītāji",
    grey_cols: set[str] | None = None,
    muted_cols: set[str] | None = None,
    fy_start_month: int = 1,
    show_cumulative: bool = True,
    lbl_width_px: int | None = None,
    freeze_sides: bool = False,
    highlight_section: str | None = None,
) -> str:
    """
    Optimizēts gan 13,3" klēpjdatoram, gan lielam monitoram:
    - Uz lielā monitora aizņem 100% platuma;
    - Uz 13,3" ekrāna katrai skaitļu kolonnai tiek garantēts lasāms minimālais platums (nekādu '120...'!),
      ar fiksētu kreiso rādītāju kolonnu un gludu horizontālo ritināšanu.
    - Ja `freeze_sides=True`, papildus sasaldē arī labās KOPĀ un AVG kolonnas.
    - Ja `highlight_section` ir 'main', 'kpi' vai 'cum', attiecīgā tabula tiek izcelta apmācības rāmī.
    """
    valid_rows = [
        r for r in amounts_df.index
        if row_kinds.get(r) not in ("EXCL", "EXCLUDED")
        and not str(r).lower().startswith("izslēgt")
        and not str(r).lower().startswith("neiekļaut")
    ]
    amounts_df = amounts_df.loc[valid_rows]
    main_df = main_df.loc[valid_rows]

    periods = list(amounts_df.columns)
    n_per = max(len(periods), 1)
    row_names = list(amounts_df.index)
    kinds = [row_kinds.get(rn, "OE_CHILD") for rn in row_names]
    grey_cols = grey_cols or set()
    muted_cols = muted_cols or set()

    display_row_name = lambda rn: "Uzturēšanas izmaksas (OE)" if rn == "OE" else rn

    if lbl_width_px is not None:
        lbl_w_px = lbl_width_px
    elif n_per <= 5:
        lbl_w_px = 360
    else:
        lbl_w_px = 230

    # Vairāk par 5 periodiem -> garantējam lasāmus platumus uz 13.3" klēpjdatora + fiksētu kreiso kolonnu!
    use_responsive_scroll = bool(n_per > 5)
    use_freeze_left = bool(freeze_sides or use_responsive_scroll)
    use_freeze_right = bool(freeze_sides and n_per >= 4)

    m_val_w, m_pct_w = 78, 38
    sum_val_w, sum_pct_w = 88, 38
    sum_pair_w = sum_val_w + sum_pct_w  # 126px

    def _build_two_subcol_colgroup(num_periods: int, total_ref_periods: int) -> tuple[str, str]:
        if use_freeze_left:
            cols_html = [f'<col style="width: {lbl_w_px}px;">']
            for p_i in range(num_periods):
                if p_i >= num_periods - 2:
                    cols_html.append(f'<col style="width: {sum_val_w}px;">')
                    cols_html.append(f'<col style="width: {sum_pct_w}px;">')
                else:
                    cols_html.append(f'<col style="width: {m_val_w}px;">')
                    cols_html.append(f'<col style="width: {m_pct_w}px;">')
            min_total_px = lbl_w_px + max(num_periods - 2, 0) * (m_val_w + m_pct_w) + min(num_periods, 2) * sum_pair_w
            w_rule = f"width: {min_total_px}px;" if use_freeze_right else f"width: 100%; min-width: {min_total_px}px;"
            return "<colgroup>" + "".join(cols_html) + "</colgroup>", f"{w_rule} table-layout: fixed;"

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
        if use_freeze_left:
            cols_html = [f'<col style="width: {lbl_w_px}px;">']
            for p_i in range(num_periods):
                w_c = sum_pair_w if p_i >= num_periods - 2 else (m_val_w + m_pct_w)
                cols_html.append(f'<col style="width: {w_c}px;">')
            min_total_px = lbl_w_px + max(num_periods - 2, 0) * (m_val_w + m_pct_w) + min(num_periods, 2) * sum_pair_w
            w_rule = f"width: {min_total_px}px;" if use_freeze_right else f"width: 100%; min-width: {min_total_px}px;"
            return "<colgroup>" + "".join(cols_html) + "</colgroup>", f"{w_rule} table-layout: fixed;"

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
    tbl_cls = "toc-unified toc-freeze" if use_freeze_left else "toc-unified"

    def _sticky_left_style(bg: str) -> str:
        if not use_freeze_left:
            return f"background-color: {bg};"
        return f"position: sticky; left: 0; z-index: 5; background-color: {bg}; box-shadow: 2px 0 4px rgba(0,0,0,0.12);"

    def _sticky_right_pair(p_idx: int, total_p: int, is_pct: bool, bg: str) -> str:
        if not use_freeze_right or p_idx < total_p - 2:
            return ""
        if p_idx == total_p - 1:
            r_px = 0 if is_pct else sum_pct_w
            return f"position: sticky; right: {r_px}px; z-index: 5; background-color: {bg};"
        if is_pct:
            return f"position: sticky; right: {sum_pair_w}px; z-index: 5; background-color: {bg};"
        return f"position: sticky; right: {sum_pair_w + sum_pct_w}px; z-index: 5; background-color: {bg}; box-shadow: -3px 0 6px rgba(0,0,0,0.15);"

    def _sticky_right_single(p_idx: int, total_p: int, bg: str) -> str:
        if not use_freeze_right or p_idx < total_p - 2:
            return ""
        if p_idx == total_p - 1:
            return f"position: sticky; right: 0px; z-index: 5; background-color: {bg};"
        return f"position: sticky; right: {sum_pair_w}px; z-index: 5; background-color: {bg}; box-shadow: -3px 0 6px rgba(0,0,0,0.15);"

    def _wrap_open(sec_id: str, badge_txt: str = "") -> str:
        if highlight_section == sec_id:
            return f'<div id="tut-focus-anchor" class="tut-spotlight"><div class="tut-spotlight-label">{badge_txt}</div>'
        return "<div>"

    html = ["""
    <style>
    .toc-table-wrap { width: 100%; overflow-x: auto; margin-bottom: 14px; -webkit-overflow-scrolling: touch; }
    .toc-sec-heading {
        font-family: 'Inter', 'Segoe UI', 'Calibri', sans-serif;
        font-size: 13.5px;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin: 12px 0 7px 0;
        padding: 5px 13px;
        background-color: #1E293B;
        color: #FFFFFF !important;
        border-radius: 6px;
        display: inline-block;
    }
    .tut-spotlight {
        border: 3px solid #4F46E5 !important;
        border-radius: 12px !important;
        padding: 10px 12px 4px 12px !important;
        background-color: rgba(79, 70, 229, 0.04) !important;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.14) !important;
        margin: 10px 0 18px 0 !important;
    }
    .tut-spotlight-label {
        display: inline-block;
        background-color: #4F46E5;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
        margin-bottom: 4px;
        letter-spacing: 0.3px;
    }
    table.toc-unified {
        border-collapse: collapse !important;
        font-family: 'Inter', 'Calibri', 'Segoe UI', sans-serif;
        font-size: 12px;
        border: 2.5px solid #000000 !important;
    }
    table.toc-unified.toc-freeze { border-collapse: separate !important; border-spacing: 0 !important; }
    table.toc-unified th, table.toc-unified td { box-sizing: border-box !important; }
    table.toc-unified th {
        background-color: #E2E8F0; color: #000000 !important; font-weight: 700; text-align: center;
        padding: 6px 5px; border-top: 2.5px solid #000000 !important; border-bottom: 2.5px solid #000000 !important;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    table.toc-unified td {
        padding: 4px 6px; white-space: nowrap; color: #000000 !important; overflow: hidden;
        text-overflow: ellipsis; border-top: 1px solid #222222 !important; border-bottom: 1px solid #222222 !important;
    }
    table.toc-unified tr.thick-bottom td { border-bottom: 2.5px solid #000000 !important; }
    table.toc-unified tr.wage-block-bottom td { border-bottom: 1.8px solid #1E293B !important; }
    table.toc-unified tr.new-prod-sep td { border-top: 1.5px solid #000000 !important; }
    table.toc-unified td.lbl-main {
        text-align: left; font-weight: 700; padding-left: 8px; padding-right: 8px;
        border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; color: #000000 !important;
    }
    table.toc-unified td.lbl-sub {
        text-align: left; font-weight: 400; padding-left: 14px; padding-right: 8px;
        border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; color: #000000 !important;
    }
    table.toc-unified td.lbl-sub-wage {
        text-align: left; font-weight: 500; padding-left: 14px; padding-right: 8px;
        border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; color: #000000 !important;
    }
    table.toc-unified td.lbl-sub-l1 {
        text-align: left; font-weight: 600; padding-left: 14px; padding-right: 8px;
        border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; color: #000000 !important;
    }
    table.toc-unified td.lbl-sub-l2 {
        text-align: left; font-weight: 400; padding-left: 24px; padding-right: 8px;
        border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; color: #1E293B !important;
    }
    span.sub-tree-bar { display: inline-block; border-left: 2px solid rgba(0, 0, 0, 0.38); padding-left: 7px; line-height: 1.15; }
    table.toc-unified td.val-col {
        text-align: right; border-left: 1px solid #222222 !important; border-right: 1px solid #222222 !important;
        font-variant-numeric: tabular-nums; color: #000000 !important;
    }
    table.toc-unified td.pct-col {
        text-align: right; font-size: 11px; padding-left: 2px; padding-right: 4px;
        border-left: 1px solid #222222 !important; border-right: 2.5px solid #000000 !important;
        color: #000000 !important; font-variant-numeric: tabular-nums;
    }
    table.toc-unified td.kpi-col {
        text-align: right; padding: 4px 8px; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important;
        font-weight: 400; font-variant-numeric: tabular-nums; color: #000000 !important;
    }
    table.toc-unified th.th-kpi {
        border-top: 2.5px solid #000000 !important; border-bottom: 2.5px solid #000000 !important;
        border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important;
    }
    table.toc-unified th.th-val { border-left: 1px solid #222222 !important; border-right: 1px solid #222222 !important; }
    table.toc-unified th.th-pct { font-size: 11px; border-left: 1px solid #222222 !important; border-right: 2.5px solid #000000 !important; }
    table.toc-unified th.th-lbl { text-align: left; padding-left: 8px; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; }
    </style>
    """]

    # 1. GALVENĀ TABULA
    html.append(_wrap_open("main", "👉 2. SOLIS — GALVENĀ TOC FINANŠU TABULA"))
    if title_main:
        html.append(f'<div class="toc-sec-heading">{title_main}</div>')
    html.append(f'<div class="toc-table-wrap"><table class="{tbl_cls}" style="{main_tbl_style}">{main_colgroup}<thead><tr>')
    html.append(f'<th class="th-lbl" style="{_sticky_left_style("#E2E8F0")}">Rādītājs / Kontugrupa</th>')
    for p_idx, p in enumerate(periods):
        if p in grey_cols:
            base_th_bg, extra_th = "#CBD5E1", "color: #000000;"
        elif p in muted_cols:
            base_th_bg, extra_th = "#F1F5F9", "color: #334155; font-weight: 600;"
        else:
            base_th_bg, extra_th = "#E2E8F0", ""
        st_v = _sticky_right_pair(p_idx, n_per, False, base_th_bg)
        st_p = _sticky_right_pair(p_idx, n_per, True, base_th_bg)
        html.append(
            f'<th class="th-val" style="background-color: {base_th_bg}; {extra_th} {st_v}" title="{p}">{p}</th>'
            f'<th class="th-pct" style="background-color: {base_th_bg}; {extra_th} {st_p}">%</th>'
        )
    html.append("</tr></thead><tbody>")

    s_grp_kinds = ("S", "S_CHILD", "S_SUBTOTAL", "S_DETAIL", "S_NEW_PROD")
    tvc_grp_kinds = ("TVC", "TVC_CHILD", "TVC_SUBTOTAL", "TVC_DETAIL", "TVC_NEW_PROD")
    t_grp_kinds = ("T", "T_CHILD", "T_SUBTOTAL", "T_NEW_PROD")
    oe_grp_kinds = ("OE", "OE_CHILD")

    l1_medium_kinds = ("S_SUBTOTAL", "TVC_SUBTOTAL", "T_SUBTOTAL", "S_NEW_PROD", "TVC_NEW_PROD", "T_NEW_PROD")
    l2_detail_kinds = ("S_DETAIL", "TVC_DETAIL")
    new_prod_kinds = ("S_NEW_PROD", "TVC_NEW_PROD", "T_NEW_PROD")

    for i, rname in enumerate(row_names):
        kind = kinds[i]
        next_kind = kinds[i + 1] if i + 1 < len(kinds) else None
        next_rname = row_names[i + 1] if i + 1 < len(row_names) else ""
        fill_hex, is_bold, _ = FILLS.get(kind, FILLS["OE_CHILD"])

        is_wage_row = kind == "OE_CHILD" and is_wage_row_label(rname)
        next_is_wage_row = next_kind == "OE_CHILD" and is_wage_row_label(next_rname)

        thick_bottom = (
            kind == "NP"
            or (kind in t_grp_kinds and next_kind not in t_grp_kinds)
            or (kind in s_grp_kinds and next_kind not in s_grp_kinds)
            or (kind in tvc_grp_kinds and next_kind not in tvc_grp_kinds)
            or (kind in oe_grp_kinds and next_kind not in oe_grp_kinds)
            or i == len(row_names) - 1
        )
        tr_classes = []
        if thick_bottom:
            tr_classes.append("thick-bottom")
        elif is_wage_row and not next_is_wage_row:
            tr_classes.append("wage-block-bottom")
        if kind in new_prod_kinds:
            tr_classes.append("new-prod-sep")
        tr_cls = f' class="{" ".join(tr_classes)}"' if tr_classes else ""

        disp_rn = display_row_name(rname)

        if kind in l1_medium_kinds:
            font_w, lbl_cls, lbl_inner = "600", "lbl-sub-l1", disp_rn
        elif kind in l2_detail_kinds:
            font_w, lbl_cls, lbl_inner = "400", "lbl-sub-l2", f'<span class="sub-tree-bar">{disp_rn}</span>'
        elif is_bold:
            font_w, lbl_cls, lbl_inner = "700", "lbl-main", disp_rn
        elif is_wage_row:
            font_w, lbl_cls, lbl_inner = "500", "lbl-sub-wage", disp_rn
        else:
            font_w, lbl_cls, lbl_inner = "400", "lbl-sub", disp_rn

        row_bg_css = f"#{'F1C7A8' if is_wage_row else fill_hex}"
        row_style = f"background-color: {row_bg_css}; color: #000000; font-weight: {font_w};"
        html.append(
            f'<tr{tr_cls} style="{row_style}">'
            f'<td class="{lbl_cls}" style="{_sticky_left_style(row_bg_css)}" title="{disp_rn}">{lbl_inner}</td>'
        )

        for p_idx, p in enumerate(periods):
            val_str = fmt_num(main_df.loc[rname, p])
            pct_str = fmt_pct_int(main_df.loc[rname, f"%|{p}"])
            if p in grey_cols:
                c_bg = "#CBD5E1" if is_bold else "#E2E8F0"
                base_cell = f"background-color: {c_bg}; color: #000000; font-weight: {font_w};"
            elif p in muted_cols:
                c_bg = MUTED_DIFF_BG.get(kind, "#F8FAFC")
                m_fw = "600" if is_bold else "400"
                base_cell = f"background-color: {c_bg}; color: #334155 !important; font-weight: {m_fw};"
            else:
                c_bg = row_bg_css
                base_cell = f"background-color: {c_bg}; color: #000000; font-weight: {font_w};"
            st_v = _sticky_right_pair(p_idx, n_per, False, c_bg)
            st_p = _sticky_right_pair(p_idx, n_per, True, c_bg)
            html.append(f'<td class="val-col" style="{base_cell} {st_v}">{val_str}</td><td class="pct-col" style="{base_cell} {st_p}">{pct_str}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div></div>")

    # 2. PRODUKTIVITĀTES (KPI) TABULA
    kpi_periods = [p for p in periods if p in kpi_df.columns]
    n_kpi_p = len(kpi_periods)
    kpi_colgroup, kpi_tbl_style = _build_single_col_kpi_colgroup(n_kpi_p, n_per)
    month_only_kpi_cols = [
        p for p in kpi_periods
        if not ("YTD" in str(p).upper() or "AVG" in str(p).upper() or "BĀZE" in str(p).upper() or "KOPĀ" in str(p).upper())
    ]

    html.append(_wrap_open("kpi", "👉 3. SOLIS — PRODUKTIVITĀTES UN ATDEVES RĀDĪTĀJI (RATIO)"))
    html.append(f'<div class="toc-sec-heading">{title_kpi}</div>')
    html.append(f'<div class="toc-table-wrap"><table class="{tbl_cls}" style="{kpi_tbl_style}">{kpi_colgroup}<thead><tr>')
    html.append(f'<th class="th-lbl" style="{_sticky_left_style("#E2E8F0")}">Produktivitātes rādītājs</th>')
    for p_idx, p in enumerate(kpi_periods):
        th_bg_c = "#CBD5E1" if p in grey_cols else "#E2E8F0"
        st_r = _sticky_right_single(p_idx, n_kpi_p, th_bg_c)
        html.append(f'<th class="th-kpi" style="background-color: {th_bg_c}; {st_r}">{p}</th>')
    html.append("</tr></thead><tbody>")

    for i, idx_name in enumerate(kpi_df.index):
        tr_cls = ' class="thick-bottom"' if i == len(kpi_df.index) - 1 else ""
        html.append(f'<tr{tr_cls} style="background-color: #FFFFFF; color: #000000;">')
        html.append(
            f'<td class="lbl-main" style="font-weight: 700; color: #000000; {_sticky_left_style("#F1F5F9")}" title="{idx_name}">{idx_name}</td>'
        )
        row_month_vals = [kpi_df.loc[idx_name, mp] for mp in month_only_kpi_cols] if month_only_kpi_cols else list(kpi_df.loc[idx_name].values)

        for p_idx, p in enumerate(kpi_periods):
            v = kpi_df.loc[idx_name, p]
            if pd.isna(v):
                s = ""
            elif idx_name.startswith("T%"):
                s = f"{v * 100:.0f}%"
            else:
                s = f"{v:.2f}"
            bg_cell = "#E2E8F0" if p in grey_cols else _kpi_cond_color(idx_name, v, row_month_vals)
            st_r = _sticky_right_single(p_idx, n_kpi_p, bg_cell)
            html.append(f'<td class="kpi-col" style="background-color: {bg_cell}; {st_r}">{s}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div></div>")

    # 3. UZKRĀJOŠO RĀDĪTĀJU TABULA
    if show_cumulative:
        cum_df, cum_periods, _ = compute_cumulative_toc_df(amounts_df, fy_start_month=fy_start_month)
        html.append(_wrap_open("cum", "👉 4. SOLIS — UZKRĀJOŠIE RĀDĪTĀJI (GADA UZKRĀJUMS UN LŪZUMA PUNKTS)"))
        html.append('<div class="toc-sec-heading">Uzkrājošie rādītāji</div>')
        html.append(f'<div class="toc-table-wrap"><table class="{tbl_cls}" style="{main_tbl_style}">{main_colgroup}<thead><tr>')
        html.append(f'<th class="th-lbl" style="{_sticky_left_style("#E2E8F0")}">Uzkrājošais rādītājs</th>')
        for p_idx, p in enumerate(periods):
            if p in cum_periods:
                html.append(f'<th class="th-val">{p}</th><th class="th-pct">%</th>')
            else:
                st_r = _sticky_right_single(p_idx, n_per, "#E2E8F0")
                html.append(
                    f'<th colspan="2" style="background-color: #E2E8F0; color: #000000; border-top: 2.5px solid #000000 !important; '
                    f'border-bottom: 2.5px solid #000000 !important; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; {st_r}">{p}</th>'
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
            lbl_bg_css = f"#{cum_label_fills.get(c_name, 'F1F5F9')}"
            is_np_cum = c_name.startswith("NP")

            html.append(f'<tr{tr_cls} style="background-color: #FFFFFF; color: #000000; font-weight: 400;">')
            html.append(
                f'<td class="lbl-main" style="font-weight: 700; color: #000000; {_sticky_left_style(lbl_bg_css)}" title="{c_name}">{c_name}</td>'
            )

            for p_idx, p in enumerate(periods):
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
                    if last_p and ("YTD" in str(p).upper() or "KOPĀ" in str(p).upper()) and "AVG" not in str(p).upper():
                        v_ytd = cum_df.loc[c_name, last_p]
                        ytd_bg = ("#D4EFDF" if v_ytd >= 0 else "#FADBD8") if is_np_cum else "#F8F9F9"
                        st_r = _sticky_right_single(p_idx, n_per, ytd_bg)
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; background-color: {ytd_bg}; color: #000000; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; font-weight: 400; {st_r}">{fmt_num(v_ytd)}</td>'
                        )
                    elif last_p and "AVG" in str(p).upper() and len(cum_periods) > 0:
                        v_avg = cum_df.loc[c_name, last_p] / len(cum_periods)
                        avg_bg = ("#D4EFDF" if v_avg >= 0 else "#FADBD8") if is_np_cum else "#F8F9F9"
                        st_r = _sticky_right_single(p_idx, n_per, avg_bg)
                        html.append(
                            f'<td colspan="2" style="text-align: right; padding: 4px 6px; background-color: {avg_bg}; color: #000000; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; font-weight: 400; {st_r}">{fmt_num(v_avg)}</td>'
                        )
                    else:
                        st_r = _sticky_right_single(p_idx, n_per, "#FFFFFF")
                        html.append(f'<td colspan="2" style="background-color: #FFFFFF; border-left: 2.5px solid #000000 !important; border-right: 2.5px solid #000000 !important; {st_r}"></td>')
            html.append("</tr>")

        html.append("</tbody></table></div></div>")

    return "".join(html)


# ------------------------------------------------------------------ THRUVIA: PAR SISTĒMU, INSTRUKCIJA, BUJ UN JURIDISKIE NOTEIKUMI
def render_about_and_help_page(is_public_view: bool = False) -> None:
    if is_public_view:
        top_b1, top_b2 = st.columns([1.2, 4.8], vertical_alignment="center")
        with top_b1:
            if st.button("← Atpakaļ uz pieslēgšanos", use_container_width=True, key="pub_about_back"):
                st.session_state.public_view = "login"
                st.rerun()

    st.subheader("THRUVIA — Par sistēmu, Instrukcija un BUJ")
    st.caption("MI instruments uzņēmuma finanšu analīzei un vadības lēmumu modelēšanai, balstoties uz TOC Caurlaidības grāmatvedību.")

    if is_public_view:
        cta_l, cta_r = st.columns([2.6, 1.4], vertical_alignment="center")
        with cta_l:
            st.info(
                "Izmēģiniet **THRUVIA** ar interaktīvu Demo piemēru uzreiz pēc reģistrācijas, iesniedziet sava uzņēmuma PZA "
                "pārbaudei un saņemiet **5 dienas bez maksas** darbam ar saviem datiem!"
            )
        with cta_r:
            if st.button("Reģistrēties (5 dienas bez maksas) →", key="pub_about_cta_top", type="primary", use_container_width=True):
                st.session_state.public_view = "register"
                st.rerun()

    tab_about, tab_guide, tab_faq, tab_legal = st.tabs(
        [
            "Par THRUVIA",
            "Instrukcija pa soļiem",
            "Biežāk uzdotie jautājumi (BUJ)",
            "Juridiskie noteikumi & Privātums",
        ]
    )

    with tab_about:
        with st.container(border=True):
            st.markdown("#### Finanšu vadība, balstoties uz Caurlaidību (*Throughput Accounting*)")
            st.markdown(
                """
                Klasiskā izmaksu grāmatvedība cenšas sadalīt visas uzņēmuma izmaksas pa atsevišķiem produktiem, aprēķinot to pašizmaksu.
                **TOC Caurlaidības grāmatvedība (TA)** raugās uz uzņēmumu kā vienotu sistēmu un atbild uz galveno vadības jautājumu:
                **kā mūsu darbība un pieņemtie lēmumi mainīs uzņēmuma kopējo finansiālo rezultātu?**
                """
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info("**S (Sales)**  \nRealizācijas ieņēmumi — nauda, kas ienāk sistēmā no pārdošanas.")
            with c2:
                st.warning("**TVC (Totally Variable Costs)**  \nPilnībā mainīgās izmaksas, kas mainās tieši 1:1 ar pārdošanu (izejvielas, materiāli, tiešā piegāde).")
            with c3:
                st.error("**OE (Operating Expenses)**  \nDarbības nodrošināšanas izmaksas (algas, telpas, mārketings, IT, administrācija).")

            st.markdown(
                """
                No šiem trim lielumiem iegūst divus izšķirošos rezultātus:
                * **T = S − TVC** — **Caurlaidība (*Throughput*)**, ātrums, kādā uzņēmums ģenerē jaunu naudu.
                * **NP = T − OE** — **TOC Neto peļņa (*Net Profit*)**, uzņēmuma kopējais darbības rezultāts bez netiešo izmaksu alokēšanas uz produktiem.
                """
            )

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            with st.container(border=True):
                st.markdown("#### Esošie dati — bez papildu grāmatvedības sistēmas")
                st.markdown(
                    """
                    **THRUVIA** nepieprasa mainīt grāmatvedības procesus — oficiālā grāmatvedība turpina kontēšanu, nodokļu un normatīvo atskaišu sagatavošanu.

                    * Sistēma ielasa uzņēmumā jau pieejamos **PZA failus** (Excel/CSV), MI palīdz saklasificēt kontus un sagatavo pārskatu ar dinamikas grafikiem.
                    * Pirmreizējiem klientiem mēs pārbaudām un sakārtojam PZA struktūru, lai visi rādītāji būtu maksimāli precīzi.
                    """
                )
        with col_a2:
            with st.container(border=True):
                st.markdown("#### No pagātnes analīzes uz nākotnes modelēšanu")
                st.markdown(
                    """
                    Vadības komanda — arī bez finanšu analītiķa pieredzes — vienas stundas laikā var izspēlēt desmitiem attīstības scenāriju, nepārrēķinot produktu pašizmaksas.

                    * Plānotās izmaiņas **S**, **TVC** vai **OE** posteņos ievada procentos, eiro, brīvā tekstā vai ar balsi, uzreiz redzot ietekmi uz **T**, **NP** un attiecību rādītājiem (**RATIO**).
                    * **Attīstības perspektīva:** nākamajos posmos THRUVIA sasaistīs finanšu prognozes ar uzņēmuma jaudām un sistēmas ierobežojumiem (**CCR**).
                    """
                )

    with tab_guide:
        st.markdown("#### Darbs ar THRUVIA 5 soļos")
        with st.container(border=True):
            st.markdown(
                """
                **1. Reģistrācija un interaktīvā Demo apmācība** — 2 minūšu apmācība uz apaļu skaitļu parauga uzņēmuma.  
                **2. Pirmā PZA faila pārbaude** — iesūtiet Excel/CSV failu pārbaudei (kamēr pārbaudām, varat izmantot Demo vai sākt uzreiz ar brīdinājumu).  
                **3. Pārskats un dinamika** — galvenā TOC tabula, produktivitātes koeficienti (`PR = T/OE`), uzkrājošie rādītāji un grafiki.  
                **4. Kontu struktūra** — pārskatiet katru PZA kontu, pārvietojiet, apvienojiet vai pārdēvējiet apakšgrupas.  
                **5. Scenāriju simulācija un MI Asistents** — modelējiet nākotnes lēmumus, jaunus produktus un jautājiet Asistentam rakstiski vai ar balsi.
                """
            )

    with tab_faq:
        st.markdown("#### Biežāk uzdotie jautājumi")
        with st.expander("1. Kāpēc darba algas un komunālie maksājumi ir pie OE, nevis pie TVC?"):
            st.markdown(
                "Pie **TVC** liek tikai tās izmaksas, kas rodas **1:1 par katru pārdoto vienību** (izejvielas, materiāli, tiešā piegāde). "
                "Algas, telpas, elektrība un administrācija ir kapacitātes uzturēšanas izmaksas (**OE**)."
            )
        with st.expander("2. Kādus kontus likt sadaļā 'Neiekļautie konti (ārpus TOC aprēķina)'?"):
            st.markdown(
                "Jebkurus kontus, kurus nevēlaties iekļaut operatīvajā TOC darbības rezultātā — piemēram, nolietojumu/amortizāciju, grāmatvedības uzkrājumus vai vienreizējus tehniskus posteņus."
            )
        with st.expander("3. Kā lasīt produktivitātes koeficientu PR = T / OE?"):
            st.markdown(
                "**PR < 1.00** — zaudējumi; **1.00–1.15** — neliela rezerve; **PR ≥ 1.15** — veselīga, stabila caurlaidības atdeve."
            )
        with st.expander("4. Kāpēc pirmais PZA fails tiek nosūtīts pārbaudei?"):
            st.markdown(
                "Pārbaudot pirmo failu, pārliecināmies par 100% precīzu kontu sadalījumu — un jūsu 5 bezmaksas izmēģinājuma dienas sākas tikai pēc apstiprināšanas!"
            )

    with tab_legal:
        render_legal_documents_tabs()

    if is_public_view:
        st.markdown("---")
        _, c_mid, _ = st.columns([1.2, 1.6, 1.2])
        with c_mid:
            if st.button("Izveidot bezmaksas kontu un izmēģināt Demo →", key="pub_about_cta_bottom", type="primary", use_container_width=True):
                st.session_state.public_view = "register"
                st.rerun()


knowledge_file = None
for fname in ["toc_knowledge.txt", "toc_knowledge.md", "knowledge.txt"]:
    if Path(fname).exists():
        knowledge_file = Path(fname).read_text(encoding="utf-8")
        break

S = st.session_state

for c_key, (c_lbl, c_mc) in S.get("custom_subgroups", {}).items():
    register_custom_subgroup(c_key, c_lbl, c_mc)


# ------------------------------------------------------------------ PELDOŠA RĪKJOSLA, TUTORIĀLA STILI UN ASISTENTA LOGS
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
        padding-top: 2.5rem !important;
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
    div.sticky-nav-marker {{ display: none !important; }}
    div[data-testid="stElementContainer"]:has(div.sticky-nav-marker),
    div.element-container:has(div.sticky-nav-marker) {{
        display: none !important; margin: 0 !important; padding: 0 !important; height: 0 !important;
    }}
    div[data-testid="stVerticalBlock"] > div:has(div.sticky-nav-marker) {{
        position: sticky !important;
        top: 2.4rem !important;
        z-index: 999 !important;
        background-color: var(--secondary-background-color, #F4F3EE) !important;
        border: 1px solid rgba(120, 120, 120, 0.25) !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06) !important;
        margin-bottom: 14px !important;
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
    .top-brand-bar {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
    .top-brand-title {{ font-size: 20px; font-weight: 800; letter-spacing: -0.4px; margin: 0; }}
    .top-brand-sub {{ font-size: 13px; opacity: 0.85; }}
    .unsaved-pill {{
        background-color: #FEF3C7; color: #92400E; border: 1px solid #F59E0B;
        padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
    }}
    .trial-pill {{
        background-color: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD;
        padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
    }}
    .demo-pill {{
        background-color: #EDE9FE; color: #5B21B6; border: 1px solid #A78BFA;
        padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700;
    }}
    .tut-compact-bar {{
        background: #F5F3FF;
        border: 2px solid #4F46E5;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }}
    .tut-step-pill {{
        display: inline-block; background-color: #4F46E5; color: #FFFFFF;
        font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; margin-right: 8px;
    }}
    .tut-section-tag {{
        display: inline-block; background-color: #4F46E5; color: #FFFFFF;
        font-size: 11px; font-weight: 700; padding: 3px 11px; border-radius: 999px;
        margin: 6px 0 6px 0; box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
    }}
    .disclaimer-urgent-box {{
        background-color: #FEF2F2; border: 2px solid #DC2626; border-radius: 10px;
        padding: 12px 16px; color: #7F1D1D; margin-bottom: 14px;
    }}
    .welcome-approved-box {{
        background: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
        border: 2px solid #10B981; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def build_global_diagnostic_context(profile_store: SupabaseProfileStore) -> str:
    parts = [f"Aktīvais uzņēmuma profils: {S.get('active_profile', 'Nav')}"]
    if S.get("is_demo_mode"):
        parts.append("REŽĪMS: Demo piemērs (SIA Demo Ražošana — mācību dati ar apaļiem skaitļiem)")
    parts.append(f"Pašreizējā atvērtā sadaļa aplikācijā: {S.get('nav_page', 'overview')}")
    parts.append(f"Aktīvā atskaite: {S.get('loaded_report_meta', {}).get('report_name', 'Jauna atskaite')}")
    parts.append(
        f"Finanšu gada sākums: {S.get('fy_start_day', 1)}. {MONTH_NAMES_LV.get(S.get('fy_start_month', 1), 'Janvāris')}"
    )

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
        sub_lines = [f"Apakšgrupa {sg} ({len(al)} konti):\n" + "\n".join(al) for sg, al in by_sub.items()]
        parts.append("\n--- KAS SLĒPJAS ZEM KATRAS APAKŠGRUPAS (PZA PAMATKONTI UN TO SUMMAS) ---\n" + "\n\n".join(sub_lines))
    elif not S.get("is_demo_mode"):
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
        parts.append("\n--- AKTĪVĀ THRUVIA (TA) TABULA (EUR) ---\n" + res.amounts.round(0).to_markdown())
        parts.append("\n--- PRODUKTIVITĀTES UN ATDEVES RĀDĪTĀJI (RATIO / KPI) ---\n" + res.kpi.round(3).to_markdown())

    return "\n\n".join(parts)


TOC_CORE_EXPERT_KNOWLEDGE = """
=== THRUVIA / TOC CAURLAIDĪBAS GRĀMATVEDĪBAS (THROUGHPUT ACCOUNTING) ZINĀŠANU BĀZE ===
1. Sistēmas loģika: THRUVIA balstās uz TOC Throughput Accounting (TA), vērtējot uzņēmumu kā vienotu sistēmu un atbildot: kā mūsu lēmums ietekmē kopējo NP?
2. Galvenie rādītāji:
   - S (Sales) — realizācijas ieņēmumi.
   - TVC (Totally Variable Costs) — pilnībā mainīgās izmaksas (1:1 ar pārdošanu: izejvielas, materiāli, tiešā piegāde). Algas, elektrība, noma un administrācija NEKAD nav TVC — tās ir OE.
   - OE (Operating Expenses) — darbības nodrošināšanas izmaksas. Zem algām ir TIKAI tiešās algas un algu nodokļi; reprezentācija un personāla pasākumi ir Pārējās administrācijas izmaksās.
   - T = S - TVC — Caurlaidība (Throughput).
   - NP = T - OE — TOC Neto peļņa.
3. Attiecību rādītāji (RATIO): PR = T/OE (< 1.00 zaudējumi, 1.00–1.15 maza rezerve, >= 1.15 veselīga atdeve); T% = T/S; Eiro Sales / Eiro algas.
4. Lēmumu prioritāte: T ↑ -> I ↓ -> OE ↓; operatīvā formula ΔT - ΔOE > 0.
5. SVARĪGS AIZLIEGUMS: Nekad nespried par "mārketinga kampaņu efektivitāti" vai reklāmas atdevi (no PZA to nomērīt nevar). Sniedz informatīvu TOC analīzi, nevis licencētus finanšu/nodokļu padomus.

=== KUR KAS ATRODAS THRUVIA APLIKĀCIJĀ (GIDS LIETOTĀJIEM) ===
- **Pārskats (`overview`)**: Galvenā TOC tabula (S, TVC, T, OE, NP), Produktivitātes rādītāji (`PR`, `T%`, Algu atdeve), Uzkrājošie rādītāji, 3 dinamikas grafiki, Excel lejupielāde, Aktīvā darba perioda izvēle (saglabājas arī Simulācijā), Versiju saglabāšana (oriģināls nepazūd) un poga *Pieprasīt eksperta ieteikumus*. Ja ir 15+ mēneši, parādās gadu pāršķiršana, gadu salīdzinājums un horizontālā ritināšana ar sasaldētām malām.
- **Scenāriju simulācija (`simulation`)**: Izmaiņu ievadīšana tabulā (% vai +/- €), situācijas aprakstīšana brīvā tekstā vai ar mikrofonu, sadaļa *Jauns produkts / projekts* un simulāciju saglabāšana arhīvā.
- **Kontu struktūra (`accounts_explorer`)**: Redzams, kuri PZA konti slēpjas zem katras apakšgrupas ar mēnešu summām; augšējā panelī 4 cilnes — pārvietot kontus, apvienot apakšgrupas, izveidot jaunu apakšgrupu, pārdēvēt apakšgrupu.
- **Atskaites un Simulācijas (`archive`)**: Visas saglabātās finanšu atskaites, versijas un scenāriju simulācijas (atvēršana un dzēšana).
- **+ Jauns PZA (`ingest`)**: Jauna Excel/CSV PZA faila ielāde, rindu kontrole un MI kontu klasifikācija.
- **Par THRUVIA (`about`)**: Teorētiskais pamatojums, instrukcija pa soļiem, BUJ un juridiskie noteikumi / privātuma politika.
- **Profils → Uzstādījumi un Abonements (`settings`)**: Abonementa statuss, Stripe maksājumu/atcelšanas saites, finanšu gada sākuma datums un paroles maiņa.

=== DARBĪBAS (ACTIONS), KO TU KĀ THRUVIA ASISTENTS VARI AUTOMĀTISKI IZPILDĪT ===
Pievieno JSON bloku tagos `<ACTION>...</ACTION>` pašās atbildes beigās, kad lietotājs vēlas veikt darbību vai nokļūt kādā sadaļā:
1) PĀRDĒVĒT APAKŠGRUPU: <ACTION>{"type": "rename_subgroup", "summary": "...", "old_name": "...", "new_name": "..."}</ACTION>
2) APVIENOT APAKŠGRUPAS: <ACTION>{"type": "merge_subgroups", "summary": "...", "source_subgroup": "...", "target_subgroup": "..."}</ACTION>
3) PĀRVIETOT KONTUS / IZVEIDOT APAKŠGRUPU: <ACTION>{"type": "split_or_move_accounts", "summary": "...", "main_code": "OE", "new_subgroup": "...", "accounts": ["..."]}</ACTION>
4) SAKĀRTOT TABULAS RINDAS (algas vienmēr paliek OE augšpusē): <ACTION>{"type": "reorder_subgroups", "summary": "...", "main_code": "OE", "mode": "desc", "keep_other_last": true}</ACTION>
5) NAVIGĒT ('overview', 'simulation', 'accounts_explorer', 'archive', 'ingest', 'about', 'settings'): <ACTION>{"type": "navigate", "summary": "...", "target_page": "accounts_explorer"}</ACTION>
"""


def execute_assistant_action(act: dict, profile_store: SupabaseProfileStore) -> tuple[bool, str]:
    act_type = act.get("type", "")

    if act_type == "navigate":
        target = act.get("target_page", "accounts_explorer")
        S.nav_page = target
        st.query_params["p"] = target
        return True, f"Atvērta sadaļa **{target}**."

    if act_type == "rename_subgroup":
        return rename_subgroup_everywhere(str(act.get("old_name") or ""), str(act.get("new_name") or ""), profile_store)

    if act_type == "merge_subgroups":
        return merge_subgroups_everywhere(str(act.get("source_subgroup") or ""), str(act.get("target_subgroup") or ""), profile_store)

    if act_type == "reorder_subgroups":
        res = S.get("result")
        if res is None:
            return False, "Nav aktīvas atskaites, ko sakārtot."
        mc = str(act.get("main_code") or "OE").upper()
        mode = str(act.get("mode") or "desc").lower()
        explicit_ord = act.get("order") if isinstance(act.get("order"), list) else None
        keep_oth = bool(act.get("keep_other_last", True))
        new_ord = reorder_table_subgroups(res, main_code=mc, mode=mode, explicit_order=explicit_ord, keep_other_last=keep_oth)
        return True, f"Apakšgrupas grupā **{mc}** ir sakārtotas jaunajā secībā: {', '.join(new_ord)}!"

    if act_type in ("split_or_move_accounts", "create_subgroup"):
        main_code = str(act.get("main_code") or "OE").upper()
        new_sub_lbl = str(act.get("new_subgroup") or act.get("target_subgroup") or "").strip()
        acc_names: list[str] = [str(x).strip() for x in (act.get("accounts") or []) if str(x).strip()]
        return move_specific_accounts_to_subgroup(acc_names, new_sub_lbl, main_code, profile_store)

    return False, "Nezināma darbība."


def ask_global_assistant(user_question: str, profile_store: SupabaseProfileStore) -> tuple[str, dict | None, int, float]:
    ctx = build_global_diagnostic_context(profile_store)
    know = f"\nPapildu lietotāja faila zināšanu bāze:\n{knowledge_file}\n" if knowledge_file else ""
    sys_prompt = (
        "Tu esi ļoti atsaucīgs, cilvēcīgs un augsti profesionāls THRUVIA aplikācijas gids un TOC Caurlaidības grāmatvedības eksperts.\n\n"
        f"{TOC_CORE_EXPERT_KNOWLEDGE}\n"
        f"{know}\n"
        "SVARĪGI NOTEIKUMI:\n"
        "1. Raksti skaidrā, īsā ČATA FORMĀTĀ (1–2 teikumi rindkopā + aizzīmes).\n"
        "2. Ja lietotājs jautā, kur aplikācijā kaut kas atrodas vai kā kaut ko izdarīt — dod precīzu, soli pa solim norādi un piedāvā `<ACTION>` pogu uzreiz atvērt to sadaļu (`navigate`)!\n"
        "3. Tu MĀKI un VARI izpildīt: pārdēvēt apakšgrupu (`rename_subgroup`), apvienot apakšgrupas (`merge_subgroups`), pārvietot kontus vai izveidot apakšgrupu (`split_or_move_accounts`), sakārtot tabulu (`reorder_subgroups`), atvērt sadaļu (`navigate`).\n"
        "4. Nekad nespried par mārketinga kampaņu efektivitāti — skaidro TOC rādītājus (T, T%, OE, PR = T/OE, Eiro Sales / Eiro algas)."
    )
    history = S.get("global_chat", [])
    clean_history = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
    messages = [
        {"role": "user", "content": f"Sistēmas un uzņēmuma finanšu konteksts:\n{ctx}"},
        {"role": "assistant", "content": "Sapratu! Esmu gatavs palīdzēt gan ar orientēšanos THRUVIA aplikācijā, gan ar TOC finanšu analīzi un kontu pārvaldību."},
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


# ------------------------------------------------------------------ 1. TĪRA PUBLISKĀ IEEJA (BEZ DUBULTĀS RĪKJOSLAS)
billing_cfg = get_billing_config()

if not S.get("authenticated", False):
    if "public_view" not in S:
        S.public_view = "login"

    if S.public_view == "about":
        render_about_and_help_page(is_public_view=True)
        render_app_footer_disclaimer()
        st.stop()

    _, col_l2, _ = st.columns([0.95, 1.7, 0.95])
    with col_l2:
        temp_store = SupabaseProfileStore()
        st.write("")
        if S.public_view == "login":
            with st.container(border=True):
                st.markdown("## THRUVIA")
                st.caption("MI instruments uzņēmuma finanšu analīzei un vadības lēmumu modelēšanai (TOC)")
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

                st.markdown("---")
                lc1, lc2 = st.columns(2)
                with lc1:
                    if st.button("Par THRUVIA", use_container_width=True, key="login_to_about"):
                        S.public_view = "about"
                        st.rerun()
                with lc2:
                    if st.button("Reģistrēties (5 dienas bez maksas)", type="primary", use_container_width=True, key="login_to_reg"):
                        S.public_view = "register"
                        st.rerun()

        else:
            if st.button("← Atpakaļ uz pieslēgšanos", key="reg_back_login"):
                S.public_view = "login"
                st.rerun()
            with st.container(border=True):
                st.markdown(f"### Reģistrēties THRUVIA ({billing_cfg['trial_days']} dienas bez maksas)")
                st.caption(
                    "1. Izveidojiet profilu → 2. Iziet īsu Demo apmācību → "
                    "3. Iesniedziet sava uzņēmuma PZA pārbaudei → 4. Pēc apstiprināšanas sākas 5 bezmaksas dienas "
                    f"(pēc tam **{billing_cfg['monthly_price_eur']} €/mēn.**, atceļams jebkurā brīdī)."
                )
                with st.form("register_trial_form"):
                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        reg_company = st.text_input("Uzņēmuma nosaukums *", placeholder="piem., SIA Mans Uzņēmums")
                        reg_contact = st.text_input("Kontaktpersona (vārds, uzvārds) *", placeholder="piem., Jānis Bērziņš")
                    with r_col2:
                        reg_email = st.text_input("Darba e-pasts *", placeholder="janis@uznemums.lv")
                        reg_pw1 = st.text_input("Izvēlieties paroli (vismaz 6 simboli) *", type="password")
                    reg_pw2 = st.text_input("Atkārtojiet paroli *", type="password")

                    reg_agree_terms = st.checkbox(
                        "Piekrītu THRUVIA Lietošanas noteikumiem un Privātuma politikai (VDAR), kā arī apliecinu, "
                        "ka saprotu — sistēma ir informatīvs analīzes rīks un nesniedz licencētas finanšu, nodokļu vai investīciju konsultācijas. *",
                        value=False,
                    )
                    reg_submit = st.form_submit_button(
                        "Izveidot profilu un sākt Demo apmācību →",
                        type="primary",
                        use_container_width=True,
                    )

                    if reg_submit:
                        rc = reg_company.strip()
                        rcont = reg_contact.strip()
                        re_mail = reg_email.strip()
                        if not rc or len(rc) < 2:
                            st.error("Lūdzu ievadiet uzņēmuma nosaukumu.")
                        elif not rcont or len(rcont) < 2:
                            st.error("Lūdzu ievadiet kontaktpersonas vārdu un uzvārdu.")
                        elif "@" not in re_mail or "." not in re_mail:
                            st.error("Lūdzu ievadiet derīgu darba e-pasta adresi.")
                        elif len(reg_pw1) < 6:
                            st.error("Parolei jābūt vismaz 6 simbolus garai.")
                        elif reg_pw1 != reg_pw2:
                            st.error("Ievadītās paroles nesakrīt.")
                        elif not reg_agree_terms:
                            st.error("Lai reģistrētos, lūdzu apstipriniet piekrišanu juridiskajiem noteikumiem.")
                        else:
                            existing_names = {c["company_name"].strip().lower() for c in temp_store.list_all_companies_admin()}
                            if rc.lower() in existing_names or rc.lower() == "admin":
                                st.error("Šāds uzņēmuma profils jau pastāv. Lūdzu pieslēdzieties vai izvēlieties citu nosaukumu.")
                            else:
                                structured_n = build_structured_notes(
                                    plan_code="TRIAL",
                                    email=re_mail,
                                    trial_start=date.today(),
                                    free_note="Jauns klients (pašreģistrācija)",
                                    onboard_status="TUTORIAL",
                                    contact_name=rcont,
                                    legal_ok_date=date.today().isoformat(),
                                    welcome_seen="0",
                                )
                                ok_reg, msg_reg = temp_store.create_or_update_company_admin(
                                    company_name=rc,
                                    password=reg_pw1,
                                    monthly_limit=10000,
                                    is_active=True,
                                    notes=structured_n,
                                    reset_usage=True,
                                )
                                if ok_reg:
                                    S.authenticated = True
                                    S.is_admin = False
                                    S.active_profile = rc
                                    S.view_mode = "workspace"
                                    S.nav_page = "overview"
                                    S.tutorial_step = 1
                                    S.auto_loaded_latest = False
                                    st.query_params["u"] = rc
                                    st.query_params["t"] = _make_auth_token(rc, False)
                                    st.rerun()
                                else:
                                    st.error(msg_reg)

                with st.expander("Lasīt Lietošanas noteikumus, Privātuma politiku (VDAR) un Atbildības ierobežojumu"):
                    render_legal_documents_tabs(compact=True)

    render_app_footer_disclaimer()
    st.stop()


# ------------------------------------------------------------------ 2. ADMINISTRATORA PANELIS (AR 1 KLIKŠĶA PZA PĀRBAUDI UN E-PASTU)
admin_store = SupabaseProfileStore()

if S.get("is_admin", False) and S.get("view_mode") == "admin_panel":
    top_a1, top_a2 = st.columns([5, 1], vertical_alignment="center")
    with top_a1:
        st.title("THRUVIA — Administratora un abonēšanas panelis")
        st.caption("Pirmreizējo PZA failu tiešā pārbaude, apstiprināšana ar e-pastu un klientu abonementu vadība.")
    with top_a2:
        if st.button("Iziet", use_container_width=True):
            st.query_params.clear()
            S.clear()
            st.rerun()

    # Ja tikko tika apstiprināts kāda klienta PZA — parādām e-pasta nosūtīšanas statusu (un mailto: rezerves pogu)
    last_mail_info = S.get("last_approval_email_result")
    if last_mail_info:
        with st.container(border=True):
            if last_mail_info["sent"]:
                st.success(last_mail_info["msg"])
            else:
                st.warning(last_mail_info["msg"])
                if last_mail_info.get("to_email"):
                    mailto_q = urllib.parse.urlencode(
                        {"subject": last_mail_info["subject"], "body": last_mail_info["body"]},
                        quote_via=urllib.parse.quote,
                    )
                    st.link_button(
                        f"✉️ Atvērt gatavu e-pastu klientam ({last_mail_info['to_email']})",
                        f"mailto:{last_mail_info['to_email']}?{mailto_q}",
                        type="primary",
                    )
            if st.button("Aizvērt paziņojumu", key="close_mail_notice"):
                S.pop("last_approval_email_result", None)
                st.rerun()

    companies = admin_store.list_all_companies_admin()
    enriched = []
    for c in companies:
        sub_st = parse_subscription_status(c.get("notes"), trial_days=billing_cfg["trial_days"])
        onb_st = parse_onboarding_status(c.get("notes"))
        c_copy = dict(c)
        c_copy["plan_code"] = sub_st["code"]
        c_copy["plan_label"] = sub_st["label"]
        c_copy["onboard_status"] = onb_st
        c_copy["onboard_label"] = ONBOARD_LABELS_LV.get(onb_st, onb_st)
        c_copy["email"] = extract_note_field(c.get("notes"), "EMAIL")
        c_copy["contact"] = extract_note_field(c.get("notes"), "CONTACT")
        c_copy["free_note"] = extract_free_notes(c.get("notes"))
        enriched.append(c_copy)

    pending_review_clients = [c for c in enriched if c["onboard_status"] in ("PENDING_WAIT", "PENDING_IMMEDIATE")]
    n_total = len(enriched)
    n_paid = sum(1 for c in enriched if c["plan_code"] == "ACTIVE" and c.get("is_active"))
    n_trial = sum(1 for c in enriched if c["plan_code"] == "TRIAL" and c.get("is_active"))
    tot_api_eur = sum(float(c.get("total_cost_eur") or 0.0) for c in enriched)
    try:
        mrr_eur = n_paid * float(billing_cfg["monthly_price_eur"])
    except ValueError:
        mrr_eur = 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Klienti kopā", n_total)
    k2.metric("Gaida PZA pārbaudi", len(pending_review_clients))
    k3.metric("Aktīvi maksas abon.", n_paid)
    k4.metric("Izmēģinājumā (5d)", n_trial)
    k5.metric("Prognozētais MRR / API", f"{mrr_eur:,.0f} €", delta=f"API: -{tot_api_eur:.2f} €", delta_color="off")

    # PZA PĀRBAUDES RINDA — 1 KLIKŠĶA TIEŠĀ PĀRBAUDE BEZ LEJUPIELĀDES!
    if pending_review_clients:
        st.markdown("---")
        st.subheader(f"📥 PZA pārbaudes rinda ({len(pending_review_clients)})")
        st.caption(
            "Spiediet **🔍 Pārbaudīt & Klasificēt PZA** — klienta fails uzreiz ielādēsies klasifikācijā. "
            "Kamēr kārtojat kontus, klients joprojām redz tikai Demo. Pēc pabeigšanas spiediet apstiprināt, un klientam aizies paziņojums!"
        )
        for pc in pending_review_clients:
            pc_name = pc["company_name"]
            pc_store = SupabaseProfileStore(company_name=pc_name)
            pc_reps = pc_store.list_reports()
            pending_files = [r for r in pc_reps if str(r.get("file_name", "")).startswith("PENDING_PZA::")]

            latest_payload = None
            if pending_files:
                loaded_p = pc_store.load_report(pending_files[0]["id"])
                if loaded_p:
                    try:
                        latest_payload = json.loads(loaded_p.get("commentary") or "{}")
                    except Exception:
                        latest_payload = None

            with st.container(border=True):
                r1, r2, r3 = st.columns([2.5, 1.5, 1.0], vertical_alignment="center")
                with r1:
                    st.markdown(f"**{pc_name}** &nbsp;|&nbsp; `{pc['onboard_label']}`")
                    contact_line = f"{pc['contact']} ({pc['email']})" if pc["contact"] else (pc["email"] or "Nav e-pasta")
                    st.caption(f"Kontakts: {contact_line}")
                    if latest_payload:
                        st.caption(
                            f"Fails: **{latest_payload.get('filename')}** ({latest_payload.get('size_kb', '?')} KB) | "
                            f"Iesūtīts: {latest_payload.get('submitted_at', '—')}"
                            + (f" | Piezīme: *{latest_payload.get('user_note')}*" if latest_payload.get("user_note") else "")
                        )
                with r2:
                    if st.button("🔍 Pārbaudīt & Klasificēt PZA", key=f"adm_direct_rev_{pc_name}", type="primary", use_container_width=True):
                        S.active_profile = pc_name
                        S.view_mode = "workspace"
                        S.admin_reviewing_client = pc_name
                        S.nav_page = "ingest"
                        S.auto_loaded_latest = False
                        S.pop("result", None)
                        S.pop("cls_df", None)
                        S.pop("structure", None)
                        S.pop("analysis", None)
                        S.pop("fid", None)
                        S.pop("clarifications_done", None)
                        S.pop("loaded_report_meta", None)
                        if latest_payload and latest_payload.get("b64"):
                            S.preloaded_pza_name = latest_payload.get("filename", "klienta_PZA.xlsx")
                            S.preloaded_pza_bytes = base64.b64decode(latest_payload["b64"])
                        st.rerun()
                with r3:
                    if latest_payload and latest_payload.get("b64"):
                        st.download_button(
                            "Lejupielādēt",
                            data=base64.b64decode(latest_payload["b64"]),
                            file_name=latest_payload.get("filename", "klienta_PZA.xlsx"),
                            key=f"adm_dl_bak_{pc_name}",
                            use_container_width=True,
                        )

    st.markdown("---")
    col_adm_left, col_adm_right = st.columns([1.85, 1.15])

    with col_adm_left:
        st.subheader("Klientu un abonementu pārskats")
        flt_c1, flt_c2, flt_c3 = st.columns([1.2, 1.4, 0.8], vertical_alignment="bottom")
        with flt_c1:
            adm_filter = st.selectbox(
                "Filtrs pēc statusa",
                options=["Visi klienti", "Gaida PZA pārbaudi", "Aktīvi maksas", "Izmēģinājumā (5d)", "Beidzies / Neaktīvi", "B4B pārvaldītie"],
                key="adm_plan_filter",
            )
        with flt_c2:
            adm_search = st.text_input("Meklēt (uzņēmums, e-pasts, piezīme)", placeholder="Meklēt...", key="adm_search_q")

        filtered_comps = []
        for c in enriched:
            if adm_filter == "Gaida PZA pārbaudi" and c["onboard_status"] not in ("PENDING_WAIT", "PENDING_IMMEDIATE"):
                continue
            if adm_filter == "Aktīvi maksas" and not (c["plan_code"] == "ACTIVE" and c.get("is_active")):
                continue
            if adm_filter == "Izmēģinājumā (5d)" and not (c["plan_code"] == "TRIAL" and c.get("is_active")):
                continue
            if adm_filter == "Beidzies / Neaktīvi" and not (c["plan_code"] in ("TRIAL_EXPIRED", "CANCELLED") or not c.get("is_active")):
                continue
            if adm_filter == "B4B pārvaldītie" and c["plan_code"] != "MANAGED":
                continue
            if adm_search.strip():
                q_low = adm_search.strip().lower()
                hay = f"{c['company_name']} {c['contact']} {c['email']} {c['free_note']} {c['plan_label']} {c['onboard_label']}".lower()
                if q_low not in hay:
                    continue
            filtered_comps.append(c)

        if not filtered_comps:
            st.info("Nav klientu, kas atbilst izvēlētajam filtram.")
        else:
            comp_df = pd.DataFrame(filtered_comps)[
                ["company_name", "onboard_label", "plan_label", "is_active", "email", "total_cost_eur", "accounts_count", "reports_count", "free_note"]
            ]
            comp_df.columns = ["Uzņēmums", "PZA statuss", "Abonements", "Aktīvs", "E-pasts", "API (€)", "Konti", "Atsk.", "Piezīmes"]
            with flt_c3:
                st.download_button(
                    "CSV eksports",
                    comp_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"THRUVIA_klienti_{date.today().isoformat()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("#### Ātrā konta un abonementa vadība")
            c_names = [c["company_name"] for c in filtered_comps]
            target_client = st.selectbox("Izvēlieties klientu darbībai", options=c_names, key="adm_target_client_sel")
            sel_obj = next((c for c in enriched if c["company_name"] == target_client), {})

            qa1, qa2, qa3, qa4, qa5 = st.columns([1.15, 1.3, 1.25, 1.15, 0.75])
            with qa1:
                if st.button("Atvērt vidi", type="primary", use_container_width=True):
                    S.active_profile = target_client
                    S.view_mode = "workspace"
                    S.nav_page = "overview"
                    S.pop("admin_reviewing_client", None)
                    S.auto_loaded_latest = False
                    S.pop("result", None)
                    S.pop("cls_df", None)
                    S.pop("loaded_report_meta", None)
                    st.rerun()
            with qa2:
                if st.button("Aktivizēt abon.", use_container_width=True):
                    new_n = build_structured_notes(
                        "ACTIVE",
                        email=sel_obj.get("email", ""),
                        free_note=sel_obj.get("free_note", ""),
                        onboard_status="APPROVED",
                        contact_name=sel_obj.get("contact", ""),
                    )
                    admin_store.create_or_update_company_admin(
                        company_name=target_client,
                        password="",
                        monthly_limit=10000,
                        is_active=True,
                        notes=new_n,
                        reset_usage=False,
                    )
                    st.toast(f"Klientam '{target_client}' aktivizēts maksas abonements!")
                    st.rerun()
            with qa3:
                if st.button("+5d izmēģināj.", use_container_width=True):
                    new_n = build_structured_notes(
                        "TRIAL",
                        email=sel_obj.get("email", ""),
                        trial_start_dt=now_riga().strftime("%Y-%m-%dT%H:%M"),
                        free_note=sel_obj.get("free_note", ""),
                        onboard_status=sel_obj.get("onboard_status", "APPROVED"),
                        contact_name=sel_obj.get("contact", ""),
                    )
                    admin_store.create_or_update_company_admin(
                        company_name=target_client,
                        password="",
                        monthly_limit=10000,
                        is_active=True,
                        notes=new_n,
                        reset_usage=False,
                    )
                    st.toast(f"Klientam '{target_client}' piešķirtas +5 izmēģinājuma dienas!")
                    st.rerun()
            with qa4:
                if st.button("Apturēt piekļ.", use_container_width=True):
                    new_n = build_structured_notes(
                        "CANCELLED",
                        email=sel_obj.get("email", ""),
                        free_note=sel_obj.get("free_note", ""),
                        onboard_status=sel_obj.get("onboard_status", "APPROVED"),
                        contact_name=sel_obj.get("contact", ""),
                    )
                    admin_store.create_or_update_company_admin(
                        company_name=target_client,
                        password="",
                        monthly_limit=10000,
                        is_active=False,
                        notes=new_n,
                        reset_usage=False,
                    )
                    st.toast(f"Klientam '{target_client}' piekļuve apturēta.")
                    st.rerun()
            with qa5:
                if st.button("Dzēst", use_container_width=True):
                    S.confirm_del_client = target_client

            if S.get("confirm_del_client") == target_client:
                st.warning(f"Vai tiešām neatgriezeniski dzēst profilu **{target_client}** un visas tā atskaites?")
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
        edit_mode = st.radio(
            "Režīms",
            options=["Izveidot jaunu klientu", "Rediģēt izvēlēto klientu"],
            horizontal=True,
            label_visibility="collapsed",
            key="adm_edit_mode_radio",
        )
        prefill = (
            next((c for c in enriched if c["company_name"] == S.get("adm_target_client_sel")), {})
            if (edit_mode == "Rediģēt izvēlēto klientu" and enriched)
            else {}
        )

        plan_opts_map = {
            "TRIAL": f"Bezmaksas izmēģinājums ({billing_cfg['trial_days']} d.)",
            "ACTIVE": "Aktīvs maksas abonements",
            "MANAGED": "B4B pārvaldīts / VIP profils",
            "CANCELLED": "Atcelts / Apturēts",
        }
        rev_plan_map = {v: k for k, v in plan_opts_map.items()}
        plan_list = list(plan_opts_map.values())
        cur_p_code = prefill.get("plan_code", "TRIAL")
        if cur_p_code == "TRIAL_EXPIRED":
            cur_p_code = "TRIAL"
        def_p_idx = plan_list.index(plan_opts_map.get(cur_p_code, plan_list[0]))

        onb_keys = list(ONBOARD_LABELS_LV.keys())
        cur_onb_k = prefill.get("onboard_status", "APPROVED")
        def_onb_idx = onb_keys.index(cur_onb_k) if cur_onb_k in onb_keys else onb_keys.index("APPROVED")

        with st.form("admin_manage_client_form"):
            f_cname = st.text_input("Lietotājvārds / Uzņēmums", value=prefill.get("company_name", ""), placeholder="piem., SIA Klients")
            f_contact = st.text_input("Kontaktpersona", value=prefill.get("contact", ""), placeholder="piem., Jānis Bērziņš")
            f_email = st.text_input("Klienta e-pasts", value=prefill.get("email", ""), placeholder="klients@uznemums.lv")
            f_pw = st.text_input("Parole (atstājiet tukšu, ja nemaināt)", type="password")
            f_plan_lbl = st.selectbox("Abonementa plāns", options=plan_list, index=def_p_idx)
            f_onb_key = st.selectbox(
                "PZA / Ievadposma statuss",
                options=onb_keys,
                index=def_onb_idx,
                format_func=lambda k: ONBOARD_LABELS_LV[k],
            )
            f_trial_start = st.date_input("Izmēģinājuma sākuma datums", value=date.today())
            f_active = st.checkbox("Profils ir aktīvs", value=bool(prefill.get("is_active", True)))
            f_reset = st.checkbox("Nullēt API statistikas skaitītāju (0)", value=False)
            f_notes = st.text_input("Piezīmes", value=prefill.get("free_note", ""))
            sub_adm = st.form_submit_button("Saglabāt klienta profilu", type="primary", use_container_width=True)

            if sub_adm:
                sel_p_code = rev_plan_map[f_plan_lbl]
                packed_notes = build_structured_notes(
                    plan_code=sel_p_code,
                    email=f_email,
                    trial_start=f_trial_start,
                    free_note=f_notes,
                    onboard_status=f_onb_key,
                    contact_name=f_contact,
                )
                ok_a, msg_a = admin_store.create_or_update_company_admin(
                    company_name=f_cname,
                    password=f_pw,
                    monthly_limit=10000,
                    is_active=f_active if sel_p_code != "CANCELLED" else False,
                    notes=packed_notes,
                    reset_usage=f_reset,
                )
                if ok_a:
                    st.success(msg_a)
                    st.rerun()
                else:
                    st.error(msg_a)
    st.stop()


# ------------------------------------------------------------------ 3. PROFILA IELĀDE, ONBOARDING UN DEMO REŽĪMS
profile = SupabaseProfileStore(company_name=S.active_profile)

if "fy_start_month" not in S:
    S.fy_start_month = 1
if "fy_start_day" not in S:
    S.fy_start_day = 1
if "nav_page" not in S:
    S.nav_page = "overview"
if "hide_top_toolbar" not in S:
    S.hide_top_toolbar = False

cur_company_row = next(
    (c for c in admin_store.list_all_companies_admin() if c["company_name"] == S.active_profile),
    {},
)
onboard_status = parse_onboarding_status(cur_company_row.get("notes"))
sub_info = parse_subscription_status(cur_company_row.get("notes"), trial_days=billing_cfg["trial_days"])
welcome_seen_flag = extract_note_field(cur_company_row.get("notes"), "WELCOME_SEEN")
S.onboard_status = onboard_status
S.subscription_info = sub_info
S.company_email = extract_note_field(cur_company_row.get("notes"), "EMAIL")
S.company_contact = extract_note_field(cur_company_row.get("notes"), "CONTACT")

is_demo_mode = (not S.get("is_admin", False)) and (onboard_status in ("TUTORIAL", "SUBMIT_PZA", "PENDING_WAIT"))
S.is_demo_mode = is_demo_mode

all_saved_items = profile.list_reports()
fin_reports = [r for r in all_saved_items if not str(r.get("file_name", "")).startswith(("SIM::", "PENDING_PZA::"))]
sim_reports = [r for r in all_saved_items if str(r.get("file_name", "")).startswith("SIM::")]
saved_acc_count = profile.count_saved()

if is_demo_mode:
    if not S.get("demo_initialized", False) or S.get("loaded_report_meta", {}).get("file_name") != "DEMO::DATA":
        demo_res, demo_acc = build_demo_company_data()
        S.result = apply_custom_row_order_to_result(demo_res)
        S.loaded_account_details = demo_acc
        S.loaded_report_meta = {
            "id": None,
            "report_name": "SIA Demo Ražošana (2026. gads)",
            "file_name": "DEMO::DATA",
        }
        S.current_report_id = None
        S.has_unsaved_changes = False
        S.demo_initialized = True
        S.pop("cls_df", None)
        S.pop("analysis", None)
else:
    if S.get("demo_initialized", False):
        S.demo_initialized = False
        S.pop("result", None)
        S.pop("loaded_report_meta", None)
        S.auto_loaded_latest = False

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
                S.result = apply_custom_row_order_to_result(strip_excluded_rows(init_res))
                S.loaded_report_meta = loaded_init
                S.current_report_id = loaded_init["id"]
                S.has_unsaved_changes = False
                comm_clean, acc_snap = unpack_commentary_and_account_details(loaded_init.get("commentary"))
                S.loaded_account_details = acc_snap
                if comm_clean:
                    S.commentary = comm_clean
                else:
                    S.pop("commentary", None)


# ------------------------------------------------------------------ VIENOTS DARBA PERIODA SINHRONIZATORS
def ensure_shared_working_period(all_months: list[str], months_by_year: dict[int, list[str]]) -> tuple[list[str], list[str]]:
    preset_opts = [f"Visi mēneši ({len(all_months)} mēn.)"]
    for yr in sorted(months_by_year.keys()):
        preset_opts.append(f"{yr}. gads ({len(months_by_year[yr])} mēn.)")
    if len(all_months) > 12:
        preset_opts.append("Pēdējie 12 mēneši")
    if len(all_months) >= 6:
        preset_opts.append("Pirmie 6 mēneši")
        preset_opts.append("Pēdējie 6 mēneši")
    preset_opts.append("Manuāla mēnešu izlase")

    if "working_period_preset" not in S or S.working_period_preset not in preset_opts:
        S.working_period_preset = preset_opts[0]
    if "working_months_active" not in S or not set(S.working_months_active).issubset(set(all_months)) or not S.working_months_active:
        S.working_months_active = list(all_months)

    return preset_opts, S.working_months_active


# ------------------------------------------------------------------ ASISTENTA IEKŠĒJAIS LOGS (THRUVIA EKSPERTS)
def _render_assistant_popover_content(form_key: str) -> None:
    top_c1, top_c2 = st.columns([1.8, 1.2], vertical_alignment="center")
    with top_c1:
        st.markdown("**THRUVIA Eksperts**")
    with top_c2:
        st.toggle(
            "Paplašināt logu",
            key=f"exp_tog_{form_key}",
            value=bool(S.get("assistant_expanded", False)),
            on_change=lambda: S.update(assistant_expanded=not S.get("assistant_expanded", False)),
        )

    if "global_chat" not in S:
        S.global_chat = []

    box_h = 430 if S.get("assistant_expanded", False) else 255
    chat_box = st.container(height=box_h)
    with chat_box:
        if not S.global_chat:
            st.caption(
                "Sveiki! Jautājiet par skaitļiem, ko nozīmē rādītāji, kur aplikācijā kas atrodas, "
                "vai palūdziet pārdēvēt, apvienot un sakārtot apakšgrupas!"
            )
        else:
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

            components.html(
                """
                <script>
                setTimeout(function() {
                    const anchor = window.parent.document.getElementById('toc-latest-answer-top');
                    if (anchor) { anchor.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
                }, 80);
                </script>
                """,
                height=0,
            )

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
        sub_q = f_c1.form_submit_button("Sūtīt tekstu", type="primary", use_container_width=True)
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
            S["asist_audio_ver"] = S.get("asist_audio_ver", 0) + 1
            st.rerun()

    if hasattr(st, "audio_input"):
        v_col1, v_col2 = st.columns([1.8, 1.2], vertical_alignment="bottom")
        with v_col1:
            asist_audio = st.audio_input(
                "🎙️ Balss jautājums (spiediet mikrofonu)",
                key=f"asist_voice_{form_key}_{S.get('asist_audio_ver', 0)}",
            )
        with v_col2:
            manual_voice_send = st.button(
                "Sūtīt balss ziņu",
                key=f"send_voice_always_{form_key}",
                type="primary",
                use_container_width=True,
            )

        if asist_audio is not None:
            a_bytes = asist_audio.getvalue()
            a_hash = hashlib.md5(a_bytes).hexdigest()
            if S.get("last_asist_audio_hash") != a_hash or manual_voice_send:
                S["last_asist_audio_hash"] = a_hash
                with st.spinner("Atpazīst balss jautājumu un gatavo atbildi..."):
                    transcribed_q, err_v = transcribe_wav_bytes_lv(a_bytes)
                    if transcribed_q:
                        S.global_chat.append({"role": "user", "content": transcribed_q})
                        try:
                            ans, act_obj, t_tok, t_eur = ask_global_assistant(transcribed_q, profile)
                            profile.log_api_usage(t_tok, t_eur)
                            if act_obj:
                                S.pending_assistant_action = act_obj
                        except Exception as e:
                            ans = f"Kļūda: {e}"
                        S.global_chat.append({"role": "assistant", "content": ans})
                        S["asist_audio_ver"] = S.get("asist_audio_ver", 0) + 1
                        st.rerun()
                    elif err_v:
                        st.warning(err_v)
        elif manual_voice_send:
            st.caption("ℹ️ Vispirms nospiediet sarkano kvadrātiņu (Stop) pie mikrofona, lai pabeigtu balss ierakstu, un tas uzreiz nosūtīsies!")


# ------------------------------------------------------------------ 4. PELDOŠĀ AUGŠĒJĀ RĪKJOSLA (THRUVIA)
if is_demo_mode:
    PRIMARY_NAV = {
        "overview": "Pārskats (Demo)",
        "simulation": "Scenāriju simulācija (Demo)",
        "accounts_explorer": "Kontu struktūra (Demo)",
        "submit_pza": "📤 Iesniegt savu PZA",
    }
else:
    PRIMARY_NAV = {
        "overview": "Pārskats",
        "simulation": "Scenāriju simulācija",
        "accounts_explorer": "Kontu struktūra",
        "archive": "Atskaites un Simulācijas",
        "ingest": "+ Jauns PZA",
    }

active_rep_label = (
    S.loaded_report_meta.get("report_name")
    if S.get("loaded_report_meta")
    else ("Jauns aprēķins" if "result" in S else "Nav izvēlēta")
)

unsaved_badge_html = (
    '&nbsp; <span class="unsaved-pill">● Nesaglabātas izmaiņas</span>'
    if (S.get("has_unsaved_changes") and not is_demo_mode)
    else ""
)
demo_badge_html = '&nbsp; <span class="demo-pill">DEMO</span>' if is_demo_mode else ""
trial_badge_html = (
    f'&nbsp; <span class="trial-pill">Izmēģinājums: {sub_info["days_left"]} d.</span>'
    if (not is_demo_mode and sub_info["code"] == "TRIAL" and sub_info["days_left"] is not None)
    else ""
)


def _render_account_menu_popover(key_sfx: str) -> None:
    st.markdown(f"**{S.active_profile}**")
    st.caption(sub_info["label"])
    st.markdown("---")
    if st.button("Uzstādījumi un Abonements", key=f"acc_menu_set_{key_sfx}", use_container_width=True):
        S.nav_page = "settings"
        st.query_params["p"] = "settings"
        st.rerun()
    if S.get("is_admin", False):
        if st.button("Administratora panelis", key=f"acc_menu_adm_{key_sfx}", type="primary", use_container_width=True):
            S.view_mode = "admin_panel"
            S.pop("admin_reviewing_client", None)
            st.rerun()
    st.markdown("---")
    if st.button("Iziet no sistēmas", key=f"acc_menu_out_{key_sfx}", use_container_width=True):
        st.query_params.clear()
        S.clear()
        st.rerun()


with st.container():
    st.markdown('<div class="sticky-nav-marker"></div>', unsafe_allow_html=True)

    if S.hide_top_toolbar:
        _, col_asist, col_prof, col_arr = st.columns([5.3, 1.0, 0.9, 0.35], vertical_alignment="center")
        with col_asist:
            with st.popover("Asistents", use_container_width=True):
                _render_assistant_popover_content("side_agent_form_min")
        with col_prof:
            with st.popover("Profils", use_container_width=True):
                _render_account_menu_popover("min")
        with col_arr:
            if st.button("▼", help="Rādīt rīkjoslu", use_container_width=True):
                S.hide_top_toolbar = False
                st.rerun()
    else:
        hdr_left, hdr_asist, hdr_about, hdr_prof, hdr_arr = st.columns([4.3, 1.0, 0.95, 0.9, 0.35], vertical_alignment="center")
        with hdr_left:
            st.markdown(
                f'<div class="top-brand-bar">'
                f'<span class="top-brand-title">THRUVIA</span>'
                f'<span class="top-brand-sub"><b>{S.active_profile}</b> &nbsp;|&nbsp; Atskaite: <b>{active_rep_label}</b>{demo_badge_html}{unsaved_badge_html}{trial_badge_html}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with hdr_asist:
            with st.popover("Asistents", use_container_width=True):
                _render_assistant_popover_content("side_agent_form")
        with hdr_about:
            about_type = "primary" if S.nav_page == "about" else "secondary"
            if st.button("Par THRUVIA", key="top_about_btn", type=about_type, use_container_width=True):
                S.nav_page = "about"
                st.query_params["p"] = "about"
                st.rerun()
        with hdr_prof:
            with st.popover("Profils", use_container_width=True):
                _render_account_menu_popover("full")
        with hdr_arr:
            if st.button("▲", help="Paslēpt rīkjoslu", use_container_width=True):
                S.hide_top_toolbar = True
                st.rerun()

        nav_widths = [1.05, 1.3, 1.25, 1.25] if is_demo_mode else [1.0, 1.25, 1.15, 1.35, 1.05]
        nav_cols = st.columns(nav_widths)
        for idx_n, (n_key, n_label) in enumerate(PRIMARY_NAV.items()):
            btn_type = "primary" if S.nav_page == n_key else "secondary"
            if nav_cols[idx_n].button(n_label, key=f"top_nav_{n_key}", type=btn_type, use_container_width=True):
                S.nav_page = n_key
                st.query_params["p"] = n_key
                st.rerun()


# ------------------------------------------------------------------ 5. ADMIN PĀRBAUDES JOSLA (KAD ADMINS PĀRBAUDA KLIENTA PZA)
def finalize_admin_pza_approval(client_name: str, client_store: SupabaseProfileStore) -> None:
    """Apstiprina klienta PZA, izdzēš PENDING_PZA melnrakstus, sāk 5d trial tieši tagad un nosūta e-pastu."""
    _, end_dt = update_company_onboarding_status(
        admin_store,
        client_name,
        "APPROVED",
        reset_trial_now=True,
        welcome_seen="0",
    )
    for pf in client_store.list_reports():
        if str(pf.get("file_name", "")).startswith("PENDING_PZA::"):
            client_store.delete_report(pf["id"])

    charge_str = (
        end_dt.strftime("%d.%m.%Y. plkst. %H:%M")
        if end_dt
        else (now_riga() + timedelta(days=billing_cfg["trial_days"])).strftime("%d.%m.%Y. plkst. %H:%M")
    )
    sent_ok, mail_msg, subj, body = send_approval_email_to_client(
        to_email=S.get("company_email", ""),
        company_name=client_name,
        contact_name=S.get("company_contact", ""),
        first_charge_str=charge_str,
        billing_cfg=billing_cfg,
    )
    S.last_approval_email_result = {
        "sent": sent_ok,
        "msg": mail_msg,
        "to_email": S.get("company_email", ""),
        "subject": subj,
        "body": body,
    }
    S.pop("admin_reviewing_client", None)
    S.view_mode = "admin_panel"
    st.rerun()


if S.get("is_admin", False) and onboard_status in ("PENDING_WAIT", "PENDING_IMMEDIATE"):
    with st.container(border=True):
        ar1, ar2, ar3 = st.columns([2.8, 1.3, 0.9], vertical_alignment="center")
        with ar1:
            st.markdown(
                f"🛠️ **Admin pārbaudes režīms (`{S.active_profile}`):** "
                "Klients šobrīd joprojām redz Demo. Kad konti un pārskats ir pārbaudīti, spiediet apstiprināt — "
                "tikai tad klientam atvērsies īstais profils un aizies apstiprinājuma e-pasts."
            )
        with ar2:
            if st.button("✅ Apstiprināt PZA un nosūtīt e-pastu", type="primary", use_container_width=True, key="adm_bar_approve_now"):
                finalize_admin_pza_approval(S.active_profile, profile)
        with ar3:
            if st.button("← Admin panelis", use_container_width=True, key="adm_bar_back"):
                S.pop("admin_reviewing_client", None)
                S.view_mode = "admin_panel"
                st.rerun()


# ------------------------------------------------------------------ 6. KLIENTA PIRMĀS IEIEŠANAS SAGAIDĪŠANA PĒC PZA APSTIPRINĀŠANAS
if (
    not S.get("is_admin", False)
    and onboard_status == "APPROVED"
    and welcome_seen_flag == "0"
    and S.nav_page not in ("about",)
):
    charge_txt = sub_info.get("first_charge_str") or "pēc 5 dienām"
    left_txt = sub_info.get("time_left_str") or f"{billing_cfg['trial_days']} dienas"
    st.markdown(
        f'<div class="welcome-approved-box">'
        f'<div style="font-size:17px; font-weight:800; color:#065F46; margin-bottom:6px;">🎉 Jūsu uzņēmuma PZA ir pārbaudīts un apstiprināts!</div>'
        f'<div style="font-size:13.5px; color:#0F172A; line-height:1.55;">'
        f'Tagad varat pilnvērtīgi darboties THRUVIA ar sava uzņēmuma apstiprinātajiem datiem.<br>'
        f'• <b>Bezmaksas izmēģinājums ir sācies:</b> <b>{billing_cfg["trial_days"]} dienas bez maksas</b> (šobrīd atlikušas: <b>{left_txt}</b>).<br>'
        f'• <b>Pirmā abonementa apmaksa ({billing_cfg["monthly_price_eur"]} €/mēn.):</b> tiks iekasēta <b>{charge_txt}</b> (Latvijas laiks), '
        f'ja iepriekš abonementu neatcelsiet sadaļā <i>Profils → Uzstādījumi un Abonements</i>.'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    wc1, wc2, _ = st.columns([1.3, 1.5, 2.2])
    with wc1:
        if st.button("Sapratu, sākt darbu ✓", type="primary", use_container_width=True, key="btn_welcome_ack"):
            update_company_onboarding_status(admin_store, S.active_profile, "APPROVED", reset_trial_now=False, welcome_seen="1")
            st.rerun()
    with wc2:
        if billing_cfg["stripe_checkout_url"]:
            st.link_button("💳 Pievienot maksājumu karti", billing_cfg["stripe_checkout_url"], use_container_width=True)


# ------------------------------------------------------------------ 7. FOKUSĒTA 8 SOĻU INTERAKTĪVĀ DEMO APMĀCĪBA (BEZ LIEKA TROKŠŅA)
TUTORIAL_STEPS = [
    {
        "step": 1, "page": "overview", "focus": "period",
        "title": "Darba periods un Excel eksports",
        "text": "Augšā izvēlieties periodu (piem., <b>Visi mēneši</b> vai <b>Pēdējie 6 mēneši</b>) — šis pats periods automātiski darbosies arī Simulācijā. Blakus variet lejupielādēt tabulu Excel formātā.",
    },
    {
        "step": 2, "page": "overview", "focus": "main",
        "title": "Galvenā TOC tabula (S, TVC, T, OE, NP)",
        "text": "<b>S</b> (dzeltens) = Ieņēmumi · <b>TVC</b> (brūns) = pilnībā mainīgās izmaksas · <b>T</b> (zils) = Caurlaidība (S − TVC) · <b>OE</b> (persiku) = darbības izmaksas (algas vienmēr augšā) · <b>NP</b> (zaļš) = TOC Neto peļņa.",
    },
    {
        "step": 3, "page": "overview", "focus": "kpi",
        "title": "Produktivitātes rādītāji (RATIO)",
        "text": "<b>PR = T / OE</b> parāda, cik eiro Caurlaidības rodas uz 1 € darbības izmaksu (zaļš ≥ 1.15, sarkans &lt; 1.00). Zemāk redzama arī <b>T%</b> un <b>algu atdeve</b> (cik eiro pārdošanas nes katrs algu eiro).",
    },
    {
        "step": 4, "page": "overview", "focus": "cum",
        "title": "Uzkrājošie rādītāji",
        "text": "Skaitļi tiek summēti no gada sākuma uz priekšu — tā uzreiz redzams brīdis, kad uzkrātā Caurlaidība pārsniedz izmaksas un izlīdzinās atsevišķu mēnešu svārstības.",
    },
    {
        "step": 5, "page": "overview", "focus": "charts",
        "title": "Dinamikas grafiki un Eksperta secinājumi",
        "text": "Augšējā grafikā redzama galveno rādītāju kustība pa mēnešiem (pārslēdzama uz <i>Uzkrājošie rādītāji</i>), zemāk — produktivitātes tendences un poga <b>Pieprasīt eksperta ieteikumus</b>.",
    },
    {
        "step": 6, "page": "accounts_explorer", "focus": "accounts",
        "title": "Kontu struktūra un apakšgrupu vadība",
        "text": "Šeit redzams, kuri PZA konti veido katru tabulas rindu. Augšējā vadības panelī ar pāris klikšķiem varat <b>pārvietot kontus</b>, <b>apvienot</b>, <b>izveidot</b> vai <b>pārdēvēt</b> apakšgrupas.",
    },
    {
        "step": 7, "page": "simulation", "focus": "sim",
        "title": "Scenāriju simulācija un Jauns projekts",
        "text": "Ievadiet izmaiņas tabulā (% vai €), aprakstiet situāciju tekstā/balsī vai ieslēdziet <b>Jauns produkts / projekts</b> — tabula zemāk uzreiz parādīs ietekmi uz T, NP un PR.",
    },
    {
        "step": 8, "page": "overview", "focus": "assistant",
        "title": "MI Asistents un sava PZA pievienošana",
        "text": "Augšā labajā stūrī atveriet <b>Asistents</b> — tas izskaidros skaitļus, parādīs kur kas atrodas un pēc pieprasījuma pārdēvēs vai sakārtos rindas. Tagad esat gatavs pievienot sava uzņēmuma PZA!",
    },
]

tut_active_focus: str | None = None

if not S.get("is_admin", False) and onboard_status == "TUTORIAL" and S.nav_page not in ("submit_pza", "about", "settings"):
    cur_t_step = max(1, min(len(TUTORIAL_STEPS), int(S.get("tutorial_step", 1))))
    t_cfg = TUTORIAL_STEPS[cur_t_step - 1]
    tut_active_focus = t_cfg["focus"]

    st.markdown(
        f'<div class="tut-compact-bar">'
        f'<span class="tut-step-pill">SOLIS {cur_t_step} / {len(TUTORIAL_STEPS)}</span>'
        f'<b style="font-size:14.5px; color:#1E1B4B;">{t_cfg["title"]}</b>'
        f'<div style="font-size:13px; color:#1E293B; margin-top:5px; line-height:1.45;">{t_cfg["text"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tb1, tb2, tb3 = st.columns([0.9, 1.3, 1.3], vertical_alignment="center")
    with tb1:
        if cur_t_step > 1 and st.button("◀ Iepriekšējais", use_container_width=True, key="tut_prev_btn"):
            new_s = cur_t_step - 1
            S.tutorial_step = new_s
            S.nav_page = TUTORIAL_STEPS[new_s - 1]["page"]
            st.rerun()
    with tb2:
        if cur_t_step < len(TUTORIAL_STEPS):
            if st.button("Nākamais solis ▶", type="primary", use_container_width=True, key="tut_next_btn"):
                new_s = cur_t_step + 1
                S.tutorial_step = new_s
                S.nav_page = TUTORIAL_STEPS[new_s - 1]["page"]
                st.rerun()
        else:
            if st.button("Pabeigt un pievienot savu PZA →", type="primary", use_container_width=True, key="tut_finish_btn"):
                update_company_onboarding_status(admin_store, S.active_profile, "SUBMIT_PZA")
                S.nav_page = "submit_pza"
                st.rerun()
    with tb3:
        if cur_t_step < len(TUTORIAL_STEPS) and st.button("Izlaist apmācību → Pievienot PZA", use_container_width=True, key="tut_skip_btn"):
            update_company_onboarding_status(admin_store, S.active_profile, "SUBMIT_PZA")
            S.nav_page = "submit_pza"
            st.rerun()

    # Automātiski pārtin lapu tieši uz izcelto apmācības elementu (#tut-focus-anchor)
    if cur_t_step in (3, 4, 5):
        components.html(
            """
            <script>
            setTimeout(function() {
                const el = window.parent.document.getElementById('tut-focus-anchor');
                if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
            }, 180);
            </script>
            """,
            height=0,
        )

elif not S.get("is_admin", False) and onboard_status == "PENDING_WAIT" and S.nav_page not in ("submit_pza", "about", "settings"):
    with st.container(border=True):
        pw_c1, pw_c2 = st.columns([2.6, 1.4], vertical_alignment="center")
        with pw_c1:
            st.markdown(
                "**⏳ Jūsu PZA fails tiek pārbaudīts!**  \n"
                "Tiklīdz pārbaudīsim un apstiprināsim jūsu PZA struktūru, nosūtīsim jums e-pastu un sāksies jūsu 5 bezmaksas izmēģinājuma dienas. "
                "Tikmēr brīvi izmēģiniet sistēmu Demo versijā."
            )
        with pw_c2:
            if billing_cfg["stripe_checkout_url"]:
                st.link_button("💳 Pievienot karti / Abonēt", billing_cfg["stripe_checkout_url"], type="primary", use_container_width=True)
            if st.button("Iesniegt citu PZA / Mainīt režīmu", use_container_width=True, key="pw_goto_sub"):
                S.nav_page = "submit_pza"
                st.rerun()

elif not S.get("is_admin", False) and onboard_status == "PENDING_IMMEDIATE" and S.nav_page not in ("about", "settings"):
    st.markdown(
        '<div class="disclaimer-urgent-box">'
        '<b>⚠️ BRĪDINĀJUMS — PZA struktūra vēl nav pārbaudīta!</b><br>'
        'Līdz mūsu speciālista apstiprinājumam automātiskā kontu grupēšana un aprēķini var saturēt neprecizitātes. '
        'Tiklīdz pārbaude būs pabeigta, nosūtīsim e-pastu un šis brīdinājums pazudīs.'
        '</div>',
        unsafe_allow_html=True,
    )

if not S.get("is_admin", False) and onboard_status == "SUBMIT_PZA" and S.nav_page not in ("submit_pza", "about", "settings"):
    S.nav_page = "submit_pza"

if sub_info["expired"] and not S.get("is_admin", False) and S.nav_page not in ("settings", "about"):
    st.warning(
        f"**{sub_info['label']}.** Lai turpinātu darbu ar saviem datiem, lūdzu aktivizējiet mēneša abonementu sadaļā **Profils → Uzstādījumi un Abonements**."
    )
    if billing_cfg["stripe_checkout_url"]:
        st.link_button("💳 Aktivizēt mēneša abonementu (Stripe)", billing_cfg["stripe_checkout_url"], type="primary")
    elif st.button("Doties uz abonementa aktivizēšanu →", type="primary"):
        S.nav_page = "settings"
        st.rerun()
    render_app_footer_disclaimer()
    st.stop()


# === BEIDZAS 2/3 DAĻA (zem šīs rindas līmēsi 3/3 daļu) ===
# === SĀKAS 3/3 DAĻA ===

# ==============================================================================
# POSMS 0: PIRMREIZĒJĀ PZA IESNIEGŠANA PĀRBAUDEI (2 VARIANTI + ABONEMENTS)
# ==============================================================================
if S.nav_page == "submit_pza":
    st.subheader("Iesniedziet sava uzņēmuma PZA failu pārbaudei")
    st.caption(
        "Lai jūsu pirmie rezultāti būtu maksimāli precīzi, mēs pārbaudām katra jaunā uzņēmuma PZA faila struktūru, "
        "starpsummas un kontu grupēšanu. Augšupielādējiet sava uzņēmuma mēnešu PZA (Excel vai CSV) un izvēlieties vēlamo režīmu."
    )

    if onboard_status == "PENDING_WAIT":
        st.success(
            "Jūsu PZA fails jau ir iesniegts pārbaudei! Kamēr pārbaudām un sakārtojam kontus, varat turpināt darboties **Demo režīmā** "
            "vai iesniegt atjaunotu failu zemāk."
        )

    sub_left, sub_right = st.columns([1.85, 1.15], vertical_alignment="top")

    with sub_left:
        with st.container(border=True):
            st.markdown("#### 1. Pievienojiet sava uzņēmuma PZA failu")
            first_pza_file = st.file_uploader(
                "Peļņas vai zaudējumu aprēķins pa mēnešiem (Excel vai CSV)",
                type=["xlsx", "xlsm", "xls", "csv"],
                key="onboarding_pza_uploader",
            )
            first_pza_note = st.text_area(
                "Komentārs vai norādes mūsu speciālistam (neobligāti)",
                placeholder="Piemēram: Kurā darblapā ir aktuālais PZA, vai kādi specifiski konti, kuriem pievērst uzmanību...",
                height=78,
                key="onboarding_pza_note",
            )

            st.markdown("#### 2. Izvēlieties, kā vēlaties rīkoties pārbaudes laikā")
            submit_choice = st.radio(
                "Pārbaudes režīms",
                options=[
                    "Nosūtīt pārbaudei un gaidīt rezultātus (Ieteicams — tikmēr darbojos Demo versijā)",
                    "Nosūtīt pārbaudei un sākt darboties ar savu failu uzreiz",
                ],
                key="onboarding_submit_mode_radio",
            )

            is_immediate_choice = submit_choice.startswith("Nosūtīt pārbaudei un sākt")
            imm_ack = True

            if is_immediate_choice:
                st.markdown(
                    '<div class="disclaimer-urgent-box">'
                    '<b>⚠️ SVARĪGS BRĪDINĀJUMS (DISCLAIMER)</b><br>'
                    'Jūsu uzņēmuma PZA faila struktūra vēl <b>nav pārbaudīta un apstiprināta</b>. '
                    'Sākot darbu uzreiz, automātiskā datu nolasīšana, starpsummu atdalīšana un kontu grupēšana '
                    '<b>var būt neprecīza vai kļūdaina</b>. Pirms pieņemt biznesa lēmumus, sagaidiet mūsu apstiprinājumu! '
                    'Fails paralēli tiks nosūtīts arī mums pārbaudei.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                imm_ack = st.checkbox(
                    "Saprotu un piekrītu, ka līdz PZA struktūras pārbaudei automātiskā nolasīšana un kontu grupēšana var saturēt kļūdas.",
                    value=False,
                    key="onboarding_imm_disclaimer_chk",
                )

            st.write("")
            btn_label = (
                "Nosūtīt pārbaudei un sākt darboties uzreiz →"
                if is_immediate_choice
                else "Nosūtīt PZA pārbaudei un turpināt Demo režīmā →"
            )
            if st.button(btn_label, type="primary", use_container_width=True, key="onboarding_submit_btn"):
                if first_pza_file is None:
                    st.error("Lūdzu vispirms augšupielādējiet sava uzņēmuma PZA failu (Excel vai CSV).")
                elif is_immediate_choice and not imm_ack:
                    st.error("Lai sāktu darboties uzreiz bez iepriekšējas pārbaudes, lūdzu atzīmējiet brīdinājuma apstiprinājumu.")
                else:
                    raw_bytes = first_pza_file.getvalue()
                    mode_tag = "IMMEDIATE" if is_immediate_choice else "WAIT"
                    with st.spinner("Nosūta PZA failu pārbaudei..."):
                        ok_sub, msg_sub = submit_pza_for_admin_review(
                            profile,
                            first_pza_file.name,
                            raw_bytes,
                            mode=mode_tag,
                            user_note=first_pza_note,
                        )
                    if ok_sub:
                        new_status = "PENDING_IMMEDIATE" if is_immediate_choice else "PENDING_WAIT"
                        update_company_onboarding_status(
                            admin_store,
                            S.active_profile,
                            new_status,
                            reset_trial_now=is_immediate_choice,
                            welcome_seen="0",
                        )
                        if is_immediate_choice:
                            S.demo_initialized = False
                            S.pop("result", None)
                            S.pop("loaded_report_meta", None)
                            S.preloaded_pza_name = first_pza_file.name
                            S.preloaded_pza_bytes = raw_bytes
                            S.nav_page = "ingest"
                        else:
                            S.pza_just_submitted = True
                            S.nav_page = "overview"
                        st.toast(msg_sub)
                        st.rerun()
                    else:
                        st.error(msg_sub)

        if st.button("← Atgriezties uz Demo apmācību", key="back_to_demo_from_sub"):
            update_company_onboarding_status(admin_store, S.active_profile, "TUTORIAL")
            S.tutorial_step = 1
            S.nav_page = "overview"
            st.rerun()

    with sub_right:
        with st.container(border=True):
            st.markdown("#### 💳 Abonements un kartes pievienošana")
            st.markdown(
                f"Pēc jūsu pirmā PZA apstiprināšanas jums sāksies **{billing_cfg['trial_days']} dienu bezmaksas izmēģinājums** ar jūsu uzņēmuma datiem."
            )
            st.markdown(
                f"Pēc izmēģinājuma perioda: **{billing_cfg['monthly_price_eur']} €/mēn.** (automātiska apmaksa caur Stripe, "
                "rēķini uz e-pastu un iespēja atcelt jebkurā brīdī)."
            )
            st.caption("Maksājumu karti varat pievienot jau tagad vai arī vēlāk sadaļā **Profils → Uzstādījumi un Abonements**.")

            if billing_cfg["stripe_checkout_url"]:
                chk_url = billing_cfg["stripe_checkout_url"]
                if S.get("company_email") and "prefilled_email" not in chk_url:
                    sep = "&" if "?" in chk_url else "?"
                    chk_url = f"{chk_url}{sep}prefilled_email={S['company_email']}"
                st.link_button("💳 Pievienot karti / Aktivizēt abonementu", chk_url, type="primary", use_container_width=True)
            else:
                st.info(f"Karšu apmaksas saite tiks pieslēgta drīzumā. Jautājumiem: **{billing_cfg['support_email']}**")

        with st.container(border=True):
            st.markdown("#### 🔒 Datu drošība")
            st.caption(
                "Jūsu augšupielādētais PZA fails tiek izmantots tikai jūsu profila iestatīšanai un analīzei. "
                "Uzņēmuma finanšu dati nekad netiek izmantoti MI modeļu apmācībai."
            )


# ==============================================================================
# POSMS 1: PĀRSKATS (OVERVIEW) — 14/15 MĒNEŠU SLIEKSNIS + 8 SOĻU SPOTLIGHT
# ==============================================================================
elif S.nav_page == "overview":
    if S.pop("pza_just_submitted", False):
        st.success(
            "Paldies! Jūsu PZA fails ir nosūtīts pārbaudei. Tiklīdz būsim to pārbaudījuši un apstiprinājuši, "
            "nosūtīsim jums e-pastu un jūsu profils tiks aktivizēts. Tikmēr droši izmēģiniet Demo versiju!"
        )

    if "result" not in S or S.result is None:
        st.info("Šajā profilā vēl nav nevienas saglabātas atskaites. Pārejiet uz sadaļu **+ Jauns PZA**, lai augšupielādētu pirmo PZA failu.")
        if st.button("Doties uz PZA faila augšupielādi →", type="primary"):
            S.nav_page = "ingest"
            st.rerun()
        render_app_footer_disclaimer()
        st.stop()

    res = apply_custom_row_order_to_result(strip_excluded_rows(S.result))
    all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c or "KOPĀ" in c or "TOTAL" in c)]
    if not all_months:
        all_months = list(res.amounts.columns)

    months_by_year: dict[int, list[str]] = {}
    for m_col in all_months:
        _, yr = parse_period_month_year(m_col)
        if yr:
            months_by_year.setdefault(yr, []).append(m_col)

    fin_preset_opts, _ = ensure_shared_working_period(all_months, months_by_year)

    def _on_working_preset_change():
        p = S.working_period_preset
        if p.startswith("Visi mēneši"):
            S.working_months_active = list(all_months)
        elif p == "Pēdējie 12 mēneši":
            S.working_months_active = list(all_months[-12:])
        elif p == "Pirmie 6 mēneši":
            S.working_months_active = list(all_months[:6])
        elif p == "Pēdējie 6 mēneši":
            S.working_months_active = list(all_months[-6:])
        else:
            for yr_k, m_list in months_by_year.items():
                if p.startswith(f"{yr_k}. gads"):
                    S.working_months_active = list(m_list)
                    break

    def _on_working_months_change():
        cur = S.working_months_active
        if cur == list(all_months):
            S.working_period_preset = fin_preset_opts[0]
        else:
            S.working_period_preset = "Manuāla mēnešu izlase"

    # Apmācības fokusētie marķieri (bez lieka teksta dublēšanas!)
    if tut_active_focus == "period":
        st.markdown('<div class="tut-section-tag">👇 1. SOLIS — ŠEIT IZVĒLIETIES DARBA PERIODU UN EXCEL EKSPORTU</div>', unsafe_allow_html=True)
    elif tut_active_focus == "assistant":
        st.markdown('<div class="tut-section-tag">👆 8. SOLIS — SPIEDIET POGU "ASISTENTS" AUGŠĒJĀ LABAJĀ STŪRĪ</div>', unsafe_allow_html=True)

    bar_c1, bar_c2, bar_c3, bar_c4, _ = st.columns([1.4, 1.2, 1.45, 1.1, 0.85], vertical_alignment="bottom")
    with bar_c1:
        st.selectbox(
            "Aktīvais darba periods (arī Simulācijai)",
            options=fin_preset_opts,
            key="working_period_preset",
            on_change=_on_working_preset_change,
        )

    cur_fin_months = S.get("working_months_active", all_months)
    with bar_c2:
        with st.popover(f"Mēnešu izlase ({len(cur_fin_months)}/{len(all_months)})", use_container_width=True):
            st.caption("Atzīmējiet mēnešus, ar kuriem vēlaties strādāt Pārskatā un Simulācijā:")
            st.multiselect(
                "Izvēlētie mēneši",
                options=all_months,
                key="working_months_active",
                on_change=_on_working_months_change,
                label_visibility="collapsed",
            )

    fin_selected_months = [m for m in all_months if m in S.get("working_months_active", all_months)]
    if not fin_selected_months:
        st.warning("Izvēlieties vismaz vienu mēnesi finanšu pārskata tabulai.")
        st.stop()

    view_res = filter_result_by_months(res, fin_selected_months, all_months)

    with bar_c3:
        if is_demo_mode:
            st.button("Demo skats (bez saglabāšanas)", disabled=True, use_container_width=True)
        else:
            save_btn_lbl = "💾 Saglabāt izmaiņas / Versiju" if S.get("has_unsaved_changes") else "Saglabāt skatu / Versiju"
            with st.popover(save_btn_lbl, use_container_width=True):
                st.markdown("**Saglabāt pašreizējo pārskatu**")
                st.caption(
                    "Oriģinālā atskaite arhīvā vienmēr paliek drošībā! Šeit varat saglabāt savas kontu izmaiņas vai izvēlēto mēnešu periodu kā **jaunu atskaites versiju** vai atjaunināt esošo."
                )
                default_ver_name = (
                    f"{active_rep_label} ({fin_selected_months[0]}–{fin_selected_months[-1]})"
                    if len(fin_selected_months) < len(all_months)
                    else f"{active_rep_label} (v2)"
                )
                new_ver_name_in = st.text_input("Nosaukums jaunai versijai", value=default_ver_name, key="ov_save_ver_name")
                save_scope = st.radio(
                    "Ko iekļaut saglabātajā atskaitē?",
                    options=[
                        f"Tikai izvēlēto darba periodu ({len(fin_selected_months)} mēn.)",
                        f"Visus faila mēnešus ({len(all_months)} mēn.)",
                    ],
                    key="ov_save_scope_radio",
                )
                target_res_obj = view_res if save_scope.startswith("Tikai") else res

                if st.button("➕ Saglabāt kā JAUNU atskaiti (vecā paliek arhīvā)", type="primary", use_container_width=True):
                    ok_sv, _ = save_report_as_new_or_update(profile, target_res_obj, new_ver_name_in.strip())
                    if ok_sv:
                        if save_scope.startswith("Tikai"):
                            S.result = view_res
                            S.pop("working_months_active", None)
                            S.pop("working_period_preset", None)
                        st.toast(f"Jauna atskaites versija '{new_ver_name_in}' saglabāta! Vecā versija joprojām ir atrodama arhīvā.")
                        st.rerun()

                if S.get("current_report_id") and st.button("🔄 Pārrakstīt pašreizējo atskaiti", use_container_width=True):
                    ok_up, _ = save_report_as_new_or_update(profile, target_res_obj, active_rep_label)
                    if ok_up:
                        st.toast(f"Atskaite '{active_rep_label}' atjaunināta!")
                        st.rerun()

    with bar_c4:
        with st.popover("Lejupielādēt Excel", use_container_width=True):
            st.caption("Izvēlieties Excel faila veidu:")
            try:
                xl_f, xl_v = io.BytesIO(), io.BytesIO()
                build_excel(view_res, xl_f, formulas=True)
                build_excel(view_res, xl_v, formulas=False)
                st.download_button(
                    "Ar Excel formulām (.xlsx)",
                    xl_f.getvalue(),
                    "THRUVIA_Finansu_Analize_formulas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )
                st.download_button(
                    "Tikai vērtības (.xlsx)",
                    xl_v.getvalue(),
                    "THRUVIA_Finansu_Analize_vertibas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                xl_buf = io.BytesIO()
                with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
                    view_res.amounts.to_excel(writer, sheet_name="THRUVIA_Summas")
                    view_res.kpi.to_excel(writer, sheet_name="KPI")
                st.download_button(
                    "Lejupielādēt (.xlsx)",
                    xl_buf.getvalue(),
                    "THRUVIA_Saglabata_Atskaite.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )

    # --- 14/15 MĒNEŠU SLIEKSNIS: LĪDZ 14 MĒN. VIENĀ LAPĀ; 15+ MĒN. -> PĀRŠĶIRŠANA VAI RITINĀŠANA AR FIKSĒTĀM MALĀM ---
    sel_by_year: dict[int, list[str]] = {}
    for m_col in fin_selected_months:
        _, yr = parse_period_month_year(m_col)
        if yr:
            sel_by_year.setdefault(yr, []).append(m_col)
    sel_years_sorted = sorted(sel_by_year.keys())

    table_res_to_render = view_res
    table_title_suffix = f"{len(fin_selected_months)} mēn."
    show_cum_table = True
    use_freeze_sides = False

    if len(fin_selected_months) >= 15:
        has_multi_years = len(sel_years_sorted) > 1
        view_opts = (
            ["Pa gadiem (pāršķirams)", "Visi mēneši (ritināms ar fiksētām malām)", "Gadu salīdzinājums"]
            if has_multi_years
            else ["Pa periodiem (12 mēn. lapā)", "Visi mēneši (ritināms ar fiksētām malām)"]
        )

        with st.container(border=True):
            my_c1, my_c2 = st.columns([2.1, 2.2], vertical_alignment="center")
            with my_c1:
                multi_year_view = st.radio(
                    f"Tabulas attēlošanas veids ({len(fin_selected_months)} mēneši)",
                    options=view_opts,
                    horizontal=True,
                    key="ov_multi_year_view",
                )
            with my_c2:
                if multi_year_view == "Pa gadiem (pāršķirams)" and has_multi_years:
                    if "ov_page_year" not in S or S.ov_page_year not in sel_years_sorted:
                        S.ov_page_year = sel_years_sorted[-1]
                    cur_yr_idx = sel_years_sorted.index(S.ov_page_year)

                    pg1, pg2, pg3, pg4 = st.columns([0.85, 1.1, 0.85, 1.5], vertical_alignment="center")
                    with pg1:
                        if st.button("◀ Iepriekšējais", disabled=(cur_yr_idx == 0), use_container_width=True, key="btn_yr_prev"):
                            S.ov_page_year = sel_years_sorted[cur_yr_idx - 1]
                            st.rerun()
                    with pg2:
                        chosen_yr = st.selectbox(
                            "Gads",
                            options=sel_years_sorted,
                            index=cur_yr_idx,
                            format_func=lambda y: f"{y}. gads ({len(sel_by_year[y])} mēn.)",
                            label_visibility="collapsed",
                            key="sel_ov_page_yr_box",
                        )
                        if chosen_yr != S.ov_page_year:
                            S.ov_page_year = chosen_yr
                            st.rerun()
                    with pg3:
                        if st.button("Nākamais ▶", disabled=(cur_yr_idx == len(sel_years_sorted) - 1), use_container_width=True, key="btn_yr_next"):
                            S.ov_page_year = sel_years_sorted[cur_yr_idx + 1]
                            st.rerun()
                    with pg4:
                        keep_all_totals = st.checkbox(
                            f"KOPĀ / AVG par visu periodu ({len(fin_selected_months)}m)",
                            value=True,
                            key="ov_keep_all_totals_chk",
                        )

                    yr_months_show = sel_by_year[S.ov_page_year]
                    if keep_all_totals:
                        table_res_to_render = filter_result_by_months(
                            res,
                            yr_months_show,
                            all_months,
                            summary_source_months=fin_selected_months,
                            summary_label_prefix=f"Visi ({len(fin_selected_months)}m)",
                        )
                    else:
                        table_res_to_render = filter_result_by_months(
                            res,
                            yr_months_show,
                            all_months,
                            summary_source_months=yr_months_show,
                            summary_label_prefix=str(S.ov_page_year),
                        )
                    table_title_suffix = f"{S.ov_page_year}. gads ({len(yr_months_show)} mēn.) no {len(fin_selected_months)} mēn."

                elif multi_year_view == "Pa periodiem (12 mēn. lapā)":
                    chunks = [fin_selected_months[i : i + 12] for i in range(0, len(fin_selected_months), 12)]
                    if "ov_chunk_idx" not in S or S.ov_chunk_idx >= len(chunks):
                        S.ov_chunk_idx = len(chunks) - 1
                    c_idx = S.ov_chunk_idx

                    pg1, pg2, pg3, pg4 = st.columns([0.85, 1.1, 0.85, 1.5], vertical_alignment="center")
                    with pg1:
                        if st.button("◀ Iepriekšējais", disabled=(c_idx == 0), use_container_width=True, key="btn_chk_prev"):
                            S.ov_chunk_idx = c_idx - 1
                            st.rerun()
                    with pg2:
                        st.markdown(f"<div style='text-align:center;font-weight:600;padding-top:6px;'>Lapa {c_idx + 1} / {len(chunks)}</div>", unsafe_allow_html=True)
                    with pg3:
                        if st.button("Nākamais ▶", disabled=(c_idx == len(chunks) - 1), use_container_width=True, key="btn_chk_next"):
                            S.ov_chunk_idx = c_idx + 1
                            st.rerun()
                    with pg4:
                        keep_all_totals = st.checkbox(
                            f"KOPĀ / AVG par visu periodu ({len(fin_selected_months)}m)",
                            value=True,
                            key="ov_keep_all_chk2",
                        )

                    ch_months = chunks[c_idx]
                    table_res_to_render = filter_result_by_months(
                        res,
                        ch_months,
                        all_months,
                        summary_source_months=fin_selected_months if keep_all_totals else ch_months,
                        summary_label_prefix=f"Visi ({len(fin_selected_months)}m)" if keep_all_totals else f"Lapa {c_idx + 1}",
                    )
                    table_title_suffix = f"{ch_months[0]}–{ch_months[-1]} (no {len(fin_selected_months)} mēn.)"

                elif multi_year_view == "Gadu salīdzinājums":
                    st.caption("Katra kolonna parāda attiecīgā gada kopsummu — ērts pārskats par vairākiem gadiem bez ritināšanas.")
                    table_res_to_render = aggregate_result_by_years(res, fin_selected_months)
                    table_title_suffix = f"Gadu salīdzinājums ({sel_years_sorted[0]}–{sel_years_sorted[-1]})"
                    show_cum_table = False

                else:
                    st.caption("Ritiniet mēnešus horizontāli — **Rādītāju nosaukumi** (pa kreisi) un **Perioda KOPĀ / AVG** (pa labi) paliek fiksēti!")
                    use_freeze_sides = True

    tbl_highlight = tut_active_focus if tut_active_focus in ("main", "kpi", "cum") else None

    st.markdown(
        render_unified_toc_tables_html(
            table_res_to_render.amounts,
            table_res_to_render.main,
            table_res_to_render.kpi,
            table_res_to_render.row_kinds,
            title_main=f"Finanšu pārskata tabula — {active_rep_label} ({table_title_suffix})",
            title_kpi="Produktivitātes un atdeves rādītāji",
            fy_start_month=S.get("fy_start_month", 1),
            show_cumulative=show_cum_table,
            freeze_sides=use_freeze_sides,
            highlight_section=tbl_highlight,
        ),
        unsafe_allow_html=True,
    )

    render_overview_charts(
        view_res.amounts,
        view_res.kpi,
        fy_start_month=S.get("fy_start_month", 1),
        highlight_charts=(tut_active_focus == "charts"),
    )

    if getattr(res, "reconciliation", None) is not None and not res.reconciliation.empty:
        with st.expander("Kontrole: Salīdzinājums ar PZA"):
            st.dataframe(res.reconciliation.style.format(fmt_num), use_container_width=True)

    st.markdown("---")
    rec_c1, rec_c2 = st.columns([3, 1], vertical_alignment="center")
    with rec_c1:
        st.subheader(f"TOC eksperta secinājumi un ieteikumi ({len(fin_selected_months)} mēn. periodam)")
    with rec_c2:
        regen_btn = st.button("Pieprasīt eksperta ieteikumus", type="primary", use_container_width=True)

    cum_df_md, _, _ = compute_cumulative_toc_df(view_res.amounts, fy_start_month=S.get("fy_start_month", 1))
    tables_md = (
        f"Izvēlētais periods: {', '.join(fin_selected_months)}\n\n"
        + view_res.amounts.round(0).to_markdown()
        + "\n\n"
        + view_res.kpi.round(3).to_markdown()
        + "\n\nUzkrājošie rādītāji:\n"
        + cum_df_md.round(0).to_markdown()
    )

    knowledge_context = f"Papildu THRUVIA / TOC zināšanu bāze:\n{knowledge_file}\n\n" if knowledge_file else ""
    COMMENT_SYSTEM = (
        "Tu esi laipns, tiešs un pieredzējis THRUVIA (TOC Caurlaidības grāmatvedības / Throughput Accounting) finanšu eksperts.\n"
        f"{TOC_CORE_EXPERT_KNOWLEDGE}\n"
        f"{knowledge_context}"
        "SVARĪGI NOTEIKUMI:\n"
        "1) Nesauc tekstu par 'Vadības kopsavilkumu'. Raksti kā TOC eksperts, kurš vadības komandai vienkāršā, saprotamā valodā izstāsta, ko rāda šie skaitļi.\n"
        "2) NEKAD neraksti par 'mārketinga efektivitāti' vai reklāmas atdevi, jo to no PZA nomērīt ir ļoti grūti un neobjektīvi.\n"
        "3) Fokusējies uz trim lietām:\n"
        "   • **Ko rāda Caurlaidības (T un T%), OE un NP skaitļi**: kā uzvedas T pret OE pa mēnešiem (un garākā periodā / rotējošajos 12 mēnešos), kurā brīdī uzkrājošais T pārsniedz OE (lūzuma punkts) un kāpēc atsevišķos mēnešos NP ir pozitīvs vai negatīvs.\n"
        "   • **Ko nozīmē šādi attiecību un produktivitātes rādītāji (RATIO: `PR = T/OE`, `T% = T/S`, `Eiro Sales / Eiro algas`)**: kurās zonās tie atrodas, kāda ir tendence un ko tas nozīmē uzņēmuma kapacitātes atdevei.\n"
        "   • **Kam pievērst uzmanību un ko modelēt tālāk**: kurus OE vai S/TVC lēmumus (pēc ΔT - ΔOE > 0 loģikas, bez pašizmaksas alokēšanas) vērts tūlīt pārbaudīt sadaļā Scenāriju simulācija."
    )

    if regen_btn:
        with st.spinner("THRUVIA eksperts analizē perioda rādītājus..."):
            try:
                r = get_client().messages.create(
                    model=MODEL,
                    max_tokens=2500,
                    system=COMMENT_SYSTEM,
                    messages=[{"role": "user", "content": f"Tabulas:\n\n{tables_md}\n\nUzraksti TOC eksperta secinājumus un ieteikumus šim periodam."}],
                )
                S.commentary = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                u_c = getattr(r, "usage", None)
                tok_c = (getattr(u_c, "input_tokens", 800) + getattr(u_c, "output_tokens", 800)) if u_c else 1600
                profile.log_api_usage(tok_c, tok_c * 0.000009)
                if S.get("current_report_id") and not is_demo_mode:
                    packed_c = pack_commentary_with_account_details(S.commentary, S.get("loaded_account_details"))
                    profile.update_report_commentary(S.current_report_id, packed_c)
                    st.toast("TOC eksperta ieteikumi saglabāti pie atskaites!")
                st.rerun()
            except Exception as e:
                st.error(f"Neizdevās uzģenerēt ieteikumus: {e}")

    if S.get("commentary"):
        with st.container(border=True):
            st.markdown(S.commentary)
            st.caption(LEGAL_DISCLAIMER_SHORT)
            if not is_demo_mode:
                with st.expander("Rediģēt vai papildināt TOC eksperta ieteikumus un saglabāt", expanded=False):
                    edited_comm = st.text_area("Ieteikumu teksts", value=S.commentary, height=240)
                    if st.button("Saglabāt ieteikumus atskaitē", type="primary"):
                        S.commentary = edited_comm
                        if S.get("current_report_id"):
                            packed_c = pack_commentary_with_account_details(edited_comm, S.get("loaded_account_details"))
                            profile.update_report_commentary(S.current_report_id, packed_c)
                            if S.get("loaded_report_meta"):
                                S.loaded_report_meta["commentary"] = packed_c
                            st.success("Ieteikumi saglabāti datubāzē!")
                        else:
                            st.toast("Ieteikumi atjaunināti sesijā.")
                        st.rerun()
    else:
        st.caption("Nospiediet pogu **Pieprasīt eksperta ieteikumus**, lai saņemtu TOC diagnostiku par izvēlēto periodu.")


# ==============================================================================
# POSMS 2: SCENĀRIJU SIMULĀCIJA
# ==============================================================================
elif S.nav_page == "simulation":
    if tut_active_focus == "sim":
        st.markdown('<div class="tut-section-tag">👇 7. SOLIS — SCENĀRIJU MODELĒŠANA UN JAUNA PROJEKTA PIEVIENOŠANA</div>', unsafe_allow_html=True)

    st.subheader("Scenāriju simulācija un Jauns projekts")
    if "result" not in S or S.result is None:
        st.info("Vispirms ielādējiet vai izveidojiet finanšu atskaiti, lai veiktu simulāciju.")
        render_app_footer_disclaimer()
        st.stop()

    res = apply_custom_row_order_to_result(strip_excluded_rows(S.result))
    all_months = [c for c in res.amounts.columns if not ("YTD" in c or "AVG" in c or "KOPĀ" in c or "TOTAL" in c)]
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

    sim_preset_opts, _ = ensure_shared_working_period(all_months, months_by_year)
    if "sim_situation_text" not in S:
        S.sim_situation_text = ""

    def _on_sim_working_preset_change():
        p = S.working_period_preset
        if p.startswith("Visi mēneši"):
            S.working_months_active = list(all_months)
        elif p == "Pēdējie 12 mēneši":
            S.working_months_active = list(all_months[-12:])
        elif p == "Pirmie 6 mēneši":
            S.working_months_active = list(all_months[:6])
        elif p == "Pēdējie 6 mēneši":
            S.working_months_active = list(all_months[-6:])
        else:
            for yr_k, m_list in months_by_year.items():
                if p.startswith(f"{yr_k}. gads"):
                    S.working_months_active = list(m_list)
                    break

    def _on_sim_working_months_change():
        cur = S.working_months_active
        if cur == list(all_months):
            S.working_period_preset = sim_preset_opts[0]
        else:
            S.working_period_preset = "Manuāla mēnešu izlase"

    p_col1, p_col2, p_col3, _ = st.columns([1.4, 1.2, 1.2, 2.2], vertical_alignment="bottom")
    with p_col1:
        st.selectbox(
            "Aktīvais darba periods (vienots ar Pārskatu)",
            options=sim_preset_opts,
            key="working_period_preset",
            on_change=_on_sim_working_preset_change,
        )

    cur_sim_months = S.get("working_months_active", all_months)
    with p_col2:
        with st.popover(f"Mēnešu izlase ({len(cur_sim_months)}/{len(all_months)})", use_container_width=True):
            st.multiselect(
                "Mēneši simulācijā",
                options=all_months,
                key="working_months_active",
                on_change=_on_sim_working_months_change,
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

    selected_months = [m for m in all_months if m in S.get("working_months_active", all_months)]
    if not selected_months:
        st.warning("Izvēlieties vismaz vienu mēnesi.")
        st.stop()

    base_sub = res.amounts[selected_months].copy().astype(float)
    base_sum = base_sub.sum(axis=1)

    with st.expander("Aprakstīt vai ierunāt situāciju (MI automātiski saliks simulācijas skaitļus)", expanded=True):
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
            st.text_area(
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
                "Pozīcija": st.column_config.TextColumn("Pozīcija", width="large"),
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
        if len(s_children) > 1:
            base_sum.loc[old_s_subtotal_lbl] = old_s_base
            sim_sum.loc[old_s_subtotal_lbl] = old_s_sim
            sim_row_kinds[old_s_subtotal_lbl] = "S_SUBTOTAL"
            s_display_rows.append(old_s_subtotal_lbl)
            for sc in s_children:
                sim_row_kinds[sc] = "S_DETAIL"
                s_display_rows.append(sc)
        elif len(s_children) == 1:
            sim_row_kinds[s_children[0]] = "S_SUBTOTAL"
            s_display_rows.append(s_children[0])

        base_sum.loc[np_s_row_lbl] = 0.0
        sim_sum.loc[np_s_row_lbl] = np_s_val
        sim_row_kinds[np_s_row_lbl] = "S_NEW_PROD"
        s_display_rows.append(np_s_row_lbl)

        if len(tvc_children) > 1:
            base_sum.loc[old_tvc_subtotal_lbl] = old_tvc_base
            sim_sum.loc[old_tvc_subtotal_lbl] = old_tvc_sim
            sim_row_kinds[old_tvc_subtotal_lbl] = "TVC_SUBTOTAL"
            tvc_display_rows.append(old_tvc_subtotal_lbl)
            for tc in tvc_children:
                sim_row_kinds[tc] = "TVC_DETAIL"
                tvc_display_rows.append(tc)
        elif len(tvc_children) == 1:
            sim_row_kinds[tvc_children[0]] = "TVC_SUBTOTAL"
            tvc_display_rows.append(tvc_children[0])

        base_sum.loc[np_tvc_row_lbl] = 0.0
        sim_sum.loc[np_tvc_row_lbl] = np_tvc_total
        sim_row_kinds[np_tvc_row_lbl] = "TVC_NEW_PROD"
        tvc_display_rows.append(np_tvc_row_lbl)

        base_sum.loc[old_t_row_lbl] = old_t_base
        sim_sum.loc[old_t_row_lbl] = old_t_sim
        sim_row_kinds[old_t_row_lbl] = "T_SUBTOTAL"
        t_display_rows.append(old_t_row_lbl)

        base_sum.loc[np_t_row_lbl] = 0.0
        sim_sum.loc[np_t_row_lbl] = np_t_total
        sim_row_kinds[np_t_row_lbl] = "T_NEW_PROD"
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
    sim_kpi_df = pd.DataFrame(
        {
            "Bāze KOPĀ": _calc_kpi_for_series(summary_amounts["Bāze KOPĀ"]),
            "Simulācija KOPĀ": _calc_kpi_for_series(summary_amounts["Simulācija KOPĀ"]),
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
            title_main=f"Simulētā tabula — {active_rep_label} ({len(selected_months)} mēn.)",
            title_kpi="Simulētie produktivitātes rādītāji",
            grey_cols={"Bāze KOPĀ"},
            muted_cols={"Δ Izmaiņa (€)"},
            fy_start_month=S.get("fy_start_month", 1),
            show_cumulative=False,
            lbl_width_px=360,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if is_demo_mode:
        st.caption("Demo režīmā simulāciju var brīvi testēt; simulāciju saglabāšana arhīvā būs pieejama pēc jūsu uzņēmuma PZA aktivizēšanas.")
    else:
        sv_c1, sv_c2 = st.columns([2, 1], vertical_alignment="bottom")
        with sv_c1:
            sim_save_name = st.text_input(
                "Simulācijas nosaukums saglabāšanai",
                value=f"Simulācija ({now_riga().strftime('%d.%m.%Y %H:%M')})",
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
    if tut_active_focus == "accounts":
        st.markdown('<div class="tut-section-tag">👇 6. SOLIS — APAKŠGRUPU UN KONTU PĀRVALDĪBA</div>', unsafe_allow_html=True)

    top_exp_c1, top_exp_c2 = st.columns([2.8, 1.2], vertical_alignment="center")
    with top_exp_c1:
        st.subheader("Kontu struktūra un Apakšgrupu vadība")
        st.caption(
            f"Aktīvā atskaite: **{active_rep_label}**"
            + (" *(Demo parauga dati — izmaiņas ir īslaicīgas izmēģinājumam)*" if is_demo_mode else " — visas izmaiņas varat saglabāt kā jaunu versiju vai atjaunināt esošo.")
        )
    with top_exp_c2:
        if S.get("has_unsaved_changes") and not is_demo_mode:
            with st.popover("💾 Saglabāt veiktās izmaiņas", use_container_width=True):
                st.markdown("**Saglabāt kontu un apakšgrupu izmaiņas**")
                ver_nm_exp = st.text_input("Nosaukums jaunai versijai", value=f"{active_rep_label} (rediģēts)", key="exp_save_ver_nm")
                if st.button("➕ Saglabāt kā JAUNU versiju (vecā paliek)", type="primary", use_container_width=True):
                    ok_sv, _ = save_report_as_new_or_update(profile, S.result, ver_nm_exp)
                    if ok_sv:
                        st.toast(f"Saglabāta jauna versija '{ver_nm_exp}'!")
                        st.rerun()
                if S.get("current_report_id") and st.button("🔄 Pārrakstīt esošo atskaiti", use_container_width=True):
                    ok_up, _ = save_report_as_new_or_update(profile, S.result, active_rep_label)
                    if ok_up:
                        st.toast("Esošā atskaite atjaunināta!")
                        st.rerun()

    has_live_pza = ("cls_df" in S and S.cls_df is not None and not S.cls_df.empty and "analysis" in S and S.analysis is not None)
    snap_details: list[dict] = S.get("loaded_account_details") or []
    saved_maps = [] if is_demo_mode else profile.list_saved_mappings()
    res_for_exp = S.get("result")
    res_month_cols = (
        [c for c in res_for_exp.amounts.columns if not ("YTD" in str(c).upper() or "AVG" in str(c).upper() or "KOPĀ" in str(c).upper())]
        if res_for_exp is not None
        else []
    )

    if not has_live_pza and not snap_details and not saved_maps:
        st.info("Šajā profilā vēl nav saglabātu kontu. Augšupielādējiet PZA failu sadaļā **+ Jauns PZA**.")
        render_app_footer_disclaimer()
        st.stop()

    if has_live_pza:
        an = S.analysis
        all_acc_items = build_account_details_snapshot(an.leaves, S.cls_df, an.ex.period_cols)
    elif snap_details:
        all_acc_items = snap_details
    else:
        all_acc_items = [
            {
                "id": str(m["id"]),
                "row": "—",
                "label": m.get("account_label", ""),
                "display_label": m.get("account_label", ""),
                "path": m.get("path_label", ""),
                "group": normalize_group_key(m.get("toc_group", "OE_OTHER")),
                "months": {},
                "total": 0.0,
            }
            for m in saved_maps
        ]

    existing_sub_labels = [lbl for k, lbl in SUBGROUP_LABELS.items() if k != "EXCLUDED"]
    mc_badge_icon = {"S": "🟡 [S]", "TVC": "🟠 [TVC]", "OE": "🍑 [OE]", "EXCL": "⚪ [Neiekļ.]"}

    def _format_account_option_clean(it: dict) -> str:
        gk = normalize_group_key(it.get("group", "OE_OTHER"))
        mc = get_main_code(gk)
        sub_l = SUBGROUP_LABELS.get(gk, gk)
        badge = mc_badge_icon.get(mc, f"[{mc}]")
        tot_s = fmt_num(abs(float(it.get("total", 0.0))))
        return f"{badge} {sub_l}  ▸  {it['label']}  ({tot_s} €)"

    with st.container(border=True):
        st.markdown("##### ⚙️ Ātrā apakšgrupu un kontu vadība")
        tab_move, tab_merge, tab_new, tab_rename = st.tabs(
            [
                "1. Pārvietot kontus (izņemt un pielikt citai)",
                "2. Apvienot apakšgrupas (pievienot pie citas)",
                "3. Izveidot jaunu apakšgrupu",
                "4. Pārdēvēt apakšgrupu",
            ]
        )

        with tab_move:
            st.caption("Vispirms varat izvēlēties apakšgrupu, no kuras vēlaties izņemt kontu, lai sarakstā rādītos tikai tās konti:")
            mv_f1, mv_f2, mv_f3, mv_f4 = st.columns([1.3, 2.0, 1.4, 0.85], vertical_alignment="bottom")
            with mv_f1:
                used_sub_names = sorted(
                    {SUBGROUP_LABELS.get(normalize_group_key(it["group"]), it["group"]) for it in all_acc_items},
                    key=lambda x: GROUP_SORT_ORDER.get(LABEL_TO_KEY.get(x, "OE_OTHER"), 50),
                )
                filter_src_sub = st.selectbox(
                    "1. No kuras grupas izņemt",
                    options=["Visas apakšgrupas"] + used_sub_names,
                    key="mgr_filter_src_sub",
                )

            filtered_items_for_move = [
                it for it in all_acc_items
                if filter_src_sub == "Visas apakšgrupas"
                or SUBGROUP_LABELS.get(normalize_group_key(it["group"]), it["group"]) == filter_src_sub
            ]
            filtered_acc_map = {_format_account_option_clean(it): it["label"] for it in filtered_items_for_move}

            with mv_f2:
                sel_acc_nice = st.multiselect(
                    f"2. Konti ({len(filtered_acc_map)} pieejami)",
                    options=list(filtered_acc_map.keys()),
                    placeholder="Izvēlieties vienu vai vairākus kontus...",
                    key="mgr_mv_accs",
                )
            with mv_f3:
                dest_sub_sel = st.selectbox(
                    "3. Uz kuru apakšgrupu pārlikt",
                    options=existing_sub_labels + [EXCLUDED_LABEL],
                    key="mgr_mv_dest",
                )
            with mv_f4:
                if st.button("Pārvietot", type="primary", use_container_width=True, key="mgr_btn_mv"):
                    if not sel_acc_nice:
                        st.warning("Izvēlieties vismaz vienu kontu.")
                    else:
                        raw_names = [filtered_acc_map[k] for k in sel_acc_nice]
                        dest_key = LABEL_TO_KEY.get(dest_sub_sel, "OE_OTHER")
                        dest_mc = get_main_code(dest_key)
                        ok_m, msg_m = move_specific_accounts_to_subgroup(raw_names, dest_sub_sel, dest_mc, profile)
                        if ok_m:
                            st.toast(msg_m)
                            st.rerun()

        with tab_merge:
            st.caption("Visi konti un summas no pirmās apakšgrupas tiks pievienoti otrajai apakšgrupai:")
            mg1, mg2, mg3 = st.columns([1.6, 1.6, 1.0], vertical_alignment="bottom")
            with mg1:
                src_sub_merge = st.selectbox("Apakšgrupa, kuru izņemt / pievienot", options=existing_sub_labels, key="mgr_mrg_src")
            with mg2:
                tgt_opts_merge = [x for x in existing_sub_labels if x != src_sub_merge] or existing_sub_labels
                tgt_sub_merge = st.selectbox("Apakšgrupa, kurai pievienot klāt", options=tgt_opts_merge, key="mgr_mrg_tgt")
            with mg3:
                if st.button("Apvienot grupas", type="primary", use_container_width=True, key="mgr_btn_mrg"):
                    ok_mg, msg_mg = merge_subgroups_everywhere(src_sub_merge, tgt_sub_merge, profile)
                    if ok_mg:
                        st.toast(msg_mg)
                        st.rerun()
                    else:
                        st.error(msg_mg)

        with tab_new:
            st.caption("Izveidojiet jaunu apakšgrupu un pēc izvēles uzreiz pārceliet uz to kontus:")
            all_acc_map_clean = {_format_account_option_clean(it): it["label"] for it in all_acc_items}
            nw1, nw2, nw3, nw4 = st.columns([1.4, 1.2, 1.6, 0.9], vertical_alignment="bottom")
            with nw1:
                new_sub_name = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Personāla pasākumi", key="mgr_new_lbl")
            with nw2:
                new_sub_mc_lbl = st.selectbox(
                    "Lielā grupa",
                    options=["Uzturēšanas izmaksas (OE)", "TVC (Mainīgās izmaksas)", "S (Ieņēmumi)"],
                    key="mgr_new_mc",
                )
            with nw3:
                new_sub_accs = st.multiselect(
                    "Konti, ko uzreiz ielikt (neobligāti)",
                    options=list(all_acc_map_clean.keys()),
                    placeholder="Var izvēlēties arī vēlāk...",
                    key="mgr_new_accs",
                )
            with nw4:
                if st.button("Izveidot grupu", type="primary", use_container_width=True, key="mgr_btn_new"):
                    if not new_sub_name.strip():
                        st.warning("Ievadiet jaunās apakšgrupas nosaukumu.")
                    else:
                        mc_code = "S" if new_sub_mc_lbl.startswith("S") else ("TVC" if new_sub_mc_lbl.startswith("TVC") else "OE")
                        raw_accs_for_new = [all_acc_map_clean[k] for k in new_sub_accs]
                        ok_nw, msg_nw = move_specific_accounts_to_subgroup(raw_accs_for_new, new_sub_name.strip(), mc_code, profile)
                        if ok_nw:
                            st.toast(msg_nw)
                            st.rerun()

        with tab_rename:
            st.caption("Nomainiet jebkuras esošās apakšgrupas nosaukumu:")
            rn1, rn2, rn3 = st.columns([1.6, 1.6, 1.0], vertical_alignment="bottom")
            with rn1:
                old_sub_sel = st.selectbox("Esošā apakšgrupa", options=existing_sub_labels, key="exp_rn_old")
            with rn2:
                new_sub_inp = st.text_input("Jaunais nosaukums", placeholder="piem., Reklāmas izdevumi", key="exp_rn_new")
            with rn3:
                if st.button("Pārdēvēt grupu", type="primary", use_container_width=True, key="exp_btn_rn_sub"):
                    if new_sub_inp.strip():
                        ok_rn, msg_rn = rename_subgroup_everywhere(old_sub_sel, new_sub_inp.strip(), profile)
                        if ok_rn:
                            st.toast(msg_rn)
                            st.rerun()
                        else:
                            st.error(msg_rn)

    main_sections = [
        ("S", "S — Ieņēmumi (Sales)", "#FFF200", "#000000"),
        ("TVC", "TVC — Patiesi mainīgās izmaksas", "#C55A11", "#FFFFFF"),
        ("OE", "Uzturēšanas izmaksas (OE)", "#F4B183", "#000000"),
        ("EXCL", EXCLUDED_LABEL, "#E2E8F0", "#0F172A"),
    ]

    for m_code, m_title, bg_hex, fg_hex in main_sections:
        grp_keys_for_main = [k for k, (_, mc) in GROUPS.items() if mc == m_code]
        main_items = [it for it in all_acc_items if normalize_group_key(it.get("group", "")) in grp_keys_for_main]
        main_tot = sum((float(it.get("total", 0.0)) if m_code == "S" else abs(float(it.get("total", 0.0)))) for it in main_items)

        st.markdown(
            f'<div style="background-color:{bg_hex}; color:{fg_hex}; padding:8px 14px; border-radius:8px; '
            f'font-weight:700; font-size:15px; margin-top:16px; margin-bottom:8px; border:1.5px solid #1E293B; '
            f'display:flex; justify-content:space-between;">'
            f'<span>{m_title} ({fmt_acc_count_lv(len(main_items))})</span>'
            f'<span>Kopā periodā: {fmt_num(main_tot)} €</span></div>',
            unsafe_allow_html=True,
        )

        if not main_items:
            st.caption("Šajā grupā nav neviena konta.")
            continue

        if m_code == "OE":
            grp_keys_for_main.sort(key=lambda gk: (0 if gk in ("OE_ADMIN_WAGES", "OE_WORKER_WAGES") else 1, GROUP_SORT_ORDER.get(gk, 50)))

        for g_key in grp_keys_for_main:
            g_items = [it for it in main_items if normalize_group_key(it.get("group", "")) == g_key]
            if not g_items:
                continue
            sub_lbl = SUBGROUP_LABELS.get(g_key, g_key)
            sub_tot = sum((float(it.get("total", 0.0)) if m_code == "S" else abs(float(it.get("total", 0.0)))) for it in g_items)

            with st.expander(f"📂 {sub_lbl}  —  {fmt_acc_count_lv(len(g_items))}  |  Kopā: {fmt_num(sub_tot)} €", expanded=False):
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
                        inf_c1, inf_c2 = st.columns([2.0, 1.5], vertical_alignment="bottom")
                        with inf_c1:
                            st.markdown(
                                f"**Oriģinālais nosaukums PZA:** `{orig_name}`  \n"
                                f"**Oriģinālā vieta / sadaļa PZA:** `{orig_path}` (rinda **#{orig_row_num}**)  \n"
                                f"**Pašreizējā apakšgrupa:** **{sub_lbl}**"
                            )
                        with inf_c2:
                            all_sub_opts = existing_sub_labels + [EXCLUDED_LABEL]
                            if sub_lbl not in all_sub_opts:
                                all_sub_opts.append(sub_lbl)

                            new_sub_choice = st.selectbox(
                                "Pārlikt šo kontu uz citu apakšgrupu",
                                options=all_sub_opts,
                                index=all_sub_opts.index(sub_lbl) if sub_lbl in all_sub_opts else 0,
                                key=f"move_live_{acc_id}",
                            )
                            if new_sub_choice != sub_lbl and new_sub_choice in LABEL_TO_KEY:
                                if st.button("Pārvietot un pārrēķināt", key=f"btn_mv_{acc_id}", type="primary", use_container_width=True):
                                    dest_mc = get_main_code(LABEL_TO_KEY[new_sub_choice])
                                    ok_mv1, msg_mv1 = move_specific_accounts_to_subgroup([orig_name], new_sub_choice, dest_mc, profile)
                                    if ok_mv1:
                                        st.toast(msg_mv1)
                                        st.rerun()

                        if m_dict_raw:
                            disp_m = {k: fmt_num(float(v)) for k, v in m_dict_raw.items()}
                            disp_m["KOPĀ"] = fmt_num(acc_tot)
                            st.markdown("**Konta nolasītie skaitļi pa mēnešiem PZA failā (€):**")
                            st.dataframe(pd.DataFrame([disp_m]), use_container_width=True, hide_index=True)


# ==============================================================================
# POSMS 4: ATSKAITES UN SIMULĀCIJAS (ARHĪVS)
# ==============================================================================
elif S.nav_page == "archive":
    st.subheader("Saglabātās atskaites un simulācijas")
    st.caption("Šeit varat pārskatīt, atvērt vai dzēst iepriekš saglabātās finanšu atskaites (pilnos periodus un saglabātās versijas) un scenāriju simulācijas.")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("#### Finanšu atskaites un saglabātās versijas")
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
                                S.result = apply_custom_row_order_to_result(strip_excluded_rows(loaded_res))
                                S.loaded_report_meta = loaded
                                S.current_report_id = loaded["id"]
                                S.has_unsaved_changes = False
                                S.pop("working_months_active", None)
                                S.pop("working_period_preset", None)
                                S.pop("ov_page_year", None)
                                S.pop("ov_chunk_idx", None)
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
                                S.result = apply_custom_row_order_to_result(strip_excluded_rows(loaded_sim_res))
                                S.loaded_report_meta = loaded_s
                                S.has_unsaved_changes = False
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
                                        S["working_months_active"] = list(sim_state["selected_months"])
                                        S["working_period_preset"] = "Manuāla mēnešu izlase"
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
# POSMS 5: JAUNS PZA / KONTU KLASIFIKĀCIJA (PRECĪZAIS KLASIFIKATORS + ADMIN TIEŠĀ PĀRBAUDE)
# ==============================================================================
elif S.nav_page == "ingest":
    st.subheader("Jauna PZA faila ielasīšana un kontu klasifikācija")

    ignore_memory = st.checkbox(
        "Ignorēt iepriekšējo profilu atmiņu un izvērtēt visus kontus pilnīgi no jauna (MI)",
        value=S.get("ignore_mem_flag", False),
        key="ignore_mem_checkbox",
    )
    S.ignore_mem_flag = ignore_memory

    uploaded = st.file_uploader("Augšupielādējiet PZA failu (CSV vai Excel)", type=["csv", "xlsx", "xlsm", "xls"])

    active_file_name = None
    active_file_bytes = None
    if uploaded is not None:
        active_file_name = uploaded.name
        active_file_bytes = uploaded.getvalue()
        S.pop("preloaded_pza_name", None)
        S.pop("preloaded_pza_bytes", None)
    elif S.get("preloaded_pza_bytes") and S.get("preloaded_pza_name"):
        active_file_name = S["preloaded_pza_name"]
        active_file_bytes = S["preloaded_pza_bytes"]
        st.info(f"Ielādēts iesūtītais klienta PZA fails pārbaudei: **{active_file_name}**")

    if active_file_bytes is None or active_file_name is None:
        st.info("Augšupielādējiet PZA failu, lai sāktu strukturēšanu un kontu klasifikāciju.")
        render_app_footer_disclaimer()
        st.stop()

    fid = hashlib.md5(active_file_bytes).hexdigest()
    if S.get("fid") != fid:
        S.fid = fid
        S.grids = load_grids(active_file_name, active_file_bytes)
        S.pop("structure", None)
        S.pop("analysis", None)
        S.pop("cls_df", None)
        S.pop("manual_leaf_overrides", None)
        S.pop("clarifications_done", None)

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
            with st.spinner("Klasificē pamatkontus ar precīzajiem TOC noteikumiem..."):
                try:
                    known, unknown = ({}, an.leaves) if ignore_memory else profile.lookup(an.leaves)
                    items = smart_classify_accounts(get_client(), MODEL, unknown) if unknown else []
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
        render_app_footer_disclaimer()
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
                cl_new_sub = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Personāla ilgtspējas pasākumi", key="clar_new_sub_in")
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
                st.markdown(f"**{disp_name}** *(MI sākotnējais ieteikums: `{agent_sub_label}`)*")
                st.caption(r["Jautājums"])

                c_m, c_s = st.columns([1.2, 1.8])
                with c_m:
                    main_opts_map = {
                        "OE": MAIN_GROUP_OPTIONS[2],
                        "TVC": MAIN_GROUP_OPTIONS[1],
                        "S": MAIN_GROUP_OPTIONS[0],
                        "EXCL": MAIN_GROUP_OPTIONS[3],
                    }
                    rev_main_map = {v: k for k, v in main_opts_map.items()}
                    m_list = list(main_opts_map.values())
                    sel_m_lbl = st.selectbox(
                        "1. Lielā grupa",
                        options=m_list,
                        index=m_list.index(main_opts_map.get(agent_main_code, MAIN_GROUP_OPTIONS[2])),
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
        render_app_footer_disclaimer()
        st.stop()

    st.markdown("#### Kontu sadalījums pa Lielajām grupām un Apakšgrupām")

    with st.expander("➕ Pievienot jaunu apakšgrupu (sadalīt izmaksas vai ieņēmumus sīkāk)", expanded=False):
        ac1, ac2, ac3 = st.columns([2, 1.5, 1], vertical_alignment="bottom")
        with ac1:
            new_sub_title = st.text_input("Jaunās apakšgrupas nosaukums", placeholder="piem., Personāla pasākumi")
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
    c1.metric("S (Ieņēmumi)", fmt_acc_count_lv(int(cls_df["Grupa"].str.startswith("S_").sum())))
    c2.metric("TVC (Mainīgās)", fmt_acc_count_lv(int(cls_df["Grupa"].str.startswith("TVC").sum())))
    c3.metric("Uzturēšanas izmaksas", fmt_acc_count_lv(int(cls_df["Grupa"].str.startswith("OE_").sum())))
    c4.metric("Neiekļautie", fmt_acc_count_lv(int((cls_df["Grupa"] == "EXCLUDED").sum())))

    cat_sections = [
        ("S", "S (Ieņēmumi)", MAIN_GROUP_OPTIONS[0]),
        ("TVC", "TVC (Mainīgās izmaksas)", MAIN_GROUP_OPTIONS[1]),
        ("OE", "Uzturēšanas izmaksas (OE)", MAIN_GROUP_OPTIONS[2]),
        ("EXCL", EXCLUDED_LABEL, MAIN_GROUP_OPTIONS[3]),
    ]

    changed_any = False
    new_groups_map: dict[str, str] = {}

    for m_code_sec, sec_title, m_cat_str in cat_sections:
        sec_mask = cls_df["Grupa"].map(get_main_code) == m_code_sec
        sec_df = cls_df[sec_mask].copy()
        if sec_df.empty:
            continue

        allowed_subs_sec = get_subgroups_for_main_code(m_code_sec)
        st.markdown(f"**{sec_title}** ({fmt_acc_count_lv(len(sec_df))})")

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
                target_mc = (
                    "S" if new_m_s.startswith("S")
                    else ("TVC" if new_m_s.startswith("TVC")
                          else ("EXCL" if (new_m_s.startswith("Izslēgt") or new_m_s.startswith("Neiekļaut")) else "OE"))
                )
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

    default_rep_name = f"{Path(active_file_name).stem} ({now_riga().strftime('%d.%m.%Y')})"
    is_admin_pending_review = bool(S.get("is_admin", False) and onboard_status in ("PENDING_WAIT", "PENDING_IMMEDIATE"))

    if is_admin_pending_review:
        col_rn, col_btn1, col_btn2 = st.columns([1.6, 1.15, 1.35], vertical_alignment="bottom")
        with col_rn:
            report_custom_name = st.text_input("Atskaites nosaukums", value=default_rep_name)
        with col_btn1:
            run_calc_draft = st.button("Saglabāt un pārbaudīt Pārskatā", use_container_width=True)
        with col_btn2:
            run_calc_approve = st.button("✅ Saglabāt, apstiprināt & sūtīt e-pastu", type="primary", use_container_width=True)
        run_calc = run_calc_draft or run_calc_approve
    else:
        col_rn, col_btn = st.columns([2, 1], vertical_alignment="bottom")
        with col_rn:
            report_custom_name = st.text_input("Atskaites nosaukums", value=default_rep_name)
        with col_btn:
            run_calc = st.button("Apstiprināt, saglabāt un atvērt Pārskatā", type="primary", use_container_width=True)
        run_calc_approve = False

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

        res = apply_custom_row_order_to_result(
            compute_toc_with_dynamic_subgroups(an.leaves, mapping, an.ex.period_cols, reported)
        )
        S.result = res
        S.has_unsaved_changes = False
        S.pop("working_months_active", None)
        S.pop("working_period_preset", None)
        S.pop("ov_page_year", None)
        S.pop("ov_chunk_idx", None)
        S.pop("preloaded_pza_name", None)
        S.pop("preloaded_pza_bytes", None)

        acc_snapshot = build_account_details_snapshot(an.leaves, cls_sorted, an.ex.period_cols)
        S.loaded_account_details = acc_snapshot

        if profile.is_connected:
            ok_map, _ = profile.save(an.leaves, mapping)
            packed_init_comm = pack_commentary_with_account_details("", acc_snapshot)
            ok_rep, rep_id = profile.save_report(
                report_name=report_custom_name.strip() or default_rep_name,
                file_name=active_file_name,
                res_obj=res,
                commentary=packed_init_comm,
            )
            profile.log_api_usage(0, 0.0)
            S.current_report_id = rep_id
            S.loaded_report_meta = {
                "id": rep_id,
                "report_name": report_custom_name.strip() or default_rep_name,
                "file_name": active_file_name,
                "periods_label": f"{len(an.ex.period_cols)} periodi",
            }
            if ok_map and ok_rep:
                st.toast("Atskaite un kontu mēnešu skaitļi saglabāti!")

        S.pop("commentary", None)
        if run_calc_approve:
            finalize_admin_pza_approval(S.active_profile, profile)
        else:
            S.nav_page = "overview"
            st.rerun()


# ==============================================================================
# POSMS 6: PAR THRUVIA (PRIEKŠVARDS, INSTRUKCIJA, BUJ UN JURIDISKIE NOTEIKUMI)
# ==============================================================================
elif S.nav_page == "about":
    render_about_and_help_page(is_public_view=False)


# ==============================================================================
# POSMS 7: UZSTĀDĪJUMI UN ABONEMENTS
# ==============================================================================
elif S.nav_page == "settings":
    st.subheader(f"Uzstādījumi un Abonements ({S.active_profile})")

    with st.container(border=True):
        st.markdown("#### 1. Profila informācija")
        pi1, pi2, pi3, pi4 = st.columns(4)
        pi1.metric("Uzņēmums / Profils", S.active_profile)
        pi2.metric("Saglabātie konti", saved_acc_count)
        pi3.metric("Finanšu atskaites", len(fin_reports))
        pi4.metric("Saglabātās simulācijas", len(sim_reports))

    with st.container(border=True):
        st.markdown("#### 2. Abonements un apmaksa")
        st.markdown(f"**Pašreizējais statuss:** `{sub_info['label']}`")
        st.markdown(f"**PZA pārbaudes statuss:** `{ONBOARD_LABELS_LV.get(onboard_status, onboard_status)}`")
        if sub_info["code"] == "TRIAL" and sub_info.get("first_charge_str"):
            st.info(
                f"Bezmaksas izmēģinājumā atlikušas: **{sub_info['time_left_str']}**.  \n"
                f"Pirmā automātiskā abonementa maksa (**{billing_cfg['monthly_price_eur']} €/mēn.**) tiks iekasēta: "
                f"**{sub_info['first_charge_str']}** (Latvijas laiks), ja abonements iepriekš netiks atcelts."
            )
        else:
            st.caption(
                f"THRUVIA mēneša abonements: **{billing_cfg['monthly_price_eur']} €/mēn.** (automātiska karšu apmaksa caur Stripe, "
                f"rēķini uz e-pastu un iespēja atcelt jebkurā brīdī)."
            )

        ab_c1, ab_c2, _ = st.columns([1.4, 1.4, 1.2])
        with ab_c1:
            if billing_cfg["stripe_checkout_url"]:
                chk_url = billing_cfg["stripe_checkout_url"]
                if S.get("company_email") and "prefilled_email" not in chk_url:
                    sep = "&" if "?" in chk_url else "?"
                    chk_url = f"{chk_url}{sep}prefilled_email={S['company_email']}"
                st.link_button("💳 Aktivizēt mēneša abonementu", chk_url, type="primary", use_container_width=True)
            else:
                st.button(
                    f"💳 Aktivizēt abonementu ({billing_cfg['monthly_price_eur']} €/mēn.)",
                    disabled=True,
                    use_container_width=True,
                    help=f"Karšu apmaksas saite tiks pieslēgta drīzumā. Sazinieties ar B4B: {billing_cfg['support_email']}",
                )
        with ab_c2:
            if billing_cfg["stripe_portal_url"]:
                st.link_button("Pārvaldīt vai atcelt abonementu", billing_cfg["stripe_portal_url"], use_container_width=True)
            else:
                st.caption(f"Abonementa izmaiņām vai rēķiniem rakstiet: **{billing_cfg['support_email']}**")

    with st.container(border=True):
        st.markdown("#### 3. Finanšu gada sākums")
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
        st.markdown("#### 4. Paroles maiņa")
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
                    cur_notes = str(cur_c.get("notes", ""))
                    cur_act = bool(cur_c.get("is_active", True))
                    ok_pw, msg_pw = admin_store.create_or_update_company_admin(
                        company_name=S.active_profile,
                        password=new_pw1.strip(),
                        monthly_limit=10000,
                        is_active=cur_act,
                        notes=cur_notes,
                        reset_usage=False,
                    )
                    if ok_pw:
                        st.success("Parole veiksmīgi nomainīta!")
                    else:
                        st.error(msg_pw)

    with st.expander("Juridiskie noteikumi, Privātuma politika (VDAR) un Atbildības ierobežojums"):
        render_legal_documents_tabs()


# ------------------------------------------------------------------ JURIDISKĀ KĀJENE VISĀS LAPĀS
render_app_footer_disclaimer()

# === BEIDZAS 3/3 DAĻA (FAILA BEIGAS) ===

"""
db.py — Supabase datubāzes integrācija (v4.0):
- Administratora (Master) klientu pārvaldība, paroles, limiti un lietojuma uzskaite
- Klientu datu stingra izolācija
- Tieša saglabāto kontu rediģēšana un atskaišu arhīvs
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
import hashlib
import json
import os
from io import StringIO
import pandas as pd
import streamlit as st
from supabase import create_client, Client

DEFAULT_SUPABASE_URL = "https://amprqueamdcxpjufigvm.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_f2rdvTMla_SXrU7bKAo4Jw_fkGCJWEN"


def _get_credentials() -> tuple[str, str]:
    url, key = "", ""
    try:
        if "supabase" in st.secrets:
            supa_cfg = st.secrets["supabase"]
            url = str(supa_cfg.get("url", "")).strip()
            key = str(supa_cfg.get("key", "")).strip()
    except Exception:
        pass

    if not url:
        url = str(os.environ.get("SUPABASE_URL", "")).strip() or DEFAULT_SUPABASE_URL
    if not key:
        key = str(os.environ.get("SUPABASE_KEY", "")).strip() or DEFAULT_SUPABASE_KEY

    return url.strip().strip('"').strip("'"), key.strip().strip('"').strip("'")


def get_admin_password() -> str:
    try:
        pw = str(st.secrets.get("ADMIN_PASSWORD", "")).strip()
        if pw:
            return pw
    except Exception:
        pass
    return os.environ.get("ADMIN_PASSWORD", "TocAdmin2026!")


@st.cache_resource
def get_supabase() -> tuple[Client | None, str | None]:
    url, key = _get_credentials()
    if not url or not key:
        return None, "Nav norādīts Supabase URL vai Key."
    try:
        client = create_client(url, key)
        return client, None
    except Exception as e:
        return None, str(e)


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _norm_part(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _norm_key(path: str, label: str) -> str:
    return f"{_norm_part(path)}||{_norm_part(label)}"


class SupabaseProfileStore:
    def __init__(self, company_name: str = ""):
        self.user_id = str(company_name).strip()
        self.client, self.init_error = get_supabase()
        self.last_error: str | None = self.init_error

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ AUTENTIFIKĀCIJA UN ADMIN PĀRVALDĪBA
    def authenticate(self, username: str, password: str) -> tuple[bool, str, bool]:
        """
        Atgriež (veiksmīgi, ziņojums, ir_admins).
        Ja username == 'admin', pārbauda administratora paroli.
        """
        u_clean = str(username or "").strip()
        if not u_clean or not password:
            return False, "Ievadiet lietotājvārdu un paroli.", False

        if u_clean.lower() == "admin":
            if password == get_admin_password():
                return True, "Administratora piekļuve apstiprināta.", True
            return False, "Nepareiza administratora parole.", False

        if not self.client:
            return False, "Nav datubāzes savienojuma.", False

        pw_hash = _hash_pw(password)
        try:
            res = (
                self.client.table("company_profiles")
                .select("*")
                .eq("company_name", u_clean)
                .limit(1)
                .execute()
            )
            rows = res.data or []
            if not rows:
                return False, "Lietotājs nav atrasts. Sazinieties ar sistēmas administratoru.", False

            row = rows[0]
            if row.get("is_active") is False:
                return False, "Šis profils ir deaktivizēts. Sazinieties ar administratoru.", False

            if row["password_hash"] == pw_hash:
                self.user_id = row["company_name"]
                return True, "Veiksmīga ielogošanās.", False
            return False, "Nepareiza parole.", False
        except Exception as e:
            return False, f"Kļūda autorizācijā: {e}", False

    def check_quota(self) -> tuple[bool, str]:
        """Pārbauda, vai klientam nav izsmelts mēneša aprēķinu limits."""
        if not self.client or not self.user_id:
            return True, ""
        try:
            res = (
                self.client.table("company_profiles")
                .select("is_active, monthly_limit, usage_count")
                .eq("company_name", self.user_id)
                .limit(1)
                .execute()
            )
            if not res.data:
                return True, ""
            r = res.data[0]
            if r.get("is_active") is False:
                return False, "Konts ir deaktivizēts."
            lim = int(r.get("monthly_limit") or 50)
            used = int(r.get("usage_count") or 0)
            if used >= lim:
                return False, f"Sasniegts profila aprēķinu limits ({used}/{lim}). Sazinieties ar administratoru."
            return True, f"{used}/{lim}"
        except Exception:
            return True, ""

    def increment_usage(self) -> None:
        if not self.client or not self.user_id:
            return
        try:
            res = (
                self.client.table("company_profiles")
                .select("id, usage_count")
                .eq("company_name", self.user_id)
                .limit(1)
                .execute()
            )
            if res.data:
                cur = int(res.data[0].get("usage_count") or 0)
                self.client.table("company_profiles").update({"usage_count": cur + 1}).eq("id", res.data[0]["id"]).execute()
        except Exception:
            pass

    def list_all_companies_admin(self) -> list[dict]:
        """Administratora saraksts ar visiem klientu profiliem, kontu skaitu un atskaišu skaitu."""
        if not self.client:
            return []
        try:
            res = self.client.table("company_profiles").select("*").order("created_at", desc=True).execute()
            companies = res.data or []

            # Saskaitām kontus un atskaites katram
            acc_res = self.client.table("account_mappings").select("user_id").execute()
            rep_res = self.client.table("saved_reports").select("profile_name").execute()

            acc_counts: dict[str, int] = {}
            for r in acc_res.data or []:
                u = r.get("user_id", "")
                acc_counts[u] = acc_counts.get(u, 0) + 1

            rep_counts: dict[str, int] = {}
            for r in rep_res.data or []:
                p = r.get("profile_name", "")
                rep_counts[p] = rep_counts.get(p, 0) + 1

            out = []
            for c in companies:
                cname = c["company_name"]
                out.append(
                    {
                        "id": c["id"],
                        "company_name": cname,
                        "is_active": bool(c.get("is_active", True)),
                        "monthly_limit": int(c.get("monthly_limit") or 50),
                        "usage_count": int(c.get("usage_count") or 0),
                        "notes": str(c.get("notes") or ""),
                        "accounts_count": acc_counts.get(cname, 0),
                        "reports_count": rep_counts.get(cname, 0),
                        "created_at": str(c.get("created_at", ""))[:10],
                    }
                )
            return out
        except Exception as e:
            self.last_error = str(e)
            return []

    def create_or_update_company_admin(
        self,
        company_name: str,
        password: str = "",
        monthly_limit: int = 50,
        is_active: bool = True,
        notes: str = "",
        reset_usage: bool = False,
    ) -> tuple[bool, str]:
        """Ļauj administratoram izveidot jaunu klientu vai atjaunināt paroli / limitus esošam."""
        if not self.client:
            return False, "Nav datubāzes savienojuma."
        c_name = company_name.strip()
        if not c_name:
            return False, "Norādiet uzņēmuma nosaukumu."
        if c_name.lower() == "admin":
            return False, "Rezervēts lietotājvārds."

        try:
            chk = self.client.table("company_profiles").select("*").eq("company_name", c_name).execute()
            if chk.data:
                upd: dict = {
                    "monthly_limit": int(monthly_limit),
                    "is_active": bool(is_active),
                    "notes": notes,
                }
                if password.strip():
                    if len(password.strip()) < 4:
                        return False, "Parolei jābūt vismaz 4 simbolus garai."
                    upd["password_hash"] = _hash_pw(password.strip())
                if reset_usage:
                    upd["usage_count"] = 0
                self.client.table("company_profiles").update(upd).eq("company_name", c_name).execute()
                return True, f"Klienta '{c_name}' iestatījumi atjaunināti."
            else:
                if len(password.strip()) < 4:
                    return False, "Jaunam profilam jānorāda parole (min. 4 simboli)."
                payload = {
                    "company_name": c_name,
                    "password_hash": _hash_pw(password.strip()),
                    "monthly_limit": int(monthly_limit),
                    "is_active": bool(is_active),
                    "usage_count": 0,
                    "notes": notes,
                }
                self.client.table("company_profiles").insert(payload).execute()
                return True, f"Klients '{c_name}' veiksmīgi izveidots."
        except Exception as e:
            return False, f"Datubāzes kļūda: {e}"

    def delete_company_admin(self, company_name: str) -> bool:
        if not self.client or not company_name:
            return False
        try:
            self.client.table("account_mappings").delete().eq("user_id", company_name).execute()
            self.client.table("saved_reports").delete().eq("profile_name", company_name).execute()
            self.client.table("company_profiles").delete().eq("company_name", company_name).execute()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    # ------------------------------------------------------------------ KONTU ATMIŅA UN REDIĢĒŠANA
    def count_saved(self) -> int:
        if not self.client or not self.user_id:
            return 0
        try:
            res = (
                self.client.table("account_mappings")
                .select("id", count="exact")
                .eq("user_id", self.user_id)
                .execute()
            )
            return int(res.count if res.count is not None else len(res.data or []))
        except Exception as e:
            self.last_error = str(e)
            return 0

    def list_saved_mappings(self) -> list[dict]:
        """Atgriež visus profila saglabātos kontus tiešai rediģēšanai."""
        if not self.client or not self.user_id:
            return []
        try:
            res = (
                self.client.table("account_mappings")
                .select("id, path_label, account_label, toc_group, is_income")
                .eq("user_id", self.user_id)
                .order("account_label")
                .execute()
            )
            return res.data or []
        except Exception as e:
            self.last_error = str(e)
            return []

    def update_saved_mappings_direct(self, rows: list[dict]) -> bool:
        """Saglabā rediģētos kontu kartējumus tieši Supabase tabulā."""
        if not self.client or not self.user_id:
            return False
        try:
            for r in rows:
                row_id = r.get("id")
                grp = str(r.get("toc_group", "OE_OTHER"))
                is_inc = grp.startswith("S_")
                if row_id:
                    self.client.table("account_mappings").update(
                        {"toc_group": grp, "is_income": is_inc}
                    ).eq("id", row_id).eq("user_id", self.user_id).execute()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def delete_saved_mapping(self, mapping_id: int) -> bool:
        if not self.client or not self.user_id:
            return False
        try:
            self.client.table("account_mappings").delete().eq("id", mapping_id).eq("user_id", self.user_id).execute()
            return True
        except Exception:
            return False

    def lookup(self, leaves) -> tuple[dict[str, dict], list]:
        if not self.client or not self.user_id:
            return {}, list(leaves)

        try:
            res = (
                self.client.table("account_mappings")
                .select("path_label, account_label, toc_group, is_income")
                .eq("user_id", self.user_id)
                .execute()
            )
            rows = res.data or []
            exact_db: dict[str, dict] = {}
            label_only_db: dict[str, list[dict]] = {}

            for row in rows:
                entry = {
                    "group": row["toc_group"],
                    "is_income": bool(row.get("is_income", False)),
                }
                full_k = _norm_key(row.get("path_label", ""), row.get("account_label", ""))
                exact_db[full_k] = entry
                lbl_k = _norm_part(row.get("account_label", ""))
                label_only_db.setdefault(lbl_k, []).append(entry)

            known: dict[str, dict] = {}
            unknown = []
            for lf in leaves:
                full_k = _norm_key(getattr(lf, "path", ""), getattr(lf, "label", ""))
                lbl_k = _norm_part(getattr(lf, "label", ""))
                if full_k in exact_db:
                    known[lf.id] = exact_db[full_k]
                elif lbl_k in label_only_db and len(label_only_db[lbl_k]) == 1:
                    known[lf.id] = label_only_db[lbl_k][0]
                else:
                    unknown.append(lf)

            return known, unknown
        except Exception as e:
            self.last_error = str(e)
            return {}, list(leaves)

    def save(self, leaves, mapping: dict[str, dict]) -> tuple[bool, int]:
        if not self.client or not self.user_id:
            return False, 0

        dedup: dict[tuple[str, str, str], dict] = {}
        for lf in leaves:
            if lf.id in mapping:
                m = mapping[lf.id]
                p_norm = _norm_part(getattr(lf, "path", ""))
                l_norm = _norm_part(getattr(lf, "label", ""))
                dedup[(self.user_id, p_norm, l_norm)] = {
                    "user_id": self.user_id,
                    "path_label": p_norm,
                    "account_label": l_norm,
                    "toc_group": str(m["group"]),
                    "is_income": bool(m["is_income"]),
                }

        rows_to_upsert = list(dedup.values())
        if not rows_to_upsert:
            return True, 0

        try:
            self.client.table("account_mappings").upsert(
                rows_to_upsert,
                on_conflict="user_id,path_label,account_label",
            ).execute()
            return True, len(rows_to_upsert)
        except Exception as e:
            self.last_error = str(e)
            return False, 0

    # ------------------------------------------------------------------ FINANŠU TABULU VĒSTURE
    def save_report(
        self,
        report_name: str,
        file_name: str,
        res_obj,
        commentary: str = "",
    ) -> tuple[bool, int | None]:
        if not self.client or not self.user_id:
            return False, None

        try:
            periods = [c for c in res_obj.amounts.columns if not ("YTD" in c or "AVG" in c)]
            periods_label = f"{periods[0]} – {periods[-1]}" if len(periods) >= 2 else (", ".join(periods) or "")

            rec_df = getattr(res_obj, "reconciliation", None)
            rec_json = rec_df.to_json(orient="split") if (rec_df is not None and not rec_df.empty) else ""

            payload = {
                "profile_name": self.user_id,
                "report_name": report_name,
                "file_name": file_name,
                "periods_label": periods_label,
                "amounts_json": res_obj.amounts.to_json(orient="split"),
                "main_json": res_obj.main.to_json(orient="split"),
                "kpi_json": res_obj.kpi.to_json(orient="split"),
                "row_kinds_json": json.dumps(res_obj.row_kinds, ensure_ascii=False),
                "reconciliation_json": rec_json,
                "commentary": commentary or "",
            }
            ins = self.client.table("saved_reports").insert(payload).execute()
            new_id = ins.data[0]["id"] if ins.data else None
            return True, new_id
        except Exception as e:
            self.last_error = str(e)
            return False, None

    def update_report_commentary(self, report_id: int, commentary: str) -> bool:
        if not self.client or not report_id:
            return False
        try:
            self.client.table("saved_reports").update({"commentary": commentary}).eq("id", report_id).execute()
            return True
        except Exception:
            return False

    def list_reports(self) -> list[dict]:
        if not self.client or not self.user_id:
            return []
        try:
            res = (
                self.client.table("saved_reports")
                .select("id, profile_name, report_name, file_name, periods_label, created_at")
                .eq("profile_name", self.user_id)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            return res.data or []
        except Exception as e:
            self.last_error = str(e)
            return []

    def load_report(self, report_id: int) -> dict | None:
        if not self.client:
            return None
        try:
            res = self.client.table("saved_reports").select("*").eq("id", report_id).limit(1).execute()
            if not res.data:
                return None
            row = res.data[0]
            amounts = pd.read_json(StringIO(row["amounts_json"]), orient="split")
            main = pd.read_json(StringIO(row["main_json"]), orient="split")
            kpi = pd.read_json(StringIO(row["kpi_json"]), orient="split")
            row_kinds = json.loads(row["row_kinds_json"])
            rec_json = row.get("reconciliation_json") or ""
            reconciliation = (
                pd.read_json(StringIO(rec_json), orient="split") if rec_json.strip() else pd.DataFrame()
            )
            return {
                "id": row["id"],
                "profile_name": row.get("profile_name", ""),
                "report_name": row.get("report_name", ""),
                "file_name": row.get("file_name", ""),
                "periods_label": row.get("periods_label", ""),
                "created_at": row.get("created_at", ""),
                "amounts": amounts,
                "main": main,
                "kpi": kpi,
                "row_kinds": row_kinds,
                "reconciliation": reconciliation,
                "commentary": row.get("commentary") or "",
            }
        except Exception as e:
            self.last_error = str(e)
            return None

    def delete_report(self, report_id: int) -> bool:
        if not self.client:
            return False
        try:
            self.client.table("saved_reports").delete().eq("id", report_id).execute()
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

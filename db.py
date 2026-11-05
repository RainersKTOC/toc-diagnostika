"""
db.py — Supabase datubāzes integrācija:
- Vairāku uzņēmumu profili (kontu klasifikācijas atmiņa)
- Saglabāto finanšu tabulu / atskaišu arhīvs (vēsture)
Project ID: amprqueamdcxpjufigvm
"""
from __future__ import annotations
import json
import os
from io import StringIO
import pandas as pd
import streamlit as st
from supabase import create_client, Client

# Tavi publiskie projekta savienojuma dati kā drošs rezerves variants (fallback),
# ja Streamlit Cloud neielasa .streamlit/secrets.toml failu:
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


@st.cache_resource
def get_supabase() -> tuple[Client | None, str | None]:
    """Atgriež (Supabase Client, kļūdas_ziņojums)."""
    url, key = _get_credentials()
    if not url or not key:
        return None, "Nav norādīts Supabase URL vai Key."
    try:
        client = create_client(url, key)
        return client, None
    except Exception as e:
        return None, str(e)


def _norm_part(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _norm_key(path: str, label: str) -> str:
    return f"{_norm_part(path)}||{_norm_part(label)}"


class SupabaseProfileStore:
    def __init__(self, profile_name: str = "Mans Uzņēmums"):
        self.user_id = str(profile_name).strip() or "Mans Uzņēmums"
        self.client, self.init_error = get_supabase()
        self.last_error: str | None = self.init_error

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------ PROFILU SARAKSTS
    def list_profiles(self) -> list[str]:
        """Atgriež visus datubāzē izveidotos uzņēmumu profilus."""
        base = ["Mans Uzņēmums"]
        if not self.client:
            return base
        found: set[str] = set()
        try:
            r1 = self.client.table("account_mappings").select("user_id").execute()
            for row in r1.data or []:
                u = str(row.get("user_id", "")).strip()
                if u and u != "default_user":
                    found.add(u)
                elif u == "default_user":
                    found.add("Mans Uzņēmums")
        except Exception:
            pass

        try:
            r2 = self.client.table("saved_reports").select("profile_name").execute()
            for row in r2.data or []:
                p = str(row.get("profile_name", "")).strip()
                if p:
                    found.add(p)
        except Exception:
            pass

        all_profiles = sorted(found | set(base))
        return all_profiles

    def count_saved(self) -> int:
        if not self.client:
            return 0
        try:
            res = (
                self.client.table("account_mappings")
                .select("id", count="exact")
                .in_("user_id", [self.user_id, "default_user"] if self.user_id == "Mans Uzņēmums" else [self.user_id])
                .execute()
            )
            self.last_error = None
            return int(res.count if res.count is not None else len(res.data or []))
        except Exception as e:
            self.last_error = str(e)
            return 0

    # ------------------------------------------------------------------ KONTU KLASIFIKĀCIJAS ATMIŅA
    def lookup(self, leaves) -> tuple[dict[str, dict], list]:
        if not self.client:
            return {}, list(leaves)

        target_users = [self.user_id, "default_user"] if self.user_id == "Mans Uzņēmums" else [self.user_id]
        try:
            res = (
                self.client.table("account_mappings")
                .select("path_label, account_label, toc_group, is_income")
                .in_("user_id", target_users)
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

            self.last_error = None
            return known, unknown
        except Exception as e:
            self.last_error = str(e)
            return {}, list(leaves)

    def save(self, leaves, mapping: dict[str, dict]) -> tuple[bool, int]:
        if not self.client:
            self.last_error = "Nav aktīva Supabase savienojuma."
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
            self.last_error = None
            return True, len(rows_to_upsert)
        except Exception as e:
            self.last_error = str(e)
            return False, 0

    # ------------------------------------------------------------------ FINANŠU TABULU VĒSTURE (ARHĪVS)
    def save_report(
        self,
        report_name: str,
        file_name: str,
        res_obj,
        commentary: str = "",
    ) -> tuple[bool, int | None]:
        """Saglabā aprēķināto TOC-FIN finanšu tabulu Supabase datubāzē."""
        if not self.client:
            self.last_error = "Nav aktīva Supabase savienojuma."
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
            self.last_error = None
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

    def list_reports(self, all_profiles: bool = False) -> list[dict]:
        """Atgriež saglabāto finanšu tabulu sarakstu."""
        if not self.client:
            return []
        try:
            q = (
                self.client.table("saved_reports")
                .select("id, profile_name, report_name, file_name, periods_label, created_at")
                .order("created_at", desc=True)
                .limit(50)
            )
            if not all_profiles:
                q = q.eq("profile_name", self.user_id)
            res = q.execute()
            return res.data or []
        except Exception as e:
            self.last_error = str(e)
            return []

    def load_report(self, report_id: int) -> dict | None:
        """Ielādē pilnu saglabāto finanšu tabulu no Supabase pēc ID."""
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
                "profile_name": row.get("profile_name", "Mans Uzņēmums"),
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

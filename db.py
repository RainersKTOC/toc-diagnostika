"""
db.py — Supabase datubāzes integrācija un kontu kartējumu glabāšana
"""
from __future__ import annotations
import os
import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def get_supabase() -> Client | None:
    try:
        url = st.secrets.get("supabase", {}).get("url") or os.environ.get("SUPABASE_URL")
        key = st.secrets.get("supabase", {}).get("key") or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None

class SupabaseProfileStore:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.client = get_supabase()

    def lookup(self, leaves) -> tuple[dict[str, dict], list]:
        """(zināmie {id: {group, is_income}}, nezināmās lapas)"""
        known, unknown = {}, []
        if not self.client:
            return {}, leaves

        try:
            res = self.client.table("account_mappings").select("*").eq("user_id", self.user_id).execute()
            data = res.data or {}
            mapping_db = {f"{row['path_label']}||{row['account_label']}": {"group": row["toc_group"], "is_income": row["is_income"]} for row in data}

            for lf in leaves:
                key = f"{lf.path.strip().lower()}||{lf.label.strip().lower()}"
                if key in mapping_db:
                    known[lf.id] = mapping_db[key]
                else:
                    unknown.append(lf)
        except Exception:
            return {}, leaves

        return known, unknown

    def save(self, leaves, mapping: dict[str, dict]) -> None:
        if not self.client:
            return
        
        rows_to_upsert = []
        for lf in leaves:
            if lf.id in mapping:
                rows_to_upsert.append({
                    "user_id": self.user_id,
                    "path_label": lf.path.strip().lower(),
                    "account_label": lf.label.strip().lower(),
                    "toc_group": mapping[lf.id]["group"],
                    "is_income": bool(mapping[lf.id]["is_income"])
                })
        
        if rows_to_upsert:
            try:
                self.client.table("account_mappings").upsert(rows_to_upsert, on_conflict="user_id,path_label,account_label").execute()
            except Exception as e:
                print("Kļūda saglabājot datubāzē:", e)

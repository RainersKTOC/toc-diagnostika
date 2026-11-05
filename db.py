"""
db.py — Supabase datubāzes integrācija un kontu kartējumu glabāšana (v2.0)
"""
from __future__ import annotations
import os
import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase() -> Client | None:
    """Izveido un kešo Supabase klientu no st.secrets vai vides mainīgajiem."""
    try:
        supa_cfg = st.secrets.get("supabase", {})
        url = str(supa_cfg.get("url", "") or os.environ.get("SUPABASE_URL", "")).strip()
        key = str(supa_cfg.get("key", "") or os.environ.get("SUPABASE_KEY", "")).strip()
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:
        st.warning(f"Supabase savienojuma brīdinājums: {e}")
        return None


def _norm_key(path: str, label: str) -> str:
    p = " ".join(str(path or "").strip().lower().split())
    l = " ".join(str(label or "").strip().lower().split())
    return f"{p}||{l}"


def _norm_part(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


class SupabaseProfileStore:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = str(user_id).strip() or "default_user"
        self.client = get_supabase()
        self.last_error: str | None = None

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    def count_saved(self) -> int:
        if not self.client:
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

    def lookup(self, leaves) -> tuple[dict[str, dict], list]:
        """Atgriež (zināmie {leaf.id: {'group': ..., 'is_income': ...}}, nezināmās_lapas)."""
        if not self.client:
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
                    # Ja nosaukums ir unikāls profilā, atpazīstam arī tad, ja ceļš nedaudz atšķiras
                    known[lf.id] = label_only_db[lbl_k][0]
                else:
                    unknown.append(lf)

            self.last_error = None
            return known, unknown
        except Exception as e:
            self.last_error = str(e)
            return {}, list(leaves)

    def save(self, leaves, mapping: dict[str, dict]) -> tuple[bool, int]:
        """Saglabā vai atjaunina kontu kartējumus Supabase. Atgriež (izdevās, skaits)."""
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

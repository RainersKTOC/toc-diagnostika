"""
profile_store.py — uzņēmuma profils: apstiprinātā kontu klasifikācija tiek atcerēta.

Šeit: vienkāršs JSON failu glabātājs (lokāli). Streamlit Cloud failu sistēma ir pagaidu —
produkcijā nomainiet ProfileStore.load/save uz Supabase/Postgres (interfeiss paliek tas pats).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("profiles")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()) or "default"


def _key(path: str, label: str) -> str:
    return f"{path.strip().lower()}||{label.strip().lower()}"


class ProfileStore:
    def __init__(self, company: str):
        self.file = ROOT / f"{_slug(company)}.json"
        self.data: dict[str, dict] = {}
        if self.file.exists():
            self.data = json.loads(self.file.read_text(encoding="utf-8"))

    def lookup(self, leaves) -> tuple[dict[str, dict], list]:
        """(zināmie {id: {group,is_income}}, nezināmās lapas)"""
        known, unknown = {}, []
        for lf in leaves:
            m = self.data.get(_key(lf.path, lf.label))
            if m:
                known[lf.id] = dict(m)
            else:
                unknown.append(lf)
        return known, unknown

    def save(self, leaves, mapping: dict[str, dict]) -> None:
        for lf in leaves:
            if lf.id in mapping:
                self.data[_key(lf.path, lf.label)] = {"group": mapping[lf.id]["group"],
                                                      "is_income": bool(mapping[lf.id]["is_income"])}
        ROOT.mkdir(exist_ok=True)
        self.file.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

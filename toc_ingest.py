"""
toc_ingest.py  —  PZA failu ielasīšana (1.–3. solis)
"""
from __future__ import annotations

import csv
import io
import itertools
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GENERIC_TOTAL = re.compile(r"^\W*(kopā|kopsumma|kopējais|kopējās|total|subtotal|summa)\W*$", re.I)
ACCOUNT_CODE = re.compile(r"^\s*(\d{4,6})\s+")
HAS_LETTER = re.compile(r"[A-Za-zĀ-žА-я]")


def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1257", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_grids(name: str, data: bytes) -> dict[str, pd.DataFrame]:
    low = name.lower()
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(data))
        return {s: xl.parse(s, header=None, dtype=object) for s in xl.sheet_names}

    text = decode_bytes(data)
    best_rows, best_score = None, -1
    for delim in (",", ";", "\t", "|"):
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        if not rows:
            continue
        lens = [len(r) for r in rows if any(c.strip() for c in r)]
        if not lens:
            continue
        mode = max(set(lens), key=lens.count)
        score = mode * lens.count(mode)
        if mode >= 2 and score > best_score:
            best_rows, best_score = rows, score
    if best_rows is None:
        best_rows = list(csv.reader(io.StringIO(text)))
    width = max(len(r) for r in best_rows)
    grid = [[(c.strip() if c.strip() != "" else None) for c in r] + [None] * (width - len(r)) for r in best_rows]
    return {"CSV": pd.DataFrame(grid, dtype=object)}


def detect_decimal_comma(grid: pd.DataFrame) -> bool:
    for v in grid.to_numpy().ravel():
        if isinstance(v, str):
            s = v.strip().replace(" ", "")
            if re.search(r"\d,\d{1,2}$", s) or re.search(r"\d\.\d{3},\d", s):
                return True
    return False


def parse_number(x, decimal_comma: bool = False) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.number)):
        return None if (isinstance(x, float) and np.isnan(x)) else float(x)
    s = str(x).strip().replace("\u00a0", "").replace("\u2212", "-").replace(" ", "")
    s = s.replace("€", "").replace("EUR", "")
    if s in ("-", "–", "—"):
        return 0.0
    if s == "" or s.endswith("%"):
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = s.replace(".", "").replace(",", ".") if decimal_comma else s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


PERIOD_KINDS = ("month", "ytd", "year", "other")


@dataclass
class Structure:
    sheet: str
    period_cols: list[dict]
    first_row: int
    last_row: int
    result_row: int | None = None
    sign_convention: str = "mixed"
    totals_position: str = "auto"
    pl_layout: str = "unclear"
    unit: str = ""
    company: str = ""
    notes: str = ""
    tolerance: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Structure":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


STRUCTURE_TOOL = {
    "name": "report_structure",
    "description": (
        "Report WHERE the profit-and-loss data sits in the sheet. Return indices only (0-based row and "
        "column numbers exactly as shown in the preview). Do not copy or calculate any amounts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sheet": {"type": "string", "description": "Name of the sheet that holds the P&L."},
            "period_cols": {
                "type": "array",
                "description": "Columns with ACTUAL amounts only. Exclude budget, prior-year, variance and % columns.",
                "items": {
                    "type": "object",
                    "properties": {
                        "col": {"type": "integer"},
                        "label": {"type": "string", "description": "'MM.YYYY.' for a month, 'YTD', or 'YYYY' for a full year."},
                        "kind": {"enum": list(PERIOD_KINDS)},
                    },
                    "required": ["col", "label", "kind"],
                },
            },
            "first_row": {"type": "integer", "description": "First row of the P&L body (first revenue line)."},
            "last_row": {"type": "integer", "description": "Last row of the P&L body. Exclude KPI blocks, notes, footers below it."},
            "result_row": {"type": ["integer", "null"],
                           "description": "The final bottom-line profit/loss row of the whole P&L (the last profit line that already includes all items)."},
            "sign_convention": {"enum": ["costs_negative", "costs_positive", "mixed"],
                                "description": "How costs are signed in the detail lines. 'mixed' if summary lines and breakdown lines differ."},
            "totals_position": {"enum": ["after", "before", "auto"],
                                "description": "Is a total printed AFTER its component lines, or BEFORE them? 'auto' if unsure."},
            "pl_layout": {"enum": ["by_nature", "by_function", "unclear"],
                          "description": "by_nature = costs by type (materials, wages, rent...). by_function = only COGS/selling/admin totals with no breakdown."},
            "unit": {"type": "string", "description": "EUR, tūkst. EUR, ..."},
            "company": {"type": ["string", "null"]},
            "notes": {"type": "string", "description": "Anything unusual a human should know (max 2 sentences)."},
        },
        "required": ["sheet", "period_cols", "first_row", "last_row", "sign_convention", "totals_position", "pl_layout"],
    },
}

CHOOSE_SHEET_TOOL = {
    "name": "choose_sheet",
    "description": "Pick the sheet that contains the profit and loss statement (PZA / P&L / income statement).",
    "input_schema": {"type": "object", "properties": {"sheet": {"type": "string"}}, "required": ["sheet"]},
}

STRUCTURE_SYSTEM = """You inspect accounting exports (Latvian or English, any accounting system) and report where the profit-and-loss data is.
You receive a cell-by-cell preview: `r<row> | c<col>:<value> ...` (0-based indices, empty cells omitted).
Rules:
- Report indices only. Never output or compute amounts.
- period_cols: ACTUAL amounts only (skip budget 'BU', prior year 'PY', variances, % columns). Monthly files: one entry per month column that has data. If the file has a YTD/cumulative actual column, add it with kind 'ytd'. A single full-year column -> kind 'year'.
- first_row/last_row delimit the P&L body. In files where a summary P&L is followed by a detailed breakdown (decoding of line items), last_row is the END of the breakdown.
- result_row is the final profit/loss line (e.g. 'Pārskata gada peļņa vai zaudējumi', 'Net Result').
- If the statement is by function (only COGS / selling / admin totals, no breakdown by cost type), say pl_layout='by_function'.
- If something is unclear, say so in notes instead of guessing."""


def grid_preview(df: pd.DataFrame, max_rows: int = 260, max_cols: int = 24, cell_chars: int = 52) -> str:
    n = len(df)
    rows = list(range(n)) if n <= max_rows else list(range(0, max_rows - 60)) + list(range(n - 60, n))
    out = []
    for r in rows:
        cells = []
        for c in range(min(df.shape[1], max_cols)):
            v = df.iat[r, c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            s = str(v).strip()
            if s == "":
                continue
            cells.append(f"c{c}:{s[:cell_chars]}")
        if cells:
            out.append(f"r{r} | " + " | ".join(cells))
    head = f"[{n} rows x {df.shape[1]} cols]"
    return head + "\n" + "\n".join(out)


def validate_structure(grid: pd.DataFrame, st: Structure) -> list[str]:
    errs: list[str] = []
    n, m = grid.shape
    if not st.period_cols:
        errs.append("period_cols is empty.")
    for p in st.period_cols:
        if not (0 <= p["col"] < m):
            errs.append(f"period col {p['col']} is outside the sheet (0..{m - 1}).")
        if p.get("kind") not in PERIOD_KINDS:
            errs.append(f"period col {p['col']}: invalid kind.")
    if not (0 <= st.first_row <= st.last_row < n):
        errs.append(f"row range {st.first_row}..{st.last_row} is invalid for {n} rows.")
    if st.result_row is not None and not (st.first_row <= st.result_row <= st.last_row):
        errs.append("result_row is outside first_row..last_row.")
    if errs:
        return errs
    dec = detect_decimal_comma(grid)
    counts = []
    for p in st.period_cols:
        col = grid.iloc[st.first_row: st.last_row + 1, p["col"]]
        counts.append(sum(parse_number(v, dec) is not None for v in col))
    if max(counts) < 3:
        errs.append("None of the period columns contains numbers in the given row range - wrong columns or rows.")
    return errs


@dataclass
class Line:
    pos: int
    row: int
    label: str
    code: str
    vals: np.ndarray
    has_values: bool
    is_ratio: bool = False


@dataclass
class Extracted:
    lines: list[Line]
    period_cols: list[dict]
    dropped_cols: list[dict]
    decimal_comma: bool
    tol: float


def extract_lines(grid: pd.DataFrame, st: Structure) -> Extracted:
    dec = detect_decimal_comma(grid)
    first_pc = min(p["col"] for p in st.period_cols)
    label_cols = list(range(0, first_pc)) or [0]

    raw = []
    for r in range(st.first_row, st.last_row + 1):
        label, code = "", ""
        for c in label_cols:
            v = grid.iat[r, c]
            if isinstance(v, str) and HAS_LETTER.search(v):
                label = v.strip()
                break
        if not label:
            for c in label_cols:
                v = grid.iat[r, c]
                if isinstance(v, str) and v.strip():
                    code = v.strip()
        m = ACCOUNT_CODE.match(label)
        if m:
            code = m.group(1)
        nums = [parse_number(grid.iat[r, p["col"]], dec) for p in st.period_cols]
        raw.append((r, label, code, nums))

    keep = []
    for j, p in enumerate(st.period_cols):
        if any((x[3][j] or 0) != 0 for x in raw):
            keep.append(j)
    dropped = [st.period_cols[j] for j in range(len(st.period_cols)) if j not in keep]
    kept_cols = [st.period_cols[j] for j in keep]

    allv = np.array([abs(x[3][j]) for x in raw for j in keep if x[3][j] is not None] or [0.0])
    integer_like = bool(np.all(np.abs(allv - np.round(allv)) < 1e-9))
    tol = st.tolerance if st.tolerance else (2.0 if integer_like else 0.05)

    lines: list[Line] = []
    for r, label, code, nums in raw:
        sel = [nums[j] for j in keep]
        has = any(v is not None for v in sel)
        if not label and not has:
            continue
        vec = np.array([v or 0.0 for v in sel], dtype=float)
        is_ratio = "%" in label or label.lower().startswith("(as ")
        lines.append(Line(len(lines), r, label or f"(rinda {r})", code, vec, has, is_ratio))
    return Extracted(lines, kept_cols, dropped, dec, tol)


def _stack_totals(order: list[int], V: np.ndarray, tol: float) -> dict[int, list[int]]:
    stack: list[int] = []
    totals: dict[int, list[int]] = {}
    for i in order:
        v = V[i]
        if np.any(np.abs(v) > tol):
            acc = np.zeros_like(v)
            best = 0
            for k in range(1, len(stack) + 1):
                acc = acc + V[stack[-k]]
                if np.all(np.abs(acc - v) <= tol):
                    best = k
            if best:
                totals[i] = stack[-best:]
                del stack[-best:]
        stack.append(i)
    return totals


@dataclass
class Leaf:
    id: str
    row: int
    label: str
    path: str
    code: str
    vals: np.ndarray
    mult: int | None


@dataclass
class Analysis:
    ex: Extracted
    direction: str
    totals: dict[int, list[int]]
    leaves: list[Leaf]
    mirrors: dict[int, tuple[int, int]]
    derived: set[int]
    zero_rows: list[int]
    notes: list[str] = field(default_factory=list)

    def roles(self) -> dict[int, str]:
        r = {}
        for ln in self.ex.lines:
            r[ln.pos] = "ratio" if ln.is_ratio else ("header" if not ln.has_values else "leaf")
        for t in self.totals:
            r[t] = "total"
        for l in self.mirrors:
            r[l] = "mirror"
        for d in self.derived:
            r[d] = "derived"
        for z in self.zero_rows:
            r[z] = "zero"
        return r


_COMBO_CACHE: dict[int, np.ndarray] = {}


def _combos(m: int) -> np.ndarray:
    if m not in _COMBO_CACHE:
        _COMBO_CACHE[m] = np.array(list(itertools.product((-1, 0, 1), repeat=m)), dtype=float)
    return _COMBO_CACHE[m]


def analyze(ex: Extracted, st: Structure) -> Analysis:
    lines = ex.lines
    tol = ex.tol
    V = np.array([ln.vals for ln in lines]) if lines else np.zeros((0, 1))
    active = [ln.pos for ln in lines if ln.has_values and not ln.is_ratio]

    after = _stack_totals(active, V, tol)
    before = _stack_totals(active[::-1], V, tol)
    pos_hint = st.totals_position
    if pos_hint == "after":
        direction, totals = "after", after
    elif pos_hint == "before":
        direction, totals = "before", before
    else:
        direction, totals = ("after", after) if len(after) >= len(before) else ("before", before)
    notes = [f"Kopsummu virziens: {direction} (atrasts: after={len(after)}, before={len(before)})."]

    parent: dict[int, int] = {}
    for t, ch in totals.items():
        for c in ch:
            parent[c] = t

    def ancestors(i: int) -> list[int]:
        out = []
        while i in parent:
            i = parent[i]
            out.append(i)
        return out

    def descendants_leaves(t: int) -> list[int]:
        out, st_ = [], [t]
        while st_:
            x = st_.pop()
            if x in totals:
                st_.extend(totals[x])
            else:
                out.append(x)
        return out

    mirrors: dict[int, tuple[int, int]] = {}
    leaf_cand = [i for i in active if i not in totals]
    for L in leaf_cand:
        if not np.any(np.abs(V[L]) > tol):
            continue
        anc = set(ancestors(L))
        best = None
        for T in totals:
            if T in anc or T == L:
                continue
            for f in (1, -1):
                if np.all(np.abs(V[L] - f * V[T]) <= tol):
                    size = len(descendants_leaves(T))
                    if best is None or size > best[2]:
                        best = (T, f, size)
        if best and best[2] >= 2:
            mirrors[L] = (best[0], best[1])

    roots = [i for i in active if i not in parent and i not in mirrors]
    derived: set[int] = set()
    nonzero_roots: list[int] = []
    for R in roots:
        if not np.any(np.abs(V[R]) > tol):
            continue
        win = nonzero_roots[-10:]
        if win:
            C = _combos(len(win))
            M = V[win]
            res = np.abs(C @ M - V[R]).max(axis=1) <= tol
            res &= (np.abs(C).sum(axis=1) >= 1)
            if res.any():
                derived.add(R)
        nonzero_roots.append(R)

    for d in list(derived):
        stack_ = [d]
        while stack_:
            x = stack_.pop()
            for c in totals.get(x, []):
                derived.add(c)
                stack_.append(c)

    zero_rows = [i for i in leaf_cand if not np.any(np.abs(V[i]) > tol) and i not in mirrors]
    header_pos = {ln.pos for ln in lines if not ln.has_values and ln.label and not ln.label.startswith("(rinda")}

    def section_name(t: int) -> str:
        lab = lines[t].label
        if not GENERIC_TOTAL.match(lab):
            return lab
        kids = sorted(totals[t])
        first = kids[0]
        p = first - 1
        run = []
        while p >= 0 and (p in header_pos):
            run.append(p)
            p -= 1
        return lines[run[-1]].label.strip() if run else lab

    chain_cache: dict[int, bool] = {}

    def is_chain(t: int) -> bool:
        if t not in chain_cache:
            chain_cache[t] = any((c in mirrors) or (c in totals and is_chain(c)) for c in totals[t])
        return chain_cache[t]

    mult_of: dict[int, int] = {}
    for L, (T, f) in mirrors.items():
        for d in descendants_leaves(T):
            mult_of.setdefault(d, f)

    skip = set(mirrors) | derived | set(zero_rows) | set(totals)
    leaves: list[Leaf] = []
    for i in leaf_cand:
        if i in skip:
            continue
        anc = [a for a in ancestors(i)]
        names = []
        for a in anc:
            if is_chain(a):
                continue
            nm = section_name(a)
            if nm and nm not in names:
                names.append(nm)
            if len(names) == 2:
                break
        path = " › ".join(reversed(names)) if names else ""
        ln = lines[i]
        if i in mult_of:
            mult: int | None = mult_of[i]
        elif st.sign_convention == "costs_positive":
            mult = None
        else:
            mult = 1
        leaves.append(Leaf(f"r{ln.row}", ln.row, ln.label, path, ln.code, V[i].copy(), mult))

    if mirrors:
        notes.append(f"{len(mirrors)} kopsavilkuma rindas ir atpazītas kā dublikāti un netiek skaitītas.")
    if derived:
        notes.append(f"{len(derived)} rindas atpazītas kā atvasinātas (peļņas rindas) un netiek skaitītas.")
    return Analysis(ex, direction, totals, leaves, mirrors, derived, zero_rows, notes)


def leaf_amounts(leaves: list[Leaf], income_ids: set[str] | None) -> tuple[np.ndarray, bool]:
    if not leaves:
        return np.zeros((0, 1)), True
    rows, complete = [], True
    for lf in leaves:
        if lf.mult is not None:
            m = lf.mult
        elif income_ids is None:
            m, complete = 0, False
        else:
            m = 1 if lf.id in income_ids else -1
        rows.append(m * lf.vals)
    return np.array(rows), complete


def reconcile(an: Analysis, st: Structure, income_ids: set[str] | None = None) -> dict:
    out = {"status": "skip", "table": None, "message": ""}
    if st.result_row is None:
        out["message"] = "Gala rezultāta rinda nav norādīta — salīdzināšana nav iespējama."
        return out
    pos = next((ln.pos for ln in an.ex.lines if ln.row == st.result_row), None)
    if pos is None:
        out["message"] = "Gala rezultāta rinda nav atrasta rindu sarakstā."
        return out
    A, complete = leaf_amounts(an.leaves, income_ids)
    if not complete:
        out["status"] = "deferred"
        out["message"] = ("Šim formātam izdevumu zīme jānosaka pēc klasifikācijas — "
                          "salīdzināšana tiks veikta pēc 4. soļa.")
        return out
    calc = A.sum(axis=0)
    rep = an.ex.lines[pos].vals
    diff = calc - rep
    tbl = pd.DataFrame({"Periods": [p["label"] for p in an.ex.period_cols],
                        "Σ lapas": calc, "Faila rezultāts": rep, "Starpība": diff})
    ok = bool(np.all(np.abs(diff) <= max(an.ex.tol * 3, 1.0)))
    out.update(status="ok" if ok else "fail", table=tbl,
               message="Visu lapu summa sakrīt ar faila gala rezultātu." if ok
               else "Lapu summa NESAKRĪT ar faila gala rezultātu — dati vai struktūra ir nepareizi nolasīti.")
    return out


def verification_report(grid: pd.DataFrame, st: Structure, an: Analysis,
                        income_ids: set[str] | None = None) -> list[dict]:
    rep = []
    errs = validate_structure(grid, st)
    rep.append({"Pārbaude": "Struktūra (kolonnas, rindu diapazons)",
                "Statuss": "OK" if not errs else "KĻŪDA", "Detaļas": "; ".join(errs) or "—"})
    if an.ex.dropped_cols:
        rep.append({"Pārbaude": "Tukšas periodu kolonnas",
                    "Statuss": "INFO",
                    "Detaļas": "Izlaistas: " + ", ".join(p["label"] for p in an.ex.dropped_cols)})
    roles = an.roles()
    counts = pd.Series(list(roles.values())).value_counts().to_dict()
    unclassified = [ln for ln in an.ex.lines if ln.has_values and not ln.is_ratio and ln.pos not in roles]
    rep.append({"Pārbaude": "Katra rinda ar skaitļiem ir ar skaidru lomu",
                "Statuss": "OK" if not unclassified else "KĻŪDA",
                "Detaļas": ", ".join(f"{k}={v}" for k, v in counts.items())})
    n_tot = len(an.totals)
    rep.append({"Pārbaude": "Kopsummas sakrīt ar komponentēm",
                "Statuss": "OK" if n_tot else "BRĪDINĀJUMS",
                "Detaļas": f"{n_tot} kopsummas atrastas un pārbaudītas ({an.direction})"
                           if n_tot else "Neviena kopsumma nav atrasta — hierarhija nav noteikta."})
    rec = reconcile(an, st, income_ids)
    status = {"ok": "OK", "fail": "KĻŪDA", "deferred": "ATLIKTS", "skip": "BRĪDINĀJUMS"}[rec["status"]]
    rep.append({"Pārbaude": "Σ(lapas) = faila gala rezultāts", "Statuss": status, "Detaļas": rec["message"]})

    cols = an.ex.period_cols
    months = [j for j, p in enumerate(cols) if p["kind"] == "month"]
    ytd = [j for j, p in enumerate(cols) if p["kind"] == "ytd"]
    if len(months) >= 2 and ytd and an.leaves:
        A, complete = leaf_amounts(an.leaves, income_ids)
        if complete:
            d = np.abs(A[:, months].sum(axis=1) - A[:, ytd[0]]).max()
            rep.append({"Pārbaude": "YTD kolonna = mēnešu summa",
                        "Statuss": "OK" if d <= an.ex.tol * 3 else "BRĪDINĀJUMS",
                        "Detaļas": f"Lielākā starpība vienai lapai: {d:,.0f}"})
    return rep


def _tool_input(resp, name: str):
    tu = next((b for b in resp.content if getattr(b, "type", "") == "tool_use" and b.name == name), None)
    return tu


def pick_sheet(client, model: str, grids: dict[str, pd.DataFrame]) -> str:
    if len(grids) == 1:
        return next(iter(grids))
    parts = [f"=== SHEET '{n}' ===\n{grid_preview(g, max_rows=30, max_cols=12)}" for n, g in list(grids.items())[:8]]
    resp = client.messages.create(
        model=model, max_tokens=300,
        system="Choose the sheet containing the profit and loss statement (income statement).",
        tools=[CHOOSE_SHEET_TOOL], tool_choice={"type": "tool", "name": "choose_sheet"},
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
    sheet = _tool_input(resp, "choose_sheet").input["sheet"]
    return sheet if sheet in grids else next(iter(grids))


def request_structure(client, model: str, sheet: str, grid: pd.DataFrame, max_tries: int = 3) -> Structure:
    msgs = [{"role": "user", "content": f"Sheet '{sheet}':\n{grid_preview(grid)}"}]
    last_errs: list[str] = []
    for _ in range(max_tries):
        resp = client.messages.create(
            model=model, max_tokens=2000, system=STRUCTURE_SYSTEM,
            tools=[STRUCTURE_TOOL], tool_choice={"type": "tool", "name": STRUCTURE_TOOL["name"]},
            messages=msgs,
        )
        tu = _tool_input(resp, STRUCTURE_TOOL["name"])
        d = dict(tu.input)
        d["sheet"] = sheet
        st = Structure.from_dict(d)
        last_errs = validate_structure(grid, st)
        if not last_errs:
            return st
        msgs += [{"role": "assistant", "content": resp.content},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                               "content": "Validation failed: " + " ".join(last_errs) + " Fix and resend."}]}]
    raise ValueError("Struktūru neizdevās noteikt automātiski: " + " ".join(last_errs))


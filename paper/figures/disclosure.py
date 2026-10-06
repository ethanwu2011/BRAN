"""Disclosure check of the BRAN Source Data.

Run from the repository root with python -m paper.figures.disclosure. It reads only paper/source_data and needs no
plotting package. It exits 1 if any rule fails.

Rules, applied to every Source Data JSON file and to the header rows of every workbook.
  1. Every column a panel declares as a count, and every person-count column found by name, holds numbers of at
     least 20. A column declared as floored holds multiples of 20.
  2. Every lower bound printed as "≥ n" and every "n = k" or "n ≥ k" in any text, legends and notes included, is at
     least 20.
  3. No column is named like a person or record identifier, no cell holds a long digit string that could be an
     identifier, and no panel has more than 400 rows, so no table can list people one by one.
The check also reports, without failing, floats that equal a fraction k/d with an odd denominator d above 20 (other
than a power of 2 times a power of 5), because such a proportion can reveal the exact count d behind it. Each figure
team reviews those cells.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import zipfile

REPO = Path(__file__).resolve().parents[2]
SOURCE_DATA = REPO / "paper" / "source_data"
MANIFEST = SOURCE_DATA / "SOURCE_DATA_MANIFEST.json"
COUNT_FLOOR = 20
MAX_ROWS = 400
PERSON_COUNT = re.compile(r"(^n$|^n_|_n$|people|persons|participants|patients|subjects|deaths|cases|admissions|"
                          r"survivors|answered|^yes_|_count$|^count_)", re.I)
NOT_COUNT_SUFFIX = ("_type", "_definition", "_note", "_source", "_sha256", "_printed")
# Columns whose name looks like a person count but whose numbers count something else, with the reason.
NOT_PERSON_COUNT = {("fig1", "c", "count_or_name"): "number of model outputs in a row of the schematic, not people"}
IDENTIFIER = re.compile(r"(^|_)(id|ids|patient|patients_id|person_id|subject|subject_id|row_id|mrn|seqn|stay_id|"
                        r"hadm_id|subject_ids|image_id|file_id|participant_id|record_id)$", re.I)
LOWER_BOUND = re.compile(r"≥\s?([0-9][0-9,]*)")
N_EQUALS = re.compile(r"\b[nN]\s*(?:=|≥)\s*([0-9][0-9,]*)\b")
LONG_DIGITS = re.compile(r"^\d{7,}$")
RATIO_MAX_DEN = 20000
RATIO_TOL = 1e-12


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def _floats(o, path=()):
    if isinstance(o, float):
        yield path, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _floats(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _floats(v, path + (str(i),))


def _odd_part(d: int) -> int:
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d


def ratio_flags(payload: dict) -> list[str]:
    """Floats in the panels that equal k/d exactly (within 1e-12) for 20 < d <= 20000 with d not of the form 2^a 5^b."""
    out = []
    for path, v in _floats(payload.get("panels", {})):
        if v == 0 or abs(v) >= 1e6:
            continue
        f = Fraction(v).limit_denominator(RATIO_MAX_DEN)
        if f.denominator > COUNT_FLOOR and _odd_part(f.denominator) > 1 and abs(float(f) - v) <= RATIO_TOL:
            out.append(f"{'/'.join(path)}={v!r} equals {f.numerator}/{f.denominator}")
    return out


def payload_problems(fid: str, payload: dict) -> list[str]:
    out = []
    for key, panel in payload["panels"].items():
        cols = panel["columns"]
        rows = panel["rows"]
        if len(rows) > MAX_ROWS:
            out.append(f"{fid} panel {key} has {len(rows)} rows")
        for c in cols:
            if IDENTIFIER.search(c):
                out.append(f"{fid} panel {key} column {c!r} looks like an identifier")
        declared = set(panel.get("count_columns", []))
        named = {c for c in cols if PERSON_COUNT.search(c) and not c.endswith(NOT_COUNT_SUFFIX)
                 and (fid, key, c) not in NOT_PERSON_COUNT}
        counts = [cols.index(c) for c in sorted(declared | named) if c in cols]
        floored = [cols.index(c) for c in panel.get("floored_columns", []) if c in cols]
        for r in rows:
            if len(r) != len(cols):
                out.append(f"{fid} panel {key} row of length {len(r)} for {len(cols)} columns")
                continue
            for i in counts:
                v = r[i]
                if isinstance(v, bool) or v is None or isinstance(v, str):
                    if cols[i] in declared and isinstance(v, (str, bool)):
                        out.append(f"{fid} panel {key} declared count {cols[i]}={v!r} is not a number")
                    continue
                if v < COUNT_FLOOR:
                    out.append(f"{fid} panel {key} count {cols[i]}={v!r} below {COUNT_FLOOR}")
            for i in floored:
                v = r[i]
                if v is not None and (int(v) != v or int(v) % COUNT_FLOOR):
                    out.append(f"{fid} panel {key} floored count {cols[i]}={v!r} is not a multiple of {COUNT_FLOOR}")
            for v in r:
                if isinstance(v, str) and LONG_DIGITS.match(v.strip()):
                    out.append(f"{fid} panel {key} cell {v!r} looks like an identifier")
    for s in _strings(payload):
        for pattern in (LOWER_BOUND, N_EQUALS):
            for m in pattern.finditer(s):
                if int(m.group(1).replace(",", "")) < COUNT_FLOOR:
                    out.append(f"{fid} prints {m.group(0)!r} in {s[max(0, m.start() - 30):m.end() + 10]!r}")
    return out


def workbook_problems(path: Path) -> list[str]:
    """Header cells of every sheet, read from the workbook XML without a spreadsheet library."""
    out = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = z.read("xl/sharedStrings.xml").decode("utf-8") if "xl/sharedStrings.xml" in names else ""
        inline = "".join(z.read(n).decode("utf-8") for n in names if n.startswith("xl/worksheets/"))
    texts = re.findall(r"<t[^>]*>([^<]*)</t>", shared + inline)
    for t in texts:
        if IDENTIFIER.search(t.strip()):
            out.append(f"{path.name} has a cell {t.strip()!r} that names an identifier")
        if LONG_DIGITS.match(t.strip()):
            out.append(f"{path.name} has a cell {t.strip()!r} that looks like an identifier")
    return out


def check_all() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = {"files": 0, "problems": [], "ratio_flags": {}}
    for item in manifest["files"]:
        payload = json.loads((SOURCE_DATA / item["json"]["path"]).read_text(encoding="utf-8"))
        report["problems"] += payload_problems(item["id"], payload)
        report["problems"] += workbook_problems(SOURCE_DATA / item["xlsx"]["path"])
        flags = ratio_flags(payload)
        if flags:
            report["ratio_flags"][item["id"]] = flags
        report["files"] += 2
    return report


def main() -> int:
    report = check_all()
    summary = {"files_checked": report["files"], "problems": report["problems"],
               "ratio_flags_for_review": {k: len(v) for k, v in report["ratio_flags"].items()},
               "rules": "counts and printed bounds at least 20, no identifier columns or cells, at most 400 rows"}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())

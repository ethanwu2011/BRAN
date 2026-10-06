"""Disclosure check of the BRAN Source Data.

Run from the repository root with python -m paper.figures.disclosure. It reads only paper/source_data and needs no
plotting package. It exits 1 if any rule fails, 2 if the only findings are the open items listed in OPEN_ITEMS, and
0 when nothing is found. An open item blocks release until the figure builder that owns it exports again. With
--allow-open the check also exits 0 when open items are its only findings.

Rules, applied to the panels, the draw section and every text of each Source Data JSON file, and to the cells of every
workbook.
  1. Every column a panel declares as a count, and every person-count column found by name, holds numbers of at
     least 20. A column declared as floored holds multiples of 20. Person-count keys of the draw section hold numbers
     of at least 20.
  2. Every lower bound printed as "≥ n" and every "n = k" or "n ≥ k" in any text, legends and notes included, is at
     least 20.
  3. No column is named like a person or record identifier, no cell holds a long digit string that could be an
     identifier, and no panel has more than 400 rows, so no table can list people one by one.
  4. No float in the panels or the draw section equals a fraction k/d with an odd denominator d above 20 (other than
     a power of 2 times a power of 5), because such a proportion can reveal the exact count d behind it. A cell that
     REVIEWED_RATIOS accepts, with the reason given there, or that an open item lists, is exempt.
"""
from __future__ import annotations

import argparse
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
# Columns or draw keys whose name looks like a person count but whose numbers count something else, with the reason.
NOT_PERSON_COUNT = {
    ("fig1", "c", "count_or_name"): "number of model outputs in a row of the schematic, not people",
    ("fig1", "draw", "n_ocular"): "number of ocular outputs of the model, not people",
    ("fig1", "draw", "n_cbc"): "number of blood count outputs of the model, not people",
    ("fig1", "draw", "n_clin_sources"): "number of clinical-only training sources, not people",
}
IDENTIFIER = re.compile(r"(^|_)(id|ids|patient|patients_id|person_id|subject|subject_id|row_id|mrn|seqn|stay_id|"
                        r"hadm_id|subject_ids|image_id|file_id|participant_id|record_id)$", re.I)
LOWER_BOUND = re.compile(r"≥\s?([0-9][0-9,]*)")
N_EQUALS = re.compile(r"\b[nN]\s*(?:=|≥)\s*([0-9][0-9,]*)\b")
LONG_DIGITS = re.compile(r"^\d{7,}$")
RATIO_MAX_DEN = 20000
RATIO_TOL = 1e-12

# Exact fractions that were reviewed and are not counts of people, as (file id, pattern over the cell path, reason).
# A cell path is panels/<panel>/rows/<row>/<column> or draw/<key>/... as ratio_cells() writes it.
REVIEWED_RATIOS = [
    ("fig1", r"panels/a/rows/\d+/circle_area_relative",
     "circle area relative to the largest cohort, the ratio of two released lower bounds that are multiples of 20"),
    ("fig2", r"panels/b strip/rows/\d+/p_(one_sided|holm)",
     "P values from 1,000 resamples, of the form (k + 1)/1,001, so the denominator counts resamples and not people"),
    ("fig5", r"panels/e/rows/\d+/holm_adjusted_p_one_sided",
     "Holm-adjusted P value from 1,000 resamples, a multiple of 1/1,001"),
    ("ed1", r"(panels/b/rows/\d+/holm_p|draw/pb/con/\w+/p)",
     "Holm-adjusted P values from 2,000 resamples, multiples of 1/2,001"),
    ("ed3", r"panels/b/rows/363/estimate",
     "mean of 22 per-condition differences printed to four decimals, so the denominator comes from the 22 conditions"),
]

# Findings that block release until the figure builder that owns them exports again. Each item names its file, the
# cells or panel it concerns, the builder that must change and the change. The tests require every item to still be
# present, so an item is removed here once its export is fixed, and to be listed in paper/source_data/README.md.
OPEN_ITEMS = {
    "fig2-retrieval-cohort": {
        "id": "fig2", "kind": "ratio", "cells": r"panels/a/rows/\d+/value",
        "owner": "private Fig. 2 builder (build_fig2.py)",
        "problem": "Panel a prints the retrieval cohort exactly and its proportions are exact fractions of it, while "
                   "the development cohort is printed exactly elsewhere and differs from it by fewer than 20 people.",
        "fix": "Print the retrieval cohort as a lower bound floored to 20, round the panel a proportions to 3 "
               "decimals, export again and resync.",
    },
    "fig5-hirid-local-fit": {
        "id": "fig5", "kind": "ratio", "cells": r"(panels/g/rows/\d+/(mae_g_dl|ci95_low|ci95_high)|draw/g/.+)",
        "owner": "private Fig. 5 builder (build_fig5.py)",
        "problem": "Panel g gives unrounded mean absolute errors of the HiRID local fit, and an exact fraction among "
                   "them constrains the admission count that every HiRID aggregate withholds.",
        "fix": "Round the panel g errors and their intervals to 3 decimals, export again and resync.",
    },
    "ed2-count-type": {
        "id": "ed2", "kind": "count_type", "panel": "b",
        "owner": "private Extended Data Fig. 2 builder (build_ed2.py)",
        "problem": "Panel b prints exact site, image, device and study-group counts in columns that do not mark "
                   "them exact. Only the panel title says that site and image counts are exact, and the About sheet "
                   "still says that counts are floored unless the column says exact.",
        "fix": "Add a count-type column to panel b, as Fig. 5 and Extended Data Table 1 have, correct the About sheet, "
               "export again and resync.",
    },
}


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


def _leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _leaves(v, path + (str(i),))
    else:
        yield path, o


def _odd_part(d: int) -> int:
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d


def _exact_fraction(v) -> Fraction | None:
    if not isinstance(v, float) or v == 0 or abs(v) >= 1e6:
        return None
    f = Fraction(v).limit_denominator(RATIO_MAX_DEN)
    if f.denominator > COUNT_FLOOR and _odd_part(f.denominator) > 1 and abs(float(f) - v) <= RATIO_TOL:
        return f
    return None


def ratio_cells(payload: dict) -> list[tuple[str, Fraction]]:
    """Floats of the panels and the draw section that equal k/d exactly (within 1e-12) for 20 < d <= 20000 with d not
    of the form 2^a 5^b. Panel cells are named by column, panels/<panel>/rows/<row>/<column>."""
    out = []
    for key, panel in payload.get("panels", {}).items():
        cols = panel.get("columns", [])
        for i, row in enumerate(panel.get("rows", [])):
            for col, v in zip(cols, row):
                for sub, leaf in _leaves(v):
                    f = _exact_fraction(leaf)
                    if f is not None:
                        out.append(("/".join(("panels", key, "rows", str(i), col) + sub), f))
    for path, leaf in _leaves(payload.get("draw", {}), ("draw",)):
        f = _exact_fraction(leaf)
        if f is not None:
            out.append(("/".join(path), f))
    return out


def ratio_flags(payload: dict) -> list[str]:
    """Every exact-fraction cell, reviewed or not, for a figure team's review. Not printed by the command line."""
    return [f"{p}={float(f)!r} equals {f.numerator}/{f.denominator}" for p, f in ratio_cells(payload)]


def reviewed_reason(fid: str, path: str) -> str | None:
    for rid, pattern, reason in REVIEWED_RATIOS:
        if rid == fid and re.fullmatch(pattern, path):
            return reason
    return None


def open_item_of(fid: str, path: str) -> str | None:
    for name, item in OPEN_ITEMS.items():
        if item["id"] == fid and item["kind"] == "ratio" and re.fullmatch(item["cells"], path):
            return name
    return None


def open_items_present(fid: str, payload: dict) -> list[str]:
    """Names of the open items that still apply to this file."""
    found = set()
    for path, _ in ratio_cells(payload):
        name = open_item_of(fid, path)
        if name:
            found.add(name)
    for name, item in OPEN_ITEMS.items():
        if item["id"] == fid and item["kind"] == "count_type":
            cols = payload.get("panels", {}).get(item["panel"], {}).get("columns", [])
            if not any("count_type" in c for c in cols):
                found.add(name)
    return sorted(found)


def ratio_problems(fid: str, payload: dict) -> list[str]:
    out = []
    for path, f in ratio_cells(payload):
        if reviewed_reason(fid, path) or open_item_of(fid, path):
            continue
        out.append(f"{fid} {path} is an exact fraction with denominator {f.denominator}, review it and round it in "
                   "the private builder or add it to REVIEWED_RATIOS with a reason")
    return out


def draw_count_problems(fid: str, payload: dict) -> list[str]:
    out = []
    for path, v in _leaves(payload.get("draw", {})):
        names = [p for p in path if not p.isdigit()]
        if not names or isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        key = names[-1]
        if not PERSON_COUNT.search(key) or key.endswith(NOT_COUNT_SUFFIX) or (fid, "draw", key) in NOT_PERSON_COUNT:
            continue
        if v < COUNT_FLOOR:
            out.append(f"{fid} draw {'/'.join(path)}={v!r} below {COUNT_FLOOR}")
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
    out += draw_count_problems(fid, payload)
    out += ratio_problems(fid, payload)
    return out


def workbook_problems(path: Path) -> list[str]:
    """Text cells of every sheet, read from the workbook XML without a spreadsheet library."""
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
    report = {"files": 0, "problems": [], "open_items": {}, "reviewed_ratio_cells": 0}
    for item in manifest["files"]:
        payload = json.loads((SOURCE_DATA / item["json"]["path"]).read_text(encoding="utf-8"))
        report["problems"] += payload_problems(item["id"], payload)
        report["problems"] += workbook_problems(SOURCE_DATA / item["xlsx"]["path"])
        for name in open_items_present(item["id"], payload):
            report["open_items"][name] = OPEN_ITEMS[name]
        report["reviewed_ratio_cells"] += sum(1 for p, _ in ratio_cells(payload) if reviewed_reason(item["id"], p))
        report["files"] += 2
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-open", action="store_true", help="exit 0 when the only findings are open items")
    a = ap.parse_args(argv)
    report = check_all()
    summary = {"files_checked": report["files"], "problems": report["problems"],
               "open_items_blocking_release": {k: {"file": v["id"], "owner": v["owner"], "problem": v["problem"],
                                                   "fix": v["fix"]} for k, v in report["open_items"].items()},
               "reviewed_exact_fraction_cells": report["reviewed_ratio_cells"],
               "rules": "counts and printed bounds at least 20, no identifier columns or cells, at most 400 rows, "
                        "no unreviewed exact fraction with an odd denominator above 20"}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if report["problems"]:
        return 1
    if report["open_items"] and not a.allow_open:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

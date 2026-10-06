"""Generate docs/source_data_code_coverage.md, which maps every Source Data row to the code behind its aggregate.

Each Source Data row names the aggregate file it was read from. This tool groups those files by run folder, assigns
each folder to the analysis that wrote it and says whether that analysis's code is in research/. It reads only the
Source Data JSON files and the file names in research/. It never imports or runs research code.

Run without arguments to rewrite the page, or with --check to fail when the committed page is stale, when a Source
Data source has no entry in SOURCES, or when a module named in SOURCES is missing from research/.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "paper" / "source_data"
RESEARCH = ROOT / "research"
PAGE = ROOT / "docs" / "source_data_code_coverage.md"
PLACEHOLDER = "PLACEHOLDER"
PATHLIKE = re.compile(r"^[\w.\-/]+\.(json|md|csv|txt)$")

# Run folder (or file, for files at the project root) -> (analysis, code). code is the module in research/ that ran
# the analysis, RECORD for a text record that is not an analysis output, or None when the code is not yet released.
RECORD = "record"
SOURCES = {
    "BRAN_R7_MODALITY_ATLAS_A3_REPORT_2026-09-25": ("A3 modality atlas", "run_bran_r7_modality_atlas_a3.py"),
    "BRAN_R7_INFORMATION_MATCHED_F1_REPORT_2026-09-25": ("F1 information-matched comparison",
                                                         "run_bran_r7_information_matched_f1.py"),
    "BRAN_R7_COMPLETION_UTILITY_U2_REPORT_2026-09-28": ("U2 completion utility",
                                                        "run_bran_r7_completion_utility_u2.py"),
    "BRAN_R7_QUANTILE_Q4_REPORT_2026-09-28": ("Q4 quantile analysis", "run_bran_r7_quantile_q4.py"),
    "BRAN_R7_INSPIRE_E1_REPORT_2026-09-23": ("INSPIRE E1", "run_bran_r7_inspire_e1.py"),
    "BRAN_HIRID_R7_H6_REPORT_2026-09-25": ("HiRID H6", "run_bran_hirid_r7_h6.py"),
    "BRAN_INTEGRATED_PAPER_2026-09-30_V1/retained_evidence/sources/BRAN_HIRID_R7_H6_REPORT_2026-09-25":
        ("HiRID H6, report copy kept with the integrated paper", "run_bran_hirid_r7_h6.py"),
    "BRAN_HIRID_R7_H5_REPORT_V2_2026-09-25": ("HiRID H5", "run_bran_hirid_r7_h5.py"),
    "BRAN_R7_HEAD_BLOCKS_I1_REPORT_2026-09-25": ("I1 head blocks", "run_bran_r7_head_blocks_i1.py"),
    "BRAN_NAMED_FM_RESULTS_V1": ("Named foundation-model results", "build_bran_named_fm_results_v1.py"),
    "BRAN_MATCHED_SCREENING_V1": ("Matched screening", None),
    "BRAN_R7_TRAINING_EXPOSURE_E4_2026-09-25": ("E4 training exposure", None),
    "BRAN_R7_READOUT_CURVE_L1_REPORT_2026-09-28": ("L1 readout curve", None),
    "BRAN_R7_SELECTIVE_Q5_REPORT_2026-09-28_v2": ("Q5 selective reliability", None),
    "BRAN_R7_SCIENTIFIC_AUDIT_A1_2026-09-23": ("A1 scientific audit", None),
    "validation_results": ("Expanded end point atlas support", None),
    "PATIENT_ATLAS_OFFICIAL_TEST_RESULT_V1.json": ("Predecessor official test result", None),
    "PATIENT_ATLAS_V5_CROSSMODAL_UTILITY_ATTEMPT2.json": ("Predecessor cross-modal utility", None),
    "PATIENT_ATLAS_RAW_SOURCE_AUDIT.json": ("Predecessor raw source audit", None),
    "PATIENT_ATLAS_EXPLORATORY_COHORT_SUMMARY.json": ("Predecessor exploratory cohort summary", None),
    "BRAN_CBC_REFERENCE_RANGES_V1.json": ("Blood count reference ranges", None),
    "BRAN_OVERNIGHT_2026-10-01/tuned_benchmark/runs/execute_attempt1": ("Tuned benchmark", None),
    "BRAN_OVERNIGHT_2026-10-01/native_vs_fm/runs/execute_attempt3": ("Native heads against foundation models", None),
    "BRAN_OVERNIGHT_2026-10-01/eye_blood_completion/EXECUTE_ATTEMPT1": ("Eye to blood completion", None),
    "BRAN_OVERNIGHT_2026-10-01/missingness_baselines/results/attempt1": ("Missingness baselines", None),
    "BRAN_OVERNIGHT_2026-10-01/knhanes_2024_external/execute_attempt2": ("KNHANES 2024 external test", None),
    "BRAN_OVERNIGHT_2026-10-01/knhanes_modalities/results/attempt_01": ("KNHANES modalities", None),
    "BRAN_OVERNIGHT_2026-10-01/discovery_recovery_deficit/outputs/D1R_EXECUTE_ATTEMPT1": ("Recovery deficit", None),
    "BRAN_OVERNIGHT_2026-10-01/discovery_retina_taught_axis/EXECUTE_ATTEMPT3": ("Retina-taught axis", None),
    "BRAN_OVERNIGHT_2026-10-01/discovery_retina_taught_axis/phenoage_amendment1/EXECUTE_ATTEMPT2":
        ("Retina-taught axis, PhenoAge amendment", None),
    "BRAN_OVERNIGHT_2026-10-01/undiagnosed_detection/results/ATTEMPT1": ("Undiagnosed detection", None),
    "BRAN_NATMED_PUSH_2026-10-02/fig4_low_hb_clinical_use/EXECUTE_ATTEMPT1": ("Low haemoglobin clinical use", None),
    "BRAN_NATMED_PUSH_2026-10-02/eye_loss_diagnostic/runs/execute_attempt1": ("Eye information loss diagnostic", None),
    "BRAN_NATMED_PUSH_2026-10-02/r8_retinal_route/runs/execute_attempt1": ("R8 retinal route", None),
    "BRAN_NATMED_PUSH_2026-10-02/fig6_discovery_expansion/EXECUTE_ATTEMPT1": ("Discovery expansion, attempt 1", None),
    "BRAN_NATMED_PUSH_2026-10-02/fig6_discovery_expansion/EXECUTE_ATTEMPT2": ("Discovery expansion, attempt 2", None),
    "BRAN_NATMED_PUSH_2026-10-02/subgroup_discovery/EXECUTE_ATTEMPT1": ("Subgroup discovery", None),
    "BRAN_NATMED_PHASE2_2026-10-05/official_test_r7/score/runs/execute_attempt1": ("Official test scoring of R7", None),
    "BRAN_NATMED_PHASE2_2026-10-05/death_replication_9904/EXECUTE_ATTEMPT1": ("Death replication", None),
    "BRAN_NATMED_PHASE2_2026-10-05/knhanes_2024_grip_replication/EXECUTE_ATTEMPT2":
        ("KNHANES 2024 grip replication", None),
    "BRAN_INTEGRATED_PAPER_2026-09-30_V1/retained_evidence/sources/BRAN_R7_EIGHT_HOUR_RESEARCH_ADDENDUM_2026-09-28_v3":
        ("R7 model card of the integrated paper", RECORD),
    "BRAN_MASTER_OUTLINE_2026-09-30_rev2.md": ("Manuscript outline", RECORD),
    "EXTERNAL_EYE_TOWER_CONTRACT.json": ("Retinal feature extractor contract", RECORD),
    "BRAN_FIGURES_2026-10-01/locked": ("Fig. 6 export manifest, a cross-reference", RECORD),
}
RANK = {"research": 0, "record": 1, "missing": 2}


def key_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else path


def category(code) -> str:
    return "research" if code not in (None, RECORD) else ("record" if code == RECORD else "missing")


def row_sources(panel: dict) -> list[list[str]]:
    cols = panel["columns"]
    idx = [i for i, c in enumerate(cols) if c == "source" or c.endswith("_source")]
    out = []
    for r in panel["rows"]:
        found = []
        for i in idx:
            v = r[i]
            if isinstance(v, str):
                found += [p.strip() for p in re.split(r"\s*[;|,]\s*|\s+and\s+", v) if PATHLIKE.match(p.strip())]
        out.append(found)
    return out


def collect() -> tuple[list[dict], dict, list[str]]:
    manifest = json.loads((SOURCE_DATA / "SOURCE_DATA_MANIFEST.json").read_text(encoding="utf-8"))
    items, per_source, problems = [], defaultdict(lambda: {"rows": 0, "items": set()}), []
    for entry in manifest["files"]:
        payload = json.loads((SOURCE_DATA / entry["json"]["path"]).read_text(encoding="utf-8"))
        counts = {"research": 0, "record": 0, "missing": 0}
        for panel_key, panel in payload["panels"].items():
            for i, sources in enumerate(row_sources(panel)):
                if not sources:
                    problems.append(f"{entry['id']} panel {panel_key} row {i} names no source file")
                    continue
                worst = "research"
                for path in sources:
                    key = key_of(path)
                    if key not in SOURCES:
                        problems.append(f"{entry['id']} panel {panel_key} row {i} source {key} has no entry in SOURCES")
                        worst = "missing"
                        continue
                    per_source[key]["rows"] += 1
                    per_source[key]["items"].add(entry["display"])
                    cat = category(SOURCES[key][1])
                    worst = max(worst, cat, key=RANK.get)
                counts[worst] += 1
        items.append({"display": entry["display"], **counts, "rows": sum(counts.values())})
    present = {p.name for p in RESEARCH.glob("*.py")}
    for key, (_, code) in SOURCES.items():
        if code not in (None, RECORD) and code not in present:
            problems.append(f"{code} named for {key} is not in research/")
    return items, per_source, problems


def build() -> tuple[str, list[str]]:
    items, per_source, problems = collect()
    total = sum(i["rows"] for i in items)
    covered = sum(i["research"] for i in items)
    records = sum(i["record"] for i in items)
    missing = sum(i["missing"] for i in items)
    order = {d: n for n, d in enumerate(i["display"] for i in items)}
    lines = [
        "# Source Data and analysis code",
        "",
        "This page is generated by `python tools/source_data_coverage.py` from the Source Data in `paper/source_data` "
        "and the file names in `research/`. Do not edit it by hand. `python tools/source_data_coverage.py --check` "
        "fails when it is stale or when a Source Data source has no entry.",
        "",
        "Every Source Data row names the aggregate file its values were read from. The tables below group those files "
        "by the run that wrote them and say whether the code of that run is in `research/`. A row read from two files "
        "counts under the less complete of the two.",
        "",
        f"Of {total:,} Source Data rows, {covered:,} come from analyses whose code is in `research/`, {records:,} "
        f"quote text records that no analysis computed, and {missing:,} come from analyses whose code is not yet in "
        f"this repository. {PLACEHOLDER} for a hash-locked snapshot of the runner code of every analysis marked "
        f"{PLACEHOLDER} below, with its entrypoint. Until then the Analyses level of "
        "[the research reproduction guide](research_reproduction.md) covers only the analyses with code here.",
        "",
        "Where the code is here, it is the code that ran the analysis. The report step that assembled its results "
        "into the aggregate table read by the figure builder is not included, except for the named foundation-model "
        "results. The code of each analysis marked PLACEHOLDER is kept in the private workspace.",
        "",
        "## By display item",
        "",
        "| Item | Rows | Code in `research/` | Text record | Code not yet released |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i in items:
        lines.append(f"| {i['display']} | {i['rows']} | {i['research']} | {i['record']} | {i['missing']} |")
    lines += ["", "## By source", "",
              "| Source folder or file | Analysis | Items | Rows | Code in `research/` |",
              "| --- | --- | --- | --- | --- |"]
    for key in sorted(per_source, key=lambda k: (RANK[category(SOURCES[k][1])], k)):
        name, code = SOURCES[key]
        shown = f"`{code}`" if category(code) == "research" else (
            "Text record, not an analysis" if code == RECORD else PLACEHOLDER)
        its = ", ".join(sorted(per_source[key]["items"], key=order.get))
        lines.append(f"| `{key}` | {name} | {its} | {per_source[key]['rows']} | {shown} |")
    unused = sorted(set(SOURCES) - set(per_source))
    if unused:
        problems.append("SOURCES entries no Source Data row uses: " + ", ".join(unused))
    return "\n".join(lines) + "\n", problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail when the committed page is stale")
    a = ap.parse_args(argv)
    text, problems = build()
    for p in problems:
        print(p, file=sys.stderr)
    if a.check:
        current = PAGE.read_text(encoding="utf-8") if PAGE.is_file() else ""
        if current != text:
            print(f"{PAGE.relative_to(ROOT)} is stale, run python tools/source_data_coverage.py", file=sys.stderr)
            return 1
        return 1 if problems else 0
    PAGE.write_text(text, encoding="utf-8")
    print("wrote", PAGE.relative_to(ROOT))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

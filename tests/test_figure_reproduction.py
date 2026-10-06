"""Checks of the paper figure layer: Source Data identity, disclosure, public code and byte-identical rebuilds.

The rebuild tests draw every figure from its Source Data and compare the PNG, PDF and JPEG hashes with
paper/figures/REFERENCE_OUTPUTS.json. They run when plotly and Kaleido are installed and the environment matches the
reference environment, or when BRAN_FIGURE_REBUILD=1 is set, and are skipped otherwise with the reason.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "paper" / "source_data"
FIGDIR = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT))
from paper.figures import build_all, disclosure  # noqa: E402

ABSOLUTE = re.compile(r"(?<![\w.~-])(?:/Users/|/home/|/Volumes/|/private/|/opt/|/System/|/Library/|/Applications/|"
                      r"/tmp/|/var/folders/|[A-Za-z]:\\{1,2}(?:Users|Documents)\\{1,2})[A-Za-z0-9_]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference() -> dict:
    return json.loads((FIGDIR / "REFERENCE_OUTPUTS.json").read_text(encoding="utf-8"))


class SourceDataTests(unittest.TestCase):
    def test_manifest_lists_every_file_with_its_hash_and_display_number(self):
        manifest = json.loads((SOURCE_DATA / "SOURCE_DATA_MANIFEST.json").read_text(encoding="utf-8"))
        listed = set()
        for item in manifest["files"]:
            self.assertRegex(item["display"], r"^(Fig\. [1-5]|Extended Data Fig\. \d+|Extended Data Table \d+)$")
            for kind in ("json", "xlsx"):
                path = SOURCE_DATA / item[kind]["path"]
                self.assertEqual(sha256(path), item[kind]["sha256"], path.name)
                self.assertEqual(path.stat().st_size, item[kind]["bytes"], path.name)
                listed.add(path.name)
        present = {p.name for p in SOURCE_DATA.iterdir()} - {"SOURCE_DATA_MANIFEST.json", "README.md"}
        self.assertEqual(present, listed)
        self.assertEqual({i["id"] for i in manifest["files"] if i["status"] == "final"}, set(build_all.FIGURES))

    def test_display_numbers_match_the_drawing_registry(self):
        manifest = json.loads((SOURCE_DATA / "SOURCE_DATA_MANIFEST.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            payload = json.loads((SOURCE_DATA / item["json"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["display"], item["display"], item["id"])
            if item["id"] in build_all.FIGURES:
                self.assertEqual(build_all.display(item["id"]), item["display"])

    def test_disclosure_rules_hold_for_every_file(self):
        report = disclosure.check_all()
        self.assertEqual(report["files"], 2 * len(json.loads(
            (SOURCE_DATA / "SOURCE_DATA_MANIFEST.json").read_text(encoding="utf-8"))["files"]))
        self.assertEqual(report["problems"], [])

    def test_disclosure_check_catches_small_cells_identifiers_and_long_tables(self):
        panel = {"title": "t", "columns": ["label", "n_floor20", "people", "subject_id", "source", "source_sha256"],
                 "count_columns": ["n_floor20"], "floored_columns": ["n_floor20"],
                 "rows": [["a", 30, 25, "x", "s", "0" * 64], ["b", 40, 12, "y", "s", "0" * 64]]}
        payload = {"legend": "Shown for n = 7 people and ≥15 deaths.", "panels": {"a": panel}}
        problems = disclosure.payload_problems("synthetic", payload)
        text = "\n".join(problems)
        self.assertIn("count people=12 below 20", text)
        self.assertIn("floored count n_floor20=30", text)
        self.assertIn("subject_id", text)
        self.assertIn("'n = 7'", text)
        self.assertIn("'≥15'", text)
        long_panel = dict(panel, rows=[["a", 40, 40, "x", "s", "0" * 64]] * 401)
        self.assertTrue(any("401 rows" in p for p in disclosure.payload_problems("synthetic", {"panels": {"a": long_panel}})))

    def test_source_data_holds_no_absolute_path(self):
        for path in sorted(SOURCE_DATA.glob("*_source_data.json")):
            match = ABSOLUTE.search(path.read_text(encoding="utf-8"))
            self.assertIsNone(match, f"{path.name} {match.group(0) if match else ''}")


class FigureCodeTests(unittest.TestCase):
    def test_public_code_holds_no_absolute_or_private_path(self):
        for path in sorted(FIGDIR.glob("*.py")):
            match = ABSOLUTE.search(path.read_text(encoding="utf-8"))
            self.assertIsNone(match, f"{path.name} {match.group(0) if match else ''}")

    def test_reference_covers_every_figure_and_the_current_code(self):
        ref = reference()
        self.assertEqual(set(ref["figures"]), set(build_all.FIGURES))
        self.assertEqual(ref["code"], build_all.code_hashes(), "drawing code changed, rerun --write-reference")
        self.assertEqual(ref["source_data_manifest_sha256"], sha256(SOURCE_DATA / "SOURCE_DATA_MANIFEST.json"))
        for fid, entry in ref["figures"].items():
            self.assertIn(entry["status"], {"reproduced", "png_reproduced_other_files_differ",
                                            "differs_from_submission_export", "blocked"}, fid)
            if entry["status"] != "blocked":
                self.assertEqual(entry["files_opened_in_repository"], [f"paper/source_data/{fid}_source_data.json"])

    def test_every_item_not_reproduced_is_documented(self):
        readme = (ROOT / "paper" / "README.md").read_text(encoding="utf-8")
        for fid, entry in reference()["figures"].items():
            if entry["status"] != "reproduced":
                self.assertIn(entry["display"], readme, f"{fid} is {entry['status']} but not explained")

    def test_legends_file_matches_the_source_data(self):
        self.assertEqual((FIGDIR / "LEGENDS.md").read_text(encoding="utf-8"),
                         build_all.legends_text(list(build_all.FIGURES)))

    def test_drawing_step_cannot_read_other_repository_files(self):
        code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "from paper.figures import build_all as b; "
                "b.guard_reads({b.SOURCE_DATA / 'fig1_source_data.json'}, []); "
                "(b.SOURCE_DATA / 'fig1_source_data.json').read_bytes(); "
                "(b.REPO / 'README.md').read_text()")
        result = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("may not read", result.stderr)
        self.assertIn("README.md", result.stderr)


def rebuild_skip_reason() -> str | None:
    for module in ("plotly", "kaleido", "PIL"):
        if importlib.util.find_spec(module) is None:
            return f"{module} is not installed, install paper/figures/requirements-figures.txt"
    if os.environ.get("BRAN_FIGURE_REBUILD") == "1":
        return None
    diff = build_all.environment_differences(build_all.environment(), reference()["environment"])
    if diff:
        return "not the reference environment (" + "; ".join(diff)[:300] + "), set BRAN_FIGURE_REBUILD=1 to force"
    return None


class FigureRebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = rebuild_skip_reason()
        if reason:
            raise unittest.SkipTest(reason)

    def test_each_figure_rebuilds_byte_for_byte_from_source_data(self):
        import tempfile
        ref = reference()
        with tempfile.TemporaryDirectory(prefix="bran_figure_test_") as out:
            for fid, entry in ref["figures"].items():
                with self.subTest(figure=entry["display"]):
                    result = build_all.run_child(fid, Path(out) / fid)
                    if entry["status"] == "blocked":
                        self.assertIn("error", result, f"{fid} builds now, refresh the reference")
                        continue
                    self.assertNotIn("error", result, result.get("error"))
                    self.assertEqual(build_all.compare(result, entry), [])
                    self.assertEqual(result["files_opened_in_repository"],
                                     [f"paper/source_data/{fid}_source_data.json"])


if __name__ == "__main__":
    unittest.main()

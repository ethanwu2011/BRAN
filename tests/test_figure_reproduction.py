"""Checks of the paper figure layer: Source Data identity, disclosure, public code and byte-identical rebuilds.

The rebuild tests draw every figure from its Source Data and compare the PNG, PDF and JPEG hashes with
paper/figures/REFERENCE_OUTPUTS.json. They run when plotly and Kaleido are installed and the environment matches the
reference environment, or when BRAN_FIGURE_REBUILD=1 is set, and are skipped otherwise with the reason.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
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


# Wording of each reference status in the status table of paper/README.md.
STATUS_TEXT = {"reproduced": "Reproduced", "blocked": "Blocked",
               "differs_from_submission_export": "Differs from the submission export",
               "png_reproduced_other_files_differ": "PNG reproduced, PDF differs"}


def status_table(text: str) -> list[tuple[str, str]]:
    """(item, status) for each row of the table under the Item, Status and Detail header."""
    lines = text.splitlines()
    start = lines.index("| Item | Status | Detail |")
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1]))
    return rows


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
        statuses = {i["id"]: i["status"] for i in manifest["files"]}
        self.assertEqual(set(statuses), set(build_all.FIGURES))
        self.assertLessEqual(set(statuses.values()), {"final", "pending"})
        blocked = {f for f, e in reference()["figures"].items() if e["status"] == "blocked"}
        self.assertEqual({f for f, v in statuses.items() if v == "pending"}, blocked,
                         "items pending in the Source Data manifest must be the items blocked in the reference")
        for item in manifest["files"]:
            if item["status"] == "pending":
                self.assertTrue(item.get("status_reason"), item["id"])

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

    def test_open_disclosure_items_are_present_and_documented(self):
        report = disclosure.check_all()
        readme = (SOURCE_DATA / "README.md").read_text(encoding="utf-8")
        for name, item in disclosure.OPEN_ITEMS.items():
            self.assertIn(name, report["open_items"], f"{name} is fixed, remove it from OPEN_ITEMS and the README")
            self.assertIn(f"`{name}`", readme, f"{name} is not explained in paper/source_data/README.md")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(disclosure.main([]), 2 if report["open_items"] else 0)
            self.assertEqual(disclosure.main(["--allow-open"]), 0)

    def test_disclosure_check_catches_unreviewed_fractions_and_draw_counts(self):
        panel = {"title": "t", "columns": ["label", "value", "p_one_sided"],
                 "rows": [["a", 5 / 1923, 0.5], ["b", 0.25, 7 / 1001]]}
        payload = {"panels": {"b strip": panel}, "draw": {"n_people": 12, "share": 40 / 1923, "n_floor": 1900}}
        text = "\n".join(disclosure.payload_problems("synthetic", payload))
        self.assertIn("panels/b strip/rows/0/value is an exact fraction with denominator 1923", text)
        self.assertIn("draw/share is an exact fraction", text)
        self.assertIn("draw n_people=12 below 20", text)
        self.assertNotIn("n_floor", text)
        reviewed = disclosure.payload_problems("fig2", payload)
        self.assertFalse(any("p_one_sided" in p for p in reviewed), "a reviewed P value cell must not be flagged")
        self.assertTrue(any("rows/0/value" in p for p in reviewed))

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

    def test_status_table_lists_every_item_with_its_reference_status(self):
        rows = status_table((ROOT / "paper" / "README.md").read_text(encoding="utf-8"))
        items = [item for item, _ in rows]
        self.assertEqual(len(items), len(set(items)), "an item is listed twice")
        ref = reference()["figures"]
        self.assertEqual(set(items), {e["display"] for e in ref.values()})
        table = dict(rows)
        for fid, entry in ref.items():
            self.assertEqual(table[entry["display"]], STATUS_TEXT[entry["status"]],
                             f"{fid} is {entry['status']} in the reference")

    def test_summary_never_reports_a_match_after_a_failed_build(self):
        page = {"size_px": [10, 10], "png_sha256": "a", "pdf_sha256": "b"}
        ref = {"figures": {"fig1": {"status": "reproduced", "pages": {"fig1": page}},
                           "fig3": {"status": "blocked", "pages": {}}}}
        results = {"fig1": {"id": "fig1", "pages": {"fig1": dict(page)}}, "fig3": {"id": "fig3", "error": "stops"}}
        code, lines = build_all.summarize(["fig1", "fig3"], results, ref, [], [], False, True)
        self.assertEqual(code, 0)
        self.assertTrue(lines[-1].startswith("Incomplete. 1 of 2 items were rebuilt"))
        self.assertFalse(any("match" in line for line in lines))
        code, _ = build_all.summarize(["fig1", "fig3"], results, ref, [], [], False, False)
        self.assertEqual(code, 2)
        ref["figures"]["fig3"]["status"] = "reproduced"
        for portable in (False, True):
            code, lines = build_all.summarize(["fig1", "fig3"], results, ref, ["chrome differs"], [], portable, True)
            self.assertEqual(code, 1, "a build failure of an item that is not blocked must fail")
            self.assertIn("Build failed: Fig. 3", lines)
        results["fig3"] = {"id": "fig3", "pages": {"fig1": dict(page)}}
        ref["figures"]["fig3"] = {"status": "reproduced", "pages": {"fig1": page}}
        code, lines = build_all.summarize(["fig1", "fig3"], results, ref, [], [], False, False)
        self.assertEqual((code, lines[-1]), (0, "All 2 items were rebuilt and match the reference."))

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


class PortableDrawingTests(unittest.TestCase):
    def test_extended_data_table_1_draws_without_helvetica_neue(self):
        """The table is drawn in Arial and its text check must not need HelveticaNeue.ttc, which Linux lacks."""
        if importlib.util.find_spec("plotly") is None or importlib.util.find_spec("PIL") is None:
            self.skipTest("plotly or Pillow is not installed")
        code = ("import json, os, sys; sys.path.insert(0, sys.argv[1]); import bran_plotly_style as st\n"
                "if not os.path.isfile(st.NM_FONT_FILES[False]): print('SKIP no Arial'); sys.exit(0)\n"
                "import ed_canvas as E; E.FONT_FILE = 'NoSuchFont.ttc'\n"
                "import draw_ed_table1 as d\n"
                "pages = d.render(json.loads(open(sys.argv[2], encoding='utf-8').read()))\n"
                "print('DRAWN', len(pages))")
        result = subprocess.run([sys.executable, "-c", code, str(FIGDIR),
                                 str(SOURCE_DATA / "ed_table1_source_data.json")],
                                capture_output=True, text=True, cwd=ROOT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if "SKIP" in result.stdout:
            self.skipTest("no Arial font file found, set BRAN_FONT_DIR")
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertIn("DRAWN 1", result.stdout)


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

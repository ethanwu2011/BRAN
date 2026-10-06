"""The generated map from Source Data rows to analysis code must be current and complete."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import source_data_coverage  # noqa: E402


class SourceDataCoverageTests(unittest.TestCase):
    def test_committed_page_is_current_and_every_source_is_mapped(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "source_data_coverage.py"), "--check"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rows_are_counted_once_under_the_less_complete_source(self):
        items, _, problems = source_data_coverage.collect()
        self.assertEqual(problems, [])
        for item in items:
            self.assertEqual(item["rows"], item["research"] + item["record"] + item["missing"], item["display"])

    def test_named_research_modules_exist(self):
        present = {p.name for p in (ROOT / "research").glob("*.py")}
        for key, (_, code) in source_data_coverage.SOURCES.items():
            if code not in (None, source_data_coverage.RECORD):
                self.assertIn(code, present, key)


if __name__ == "__main__":
    unittest.main()

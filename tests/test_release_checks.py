"""Checks of scripts/verify_release.py on generated files and on this tree."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_release  # noqa: E402


class ReleaseCheckTests(unittest.TestCase):
    def check_in(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            saved = verify_release.ROOT
            verify_release.ROOT = root
            try:
                historical: dict[str, int] = {}
                findings = verify_release.check_files(sorted(files), historical)
            finally:
                verify_release.ROOT = saved
        return findings, historical

    def test_absolute_home_path_outside_research_is_rejected(self):
        home = "/" + "Users" + "/someone/data/table.txt"
        findings, historical = self.check_in({"docs/note.md": f"Read {home} first.\n"})
        self.assertEqual(findings, [{"file": "docs/note.md", "reason": "absolute_local_path"}])
        self.assertEqual(historical, {})

    def test_absolute_path_inside_research_is_reported_not_rejected(self):
        volume = "/" + "Volumes" + "/Drive/source"
        findings, historical = self.check_in({"research/old_runner.py": f'ROOT = "{volume}"\n'})
        self.assertEqual(findings, [])
        self.assertEqual(historical, {"research/old_runner.py": 1})

    def test_relative_and_system_paths_are_not_flagged(self):
        text = "paper/source_data/fig1.json and /usr/share/fonts and docs/Users/guide.md\n"
        findings, historical = self.check_in({"docs/ok.md": text})
        self.assertEqual(findings, [])
        self.assertEqual(historical, {})

    def test_tree_passes_and_reports_the_documented_historical_paths(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_release.py")],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["historical_absolute_paths"]["files"], 26)
        self.assertEqual(report["historical_absolute_paths"]["occurrences"], 59)
        documented = (ROOT / "docs" / "privacy_and_access.md").read_text(encoding="utf-8")
        for name in report["historical_absolute_paths"]["per_file"]:
            self.assertIn(f"`{name}`", documented)


if __name__ == "__main__":
    unittest.main()

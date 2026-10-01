"""Verify research source identity without importing or executing study runners."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ResearchSnapshotTests(unittest.TestCase):
    def test_verifier_needs_no_original_workspace(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "research_snapshot.py"), "verify"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["module_count"], 425)

    def test_snapshot_contains_only_bound_parseable_python_modules(self):
        directory = ROOT / "research"
        manifest = json.loads((directory / "snapshot_manifest.json").read_text())
        self.assertEqual(manifest["schema"], "bran-research-code-snapshot-v1")
        expected = set(manifest["modules"])
        self.assertEqual(expected, {path.name for path in directory.glob("*.py")})
        for name, receipt in manifest["modules"].items():
            content = (directory / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), receipt["sha256"], name)
            ast.parse(content, filename=name)


if __name__ == "__main__":
    unittest.main()

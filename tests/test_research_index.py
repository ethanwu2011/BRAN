"""The generated research index must match the snapshot manifest."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_index  # noqa: E402


class ResearchIndexTests(unittest.TestCase):
    def test_committed_index_is_current(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "research_index.py"), "--check"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_family_strips_version_attempt_and_stage_tokens(self):
        self.assertEqual(research_index.family("soft_patient_atlas_v6_2.py"), "soft_patient_atlas")
        self.assertEqual(research_index.family("bran_named_fm_extraction_v1_attempt2.py"),
                         "bran_named_fm_extraction")
        self.assertEqual(research_index.family("run_bran_hirid_r7_h6.py"), "run_bran_hirid")
        self.assertEqual(research_index.family("bran_cbc_mcl_units_v3.py"), "bran_cbc_mcl_units")

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


RELEASE = Path(__file__).resolve().parents[1]
SCHEMA_PATH = RELEASE / "schemas" / "clinical_fields.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ClinicalFieldReleaseSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_container_width_and_ordered_field_blocks(self):
        fields = self.schema["clinical_fields"]
        self.assertEqual(len(fields), 59)
        self.assertEqual([row["index"] for row in fields], list(range(59)))
        self.assertEqual(sum(row["block"] == "continuous" for row in fields), 48)
        self.assertEqual(sum(row["block"] == "history" for row in fields), 11)
        self.assertEqual(len({row["name"] for row in fields}), 59)

    def test_eligible_continuous_positions_and_disabled_history(self):
        container = self.schema["container"]
        eligible = [row["index"] for row in self.schema["clinical_fields"]
                    if row["block"] == "continuous" and row["r7_eligible"]]
        self.assertEqual(container["clinical_values_shape"], ["batch", 59])
        self.assertEqual(len(eligible), 43)
        self.assertEqual(eligible, container["eligible_continuous_indices"])
        self.assertEqual([row["index"] for row in self.schema["clinical_fields"]
                          if row["block"] == "history"], list(range(48, 59)))
        self.assertTrue(container["history_slots_disabled"])

    def test_cbc_output_and_canonical_slot_mapping(self):
        targets = self.schema["cbc_targets"]
        self.assertEqual(len(targets), 9)
        self.assertEqual([target["output_index"] for target in targets], list(range(9)))
        self.assertEqual([target["continuous_index"] for target in targets],
                         [17, 19, 22, 23, 24, 26, 29, 30, 37])
        self.assertEqual([target["field"] for target in targets],
                         ["hct", "hemoglobin", "mch", "mchc", "mcv",
                          "plt", "rbc", "rdw", "wbc"])
        self.assertTrue(all(target["source_admission"] is False for target in targets))
        self.assertEqual(self.schema["clinical_fields"][8]["name"], "c_peptide")
        self.assertEqual(self.schema["clinical_fields"][8]["unit"], None)

    def test_age_and_retinal_shapes(self):
        age = self.schema["age"]
        retinal = self.schema["retinal"]
        self.assertEqual(age["shape"], ["batch", 7])
        self.assertEqual(len(age["features"]), 7)
        self.assertEqual(age["kinds"],
                         ["reported", "interval", "right_censored", "unknown"])
        self.assertEqual(retinal["shape"], ["batch", "retinal_count", 384])
        self.assertEqual(retinal["visibility_mask_shape"], ["batch", "retinal_count"])

    def test_bundled_hashes_recheck_and_external_hashes_are_labeled(self):
        provenance = self.schema["provenance"]
        self.assertEqual(len(provenance), 6)
        for item in provenance:
            self.assertRegex(item["sha256"], SHA256)
            if item["distribution_status"] == "bundled":
                path = RELEASE / item["release_path"]
                self.assertTrue(path.is_file(), item["release_path"])
                value = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(value, item["sha256"], item["release_path"])
                self.assertEqual(item["hash_status"], "rechecked_against_bundled_copy")
            else:
                self.assertEqual(item["distribution_status"], "reference_not_bundled")
                self.assertNotIn("release_path", item)
                self.assertEqual(item["hash_status"], "verified_in_original_workspace_only")
                self.assertFalse(Path(item["path"]).is_absolute())
                self.assertEqual(Path(item["path"]).name, item["path"])

    def test_contract_metadata_is_not_misidentified_as_json_schema(self):
        self.assertEqual(self.schema["schema_version"], "bran-clinical-input-contract-v1")
        self.assertNotIn("$schema", self.schema)

    def test_checkpoint_binding_limit_is_explicit(self):
        binding = self.schema["checkpoint_binding"]
        self.assertEqual(binding["status"], "not_recreated_from_allowlisted_code")
        self.assertTrue(binding["caller_supplied_eligible_indices_are_validated_but_not_defined_in_model_code"])
        self.assertTrue(binding["registry_order_is_documented_but_not_loaded_from_checkpoint_here"])
        self.assertTrue(binding["retinal_dimension_is_documented_but_not_rechecked_against_checkpoint_here"])


if __name__ == "__main__":
    unittest.main()

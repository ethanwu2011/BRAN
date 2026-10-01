from __future__ import annotations

import copy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bran_multisource_age_v2 import AgeBatch, normalize_age
from bran_multisource_model_v2 import erase_cbc_for_completion
from bran_robust_clinical_r7 import BRANRobustClinicalR7
import bran_r7_release.inference as inference
from bran_r7_release.inference import (
    infer,
    load_checkpoint,
    make_field_map_binding,
    make_transform_binding,
    save_checkpoint,
)


ELIGIBLE = tuple(range(43))
CBC = tuple(range(9))
FEATURE_NAMES = tuple(f"synthetic_feature_{index:02d}" for index in range(59))


def synthetic_transform():
    eligible = torch.zeros(59, dtype=torch.bool)
    eligible[list(ELIGIBLE)] = True
    return SimpleNamespace(
        clinical_median=torch.zeros(59, dtype=torch.float64),
        clinical_iqr=torch.ones(59, dtype=torch.float64),
        retinal_mean=torch.zeros(384, dtype=torch.float64),
        retinal_scale=torch.ones(384, dtype=torch.float64),
        age_mean=50.0,
        age_scale=15.0,
        eligible=eligible,
        heldout_fold=0,
        fold_identity_sha256="a" * 64,
        training_indices_sha256="b" * 64,
    )


def unknown_age(batch_size):
    raw = AgeBatch(
        value=torch.full((batch_size,), float("nan")),
        lower=torch.full((batch_size,), float("nan")),
        upper=torch.full((batch_size,), float("nan")),
        kind=torch.full((batch_size,), 3, dtype=torch.long),
    )
    return normalize_age(raw, mean=50.0, scale=15.0)


class ReleaseInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(101)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls.tempdir.name) / "synthetic_r7.pt"
        cls.model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC)
        cls.field_map = make_field_map_binding(FEATURE_NAMES, ELIGIBLE, CBC)
        cls.transform = synthetic_transform()
        cls.transform_binding = make_transform_binding(cls.transform)
        cls.binding = save_checkpoint(
            cls.checkpoint_path,
            cls.model,
            field_map_binding=cls.field_map,
            input_transform=cls.transform,
        )
        cls.loaded = load_checkpoint(
            cls.checkpoint_path,
            expected_binding=cls.binding,
            expected_field_map_binding=cls.field_map,
            expected_transform_binding=cls.transform_binding,
            input_transform=cls.transform,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def make_inputs(self, batch=3):
        torch.manual_seed(103)
        return (
            torch.randn(batch, 59),
            torch.ones(batch, 59, dtype=torch.bool),
            torch.randn(batch, 2, 384),
            torch.ones(batch, 2, dtype=torch.bool),
            unknown_age(batch),
        )

    def test_safe_checkpoint_round_trip_native_heads_and_original_units(self):
        inputs = self.make_inputs()
        result = infer(self.loaded, *inputs, expected_binding=self.binding)
        self.assertEqual(tuple(result.state_mean.shape), (3, 192))
        self.assertEqual(tuple(result.screening_probability.shape), (3, 26))
        self.assertEqual(tuple(result.cbc_standardized.shape), (3, 9))
        self.assertEqual(tuple(result.cbc_original_units.shape), (3, 9))
        self.assertTrue(torch.isfinite(result.screening_probability).all())
        self.assertTrue(torch.isfinite(result.cbc_standardized).all())
        torch.testing.assert_close(result.cbc_original_units, result.cbc_standardized)

    def test_original_units_are_unscaled_only_from_compatible_transform(self):
        transform = copy.deepcopy(self.transform)
        transform.clinical_median[0] = 20.0
        transform.clinical_iqr[0] = 2.5
        field_map = self.field_map
        transform_binding = make_transform_binding(transform)
        model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bound.pt"
            binding = save_checkpoint(path, model, field_map_binding=field_map, input_transform=transform)
            loaded = load_checkpoint(
                path,
                expected_binding=binding,
                expected_field_map_binding=field_map,
                expected_transform_binding=transform_binding,
                input_transform=transform,
            )
            result = infer(loaded, *self.make_inputs(), expected_binding=binding)
        medians = transform.clinical_median[list(CBC)].to(torch.float32)
        scales = transform.clinical_iqr[list(CBC)].to(torch.float32)
        expected = result.cbc_standardized * scales[None, :] + medians[None, :]
        torch.testing.assert_close(result.cbc_original_units, expected)

    def test_checkpoint_without_transform_returns_standardized_values_only(self):
        loaded = load_checkpoint(
            self.checkpoint_path,
            expected_binding=self.binding,
            expected_field_map_binding=self.field_map,
            expected_transform_binding=self.transform_binding,
        )
        result = infer(loaded, *self.make_inputs(), expected_binding=self.binding)
        self.assertIsNone(result.cbc_original_units)
        self.assertTrue(torch.isfinite(result.cbc_standardized).all())

    def test_completion_erases_targets_before_encoder_and_reports_support_mask(self):
        inputs = self.make_inputs()
        original_mask = inputs[1][:, list(CBC)].clone()
        completed = infer(self.loaded, *inputs, expected_binding=self.binding,
                          erase_cbc_targets=True)
        erased_values, erased_mask = erase_cbc_for_completion(inputs[0], inputs[1], CBC)
        expected = infer(self.loaded, erased_values, erased_mask, *inputs[2:],
                         expected_binding=self.binding)
        self.assertIsNotNone(completed.completion_target_mask)
        self.assertTrue(torch.equal(completed.completion_target_mask, original_mask))
        self.assertEqual(tuple(completed.cbc_standardized.shape), (3, 9))
        torch.testing.assert_close(completed.state_mean, expected.state_mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(completed.cbc_standardized, expected.cbc_standardized, rtol=0.0, atol=0.0)

    def test_all_missing_physiology_abstains_with_age_available(self):
        clinical = torch.full((2, 59), float("nan"))
        clinical_observed = torch.zeros((2, 59), dtype=torch.bool)
        retinal = torch.full((2, 1, 384), float("nan"))
        retinal_visible = torch.zeros((2, 1), dtype=torch.bool)
        result = infer(self.loaded, clinical, clinical_observed, retinal,
                       retinal_visible, unknown_age(2), expected_binding=self.binding)
        self.assertTrue(bool(result.abstained.all()))
        self.assertTrue(torch.isnan(result.screening_probability).all())
        self.assertTrue(torch.isnan(result.cbc_standardized).all())
        self.assertTrue(torch.isnan(result.cbc_original_units).all())

    def test_nonfinite_declared_observation_is_rejected(self):
        clinical, observed, retinal, retinal_mask, age = self.make_inputs()
        clinical[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            infer(self.loaded, clinical, observed, retinal, retinal_mask, age,
                  expected_binding=self.binding)

    def test_nonfinite_masked_payload_is_ignored(self):
        clinical, observed, retinal, retinal_mask, age = self.make_inputs()
        clinical[0, 0] = float("nan")
        observed[0, 0] = False
        retinal[0, 0, :] = float("nan")
        retinal_mask[0, 0] = False
        result = infer(self.loaded, clinical, observed, retinal, retinal_mask, age,
                       expected_binding=self.binding)
        self.assertFalse(bool(result.abstained.any()))
        self.assertTrue(torch.isfinite(result.screening_probability).all())

    def test_postload_model_mutation_breaks_bound_checkpoint(self):
        loaded = copy.deepcopy(self.loaded)
        with torch.no_grad():
            loaded.model.cbc_joint_head.bias[0] += 1.0
        with self.assertRaises(ValueError):
            infer(loaded, *self.make_inputs(), expected_binding=self.binding)

    def test_field_map_and_transform_bindings_are_required_to_match(self):
        wrong_map = make_field_map_binding(
            ("different_name",) + FEATURE_NAMES[1:], ELIGIBLE, CBC
        )
        with self.assertRaises(ValueError):
            load_checkpoint(
                self.checkpoint_path,
                expected_binding=self.binding,
                expected_field_map_binding=wrong_map,
                expected_transform_binding=self.transform_binding,
                input_transform=self.transform,
            )
        wrong_transform = copy.deepcopy(self.transform)
        wrong_transform.clinical_iqr[0] = 3.0
        with self.assertRaises(ValueError):
            load_checkpoint(
                self.checkpoint_path,
                expected_binding=self.binding,
                expected_field_map_binding=self.field_map,
                expected_transform_binding=self.transform_binding,
                input_transform=wrong_transform,
            )

    def test_state_dict_tampering_fails_caller_pinned_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "tampered.pt"
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            payload["state_dict"]["screening_joint_head.bias"][0] += 1.0
            torch.save(payload, tampered)
            with self.assertRaises(ValueError):
                load_checkpoint(
                    tampered,
                    expected_binding=self.binding,
                    expected_field_map_binding=self.field_map,
                    expected_transform_binding=self.transform_binding,
                    input_transform=self.transform,
                )

    def test_failed_save_does_not_delete_existing_file(self):
        before = self.checkpoint_path.read_bytes()
        with self.assertRaises(ValueError):
            save_checkpoint(
                self.checkpoint_path,
                self.model,
                field_map_binding=self.field_map,
                input_transform=self.transform,
            )
        self.assertEqual(self.checkpoint_path.read_bytes(), before)

    def test_inference_requires_the_callers_pinned_binding(self):
        wrong_binding = dict(self.binding)
        wrong_binding["state_dict_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            infer(self.loaded, *self.make_inputs(), expected_binding=wrong_binding)

    def test_file_size_cap_is_checked_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory) / "oversized.pt"
            with oversized.open("wb") as stream:
                stream.truncate(inference.MAX_CHECKPOINT_BYTES + 1)
            with mock.patch.object(inference.torch, "load", side_effect=AssertionError("must not load")) as loader:
                with self.assertRaises(ValueError):
                    load_checkpoint(
                        oversized,
                        expected_binding=self.binding,
                        expected_field_map_binding=self.field_map,
                        expected_transform_binding=self.transform_binding,
                    )
            loader.assert_not_called()

    def test_zip_member_cap_is_checked_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "large_member.pt"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("archive/data/0", b"x" * 101)
            with mock.patch.object(inference, "MAX_ARCHIVE_MEMBER_BYTES", 100):
                with mock.patch.object(inference.torch, "load", side_effect=AssertionError("must not load")) as loader:
                    with self.assertRaises(ValueError):
                        load_checkpoint(
                            archive_path,
                            expected_binding=self.binding,
                            expected_field_map_binding=self.field_map,
                            expected_transform_binding=self.transform_binding,
                        )
            loader.assert_not_called()

    def test_state_hash_rejects_excess_elements_and_key_bytes_before_hashing(self):
        oversized = torch.empty((inference.MAX_STATE_ELEMENTS + 1,), device="meta")
        with self.assertRaises(ValueError):
            inference._state_hash({"oversized": oversized})
        with self.assertRaises(ValueError):
            inference._state_hash({"k" * (inference.MAX_STATE_KEY_BYTES + 1): torch.zeros(1)})


if __name__ == "__main__":
    unittest.main()

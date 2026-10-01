from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from bran_multisource_anchored_model_v3 import BRANMultisourceAnchoredModelV3
from bran_multisource_model_v2 import erase_cbc_for_completion
from bran_robust_clinical_r7 import BRANRobustClinicalR7, INPUT_MAP
from bran_r7_release import AgeBatch, normalize_age


ELIGIBLE = tuple(range(43))
CBC = tuple(range(9))


class R7CoreTests(unittest.TestCase):
    def test_input_map_is_applied_to_both_inherited_clinical_paths(self):
        parent = BRANMultisourceAnchoredModelV3("mlp", ELIGIBLE, CBC, seed=2026)
        model = BRANRobustClinicalR7.from_parent(parent).eval()
        clean = torch.linspace(-8.0, 8.0, 3 * 59, dtype=torch.float32).reshape(3, 59)
        valid = torch.ones_like(clean, dtype=torch.bool)
        valid[:, 43:] = False
        clean[:, 43:] = 0.0
        age = torch.zeros((3, 7), dtype=torch.float32)
        age[:, 6] = 1.0
        mapped = 3.0 * torch.asinh(clean / 3.0)

        expected = parent._clinical_hidden(mapped, valid, age)
        actual = model._clinical_hidden(clean, valid, age)

        self.assertEqual(INPUT_MAP["scale"], 3.0)
        self.assertFalse(INPUT_MAP["targets_changed"])
        self.assertFalse(INPUT_MAP["age_changed"])
        self.assertFalse(INPUT_MAP["masks_changed"])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_r7_config_round_trip_is_closed(self):
        model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC)
        restored = BRANRobustClinicalR7.from_config(model.export_config())
        self.assertEqual(restored.export_config(), model.export_config())
        invalid = dict(model.export_config())
        invalid["clinical_encoder_input_map"] = {"name": "identity"}
        with self.assertRaises(ValueError):
            BRANRobustClinicalR7.from_config(invalid)

    def test_synthetic_encode_has_fixed_state_shape_and_ignores_hidden_payload(self):
        torch.manual_seed(7)
        model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC).eval()
        values = torch.randn(2, 59)
        observed = torch.ones((2, 59), dtype=torch.bool)
        observed[:, 43:] = False
        values[:, 43:] = float("nan")
        changed = values.clone()
        changed[:, 43:] = 1.0e20
        retinal = torch.randn(2, 2, 384)
        retinal_visible = torch.ones((2, 2), dtype=torch.bool)
        age = torch.zeros((2, 7))
        age[:, 6] = 1.0

        first = model.encode(values, observed, retinal, retinal_visible, age)
        second = model.encode(changed, observed, retinal, retinal_visible, age)

        self.assertEqual(tuple(first.mean.shape), (2, 192))
        self.assertEqual(tuple(first.logvar.shape), (2, 192))
        self.assertTrue(torch.isfinite(first.mean).all())
        torch.testing.assert_close(first.mean, second.mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.logvar, second.logvar, rtol=0.0, atol=0.0)

    def test_cbc_completion_erases_values_and_masks(self):
        values = torch.arange(2 * 59, dtype=torch.float32).reshape(2, 59)
        observed = torch.ones_like(values, dtype=torch.bool)
        erased_values, erased_mask = erase_cbc_for_completion(values, observed, CBC)
        self.assertTrue(torch.equal(erased_values[:, CBC], torch.zeros((2, 9))))
        self.assertFalse(bool(erased_mask[:, CBC].any()))
        self.assertTrue(torch.equal(erased_values[:, 9:], values[:, 9:]))
        self.assertTrue(torch.equal(erased_mask[:, 9:], observed[:, 9:]))

    def test_empty_physiology_with_age_returns_native_abstention(self):
        model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC).eval()
        batch = 2
        values = torch.full((batch, 59), float("nan"))
        observed = torch.zeros((batch, 59), dtype=torch.bool)
        retinal = torch.full((batch, 2, 384), float("nan"))
        retinal_visible = torch.zeros((batch, 2), dtype=torch.bool)
        age = torch.zeros((batch, 7))
        age[:, 6] = 1.0

        state = model.encode(values, observed, retinal, retinal_visible, age)

        self.assertTrue(bool(state.abstain.all()))
        self.assertFalse(bool(state.clinical_available.any()))
        self.assertFalse(bool(state.retinal_available.any()))
        self.assertTrue(torch.isfinite(state.mean).all())

    def test_cbc_target_erasure_makes_state_invariant_to_synthetic_targets(self):
        torch.manual_seed(19)
        model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC).eval()
        values = torch.randn(3, 59)
        observed = torch.ones((3, 59), dtype=torch.bool)
        changed = values.clone()
        changed[:, CBC] += 1000.0
        erased_values, erased_observed = erase_cbc_for_completion(values, observed, CBC)
        changed_erased, changed_erased_observed = erase_cbc_for_completion(changed, observed, CBC)
        retinal = torch.randn(3, 1, 384)
        retinal_visible = torch.ones((3, 1), dtype=torch.bool)
        age = torch.zeros((3, 7))
        age[:, 6] = 1.0

        first = model.encode(erased_values, erased_observed, retinal, retinal_visible, age)
        second = model.encode(changed_erased, changed_erased_observed, retinal, retinal_visible, age)

        torch.testing.assert_close(first.mean, second.mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.logvar, second.logvar, rtol=0.0, atol=0.0)

    def test_missing_retinal_branch_does_not_use_invisible_embeddings(self):
        torch.manual_seed(23)
        model = BRANRobustClinicalR7("mlp", ELIGIBLE, CBC).eval()
        values = torch.randn(2, 59)
        observed = torch.ones((2, 59), dtype=torch.bool)
        retinal_visible = torch.zeros((2, 3), dtype=torch.bool)
        retinal_a = torch.randn(2, 3, 384)
        retinal_b = torch.full((2, 3, 384), 1.0e20)
        age = torch.zeros((2, 7))
        age[:, 6] = 1.0

        first = model.encode(values, observed, retinal_a, retinal_visible, age)
        second = model.encode(values, observed, retinal_b, retinal_visible, age)

        self.assertFalse(bool(first.retinal_available.any()))
        self.assertTrue(bool(first.clinical_available.all()))
        torch.testing.assert_close(first.mean, second.mean, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.logvar, second.logvar, rtol=0.0, atol=0.0)

    def test_public_facade_normalizes_synthetic_unknown_age(self):
        batch = AgeBatch(
            value=torch.full((1,), float("nan")),
            lower=torch.full((1,), float("nan")),
            upper=torch.full((1,), float("nan")),
            kind=torch.tensor([3]),
        )
        age = normalize_age(batch, mean=50.0, scale=15.0)
        self.assertEqual(tuple(age.shape), (1, 7))
        self.assertEqual(float(age[0, 6]), 1.0)


if __name__ == "__main__":
    unittest.main()

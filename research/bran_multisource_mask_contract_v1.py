"""Closed in-memory contracts for the prospective combined multisource-mask study.

These validators do not read files, train models, or serialize patient data.
They bind runner-produced aggregates, checkpoint bundles, manifests, and audits
to the already-established native rehearsal schemas.
"""
from __future__ import annotations

import math
import json

import numpy as np

import run_bran_native_rehearsal_v1 as old
import bran_supervised_mask_uncertainty_v1 as uncertainty


ERROR = "bran_multisource_mask_contract_v1_failed"
PATTERNS = ("single_target_hidden", "whole_cbc_hidden")
EVALPATTERNS = ("single_target_hidden", "whole_cbc_hidden",
                "single_target_no_retina", "whole_cbc_no_retina")
FLAGS = {**old.FLAGS, "external_rehearsal_used": True, "retinal_input_changed": False}
NORMALIZERS = tuple(old.origin.NORMALIZERS)
CBC_FIELDS = tuple(old.metrics.CBC_FIELDS)
VECTOR_NORMALIZERS = ("clinical_median", "clinical_iqr", "retinal_mean", "retinal_scale")
AGE_NORMALIZERS = ("age_mean", "age_scale")


def decisions(screen, completion, missingness, completion_no_retina):
    """Retain old gates exactly, then add the prospective no-retina check."""
    base = old.decisions(screen, completion, missingness)
    checks = dict(base["checks"])
    checks["no_retina_priority_point_nonworse"] = all(
        completion_no_retina[pattern][field]["overall"]["status"] == "supported"
        and all(completion_no_retina[pattern][field]["overall"]["contrasts"][reference]["delta"] <= 0
                for reference in ("initial", "continued"))
        for pattern in PATTERNS for field in ("hemoglobin", "plt", "wbc")
    )
    return {"checks": checks, "advancement_supported": bool(all(checks.values())),
            "automatic_promotion": False}


def _fail():
    raise ValueError(ERROR) from None


def _require(value):
    if not value:
        _fail()


def _hash(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _strict_json_equal(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
            right, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _source(p):
    try:
        source = p["native_source"]["source"]
        auth, names = source["authentication"], source["endpoint_names"]
        _require(type(p) is dict and type(source) is dict and type(auth) is dict)
        _require(isinstance(names, (list, tuple)) and len(names) == 26)
        _require(len(set(names)) == 26 and all(type(name) is str and name for name in names))
        return source
    except Exception:
        _fail()


def validate_result(a, p, pin):
    """Validate original metrics and the additional no-retina completion paths."""
    try:
        source = _source(p)
        old_keys = {"schema", "status", "screening", "completion", "completion_no_retina", "calibrated_completion", "missingness", "decisions",
                    "state_width", "paired_people", "recorded_conditions", "fold_authentication", *old.FLAGS}
        expected = old_keys | {"protocol_sha256", "external_rehearsal_used", "retinal_input_changed"}
        _require(type(a) is dict and set(a) == expected and _hash(pin))
        _require(a["schema"] == "bran-multisource-mask-aggregate-v1" and a["status"] == "completed"
                 and a["protocol_sha256"] == pin)
        _require(all(a[key] is value for key, value in FLAGS.items()))
        for key, value in (("state_width", 192), ("paired_people", 1928), ("recorded_conditions", 26)):
            _require(type(a[key]) is int and a[key] == value)
        _require(_strict_json_equal(a["fold_authentication"], source["authentication"]))
        legacy = {key: value for key, value in a.items()
                  if key not in {"protocol_sha256", "external_rehearsal_used", "retinal_input_changed", "completion_no_retina", "calibrated_completion"}}
        legacy["schema"] = "bran-native-rehearsal-aggregate-v1"
        legacy["decisions"] = old.decisions(a["screening"], a["completion"], a["missingness"])
        old.validate_result(legacy, p)
        _require(type(a["completion_no_retina"]) is dict and set(a["completion_no_retina"]) == set(PATTERNS))
        for pattern in PATTERNS:
            old.metrics.validate(a["screening"], a["completion_no_retina"][pattern],
                                 old.metrics.decisions(a["screening"], a["completion_no_retina"][pattern]),
                                 source["endpoint_names"])
        _require(_strict_json_equal(a["decisions"], decisions(
            a["screening"], a["completion"], a["missingness"], a["completion_no_retina"])))
        calibrated = a["calibrated_completion"]
        _require(type(calibrated) is dict and set(calibrated) == {"split_authentication", "patterns"})
        _require(type(calibrated["split_authentication"]) is dict
                 and set(calibrated["split_authentication"]) == {"patient_order_sha256", "roles_sha256"}
                 and all(_hash(value) for value in calibrated["split_authentication"].values()))
        uncertainty.validate_result(calibrated["patterns"])
    except Exception:
        _fail()


def validate_bundle(bundle, original, p, pin, fold):
    """Validate a no-private-key checkpoint bundle against an original state."""
    import torch

    try:
        source = _source(p)
        _require(_hash(pin) and type(fold) is int and 0 <= fold <= 4)
        expected_keys = set(NORMALIZERS) | {"continued", "student", "protocol_sha256",
            "initial_checkpoint_sha256", "fold", "endpoint_names", "cbc_fields"}
        _require(type(bundle) is dict and set(bundle) == expected_keys and type(original) is dict)
        _require(bundle["protocol_sha256"] == pin and type(bundle["fold"]) is int and bundle["fold"] == fold)
        _require(_hash(bundle["initial_checkpoint_sha256"])
                 and bundle["initial_checkpoint_sha256"] == p["native_source"]["checkpoint_sha256"]["fold" + str(fold)])
        _require(bundle["endpoint_names"] == source["endpoint_names"]
                 and bundle["cbc_fields"] == list(CBC_FIELDS))
        _require(set(NORMALIZERS) == set(VECTOR_NORMALIZERS) | set(AGE_NORMALIZERS))
        for key in VECTOR_NORMALIZERS:
            left, right = bundle[key], original[key]
            _require(type(left) is np.ndarray and type(right) is np.ndarray
                     and left.dtype == right.dtype and left.shape == right.shape
                     and np.isfinite(left).all() and np.isfinite(right).all()
                     and np.array_equal(left, right))
        for key in AGE_NORMALIZERS:
            left, right = bundle[key], original[key]
            _require(type(left) is float and type(right) is float
                     and math.isfinite(left) and math.isfinite(right) and left == right)
        _require(bundle["age_scale"] > 0 and original["age_scale"] > 0)
        expected = original["candidate"]
        _require(isinstance(expected, dict) and expected)
        for key, value in expected.items():
            _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and bool(torch.isfinite(value).all()))
        for version in ("continued", "student"):
            state = bundle[version]
            _require(isinstance(state, dict) and set(state) == set(expected))
            for key, value in state.items():
                reference = expected[key]
                _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                         and value.dtype == reference.dtype and value.shape == reference.shape
                         and bool(torch.isfinite(value).all()))
    except Exception:
        _fail()


def _checkpoint_pins(value):
    _require(type(value) is dict and set(value) == {"fold" + str(index) for index in range(5)})
    _require(all(_hash(pin) for pin in value.values()))


def validate_manifest(m, pin, aggregate_pin, checkpoint_pins, baseline_pin, calibration_pin):
    try:
        _require(_hash(pin) and _hash(aggregate_pin) and _hash(baseline_pin) and _hash(calibration_pin))
        _checkpoint_pins(checkpoint_pins)
        expected = {"protocol_sha256", "aggregate_sha256", "checkpoint_sha256",
                    "baseline_replay_sha256", "calibration_sha256", "elapsed_seconds", "patient_level_output_emitted"}
        _require(type(m) is dict and set(m) == expected)
        _require(m["protocol_sha256"] == pin and m["aggregate_sha256"] == aggregate_pin
                 and m["checkpoint_sha256"] == checkpoint_pins and m["baseline_replay_sha256"] == baseline_pin
                 and m["calibration_sha256"] == calibration_pin
                 and m["patient_level_output_emitted"] is False)
        _require(type(m["elapsed_seconds"]) in (int, float) and not isinstance(m["elapsed_seconds"], bool)
                 and math.isfinite(m["elapsed_seconds"]) and m["elapsed_seconds"] >= 0)
    except Exception:
        _fail()


def validate_audit(a, p, pin, aggregate_pin, manifest_pin, checkpoint_pins, calibration_pin):
    try:
        source = _source(p)
        _require(_hash(pin) and _hash(aggregate_pin) and _hash(manifest_pin) and _hash(calibration_pin))
        _checkpoint_pins(checkpoint_pins)
        expected = {"schema", "status", "protocol_sha256", "aggregate_sha256", "manifest_sha256",
                    "checkpoint_sha256", "fold_authentication", "all_aggregates_replayed",
                    "checkpoint_predictions_replayed", "candidate_training_repeated",
                    "raw_reference_heads_refit", "calibration_sha256", "calibration_replayed", "patient_level_output_emitted"}
        _require(type(a) is dict and set(a) == expected)
        _require(a["schema"] == "bran-multisource-mask-audit-v1" and a["status"] == "authenticated"
                 and a["protocol_sha256"] == pin and a["aggregate_sha256"] == aggregate_pin
                 and a["manifest_sha256"] == manifest_pin and a["checkpoint_sha256"] == checkpoint_pins
                 and _strict_json_equal(a["fold_authentication"], source["authentication"])
                 and a["all_aggregates_replayed"] is True and a["checkpoint_predictions_replayed"] is True
                 and a["candidate_training_repeated"] is False and a["raw_reference_heads_refit"] is True
                 and a["calibration_sha256"] == calibration_pin and a["calibration_replayed"] is True
                 and a["patient_level_output_emitted"] is False)
    except Exception:
        _fail()

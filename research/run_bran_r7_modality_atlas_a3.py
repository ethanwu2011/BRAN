"""Quiet local-only A3 evaluation of the frozen native R7 screening head."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import bran_r7_modality_atlas_a3 as kernel
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json


ROOT = Path(__file__).resolve().parent
ERROR = "bran_r7_modality_atlas_a3_failed"
SCHEMA = "bran-r7-modality-atlas-a3-aggregate-v1"
PROTOCOL_SCHEMA = "bran-r7-modality-atlas-a3-protocol-v1"
ATTEMPT_DIR = "BRAN_R7_MODALITY_ATLAS_A3_ATTEMPT1"
CODE = (
    "BRAN_R7_MODALITY_ATLAS_A3_DESIGN.md",
    "bran_r7_modality_atlas_a3.py",
    "test_bran_r7_modality_atlas_a3.py",
    "run_bran_r7_modality_atlas_a3.py",
    "test_run_bran_r7_modality_atlas_a3.py",
    "run_bran_r7_modality_atlas_a3_attempt1.sh",
    "bran_multisource_outcome_metrics_v2.py",
)
PHASES = (
    "authentication", "source_loading", "state_inference", "aggregate_summary",
    "post_authentication", "completed",
)


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def _sha256(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and set(value) <= set("0123456789abcdef"))


def paths(attempt: int = 1) -> Path:
    require(type(attempt) is int and attempt == 1)
    return ROOT / ATTEMPT_DIR


def read(path: Path) -> dict:
    from run_bran_r7_fixed_state_p1 import _read
    try:
        return _read(path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def authenticate() -> dict:
    """Reuse I1's source/fold/checkpoint closure and add A3 code pins."""
    import run_bran_r7_head_blocks_i1 as i1

    i1_binding = i1.authenticate()
    require(type(i1_binding) is dict
            and i1_binding.get("patient_level_output_emitted") is False
            and type(i1_binding.get("code_sha256")) is dict)
    for name, pin in i1_binding["code_sha256"].items():
        require(type(name) is str and _sha256(pin) and sha(ROOT / name) == pin)
    code_sha256 = dict(i1_binding["code_sha256"])
    code_sha256.update({name: sha(ROOT / name) for name in CODE})
    return {
        "i1_binding": i1_binding,
        "code_sha256": {name: code_sha256[name] for name in sorted(code_sha256)},
        "encoder_updated": False,
        "head_fitted": False,
        "patient_level_output_emitted": False,
    }


def sources(binding: dict):
    import run_bran_r7_head_blocks_i1 as i1

    require(type(binding) is dict and set(binding) == {
        "i1_binding", "code_sha256", "encoder_updated", "head_fitted",
        "patient_level_output_emitted",
    } and binding["encoder_updated"] is False and binding["head_fitted"] is False
        and binding["patient_level_output_emitted"] is False)
    value = i1.sources(binding["i1_binding"])
    require(value.receipt() == binding["i1_binding"]["source_binding"])
    return value


def _native_route_probabilities(model, routed):
    import torch

    head = model.screening_joint_head
    require(type(head) is torch.nn.Linear and head.in_features == 192
            and head.out_features == 26 and head.bias is not None)
    values = {}
    with torch.no_grad():
        for route in kernel.ROUTES:
            state = routed.states[route]
            available = routed.available[route]
            require(isinstance(state, torch.Tensor) and state.ndim == 2
                    and state.shape[1] == 192 and torch.isfinite(state).all()
                    and isinstance(available, torch.Tensor) and available.dtype == torch.bool
                    and available.shape == (len(state),)
                    and bool((state[~available] == 0).all()))
            probability = torch.sigmoid(head(state)).detach().clone()
            require(probability.shape == (len(state), 26)
                    and torch.isfinite(probability[available]).all()
                    and bool(((probability[available] >= 0)
                              & (probability[available] <= 1)).all()))
            values[route] = probability
    return values


def _store_heldout_route_predictions(predictions: dict, probabilities: dict,
                                    availability: dict, heldout: np.ndarray) -> None:
    """Copy only held-out, route-available scores into NaN-initialized arrays."""
    import numpy as np

    require(type(predictions) is dict and set(predictions) == set(kernel.ROUTES)
            and type(probabilities) is dict and set(probabilities) == set(kernel.ROUTES)
            and type(availability) is dict and set(availability) == set(kernel.ROUTES)
            and isinstance(heldout, np.ndarray) and heldout.dtype == np.dtype(bool)
            and heldout.ndim == 1)
    for route in kernel.ROUTES:
        dest, values, available = predictions[route], probabilities[route], availability[route]
        require(isinstance(dest, np.ndarray) and dest.shape == (len(heldout), 26)
                and dest.dtype.kind == "f"
                and isinstance(values, np.ndarray) and values.shape == dest.shape
                and np.isfinite(values).all()
                and np.all((values >= 0) & (values <= 1))
                and isinstance(available, np.ndarray) and available.dtype == np.dtype(bool)
                and available.shape == heldout.shape)
        keep = heldout & available
        dest[keep] = values[keep]


def evaluate(bound, binding: dict, callback) -> dict:
    """Score each held-out route with one unchanged R7 head; never fit."""
    import numpy as np
    import torch
    import bran_r7_fixed_state_p1 as p1
    import run_bran_robust_clinical_r7 as r7
    from bran_clinical_semantics_v1 import CBC_FIELDS
    from bran_multisource_batches_v2 import tensor
    from bran_multisource_outcomes_v2 import original_age
    from bran_v5_state_routes import state_routes

    torch.set_num_threads(2)
    paired = bound.paired
    folds = np.asarray(paired.folds)
    p1.validate_evaluation_inputs(paired, lambda *args: None)
    age = original_age(paired)
    slots = tuple(paired.names.index(field) for field in CBC_FIELDS)
    predictions = {
        route: np.full((len(folds), 26), np.nan, dtype=np.float64)
        for route in kernel.ROUTES
    }
    i1_binding = binding["i1_binding"]
    manifest = i1_binding["checkpoint_manifest"]
    require(type(manifest) is dict and manifest.get("component_role") == "R"
            and type(manifest.get("folds")) is list and len(manifest["folds"]) == 5)
    _, private = r7.paths("fit", 1)

    for fold in range(5):
        callback("state_inference", fold)
        item = manifest["folds"][fold]
        require(type(item) is dict and item.get("fold") == fold)

        def load():
            return r7.load_checkpoint(
                private / f"fold{fold}_R.pt", item["checkpoint_sha256"], item["binding"]
            )

        model, transform = load()
        before, transform_pin, gradients = p1._validate_r7(
            model, transform, fold, slots, paired.transforms[fold]
        )
        c, cm = transform.clinical(paired.c, paired.cm)
        r, rm = transform.retinal(paired.r, paired.rm)
        args = (
            tensor(c), tensor(cm, torch.bool), tensor(r), tensor(rm, torch.bool),
            age, transform.age_mean, transform.age_scale,
        )
        routed = state_routes(model, *args)
        route_probabilities = _native_route_probabilities(model, routed)
        require(p1._role_unchanged(before, transform_pin, gradients, model, transform))

        replay, replay_transform = load()
        replay_before, replay_pin, replay_gradients = p1._validate_r7(
            replay, replay_transform, fold, slots, paired.transforms[fold]
        )
        replayed = state_routes(replay, *args)
        replay_probabilities = _native_route_probabilities(replay, replayed)
        require(replay is not model and p1.replay_equal(routed, replayed)
                and all(torch.equal(route_probabilities[route], replay_probabilities[route])
                        for route in kernel.ROUTES)
                and p1._role_unchanged(replay_before, replay_pin, replay_gradients,
                                       replay, replay_transform))

        heldout = folds == fold
        route_availability = {
            route: routed.available[route].detach().cpu().numpy()
            for route in kernel.ROUTES
        }
        route_values = {
            route: route_probabilities[route].detach().cpu().numpy().astype(
                np.float64, copy=False
            ) for route in kernel.ROUTES
        }
        _store_heldout_route_predictions(
            predictions, route_values, route_availability, heldout
        )
        del model, transform, replay, replay_transform, routed, replayed
        del route_probabilities, replay_probabilities, c, cm, r, rm

    callback("aggregate_summary", None)
    counts = p1.paired_counts(folds, draws=1000, seed=98571)
    report = kernel.summarize(
        predictions, paired.labels, paired.labelmask, folds,
        tuple(paired.endpoint_names), counts,
    )
    require(report == kernel.summarize(
        predictions, paired.labels, paired.labelmask, folds,
        tuple(paired.endpoint_names), counts,
    ))
    kernel.validate_result(report)
    return report


def progress(out: Path, state: dict, phase: str, fold: int | None = None) -> None:
    require(phase in PHASES and (fold is None or type(fold) is int and fold in range(5)))
    state["phase"] = phase
    write_json(out / "progress.next.json", {
        "phase": phase,
        "fold": fold,
        "encoder_updated": False,
        "head_fitted": False,
        "patient_level_output_emitted": False,
    })
    os.replace(out / "progress.next.json", out / "progress.json")


def validate_aggregate(value: object) -> None:
    require(type(value) is dict and set(value) == {
        "schema", "protocol_sha256", "report", "state_replay_equal",
        "aggregate_replay_equal", "native_head_identity_checked", "encoder_updated",
        "head_fitted", "patient_level_output_emitted",
    } and value["schema"] == SCHEMA and _sha256(value["protocol_sha256"])
        and value["state_replay_equal"] is True
        and value["aggregate_replay_equal"] is True
        and value["native_head_identity_checked"] is True
        and value["encoder_updated"] is False and value["head_fitted"] is False
        and value["patient_level_output_emitted"] is False)
    kernel.validate_result(value["report"])


def _validate_protocol(value: object) -> None:
    require(type(value) is dict and set(value) == {
        "schema", "binding", "parameters", "frozen_before_inference",
        "encoder_updated", "head_fitted", "patient_level_output_emitted",
    } and value["schema"] == PROTOCOL_SCHEMA
        and type(value["binding"]) is dict
        and value["parameters"] == kernel.PARAMETERS
        and value["frozen_before_inference"] is True
        and value["encoder_updated"] is False and value["head_fitted"] is False
        and value["patient_level_output_emitted"] is False)
    binding = value["binding"]
    require(set(binding) == {
        "i1_binding", "code_sha256", "encoder_updated", "head_fitted",
        "patient_level_output_emitted",
    } and type(binding["i1_binding"]) is dict
        and type(binding["code_sha256"]) is dict
        and binding["encoder_updated"] is False and binding["head_fitted"] is False
        and binding["patient_level_output_emitted"] is False
        and all(type(name) is str and _sha256(pin)
                for name, pin in binding["code_sha256"].items()))
    i1_binding = binding["i1_binding"]
    require(set(i1_binding) == {
        "source_binding", "checkpoint_manifest", "r7_fit_receipt", "p1_receipt",
        "code_sha256", "patient_level_output_emitted",
    } and type(i1_binding["source_binding"]) is dict
        and type(i1_binding["checkpoint_manifest"]) is dict
        and i1_binding["checkpoint_manifest"].get("component_role") == "R"
        and type(i1_binding["checkpoint_manifest"].get("folds")) is list
        and len(i1_binding["checkpoint_manifest"]["folds"]) == 5
        and type(i1_binding["r7_fit_receipt"]) is dict
        and type(i1_binding["p1_receipt"]) is dict
        and type(i1_binding["code_sha256"]) is dict
        and i1_binding["patient_level_output_emitted"] is False
        and all(type(name) is str and _sha256(pin)
                for name, pin in i1_binding["code_sha256"].items()))
    expected_names = set(i1_binding["code_sha256"]) | set(CODE)
    require(set(binding["code_sha256"]) == expected_names
            and all(binding["code_sha256"][name] == i1_binding["code_sha256"][name]
                    for name in set(i1_binding["code_sha256"]) & set(CODE)))


def run(attempt: int = 1) -> dict:
    out = paths(attempt)
    require(not out.exists() and not out.is_symlink())
    with LOCK.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists() and not out.is_symlink())
        out.mkdir()
        state = {"phase": "authentication"}
        try:
            with quiet():
                callback = lambda phase, fold=None: progress(out, state, phase, fold)
                callback("authentication")
                binding = authenticate()
                protocol = {
                    "schema": PROTOCOL_SCHEMA,
                    "binding": binding,
                    "parameters": kernel.PARAMETERS,
                    "frozen_before_inference": True,
                    "encoder_updated": False,
                    "head_fitted": False,
                    "patient_level_output_emitted": False,
                }
                _validate_protocol(protocol)
                write_json(out / "protocol.json", protocol)
                protocol_pin = sha(out / "protocol.json")

                callback("source_loading")
                bound = sources(binding)
                report = evaluate(bound, binding, callback)

                callback("post_authentication")
                require(authenticate() == binding
                        and sources(binding).receipt() == binding["i1_binding"]["source_binding"]
                        and sha(out / "protocol.json") == protocol_pin)
                aggregate = {
                    "schema": SCHEMA,
                    "protocol_sha256": protocol_pin,
                    "report": report,
                    "state_replay_equal": True,
                    "aggregate_replay_equal": True,
                    "native_head_identity_checked": True,
                    "encoder_updated": False,
                    "head_fitted": False,
                    "patient_level_output_emitted": False,
                }
                validate_aggregate(aggregate)
                write_json(out / "aggregate.json", aggregate)
                callback("completed")
                write_json(out / "completed.json", {
                    "status": "authenticated_completed",
                    "protocol_sha256": protocol_pin,
                    "aggregate_sha256": sha(out / "aggregate.json"),
                    "encoder_updated": False,
                    "head_fitted": False,
                    "patient_level_output_emitted": False,
                })
            return {"status": "completed_pending_separate_audit",
                    "encoder_updated": False, "head_fitted": False,
                    "patient_level_output_emitted": False}
        except BaseException:
            require(not (out / "completed.json").exists())
            write_json(out / "failure.json", {
                "status": "technical_failure",
                "phase": state["phase"],
                "error_code": ERROR,
                "encoder_updated": False,
                "head_fitted": False,
                "patient_level_output_emitted": False,
            })
            return {"status": "technical_failure", "phase": state["phase"],
                    "encoder_updated": False, "head_fitted": False,
                    "patient_level_output_emitted": False}


def audit(attempt: int = 1, replay: bool = True) -> dict:
    with LOCK.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with quiet():
            out = paths(attempt)
            require(out.is_dir() and not out.is_symlink()
                    and {item.name for item in out.iterdir()} == {
                        "protocol.json", "aggregate.json", "completed.json", "progress.json"
                    }
                    and all(item.is_file() and not item.is_symlink()
                            and item.stat().st_nlink == 1 for item in out.iterdir()))
            protocol = read(out / "protocol.json")
            result = read(out / "aggregate.json")
            done = read(out / "completed.json")
            progress_record = read(out / "progress.json")
            pin = sha(out / "protocol.json")
            terminal_hashes = {
                name: sha(out / name)
                for name in ("protocol.json", "aggregate.json", "completed.json", "progress.json")
            }
            _validate_protocol(protocol)
            require(type(result) is dict and result.get("protocol_sha256") == pin
                and done == {
                "status": "authenticated_completed",
                "protocol_sha256": pin,
                "aggregate_sha256": sha(out / "aggregate.json"),
                "encoder_updated": False,
                "head_fitted": False,
                "patient_level_output_emitted": False,
            }
                and progress_record == {
                    "phase": "completed", "fold": None,
                    "encoder_updated": False, "head_fitted": False,
                    "patient_level_output_emitted": False,
                })
            validate_aggregate(result)
            binding = protocol["binding"]
            require(authenticate() == binding)
            bound = sources(binding)
            if replay:
                require(evaluate(bound, binding, lambda *args: None) == result["report"])
                require(authenticate() == binding
                        and sources(binding).receipt() == binding["i1_binding"]["source_binding"])
            terminal_names = {
                "protocol.json", "aggregate.json", "completed.json", "progress.json"
            }
            require({item.name for item in out.iterdir()} == terminal_names
                    and all(item.is_file() and not item.is_symlink()
                            and item.stat().st_nlink == 1 for item in out.iterdir())
                    and {name: sha(out / name) for name in terminal_hashes} == terminal_hashes
                    and read(out / "protocol.json") == protocol
                    and read(out / "aggregate.json") == result
                    and read(out / "completed.json") == done
                    and read(out / "progress.json") == progress_record)
            return {
                "status": "aggregate_terminal_authenticated",
                "protocol_sha256": pin,
                "aggregate_sha256": done["aggregate_sha256"],
                "independent_source_inference_replay": replay,
                "encoder_updated": False,
                "head_fitted": False,
                "patient_level_output_emitted": False,
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    try:
        with quiet():
            result = audit(args.attempt) if args.audit else run(args.attempt)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] in (
            "completed_pending_separate_audit", "aggregate_terminal_authenticated"
        ) else 1
    except BaseException:
        print(json.dumps({"status": "closed_failure",
                          "encoder_updated": False, "head_fitted": False,
                          "patient_level_output_emitted": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

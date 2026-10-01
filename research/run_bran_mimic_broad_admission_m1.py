"""Local-only lifecycle for the broader MIMIC admission M1 hand-off.

This runner is deliberately a source-bound admission job, not a model or
clinical-utility experiment.  It authenticates the corrected C2 terminal,
reuses the qualified laboratory and disease bindings, and writes one private
selected-admission artifact plus disclosure-safe coarsened aggregates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import bran_clinical_discovery_landmark_v3 as discovery
import bran_clinical_source_reader_v1 as source_reader
import bran_joint_lab_cache_v1 as lab_cache
import bran_mimic_landmark_join_v3 as landmark_join
import bran_mimic_landmark_source_v1 as source
import bran_mimic_prior_disease_dictionary_v1 as disease_dictionary
import bran_mimic_retrospective_membership_v2 as membership
import run_bran_mimic_landmark_linkage_v1 as linked
import run_bran_mimic_joint_labs_v1 as parent
import run_bran_mimic_upstream_coverage_c2_attempt2 as c2
from bran_clinical_source_reader_v1 import iter_projected_csv
from bran_joint_lab_online_v1 import authenticate_code_binding
from diagnose_bran_agefree_reference_failure_v1 import safe_trace
from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json

from bran_mimic_broad_snapshot_m1 import BroadSnapshot, BroadSnapshotAccumulator


ROOT = Path(__file__).resolve().parent
DESIGN = ROOT / "BRAN_MIMIC_BROAD_ADMISSION_M1_DESIGN.md"
OUT = ROOT / "BRAN_MIMIC_BROAD_ADMISSION_M1_ATTEMPT1"
PRIVATE = ROOT / "private_artifacts" / "bran_mimic_broad_admission_m1_attempt1"
SALT_PATH = Path(parent.SALT)
LIMITS = {
    "patients": 2_000_000,
    "admissions": 2_000_000,
    "labs": 200_000_000,
    "diagnoses": 20_000_000,
}
PHASES = (
    "authentication",
    "source_hashes",
    "metadata",
    "lab_scan",
    "admissions",
    "diagnoses",
    "selection",
    "private_artifact",
    "post_authentication",
    "completed",
)
ERROR = "mimic_broad_admission_m1_contract_failed"
AGE_KINDS = tuple(lab_cache.AGE_KINDS)
FIELDS = tuple(lab_cache.FIELDS)
FAMILIES = tuple(membership.FAMILIES)
COHORT_KEYS = (
    "values",
    "observed",
    "specimen_minutes",
    "available_minutes",
    "age_triplet",
    "age_kind",
    "person",
    "episode",
    "row_binding",
    "roles",
    "membership",
    "labels",
    "outcome",
)
COUNT_KEYS = (
    "source_local_people_lower_bound_20",
    "admitted_people_lower_bound_20",
    "chemistry_only_people_lower_bound_20",
    "fewer_than_two_cbc_people_lower_bound_20",
)
FALSE_FLAGS = {
    "patient_level_output_emitted": False,
    "model_inference_performed": False,
    "clustering_performed": False,
    "training_exposure_changed": False,
    "clinically_adjudicated_membership": False,
}
PUBLIC_BINDING_KEYS = frozenset({
    "c2_binding",
    "source_sha256",
    "mapping_sha256",
    "disease_dictionary_sha256",
    "disease_receipt_sha256",
    "disease_audit_sha256",
    "approved_code_map_sha256",
    "split_salt_sha256",
})
PRIVATE_BINDING_KEYS = frozenset({"_mapping", "_approved_codes"})
KNOWN_MINOR_RULE = "exclude only when finite documented age upper_years <= 18; retain unknown age"
SELECTION_RULE = "earliest eligible admission per person before membership or known-outcome filtering"
MEMBERSHIP_RULE = "current-discharge dictionary-bound retrospective administrative membership"
LANDMARK_RULE = "strictly beyond 24h observed endpoint; ambiguous outcome remains unknown"
JSON_MAX_BYTES = 8 * 1024 * 1024


def require(condition: bool) -> None:
    if not condition:
        raise ValueError(ERROR) from None


def _public(value):
    """Remove in-memory authenticated objects before protocol serialization."""
    require(isinstance(value, dict))
    keys = set(value)
    require(keys <= PUBLIC_BINDING_KEYS | PRIVATE_BINDING_KEYS)
    require(keys - PRIVATE_BINDING_KEYS == PUBLIC_BINDING_KEYS)
    return {key: value[key] for key in sorted(PUBLIC_BINDING_KEYS)}


def _json_digest(value: object) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except Exception:
        raise ValueError(ERROR) from None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_inputs() -> dict[str, Path]:
    """Return only the approved original source paths, without opening them."""
    return {
        "patients": Path(linked.INPUTS["patients"]),
        "admissions": Path(linked.INPUTS["admissions"]),
        "labs": Path(linked.INPUTS["labs"]),
        "lab_dictionary": Path(linked.INPUTS["dictionary"]),
        "diagnoses": Path(linked.PHENOTYPE_INPUTS["diagnoses"]),
        "disease_dictionary": Path(linked.PHENOTYPE_INPUTS["dictionary"]),
    }


SOURCE_INPUTS = _source_inputs()


def _canonical_codes(approved: dict[tuple[str, str], str]) -> list[list[object]]:
    require(isinstance(approved, dict))
    return [[key[0], key[1], value] for key, value in sorted(approved.items())]


def dependency_binding() -> dict[str, object]:
    """Authenticate C2 and all nonpatient mappings before source rows.

    The underscored entries are intentionally process-local objects used by
    ``run``.  They are removed by ``_public`` before writing protocol JSON.
    """
    try:
        c2_out, _aggregate, receipt = c2.authenticate(2)
        require(isinstance(receipt, dict))
        require(receipt.get("status") == "authenticated")
        require(receipt.get("patient_level_output_emitted") is False)
        c2_out = Path(c2_out)
        require(c2_out.is_dir())
        for name in ("protocol.json", "aggregate.json", "completed.json"):
            path = c2_out / name
            require(path.is_file() and not path.is_symlink())
        mapping = parent.mapping()
        approved = disease_dictionary.authenticate_audit()
        require(len(SALT_PATH.read_bytes()) == 32)
        source_sha = {key: sha(Path(path)) for key, path in SOURCE_INPUTS.items()}
        return {
            "c2_binding": {
                "protocol_sha256": sha(c2_out / "protocol.json"),
                "aggregate_sha256": sha(c2_out / "aggregate.json"),
                "completed_sha256": sha(c2_out / "completed.json"),
            },
            "source_sha256": source_sha,
            "mapping_sha256": _json_digest(sorted(mapping.items())),
            "disease_dictionary_sha256": sha(Path(disease_dictionary.DICTIONARY)),
            "disease_receipt_sha256": sha(Path(disease_dictionary.OUT)),
            "disease_audit_sha256": sha(Path(disease_dictionary.AUDIT)),
            "approved_code_map_sha256": _json_digest(_canonical_codes(approved)),
            "split_salt_sha256": sha(SALT_PATH),
            "_mapping": mapping,
            "_approved_codes": approved,
        }
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def source_hashes(binding: dict[str, object]) -> None:
    try:
        expected = binding["source_sha256"]
        require(isinstance(expected, dict) and set(expected) == set(SOURCE_INPUTS))
        for key, path in SOURCE_INPUTS.items():
            require(sha(Path(path)) == expected[key])
        require(sha(SALT_PATH) == binding["split_salt_sha256"])
    except Exception:
        raise ValueError(ERROR) from None


def code_hashes() -> dict[str, str]:
    names = set(getattr(linked, "CODE", ())) | {
        "run_bran_mimic_broad_admission_m1.py",
        "test_run_bran_mimic_broad_admission_m1.py",
        "bran_mimic_broad_snapshot_m1.py",
        "test_bran_mimic_broad_snapshot_m1.py",
        "BRAN_MIMIC_BROAD_ADMISSION_M1_DESIGN.md",
        "bran_mimic_landmark_join_v3.py",
        "bran_mimic_retrospective_membership_v2.py",
        "bran_clinical_discovery_landmark_v3.py",
        "bran_joint_lab_online_v1.py",
        "bran_joint_lab_cache_v1.py",
        "bran_mimic_landmark_source_v1.py",
        "bran_mimic_prior_disease_dictionary_v1.py",
        "bran_clinical_source_reader_v1.py",
        "diagnose_bran_agefree_reference_failure_v1.py",
        "run_bran_multisource_retinal_features_v2.py",
        "run_bran_mimic_upstream_coverage_c2.py",
        "bran_mimic_upstream_coverage_c2.py",
        "run_bran_mimic_upstream_coverage_c2_attempt2.py",
    }
    return {name: sha(ROOT / name) for name in sorted(names)}


def _protocol(binding: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "bran-mimic-broad-admission-m1-protocol",
        "status": "frozen_before_source_rows",
        "design_sha256": sha(DESIGN),
        "code_sha256": code_hashes(),
        "dependency_binding": _public(binding),
        "fields": list(FIELDS),
        "age_kinds": list(AGE_KINDS),
        "families": list(FAMILIES),
        "max_rows": dict(LIMITS),
        "lab_window_minutes": 1440.0,
        "minimum_observed_fields": 1,
        "known_minor_rule": KNOWN_MINOR_RULE,
        "selection_rule": SELECTION_RULE,
        "membership_rule": MEMBERSHIP_RULE,
        "landmark_rule": LANDMARK_RULE,
        "patient_level_output_emitted": False,
        "model_inference_performed": False,
        "clustering_performed": False,
        "training_exposure_changed": False,
    }


def validate_protocol(protocol: object, *, expected_binding: dict[str, object] | None = None) -> None:
    try:
        require(isinstance(protocol, dict))
        expected_keys = {
            "schema", "status", "design_sha256", "code_sha256", "dependency_binding",
            "fields", "age_kinds", "families", "max_rows", "lab_window_minutes",
            "minimum_observed_fields", "known_minor_rule", "selection_rule",
            "membership_rule", "landmark_rule", "patient_level_output_emitted",
            "model_inference_performed", "clustering_performed", "training_exposure_changed",
        }
        require(set(protocol) == expected_keys)
        require(protocol["schema"] == "bran-mimic-broad-admission-m1-protocol")
        require(protocol["status"] == "frozen_before_source_rows")
        require(protocol["design_sha256"] == sha(DESIGN))
        require(protocol["code_sha256"] == code_hashes())
        require(protocol["fields"] == list(FIELDS) and protocol["age_kinds"] == list(AGE_KINDS))
        require(protocol["families"] == list(FAMILIES) and protocol["max_rows"] == LIMITS)
        require(protocol["lab_window_minutes"] == 1440.0 and protocol["minimum_observed_fields"] == 1)
        require(protocol["known_minor_rule"] == KNOWN_MINOR_RULE)
        require(protocol["selection_rule"] == SELECTION_RULE)
        require(protocol["membership_rule"] == MEMBERSHIP_RULE)
        require(protocol["landmark_rule"] == LANDMARK_RULE)
        require(protocol["patient_level_output_emitted"] is False)
        require(protocol["model_inference_performed"] is False)
        require(protocol["clustering_performed"] is False)
        require(protocol["training_exposure_changed"] is False)
        if expected_binding is None:
            expected_binding = _public(dependency_binding())
        require(protocol["dependency_binding"] == _public(expected_binding))
    except Exception:
        raise ValueError(ERROR) from None


def _read_json(path: Path) -> object:
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)
    require(path.stat().st_size <= JSON_MAX_BYTES)

    def reject_constant(_value):
        raise ValueError(ERROR) from None

    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result)
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except Exception:
        raise ValueError(ERROR) from None


def _valid_hex(value: object, width: int = 64) -> bool:
    return (
        type(value) is str
        and len(value) == width
        and value.isascii()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_arrays(arrays: object) -> None:
    """Closed private-artifact schema validator; never prints array contents."""
    try:
        require(isinstance(arrays, dict) and set(arrays) == set(COHORT_KEYS))
        values, observed = arrays["values"], arrays["observed"]
        require(type(values) is np.ndarray and values.dtype == np.dtype(np.float64) and values.ndim == 2)
        n = values.shape[0]
        require(n >= 1 and values.shape[1] == len(FIELDS))
        require(type(observed) is np.ndarray and observed.dtype == np.dtype(bool) and observed.shape == values.shape)
        require(np.isfinite(values).all() and (values[~observed] == 0).all() and (values[observed] >= 0).all())
        require((values[:, :9][observed[:, :9]] > 0).all() and observed.any(axis=1).all())
        for name in ("specimen_minutes", "available_minutes"):
            array = arrays[name]
            require(type(array) is np.ndarray and array.dtype == np.dtype(np.float64) and array.shape == values.shape)
            require(np.isnan(array[~observed]).all())
            require(np.isfinite(array[observed]).all() and (array[observed] >= 0).all()
                    and (array[observed] <= 1440).all())
        require((arrays["specimen_minutes"][observed] <= arrays["available_minutes"][observed]).all())

        age_triplet, age_kind = arrays["age_triplet"], arrays["age_kind"]
        require(type(age_triplet) is np.ndarray and age_triplet.dtype == np.dtype(np.float64)
                and age_triplet.shape == (n, 3))
        require(type(age_kind) is np.ndarray and age_kind.dtype == np.dtype(np.uint8)
                and age_kind.shape == (n,) and np.isin(age_kind, np.arange(len(AGE_KINDS))).all())
        require(not np.isneginf(age_triplet).any())
        require(np.isnan(age_triplet[age_kind == AGE_KINDS.index("missing_or_invalid")]).all())

        for name in ("person", "episode", "row_binding"):
            array = arrays[name]
            require(type(array) is np.ndarray and array.dtype == np.dtype("U64") and array.shape == (n,))
            values_list = array.tolist()
            require(all(type(value) is str and value for value in values_list))
            require(len(set(values_list)) == n)
        require(all(_valid_hex(value) for value in arrays["row_binding"].tolist()))

        roles = arrays["roles"]
        require(type(roles) is np.ndarray and roles.dtype == np.dtype(np.uint8) and roles.shape == (n,))
        require(np.isin(roles, (0, 1, 2)).all())
        current = arrays["membership"]
        require(type(current) is np.ndarray and current.dtype == np.dtype(bool)
                and current.shape == (n, len(FAMILIES)))
        labels = arrays["labels"]
        require(type(labels) is np.ndarray and labels.dtype == np.dtype(np.int8)
                and labels.shape == (n, len(FAMILIES)) and np.isin(labels, (-1, 0, 1)).all())
        outcome = arrays["outcome"]
        require(type(outcome) is np.ndarray and outcome.dtype == np.dtype(np.int8)
                and outcome.shape == (n,) and np.isin(outcome, (0, 1, 2)).all())
        expected = np.full(labels.shape, -1, dtype=np.int8)
        known = np.isin(outcome, (1, 2))
        expected[known] = np.where(outcome[known, None] == 1, 1, 0)
        expected[known] = np.where(current[known], expected[known], -1)
        require(np.array_equal(labels, expected))
    except Exception:
        raise ValueError(ERROR) from None


def _coarse(value: int) -> int | None:
    require(type(value) is int and value >= 0)
    return lab_cache.coarse_count(value)


def summary(arrays: dict[str, np.ndarray], private_sha256: str) -> dict[str, object]:
    validate_arrays(arrays)
    people = arrays["person"]
    observed = arrays["observed"]
    cbc_count = observed[:, :9].sum(axis=1)
    chemistry = ~observed[:, :9].any(axis=1)
    known = np.isin(arrays["outcome"], (1, 2))
    result = {
        "admitted_people_lower_bound_20": _coarse(int(len(people))),
        "source_local_people_lower_bound_20": _coarse(int(len(people))),
        "chemistry_only_people_lower_bound_20": _coarse(int(chemistry.sum())),
        "fewer_than_two_cbc_people_lower_bound_20": _coarse(int((cbc_count < 2).sum())),
        "recorded_membership_people_lower_bound_20": {
            family: _coarse(int(arrays["membership"][:, index].sum()))
            for index, family in enumerate(FAMILIES)
        },
        "recorded_membership_known_outcome_people_lower_bound_20": {
            family: _coarse(int((arrays["membership"][:, index] & known).sum()))
            for index, family in enumerate(FAMILIES)
        },
        "private_sha256": {"cohort.npz": private_sha256},
        "scope": "selected broad hospital-admission snapshots; source-local counts are non-additive",
        "interpretation": "admission observations and retrospective administrative membership, not training exposure or clinical adjudication",
    }
    return result


def validate_result(result: object) -> None:
    try:
        require(isinstance(result, dict))
        expected = set(COUNT_KEYS) | {
            "recorded_membership_people_lower_bound_20",
            "recorded_membership_known_outcome_people_lower_bound_20",
            "private_sha256", "scope", "interpretation", "schema", "status", *FALSE_FLAGS,
        }
        require(set(result) == expected)
        require(result["schema"] == "bran-mimic-broad-admission-m1"
                and result["status"] == "completed")
        for key in COUNT_KEYS:
            value = result[key]
            require(value is None or (type(value) is int and value >= 20 and value % 20 == 0))
        for key in ("recorded_membership_people_lower_bound_20",
                    "recorded_membership_known_outcome_people_lower_bound_20"):
            cells = result[key]
            require(isinstance(cells, dict) and set(cells) == set(FAMILIES))
            for value in cells.values():
                require(value is None or (type(value) is int and value >= 20 and value % 20 == 0))
        require(isinstance(result["private_sha256"], dict)
                and set(result["private_sha256"]) == {"cohort.npz"}
                and _valid_hex(result["private_sha256"]["cohort.npz"]))
        require(result["scope"] == "selected broad hospital-admission snapshots; source-local counts are non-additive")
        require(result["interpretation"] == "admission observations and retrospective administrative membership, not training exposure or clinical adjudication")
        require(all(result[key] is False for key in FALSE_FLAGS))
    except Exception:
        raise ValueError(ERROR) from None


def _row_binding(salt: bytes, person: str, episode: str) -> str:
    require(type(salt) is bytes and len(salt) == 32)
    require(type(person) is str and type(episode) is str)
    message = b"bran-mimic-broad-admission-m1\0" + person.encode("ascii") + b"\0" + episode.encode("ascii")
    return hmac.new(salt, message, hashlib.sha256).hexdigest()


def _age_is_known_minor(age) -> bool:
    try:
        return math.isfinite(float(age.upper_years)) and float(age.upper_years) <= 18.0
    except Exception:
        return False


def _age_triplet(age) -> tuple[float, float, float]:
    return (float(age.reported_years), float(age.lower_years), float(age.upper_years))


def _read_rows(key: str, columns: tuple[str, ...], limit: int):
    require(key in SOURCE_INPUTS and type(limit) is int and limit > 0)
    return iter_projected_csv(Path(SOURCE_INPUTS[key]), columns, max_rows=limit)


def _write_private(arrays: dict[str, np.ndarray]) -> Path:
    validate_arrays(arrays)
    require(not PRIVATE.exists() and not PRIVATE.is_symlink())
    PRIVATE.mkdir(parents=True, mode=0o700)
    os.chmod(PRIVATE, 0o700)
    path = PRIVATE / "cohort.npz"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)
    return path


def _load_private(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)
    require(path.stat().st_mode & 0o777 == 0o600)
    try:
        with np.load(path, allow_pickle=False) as handle:
            arrays = {name: handle[name] for name in handle.files}
    except Exception:
        raise ValueError(ERROR) from None
    validate_arrays(arrays)
    return arrays


def _progress(phase: str, bucket: int | None = None, *, start: float = 0.0) -> None:
    require(phase in PHASES)
    require(bucket is None or type(bucket) is int and bucket >= 0 and bucket % 1_000_000 == 0)
    elapsed = time.monotonic() - start
    require(type(elapsed) is float and math.isfinite(elapsed) and elapsed >= 0.0)
    value = {
        "schema": "bran-mimic-broad-admission-m1-progress",
        "phase": phase,
        "processed_lab_rows_lower_bound_million": bucket,
        "elapsed_seconds": elapsed,
        "patient_level_output_emitted": False,
        "model_inference_performed": False,
        "clustering_performed": False,
    }
    temporary = OUT / "progress.next.json"
    write_json(temporary, value)
    os.replace(temporary, OUT / "progress.json")


def _validate_terminal(terminal: object, protocol_pin: str, aggregate_pin: str) -> None:
    require(isinstance(terminal, dict) and set(terminal) == {
        "status", "protocol_sha256", "aggregate_sha256", "patient_level_output_emitted",
    })
    require(terminal["status"] == "authenticated_completed"
            and terminal["protocol_sha256"] == protocol_pin
            and terminal["aggregate_sha256"] == aggregate_pin
            and terminal["patient_level_output_emitted"] is False)


def _expected_output_names(pending: bool) -> set[str]:
    return {"protocol.json", "aggregate.json", "progress.json"} | ({"completed.json"} if not pending else set())


def authenticate(*, _pending_terminal: dict[str, object] | None = None):
    """Replay all closed schemas and private arrays; no source rows are read."""
    try:
        require(OUT.is_dir() and not OUT.is_symlink())
        require(set(path.name for path in OUT.iterdir()) == _expected_output_names(_pending_terminal is not None))
        protocol = _read_json(OUT / "protocol.json")
        aggregate = _read_json(OUT / "aggregate.json")
        validate_protocol(protocol)
        binding = protocol["dependency_binding"]
        source_hashes(binding)
        validate_result(aggregate)
        require(PRIVATE.is_dir() and not PRIVATE.is_symlink() and PRIVATE.stat().st_mode & 0o777 == 0o700)
        require(set(path.name for path in PRIVATE.iterdir()) == {"cohort.npz"})
        private = PRIVATE / "cohort.npz"
        arrays = _load_private(private)
        require(sha(private) == aggregate["private_sha256"]["cohort.npz"])
        expected_summary = summary(arrays, sha(private))
        for key, value in expected_summary.items():
            require(aggregate[key] == value)
        progress = _read_json(OUT / "progress.json")
        require(isinstance(progress, dict) and set(progress) == {
            "schema", "phase", "processed_lab_rows_lower_bound_million", "elapsed_seconds",
            "patient_level_output_emitted", "model_inference_performed", "clustering_performed",
        })
        require(progress["schema"] == "bran-mimic-broad-admission-m1-progress"
                and progress["phase"] == "completed"
                and (progress["processed_lab_rows_lower_bound_million"] is None
                     or (type(progress["processed_lab_rows_lower_bound_million"]) is int
                         and progress["processed_lab_rows_lower_bound_million"] >= 0
                         and progress["processed_lab_rows_lower_bound_million"] % 1_000_000 == 0))
                and type(progress["elapsed_seconds"]) is float
                and math.isfinite(progress["elapsed_seconds"])
                and progress["elapsed_seconds"] >= 0.0
                and progress["patient_level_output_emitted"] is False
                and progress["model_inference_performed"] is False
                and progress["clustering_performed"] is False)
        terminal = _pending_terminal if _pending_terminal is not None else _read_json(OUT / "completed.json")
        _validate_terminal(terminal, sha(OUT / "protocol.json"), sha(OUT / "aggregate.json"))
        return aggregate, terminal
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def _rows_for_current_admission(row: dict[str, str]) -> dict[str, str]:
    return {key: row[key] for key in ("subject_id", "hadm_id", "admittime", "dischtime")}


def run(state: dict[str, object] | None = None) -> None:
    require(not OUT.exists() and not OUT.is_symlink())
    require(not PRIVATE.exists() and not PRIVATE.is_symlink())
    OUT.mkdir(mode=0o700)
    if state is not None:
        state["owned"] = True
        state["phase"] = "authentication"
    os.chmod(OUT, 0o700)
    started = time.monotonic()
    def progress(phase: str, bucket: int | None = None) -> None:
        if state is not None:
            state["phase"] = phase
        _progress(phase, bucket, start=started)
    try:
        progress("authentication")
        full_binding = dependency_binding()
        binding = _public(full_binding)
        mapping = full_binding.get("_mapping")
        approved_codes = full_binding.get("_approved_codes")
        if mapping is None:
            mapping = parent.mapping()
        if approved_codes is None:
            approved_codes = disease_dictionary.authenticate_audit()
        require(isinstance(mapping, dict) and isinstance(approved_codes, dict))
        protocol = _protocol(binding)
        validate_protocol(protocol, expected_binding=binding)
        write_json(OUT / "protocol.json", protocol)

        progress("source_hashes")
        source_hashes(binding)
        progress("metadata")
        metadata = source.predictor_metadata(
            _read_rows("patients", source.PATIENT_COLUMNS, LIMITS["patients"]),
            _read_rows("admissions", source.ADMISSION_COLUMNS, LIMITS["admissions"]),
        )
        accumulator = BroadSnapshotAccumulator(metadata)
        code_binding = authenticate_code_binding("mimic", mapping)

        def lab_rows():
            for number, row in enumerate(_read_rows("labs", source.LAB_COLUMNS, LIMITS["labs"]), 1):
                if number % 1_000_000 == 0:
                    progress("lab_scan", number)
                yield row

        progress("lab_scan", 0)
        stream = source.available_events(lab_rows(), metadata, code_binding)
        try:
            for item in stream:
                accumulator.add(item)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        snapshots = [snap for snap in accumulator.private_snapshots() if not _age_is_known_minor(snap.age)]
        require(snapshots)

        progress("admissions")
        outcome_rows = list(_read_rows("admissions", source.OUTCOME_COLUMNS, LIMITS["admissions"]))
        salt = SALT_PATH.read_bytes()
        require(len(salt) == 32)
        target_rows = []
        for snapshot in snapshots:
            target_rows.append({
                "person": snapshot.person,
                "episode": snapshot.episode,
                "row_binding": _row_binding(salt, snapshot.person, snapshot.episode),
            })
        progress("diagnoses")
        current_rows = [_rows_for_current_admission(row) for row in outcome_rows]
        current_membership = membership.assemble_current_membership(
            target_rows,
            current_rows,
            _read_rows("diagnoses", ("subject_id", "hadm_id", "icd_code", "icd_version"), LIMITS["diagnoses"]),
            approved_codes,
        )
        progress("selection")
        joined = landmark_join.assemble_landmark_join(target_rows, outcome_rows, current_membership)
        selected = np.flatnonzero(joined.selection.index_admission)
        require(len(selected) > 0 and len(np.unique([snapshots[index].person for index in selected])) == len(selected))
        selected_snapshots = [snapshots[index] for index in selected]
        arrays = {
            "values": np.asarray([snap.values for snap in selected_snapshots], dtype=np.float64),
            "observed": np.asarray([snap.observed for snap in selected_snapshots], dtype=bool),
            "specimen_minutes": np.asarray([snap.specimen_minutes for snap in selected_snapshots], dtype=np.float64),
            "available_minutes": np.asarray([snap.available_minutes for snap in selected_snapshots], dtype=np.float64),
            "age_triplet": np.asarray([_age_triplet(snap.age) for snap in selected_snapshots], dtype=np.float64),
            "age_kind": np.asarray([AGE_KINDS.index(snap.age.kind) for snap in selected_snapshots], dtype=np.uint8),
            "person": np.asarray([snap.person for snap in selected_snapshots], dtype="U64"),
            "episode": np.asarray([snap.episode for snap in selected_snapshots], dtype="U64"),
            "row_binding": np.asarray([target_rows[index]["row_binding"] for index in selected], dtype="U64"),
            "roles": np.asarray(joined.selection.roles[selected], dtype=np.uint8),
            "membership": np.asarray(joined.current_membership[selected], dtype=bool),
            "labels": np.asarray(joined.selection.labels[selected], dtype=np.int8),
            "outcome": np.asarray(joined.outcome_status[selected], dtype=np.int8),
        }
        validate_arrays(arrays)
        progress("private_artifact")
        private_path = _write_private(arrays)
        progress("post_authentication")
        source_hashes(binding)
        require(_public(dependency_binding()) == binding)
        require(code_hashes() == protocol["code_sha256"] and sha(DESIGN) == protocol["design_sha256"])
        aggregate = {
            "schema": "bran-mimic-broad-admission-m1",
            "status": "completed",
            "summary": summary(arrays, sha(private_path)),
            "private_sha256": {"cohort.npz": sha(private_path)},
            **FALSE_FLAGS,
        }
        # Keep summary fields at the top level so public consumers cannot
        # mistake a nested free-form object for a released row structure.
        safe_summary = aggregate.pop("summary")
        aggregate.update({key: safe_summary[key] for key in safe_summary if key != "private_sha256"})
        validate_result(aggregate)
        write_json(OUT / "aggregate.json", aggregate)
        progress("completed")
        terminal = {
            "status": "authenticated_completed",
            "protocol_sha256": sha(OUT / "protocol.json"),
            "aggregate_sha256": sha(OUT / "aggregate.json"),
            "patient_level_output_emitted": False,
        }
        # Authenticate while completed.json is absent; this is the exclusive
        # pending-terminal check required before committing success.
        authenticate(_pending_terminal=terminal)
        write_json(OUT / "completed.json", terminal)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    state: dict[str, object] = {"owned": False, "phase": "authentication"}
    ok = False
    terminal = None
    with quiet():
        try:
            with LOCK.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.audit_only:
                    _aggregate, terminal = authenticate()
                else:
                    run(state)
                ok = True
        except Exception as exc:
            if state["owned"] and OUT.is_dir() and not (OUT / "completed.json").exists():
                try:
                    allowed = set(code_hashes()) | set(getattr(linked, "CODE", ())) | {
                        "run_bran_mimic_broad_admission_m1.py"
                    }
                    write_json(OUT / "failure.json", {
                        "status": "technical_failure",
                        "phase": state["phase"],
                        "safe_exception_chain": safe_trace(exc, ROOT, allowed),
                        "patient_level_output_emitted": False,
                    })
                except Exception:
                    pass
    answer = terminal if ok and args.audit_only else {
        "status": "completed" if ok else "not_completed",
        "phase": state["phase"],
        "patient_level_output_emitted": False,
    }
    print(json.dumps(answer, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

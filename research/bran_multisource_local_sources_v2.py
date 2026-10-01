"""Private, fail-closed V2 source reconstruction after local receipt authentication.

This module is intentionally not a runner.  ``load_prepared_sources`` may only
be called by an FD-quiet caller which already owns the shared source lock.  It
returns private in-memory arrays and never prints, serializes, trains, or makes
source admission decisions.  Every supplied pool is rebuilt from the named
successful local attempt and checked against its whole-source receipt hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from bran_multisource_protocol_v2 import EVIDENCE_KEYS, is_hash
from bran_multisource_clinical_v2 import ClinicalPoolV2, bridge, eligibility_hash as clinical_eligibility_hash, safe_summary
from bran_multisource_nwicu_v2 import convert as convert_nwicu, eligibility_hash as nwicu_eligibility_hash, safe_summary as nwicu_summary, validate_pool as validate_nwicu_pool
from bran_multisource_batches_v2 import RetinalPoolV2


ROOT = Path(__file__).resolve().parent
ERROR = "multisource local source binding rejected"
MIMICIII_PROTOCOL_PIN = "ecc3673dc1c02998f9d3d93760e768afccacd8e388e6f6a5c6b0c879692abbac"
SOURCE_NAMES = ("aireadi", "mimiciii", "mimiciv", "eicu", "nhanes_exposed", "nwicu", "brset")


def require(ok: bool) -> None:
    if not ok:
        raise ValueError(ERROR) from None


def _sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink())
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> dict:
    try:
        require(path.is_file() and not path.is_symlink())
        value = json.loads(path.read_text())
        require(type(value) is dict)
        return value
    except Exception:
        require(False)


def _public_attempt(out: Path, *, require_audit: bool = False) -> tuple[dict, dict, dict | None]:
    """Authenticate aggregate/manifest (and, where required, audit) receipts."""
    try:
        require(out.is_dir() and not out.is_symlink())
        require(not os.path.lexists(out / "failure.json") and not os.path.lexists(out / "audit_failure.json"))
        aggregate = _json(out / "aggregate.json")
        manifest = _json(out / "manifest.json")
        require(manifest.get("aggregate_sha256") == _sha(out / "aggregate.json"))
        audit = None
        if require_audit:
            audit = _json(out / "audit.json")
            require(audit.get("aggregate_sha256") == _sha(out / "aggregate.json"))
        return aggregate, manifest, audit
    except Exception:
        require(False)


def _code_matches(receipt: object) -> None:
    require(type(receipt) is dict and receipt)
    for name, pin in receipt.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_]+\.py", name) is not None
                and is_hash(pin) and _sha(ROOT / name) == pin)


def _evidence(**roles: str) -> dict[str, str]:
    require(set(roles) == EVIDENCE_KEYS and all(is_hash(value) for value in roles.values()))
    return dict(roles)


def _combined_receipt_hash(**receipts: str) -> str:
    """Bind more than one whole-source receipt into one protocol hash role."""
    require(bool(receipts) and all(type(name) is str and is_hash(value)
                                   for name, value in receipts.items()))
    return _digest(dict(receipts))


def _readonly(value: object, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _retinal_pool(features: object, groups: object, ages: object) -> RetinalPoolV2:
    """Make a BRSET-only pool; numeric ages are retained only as documented values.

    The authenticated BRSET grouping artifact has a documented ``ages`` field.
    Non-numeric/missing values remain typed unknown; infinities and negative
    purported ages are source-contract failures rather than imputed values.
    """
    try:
        f = np.asarray(features)
        g = np.asarray(groups)
        a = np.asarray(ages)
        require(f.ndim == 2 and f.shape[1] == 384 and f.dtype == np.float32 and np.isfinite(f).all())
        n = f.shape[0]
        require(n > 0 and g.shape == (n,) and g.dtype.kind in "iu" and np.all(g >= 0))
        require(a.shape == (n,) and a.dtype.kind == "f")
        require(not np.isinf(a).any() and np.all(a[np.isfinite(a)] >= 0))
        reported = np.isfinite(a)
        value = np.full(n, np.nan, dtype=np.float64)
        value[reported] = a[reported]
        unknown = np.full(n, np.nan, dtype=np.float64)
        kind = np.where(reported, 0, 3).astype(np.int64)
        return RetinalPoolV2("brset", _readonly(f, np.float32), _readonly(g, np.int64),
                             _readonly(value, np.float64), _readonly(unknown, np.float64),
                             _readonly(unknown, np.float64), _readonly(kind, np.int64))
    except Exception:
        require(False)


def _load_paired() -> tuple[object, dict[str, str], dict[str, object]]:
    import run_bran_multisource_paired_binding_v2 as paired

    aggregate, manifest, _ = _public_attempt(paired.OUT)
    require(aggregate.get("schema") == "bran-multisource-paired-binding-v2"
            and aggregate.get("status") == "paired_source_and_five_fold_transforms_authenticated"
            and aggregate.get("training_started") is False
            and manifest == {"aggregate_sha256": _sha(paired.OUT / "aggregate.json"),
                             "patient_level_output_emitted": False})
    _code_matches(aggregate.get("code_sha256"))
    value = paired.load_private()
    require(isinstance(value, paired.PairedSourceV2) and value.receipt == aggregate)
    evidence = _evidence(
        qualification_sha256=aggregate["source_receipt_sha256"],
        canonical_input_sha256=_combined_receipt_hash(
            clinical_source_receipt_sha256=aggregate["source_receipt_sha256"],
            retinal_binding_aggregate_sha256=aggregate["retinal_binding"]["aggregate_sha256"],
        ),
        eligibility_sha256=aggregate["fold_array_sha256"],
        grouping_sha256=aggregate["outer_fold_sha256"],
        exposure_audit_sha256=aggregate["retinal_binding"]["audit_sha256"],
    )
    pins = {"attempt": "BRAN_MULTISOURCE_PAIRED_BINDING_V2_ATTEMPT1",
            "aggregate_sha256": _sha(paired.OUT / "aggregate.json"),
            "code_sha256": dict(aggregate["code_sha256"])}
    return value, evidence, pins


def _load_clinical() -> tuple[dict[str, ClinicalPoolV2], dict[str, dict[str, str]], dict[str, object]]:
    import run_bran_native_source_qualification_v1 as legacy
    import run_bran_multisource_clinical_binding_v2 as binding
    from run_bran_multisource_inventory_v2 import verify_legacy_native_metadata

    aggregate, manifest, _ = _public_attempt(binding.OUT)
    require(aggregate.get("schema") == "bran-multisource-clinical-binding-v2"
            and aggregate.get("status") == "sources_reauthenticated_and_v2_eligibility_bound"
            and aggregate.get("training_started") is False
            and manifest == {"aggregate_sha256": _sha(binding.OUT / "aggregate.json"),
                             "patient_level_output_emitted": False})
    _code_matches(aggregate.get("code_sha256"))
    require(_sha(legacy.PROTOCOL) == aggregate.get("legacy_protocol_sha256"))
    protocol = _json(legacy.PROTOCOL)
    legacy.validate_protocol(protocol)
    require(verify_legacy_native_metadata(ROOT).get("status") == "legacy_artifact_links_authenticated")
    expected = {"mimic": "mimiciv", "eicu": "eicu", "nhanes": "nhanes_exposed"}
    require(type(aggregate.get("sources")) is dict and set(aggregate["sources"]) == set(expected.values()))
    pools: dict[str, ClinicalPoolV2] = {}
    evidence: dict[str, dict[str, str]] = {}
    for legacy_name, source_name in expected.items():
        receipt = protocol["sources"][legacy_name]
        arrays = legacy.load_one(legacy_name, receipt)
        pool = bridge(legacy_name, arrays)
        item = aggregate["sources"][source_name]
        require(safe_summary(pool) == item.get("summary")
                and clinical_eligibility_hash(pool) == item.get("eligibility_sha256"))
        require(type(item.get("legacy_source_receipt")) is dict and item["legacy_source_receipt"] == receipt)
        pools[source_name] = pool
        evidence[source_name] = _evidence(
            qualification_sha256=receipt["aggregate_sha256"],
            canonical_input_sha256=receipt["private_cache_sha256"],
            eligibility_sha256=item["eligibility_sha256"],
            grouping_sha256=receipt["private_cache_sha256"],
            exposure_audit_sha256=receipt["manifest_sha256"],
        )
    legacy.validate_protocol(protocol)
    require(verify_legacy_native_metadata(ROOT).get("status") == "legacy_artifact_links_authenticated")
    pins = {"attempt": "BRAN_MULTISOURCE_CLINICAL_BINDING_V2_ATTEMPT1",
            "aggregate_sha256": _sha(binding.OUT / "aggregate.json"),
            "legacy_protocol_sha256": aggregate["legacy_protocol_sha256"],
            "code_sha256": dict(aggregate["code_sha256"])}
    return pools, evidence, pins


def _load_nwicu() -> tuple[ClinicalPoolV2, dict[str, str], dict[str, object]]:
    import run_bran_nwicu_cohort_adult_v1 as legacy
    import run_bran_multisource_nwicu_binding_v2 as binding

    aggregate, manifest, _ = _public_attempt(binding.OUT)
    require(aggregate.get("schema") == "bran-multisource-nwicu-binding-v2"
            and aggregate.get("status") == "completed_aggregate_only"
            and aggregate.get("new_prospective_training_admission") is True
            and aggregate.get("legacy_qualification_training_permitted") is False
            and aggregate.get("training_started") is False
            and manifest == {"aggregate_sha256": _sha(binding.OUT / "aggregate.json"),
                             "patient_level_output_emitted": False})
    _code_matches(aggregate.get("code_sha256"))
    protocol_pin, audit_pin = aggregate.get("legacy_protocol_sha256"), aggregate.get("legacy_audit_sha256")
    require(is_hash(protocol_pin) and is_hash(audit_pin))
    legacy.verify_audit(protocol_pin, audit_pin)
    receipt = legacy.source_receipt()
    require(_digest(receipt) == aggregate.get("source_receipt_sha256"))
    pool = convert_nwicu(legacy.load_private(receipt))
    validate_nwicu_pool(pool)
    require(nwicu_summary(pool) == aggregate.get("safe_summary")
            and nwicu_eligibility_hash(pool) == aggregate.get("eligibility_sha256"))
    legacy.verify_audit(protocol_pin, audit_pin)
    require(legacy.source_receipt() == receipt)
    evidence = _evidence(
        qualification_sha256=protocol_pin,
        canonical_input_sha256=receipt["private_cache_sha256"],
        eligibility_sha256=aggregate["eligibility_sha256"],
        grouping_sha256=receipt["pool_manifest_sha256"],
        exposure_audit_sha256=audit_pin,
    )
    pins = {"attempt": "BRAN_MULTISOURCE_NWICU_BINDING_V2_ATTEMPT1",
            "aggregate_sha256": _sha(binding.OUT / "aggregate.json"),
            "code_sha256": dict(aggregate["code_sha256"])}
    return pool, evidence, pins


def _load_mimiciii() -> tuple[ClinicalPoolV2, dict[str, str], dict[str, object]]:
    import run_bran_mimiciii_observations_v2 as runner
    from bran_mimiciii_observations_v2 import MimicIIIObservationArrays, validate_observations

    out, private = runner.paths(1)
    aggregate, manifest, audit = _public_attempt(out, require_audit=True)
    require(aggregate.get("schema") == "bran-mimiciii-observations-v2"
            and aggregate.get("status") == "observed_cache_materialized_not_trained"
            and aggregate.get("protocol_sha256") == MIMICIII_PROTOCOL_PIN
            and aggregate.get("training_started") is False
            and aggregate.get("new_encoder_trained") is False
            and manifest == {"aggregate_sha256": _sha(out / "aggregate.json"),
                             "protocol_sha256": MIMICIII_PROTOCOL_PIN,
                             "patient_level_output_emitted": False}
            and audit == {"status": "authenticated", "protocol_sha256": MIMICIII_PROTOCOL_PIN,
                          "aggregate_sha256": _sha(out / "aggregate.json"),
                          "private_cache_sha256": aggregate.get("private_cache_sha256"),
                          "cache_schema_and_training_eligibility_replayed": True,
                          "full_source_event_reextraction_replayed": False,
                          "patient_level_output_emitted": False, "training_started": False})
    protocol, _ = runner.authenticate(1, MIMICIII_PROTOCOL_PIN)
    require(_sha(out / "protocol.json") == MIMICIII_PROTOCOL_PIN
            and protocol.get("code_sha256") == runner.code_hashes())
    cache = private / "observations.npz"
    require(cache.is_file() and not cache.is_symlink() and cache.stat().st_nlink == 1
            and cache.stat().st_mode & 0o777 == 0o600 and _sha(cache) == aggregate.get("private_cache_sha256"))
    with np.load(cache, allow_pickle=False) as archive:
        require(set(archive.files) == set(MimicIIIObservationArrays.__dataclass_fields__))
        arrays = {name: archive[name].copy() for name in archive.files}
    for value in arrays.values():
        value.setflags(write=False)
    observations = MimicIIIObservationArrays(**arrays)
    validate_observations(observations)
    pool = runner.training_pool(observations)
    require(safe_summary(pool) == aggregate.get("training_summary")
            and clinical_eligibility_hash(pool) == aggregate.get("eligibility_sha256"))
    require(runner.authenticate(1, MIMICIII_PROTOCOL_PIN)[0] == protocol)
    evidence = _evidence(
        qualification_sha256=MIMICIII_PROTOCOL_PIN,
        canonical_input_sha256=aggregate["private_cache_sha256"],
        eligibility_sha256=aggregate["eligibility_sha256"],
        grouping_sha256=aggregate["private_cache_sha256"],
        exposure_audit_sha256=_sha(out / "audit.json"),
    )
    pins = {"attempt": "BRAN_MIMICIII_OBSERVATIONS_V2_ATTEMPT1",
            "aggregate_sha256": _sha(out / "aggregate.json"),
            "protocol_sha256": MIMICIII_PROTOCOL_PIN,
            "code_sha256": dict(protocol["code_sha256"])}
    return pool, evidence, pins


def _load_brset() -> tuple[RetinalPoolV2, dict[str, str], dict[str, object]]:
    import run_bran_multisource_retinal_features_v2 as runner

    out, private = runner.paths(2)
    aggregate, manifest, audit = _public_attempt(out, require_audit=True)
    protocol = _json(out / "protocol.json")
    require(protocol.get("schema") == "bran-multisource-retinal-features-v2"
            and protocol.get("status") == "frozen_before_feature_extraction"
            and protocol.get("parameters") == runner.PARAMETERS
            and protocol.get("unadapted_dino_cache_reused") is False
            and protocol.get("v2_model_trained") is False)
    _code_matches(protocol.get("code_sha256"))
    require(private.is_dir() and not private.is_symlink() and private.stat().st_mode & 0o777 == 0o700)
    # Existing runner authentication recomputes source membership and frozen
    # backbone pins.  It does not construct an encoder or replay inference.
    backend = runner.LocalBackend()
    authenticated_protocol, data, use = runner.authenticate_spec(out, backend)
    require(authenticated_protocol == protocol and isinstance(data, dict)
            and type(use) is np.ndarray and use.ndim == 1 and use.dtype.kind in "iu"
            and len(use) > 0 and len(np.unique(use)) == len(use) and np.all(use >= 0))
    source = protocol.get("source")
    require(type(source) is dict and set(source) == {"admission_protocol_sha256", "admission_audit_sha256",
            "membership_sha256", "eye_checkpoint_sha256", "eye_contract_sha256", "images_lower_bound_20",
            "source_local_people_lower_bound_20"}
            and all(is_hash(source[name]) for name in ("admission_protocol_sha256", "admission_audit_sha256",
                "membership_sha256", "eye_checkpoint_sha256", "eye_contract_sha256")))
    require(aggregate == {"schema": "bran-multisource-retinal-features-v2",
        "status": "features_extracted_pending_audit", "protocol_sha256": _sha(out / "protocol.json"),
        "features_sha256": aggregate.get("features_sha256"), "grouping_sha256": aggregate.get("grouping_sha256"),
        "dimension": 384, "dtype": "float32", "elapsed_seconds": aggregate.get("elapsed_seconds"),
        "images_lower_bound_20": source["images_lower_bound_20"],
        "source_local_people_lower_bound_20": source["source_local_people_lower_bound_20"],
        "model_training_started": False, "patient_level_output_emitted": False}
        and is_hash(aggregate["features_sha256"]) and is_hash(aggregate["grouping_sha256"])
        and type(aggregate["elapsed_seconds"]) in (int, float) and np.isfinite(aggregate["elapsed_seconds"])
        and aggregate["elapsed_seconds"] >= 0
        and manifest == {"protocol_sha256": _sha(out / "protocol.json"),
                         "aggregate_sha256": _sha(out / "aggregate.json")}
        and audit == {"schema": "bran-multisource-retinal-features-audit-v2", "status": "authenticated",
                      "protocol_sha256": _sha(out / "protocol.json"), "aggregate_sha256": _sha(out / "aggregate.json"),
                      "manifest_sha256": _sha(out / "manifest.json"), "features_sha256": aggregate["features_sha256"],
                      "grouping_sha256": aggregate["grouping_sha256"], "selected_image_replay_passed": True,
                      "model_training_started": False, "patient_level_output_emitted": False})
    features_path, grouping_path = private / "features.npy", private / "grouping.npz"
    for path, pin in ((features_path, aggregate["features_sha256"]), (grouping_path, aggregate["grouping_sha256"])):
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
                and path.stat().st_mode & 0o777 == 0o600 and _sha(path) == pin)
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    with np.load(grouping_path, allow_pickle=False) as grouped:
        require(set(grouped.files) == {"source_rows", "groups", "ages"})
        source_rows, groups, ages = (grouped[name].copy() for name in ("source_rows", "groups", "ages"))
    require(features.shape == (len(use), 384) and source_rows.shape == (len(use),)
            and source_rows.dtype.kind in "iu" and len(np.unique(source_rows)) == len(source_rows)
            and groups.shape == (len(use),) and groups.dtype.kind in "iu" and np.all(groups >= 0)
            and ages.shape == (len(use),) and ages.dtype.kind == "f")
    source_arrays = data.get("arrays")
    require(type(source_arrays) is dict and all(name in source_arrays for name in ("source_rows", "groups", "ages")))
    require(np.array_equal(source_rows, source_arrays["source_rows"][use])
            and np.array_equal(groups, source_arrays["groups"][use])
            and np.array_equal(ages, source_arrays["ages"][use], equal_nan=True))
    feature_copy = np.array(features, dtype=np.float32, copy=True)
    # Copying is private; rehash the immutable backing artifacts immediately.
    require(_sha(features_path) == aggregate["features_sha256"]
            and _sha(grouping_path) == aggregate["grouping_sha256"])
    pool = _retinal_pool(feature_copy, groups, ages)
    require(runner.authenticate_spec(out, backend)[0] == protocol)
    evidence = _evidence(
        qualification_sha256=source["admission_protocol_sha256"],
        canonical_input_sha256=aggregate["features_sha256"],
        eligibility_sha256=source["membership_sha256"],
        grouping_sha256=aggregate["grouping_sha256"],
        exposure_audit_sha256=_sha(out / "audit.json"),
    )
    pins = {"attempt": "BRAN_MULTISOURCE_RETINAL_FEATURES_V2_ATTEMPT2",
            "aggregate_sha256": _sha(out / "aggregate.json"), "protocol_sha256": _sha(out / "protocol.json"),
            "frozen_eye_checkpoint_sha256": source["eye_checkpoint_sha256"],
            "frozen_eye_contract_sha256": source["eye_contract_sha256"],
            "code_sha256": dict(protocol["code_sha256"])}
    return pool, evidence, pins


@dataclass(frozen=True, repr=False)
class PreparedSourcesV2:
    paired: object
    pools: MappingProxyType
    source_evidence: MappingProxyType
    artifact_pins: MappingProxyType


def load_prepared_sources() -> PreparedSourcesV2:
    """Rebuild exactly the seven named V2 sources in caller-owned quiet/lock scope."""
    try:
        paired, paired_evidence, paired_pins = _load_paired()
        clinical, clinical_evidence, clinical_pins = _load_clinical()
        nwicu, nwicu_evidence, nwicu_pins = _load_nwicu()
        mimiciii, mimiciii_evidence, mimiciii_pins = _load_mimiciii()
        brset, brset_evidence, brset_pins = _load_brset()
        pools = {"mimiciii": mimiciii, **clinical, "nwicu": nwicu, "brset": brset}
        evidence = {"aireadi": paired_evidence, "mimiciii": mimiciii_evidence,
                    **clinical_evidence, "nwicu": nwicu_evidence, "brset": brset_evidence}
        pins = {"aireadi": paired_pins, "mimiciii": mimiciii_pins,
                **{source: clinical_pins for source in clinical},
                "nwicu": nwicu_pins, "brset": brset_pins}
        require(set(pools) | {"aireadi"} == set(SOURCE_NAMES)
                and set(evidence) == set(SOURCE_NAMES) and set(pins) == set(SOURCE_NAMES))
        require(all(isinstance(pool, (ClinicalPoolV2, RetinalPoolV2)) for pool in pools.values()))
        require(all(type(value) is dict and set(value) == EVIDENCE_KEYS
                    and all(is_hash(pin) for pin in value.values()) for value in evidence.values()))
        return PreparedSourcesV2(paired, MappingProxyType(dict(pools)),
                                 MappingProxyType({key: MappingProxyType(dict(value)) for key, value in evidence.items()}),
                                 MappingProxyType({key: MappingProxyType(dict(value)) for key, value in pins.items()}))
    except Exception:
        require(False)

"""Run the first aggregate-only AI-READI disease-target availability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, Mapping, Sequence

from patient_atlas_disease_targets import load_development_disease_targets
from patient_atlas_real_data import load_exploratory_raw_cohort
from run_patient_atlas_exploratory_stage2 import _redacted_error, _write_exclusive_json


SCHEMA_VERSION = "patient-atlas-v5-disease-target-audit-run-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _contains_exact_patient_identity(
    payload: Any,
    patient_ids: Sequence[str],
) -> bool:
    """Reject identity-valued JSON leaves without confusing numeric counts for IDs.

    AI-READI identities can be numeric strings.  Searching serialized JSON by substring
    therefore falsely treats ordinary aggregate counts as identifiers.  This recursive
    check examines only JSON string keys and values; the audit report has no free-text
    fields derived from patient rows.
    """

    identities = frozenset(str(value) for value in patient_ids)

    def visit(value: Any) -> bool:
        if isinstance(value, str):
            return value in identities
        if isinstance(value, MappingABC):
            return any(
                (isinstance(key, str) and key in identities) or visit(item)
                for key, item in value.items()
            )
        if isinstance(value, SequenceABC) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return any(visit(item) for item in value)
        return False

    return visit(payload)


def run_disease_target_audit(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    clinical_project_root: str | Path,
    output_path: str | Path,
    failure_path: str | Path,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    failure = Path(failure_path).resolve()
    if output.exists() or failure.exists():
        raise FileExistsError("disease target audit outputs must be new")
    patient_ids: tuple[str, ...] = ()
    started = time.perf_counter()
    try:
        cohort = load_exploratory_raw_cohort(
            project_root=root,
            dataset_root=dataset_root,
            clinical_project_root=clinical_project_root,
        )
        patient_ids = cohort.patient_ids
        targets, policy = load_development_disease_targets(
            project_root=root,
            dataset_root=dataset_root,
            cohort=cohort,
        )
        audit = targets.aggregate_summary(policy)
        passing = sorted(
            column
            for column, record in audit["targets"].items()
            if record["total_gate_pass"] is True
        )
        blocked = {
            column: str(record["gate_reason"])
            for column, record in sorted(audit["targets"].items())
            if record["total_gate_pass"] is not True
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "source_and_total_eligibility_audit_complete_fold_freeze_required",
            "runtime_seconds": time.perf_counter() - started,
            "scope": "official_train_validation_only_no_official_test_target_access",
            "target_audit": audit,
            "execution_decision": {
                "total_gate_passing_target_columns": passing,
                "total_gate_blocked_target_columns": blocked,
                "fold_specific_eligibility_pending": True,
                "disease_model_scoring_allowed": False,
                "next_required_gate": "freeze canonical outer fold map and per-fold class counts",
            },
            "bindings": {
                "target_source_policy_sha256": targets.source_policy_sha256,
                "disease_registry_sha256": _sha256(
                    root / "PATIENT_ATLAS_V5_DISEASE_UTILITY_BENCHMARK_REGISTRY_V1.json"
                ),
                "target_loader_sha256": _sha256(root / "patient_atlas_disease_targets.py"),
                "audit_runner_sha256": _sha256(
                    root / "run_patient_atlas_v5_disease_target_audit.py"
                ),
            },
            "privacy": {
                "patient_derived_processing": "local_only",
                "patient_rows_emitted": False,
                "patient_identifiers_emitted": False,
                "target_values_emitted": False,
                "predictions_emitted": False,
                "small_cells_suppressed": True,
            },
            "claim_limits": {
                "performance_evaluated": False,
                "external_validation_completed": False,
                "clinical_benefit_claim_allowed": False,
            },
        }
        json.dumps(report, sort_keys=True, allow_nan=False)
        if _contains_exact_patient_identity(report, patient_ids):
            raise RuntimeError("aggregate disease target report contains a patient identity")
        _write_exclusive_json(output, report)
        return report
    except BaseException as error:
        _write_exclusive_json(
            failure,
            {
                "schema_version": "patient-atlas-v5-disease-target-audit-failure-v1",
                "terminal_for_attempt": True,
                "same_policy_technical_retry_allowed": True,
                "error_type": type(error).__name__,
                "error_message": _redacted_error(error, patient_ids),
                "runtime_seconds": time.perf_counter() - started,
                "official_test_targets_loaded": False,
                "patient_rows_identifiers_or_target_values_emitted": False,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_disease_target_audit(
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            clinical_project_root=args.clinical_project_root,
            output_path=args.output,
            failure_path=args.failure,
        )
    except BaseException:
        print(
            json.dumps(
                {
                    "event": "patient_atlas_v5_disease_target_audit_failed",
                    "details_emitted": False,
                    "failure_artifact_written": args.failure.is_file(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "event": "patient_atlas_v5_disease_target_audit_completed",
                "status": report["status"],
                "patient_details_emitted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "_contains_exact_patient_identity",
    "run_disease_target_audit",
]

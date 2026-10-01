"""Build and verify a reviewed, code-only historical BRAN research snapshot.

This tool parses source text with ast. It never imports a project module or
opens a source dataset. The reviewed hash list is an explicit allowlist.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil


RELEASE = Path(__file__).resolve().parents[1]
SOURCE = RESEARCH = RELEASE / "research"
REVIEWED = RELEASE / "tools" / "reviewed_sources.txt"
MANIFEST = RESEARCH / "snapshot_manifest.json"
ENTRYPOINTS = (
    "run_bran_robust_clinical_r7.py",
    "run_bran_robust_clinical_evaluation_r7.py",
    "run_bran_r7_modality_atlas_a3.py",
    "run_bran_r7_information_matched_f1.py",
    "run_bran_mimic_broad_state_m2.py",
    "bran_r7_missingness_figure_m2.py",
    "run_bran_r7_completion_utility_u2.py",
    "run_bran_r7_quantile_q4.py",
    "run_bran_r7_clinical_s4.py",
    "run_bran_expanded_clinical_s7.py",
    "run_bran_r7_inspire_e1.py",
    "run_bran_hirid_r7_h6.py",
)
SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\bsk-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bAKIA[0-9A-Z]{16}\b"
)
PATIENT_KEY_NAMES = (
    "subject_id", "patient_id", "hadm_id", "stay_id", "person_id", "episode_id"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_path(name: str) -> Path:
    if not name.endswith(".py") or Path(name).name != name:
        raise ValueError("non_module_path")
    path = SOURCE / name
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing_or_linked_module")
    return path


def local_name(module: str) -> str | None:
    first = module.split(".", 1)[0]
    candidate = first + ".py"
    return candidate if (SOURCE / candidate).is_file() else None


def risk_classes(tree: ast.AST) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if SECRET.search(node.value):
                found.add("credential_literal")
        if isinstance(node, ast.Compare):
            parts = [node.left, *node.comparators]
            names = [part.id.lower() for part in parts if isinstance(part, ast.Name)]
            literals = [part.value for part in parts if isinstance(part, ast.Constant)]
            if any(any(key in name for key in PATIENT_KEY_NAMES) for name in names):
                if any((type(value) is int and value >= 10000)
                       or (type(value) is str and value.isdecimal() and len(value) >= 5)
                       for value in literals):
                    found.add("hardcoded_patient_key_comparison")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) >= 200:
            if all(isinstance(item, ast.Constant) and
                   type(item.value) in (int, float) for item in node.elts):
                found.add("large_numeric_literal_sequence")
    return sorted(found)


def inspect_module(name: str) -> dict:
    data = source_path(name).read_bytes()
    tree = ast.parse(data, filename=name)
    imports: set[str] = set()
    unresolved_dynamic: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = local_name(alias.name)
                if local:
                    imports.add(local)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            local = local_name(node.module)
            if local:
                imports.add(local)
        elif isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "__import__") or
            (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
        ):
            literal = (node.args[0].value if node.args and
                       isinstance(node.args[0], ast.Constant) and
                       isinstance(node.args[0].value, str) else None)
            if literal is None:
                unresolved_dynamic.add(node.lineno)
            else:
                local = local_name(literal)
                if local:
                    imports.add(local)
    return {
        "sha256": digest(data), "bytes": len(data),
        "local_imports": sorted(imports),
        "unresolved_dynamic_import_lines": sorted(unresolved_dynamic),
        "risk_classes": risk_classes(tree),
    }


def inventory() -> dict:
    modules: dict[str, dict] = {}

    def visit(name: str) -> None:
        if name in modules:
            return
        info = inspect_module(name)
        modules[name] = info
        for dependency in info["local_imports"]:
            visit(dependency)

    for entry in ENTRYPOINTS:
        visit(entry)
    per_entrypoint = {}
    for entry in ENTRYPOINTS:
        reached: set[str] = set()
        todo = [entry]
        while todo:
            name = todo.pop()
            if name not in reached:
                reached.add(name)
                todo.extend(modules[name]["local_imports"])
        per_entrypoint[entry] = sorted(reached)
    return {
        "schema": "bran-research-code-snapshot-v1",
        "scope": "reviewed_python_source_only_no_patient_data_or_fitted_artifacts",
        "entrypoints": list(ENTRYPOINTS),
        "module_count": len(modules),
        "source_bytes": sum(item["bytes"] for item in modules.values()),
        "modules": {name: modules[name] for name in sorted(modules)},
        "entrypoint_dependencies": per_entrypoint,
    }


def reviewed_hashes() -> dict[str, str]:
    lines = REVIEWED.read_text(encoding="ascii").splitlines()
    result: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split(" ")
        if len(fields) != 2 or len(fields[0]) != 64 or fields[1] in result:
            raise ValueError("invalid_reviewed_allowlist")
        result[fields[1]] = fields[0]
    return result


def validate_review(plan: dict) -> None:
    expected = {name: info["sha256"] for name, info in plan["modules"].items()}
    if reviewed_hashes() != expected:
        raise ValueError("unreviewed_or_changed_source")
    flagged = {name: info["risk_classes"] for name, info in plan["modules"].items()
               if info["risk_classes"]}
    if flagged:
        raise ValueError("flagged_source_not_admitted")


def build() -> dict:
    plan = inventory()
    validate_review(plan)
    RESEARCH.mkdir(exist_ok=True)
    expected = set(plan["modules"]) | {MANIFEST.name}
    existing = {item.name for item in RESEARCH.iterdir()}
    if existing - expected:
        raise ValueError("unexpected_research_file")
    for name, info in plan["modules"].items():
        target = RESEARCH / name
        if target.exists() and (target.is_symlink() or digest(target.read_bytes()) != info["sha256"]):
            raise ValueError("existing_snapshot_drift")
        if not target.exists():
            shutil.copyfile(source_path(name), target)
        if digest(target.read_bytes()) != info["sha256"]:
            raise ValueError("copy_mismatch")
    rendered = (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if MANIFEST.exists() and MANIFEST.read_bytes() != rendered:
        raise ValueError("existing_manifest_drift")
    if not MANIFEST.exists():
        MANIFEST.write_bytes(rendered)
    return plan


def verify() -> dict:
    plan = inventory()
    validate_review(plan)
    if not RESEARCH.is_dir() or RESEARCH.is_symlink():
        raise ValueError("missing_research_snapshot")
    expected = set(plan["modules"]) | {MANIFEST.name}
    if {item.name for item in RESEARCH.iterdir()} != expected:
        raise ValueError("snapshot_file_set_changed")
    for name, info in plan["modules"].items():
        path = RESEARCH / name
        if not path.is_file() or path.is_symlink() or digest(path.read_bytes()) != info["sha256"]:
            raise ValueError("snapshot_file_changed")
    if json.loads(MANIFEST.read_text(encoding="utf-8")) != plan:
        raise ValueError("manifest_changed")
    return plan


def main() -> None:
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scan", "build", "verify"))
    parser.add_argument("--source", type=Path, help="explicit original code directory for scan or build")
    args = parser.parse_args()
    if args.mode in {"scan", "build"}:
        if args.source is None or not args.source.is_dir():
            parser.error("scan and build require --source pointing to original code")
        SOURCE = args.source.resolve()
    elif args.source is not None:
        parser.error("verify operates only on this repository snapshot")
    plan = inventory() if args.mode == "scan" else build() if args.mode == "build" else verify()
    summary = {
        "module_count": plan["module_count"], "source_bytes": plan["source_bytes"],
        "entrypoint_counts": {name: len(items) for name, items in
                              plan["entrypoint_dependencies"].items()},
        "flagged_modules": {name: info["risk_classes"] for name, info in
                            plan["modules"].items() if info["risk_classes"]},
        "unresolved_dynamic_modules": sorted(name for name, info in
                                             plan["modules"].items()
                                             if info["unresolved_dynamic_import_lines"]),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

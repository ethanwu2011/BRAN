"""Check a code-only release without importing research runners or reading data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".csv", ".tsv",
    ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".dcm", ".jpg",
    ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".pptx", ".zip", ".tar",
    ".gz", ".sas7bdat", ".feather", ".arrow", ".log", ".sqlite", ".db", ".svg", ".jsonl",
}
BLOCKED_PARTS = {"private_artifacts", "data", "datasets", "checkpoints", "outputs"}
SKIP_PARTS = {".git", ".venv", "__pycache__", "build", "dist", ".pytest_cache"}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def check_files(paths: list[str]) -> list[dict[str, str]]:
    findings = []
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink():
            findings.append({"file": relative, "reason": "symlink_not_allowed"})
            continue
        if not path.is_file():
            findings.append({"file": relative, "reason": "missing_file"})
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES or set(Path(relative).parts) & BLOCKED_PARTS:
            findings.append({"file": relative, "reason": "data_or_binary_artifact_not_allowed"})
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"file": relative, "reason": "non_text_file_not_allowed"})
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            findings.append({"file": relative, "reason": "credential_pattern"})
    return findings


def check_core_hashes() -> list[dict[str, str]]:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    findings = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            findings.append({"file": item["path"], "reason": "core_source_hash_mismatch"})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-tree", action="store_true")
    args = parser.parse_args()
    if args.working_tree:
        paths = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                       if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP_PARTS
                       and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts))
    else:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
        paths = [p for p in result.stdout.decode().split("\0") if p]
    findings = check_files(paths) + check_core_hashes()
    print(json.dumps({"files_checked": len(paths), "findings": findings,
                      "scope": "File types, selected credential patterns and core source hashes only",
                      "manual_privacy_review_still_required": True}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

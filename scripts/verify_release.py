"""Check a code-only release without importing research runners or reading data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".csv", ".tsv",
    ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".dcm", ".jpg",
    ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".pptx", ".zip", ".tar",
    ".gz", ".sas7bdat", ".feather", ".arrow", ".log", ".sqlite", ".db", ".svg", ".jsonl",
}
BLOCKED_PARTS = {"private_artifacts", "data", "datasets", "checkpoints", "outputs"}
SKIP_PARTS = {".git", ".venv", ".venv-figures", "__pycache__", "build", "dist", ".pytest_cache"}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
# Absolute paths into a personal home folder or an external volume reveal a local directory layout.
ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?<![\w.~-])(?:/Users/|/home/|/Volumes/|[A-Za-z]:\\{1,2}Users\\{1,2})[A-Za-z0-9_][^/\\\s\"']*")
# research/ is a hash-locked historical snapshot. Its absolute paths are reported and documented in
# docs/privacy_and_access.md, not rejected, because editing those files would break the snapshot hashes.
HISTORICAL_PREFIX = "research/"
# Source Data of the paper figures. Each file, the binary workbooks included, must be listed with its SHA-256 in the
# Source Data manifest. Workbook text is read from the XML parts and scanned like any other text.
SOURCE_DATA_DIR = "paper/source_data/"
SOURCE_DATA_MANIFEST = SOURCE_DATA_DIR + "SOURCE_DATA_MANIFEST.json"
SOURCE_DATA_UNLISTED_OK = {SOURCE_DATA_MANIFEST, SOURCE_DATA_DIR + "README.md"}
# Generated figure outputs. Git ignores this folder, so the working-tree scan skips it as well.
GENERATED_PREFIXES = ("paper/output/",)


def source_data_hashes() -> dict[str, str]:
    path = ROOT / SOURCE_DATA_MANIFEST
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for item in manifest["files"]:
        for kind in ("json", "xlsx"):
            out[SOURCE_DATA_DIR + item[kind]["path"]] = item[kind]["sha256"]
    return out


def workbook_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "\n".join(archive.read(name).decode("utf-8", errors="replace") for name in archive.namelist()
                         if name.endswith((".xml", ".rels")))


def check_files(paths: list[str], historical: dict[str, int] | None = None) -> list[dict[str, str]]:
    findings = []
    listed = source_data_hashes()
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
        if relative.startswith(SOURCE_DATA_DIR) and relative not in SOURCE_DATA_UNLISTED_OK:
            if relative not in listed:
                findings.append({"file": relative, "reason": "source_data_file_not_in_manifest"})
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != listed[relative]:
                findings.append({"file": relative, "reason": "source_data_hash_mismatch"})
        if relative.startswith(SOURCE_DATA_DIR) and path.suffix.lower() == ".xlsx":
            try:
                content = workbook_text(path)
            except zipfile.BadZipFile:
                findings.append({"file": relative, "reason": "unreadable_workbook"})
                continue
        else:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append({"file": relative, "reason": "non_text_file_not_allowed"})
                continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            findings.append({"file": relative, "reason": "credential_pattern"})
        paths_found = len(ABSOLUTE_LOCAL_PATH.findall(content))
        if paths_found:
            if relative.startswith(HISTORICAL_PREFIX):
                if historical is not None:
                    historical[relative] = paths_found
            else:
                findings.append({"file": relative, "reason": "absolute_local_path"})
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
                       and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)
                       and not str(p.relative_to(ROOT)).startswith(GENERATED_PREFIXES))
    else:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
        paths = [p for p in result.stdout.decode().split("\0") if p]
    historical: dict[str, int] = {}
    findings = check_files(paths, historical) + check_core_hashes()
    missing = sorted(set(source_data_hashes()) - set(paths))
    findings += [{"file": name, "reason": "source_data_file_missing"} for name in missing]
    print(json.dumps({"files_checked": len(paths), "findings": findings,
                      "historical_absolute_paths": {
                          "files": len(historical), "occurrences": sum(historical.values()),
                          "per_file": dict(sorted(historical.items())),
                          "status": "documented in docs/privacy_and_access.md, kept by author decision"},
                      "scope": "File types, selected credential patterns, absolute local paths, core source hashes and "
                               "Source Data hashes only",
                      "manual_privacy_review_still_required": True}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

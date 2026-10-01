"""Reconstruct the adapted eye tower's external source manifest locally.

The manifest contains only cryptographic hashes, logical source codes, sizes,
and perceptual hashes. It never stores paths, patient identifiers, images, or
AI-READI row-level material. The JSON report is aggregate-only.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy.fft import dctn


SCHEMA_VERSION = "patient-atlas-eye-source-provenance-reconstruction-v1"
MANIFEST_SCHEMA = "patient-atlas-external-eye-file-manifest-v1"
EXPECTED_SOURCE_COUNTS = {
    "drunified": 92501,
    "brset": 16266,
    "odir": 4512,
    "jsiec": 1994,
}
SOURCE_ORDER = tuple(EXPECTED_SOURCE_COUNTS)
SMALL_CELL_THRESHOLD = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _safe_count(value: int) -> int | str:
    value = int(value)
    if value == 0 or value >= SMALL_CELL_THRESHOLD:
        return value
    return f"<{SMALL_CELL_THRESHOLD}"


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.digest()


def _decode_external(path: Path, size: int = 128) -> np.ndarray:
    with Image.open(path) as image:
        image.draft("RGB", (size, size))
        return np.asarray(
            image.convert("RGB").resize(
                (size, size), Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )


def _hash64(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.bool_).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _perceptual_hashes(array: np.ndarray) -> tuple[int, int]:
    image = Image.fromarray(array, mode="RGB").convert("L")
    phash_input = np.asarray(
        image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    low = dctn(phash_input, type=2, norm="ortho")[:8, :8]
    phash = _hash64(low > np.median(low.reshape(-1)[1:]))
    dhash_input = np.asarray(
        image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16
    )
    dhash = _hash64(dhash_input[:, 1:] > dhash_input[:, :-1])
    return phash, dhash


def _enumerate_sources(source_roots: dict[str, Path]) -> tuple[list[Path], list[str], list[str]]:
    paths: list[Path] = []
    tags: list[str] = []
    relative_paths: list[str] = []
    if tuple(source_roots) != SOURCE_ORDER:
        raise ValueError("source roots must follow the frozen source order")
    for tag, root in source_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"external source root unavailable: {tag}")
        selected = sorted(
            Path(directory) / filename
            for directory, _, filenames in os.walk(root)
            for filename in filenames
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(selected) != EXPECTED_SOURCE_COUNTS[tag]:
            raise ValueError(f"external source count differs for {tag}")
        paths.extend(selected)
        tags.extend([tag] * len(selected))
        relative_paths.extend(
            path.relative_to(root).as_posix() for path in selected
        )
    return paths, tags, relative_paths


def _load_ai_decode(clinical_project_root: Path) -> tuple[Callable[[str, int], Image.Image], str]:
    """Load only the frozen decoder function, without importing model packages.

    ``encode_aireadi.py`` imports torch and timm for embedding inference, but its
    DICOM decoder needs neither.  Executing only the authenticated ``decode``
    AST keeps this provenance audit bound to the historical implementation and
    avoids silently installing or importing an unrelated model stack.
    """

    code_path = clinical_project_root / "encode_aireadi.py"
    source = code_path.read_text()
    tree = ast.parse(source, filename=str(code_path))
    decoder_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "decode"
    ]
    if len(decoder_nodes) != 1 or not isinstance(decoder_nodes[0], ast.FunctionDef):
        raise RuntimeError("frozen AI-READI decoder definition is missing or ambiguous")

    import pydicom
    from pydicom.encaps import generate_pixel_data_frame

    namespace: dict[str, Any] = {
        "Image": Image,
        "generate_pixel_data_frame": generate_pixel_data_frame,
        "io": io,
        "np": np,
        "pydicom": pydicom,
    }
    decoder_module = ast.Module(body=[decoder_nodes[0]], type_ignores=[])
    ast.fix_missing_locations(decoder_module)
    exec(compile(decoder_module, str(code_path), "exec"), namespace)
    decoder = namespace.get("decode")
    if not callable(decoder):
        raise RuntimeError("cannot load the frozen AI-READI decoder")
    return decoder, _sha256(code_path)


def _candidate_tables(phashes: np.ndarray) -> tuple[list[dict[int, list[int]]], tuple[tuple[int, int], ...]]:
    segments = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
    tables: list[dict[int, list[int]]] = []
    for start, stop in segments:
        mask = (1 << (stop - start)) - 1
        table: dict[int, list[int]] = defaultdict(list)
        for index, value in enumerate(phashes):
            table[(int(value) >> start) & mask].append(index)
        tables.append(dict(table))
    return tables, segments


def _candidate_indices(
    value: int,
    tables: Sequence[dict[int, list[int]]],
    segments: Sequence[tuple[int, int]],
) -> set[int]:
    candidates: set[int] = set()
    for table, (start, stop) in zip(tables, segments):
        mask = (1 << (stop - start)) - 1
        candidates.update(table.get((value >> start) & mask, ()))
    return candidates


def _load_existing_manifest(
    path: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load a completed path-free manifest and fail closed on its structure."""

    expected_names = {
        "metadata_json",
        "source_code",
        "file_size",
        "logical_relative_path_sha256",
        "raw_file_sha256",
        "decoded_128_rgb_sha256",
        "perceptual_hash64",
        "difference_hash64",
    }
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != expected_names:
            raise ValueError("existing eye manifest tensor names differ")
        try:
            metadata = json.loads(bytes(bundle["metadata_json"].astype(np.uint8)))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("existing eye manifest metadata is invalid") from exc
        source_code = np.asarray(bundle["source_code"], dtype=np.uint8).copy()
        file_size = np.asarray(bundle["file_size"], dtype=np.int64).copy()
        raw_sha256 = np.asarray(bundle["raw_file_sha256"], dtype=np.uint8).copy()
        decoded_sha256 = np.asarray(
            bundle["decoded_128_rgb_sha256"], dtype=np.uint8
        ).copy()
        phash64 = np.asarray(bundle["perceptual_hash64"], dtype=np.uint64).copy()
        dhash64 = np.asarray(bundle["difference_hash64"], dtype=np.uint64).copy()
        logical_hash = np.asarray(
            bundle["logical_relative_path_sha256"], dtype=np.uint8
        )

    count = sum(EXPECTED_SOURCE_COUNTS.values())
    if (
        metadata.get("schema_version") != MANIFEST_SCHEMA
        or metadata.get("row_count") != count
        or metadata.get("source_order") != list(SOURCE_ORDER)
        or metadata.get("source_counts") != EXPECTED_SOURCE_COUNTS
        or metadata.get("paths_emitted") is not False
        or metadata.get("patient_identifiers_emitted") is not False
        or metadata.get("images_emitted") is not False
    ):
        raise ValueError("existing eye manifest metadata contract differs")
    if (
        source_code.shape != (count,)
        or file_size.shape != (count,)
        or raw_sha256.shape != (count, 32)
        or decoded_sha256.shape != (count, 32)
        or logical_hash.shape != (count, 32)
        or phash64.shape != (count,)
        or dhash64.shape != (count,)
        or bool((file_size <= 0).any())
    ):
        raise ValueError("existing eye manifest array contract differs")
    expected_source_code = np.concatenate(
        [
            np.full(EXPECTED_SOURCE_COUNTS[tag], index, dtype=np.uint8)
            for index, tag in enumerate(SOURCE_ORDER)
        ]
    )
    if not np.array_equal(source_code, expected_source_code):
        raise ValueError("existing eye manifest source order differs")
    return metadata, source_code, file_size, raw_sha256, decoded_sha256, phash64, dhash64


def reconstruct_eye_source_provenance(
    *,
    source_roots: dict[str, Path],
    cache_path: Path,
    metadata_path: Path,
    dataset_root: Path,
    clinical_project_root: Path,
    output_manifest: Path,
    output_report: Path,
    workers: int = 8,
    reuse_existing_manifest: bool = False,
) -> dict[str, Any]:
    if output_report.exists():
        raise FileExistsError("eye provenance report output must be new")
    if output_manifest.exists() != reuse_existing_manifest:
        raise FileExistsError(
            "existing manifest must be explicitly reused, and reuse requires an existing manifest"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    count = sum(EXPECTED_SOURCE_COUNTS.values())
    cache = np.load(cache_path, mmap_mode="r")
    metadata = np.load(metadata_path, allow_pickle=True)
    if cache.shape != (count, 128, 128, 3) or metadata.shape != (2, count):
        raise ValueError("historical adaptation cache shape differs")
    expected_tags = np.concatenate(
        [
            np.full(EXPECTED_SOURCE_COUNTS[tag], tag, dtype=object)
            for tag in SOURCE_ORDER
        ]
    )
    if not np.array_equal(expected_tags, metadata[0]):
        raise ValueError("reconstructed source-tag order differs from cache metadata")
    if set(map(str, metadata[1])) != {"1"}:
        raise ValueError("historical cache contains failed decodes")

    if reuse_existing_manifest:
        (
            manifest_metadata,
            source_code,
            file_size,
            raw_sha256,
            decoded_sha256,
            phash64,
            dhash64,
        ) = _load_existing_manifest(output_manifest)
        pixel_matches = np.ones(count, dtype=np.bool_)
    else:
        paths, tags, relative_paths = _enumerate_sources(source_roots)
        source_code = np.asarray(
            [SOURCE_ORDER.index(tag) for tag in tags], dtype=np.uint8
        )
        file_size = np.zeros(count, dtype=np.int64)
        path_sha256 = np.zeros((count, 32), dtype=np.uint8)
        raw_sha256 = np.zeros((count, 32), dtype=np.uint8)
        decoded_sha256 = np.zeros((count, 32), dtype=np.uint8)
        phash64 = np.zeros(count, dtype=np.uint64)
        dhash64 = np.zeros(count, dtype=np.uint64)
        pixel_matches = np.zeros(count, dtype=np.bool_)

        def external_record(index: int) -> tuple[int, int, bytes, bytes, bytes, int, int, bool]:
            path = paths[index]
            decoded = _decode_external(path)
            phash, dhash = _perceptual_hashes(decoded)
            logical_path = f"{tags[index]}/{relative_paths[index]}".encode()
            return (
                index,
                path.stat().st_size,
                hashlib.sha256(logical_path).digest(),
                _file_digest(path),
                hashlib.sha256(decoded.tobytes()).digest(),
                phash,
                dhash,
                bool(np.array_equal(decoded, cache[index])),
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, count, 1024):
                stop = min(start + 1024, count)
                for record in executor.map(external_record, range(start, stop)):
                    index, size, path_hash, raw_hash, decoded_hash, phash, dhash, match = record
                    file_size[index] = size
                    path_sha256[index] = np.frombuffer(path_hash, dtype=np.uint8)
                    raw_sha256[index] = np.frombuffer(raw_hash, dtype=np.uint8)
                    decoded_sha256[index] = np.frombuffer(decoded_hash, dtype=np.uint8)
                    phash64[index] = phash
                    dhash64[index] = dhash
                    pixel_matches[index] = match

        if not bool(pixel_matches.all()):
            raise RuntimeError(
                "reconstructed source files do not exactly reproduce every cache row"
            )

        manifest_metadata = {
            "schema_version": MANIFEST_SCHEMA,
            "row_count": count,
            "source_order": list(SOURCE_ORDER),
            "source_counts": dict(EXPECTED_SOURCE_COUNTS),
            "row_order": "source order then lexicographically sorted absolute path within source",
            "stored_fields": [
                "source_code",
                "file_size",
                "logical_relative_path_sha256",
                "raw_file_sha256",
                "decoded_128_rgb_sha256",
                "perceptual_hash64",
                "difference_hash64",
            ],
            "paths_emitted": False,
            "patient_identifiers_emitted": False,
            "images_emitted": False,
        }
        with output_manifest.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.frombuffer(
                    json.dumps(
                        manifest_metadata, sort_keys=True, separators=(",", ":")
                    ).encode(),
                    dtype=np.uint8,
                ),
                source_code=source_code,
                file_size=file_size,
                logical_relative_path_sha256=path_sha256,
                raw_file_sha256=raw_sha256,
                decoded_128_rgb_sha256=decoded_sha256,
                perceptual_hash64=phash64,
                difference_hash64=dhash64,
            )

    external_decoded = {bytes(row) for row in decoded_sha256}
    tables, segments = _candidate_tables(phash64)
    ai_decode, ai_decoder_sha256 = _load_ai_decode(clinical_project_root)
    ai_paths = sorted(
        glob.glob(
            str(dataset_root / "retinal_photography" / "cfp" / "**" / "*.dcm"),
            recursive=True,
        )
    )
    if len(ai_paths) != 50315:
        raise ValueError("AI-READI CFP count differs from the frozen source audit")

    def ai_record(path: str) -> tuple[bytes, int, int, bool, float | None]:
        decoded = np.asarray(
            ai_decode(path, 128).resize((128, 128), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        digest = hashlib.sha256(decoded.tobytes()).digest()
        phash, dhash = _perceptual_hashes(decoded)
        candidates = _candidate_indices(phash, tables, segments)
        strong = [
            index
            for index in candidates
            if (phash ^ int(phash64[index])).bit_count() <= 4
            and (dhash ^ int(dhash64[index])).bit_count() <= 4
        ]
        minimum_mae = None
        if strong:
            minimum_mae = min(
                float(
                    np.mean(
                        np.abs(
                            decoded.astype(np.int16)
                            - np.asarray(cache[index], dtype=np.int16)
                        )
                    )
                )
                for index in strong
            )
        return digest, phash, dhash, bool(strong), minimum_mae

    exact_overlap = 0
    perceptual_candidates = 0
    perceptual_candidate_minimum_mae: float | None = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(ai_paths), 512):
            stop = min(start + 512, len(ai_paths))
            for digest, _, _, has_candidate, minimum_mae in executor.map(
                ai_record, ai_paths[start:stop]
            ):
                exact_overlap += int(digest in external_decoded)
                perceptual_candidates += int(has_candidate)
                if minimum_mae is not None:
                    perceptual_candidate_minimum_mae = (
                        minimum_mae
                        if perceptual_candidate_minimum_mae is None
                        else min(perceptual_candidate_minimum_mae, minimum_mae)
                    )

    raw_unique = len({bytes(row) for row in raw_sha256})
    decoded_unique = len(external_decoded)
    source_counts = dict(EXPECTED_SOURCE_COUNTS)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "external_eye_adaptation_cache_reconstruction_and_ai_readi_overlap_audit",
        "reconstruction": {
            "source_counts": source_counts,
            "total_files": count,
            "source_tag_order_exact": True,
            "successful_decode_flags": count,
            "cache_rows_pixel_exact": int(pixel_matches.sum()),
            "all_cache_rows_pixel_exact": bool(pixel_matches.all()),
            "raw_file_hash_unique_count": raw_unique,
            "decoded_pixel_hash_unique_count": decoded_unique,
            "per_file_identity_manifest_available": True,
            "completed_external_manifest_reused": reuse_existing_manifest,
            "paths_emitted": False,
        },
        "ai_readi_overlap_audit": {
            "ai_readi_cfp_images": len(ai_paths),
            "exact_decoded_pixel_overlap_count": _safe_count(exact_overlap),
            "any_exact_decoded_pixel_overlap": exact_overlap > 0,
            "dual_hash_hamming_le_4_candidate_count": _safe_count(
                perceptual_candidates
            ),
            "any_dual_hash_hamming_le_4_candidate": perceptual_candidates > 0,
            "minimum_candidate_mean_absolute_pixel_difference": (
                perceptual_candidate_minimum_mae
                if perceptual_candidates >= SMALL_CELL_THRESHOLD
                else None
            ),
            "perceptual_candidate_count_small_cell_suppressed": bool(
                0 < perceptual_candidates < SMALL_CELL_THRESHOLD
            ),
            "patient_identifiers_compared": False,
            "subject_identity_overlap_proven": False,
            "interpretation": "exact and dual-hash image overlap audit; not a cross-study subject-identity linkage",
        },
        "gates": {
            "per_file_manifest_reconstruction_passed": bool(pixel_matches.all()),
            "exact_image_overlap_gate_passed": exact_overlap == 0,
            "perceptual_overlap_screen_passed": perceptual_candidates == 0,
            "strongest_subject_identity_claim_authorized": False,
        },
        "bindings": {
            "cache_file_sha256": _sha256(cache_path),
            "metadata_file_sha256": _sha256(metadata_path),
            "manifest_file": output_manifest.name,
            "manifest_file_sha256": _sha256(output_manifest),
            "ai_readi_decoder_sha256": ai_decoder_sha256,
            "audit_code_sha256": _sha256(Path(__file__).resolve()),
        },
        "privacy": {
            "patient_rows_emitted": False,
            "patient_identifiers_emitted": False,
            "source_paths_emitted": False,
            "images_emitted": False,
            "ai_readi_hashes_emitted": False,
            "small_cell_threshold": SMALL_CELL_THRESHOLD,
        },
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drunified-root", type=Path, required=True)
    parser.add_argument("--brset-root", type=Path, required=True)
    parser.add_argument("--odir-root", type=Path, required=True)
    parser.add_argument("--jsiec-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clinical-project-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--reuse-existing-manifest",
        action="store_true",
        help="resume only the overlap phase from a structurally validated completed manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = reconstruct_eye_source_provenance(
        source_roots={
            "drunified": args.drunified_root.resolve(),
            "brset": args.brset_root.resolve(),
            "odir": args.odir_root.resolve(),
            "jsiec": args.jsiec_root.resolve(),
        },
        cache_path=args.cache.resolve(),
        metadata_path=args.metadata.resolve(),
        dataset_root=args.dataset_root.resolve(),
        clinical_project_root=args.clinical_project_root.resolve(),
        output_manifest=args.manifest_output.resolve(),
        output_report=args.report_output.resolve(),
        workers=args.workers,
        reuse_existing_manifest=args.reuse_existing_manifest,
    )
    print(
        json.dumps(
            {
                "event": "patient_atlas_eye_source_provenance_completed",
                "external_file_count": report["reconstruction"]["total_files"],
                "all_cache_rows_pixel_exact": report["reconstruction"][
                    "all_cache_rows_pixel_exact"
                ],
                "any_exact_overlap": report["ai_readi_overlap_audit"][
                    "any_exact_decoded_pixel_overlap"
                ],
                "any_perceptual_candidate": report["ai_readi_overlap_audit"][
                    "any_dual_hash_hamming_le_4_candidate"
                ],
                "report_output": str(args.report_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_SCHEMA",
    "SCHEMA_VERSION",
    "reconstruct_eye_source_provenance",
]

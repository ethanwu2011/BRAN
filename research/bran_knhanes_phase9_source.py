"""Local authenticated SAS-to-private-table boundary for phase-9 KNHANES.

No authority is created here.  A caller must supply a qualifying source-use
receipt and attest that it is inside its FD-quiet exclusive-lock context before
this module opens a SAS member.  The module has no output, cache, logging, or
aggregate release path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from collections.abc import Mapping as AbstractMapping
import zipfile

import numpy as np
import pandas as pd
from pandas.io.sas.sas7bdat import SAS7BDATReader

from audit_knhanes_metadata_v1 import json_sha, read_metadata


_ERROR = "knhanes_phase9_source_contract_failed"
_YEARS = (2022, 2023)
_RECEIPT_KEYS = {
    "schema", "binding_sha256", "permission_confirmed", "no_prior_encoder_training",
    "no_prior_model_selection", "source_provenance", "scope",
    "published_measurements_original_observed", "published_he_hb_original_observed_assay",
}
_CORE_NUMERIC = ("year", "age", "sex", "HE_prg", "HE_HB", "wt_itvex", "kstrata")
_CORE_TEXT = ("ID", "ID_fam", "psu")
_COMPANIONS = {"HE_alt": "HE_alt_etc", "HE_hsCRP": "HE_hsCRP_etc"}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _path(value: Any, *, file: bool) -> Path:
    _require(isinstance(value, (str, Path)))
    path = Path(value)
    _require((path.is_file() if file else path.is_dir()) and not path.is_symlink())
    return path


def _bound_child(root: Path, relative: Any, expected: Any) -> Path:
    _require(isinstance(relative, str) and isinstance(expected, str) and len(expected) == 64)
    candidate = Path(relative)
    _require(not candidate.is_absolute() and ".." not in candidate.parts)
    path = root / candidate
    _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
             and _sha256(path) == expected)
    return path


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy(); result.setflags(write=False); return result


@dataclass(frozen=True, repr=False)
class AuthenticatedPhase9Binding:
    """Byte/schema-authenticated, but not source-admitted, loader inputs."""

    binding: Mapping[str, Any]
    metadata: Mapping[str, Any]
    source_root: Path
    source_paths: Mapping[int, Path]
    receipt: Mapping[str, Any]

    def __repr__(self) -> str:
        return "<AuthenticatedPhase9Binding>"

    def __reduce__(self):
        raise TypeError(_ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(_ERROR)


@dataclass(frozen=True, repr=False)
class PrivatePhase9SourceTables(AbstractMapping):
    """Private adapter-shaped source tables; never serialize or print rows."""

    tables: Mapping[int, Mapping[str, Any]]

    def __repr__(self) -> str:
        return "<PrivatePhase9SourceTables>"

    def __reduce__(self):
        raise TypeError(_ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(_ERROR)

    def __getitem__(self, key: int) -> Mapping[str, Any]:
        return self.tables[key]

    def __iter__(self):
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)


def _source_use_receipt(value: Any, binding: Mapping[str, Any]) -> None:
    _require(isinstance(value, Mapping) and set(value) == _RECEIPT_KEYS
             and value["schema"] == "bran-knhanes-phase9-source-use-v1"
             and value["binding_sha256"] == _canonical_sha256(binding)
             and value["permission_confirmed"] is True
             and value["no_prior_encoder_training"] is True
             and value["no_prior_model_selection"] is True
             and value["published_measurements_original_observed"] is True
             and value["published_he_hb_original_observed_assay"] is True
             and value["source_provenance"] == "KNHANES phase-9 released ALL-table measurements"
             and value["scope"] == "bran-v5-phase9-external-hb-fixed-readout")


def _source_rows(binding: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    _require(isinstance(binding, Mapping)
             and binding.get("schema") == "bran-knhanes-phase9-binding-v1"
             and binding.get("status") == "bytes_and_schemas_bound"
             and binding.get("patient_rows_decoded") is False
             and binding.get("model_scored") is False
             and binding.get("source_qualified_for_evaluation") is False
             and isinstance(binding.get("sources"), list) and len(binding["sources"]) == 2)
    rows = {item.get("year"): item for item in binding["sources"] if isinstance(item, Mapping)}
    _require(set(rows) == set(_YEARS))
    _require(isinstance(metadata, Mapping) and metadata.get("schema") == "bran-knhanes-metadata-v1"
             and metadata.get("status") == "completed_metadata_only"
             and metadata.get("patient_rows_decoded") is False and isinstance(metadata.get("tables"), list))
    for year, row in rows.items():
        _require(isinstance(row.get("mapping_candidates"), list) and len(row["mapping_candidates"]) == 17
                 and row.get("member") == f"hn{year % 100}_all.sas7bdat")
        meta = [item for item in metadata["tables"] if isinstance(item, Mapping)
                and item.get("year") == year and item.get("module") == "all"]
        _require(len(meta) == 1)
        meta = meta[0]
        _require(all(row.get(key) == meta.get(key) for key in ("source_file", "source_sha256", "member", "schema_sha256"))
                 and json_sha(meta.get("columns")) == row["schema_sha256"])
    return rows


def authenticate_sources(admission_path: str | Path) -> AuthenticatedPhase9Binding:
    """Authenticate row-free bytes and the caller's receipt; never decode SAS rows."""
    try:
        admission_file = _path(admission_path, file=True)
        with admission_file.open("rb") as handle:
            admission = json.loads(handle.read().decode("utf-8"))
        _require(isinstance(admission, Mapping) and set(admission) == {
            "schema", "binding_path", "metadata_path", "guide_path", "unit_path", "source_root", "source_use_receipt",
        } and admission["schema"] == "bran-knhanes-phase9-source-admission-v1")
        binding_file = _path(admission["binding_path"], file=True)
        with binding_file.open("rb") as handle:
            binding = json.loads(handle.read().decode("utf-8"))
        _require(isinstance(binding, Mapping))
        _source_use_receipt(admission["source_use_receipt"], binding)
        metadata_path = _path(admission["metadata_path"], file=True)
        guide_path = _path(admission["guide_path"], file=True)
        unit_path = _path(admission["unit_path"], file=True)
        source_root = _path(admission["source_root"], file=False)
        _require(_sha256(metadata_path) == binding.get("metadata_sha256")
                 and _sha256(guide_path) == binding.get("guide_sha256")
                 and _sha256(unit_path) == binding.get("model_units_sha256"))
        with metadata_path.open("rb") as handle:
            metadata = json.loads(handle.read().decode("utf-8"))
        rows = _source_rows(binding, metadata)
        paths = {year: _bound_child(source_root, rows[year]["source_file"], rows[year]["source_sha256"])
                 for year in _YEARS}
        receipt = MappingProxyType({
            "schema": "bran-knhanes-phase9-source-authentication-v1",
            "admission_sha256": _sha256(admission_file), "binding_sha256": _canonical_sha256(binding),
            "binding_file_sha256": _sha256(binding_file), "metadata_sha256": _sha256(metadata_path),
            "guide_sha256": _sha256(guide_path), "model_units_sha256": _sha256(unit_path),
            "source_use_receipt_sha256": _canonical_sha256(admission["source_use_receipt"]),
            "permission_confirmed": True, "no_prior_encoder_training": True, "no_prior_model_selection": True,
            "source_archives": tuple((year, rows[year]["source_sha256"]) for year in _YEARS),
        })
        return AuthenticatedPhase9Binding(MappingProxyType(dict(binding)), MappingProxyType(dict(metadata)),
                                          source_root, MappingProxyType(paths), receipt)
    except (TypeError, ValueError, KeyError, IndexError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(_ERROR) from None


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        for encoding in ("utf-8", "cp949"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                pass
        _require(False)
    _require(isinstance(value, str))
    return value


def _header_lookup(columns: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for column in columns:
        _require(isinstance(column, Mapping) and set(column) == {"name", "label", "type"}
                 and isinstance(column["name"], str) and column["type"] in {"numeric", "text"})
        key = column["name"].casefold(); _require(key not in output); output[key] = column
    return output


def _needed(source: Mapping[str, Any], headers: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    candidates = source["mapping_candidates"]
    names = [row.get("source_column") for row in candidates]
    _require(all(isinstance(name, str) for name in names) and len({name.casefold() for name in names}) == 17)
    companions: list[str] = []
    for source_name, companion in _COMPANIONS.items():
        rows = [row for row in candidates if row.get("source_column", "").casefold() == source_name.casefold()]
        _require(len(rows) == 1 and isinstance(rows[0].get("companion_column"), str))
        companions.append(rows[0]["companion_column"])
    required = list(_CORE_NUMERIC) + list(_CORE_TEXT) + names + companions
    _require(len({name.casefold() for name in required}) == len(required))
    for name in _CORE_NUMERIC:
        _require(name.casefold() in headers and headers[name.casefold()]["type"] == "numeric")
    for name in _CORE_TEXT + tuple(companions):
        _require(name.casefold() in headers and headers[name.casefold()]["type"] == "text")
    for name in names:
        _require(name.casefold() in headers and headers[name.casefold()]["type"] == "numeric")
    return required, names


def _frame_columns(frame: Any, required: list[str], headers: Mapping[str, Mapping[str, Any]]) -> Mapping[str, np.ndarray]:
    _require(isinstance(frame, pd.DataFrame) and frame.ndim == 2)
    actual: dict[str, Any] = {}
    for column in frame.columns:
        name = _decode(column); key = name.casefold(); _require(key not in actual); actual[key] = column
    _require(set(actual) == set(headers))
    output: dict[str, np.ndarray] = {}
    for name in required:
        column = actual[name.casefold()]
        values = frame[column].to_numpy(copy=True)
        if headers[name.casefold()]["type"] == "numeric":
            _require(values.ndim == 1 and values.dtype.kind in "fiu")
            output[name] = _readonly(values.astype(np.float64, copy=False))
        else:
            _require(values.ndim == 1)
            output[name] = _readonly(np.asarray([_decode(item) for item in values], dtype="U"))
    rows = next(iter(output.values())).shape[0]
    _require(rows > 0 and all(value.shape == (rows,) for value in output.values()))
    return MappingProxyType(output)


def _read_frame(archive_path: Path, source: Mapping[str, Any], expected_columns: list[Mapping[str, Any]],
                reader_factory: Callable[..., Any]) -> Any:
    _require(_sha256(archive_path) == source["source_sha256"])
    with zipfile.ZipFile(archive_path) as archive:
        matches = [info for info in archive.infolist() if info.filename == source["member"]]
        _require(len(matches) == 1)
        info = matches[0]
        _require(not info.is_dir() and not info.flag_bits & 1 and 0 < info.file_size <= 2 * 1024 ** 3
                 and info.compress_size > 0 and info.file_size / info.compress_size < 2500)
        # Metadata is read before any row payload.  A header mismatch therefore
        # fails before ``reader.read`` can decode participants.
        with archive.open(info) as stream:
            observed = read_metadata(stream, reader_factory)
        _require(observed == expected_columns and json_sha(observed) == source["schema_sha256"])
        with archive.open(info) as stream:
            # pandas calls this option ``blank_missing``.  Keeping it false is
            # essential: an empty SAS report-limit companion is meaningful
            # empty text, not a missing object/NaN sentinel.
            reader = reader_factory(stream, convert_header_text=False, blank_missing=False)
            try:
                _require(getattr(reader, "_current_row_in_file_index") == 0)
                frame = reader.read()
            finally:
                reader.close()
    return frame


def load_tables(admitted_binding: AuthenticatedPhase9Binding, *, reader_factory: Callable[..., Any] = SAS7BDATReader,
                fd_quiet: bool = False, exclusive_lock_held: bool = False) -> PrivatePhase9SourceTables:
    """Decode only the bound members after ``authenticate_sources`` succeeds."""
    try:
        _require(isinstance(admitted_binding, AuthenticatedPhase9Binding) and callable(reader_factory)
                 and fd_quiet is True and exclusive_lock_held is True)
        sources = _source_rows(admitted_binding.binding, admitted_binding.metadata)
        output: dict[int, Mapping[str, Any]] = {}
        for year in _YEARS:
            source = sources[year]
            meta = [item for item in admitted_binding.metadata["tables"]
                    if item["year"] == year and item["module"] == "all"][0]
            expected = meta["columns"]
            headers = _header_lookup(expected)
            required, measurement_names = _needed(source, headers)
            frame = _read_frame(admitted_binding.source_paths[year], source, expected, reader_factory)
            columns = _frame_columns(frame, required, headers)
            rows = next(iter(columns.values())).shape[0]
            exact = {name: _readonly(np.isfinite(columns[name])) for name in measurement_names}
            imputed = {name: _readonly(np.zeros(rows, dtype=bool)) for name in measurement_names}
            output[year] = MappingProxyType({
                "columns": columns, "exact_observed": MappingProxyType(exact), "imputed": MappingProxyType(imputed),
                # This is a source-contract attestation, deliberately not an
                # inference from HE_HB finiteness.  The target kernel still
                # applies its finite/range/domain requirements.
                "target_original_observed": _readonly(np.ones(rows, dtype=bool)),
                "target_observation_contract": True,
            })
        return PrivatePhase9SourceTables(MappingProxyType(output))
    except Exception:
        raise ValueError(_ERROR) from None

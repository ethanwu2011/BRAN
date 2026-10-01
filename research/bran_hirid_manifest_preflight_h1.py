"""Metadata-only HiRID manifest qualification before a local reader is called.

This module neither opens a source path nor authenticates study approval itself.
The caller supplies a complete, already-authenticated inventory and a trusted
approval receipt.  Admission membership remains private to the local caller.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType

from bran_hirid_raw_reader_v1 import COLUMNS, SCHEMA_PDF_SHA256, _id


ERROR = "hirid_manifest_preflight_h1_contract_failed"
SOURCE = "HiRID"
VERSION = "1.1.1"
_HASH = re.compile(r"[0-9a-f]{64}")
_NO_REPEAT_PERSON_LINKAGE = "absent"


class _Invalid(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _hash(value: object) -> str:
    _require(type(value) is str and _HASH.fullmatch(value) is not None)
    return value


def _inventory_hash(partition_hashes: Iterable[str]) -> str:
    # This is an aggregate receipt only; it does not contain a source path,
    # filename, admission identifier, or partition count.
    return hashlib.sha256("\n".join(sorted(partition_hashes)).encode("ascii")).hexdigest()


class PrivatePartitionMembership(Mapping):
    """Read-only partition-to-admission handles for a trusted local callback."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, tuple[int, ...]]):
        self._values = MappingProxyType(dict(values))

    def __setattr__(self, name, value):
        if name == "_values" and hasattr(self, "_values"):
            raise TypeError(ERROR)
        object.__setattr__(self, name, value)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self) -> str:
        return "<PrivateHiRIDPartitionMembership>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


@dataclass(frozen=True, slots=True, repr=False)
class PrivateHiRIDManifest:
    """Qualified metadata and private membership; no source locations or rows."""

    source_sha256: str
    reference_sha256: str
    schema_sha256: str
    raw_partition_inventory_sha256: str
    membership: PrivatePartitionMembership
    repeat_person_linkage: str

    def __repr__(self) -> str:
        return "<PrivateHiRIDManifest>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


@dataclass(frozen=True, slots=True, repr=False)
class AuthenticatedStudyApproval:
    """A trusted caller's explicit approval receipt, not an approval generator."""

    source: str
    version: str
    status: str
    receipt_sha256: str

    def __repr__(self) -> str:
        return "<AuthenticatedHiRIDStudyApproval>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


@dataclass(frozen=True, slots=True, repr=False)
class PreflightReport:
    """Safe-to-log outcome: status plus receipts, never membership or counts."""

    status: str
    source_sha256: str
    reference_sha256: str
    schema_sha256: str
    raw_partition_inventory_sha256: str
    approval_receipt_sha256: str

    def __repr__(self) -> str:
        return "<HiRIDManifestPreflightReport>"

    def __reduce__(self):
        raise TypeError(ERROR)

    def __reduce_ex__(self, protocol):
        raise TypeError(ERROR)


def qualify_manifest(*, source: object, version: object, source_sha256: object,
                     reference_sha256: object, schema_sha256: object,
                     raw_header: object, expected_partition_hashes: object,
                     raw_partition_inventory_sha256: object,
                     partition_entries: object,
                     repeat_person_linkage: object) -> PrivateHiRIDManifest:
    """Validate supplied metadata without opening a source or reading any rows.

    ``partition_entries`` is an iterable of ``(raw_partition_sha256,
    admission_ids)`` pairs.  Its hashes must be exactly the complete,
    authenticated expected inventory.  IDs are normalized with the reader's
    existing admission-ID helper and are kept in a repr-hidden container.
    """
    try:
        _require(source == SOURCE and version == VERSION)
        source_digest = _hash(source_sha256)
        reference_digest = _hash(reference_sha256)
        schema_digest = _hash(schema_sha256)
        _require(schema_digest == SCHEMA_PDF_SHA256)

        _require(not isinstance(raw_header, (str, bytes)))
        header = tuple(raw_header)
        _require(len(header) == len(set(header)) and set(header) == set(COLUMNS))

        _require(not isinstance(expected_partition_hashes, (str, bytes)))
        expected = tuple(_hash(value) for value in expected_partition_hashes)
        _require(bool(expected) and len(expected) == len(set(expected)))
        inventory_digest = _hash(raw_partition_inventory_sha256)
        _require(inventory_digest == _inventory_hash(expected))

        _require(not isinstance(partition_entries, (str, bytes, Mapping)))
        entries = tuple(partition_entries)
        _require(bool(entries))
        memberships: dict[str, tuple[int, ...]] = {}
        admitted_anywhere: set[int] = set()
        for entry in entries:
            _require(type(entry) is tuple and len(entry) == 2)
            partition_digest = _hash(entry[0])
            _require(partition_digest not in memberships)
            members_input = entry[1]
            _require(not isinstance(members_input, (str, bytes, Mapping)))
            members = tuple(_id(value) for value in members_input)
            _require(bool(members) and len(members) == len(set(members)))
            _require(not admitted_anywhere.intersection(members))
            admitted_anywhere.update(members)
            memberships[partition_digest] = members
        _require(set(memberships) == set(expected))
        _require(repeat_person_linkage == _NO_REPEAT_PERSON_LINKAGE)
        return PrivateHiRIDManifest(
            source_digest,
            reference_digest,
            schema_digest,
            inventory_digest,
            PrivatePartitionMembership(memberships),
            _NO_REPEAT_PERSON_LINKAGE,
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


def authorize_reader(manifest: object, approval: object, source_reader_callback: object) -> PreflightReport:
    """Invoke a caller-injected local reader only after explicit approval.

    Approval authenticity is established outside this pure module.  The exact
    receipt, source, version, and approved status are required here; omitted,
    false, or malformed approval fails before the callback can run.  This is
    not an FD-level privacy wrapper: a real trusted callback must run inside
    the project's separate FD-quiet boundary.
    """
    try:
        _require(type(manifest) is PrivateHiRIDManifest)
        _require(type(approval) is AuthenticatedStudyApproval)
        _require(approval.source == SOURCE and approval.version == VERSION)
        _require(approval.status == "approved")
        receipt_digest = _hash(approval.receipt_sha256)
        _require(callable(source_reader_callback))
        source_reader_callback(manifest)
        return PreflightReport(
            "reader_authorized",
            manifest.source_sha256,
            manifest.reference_sha256,
            manifest.schema_sha256,
            manifest.raw_partition_inventory_sha256,
            receipt_digest,
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError(ERROR) from None


__all__ = [
    "ERROR", "SOURCE", "VERSION", "AuthenticatedStudyApproval", "PreflightReport",
    "PrivateHiRIDManifest", "PrivatePartitionMembership", "authorize_reader", "qualify_manifest",
]

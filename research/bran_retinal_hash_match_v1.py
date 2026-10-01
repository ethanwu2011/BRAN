"""Private, in-memory hash matching for a content-admission audit.

Matches are private row indexes only.  They do not establish a person identity,
clinical identity, or duplicate anatomy, and must not be published individually.
Only separately validated, disclosure-safe aggregate summaries may be released.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


_SEGMENTS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
_RADIUS = 4
_HASH_LIMIT = 1 << 64


def _invalid() -> ValueError:
    return ValueError("invalid private hash input")


def _require_array(value: object, *, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype:
        raise _invalid()
    if shape is not None and value.shape != shape:
        raise _invalid()
    return value


def _require_hash64(value: object) -> int:
    if type(value) is not int or value < 0 or value >= _HASH_LIMIT:
        raise _invalid()
    return value


def _hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class PrivateHashIndex:
    """Private exact/near hash index; it makes no identity or anatomy claim."""

    def __init__(
        self,
        decoded_sha256: np.ndarray,
        perceptual_hash64: np.ndarray,
        difference_hash64: np.ndarray,
    ) -> None:
        decoded = _require_array(decoded_sha256, dtype=np.dtype(np.uint8))
        if decoded.ndim != 2 or decoded.shape[0] == 0 or decoded.shape[1] != 32:
            raise _invalid()
        n_rows = decoded.shape[0]
        perceptual = _require_array(perceptual_hash64, dtype=np.dtype(np.uint64), shape=(n_rows,))
        difference = _require_array(difference_hash64, dtype=np.dtype(np.uint64), shape=(n_rows,))

        self._decoded = np.array(decoded, dtype=np.uint8, copy=True)
        self._perceptual = np.array(perceptual, dtype=np.uint64, copy=True)
        self._difference = np.array(difference, dtype=np.uint64, copy=True)
        self._decoded.setflags(write=False)
        self._perceptual.setflags(write=False)
        self._difference.setflags(write=False)

        digest_rows: dict[bytes, list[int]] = defaultdict(list)
        segment_tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in _SEGMENTS]
        for index, (digest, perceptual_value) in enumerate(zip(self._decoded, self._perceptual)):
            digest_rows[bytes(digest)].append(index)
            value = int(perceptual_value)
            for table, (start, stop) in zip(segment_tables, _SEGMENTS):
                table[(value >> start) & ((1 << (stop - start)) - 1)].append(index)
        self._digest_rows = {digest: tuple(indexes) for digest, indexes in digest_rows.items()}
        self._segment_tables = tuple(
            {key: tuple(indexes) for key, indexes in table.items()} for table in segment_tables
        )

    def __repr__(self) -> str:
        return "<PrivateHashIndex private>"

    def match(
        self,
        decoded_sha256: bytes,
        perceptual_hash64: int,
        difference_hash64: int,
    ) -> frozenset[int]:
        """Return private matching row indexes; never treat them as identity proof."""
        if type(decoded_sha256) is not bytes or len(decoded_sha256) != 32:
            raise _invalid()
        perceptual = _require_hash64(perceptual_hash64)
        difference = _require_hash64(difference_hash64)

        matches = set(self._digest_rows.get(decoded_sha256, ()))
        candidates: set[int] = set()
        for table, (start, stop) in zip(self._segment_tables, _SEGMENTS):
            segment = (perceptual >> start) & ((1 << (stop - start)) - 1)
            candidates.update(table.get(segment, ()))
        for index in candidates:
            if (
                _hamming64(perceptual, int(self._perceptual[index])) <= _RADIUS
                and _hamming64(difference, int(self._difference[index])) <= _RADIUS
            ):
                matches.add(index)
        return frozenset(matches)

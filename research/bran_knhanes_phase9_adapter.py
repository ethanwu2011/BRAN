"""Pure authenticated-table adapter for the prospective KNHANES phase-9 Hb study.

The caller owns SAS decoding, permission, source-use and byte/schema
authentication.  This module accepts only already-authenticated in-memory
tables; it performs no I/O and does not mark a source admitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from bran_knhanes_grouped_folds_v1 import GroupedKNHANESFolds, assign
from bran_knhanes_input_kernel_v1 import ColumnRule, PreparedInputs, prepare_inputs
from bran_knhanes_target_v1 import TargetMasks, prepare_target
from build_bran_knhanes_phase9_binding_v1 import MAPPINGS


_ERROR = "knhanes_phase9_adapter_contract_failed"
_YEARS = (2022, 2023)
_NUMERIC = ("year", "age", "sex", "HE_prg", "HE_HB", "wt_itvex", "kstrata")
_TEXT = ("ID", "ID_fam", "psu")
_COMPANIONS = {"HE_alt": "HE_alt_etc", "HE_hsCRP": "HE_hsCRP_etc"}


def _require(ok: bool) -> None:
    if not ok: raise ValueError(_ERROR)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy(); result.setflags(write=False); return result


@dataclass(frozen=True, repr=False)
class PrivateKNHANESPhase9Prepared:
    """Private material for a later local fixed-readout/evaluator wrapper."""
    inputs: PreparedInputs
    target_masks: TargetMasks
    hemoglobin: np.ndarray
    design_weights: np.ndarray
    years: np.ndarray
    grouped_folds: GroupedKNHANESFolds
    source_admitted: bool

    def __repr__(self): return "<PrivateKNHANESPhase9Prepared>"
    def __reduce__(self): raise TypeError(_ERROR)
    def __reduce_ex__(self, protocol): raise TypeError(_ERROR)

    @property
    def folds(self) -> np.ndarray:
        return self.grouped_folds.folds

    @property
    def psu_groups(self) -> np.ndarray:
        return self.grouped_folds.psu_groups

    @property
    def stratum_groups(self) -> np.ndarray:
        return self.grouped_folds.stratum_groups


def _headers(table: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, str], int]:
    _require(isinstance(table, Mapping) and set(table) == {"columns", "exact_observed", "imputed", "target_original_observed", "target_observation_contract"})
    columns = table["columns"]
    _require(isinstance(columns, Mapping) and bool(columns))
    lookup: dict[str, str] = {}
    rows = None
    for name, values in columns.items():
        _require(isinstance(name, str) and bool(name.strip()))
        folded = name.casefold(); _require(folded not in lookup); lookup[folded] = name
        _require(type(values) is np.ndarray and values.ndim == 1)
        rows = values.shape[0] if rows is None else rows
        _require(values.shape == (rows,))
    _require(rows is not None and rows > 0)
    for name in _NUMERIC:
        key = lookup.get(name.casefold()); _require(key is not None and columns[key].dtype.kind in "fiu")
    for name in _TEXT:
        key = lookup.get(name.casefold()); _require(key is not None and columns[key].dtype.kind in "SU")
    _require(type(table["target_original_observed"]) is np.ndarray and table["target_original_observed"].shape == (rows,)
             and table["target_original_observed"].dtype == np.dtype(bool) and table["target_observation_contract"] is True)
    for key in ("exact_observed", "imputed"):
        value = table[key]; _require(isinstance(value, Mapping))
        for header, array in value.items():
            _require(header in columns and type(array) is np.ndarray and array.shape == (rows,) and array.dtype == np.dtype(bool))
    return columns, lookup, rows


def _text(values: np.ndarray) -> np.ndarray:
    output = []
    for value in values:
        if isinstance(value, bytes):
            try: value = value.decode("utf-8")
            except UnicodeDecodeError: _require(False)
        _require(isinstance(value, str))
        item = value.strip(); _require(bool(item)); output.append(item)
    return np.asarray(output, dtype="U")


def _rules(binding: Mapping[str, Any], lookup: Mapping[str, str]) -> tuple[list[dict[str, Any]], tuple[ColumnRule, ...]]:
    _require(isinstance(binding, Mapping) and binding.get("schema") == "bran-knhanes-phase9-binding-v1"
             and binding.get("source_qualified_for_evaluation") is False and isinstance(binding.get("sources"), list))
    _require(len(binding["sources"]) == len(_YEARS))
    for pin_name in ("guide_sha256", "model_units_sha256"):
        pin = binding.get(pin_name)
        _require(isinstance(pin, str) and len(pin) == 64 and all(char in "0123456789abcdef" for char in pin))
    sources = {item.get("year"): item for item in binding["sources"] if isinstance(item, Mapping)}
    _require(set(sources) == set(_YEARS))
    candidates = []
    for year in _YEARS:
        rows = sources[year].get("mapping_candidates"); _require(isinstance(rows, list) and len(rows) == 17)
        _require(all(isinstance(row, Mapping) for row in rows))
        candidates.extend(rows)
    # Both supplied releases must carry the identical authenticated 17-field map.
    first = candidates[:17]
    for row in candidates[17:]:
        _require(row in first)
    _require(len({row.get("canonical_name") for row in first}) == 17
             and len({row.get("canonical_name") for row in candidates[17:]}) == 17)
    expected = {key.casefold():(name,unit,page) for key,(name,unit,page) in MAPPINGS.items()}
    _require({str(row.get('source_column')).casefold() for row in first} == set(expected))
    rules = []
    for row in first:
        source_name = row.get("source_column"); canonical = row.get("canonical_name"); index = row.get("canonical_index")
        _require(isinstance(source_name, str) and source_name.casefold() in lookup and isinstance(canonical, str)
                 and type(index) is int and row.get("conversion_factor") == 1.0
                 and row.get("source_unit") == row.get("target_unit") and row.get("companion_column") in (None, _COMPANIONS.get(source_name)))
        name,unit,_page=expected[source_name.casefold()]
        expected_companion={k.casefold():v for k,v in _COMPANIONS.items()}.get(source_name.casefold())
        _require(canonical==name and row['source_unit']==unit
                 and row.get('companion_column')==expected_companion)
        rules.append(ColumnRule(lookup[source_name.casefold()], canonical, index, row["source_unit"], row["target_unit"], 1., (),
                                binding["guide_sha256"], binding["model_units_sha256"]))
    return first, tuple(rules)


def prepare(tables: Mapping[int, Mapping[str, Any]], binding: Mapping[str, Any]) -> PrivateKNHANESPhase9Prepared:
    """Prepare whole-CBC-hidden inputs plus target/design/grouped-fold material.

    Values for unobserved input cells are never interpreted: their caller-supplied
    exact-observed and imputed flags are mandatory.  Positive-weight target-
    ineligible people stay in the returned survey design/grouping population.
    """
    try:
        _require(isinstance(tables, Mapping) and set(tables) == set(_YEARS))
        prepared = []
        for year in _YEARS:
            table = tables[year]; columns, lookup, rows = _headers(table); candidate_rows, rules = _rules(binding, lookup)
            numeric = {name: columns[lookup[name.casefold()]].astype(np.float64, copy=False) for name in _NUMERIC}
            keep = np.isfinite(numeric["wt_itvex"]) & (numeric["wt_itvex"] > 0.0)
            _require(bool(keep.any()))
            _require(bool(np.isfinite(numeric["year"][keep]).all()) and bool((numeric["year"][keep] == year).all())
                     and bool(np.isfinite(numeric["kstrata"][keep]).all())
                     and bool((numeric["kstrata"][keep] == np.floor(numeric["kstrata"][keep])).all()))
            identifiers = {name: _text(columns[lookup[name.casefold()]][keep]) for name in _TEXT}
            source_names = [rule.source_column for rule in rules]
            _require(all(columns[name].dtype.kind in 'fiu' for name in source_names))
            raw = np.column_stack([columns[name][keep].astype(np.float64, copy=False) for name in source_names])
            # Missing observation metadata is not evidence of a missing value.
            _require(all(name in table["exact_observed"] and name in table["imputed"] for name in source_names))
            exact = np.column_stack([table["exact_observed"][name][keep] for name in source_names])
            imputed = np.column_stack([table["imputed"][name][keep] for name in source_names])
            for index, row in enumerate(candidate_rows):
                companion = row.get("companion_column")
                if companion is not None:
                    key = lookup.get(companion.casefold()); _require(key is not None and columns[key].dtype.kind in "SU")
                    nonempty = np.asarray([bool((item.decode("utf-8") if isinstance(item, bytes) else item).strip()) for item in columns[key][keep]], bool)
                    exact[nonempty, index] = False
            inputs = prepare_inputs(raw, tuple(source_names), exact, imputed, numeric["age"][keep], rules, mask_pattern="whole_cbc")
            targets = prepare_target(numeric["HE_HB"][keep], numeric["age"][keep], numeric["sex"][keep], numeric["HE_prg"][keep],
                                     table["target_original_observed"][keep])
            prepared.append((inputs, targets, numeric, identifiers, keep))
        inputs = _combine_inputs([item[0] for item in prepared])
        target_masks = _combine_targets([item[1] for item in prepared])
        hb = np.concatenate([item[2]["HE_HB"][item[4]] for item in prepared])
        weight = np.concatenate([item[2]["wt_itvex"][item[4]] / 2.0 for item in prepared])
        years = np.concatenate([item[2]["year"][item[4]].astype(np.int64) for item in prepared])
        psu = np.concatenate([item[3]["psu"] for item in prepared]); strata = np.concatenate([item[2]["kstrata"][item[4]].astype(np.int64) for item in prepared])
        household = np.concatenate([item[3]["ID_fam"] for item in prepared]); person = np.concatenate([item[3]["ID"] for item in prepared])
        groups = assign(years, person, psu, strata, household)
        return PrivateKNHANESPhase9Prepared(inputs, target_masks, _readonly(hb), _readonly(weight), _readonly(years), groups, False)
    except (TypeError, ValueError, KeyError, IndexError, UnicodeDecodeError, AttributeError):
        raise ValueError(_ERROR) from None


def _combine_inputs(items: list[PreparedInputs]) -> PreparedInputs:
    return PreparedInputs(*(_readonly(np.concatenate([getattr(item, name) for item in items], axis=0))
                            for name in ("clinical_values", "clinical_mask", "retinal_features", "retinal_mask", "ages", "eligible")))


def _combine_targets(items: list[TargetMasks]) -> TargetMasks:
    return TargetMasks(*(_readonly(np.concatenate([getattr(item, name) for item in items])) for name in ("eligible", "low", "low_definition_available")))

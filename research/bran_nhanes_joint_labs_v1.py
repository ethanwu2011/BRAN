"""Source-local CBC + BIOPRO joins. Synthetic-testable; no IO or printing."""
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

from bran_clinical_semantics_v1 import CBC_FIELDS, NHANES_CBC_CODES, nhanes_cbc_observation
from bran_clinical_chemistry_semantics_v1 import CHEMISTRY_FIELDS
from bran_nhanes_chemistry_binding_v1 import NHANES_BIOPRO_RULES
from bran_nhanes_chemistry_observation_v2 import observed_biopro
from bran_nhanes_source_reader_v1 import link_nhanes_cycle, nhanes_numeric_key

FIELDS = CBC_FIELDS + CHEMISTRY_FIELDS


@dataclass(frozen=True, repr=False)
class JointRecord:
    person: str
    values: np.ndarray
    observed: np.ndarray
    calibration: np.ndarray
    age: object
    cycle: str


def joint_cycle_records(cycle, demographics, cbc_rows, chemistry_rows):
    """CBC anchors are retained even if BIOPRO is absent, never positional joins.

    Each laboratory file must link uniquely to the authenticated demographic
    cycle. CBC and BIOPRO are examination-cycle context, not proven same draw.
    """
    demographics, cbc_rows, chemistry_rows = list(demographics), list(cbc_rows), list(chemistry_rows)
    cbc_links = link_nhanes_cycle(cycle, demographics, cbc_rows)
    link_nhanes_cycle(cycle, demographics, chemistry_rows)
    rules = NHANES_BIOPRO_RULES[cycle]
    chemistry = {}
    for row in chemistry_rows:
        if not isinstance(row, Mapping) or not {'SEQN', *(r.column for r in rules.values())}.issubset(row):
            raise ValueError('invalid chemistry observation schema')
        chemistry[nhanes_numeric_key(row['SEQN'])] = row
    for row in cbc_rows:
        if not isinstance(row, Mapping) or not {'SEQN', *NHANES_CBC_CODES}.issubset(row):
            raise ValueError('invalid CBC observation schema')
        person = nhanes_numeric_key(row['SEQN'])
        values = np.full(len(FIELDS), np.nan)
        observed = np.zeros(len(FIELDS), bool)
        calibration = np.zeros(len(FIELDS), np.uint8)
        for code, field in NHANES_CBC_CODES.items():
            item = nhanes_cbc_observation(cycle, code, row[code], provenance=1)
            if item.observed and item.value > 0:
                j = FIELDS.index(field)
                values[j], observed[j] = item.value, True
        if observed[:len(CBC_FIELDS)].sum() < 2:
            continue
        if person in chemistry:
            chem = chemistry[person]
            for field, rule in rules.items():
                item = observed_biopro(cycle, field, rule.column, chem[rule.column], rule.source_unit, provenance=1)
                if item.observed:
                    j = FIELDS.index(field)
                    values[j], observed[j], calibration[j] = item.value, True, item.calibration_code
        yield JointRecord(person, values, observed, calibration, cbc_links.person_age[person], cycle)

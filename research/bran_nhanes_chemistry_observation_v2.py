"""Observed-only BIOPRO admission, including cycle-specific calibration.

V1 header/unit binding is preserved. This wrapper adds original-observation
provenance and the CDC analytic notes; no learned/imputed value is admitted.
"""
from dataclasses import dataclass
import math
from numbers import Integral

from bran_nhanes_chemistry_binding_v1 import convert_nhanes_biopro_value

ASSAY_POLICY = {
    'D_creatinine': 'apply -0.016 + 0.978 * original LBXSCR mg/dL once',
    'E_creatinine': 'use released LBXSCR; CDC already adjusted 2007 values; no second correction',
    'glucose': 'BIOPRO serum glucose context only; not fasting reference LBXGLU or an undiagnosed diabetes/prediabetes label',
    'ALT_AST': 'deferred U/L to canonical IU/L assay-unit harmonization',
    'provenance': 'source measured value only; assay calibration is not model imputation',
    'population': 'no population prevalence inference without survey design and weights',
}


@dataclass(frozen=True, repr=False)
class CalibratedObservation:
    value: float
    observed: bool
    calibration_code: int  # 0 none/missing; 1 D affine; 2 E source-calibrated.
    status: str


def observed_biopro(cycle, field, column, value, unit, *, provenance):
    if isinstance(provenance, bool) or not isinstance(provenance, Integral) or provenance != 1:
        return CalibratedObservation(math.nan, False, 0, 'not_original_observation')
    item = convert_nhanes_biopro_value(cycle, field, column, value, unit)
    if not item.observed:
        return CalibratedObservation(math.nan, False, 0, item.provenance)
    number = item.value
    calibration = 0
    if field == 'creatinine':
        if cycle == 'D':
            number = -0.016 + 0.978 * number
            calibration = 1
        else:
            calibration = 2
    if not math.isfinite(number) or number < 0:
        return CalibratedObservation(math.nan, False, 0, 'invalid_after_calibration')
    return CalibratedObservation(number, True, calibration, 'observed_source_assay')

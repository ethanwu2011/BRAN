"""Pure candidate target rules for the prespecified 2022/2023 adult study.

Not a source-admission decision or data loader. Inputs and returned masks
are private; callers must never send them to model-visible output. Rules
are grounded in guide PDF pages 52, 101, 182, 188, 247 and explicitly exclude
assay-report boundaries as a conservative analysis choice.
"""
from dataclasses import dataclass
import numpy as np

GUIDE_SHA256 = '921818c62267bd2949dd08e7d0143ef8cd30eb3086722e696481aa162ba42ce3'


def require(condition):
    if not condition:
        raise ValueError('knhanes_target_contract_failed')


@dataclass(frozen=True, repr=False)
class TargetMasks:
    eligible: np.ndarray
    low: np.ndarray
    low_definition_available: np.ndarray

    def __repr__(self):
        return '<PrivateKNHANESTargetMasks>'

    def __reduce__(self):
        raise TypeError('private_target_masks_not_serializable')


def prepare_target(hb, age, sex, pregnancy, original_observation):
    """Adult 19–79 analysis domain; no interpretation of top-coded age 80.

    original_observation must come from the admitted source loader, not from
    another prediction or an imputed value. This kernel cannot authenticate
    that flag. Missing pregnancy does not exclude someone from the primary
    error analysis, but prevents assigning a female clinical low-Hb stratum.
    HE_prg is 0=no, 1=yes, 8=male N/A (NOT 2=no).
    """
    require(type(hb) is np.ndarray and hb.ndim == 1 and hb.dtype.kind in 'fiu')
    for a in (age, sex, pregnancy):
        require(type(a) is np.ndarray and a.shape == hb.shape and a.dtype.kind in 'fiu')
    require(type(original_observation) is np.ndarray
            and original_observation.shape == hb.shape
            and original_observation.dtype == np.dtype(bool))
    eligible = (original_observation & np.isfinite(hb) & (hb > .8) & (hb < 26.)
                & np.isfinite(age) & (age >= 19) & (age <= 79) & (age == np.floor(age)))
    male = eligible & (sex == 1)
    nonpregnant = eligible & (sex == 2) & (pregnancy == 0)
    pregnant = eligible & (sex == 2) & (pregnancy == 1)
    known = male | nonpregnant | pregnant
    low = (male & (hb < 13)) | (nonpregnant & (hb < 12)) | (pregnant & (hb < 11))
    arrays = [np.array(a, copy=True) for a in (eligible, low, known)]
    for a in arrays:
        a.setflags(write=False)
    return TargetMasks(*arrays)

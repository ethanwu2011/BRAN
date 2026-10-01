"""No-I/O input removal for a frozen BRAN missingness diagnostic.

Inputs and outputs remain caller-local. The caller must authenticate observed
provenance, eligibility, identity and splits; this module cannot do that.
"""
from dataclasses import dataclass

import numpy as np

PATTERNS = (
    "available", "no_retina", "clinical_drop25", "clinical_drop50",
    "clinical_drop75", "whole_cbc_hidden", "whole_cbc_no_retina",
    "all_clinical_hidden",
)
MASK_SEED = 93711


@dataclass(frozen=True, repr=False)
class MaskedInputs:
    clinical: np.ndarray
    clinical_mask: np.ndarray
    retinal: np.ndarray
    retinal_mask: np.ndarray
    available: np.ndarray


def remove_inputs(c, cm, r, rm, slots, pattern, *, seed=MASK_SEED):
    """Remove extra evidence without inventing observations or mutating arrays.

    Random dropout is over eligible observed clinical *fields*, not just blood.
    The fixed random matrix is shared across severities, so masks are nested.
    Age is intentionally not an input here and is left unchanged by the caller.
    """
    invalid = "missingness_stress_input_invalid"
    if not all(isinstance(x, np.ndarray) for x in (c, cm, r, rm)):
        raise ValueError(invalid)
    if c.ndim != 2 or c.shape[1] != 59:
        raise ValueError(invalid)
    n = len(c)
    if (n < 1 or cm.shape != c.shape or r.shape != (n, 384) or rm.shape != (n,)
            or cm.dtype != bool or rm.dtype != bool
            or c.dtype.kind not in "fiu" or r.dtype.kind not in "fiu"
            or not isinstance(pattern, str) or pattern not in PATTERNS
            or type(seed) is not int or seed < 0
            or not isinstance(slots, tuple) or len(slots) != 9
            or any(type(s) is not int or not 0 <= s < 48 for s in slots)
            or len(set(slots)) != 9):
        raise ValueError(invalid)
    vc, vr = cm.copy(), rm.copy()
    if pattern in ("no_retina", "whole_cbc_no_retina"):
        vr[:] = False
    if pattern in ("whole_cbc_hidden", "whole_cbc_no_retina"):
        vc[:, slots] = False
    if pattern == "all_clinical_hidden":
        vc[:] = False
    fractions = {"clinical_drop25": .25, "clinical_drop50": .5, "clinical_drop75": .75}
    if pattern in fractions:
        vc &= np.random.default_rng(seed).random(cm.shape) >= fractions[pattern]
    # Do not multiply masked NaN/Inf by zero. Hidden payloads must be irrelevant.
    clean_c = np.where(vc, c, 0.)
    clean_r = np.where(vr[:, None], r, 0.)
    if not np.isfinite(clean_c).all() or not np.isfinite(clean_r).all():
        raise ValueError(invalid)
    return MaskedInputs(clean_c, vc, clean_r, vr, vc.any(1) | vr)


def assert_no_input_leak(masked, original_cm, original_rm, slots, pattern):
    """Data-independent contract assertions; never embed values in errors."""
    ok = (not np.any(masked.clinical_mask & ~original_cm)
          and not np.any(masked.retinal_mask & ~original_rm)
          and np.all(masked.clinical[~masked.clinical_mask] == 0)
          and np.all(masked.retinal[~masked.retinal_mask] == 0)
          and np.array_equal(masked.available, masked.clinical_mask.any(1) | masked.retinal_mask))
    if pattern in ("whole_cbc_hidden", "whole_cbc_no_retina"):
        ok = ok and not masked.clinical_mask[:, slots].any() and not masked.clinical[:, slots].any()
    if not ok:
        raise ValueError("missingness_stress_mask_contract_failed")

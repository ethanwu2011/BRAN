"""Local fixed-checkpoint old/new retinal-input bridge; no fitting or I/O."""
import numpy as np
import bran_native_screening_kernel_v1 as screening
import bran_native_cbc_decoders_v1 as decoding
from run_bran_missingness_stress_v1 import parameter_digest

VERSIONS = ("historical", "authenticated")
ROUTES = ("both", "clinical", "retinal")


def require(ok):
    if not ok:
        raise ValueError("retinal_input_bridge_contract_failed")


def infer(model, transform, clinical, observed, eligible, historical_retinal,
          new_retinal, retinal_present, new_present, age, names, *, batch_size=256):
    """Use the same supplied historical-training normalizers for both inputs.

    The caller authenticates checkpoint, folds, normalizers and new input bytes.
    All arrays returned here are private caller-local arrays, never a report.
    """
    require(not model.training)
    historical_retinal = np.asarray(historical_retinal)
    new_retinal = np.asarray(new_retinal)
    retinal_present, new_present = np.asarray(retinal_present), np.asarray(new_present)
    n = len(historical_retinal)
    require(historical_retinal.shape == new_retinal.shape == (n,384)
            and retinal_present.shape == new_present.shape == (n,)
            and retinal_present.dtype == new_present.dtype == np.dtype(bool)
            and np.array_equal(retinal_present,new_present)
            and np.isfinite(historical_retinal[retinal_present]).all()
            and np.isfinite(new_retinal[new_present]).all())
    normalizer_names = ("clinical_median","clinical_iqr","retinal_mean","retinal_scale","age_mean","age_scale")
    before_norm = {key:np.array(getattr(transform,key),copy=True) for key in normalizer_names}
    before = parameter_digest(model)
    screens, blood, support = {}, {}, {}
    clinical_payload = None
    for version, retina in zip(VERSIONS,(historical_retinal,new_retinal)):
        c,cm,r,aa = transform.apply(clinical,observed,eligible,retina,retinal_present,age)
        if clinical_payload is None:
            clinical_payload = tuple(np.array(x,copy=True) for x in (c,cm,aa))
        else:
            require(all(np.array_equal(x,y,equal_nan=True) for x,y in zip(clinical_payload,(c,cm,aa))))
        screens[version] = screening.predict_native(model,c,cm,r,retinal_present,aa,batch_size=batch_size)
        decoded, present = decoding.infer(model,c,cm,r,retinal_present,aa,names,
            transform.clinical_median,transform.clinical_iqr,
            pattern="whole_cbc_hidden",route="both",batch_size=batch_size)
        blood[version] = decoded["native"]
        support[version] = present
    require(before == parameter_digest(model)
            and all(np.array_equal(before_norm[key],getattr(transform,key),equal_nan=True) for key in normalizer_names))
    require(np.array_equal(screens["historical"]["clinical"],screens["authenticated"]["clinical"],equal_nan=True)
            and np.array_equal(support["historical"],support["authenticated"]))
    for route in ROUTES:
        require(np.array_equal(np.isfinite(screens["historical"][route]),np.isfinite(screens["authenticated"][route])))
    for version in VERSIONS:
        for x in screens[version].values(): x.setflags(write=False)
    return screens,blood,support["historical"]

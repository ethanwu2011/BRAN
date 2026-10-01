"""Local no-fit comparison of two existing heads on the SAME masked BRAN state."""
import numpy as np
from bran_clinical_semantics_v1 import CBC_FIELDS
from bran_native_screening_kernel_v1 import _validate

PATTERNS = ("single_target_hidden", "whole_cbc_hidden", "redcell_block_hidden")
ROUTES = ("both", "clinical")
HEADS = ("native", "generative")
REDCELL = ("hct", "hemoglobin", "mch", "mchc", "mcv", "rbc")


def require(ok):
    if not ok:
        raise ValueError("native_cbc_decoder_contract_failed")


def masked_targets(pattern):
    require(pattern in PATTERNS)
    return tuple(range(9)) if pattern != "redcell_block_hidden" else tuple(CBC_FIELDS.index(x) for x in REDCELL)


def infer(model, c, cm, r, rm, age, field_names, median, iqr, *, pattern, route, batch_size=256):
    import torch
    require(pattern in PATTERNS and route in ROUTES)
    c, cm, r, rm, age, batch_size = _validate(model, c, cm, r, rm, age, batch_size)
    require(not model.training and all(x.device.type == "cpu" for x in model.parameters()))
    require(callable(getattr(model, "decode", None)) and getattr(model.config, "clinical_continuous_dim", None) == 48)
    head = getattr(model, "cbc_joint_head", None)
    require(isinstance(head, torch.nn.Linear) and tuple(head.weight.shape) == (9,192) and head.bias is not None)
    require(len(field_names) == len(set(field_names)) == 59 and set(CBC_FIELDS) <= set(field_names))
    slots = tuple(field_names.index(x) for x in CBC_FIELDS)
    require(all(0 <= j < 48 for j in slots))
    median, iqr = np.asarray(median), np.asarray(iqr)
    require(median.shape == iqr.shape == (59,) and np.isfinite(median).all()
            and np.isfinite(iqr).all() and (iqr > 0).all())
    outputs = {name: np.full((len(c),9), np.nan) for name in HEADS}
    available = np.zeros((len(c),9), bool)
    targets = masked_targets(pattern)
    batches = [(j,) for j in targets] if pattern == "single_target_hidden" else [targets]
    with torch.inference_mode():
        for selected in batches:
            remove = [slots[j] for j in selected]
            hc, hm = np.array(c, copy=True), np.array(cm, copy=True)
            hc[:, remove] = 0; hm[:, remove] = False
            hc[~hm] = 0
            hrm = np.array(rm, copy=True) if route == "both" else np.zeros(len(rm), bool)
            hr = np.zeros_like(r); hr[hrm] = r[hrm]
            require(not hc[:, remove].any() and not hm[:, remove].any())
            expected = hm.any(1) | hrm
            for start in range(0,len(c),batch_size):
                stop = min(start+batch_size,len(c))
                at = torch.tensor(age[start:stop], dtype=torch.float32)
                state = model.encode(torch.tensor(hc[start:stop],dtype=torch.float32),
                    torch.tensor(hm[start:stop],dtype=torch.bool),
                    torch.tensor(hr[start:stop,None],dtype=torch.float32),
                    torch.tensor(hrm[start:stop,None],dtype=torch.bool),at)
                require(state.mean.shape == (stop-start,192)
                        and np.array_equal(~state.abstain.numpy(),expected[start:stop]))
                normalized = {"native": head(state.mean).numpy(),
                    "generative": model.decode(state,at)["continuous_mean"].numpy()[:,slots]}
                require(all(x.shape == (stop-start,9) and np.isfinite(x).all() for x in normalized.values()))
                rows = np.flatnonzero(expected[start:stop])
                cols = np.asarray(selected)
                for name, x in normalized.items():
                    original = x*iqr[list(slots)] + median[list(slots)]
                    require(np.isfinite(original).all())
                    outputs[name][np.ix_(rows+start,cols)] = original[np.ix_(rows,cols)]
                available[np.ix_(rows+start,cols)] = True
    for x in outputs.values():
        require(np.array_equal(np.isfinite(x),available))
        x.setflags(write=False)
    available.setflags(write=False)
    return outputs, available

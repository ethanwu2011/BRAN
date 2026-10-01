"""DRAFT arithmetic only. Not connected to any model fit or release.

Teacher and scale must be outer-training-only. Teacher/student must see exactly
the same available clinical values, age and observation indicators. Teacher sees
no retinal evidence. Caller must remove completion targets before BOTH routes.
This function cannot authenticate that upstream provenance or masking contract.
"""
import torch


def clinical_preservation_loss(student_mean, teacher_mean, teacher_scale, clinical_present):
    """Mean scale-normalized MSE of clinical-private coordinates, rows equally weighted.

The returned loss is unweighted. This draft proposes coefficient0.1, but does
not apply it or authorize an experiment. Scale is max(1, training SD) per
teacher coordinate; neither teacher nor scale receives gradients. No age-only
or physiology-empty row enters the loss. State order is shared64, retina64,
clinical64. A block-local loss can still backpropagate through shared weights.
"""
    tensors = (student_mean, teacher_mean, teacher_scale, clinical_present)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError('preservation inputs must be tensors')
    if (student_mean.ndim != 2 or student_mean.shape[1] != 192
        or teacher_mean.shape != student_mean.shape or teacher_scale.shape != (64,)
        or clinical_present.shape != (student_mean.shape[0],) or student_mean.shape[0] == 0):
        raise ValueError('preservation shape contract invalid')
    if (student_mean.dtype not in (torch.float32, torch.float64)
        or teacher_mean.dtype != student_mean.dtype or teacher_scale.dtype != student_mean.dtype
        or clinical_present.dtype != torch.bool
        or any(t.device != student_mean.device for t in tensors)):
        raise ValueError('preservation dtype or device contract invalid')
    if (any(not torch.isfinite(t).all() for t in (student_mean,teacher_mean,teacher_scale))
        or torch.any(teacher_scale < 1)):
        raise ValueError('preservation inputs or scale invalid')
    # Select supported rows before squaring, so excluded finite-but-extreme
    # coordinates cannot yield inf * 0 and contaminate the scalar.
    residual = (student_mean[clinical_present,128:192]
                - teacher_mean[clinical_present,128:192].detach()) / teacher_scale.detach()
    row_loss = residual.square().mean(dim=1)
    loss = row_loss.sum() / max(1, row_loss.shape[0])
    if not torch.isfinite(loss):
        raise ValueError('preservation loss is nonfinite')
    return loss

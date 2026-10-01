"""Pure two-source matched optimizer; no source authorization, I/O or sampling.

Returned weights/optimizer/RNG receipts are private. They are not a resume API.
"""
import torch

import bran_retinal_adaptation_training_v1 as old
from bran_retinal_multisource_patient_kernel_v1 import RetinalMultisourcePatientKernel

ERROR = 'retinal multisource training failed'
SOURCES = ('brset', 'odir')
FIELDS = ('images', 'patch_mask', 'group_index', 'labels', 'observed')


def require(condition):
    if not condition:
        raise ValueError(ERROR) from None


def batch(value, device):
    require(type(value) is dict and set(value) == set(SOURCES))
    result = {}
    for source in SOURCES:
        item = value[source]
        require(type(item) is dict and set(item) == set(FIELDS))
        require(all(isinstance(item[field], torch.Tensor) for field in FIELDS))
        result[source] = {key: item[key].to(device=device) for key in FIELDS}
    return result


def train_arm(base_encoder, batches, *, source_weight, positive_weights, steps=512,
              seed=74191, device='cpu', checkpoint_callback=None):
    """Matched BRSET + (0 or 1)*ODIR continuation from the same general encoder.

    Both arms execute both source objectives. The coefficient is fixed for the
    arm, not a fit-selected loss weight. Callers supply only source-admitted
    training patients, preprocessing, stream binding and private persistence.
    """
    try:
        require(type(source_weight) in (int, float) and source_weight in (0, 1))
        require(type(steps) is int and steps > 0 and type(seed) is int and 0 <= seed < 2**32)
        require(checkpoint_callback is None or callable(checkpoint_callback))
        require(type(positive_weights) is dict and set(positive_weights) == set(SOURCES))
        destination = torch.device(device)
        require(destination.type in ('cpu', 'mps'))
        require(destination.type != 'mps' or torch.backends.mps.is_available())
        weights = {}
        for source, width in (('brset', 13), ('odir', 8)):
            value = positive_weights[source]
            require(isinstance(value, torch.Tensor) and value.is_floating_point()
                    and value.shape == (width,) and torch.isfinite(value).all().item()
                    and (value > 0).all().item())
            weights[source] = value.to(device=destination)
        iterator = iter(batches)
        old._seed_everything(seed)
        kernel = RetinalMultisourcePatientKernel(base_encoder).to(destination).train()
        trainable = [parameter for parameter in kernel.parameters() if parameter.requires_grad]
        require(trainable and old._values_finite(list(kernel.parameters())))
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=.04, betas=(.9, .95))
        teacher_named = dict(kernel.teacher.named_parameters())
        teacher_updated = [teacher_named[name] for name, parameter in kernel.student.named_parameters()
                           if parameter.requires_grad]

        def receipt(completed):
            value = old._receipt(kernel, optimizer, completed_step=completed, steps=steps,
                                 lr=1e-4, label_weight=1, weight_decay=.04, ema=.996, seed=seed)
            value['schema'] = 'bran-retinal-multisource-training-receipt-v1'
            value['training_config'].pop('label_weight')
            value['training_config']['source_weight'] = float(source_weight)
            value['training_config']['source_heads'] = {'brset': 13, 'odir': 8}
            value['source_forwards_per_step'] = {'brset': 1, 'odir': 1}
            return value

        for step in range(steps):
            current = batch(next(iterator), destination)
            for group in optimizer.param_groups:
                group['lr'] = old.learning_rate_at_step(step, steps=steps, lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            loss_brset = kernel.grouped_objective('brset', **current['brset'], positive_weight=weights['brset'])
            loss_odir = kernel.grouped_objective('odir', **current['odir'], positive_weight=weights['odir'])
            # A broken added-source branch must fail even when its weight is 0.
            require(bool(torch.isfinite(loss_brset).item()) and bool(torch.isfinite(loss_odir).item()))
            loss = loss_brset + float(source_weight) * loss_odir
            require(bool(torch.isfinite(loss).item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step(); kernel.update_teacher(.996)
            require(old._values_finite([*trainable, *teacher_updated]))
            completed = step + 1
            if checkpoint_callback is not None and (completed % 32 == 0 or completed == steps):
                checkpoint_callback(receipt(completed))
        kernel.eval(); kernel.teacher.eval()
        return kernel, receipt(steps)
    except Exception:
        raise ValueError(ERROR) from None

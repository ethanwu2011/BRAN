"""Tiny generated-tensor train-save-load-infer walkthrough."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from bran_multisource_age_v2 import AgeBatch, normalize_age
from bran_robust_clinical_r7 import BRANRobustClinicalR7
from bran_r7_release.inference import (
    infer,
    load_checkpoint,
    make_field_map_binding,
    make_transform_binding,
    save_checkpoint,
)


def synthetic_transform(eligible_indices):
    eligible = torch.zeros(59, dtype=torch.bool)
    eligible[list(eligible_indices)] = True
    return SimpleNamespace(
        clinical_median=torch.zeros(59, dtype=torch.float64),
        clinical_iqr=torch.ones(59, dtype=torch.float64),
        retinal_mean=torch.zeros(384, dtype=torch.float64),
        retinal_scale=torch.ones(384, dtype=torch.float64),
        age_mean=50.0,
        age_scale=15.0,
        eligible=eligible,
        heldout_fold=0,
        fold_identity_sha256="c" * 64,
        training_indices_sha256="d" * 64,
    )


def synthetic_age(batch_size):
    raw = AgeBatch(
        value=torch.full((batch_size,), float("nan")),
        lower=torch.full((batch_size,), float("nan")),
        upper=torch.full((batch_size,), float("nan")),
        kind=torch.full((batch_size,), 3, dtype=torch.long),
    )
    return normalize_age(raw, mean=50.0, scale=15.0)


def main():
    torch.manual_seed(509)
    torch.set_num_threads(1)
    batch_size = 8
    eligible = tuple(range(43))
    cbc = tuple(range(9))
    names = tuple(f"synthetic_feature_{index:02d}" for index in range(59))
    field_map = make_field_map_binding(names, eligible, cbc)
    transform = synthetic_transform(eligible)
    transform_binding = make_transform_binding(transform)

    model = BRANRobustClinicalR7("mlp", eligible, cbc)
    clinical = torch.randn(batch_size, 59)
    clinical_observed = torch.ones(batch_size, 59, dtype=torch.bool)
    retinal = torch.randn(batch_size, 2, 384)
    retinal_visible = torch.ones(batch_size, 2, dtype=torch.bool)
    age = synthetic_age(batch_size)
    pseudo_target = (clinical[:, 0] > 0).to(torch.float32)

    # Three tiny updates touch one head only. This is a mechanics demo, not a
    # training recipe and not a fit to retained study data.
    optimizer = torch.optim.Adam(model.screening_joint_head.parameters(), lr=1e-3)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(clinical, clinical_observed, retinal, retinal_visible, age)
        logits = model.screening_joint_head(state.mean.detach())[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, pseudo_target)
        loss.backward()
        optimizer.step()

    with tempfile.TemporaryDirectory(prefix="bran-r7-synthetic-") as directory:
        checkpoint_path = Path(directory) / "synthetic.pt"
        binding = save_checkpoint(
            checkpoint_path,
            model,
            field_map_binding=field_map,
            input_transform=transform,
        )
        loaded = load_checkpoint(
            checkpoint_path,
            expected_binding=binding,
            expected_field_map_binding=field_map,
            expected_transform_binding=transform_binding,
            input_transform=transform,
        )
        native = infer(loaded, clinical, clinical_observed, retinal, retinal_visible, age,
                       expected_binding=binding)
        completed = infer(loaded, clinical, clinical_observed, retinal, retinal_visible, age,
                          expected_binding=binding, erase_cbc_targets=True)

        empty_clinical = torch.full_like(clinical, float("nan"))
        empty_clinical_mask = torch.zeros_like(clinical_observed)
        empty_retinal = torch.full_like(retinal, float("nan"))
        empty_retinal_mask = torch.zeros_like(retinal_visible)
        empty = infer(loaded, empty_clinical, empty_clinical_mask, empty_retinal,
                      empty_retinal_mask, age, expected_binding=binding)

        print(f"state shape {tuple(native.state_mean.shape)}")
        print(f"screening shape {tuple(native.screening_probability.shape)}")
        print(f"CBC output shapes {tuple(native.cbc_standardized.shape)} and "
              f"{tuple(native.cbc_original_units.shape)}")
        print(f"completion target flags {int(completed.completion_target_mask.sum())}")
        print(f"all-empty abstentions {int(empty.abstained.sum())}")


if __name__ == "__main__":
    main()

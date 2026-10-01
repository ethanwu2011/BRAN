"""Generate fake tensors and run one BRAN R7 encoder pass."""

import torch

from bran_r7_release import AgeBatch, BRANRobustClinicalR7, normalize_age


def main():
    torch.manual_seed(41)
    batch_size = 4

    # These are fake positional indices for shape testing, not field mappings.
    fake_eligible_indices = tuple(range(43))
    fake_cbc_indices = tuple(range(9))
    model = BRANRobustClinicalR7("mlp", fake_eligible_indices, fake_cbc_indices).eval()

    clinical = torch.randn(batch_size, 59)
    clinical_observed = torch.rand(batch_size, 59) > 0.2
    retinal = torch.randn(batch_size, 2, 384)
    retinal_visible = torch.ones(batch_size, 2, dtype=torch.bool)
    age_source = AgeBatch(
        value=torch.full((batch_size,), float("nan")),
        lower=torch.full((batch_size,), float("nan")),
        upper=torch.full((batch_size,), float("nan")),
        kind=torch.full((batch_size,), 3, dtype=torch.long),
    )
    age7 = normalize_age(age_source, mean=50.0, scale=15.0)

    with torch.no_grad():
        state = model.encode(clinical, clinical_observed, retinal, retinal_visible, age7)

    print(f"synthetic state shape {tuple(state.mean.shape)}")
    print(f"synthetic abstentions {int(state.abstain.sum())}")


if __name__ == "__main__":
    main()

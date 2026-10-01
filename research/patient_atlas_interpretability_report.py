"""Aggregate, fail-closed interpretation report for paired Atlas factors."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import torch

from interpret_soft_patient_atlas import (
    FactorProfileThresholds,
    FactorRelevanceReport,
    classify_factor_modality_profiles,
    likelihood_normalized_loading_signatures,
)
from soft_patient_atlas import SoftPatientAtlas


DEFAULT_PROFILE_THRESHOLDS = FactorProfileThresholds(
    min_amplitude=1e-4,
    min_expected_fisher=1e-8,
    shared_minimum_share=0.25,
)


def build_aggregate_factor_report(
    *,
    model: SoftPatientAtlas,
    relevance: FactorRelevanceReport,
    clinical_feature_names: Sequence[str],
    mean_eye_precision_increment: torch.Tensor,
    mean_blood_precision_increment: torch.Tensor,
    provenance: Mapping[str, Any],
    thresholds: FactorProfileThresholds = DEFAULT_PROFILE_THRESHOLDS,
    top_clinical_features: int = 5,
) -> dict[str, Any]:
    """Describe every matched factor pair without assigning biological names.

    Axes remain explicitly unresolved until independent real-data refits pass
    signed-permutation stability gates. Top clinical fields are decoder-loading
    descriptions, not causal effects or approved factor names.
    """

    latent_dim = int(model.config.latent_dim)
    feature_names = tuple(str(value) for value in clinical_feature_names)
    if len(feature_names) != model.config.num_blood_features:
        raise ValueError("clinical feature names do not match the model schema")
    if len(feature_names) != len(set(feature_names)) or any(not value for value in feature_names):
        raise ValueError("clinical feature names must be nonempty and unique")
    if relevance.latent_dim != latent_dim:
        raise ValueError("factor relevance and model latent dimensions differ")
    if (
        mean_eye_precision_increment.shape != (latent_dim,)
        or mean_blood_precision_increment.shape != (latent_dim,)
    ):
        raise ValueError("mean evidence precisions must have shape [latent_dim]")
    if top_clinical_features <= 0 or top_clinical_features > len(feature_names):
        raise ValueError("top_clinical_features is outside the clinical schema")
    if not bool(
        torch.isfinite(mean_eye_precision_increment).all()
        and torch.isfinite(mean_blood_precision_increment).all()
    ):
        raise ValueError("mean evidence precisions must be finite")

    identifiable = torch.zeros(
        latent_dim,
        dtype=torch.bool,
        device=relevance.eye_expected_fisher.device,
    )
    profiles = classify_factor_modality_profiles(
        eye_amplitude=model.decoder.eye_amplitude.detach(),
        blood_amplitude=model.decoder.blood_amplitude.detach(),
        eye_expected_fisher=relevance.eye_expected_fisher.detach(),
        blood_expected_fisher=relevance.blood_clinical_expected_fisher.detach(),
        identifiable_axis=identifiable,
        thresholds=thresholds,
    )
    signatures = likelihood_normalized_loading_signatures(model.decoder).detach().cpu()
    clinical_signatures = signatures[:, model.config.eye_dim :]
    if clinical_signatures.shape != (latent_dim, len(feature_names)):
        raise ValueError("likelihood-normalized clinical signature has the wrong shape")

    factors: list[dict[str, Any]] = []
    evidence_profiles: list[str] = []
    for factor, profile in enumerate(profiles):
        values = clinical_signatures[factor]
        order = sorted(
            range(len(feature_names)),
            key=lambda index: (-abs(float(values[index])), feature_names[index]),
        )[:top_clinical_features]
        eye_precision = float(mean_eye_precision_increment[factor])
        blood_precision = float(mean_blood_precision_increment[factor])
        precision_total = max(eye_precision + blood_precision, 1e-12)
        eye_precision_share = eye_precision / precision_total
        if eye_precision_share > 0.75:
            evidence_profile = "eye_evidence_dominant"
        elif eye_precision_share < 0.25:
            evidence_profile = "blood_clinical_evidence_dominant"
        else:
            evidence_profile = "shared_evidence"
        evidence_profiles.append(evidence_profile)
        factors.append(
            {
                "factor_index": factor,
                "paired_vector_fields": {
                    "eye_index": factor,
                    "blood_clinical_index": latent_dim + factor,
                },
                "decoder_likelihood_profile": profile.profile,
                "evidence_precision_profile": evidence_profile,
                "eye_evidence_precision_share": eye_precision_share,
                "blood_clinical_evidence_precision_share": 1.0
                - eye_precision_share,
                "eye_combined_share": profile.eye_combined_share,
                "blood_clinical_combined_share": profile.blood_combined_share,
                "eye_decoder_amplitude": profile.eye_amplitude,
                "blood_clinical_decoder_amplitude": profile.blood_amplitude,
                "eye_expected_fisher": profile.eye_expected_fisher,
                "blood_clinical_expected_fisher": profile.blood_expected_fisher,
                "eye_heldout_deviance_reduction": float(
                    relevance.eye_deviance_reduction[factor]
                ),
                "blood_clinical_heldout_deviance_reduction": float(
                    relevance.blood_clinical_deviance_reduction[factor]
                ),
                "mean_eye_evidence_precision_increment": eye_precision,
                "mean_blood_clinical_evidence_precision_increment": blood_precision,
                "top_clinical_decoder_fields": [
                    {
                        "field": feature_names[index],
                        "signed_likelihood_normalized_loading": float(values[index]),
                        "absolute_likelihood_normalized_loading": abs(
                            float(values[index])
                        ),
                    }
                    for index in order
                ],
                "axis_identifiable_across_real_refits": False,
                "axis_interpretation_allowed": False,
                "approved_name": None,
            }
        )

    decoder_counts = Counter(profile.profile for profile in profiles)
    evidence_counts = Counter(evidence_profiles)
    return {
        "schema_version": "patient-atlas-paired-factor-interpretability-v2",
        "scope": "aggregate_heldout_decoder_relevance",
        "vector_schema": "paired_evidence_atlas_129",
        "factor_count": latent_dim,
        "profile_thresholds": {
            "min_amplitude": thresholds.min_amplitude,
            "min_expected_fisher": thresholds.min_expected_fisher,
            "shared_minimum_share": thresholds.shared_minimum_share,
        },
        "decoder_likelihood_profile_counts": dict(sorted(decoder_counts.items())),
        "evidence_precision_profile_counts": dict(sorted(evidence_counts.items())),
        "heldout_observation_counts": {
            "patients": relevance.patient_count,
            "eye_images": relevance.eye_image_count,
            "continuous_values": relevance.continuous_value_count,
            "binary_values": relevance.binary_value_count,
        },
        "predictive_uncertainty_integrated": relevance.predictive_uncertainty_integrated,
        "factors": factors,
        "axis_stability": {
            "real_refit_replicates_available": 1,
            "individual_axes_identifiable": 0,
            "individual_axes_named": 0,
            "signed_permutation_stability_required_before_naming": True,
            "unresolved_axes_must_be_reported_as_subspaces": True,
        },
        "interpretation_rules": {
            "top_fields_are_descriptive_decoder_loadings": True,
            "decoder_likelihood_profiles_are_not_cross_family_calibrated": True,
            "evidence_precision_profiles_describe_encoder_confidence_not_outcome_utility": True,
            "factor_ablation_is_not_a_causal_intervention": True,
            "modality_profile_is_not_an_approved_biological_name": True,
            "patient_level_explanations_released": False,
        },
        "provenance": dict(provenance),
        "patient_rows_emitted": False,
        "patient_identifiers_emitted": False,
        "patient_vectors_emitted": False,
        "per_patient_contributions_emitted": False,
    }


__all__ = ["DEFAULT_PROFILE_THRESHOLDS", "build_aggregate_factor_report"]

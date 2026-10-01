"""Private source-bound context for the prospective age-free clinical transfer.

This module does not replace retinal coordinates, load external clinical pools,
fit a model, or authorize any source access.  A locked local caller owns all
actual source calls and artifact authentication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bran_agefree_unified_jobs_v1 as jobs
import bran_agefree_unified_oof_v1 as oof
import bran_agefree_unified_source_v1 as clinical
import bran_retinal_unified_refit_source_v1 as oldsource
import run_bran_blood_learning_curve_v1 as blood
import run_bran_fm_learning_curve_v1 as fm
import run_bran_joint_subtyping_v1 as phenotype
import run_bran_raw_teacher_distillation_v1 as retained
import run_bran_named_fm_cache_v1 as fm_cache


ERROR = 'age-free unified context rejected'
PEOPLE = 1928
ROOT = oldsource.ROOT


def require(value):
    if not value:
        raise ValueError(ERROR) from None


@dataclass(frozen=True, repr=False)
class PrivateInputs:
    source_sha256: str
    data: dict
    labels: np.ndarray
    label_observed: np.ndarray
    registry_names: tuple
    endpoint_names: tuple
    inner_folds: dict
    plan: dict
    foundation_features: dict
    structure_values: dict

    def binding(self):
        return {'schema': 'bran-agefree-unified-private-inputs-v1',
                'source_sha256': self.source_sha256, 'data_sha256': oof.private_digest(self.data),
                'labels_sha256': oof.private_digest({'labels': self.labels, 'observed': self.label_observed}),
                'inner_sha256': oof.private_digest({str(key): value for key, value in self.inner_folds.items()}),
                'registry_names': list(self.registry_names), 'endpoint_names': list(self.endpoint_names),
                'plan_sha256': jobs.handoff.digest(self.plan),
                'foundation_sha256': oof.private_digest(self.foundation_features),
                'structure_sha256': oof.private_digest(self.structure_values)}

    def context(self, protocol_sha256):
        return {'data': self.data, 'plan': self.plan, 'plan_sha256': jobs.handoff.digest(self.plan),
                'inner_folds': self.inner_folds, 'registry_names': self.registry_names,
                'endpoint_names': self.endpoint_names, 'protocol_sha256': protocol_sha256,
                'source_descriptor_sha256': self.source_sha256}

    def structure_arguments(self):
        return {**self.structure_values, 'inner_folds': self.inner_folds}


def _cache_receipt():
    pin = clinical.safe.sha(fm_cache.PROTOCOL)
    record = fm_cache.verify(pin)
    protocol = fm_cache.read(fm_cache.PROTOCOL)
    return {'protocol_sha256': pin,
            'result_sha256': clinical.safe.sha(fm_cache.OUT / 'result.json'),
            'cache_manifest_sha256': record['cache_manifest_sha256'],
            'code_sha256': protocol['code_sha256']}


def _foundation():
    # The old unexecuted curve preparer reads endpoint identities from a field
    # absent in the count-only success report. Use the cache's authenticated
    # protocol-scope adapter; no historical code or reference value is changed.
    pin = clinical.safe.sha(fm_cache.PROTOCOL)
    protocol = fm_cache.load_protocol(pin)
    source = protocol['source']
    cache = fm._cache_binding(fm.DEFAULT_CACHE_MANIFEST)
    require(cache['outer_fold_sha256'] == source['learning_source']['outer_fold_sha256']
            and cache['inner_fold_sha256'] == source['learning_source']['inner_fold_sha256'])
    return {'schema': 'bran-cached-foundation-sources-v1',
            'learning_source': source['learning_source'], 'approved_reference': source['approved_reference'],
            'cache': cache, 'code_sha256': fm._code_hashes(), 'runtime': fm._runtime()}


def describe():
    """Authenticate source receipts; underlying local checks remain FD-silent."""
    with clinical.safe.quiet():
        try:
            cache_receipt = _cache_receipt()
            native, fm_source, blood_source = retained.prepare(), _foundation(), blood.prepare()
            require(fm_source['cache']['manifest_sha256'] == cache_receipt['cache_manifest_sha256'])
            structure_source, external = phenotype.prepare(), clinical.describe()
            names = native['native_source']['source']['endpoint_names']
            auth = native['native_source']['source']['authentication']
            require(names == fm_source['learning_source']['endpoint_names']
                    == structure_source['source']['endpoint_names']
                    and auth['outer_fold_sha256'] == fm_source['learning_source']['outer_fold_sha256']
                    == blood_source['outer_fold_sha256']
                    and auth['inner_fold_sha256'] == fm_source['learning_source']['inner_fold_sha256']
                    == blood_source['inner_fold_sha256'])
            expected = oldsource._reference_expectations(native, blood_source, fm_source)
            require(retained.prepare() == native and _foundation() == fm_source
                    and blood.prepare() == blood_source and phenotype.prepare() == structure_source
                    and clinical.describe(verify_raw=False) == external
                    and _cache_receipt() == cache_receipt)
            return {'schema': 'bran-agefree-unified-context-sources-v1', 'retained': native,
                    'foundation': fm_source, 'blood': blood_source, 'structure': structure_source,
                    'external_clinical': external, 'expected_reference': expected,
                    'foundation_cache': cache_receipt,
                    'retinal_inputs_replaced': False, 'retinal_candidate_contract_required': False,
                    'new_odir_source_used': False, 'source_permission_created': False,
                    'patient_level_output_permitted': False}
        except Exception:
            raise ValueError(ERROR) from None


def _copy(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    return result


def _assemble(description, context, foundation_features, cgm):
    """Bind unchanged historical retinal inputs to new age-free job inputs."""
    try:
        ctx, folds, clinical_values, observed, eligible, historical_retinal, present, age, names = context
        source = retained.native.source
        fm_source = description['foundation']
        endpoints, ids = fm._validate_context(fm_source, source, ctx, folds, clinical_values, observed,
                                              eligible, historical_retinal, present, age, names)
        require(len(ids) == len(set(ids)) == PEOPLE and len(names) == len(set(names)) == 59
                and list(ctx['feature_cohort'].patient_ids) == ids
                and set(ctx['raw_cohort'].split_labels) <= {'train', 'val'}
                and tuple(endpoints) == tuple(description['retained']['native_source']['source']['endpoint_names']))
        original_inner = fm._inner_assignments(fm_source, source, ctx, folds)
        auth = description['retained']['native_source']['source']['authentication']
        require(auth['inner_fold_sha256'] == fm_source['learning_source']['inner_fold_sha256'])
        inner = {}
        for fold in range(5):
            value = np.full(len(ids), -1, dtype=np.int64)
            value[np.asarray(folds) != fold] = original_inner[fold]
            inner[fold] = value
        labels = _copy(np.column_stack([ctx['labels_by_source'][item] for item in endpoints]), np.float64)
        label_observed = _copy(np.column_stack([ctx['observed_by_source'][item] for item in endpoints]), bool)
        memberships = {code: _copy(np.asarray(ctx['observed_by_source'][code], bool)
                                   & (np.asarray(ctx['labels_by_source'][code]) == 1), bool)
                       for code in oldsource.structure.CODES}
        require(type(cgm) is tuple and len(cgm) == 2)
        data = {'patient_ids': list(ids), 'folds': _copy(folds, np.int64),
                'clinical': _copy(clinical_values), 'observed': _copy(observed, bool),
                'eligible': _copy(eligible, bool), 'retinal': _copy(historical_retinal),
                'retinal_present': _copy(present, bool), 'age': _copy(age)}
        values = {'clinical59': data['clinical'], 'clinical_mask': data['observed'] & data['eligible'],
                  'profile_age': data['age'], 'site_ids': tuple(ctx['raw_cohort'].site_ids),
                  'outer_folds': data['folds'], 'memberships': memberships,
                  'clinical_labels': labels, 'label_observed': label_observed,
                  'cgmmean': _copy(cgm[0]), 'cgm_observed': _copy(cgm[1], bool),
                  'patient_ids': data['patient_ids'], 'inner_folds': inner}
        oldsource._validate_structure_values(values)
        plan = jobs.membership.build(data['patient_ids'], data['folds'], inner)
        jobs.context(data, plan, jobs.handoff.digest(plan), 'structure', inner, tuple(names), tuple(endpoints))
        values.update(plan=plan, plan_sha256=jobs.handoff.digest(plan))
        content = {key: value for key, value in values.items() if key != 'inner_folds'}
        result = PrivateInputs(oof.private_digest(description), data, labels, label_observed,
                               tuple(names), tuple(endpoints), inner, plan, foundation_features, content)
        oof._seal({'data': data, 'labels': labels, 'observed': label_observed,
                   'features': foundation_features, 'structure': content})
        result.binding()
        return result
    except Exception:
        raise ValueError(ERROR) from None


def load(description, expected_source_sha256):
    """Load the historical paired context only after twice-authenticated receipts."""
    with clinical.safe.quiet():
        try:
            require(type(description) is dict and oof.private_digest(description) == expected_source_sha256
                    and describe() == description)
            context = retained.native.source.io.load_context()
            ctx, folds, clinical_values, observed, eligible, historical_retinal, present, age, names = context
            ids = list(map(str, ctx['raw_cohort'].patient_ids))
            cache = description['foundation']['cache']
            fms = fm.load_fm_cache(
                cache, expected_current_inputs_sha256=fm.current_inputs_sha256(
                    clinical_values, observed, eligible, historical_retinal, present, age, names),
                expected_row_order_sha256=fm.row_order_sha256(ids),
                expected_outer_fold_sha256=description['retained']['native_source']['source']['authentication']['outer_fold_sha256'],
                expected_inner_fold_sha256=description['retained']['native_source']['source']['authentication']['inner_fold_sha256'],
            )
            protocol = oldsource._read(ROOT / phenotype.SOURCE_PROTOCOL)
            cgm_path = phenotype.cgm_source._target_path(protocol)
            require(clinical.safe.sha(cgm_path) == description['structure']['phenotype']['cgm_sha256'])
            with cgm_path.open('r', encoding='utf-8', newline='') as handle:
                targets = phenotype.target_io.parse_manifest(handle)
            cgm = phenotype.target_io.align(targets, tuple(ids))
            require(clinical.safe.sha(cgm_path) == description['structure']['phenotype']['cgm_sha256'])
            result = _assemble(description, context, fms, cgm)
            fm._verify_cache_files(cache, verify_embeddings=False)
            require(describe() == description)
            return result
        except Exception:
            raise ValueError(ERROR) from None

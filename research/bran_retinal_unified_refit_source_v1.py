"""Local, quiet source authentication for the prospective unified refit.

Caller owns the single heavy-job lock. Source admission and successful retinal
audit precede clinical/feature access. No action on import, fitting, publication,
new permissions, or invented patient matching. Returned objects are PRIVATE.
"""
import copy
from dataclasses import dataclass
import json

import numpy as np

import run_bran_multisource_paired_extraction_v1 as paired
import run_bran_raw_teacher_distillation_v1 as retained
import run_bran_fm_learning_curve_v1 as fm
import run_bran_blood_learning_curve_v1 as blood
import run_bran_joint_subtyping_v1 as phenotype
import bran_retinal_unified_refit_jobs_v1 as jobs
import bran_retinal_unified_refit_oof_v1 as oof
import bran_retinal_unified_refit_reference_checks_v1 as checks
import bran_joint_subtyping_kernel_v1 as structure

ROOT = paired.ROOT
ERROR = 'unified refit source authentication rejected'
PEOPLE = 1928


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def _read(path):
    return json.loads(path.read_text())


def _reference_expectations(native, blood_source, fm_source):
    """Only previously authenticated aggregate scalars, not new predictions."""
    names = tuple(native['native_source']['source']['endpoint_names'])
    native_report = _read(retained.native.OUT / 'aggregate.json')
    raw = retained.previous.baseline()
    cbc = _read(retained.native.source.OUT / 'aggregate.json')['retained_cbc_head_whole_panel']
    screen = {}
    for route in oof.ROUTES:
        screen['initial_' + route] = {name: native_report['results']['endpoints'][name]['arms']['native_' + route]['auroc']
                                      for name in names}
    for arm in ('raw_clinical', 'raw_retinal', 'raw_concat', 'late_average'):
        screen[arm] = {name: raw['endpoints'][name]['arms'][arm]['auroc'] for name in names}
    screen['blood_age'] = copy.deepcopy(blood_source['reference_endpoint_auroc'])
    screen.update(copy.deepcopy(fm_source['approved_reference']['endpoint_auroc']))
    require(all(cbc[field]['status'] == 'complete' for field in checks.CBC_FIELDS))
    result = {'screening': screen, 'native_whole_cbc': {
        field: {stat: cbc[field][stat] for stat in ('mae', 'mse')} for field in checks.CBC_FIELDS}}
    checks.validate_expected(result, names)
    return result


def describe(extraction_protocol_sha256, extraction_audit_sha256):
    """Authenticate prerequisite sources; returns a private source descriptor.

    This does not freeze a new protocol. Full source verification can process
    patient-derived material internally; never print this descriptor or call
    without the shared local compute lock.
    """
    with paired.r.quiet():
        try:
            # Positive source-stage admission must fail BEFORE other patient
            # sources/labels/features are opened. load_protocol enforces it.
            ep = paired.load_protocol(extraction_protocol_sha256)
            paired.authenticate_audit(extraction_protocol_sha256, extraction_audit_sha256)
            manifest = paired.read_json(paired.OUT / 'manifest.json')
            paired.authenticate_private(manifest['private_sha256'])
            native, fm_source, blood_source = retained.prepare(), fm.prepare(), blood.prepare()
            structure_source = phenotype.prepare()
            names = native['native_source']['source']['endpoint_names']
            auth = native['native_source']['source']['authentication']
            require(names == fm_source['learning_source']['endpoint_names']
                    == structure_source['source']['endpoint_names']
                    and auth['outer_fold_sha256'] == fm_source['learning_source']['outer_fold_sha256']
                    == blood_source['outer_fold_sha256']
                    and auth['inner_fold_sha256'] == fm_source['learning_source']['inner_fold_sha256']
                    == blood_source['inner_fold_sha256'])
            expected = _reference_expectations(native, blood_source, fm_source)
            # Recheck aggregate-source bindings after selecting canary values.
            require(retained.prepare() == native and blood.prepare() == blood_source
                    and fm.prepare() == fm_source and phenotype.prepare() == structure_source)
            paired.authenticate_private(manifest['private_sha256'])
            return {'schema': 'bran-retinal-unified-refit-sources-v1',
                'paired_extraction': {'protocol_sha256': extraction_protocol_sha256,
                    'audit_sha256': extraction_audit_sha256, 'protocol': ep,
                    'manifest_sha256': paired.r.sha(paired.OUT / 'manifest.json'),
                    'private_sha256': manifest['private_sha256']},
                'retained': native, 'foundation': fm_source, 'blood': blood_source,
                'structure': structure_source, 'expected_reference': expected,
                'source_permission_created': False, 'patient_level_output_permitted': False}
        except Exception:
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
        """Private JSON-only content pins, not a source-permission attestation."""
        return {'schema': 'bran-retinal-unified-refit-private-inputs-v1',
            'source_sha256': self.source_sha256, 'data_sha256': oof.private_digest(self.data),
            'labels_sha256': oof.private_digest({'labels': self.labels, 'observed': self.label_observed}),
            'inner_sha256': oof.private_digest({str(k): v for k, v in self.inner_folds.items()}),
            'registry_names': list(self.registry_names), 'endpoint_names': list(self.endpoint_names),
            'plan_sha256': jobs.handoff.digest(self.plan),
            'foundation_sha256': oof.private_digest(self.foundation_features),
            'structure_sha256': oof.private_digest(self.structure_values)}

    def context(self, protocol_sha256):
        return dict(data=self.data, plan=self.plan, plan_sha256=jobs.handoff.digest(self.plan),
            inner_folds=self.inner_folds, registry_names=self.registry_names,
            endpoint_names=self.endpoint_names, protocol_sha256=protocol_sha256)

    def structure_arguments(self):
        return {**self.structure_values, 'inner_folds': self.inner_folds}


def _validate_structure_values(values):
    """Reject malformed structure inputs before any transform/model preparation.

    This is only an interface/coherence check for the established structure
    panel.  It deliberately does not require disease-cohort support or CGM
    observation: both are represented downstream as unsupported/not-applicable
    aggregate evidence.
    """
    try:
        patient_ids = values['patient_ids']
        require(type(patient_ids) is list and patient_ids
                and all(type(value) is str and value for value in patient_ids)
                and len(set(patient_ids)) == len(patient_ids))
        n = len(patient_ids)
        clinical = values['clinical59']
        clinical_mask = values['clinical_mask']
        profile_age = values['profile_age']
        outer_folds = values['outer_folds']
        labels = values['clinical_labels']
        label_observed = values['label_observed']
        cgmmean = values['cgmmean']
        cgm_observed = values['cgm_observed']
        require(type(clinical) is np.ndarray and clinical.dtype.kind == 'f'
                and clinical.shape == (n, 59)
                and type(clinical_mask) is np.ndarray and clinical_mask.dtype == np.dtype(bool)
                and clinical_mask.shape == clinical.shape
                and np.isfinite(clinical[clinical_mask]).all()
                and type(profile_age) is np.ndarray and profile_age.dtype.kind == 'f'
                and profile_age.shape == (n,) and np.isfinite(profile_age).all()
                and type(outer_folds) is np.ndarray and outer_folds.dtype == np.dtype(np.int64)
                and outer_folds.shape == (n,) and set(outer_folds.tolist()) == set(range(5)))
        site_ids = values['site_ids']
        require(type(site_ids) is tuple and len(site_ids) == n
                and all(type(value) is str and value for value in site_ids)
                and type(labels) is np.ndarray and labels.dtype.kind == 'f'
                and labels.shape == (n, 26)
                and type(label_observed) is np.ndarray and label_observed.dtype == np.dtype(bool)
                and label_observed.shape == labels.shape
                and np.all(~label_observed | (np.isfinite(labels) & ((labels == 0) | (labels == 1))))
                and type(cgmmean) is np.ndarray and cgmmean.dtype.kind == 'f'
                and cgmmean.shape == (n,)
                and type(cgm_observed) is np.ndarray and cgm_observed.dtype == np.dtype(bool)
                and cgm_observed.shape == (n,)
                and np.isfinite(cgmmean[cgm_observed]).all()
                and np.all(cgmmean[cgm_observed] > 0))
        memberships = values['memberships']
        require(type(memberships) is dict and set(memberships) == set(structure.CODES)
                and all(type(value) is np.ndarray and value.dtype == np.dtype(bool)
                        and value.shape == (n,) for value in memberships.values()))
    except Exception:
        raise ValueError(ERROR) from None


def _assemble(description, context, candidate, foundation_features, cgm):
    """Bind authenticated raw inputs to the unchanged genuine paired cohort."""
    ctx, folds, clinical, observed, eligible, old_retina, present, age, names = context
    source = retained.native.source
    fm_source = description['foundation']
    endpoints, ids = fm._validate_context(fm_source, source, ctx, folds, clinical, observed,
                                         eligible, old_retina, present, age, names)
    require(len(ids) == len(set(ids)) == PEOPLE and len(names) == len(set(names)) == 59
            and list(ctx['feature_cohort'].patient_ids) == ids
            and set(ctx['raw_cohort'].split_labels) <= {'train', 'val'}
            and tuple(endpoints) == tuple(description['retained']['native_source']['source']['endpoint_names']))
    cohort, records, features, contract = candidate
    require(cohort['patient_ids'] == ids and np.array_equal(cohort['folds'], folds)
            and np.array_equal(cohort['retinal_present'], present)
            and contract == description['paired_extraction']['protocol']['encoder'])
    paired.source.extraction.old.kernel.validate_inventory(records)
    selection = description['paired_extraction']['protocol']['paired_source']
    # The paired source descriptor is a receipt binding, not a row-selection
    # identity. The actual selected row identities come from the original V3
    # protocol verified by QualifiedPairedSource during the completed audit.
    require(selection == paired.source.authenticate_binding())
    identity = {'patient_order_sha256': jobs.handoff.digest(ids),
        'fold_order_sha256': jobs.handoff.digest(folds.tolist()),
        'selection_sha256': jobs.handoff.inventory_sha256([{**r, 'source_sha256': '0' * 64} for r in records]),
        'rows': len(records), 'people': len(ids)}
    original_inner = fm._inner_assignments(fm_source, source, ctx, folds)
    inner = {}
    auth = description['retained']['native_source']['source']['authentication']
    require(auth['inner_fold_sha256'] == fm_source['learning_source']['inner_fold_sha256'])
    for fold in range(5):
        values = np.full(len(ids), -1, np.int64)
        values[folds != fold] = original_inner[fold]
        inner[fold] = values
    labels = np.column_stack([ctx['labels_by_source'][e] for e in endpoints]).astype(float)
    lm = np.column_stack([ctx['observed_by_source'][e] for e in endpoints]).astype(bool)
    memberships = {e: np.asarray(ctx['observed_by_source'][e], bool) & (np.asarray(ctx['labels_by_source'][e]) == 1)
                   for e in structure.CODES}
    require(type(cgm) is tuple and len(cgm) == 2)
    data = dict(records=records, image_features=features, patient_ids=ids, folds=folds,
        selection=identity, candidate_contract=contract, expected_checkpoint_sha256=contract['checkpoint_sha256'],
        original_retinal=old_retina, retinal_present=present, clinical=clinical,
        observed=observed, eligible=eligible, age=age)
    values = {'clinical59': clinical, 'clinical_mask': observed & eligible, 'profile_age': age,
        'site_ids': tuple(ctx['raw_cohort'].site_ids), 'outer_folds': folds,
        'memberships': memberships, 'clinical_labels': labels, 'label_observed': lm,
        'cgmmean': cgm[0], 'cgm_observed': cgm[1], 'patient_ids': ids,
        'inner_folds': inner}
    _validate_structure_values(values)
    plan = jobs.membership.build(ids, folds, inner)
    # Validate real pooling/membership and eligibility without fitting a model.
    jobs.context(data, plan, jobs.handoff.digest(plan), 'structure', inner, tuple(names), tuple(endpoints))
    values.update(plan=plan, plan_sha256=jobs.handoff.digest(plan))
    # The private digest accepts string-key dictionaries. Structure kwargs
    # retain int-key inner folds for the actual membership API, pinned by plan.
    content = {k: v for k, v in values.items() if k != 'inner_folds'}
    result = PrivateInputs(oof.private_digest(description), data, labels, lm, tuple(names),
                           tuple(endpoints), inner, plan, foundation_features, content)
    oof._seal({'data': data, 'labels': labels, 'observed': lm, 'features': foundation_features, 'structure': content})
    result.binding()
    return result


def load(description, expected_source_sha256):
    """Return exact private inputs only after the complete source authentication."""
    with paired.r.quiet():
        try:
            require(type(description) is dict and oof.private_digest(description) == expected_source_sha256)
            p = description['paired_extraction']
            require(describe(p['protocol_sha256'], p['audit_sha256']) == description)
            context = retained.native.source.io.load_context()
            ctx, folds, clinical, observed, eligible, old_retina, present, age, names = context
            ids = list(map(str, ctx['raw_cohort'].patient_ids))
            paired.authenticate_private(p['private_sha256'])
            cohort = paired.read_json(paired.PRIVATE / 'cohort.json')
            pool, candidate_present, contract = paired.load_private_refit_input(
                p['protocol_sha256'], p['audit_sha256'], patient_ids=ids,
                folds=np.asarray(folds, dtype=np.int64), selection=cohort['selection'])
            require(np.array_equal(candidate_present, present))
            records = paired.read_json(paired.PRIVATE / 'inventory.json')
            features = np.load(paired.PRIVATE / 'features.npy', allow_pickle=False)
            require(np.array_equal(paired.pool_arrays(records, features, ids)[0], pool))
            cache = description['foundation']['cache']
            fms = fm.load_fm_cache(cache,
                expected_current_inputs_sha256=fm.current_inputs_sha256(clinical, observed, eligible, old_retina, present, age, names),
                expected_row_order_sha256=fm.row_order_sha256(ids),
                expected_outer_fold_sha256=description['retained']['native_source']['source']['authentication']['outer_fold_sha256'],
                expected_inner_fold_sha256=description['retained']['native_source']['source']['authentication']['inner_fold_sha256'])
            ph = _read(ROOT / phenotype.SOURCE_PROTOCOL)
            cgm_path = phenotype.cgm_source._target_path(ph)
            require(paired.r.sha(cgm_path) == description['structure']['phenotype']['cgm_sha256'])
            with cgm_path.open('r', encoding='utf-8', newline='') as handle:
                targets = phenotype.target_io.parse_manifest(handle)
            cgm = phenotype.target_io.align(targets, tuple(ids))
            require(paired.r.sha(cgm_path) == description['structure']['phenotype']['cgm_sha256'])
            result = _assemble(description, context, (cohort, records, features, contract), fms, cgm)
            paired.authenticate_private(p['private_sha256'])
            fm._verify_cache_files(cache, verify_embeddings=False)
            require(describe(p['protocol_sha256'], p['audit_sha256']) == description)
            return result
        except Exception:
            raise ValueError(ERROR) from None

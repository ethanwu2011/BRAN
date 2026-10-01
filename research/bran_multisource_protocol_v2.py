"""V2 source closure and fixed-budget protocol; no patient I/O or model fitting.

Receipts must be supplied by a separately authenticated local source adapter.
File presence is never qualification. A constructed protocol is a specification,
not a training result or a promoted release. Exact source example counts, if
supplied to build_protocol, are private caller-local values and are not returned.
"""
from __future__ import annotations

import hashlib
import json
import math
import re


SOURCE_POLICY = {
    'aireadi': ('paired', 'development', 'aireadi'),
    'mimiciii': ('clinical', 'development', 'mimic'),
    'mimiciv': ('clinical', 'development', 'mimic'),
    'eicu': ('clinical', 'development', 'eicu'),
    'nhanes_exposed': ('clinical', 'development', 'nhanes'),
    'nwicu': ('clinical', 'development', 'nwicu'),
    'sicdb': ('clinical', 'development', 'sicdb'),
    'zigong': ('clinical', 'development', 'zigong'),
    'hirid': ('clinical', 'protected', 'hirid'),
    'inspire': ('clinical', 'protected', 'inspire'),
    'knhanes': ('clinical_structured_eye', 'protected', 'knhanes'),
    'brset': ('retinal', 'development', 'brset'),
    'dr_unified': ('retinal', 'development', 'dr_unified'),
    'odir': ('retinal', 'development', 'odir'),
    'jsiec': ('retinal', 'development', 'jsiec'),
    'fd3611': ('retinal', 'protected', 'fd3611'),
    'trihemo_mcv': ('clinical', 'protected', 'trihemo_mcv'),
    'amsterdamumcdb': ('clinical', 'deferred', 'amsterdamumcdb'),
}
ARMS = ('mlp', 'token')
PARAMETERS = {
    'state_width': 192, 'clinical_slots': 59, 'eligible_continuous_count': 43,
    'retinal_width': 384, 'native_screening_outputs': 26, 'native_cbc_outputs': 9,
    'age_width': 7, 'age_augmentation': [.1, .1, .8],
    'token_layers': 2, 'token_width': 128, 'token_heads': 4,
    'fold_count': 5, 'seed_base': 94101, 'learning_rate': .0001,
    'weight_decay': .0001, 'gradient_clip': 5., 'stage_a_batch': 256,
    'stage_b_steps': 1500, 'stage_c_steps': 1500, 'paired_batch': 96,
    'rehearsal_weight': .5, 'screening_weight': 1., 'cbc_weight': .5,
    'visible_weight': .1, 'kl_max': .001, 'kl_warmup_steps': 300,
    'backbone_frozen': True, 'coordinate_preservation_weight': 0.,
    'source_id_is_model_input': False, 'automatic_promotion': False,
    'screening_routes': ['both', 'both', 'both', 'clinical', 'retinal'],
    'completion_patterns': ['single_target_hidden', 'whole_cbc_hidden',
        'red_cell_hidden', 'single_target_no_retina', 'whole_cbc_no_retina',
        'red_cell_no_retina'],
    'bootstrap_draws': 1000, 'minimum_valid_bootstrap_draws': 900,
    'normalization': 'recipient_ai_outer_training_only',
}
REASONS = {'qualified', 'pending_source_qualification', 'pending_access',
    'pending_units', 'pending_grouping', 'pending_exposure',
    'pending_feature_binding', 'not_located', 'user_deferred',
    'incompatible_observed_truth'}
EVIDENCE_KEYS = {'qualification_sha256', 'canonical_input_sha256',
    'eligibility_sha256', 'grouping_sha256', 'exposure_audit_sha256'}


def require(ok):
    if not ok:
        raise ValueError('bran_multisource_v2_contract_failed')


def is_hash(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def pending_receipts():
    """Closed, row-free initial ledger; never infer qualification from existence."""
    return {name: {'disposition': 'deferred' if role == 'deferred' else 'pending',
                   'reason': 'user_deferred' if role == 'deferred'
                   else 'pending_source_qualification',
                   'evidence': None}
            for name, (_, role, _) in SOURCE_POLICY.items()}


def validate_receipts(receipts):
    require(type(receipts) is dict and set(receipts) == set(SOURCE_POLICY))
    for name, cell in receipts.items():
        require(type(cell) is dict and set(cell) == {'disposition', 'reason', 'evidence'})
        require(type(cell['disposition']) is str and type(cell['reason']) is str)
        require(cell['reason'] in REASONS)
        role = SOURCE_POLICY[name][1]
        allowed = {'pending', 'excluded', 'deferred'}
        if role == 'development': allowed.add('training')
        if role == 'protected': allowed.add('protected_evaluation')
        if role == 'deferred': allowed = {'deferred'}
        require(cell['disposition'] in allowed)
        if cell['disposition'] in ('training', 'protected_evaluation'):
            require(cell['reason'] == 'qualified')
            ev = cell['evidence']
            require(type(ev) is dict and set(ev) == EVIDENCE_KEYS)
            require(all(is_hash(v) for v in ev.values()))
        else:
            require(cell['reason'] != 'qualified' and cell['evidence'] is None)
        if role == 'deferred': require(cell['reason'] == 'user_deferred')


def readiness(receipts):
    validate_receipts(receipts)
    training = [s for s, c in receipts.items() if c['disposition'] == 'training']
    checks = {
        'paired_source_admitted': 'aireadi' in training,
        'external_clinical_admitted': any(SOURCE_POLICY[s][0] == 'clinical' for s in training),
        'external_retinal_admitted': any(SOURCE_POLICY[s][0] == 'retinal' for s in training),
        # Unqualified protected validation must not stall development training.
        # It also must never become an implicit training fallback.
        'source_closure_resolved': all(c['disposition'] != 'pending'
            for s, c in receipts.items() if SOURCE_POLICY[s][1] == 'development'),
    }
    return {'checks': checks, 'source_ready': all(checks.values()),
            'training_source_names': sorted(training),
            'model_trained': False, 'model_benefit_established': False}


def build_protocol(receipts, *, outer_folds_sha256, transform_sha256,
                   inner_folds_sha256, code_sha256, training_examples):
    """Bind source certificates and five fold identities before any real fit.

    Authentication of the certificate hashes against actual local artifacts is
    the runner's responsibility; hashes alone are not independent attestation.
    """
    state = readiness(receipts)
    require(state['source_ready'])
    require(is_hash(outer_folds_sha256))
    for values in (transform_sha256, inner_folds_sha256):
        require(type(values) in (list, tuple) and len(values) == 5)
        require(all(is_hash(v) for v in values))
    require(type(code_sha256) is dict and bool(code_sha256))
    require(all(type(k) is str and re.fullmatch(r'[A-Za-z0-9_]+\.py', k)
                and is_hash(v) for k, v in code_sha256.items()))
    admitted = state['training_source_names']
    require(type(training_examples) is dict and set(training_examples) == set(admitted))
    require(all(type(n) is int and n > 0 for n in training_examples.values()))
    # Paired participants belong to B/C, not the unpaired A exposure budget.
    unpaired = [s for s in admitted if SOURCE_POLICY[s][0] != 'paired']
    steps = math.ceil(sum(training_examples[s] for s in unpaired) / PARAMETERS['stage_a_batch'])
    require(steps > 0)
    result = {'schema': 'bran-multisource-protocol-v2',
        'status': 'specification_requires_local_authentication',
        'parameters': json.loads(json.dumps(PARAMETERS)), 'arms': list(ARMS),
        'sources': json.loads(json.dumps(receipts)),
        'source_family': {s: SOURCE_POLICY[s][2] for s in admitted},
        'outer_folds_sha256': outer_folds_sha256,
        'inner_folds_sha256': list(inner_folds_sha256),
        'transform_sha256': list(transform_sha256), 'code_sha256': dict(code_sha256),
        'stage_a_steps_per_arm_fold': steps,
        'goal_achieved': False, 'scientific_results_present': False}
    return result


def validate_terminal_inventory(success_exists, failure_exists):
    require(type(success_exists) is bool and type(failure_exists) is bool)
    require(success_exists != failure_exists)


def validate_exposure_claim(*, source_local_people, global_unique_people,
                            cross_source_linkage_authenticated):
    """Do not permit an additive global patient headline without linkage."""
    require(type(cross_source_linkage_authenticated) is bool)
    require(type(source_local_people) is dict and all(s in SOURCE_POLICY for s in source_local_people))
    require(all(type(n) is int and n >= 0 for n in source_local_people.values()))
    if global_unique_people is not None:
        require(cross_source_linkage_authenticated)
        require(type(global_unique_people) is int and bool(source_local_people))
        require(max(source_local_people.values()) <= global_unique_people <= sum(source_local_people.values()))

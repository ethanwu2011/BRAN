"""Training-only, no-update, FD-quiet source diagnostic. No patient output."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from run_bran_multisource_retinal_features_v2 import LOCK, quiet, sha, write_json
import run_bran_multisource_fit_v3 as fit
from bran_multisource_binding_v3 import load_bound_sources, bind_fold
from bran_multisource_batches_v2 import RetinalPoolV2, tensor, typed_age
from bran_multisource_clinical_v2 import ClinicalPoolV2
from bran_multisource_training_v2 import MaterializedBatch
from bran_multisource_age_v2 import AgeBatch
from bran_joint_lab_task_contract_v1 import project_joint_labs_to_registry
from bran_multisource_protocol_v2 import PARAMETERS as V2

ROOT = Path(__file__).resolve().parent
SOURCES = ('brset', 'eicu', 'mimiciii', 'mimiciv', 'nhanes_exposed', 'nwicu')
CONTEXTS = tuple(V2['completion_patterns'])
TASKS = ('paired_screening', 'paired_cbc', 'paired_hb', 'paired_low_hb',
         'source_generative', 'source_cbc')
WEIGHTS = {'paired_screening': 1., 'paired_cbc': .5, 'paired_hb': 1.,
           'paired_low_hb': 1., 'source_generative': .1, 'source_cbc': .05}
GROUPS = ('encoder', 'native_heads')
HB_CELLS = ('paired_hb', 'paired_low_hb', 'source_hb', 'source_low_hb')
CODE = tuple(sorted(set(fit.CODE) | {
    'BRAN_SOURCE_DIAGNOSTIC_V1_PLAN.md',
    'run_bran_multisource_source_diagnostic_v1.py',
    'bran_multisource_source_diagnostic_v1.py', 'bran_gradient_geometry_v1.py',
    'test_bran_multisource_source_diagnostic_v1.py',
    'test_run_bran_multisource_source_diagnostic_v1.py'}))
PARAMETERS = {'folds': 5, 'repeats': 4, 'paired_batch': 512, 'source_batch': 128,
    'seed_base': 96101, 'first_step': 330, 'repeat_step_stride': 54,
    'minimum_people_per_batch': 20, 'minimum_batches_per_release': 20,
    'training_updates': 0, 'checkpoint': 'unchanged_initial_typed_age_replay',
    'single_target': 'hemoglobin', 'contexts': list(CONTEXTS),
    'source_weights_for_diagnostic': {'generative': .1, 'cbc': .05},
    'paired_weights_for_diagnostic': {'screening': 1., 'cbc': .5},
    'heldout_scoring': False, 'protected_sources_used': False,
    'automatic_candidate_training': False, 'patient_level_output_permitted': False}


def require(ok):
    if not ok:
        raise ValueError('source_diagnostic_contract_failed')


def code_hashes():
    return {name: sha(ROOT/name) for name in CODE}


def group_index(pool):
    groups, order = np.unique(pool.person_group, return_inverse=True)
    sort = np.argsort(order, kind='stable')
    edges = np.r_[0, np.cumsum(np.bincount(order))]
    return len(groups), sort, edges


def source_batch(pool, transform, registry, rng, size=128, *, grouping=None):
    """Distinct groups, one randomly selected eligible encounter/view per group."""
    count, sort, edges = group_index(pool) if grouping is None else grouping
    require(count >= size)
    chosen = rng.choice(count, size=size, replace=False)
    rows = np.asarray([sort[rng.integers(edges[g], edges[g+1])] for g in chosen])
    require(len(np.unique(pool.person_group[rows])) == size)
    c = np.zeros((size, 59), np.float32); cm = np.zeros((size, 59), bool)
    r = np.zeros((size, 1, 384), np.float32); rm = np.zeros((size, 1), bool)
    if isinstance(pool, ClinicalPoolV2):
        raw, observed = project_joint_labs_to_registry(pool.values[rows],
            pool.observed[rows], pool.observed[rows].astype(np.uint8), registry,
            np.zeros(59), np.ones(59))
        c, cm = transform.clinical(raw, observed)
    else:
        require(isinstance(pool, RetinalPoolV2))
        r[:, 0], rm[:, 0] = transform.retinal(pool.features[rows], np.ones(size, bool))
    return MaterializedBatch(tensor(c), tensor(cm, torch.bool), tensor(r),
                             tensor(rm, torch.bool), typed_age(pool, rows))


def paired_batch(sampler, rng, size=512):
    require(len(sampler.train) >= size)
    rows = rng.choice(sampler.train, size=size, replace=False)
    require(len(np.unique(rows)) == size)
    index = tensor(rows, torch.long)
    age = AgeBatch(*(getattr(sampler.age, key)[index]
                     for key in ('value', 'lower', 'upper', 'kind')))
    return MaterializedBatch(tensor(sampler.c[rows]), tensor(sampler.cm[rows], torch.bool),
        tensor(sampler.r[rows]), tensor(sampler.rm[rows], torch.bool), age,
        tensor(sampler.labels[rows]), tensor(sampler.labelmask[rows], torch.bool))


def pooled(values, *, cosine=False):
    """No exact suppressed count, raw samples, or complementary totals."""
    require(all(v is None or type(v) in (float, int) and math.isfinite(v) for v in values))
    valid = [v for v in values if v is not None]
    if len(valid) < 20:
        return {'status': 'withheld'}
    if cosine:
        require(all(-1 <= v <= 1 for v in valid))
    return {'status': 'released', 'mean': float(np.mean(valid)),
            'median': float(np.median(valid))}


def aggregate(records):
    """Fixed keys only; private input records must never be serialized."""
    require(type(records) is list and len(records) == 20)
    result = {'support': {}, 'losses': {}, 'gradient_geometry': {}, 'hb_training': {}}
    for task in TASKS:
        count = sum(record['support'].get(task, 0) >= 20 for record in records)
        result['support'][task] = ('supported_all_batches' if count == 20 else
                                   'insufficient_for_pooled_release')
        result['losses'][task] = pooled([record['geometry']['loss_values'].get(task)
            if record['support'].get(task, 0) >= 20 else None for record in records])
    for group in GROUPS:
        entry = {'norms': {}, 'source_vs_paired_cosine': {}, 'weighted_norm_ratio': {}}
        for task in TASKS:
            entry['norms'][task] = pooled([record['geometry']['groups'].get(group, {}).get('norms', {}).get(task)
                if record['support'].get(task, 0) >= 20 else None for record in records])
        for source in ('source_generative', 'source_cbc'):
            for paired in TASKS[:4]:
                pair = paired+'|'+source
                cosines, ratios = [], []
                for record in records:
                    cell = record['geometry']['groups'].get(group, {})
                    supported = min(record['support'].get(source, 0), record['support'].get(paired, 0)) >= 20
                    cos = cell.get('cosines', {}).get(pair)
                    if cos is None:
                        cos = cell.get('cosines', {}).get(source+'|'+paired)
                    cosines.append(cos if supported else None)
                    norms = cell.get('norms', {})
                    denom = norms.get(paired, 0)*WEIGHTS[paired]
                    ratios.append(norms[source]*WEIGHTS[source]/denom
                        if supported and source in norms and denom > 0 else None)
                entry['source_vs_paired_cosine'][pair] = pooled(cosines, cosine=True)
                # Hb MAE has physical units and no V3 optimizer coefficient.
                if paired in ('paired_screening', 'paired_cbc'):
                    entry['weighted_norm_ratio'][pair] = pooled(ratios)
        result['gradient_geometry'][group] = entry
    for name in HB_CELLS:
        cells = [record['hb'][name] for record in records if record['hb'][name]['count'] >= 20]
        if len(cells) < 20:
            result['hb_training'][name] = {'status': 'withheld'}
        else:
            count = sum(item['count'] for item in cells)
            result['hb_training'][name] = {'status': 'released',
                'mae_g_dl': sum(item['abs_error_sum'] for item in cells)/count,
                'bias_g_dl': sum(item['signed_error_sum'] for item in cells)/count}
    validate_cell(result)
    return result


def validate_cell(result):
    require(set(result) == {'support', 'losses', 'gradient_geometry', 'hb_training'})
    require(set(result['support']) == set(TASKS) and set(result['losses']) == set(TASKS))
    require(set(result['gradient_geometry']) == set(GROUPS) and set(result['hb_training']) == set(HB_CELLS))
    require(all(v in ('supported_all_batches', 'insufficient_for_pooled_release') for v in result['support'].values()))
    def check(value, keys, nonnegative=False, cosine=False):
        if value == {'status': 'withheld'}:
            return
        require(set(value) == {'status', *keys} and value['status'] == 'released')
        require(all(type(value[k]) in (float, int) and math.isfinite(value[k]) for k in keys))
        if nonnegative: require(all(value[k] >= 0 for k in keys))
        if cosine: require(all(-1 <= value[k] <= 1 for k in keys))
    for value in result['losses'].values(): check(value, ('mean', 'median'))
    pairs = {a+'|'+b for a in TASKS[:4] for b in TASKS[4:]}
    ratio_pairs = {a+'|'+b for a in TASKS[:2] for b in TASKS[4:]}
    for group in result['gradient_geometry'].values():
        require(set(group) == {'norms', 'source_vs_paired_cosine', 'weighted_norm_ratio'})
        require(set(group['norms']) == set(TASKS))
        require(set(group['source_vs_paired_cosine']) == pairs)
        require(set(group['weighted_norm_ratio']) == ratio_pairs)
        for value in group['norms'].values(): check(value, ('mean', 'median'), nonnegative=True)
        for value in group['source_vs_paired_cosine'].values(): check(value, ('mean', 'median'), cosine=True)
        for value in group['weighted_norm_ratio'].values(): check(value, ('mean', 'median'), nonnegative=True)
    for value in result['hb_training'].values():
        check(value, ('mae_g_dl', 'bias_g_dl'))
        if value['status'] == 'released': require(value['mae_g_dl'] >= abs(value['bias_g_dl'])-1e-9)
    json.dumps(result, allow_nan=False)


def progress(out, phase, fold=None):
    require(phase in ('source_authentication', 'gradient_diagnostic', 'aggregate_authentication', 'completed'))
    require(fold is None or type(fold) is int and fold in range(5))
    temporary = out/'progress.next.json'
    write_json(temporary, {'phase': phase, 'fold': fold, 'pid': os.getpid(),
        'patient_level_output_emitted': False, 'training_updates': 0})
    os.replace(temporary, out/'progress.json')


def execute(out):
    from bran_multisource_source_diagnostic_v1 import diagnostic
    torch.set_num_threads(2)
    start = time.monotonic()
    progress(out, 'source_authentication')
    # Completed fit authentication also pins source loading and warm-start code.
    old = json.loads((ROOT/'BRAN_MULTISOURCE_FIT_V3_ATTEMPT1/protocol.json').read_text())
    require(old['code_sha256'] == fit.code_hashes())
    sources = load_bound_sources()
    require(old['source_binding'] == sources.receipt())
    pools = {pool.source: pool for pool in sources.pools}
    require(set(pools) == set(SOURCES))
    groupings = {name: group_index(pool) for name, pool in pools.items()}
    code = code_hashes()
    protocol = {'schema': 'bran-source-diagnostic-protocol-v1', 'parameters': PARAMETERS,
        'code_sha256': code, 'source_binding': sources.receipt(),
        'V3_fit_protocol_sha256': sha(ROOT/'BRAN_MULTISOURCE_FIT_V3_ATTEMPT1/protocol.json'),
        'V3_prior_results_inspected': True, 'adaptive_development': True,
        'minimum_release_support': '20 distinct supported people per batch and 20 batches',
        'gradient_diagnostic_is_not_causal_attribution': True, 'candidate_promoted': False}
    write_json(out/'protocol.json', protocol)
    pin = sha(out/'protocol.json')
    records = {(source, context): [] for source in SOURCES for context in CONTEXTS}
    for fold in range(5):
        progress(out, 'gradient_diagnostic', fold)
        bound = bind_fold(sources, fold)
        model = bound.model
        for parameter in model.parameters(): parameter.requires_grad_(True)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        factory = bound.paired_factory()
        source_rngs = {name: np.random.default_rng(96101+fold+1000*(j+1)) for j, name in enumerate(SOURCES)}
        paired_rng = np.random.default_rng(96101+fold)
        hb = sources.paired.names.index('hemoglobin')
        for repeat in range(4):
            for pattern, context in enumerate(CONTEXTS):
                paired = paired_batch(factory.private_sampler, paired_rng)
                step = 330+54*repeat+pattern
                for name in SOURCES:
                    batch = source_batch(pools[name], bound.transform, sources.paired.names,
                        source_rngs[name], grouping=groupings[name])
                    private = diagnostic(model, paired, batch, step=step, seed=96101+fold,
                        age_mean=bound.transform.age_mean, age_scale=bound.transform.age_scale,
                        positive_weight=factory.positive_weights,
                        hb_median=float(bound.transform.clinical_median[hb]),
                        hb_iqr=float(bound.transform.clinical_iqr[hb]))
                    records[(name, context)].append(private)
        require(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
        require(torch.equal(torch.get_rng_state(), rng_before))
        require(not model.training and all(parameter.grad is None for parameter in model.parameters()))
    progress(out, 'aggregate_authentication')
    summary = {name: {context: aggregate(records[(name, context)]) for context in CONTEXTS} for name in SOURCES}
    require(code_hashes() == code and sha(out/'protocol.json') == pin)
    require(load_bound_sources().receipt() == protocol['source_binding'])
    aggregate_result = {'schema': 'bran-source-diagnostic-aggregate-v1', 'status': 'completed',
        'protocol_sha256': pin, 'summary': summary, 'five_fold_initial_models_unchanged': True,
        'training_only': True, 'training_updates': 0, 'diagnostic_batches_per_source_context': 20,
        'batch_repeats_are_not_unique_people': True, 'gradient_arrays_or_patient_output_emitted': False,
        'protected_sources_used': False, 'heldout_scoring_performed': False,
        'candidate_promoted': False, 'causal_attribution_established': False}
    write_json(out/'aggregate.json', aggregate_result)
    write_json(out/'manifest.json', {'protocol_sha256': pin, 'aggregate_sha256': sha(out/'aggregate.json'),
        'code_sha256': code, 'elapsed_seconds': time.monotonic()-start,
        'patient_level_output_emitted': False})
    # Exclusive terminal receipt is authoritative, not the presence of a partial
    # aggregate after an interrupted manifest write.
    require(not (out/'failure.json').exists())
    reread = json.loads((out/'aggregate.json').read_text())
    require(reread == aggregate_result)
    for source in SOURCES:
        for context in CONTEXTS:
            validate_cell(reread['summary'][source][context])
    progress(out, 'completed')
    write_json(out/'completed.json', {'status': 'authenticated',
        'protocol_sha256': pin, 'aggregate_sha256': sha(out/'aggregate.json'),
        'manifest_sha256': sha(out/'manifest.json'),
        'patient_level_output_emitted': False, 'training_updates': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    require(1 <= args.attempt <= 99)
    out = ROOT/f'BRAN_SOURCE_DIAGNOSTIC_V1_ATTEMPT{args.attempt}'
    owned, ok, lock_busy = False, False, False
    with quiet():
        try:
            with LOCK.open('a+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_busy = True
                    raise
                out.mkdir(exist_ok=False); owned = True
                execute(out); ok = True
        except Exception:
            if owned and not (out/'completed.json').exists():
                phase = 'source_authentication'
                if (out/'progress.json').exists():
                    phase = json.loads((out/'progress.json').read_text())['phase']
                write_json(out/'failure.json', {'status': 'technical_failure', 'phase': phase,
                    'patient_level_output_emitted': False, 'training_updates': 0})
    print(json.dumps({'status': 'completed' if ok else 'lock_busy' if lock_busy else 'failed',
        'attempt': args.attempt, 'patient_level_output_emitted': False}))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__': main()

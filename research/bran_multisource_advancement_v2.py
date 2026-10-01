"""Map each V2 candidate into immutable historical advancement roles.

No source/checkpoint I/O or fitting. Caller authenticates historical references,
their replay, V2 checkpoints, patient alignment and all target-erased readouts.
Historical `student` means the candidate being tested, never the other V2 arm.
The primary gates use their original training-quantile tail definition; the
separately reported Hb <12 g/dL research stratum must not replace that gate.
"""
from dataclasses import dataclass
import numpy as np

import bran_multisource_mask_contract_v1 as contract
import run_bran_native_rehearsal_v1 as old
from bran_multisource_outcome_metrics_v2 import validate_counts

ARMS = ('mlp', 'token')
ROUTES = ('both', 'clinical', 'retinal')
SCREENS = tuple(version+'_'+route for version in ('initial','continued',*ARMS) for route in ROUTES) + (
    'raw_clinical','raw_retinal','raw_concat','late_average')
COMPLETIONS = ('initial','continued','mlp','token','raw_clinical','raw_concat')
ROLE_DESCRIPTION = {
    'initial':'authenticated original held-out native checkpoint',
    'continued':'authenticated historical multisource-mask continued-control checkpoint',
    'student':'the explicitly named V2 candidate arm under evaluation',
    'raw_controls':'unchanged historical fitted-readout recipe and replay population',
    'comparison':'historical advancement test, not the new information-matched FM benchmark',
    'low_hemoglobin_gate':'historical outer-training quantile tail, not a redefined clinical threshold',
    'official_figures_exclude':'late_average',
}


def require(ok):
    if not ok: raise ValueError('multisource_advancement_roles_failed')


@dataclass(frozen=True, repr=False)
class RoleViews:
    screening: dict
    completion: dict
    missingness: dict


def role_views(arm, screening, completion, missingness):
    require(arm in ARMS and type(screening) is dict and len(screening) == 26)
    require(all(type(value) is dict and set(value) == set(SCREENS) for value in screening.values()))
    require(type(completion) is dict and set(completion) == set(contract.EVALPATTERNS)
            and all(type(value) is dict and set(value) == set(COMPLETIONS) for value in completion.values()))
    require(type(missingness) is dict and set(missingness) == {'initial','continued','mlp','token'})
    sp = {}
    for endpoint, values in screening.items():
        sp[endpoint] = {key:values[arm+'_'+key[len('student_'):]] if key.startswith('student_') else values[key]
                        for key in old.metrics.S_ARMS}
    cp = {pattern:{key:values[arm] if key == 'student' else values[key] for key in old.metrics.C_ARMS}
          for pattern, values in completion.items()}
    mp = {key:missingness[arm] if key == 'student' else missingness[key] for key in old.metrics.VERSIONS}
    return RoleViews(sp, cp, mp)


def evaluate(arm, screening, completion, missingness, target, observed, groups,
             labels, labelmask, folds, names, counts):
    """Return supported aggregate metrics plus unchanged gate decisions.

`missingness` contains already-validated aggregate stress summaries, never state
vectors. `observed`/`groups` are private pattern-specific target masks and tails.
Caller must obtain them before seeing candidate performance.
"""
    validate_counts(counts,folds)
    require(tuple(screening) == tuple(names) and len(names) == len(set(names)) == 26
            and set(labels) == set(labelmask) == set(names))
    require(set(observed) == set(groups) == set(contract.EVALPATTERNS))
    require(target.shape == (len(folds),9))
    view = role_views(arm,screening,completion,missingness)
    for version in old.metrics.VERSIONS: old.stress_metrics.validate_result(view.missingness[version],names)
    # The fixed historical initial-route support defines the comparison, not
    # whichever subset makes the new candidate look better.
    masks = old.screen_masks(view.screening,labelmask,names,labels,folds)
    screen = old.metrics.screening(view.screening,labels,masks,folds,names,counts)
    summaries = {}
    for pattern in contract.EVALPATTERNS:
        obs = observed[pattern]
        require(obs.shape == target.shape and obs.dtype == bool and np.isfinite(target[obs]).all())
        require(set(groups[pattern]) == set(old.metrics.GROUPS)
                and all(value.shape == target.shape and value.dtype == bool for value in groups[pattern].values()))
        require(all(value.shape == target.shape and np.isfinite(value[obs]).all()
                    for value in view.completion[pattern].values()))
        summaries[pattern] = old.metrics.completion(target,obs,view.completion[pattern],
                                    old.safe_tail_groups(obs,groups[pattern]),counts)
    paired = {pattern:summaries[pattern] for pattern in contract.PATTERNS}
    no_retina = {pattern:summaries[pattern.replace('_hidden','_no_retina')] for pattern in contract.PATTERNS}
    decisions = contract.decisions(screen,paired,view.missingness,no_retina)
    for item in summaries.values():
        old.metrics.validate(screen,item,old.metrics.decisions(screen,item),names)
    return {'schema':'bran-multisource-advancement-v2','candidate_arm':arm,
            'screening':screen,'completion':paired,'completion_no_retina':no_retina,
            'missingness':view.missingness,'decisions':decisions,'role_description':dict(ROLE_DESCRIPTION),
            'historical_gate_definitions_changed':False,'candidate_promoted':False,
            'scientific_goal_achieved':False,'patient_level_output_emitted':False}

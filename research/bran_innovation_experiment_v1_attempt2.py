"""Technical retry: select the frozen eligible outcome subset at the loader boundary.

Attempt1 stopped at input-contract validation before any encoder fit. Its source,
protocol and failure are immutable. No scientific parameter or metric changes.
"""
import json
from pathlib import Path
import sys
import bran_innovation_experiment_v1 as base

FAILURE_NAME='validation_results/BRAN_INNOVATION_READOUT_V1/FAILURE.json'
FAILURE_SHA='fbec15ba523f1319389b820d8776e95ca4c535f417043f5c304fd804ea37b664'
ATTEMPT1_PROTOCOL_SHA='d664d0d20bde794a9cd02e238f3d80211b7f980e41a4b3e7de75c2f95ccc1537'
_evaluate=base.evaluate
_validate_protocol=base.validate_protocol

def eligible_evaluate(c,cm,elig,r,rm,names,ages,outer,inners,labels,observed,sources,**kwargs):
    """The canonical loader intentionally retains unsupported source outcomes too."""
    sources=tuple(sources)
    base.require(len(sources)==len(set(sources))==26 and set(sources)<=set(labels) and set(sources)<=set(observed))
    selected_labels={s:labels[s] for s in sources};selected_observed={s:observed[s] for s in sources}
    return _evaluate(c,cm,elig,r,rm,names,ages,outer,inners,selected_labels,selected_observed,sources,**kwargs)

def validate_protocol(root=base.ROOT):
    root=Path(root)
    base.require(base.sha(root/'BRAN_INNOVATION_READOUT_PROTOCOL_V1.json')==ATTEMPT1_PROTOCOL_SHA)
    base.require(base.sha(root/FAILURE_NAME)==FAILURE_SHA)
    base.require(not(root/'validation_results/BRAN_INNOVATION_READOUT_V1/SUCCESS.json').exists())
    return _validate_protocol(root)

# Deliberately isolated entry-point adaptation, matching existing versioned runners.
base.PROTOCOL_NAME='BRAN_INNOVATION_READOUT_PROTOCOL_V1_ATTEMPT2.json'
base.SCHEMA='bran-innovation-experiment-v1-attempt2'
base.OUTDIR='validation_results/BRAN_INNOVATION_READOUT_V1_ATTEMPT2'
base.PATHS={k:base.OUTDIR+'/'+Path(v).name for k,v in base.PATHS.items()}
base.NEW_FILES=base.NEW_FILES|{'bran_innovation_experiment_v1_attempt2.py','test_bran_innovation_experiment_v1_attempt2.py','BRAN_INNOVATION_READOUT_PROTOCOL_V1.json'}
base.evaluate=eligible_evaluate
base.validate_protocol=validate_protocol

if __name__=='__main__':
    try:print(json.dumps({'freeze':base.freeze,'run':base.run,'audit':base.audit}[sys.argv[1]](),sort_keys=True,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status':'blocked_without_disclosure','error_class':type(exc).__name__,'contents_emitted':False}));raise SystemExit(1)

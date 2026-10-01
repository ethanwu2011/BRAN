"""No-fit clinical bridge for the newly authenticated retinal production."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import sha,exclusive_json
import run_bran_raw_teacher_distillation_v1 as origin
import run_bran_retinal_extraction_v1 as extraction
import bran_authenticated_retinal_input_v1 as inputs
import bran_retinal_input_bridge_v1 as kernel
import bran_retinal_input_bridge_metrics_v1 as metrics
from bran_clinical_semantics_v1 import CBC_FIELDS

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/"BRAN_RETINAL_INPUT_BRIDGE_PROTOCOL_V1.json"
OUT=ROOT/"BRAN_RETINAL_INPUT_BRIDGE_V1"
AUDIT=ROOT/"BRAN_RETINAL_INPUT_BRIDGE_AUDIT_V1"
EXTRACTION_PIN="e82f307d81f8f8ff3cce84ec130d1eab0512d0dc4cc9a217d5565db77fb8f3e1"
PARAMETERS={"training_steps":0,"state_width":192,"torch_threads":2,"inference_batch":256,
    "normalizers":"unchanged historical outer-training; never fit to new retinal input",
    "retinal_pooling":"new authenticated float64 arithmetic mean; original historical pools unchanged",
    "screening":"all26 endpoints; all six routes common nonabstaining support; fold-weighted AUROC",
    "cbc":"same retained native head, all9 values and indicators hidden, original units",
    "bootstrap_draws":1000,"bootstrap_seed":94801,"minimum_valid_draws":900,"minimum_release_support":20,
    "bootstrap_scope":"paired patients within original outer folds; fixed checkpoints, no refits",
    "historical_screen_replay_atol":1e-10,"historical_cbc_replay_atol":1e-6,"historical_cbc_replay_rtol":1e-6,
    "equivalence":{"rtol":1e-4,"atol":1e-5},
    "decision":"no automatic replacement; review eligibility only if pooled inputs, screening probabilities and wholeCBC predictions all numerically equivalent"}
FLAGS={"patient_level_output_emitted":False,"official_test_used":False,"adaptive_development":True,
    "clinical_benefit_claim":False,"model_promoted":False,"retinal_input_replaced":False,
    "training_performed":False,"new_subtype_claim":False,"historical_reference_replay_passed":True,
    "unchanged_clinical_route":True,"normalizers_and_checkpoints_unchanged":True}
FILES=("run_bran_retinal_input_bridge_v1.py","test_run_bran_retinal_input_bridge_v1.py",
    "bran_retinal_input_bridge_v1.py","test_bran_retinal_input_bridge_v1.py",
    "bran_retinal_input_bridge_metrics_v1.py","test_bran_retinal_input_bridge_metrics_v1.py",
    "bran_authenticated_retinal_input_v1.py","test_bran_authenticated_retinal_input_v1.py",
    "bran_native_cbc_decoders_v1.py","run_bran_missingness_stress_v1.py",
    "BRAN_RETINAL_INPUT_BRIDGE_DESIGN_V1.md")
require=kernel.require


def prepare(audit_pin):
    require(inputs._is_digest(audit_pin) and sha(extraction.PROTOCOL)==EXTRACTION_PIN)
    require(not (extraction.OUT/"failure.json").exists() and not (extraction.AUDIT/"failure.json").exists())
    ep=json.loads(extraction.PROTOCOL.read_text());extraction.validate_protocol(ep)
    ea=json.loads((extraction.OUT/"aggregate.json").read_text());extraction.validate_result(ea,ep,EXTRACTION_PIN)
    require(sha(extraction.AUDIT/"audit.json")==audit_pin)
    audit=json.loads((extraction.AUDIT/"audit.json").read_text());inputs._validate_audit(audit,ep,EXTRACTION_PIN)
    require(audit["aggregate_sha256"]==sha(extraction.OUT/"aggregate.json")
            and audit["output_sha256"]==ea["output_sha256"]==sha(extraction.PRIVATE/"features.npy")
            and audit["inventory_file_sha256"]==ea["inventory_file_sha256"]==sha(extraction.PRIVATE/"inventory.json"))
    base=origin.prepare()
    return {"schema":"bran-retinal-input-bridge-protocol-v1","status":"frozen_before_execution",
        "parameters":PARAMETERS,"native_source":base["native_source"],
        "native_aggregate_sha256":base["native_aggregate_sha256"],"native_audit_sha256":base["native_audit_sha256"],
        "retinal_protocol_sha256":EXTRACTION_PIN,"retinal_audit_sha256":audit_pin,
        "retinal_aggregate_sha256":audit["aggregate_sha256"],"retinal_output_sha256":audit["output_sha256"],
        "retinal_selection":audit["selection"],"source_policy_sha256":ep["external_sha256"]["source_policy"],
        "code_sha256":{**base["code_sha256"],**ep["code_sha256"],**{name:sha(ROOT/name) for name in FILES}},
        "runtime":base["runtime"]}


def validate_protocol(p):require(p==prepare(p["retinal_audit_sha256"]))


def validate_result(a,p):
    require(type(a)is dict and set(a)=={"schema","status","results","equivalence","fold_authentication","retinal_selection"}|set(FLAGS))
    require(a["schema"]=="bran-retinal-input-bridge-aggregate-v1" and a["status"]=="completed")
    require(all(a[k] is v for k,v in FLAGS.items()))
    require(a["fold_authentication"]==p["native_source"]["source"]["authentication"]
            and a["retinal_selection"]==p["retinal_selection"])
    metrics.validate(a["results"],p["native_source"]["source"]["endpoint_names"])
    e=a["equivalence"]
    require(type(e)is dict and set(e)=={"pooled_input","screening","whole_cbc","replacement_review_eligible"}
            and all(type(v)is bool for v in e.values())
            and e["replacement_review_eligible"]==all(e[k] for k in ("pooled_input","screening","whole_cbc")))


def compute(p,notify=None):
    import torch
    torch.set_num_threads(2);source=origin.native.source
    ctx,folds,c0,cm0,eligible,r0,rm,ages,names=source.io.load_context()
    ids=list(map(str,ctx["raw_cohort"].patient_ids));require(len(ids)==1928)
    rnew,rmnew=inputs.load_pooled_features(protocol_pin=EXTRACTION_PIN,audit_pin=p["retinal_audit_sha256"],
        patient_ids=ids,folds=folds,source_policy_sha256=p["source_policy_sha256"])
    require(np.array_equal(rm,rmnew))
    endpoints=p["native_source"]["source"]["endpoint_names"]
    screen={arm:np.full((len(folds),26),np.nan) for arm in metrics.ARMS}
    blood={v:np.full((len(folds),9),np.nan) for v in kernel.VERSIONS}
    support=np.zeros((len(folds),9),bool)
    slots=tuple(names.index(f) for f in CBC_FIELDS)
    for f in range(5):
        if notify:notify("fixed_checkpoint_bridge",f)
        fit=np.flatnonzero(folds!=f);test=np.flatnonzero(folds==f)
        _,identity=source.base._inner_context(ctx,fit,f)
        require(identity==p["native_source"]["source"]["authentication"]["inner_fold_sha256"][f])
        transform=source.base.FoldTransform(c0,cm0,eligible,r0,rm,ages,fit)
        model=origin.load_initial(f,transform,p)
        screens,cbc,available=kernel.infer(model,transform,c0[test],cm0[test],eligible[test],r0[test],rnew[test],
            rm[test],rmnew[test],ages[test],names,batch_size=256)
        for v in kernel.VERSIONS:
            for route in kernel.ROUTES:screen[v+"_"+route][test]=screens[v][route]
            blood[v][test]=cbc[v]
        support[test]=available
    if notify:notify("reference_replay")
    labels=np.column_stack([ctx["labels_by_source"][name] for name in endpoints])
    observed=np.column_stack([ctx["observed_by_source"][name] for name in endpoints]).astype(bool)
    baseline=json.loads((origin.native.OUT/"aggregate.json").read_text())["results"]["endpoints"]
    common=np.logical_and.reduce([np.isfinite(screen["historical_"+r]).all(1) for r in kernel.ROUTES])
    for j,name in enumerate(endpoints):
        for route in kernel.ROUTES:
            auc=source.base.fold_weighted_auc(labels[:,j],screen["historical_"+route][:,j],observed[:,j]&common,folds)
            require(abs(auc-baseline[name]["arms"]["native_"+route]["auroc"])<=PARAMETERS["historical_screen_replay_atol"])
    bobs=(cm0&eligible)[:,slots]&support;truth=c0[:,slots]
    reference=json.loads((source.OUT/"aggregate.json").read_text())["retained_cbc_head_whole_panel"]
    for j,field in enumerate(CBC_FIELDS):
        error=blood["historical"][bobs[:,j],j]-truth[bobs[:,j],j]
        for key,value in (("mae",np.abs(error).mean()),("mse",np.square(error).mean())):
            require(np.isclose(value,reference[field][key],atol=PARAMETERS["historical_cbc_replay_atol"],rtol=PARAMETERS["historical_cbc_replay_rtol"]))
    if notify:notify("paired_aggregate")
    counts=source.ev.paired_counts(folds,draws=1000,seed=94801)
    results=metrics.summarize(screen,labels,observed,folds,endpoints,counts,truth,blood,bobs)
    e={"pooled_input":bool(np.allclose(r0[rm],rnew[rm],**PARAMETERS["equivalence"])),
       "screening":bool(all(np.allclose(screen["historical_"+r],screen["authenticated_"+r],equal_nan=True,**PARAMETERS["equivalence"]) for r in kernel.ROUTES)),
       "whole_cbc":bool(np.allclose(blood["historical"],blood["authenticated"],equal_nan=True,**PARAMETERS["equivalence"]))}
    e["replacement_review_eligible"]=all(e.values())
    result={"schema":"bran-retinal-input-bridge-aggregate-v1","status":"completed","results":results,"equivalence":e,
        "fold_authentication":p["native_source"]["source"]["authentication"],"retinal_selection":p["retinal_selection"],**FLAGS}
    validate_result(result,p);return result


def audit(p,pin):
    require(not (OUT/"failure.json").exists())
    a=json.loads((OUT/"aggregate.json").read_text());validate_result(a,p)
    m=json.loads((OUT/"manifest.json").read_text())
    require(type(m)is dict and set(m)=={"protocol_sha256","aggregate_sha256","elapsed_seconds","patient_level_output_emitted"}
            and m["protocol_sha256"]==pin and m["aggregate_sha256"]==sha(OUT/"aggregate.json")
            and m["patient_level_output_emitted"] is False
            and metrics.finite(m["elapsed_seconds"]) and m["elapsed_seconds"]>=0)
    require(compute(p)==a);validate_protocol(p)
    return {"schema":"bran-retinal-input-bridge-audit-v1","status":"authenticated","protocol_sha256":pin,
        "aggregate_sha256":sha(OUT/"aggregate.json"),"manifest_sha256":sha(OUT/"manifest.json"),
        "retinal_audit_sha256":p["retinal_audit_sha256"],"fold_authentication":a["fold_authentication"],
        "all_aggregates_replayed":True,"patient_level_output_emitted":False,"model_promoted":False}


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    for op in ("prepare","run","audit"):g.add_argument("--"+op,action="store_true")
    parser.add_argument("--protocol-sha256");parser.add_argument("--retinal-audit-sha256");args=parser.parse_args()
    ok,owned,phase,start=False,False,"protocol",time.monotonic();dest=AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists())
                exclusive_json(PROTOCOL,prepare(args.retinal_audit_sha256))
            else:
                require(args.protocol_sha256 and sha(PROTOCOL)==args.protocol_sha256)
                p=json.loads(PROTOCOL.read_text());validate_protocol(p);dest.mkdir();owned=True
                with open("/private/tmp/bran_retinal_input_bridge_v1.lock","a") as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if args.run:
                        phase="bridge_comparison"
                        result=compute(p,lambda phase,fold=None:origin.native.source.base._atomic_progress(OUT/"progress.json",phase,fold))
                        validate_protocol(p);staged=OUT/"aggregate.staged.json";exclusive_json(staged,result)
                        exclusive_json(OUT/"manifest.json",{"protocol_sha256":args.protocol_sha256,"aggregate_sha256":sha(staged),
                            "elapsed_seconds":round(time.monotonic()-start,1),"patient_level_output_emitted":False})
                        origin.native.source.base._atomic_completed(OUT/"progress.json")
                        os.link(staged,OUT/"aggregate.json")
                    else:
                        phase="audit";exclusive_json(AUDIT/"audit.json",audit(p,args.protocol_sha256))
            ok=True
        except Exception as error:
            if owned:
                frames=[{"file":Path(x.filename).name,"line":x.lineno} for x in traceback.extract_tb(error.__traceback__)
                    if Path(x.filename).parent==ROOT and Path(x.filename).name in FILES]
                exclusive_json(dest/"failure.json",{"status":"execution_failed","phase":phase,
                    "error_class":type(error).__name__ if type(error) in (ValueError,TypeError,KeyError,RuntimeError,OSError) else "other",
                    "code_frames":frames,"patient_level_output_emitted":False})
    print(json.dumps({"operation":"prepare" if args.prepare else "audit" if args.audit else "run",
        "status":"completed" if ok else "execution_failed","patient_level_output_emitted":False}))
    return 0 if ok else 1


if __name__=="__main__":raise SystemExit(main())

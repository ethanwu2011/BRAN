"""No-fit native-state CBC decoder comparison and honest local calibration."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np

from bran_clinical_dictionary_binding_v1 import _quiet
from run_bran_source_linkage_audit_v1 import exclusive_json, sha
from bran_clinical_semantics_v1 import CBC_FIELDS, CANONICAL_UNITS
import run_bran_raw_teacher_distillation_v1 as origin
import bran_native_calibration_split_v1 as splitting
import bran_native_cbc_decoders_v1 as decoding
import bran_native_cbc_calibration_metrics_v1 as metrics
from run_bran_missingness_stress_v1 import parameter_digest
from bran_missingness_stress_metrics_v1 import safe_coverage

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "BRAN_NATIVE_CBC_CALIBRATION_PROTOCOL_V1.json"
OUT = ROOT / "BRAN_NATIVE_CBC_CALIBRATION_V1"
AUDIT = ROOT / "BRAN_NATIVE_CBC_CALIBRATION_AUDIT_V1"
PRIVATE = ROOT / "private_artifacts/bran_native_cbc_calibration_v1"
PARAMETERS = {"state_width":192, "patterns":list(decoding.PATTERNS), "routes":list(decoding.ROUTES),
    "heads":list(decoding.HEADS), "units":dict(CANONICAL_UNITS), "redcell_fields":list(decoding.REDCELL),
    "input":"unchanged authenticated historical retinal features; no new extraction input swap",
    "primary":"native retained CBC head; generative continuous decoder secondary, no posthoc routing",
    "calibration_split_salt":splitting.SALT, "calibration_fraction":.5,
    "calibration_selection":"lowest floor(n/2) ID hashes WITHIN own heldout fold; others score",
    "alpha":.1, "radius":"ceil((n+1)*0.9)-th absolute residual, min20; no clipping",
    "minimum_support":20, "torch_threads":2, "inference_batch":256, "training_steps":0,
    "bootstrap_draws":1000, "bootstrap_seed":94701, "minimum_valid_draws":900,
    "bootstrap_scope":"scoring halves only, within outer fold; fixed fitted models and calibration radii",
    "tail_quantiles":[.1,.9], "tails":"original-unit outer-training thresholds; suppress all tails if any n<20",
    "calibration_unsupported":"field/fold interval abstains; report common calibrated scoring support",
    "screening_replay_tolerance":1e-10, "cbc_replay_atol":1e-6, "cbc_replay_rtol":1e-6,
    "cbc_replay_scope":"native all9 wholeCBC heldout MAE/MSE; tolerance accommodates batched forward ordering",
    "head_advancement":"no automatic selection; require paired MAE CI upper<0 for all3 priority single targets and no worse wholeCBC points before any further study"}
FLAGS = {"patient_level_output_emitted":False, "clinical_use":False, "model_promoted":False,
    "training_performed":False, "retinal_input_replaced":False, "official_test_used":False,
    "adaptive_development":True, "unconditional_coverage_guarantee":False, "new_subtype_claim":False,
    "baseline_replay_passed":True, "parameters_unchanged":True}
FILES = ("run_bran_native_cbc_calibration_v1.py", "test_run_bran_native_cbc_calibration_v1.py",
    "bran_native_cbc_decoders_v1.py", "test_bran_native_cbc_decoders_v1.py",
    "bran_native_cbc_calibration_metrics_v1.py", "test_bran_native_cbc_calibration_metrics_v1.py",
    "bran_native_calibration_split_v1.py", "test_bran_native_calibration_split_v1.py",
    "run_bran_missingness_stress_v1.py", "bran_missingness_stress_metrics_v1.py",
    "BRAN_NATIVE_CBC_CALIBRATION_DESIGN_V1.md")


def require(ok):
    if not ok:
        raise ValueError("native_cbc_calibration_contract_failed")


def digest_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",",":"), allow_nan=False).encode()).hexdigest()


def prepare():
    base = origin.prepare()
    return {"schema":"bran-native-cbc-calibration-protocol-v1", "status":"frozen_before_execution",
        "parameters":PARAMETERS, "native_source":base["native_source"],
        "native_aggregate_sha256":base["native_aggregate_sha256"], "native_audit_sha256":base["native_audit_sha256"],
        "code_sha256":{**base["code_sha256"], **{name:sha(ROOT/name) for name in FILES}}, "runtime":base["runtime"]}


def validate_protocol(p):
    require(p == prepare())


def check_coverage(x):
    if x == {"status":"withheld"}:
        return
    require(type(x) is dict and set(x) == {"status","supported","total"} and x["status"] == "released")
    require(type(x["supported"]) is int and type(x["total"]) is int and 0<=x["supported"]<=x["total"] and x["total"]>=20)
    require(all(n == 0 or n >=20 for n in (x["supported"],x["total"]-x["supported"])))


def decision(results):
    primary = results["single_target_hidden"]["both"]["metrics"]
    whole = results["whole_cbc_hidden"]["both"]["metrics"]
    checks = {}
    for field in ("hemoglobin","plt","wbc"):
        for label, value, interval in (("single_",primary[field]["overall"],True),("whole_",whole[field]["overall"],False)):
            supported = value["status"] == "supported"
            contrast = value.get("contrast", {}).get("generative_minus_native", {})
            checks[label+field] = bool(supported and (contrast["ci95"][1] < 0 if interval else contrast["mae_delta"] <= 0))
    return {"checks":checks, "further_decoder_study_supported":bool(all(checks.values())), "automatic_selection":False}


def validate_result(a,p):
    require(type(a) is dict and set(a) == {"schema","status","results","split_authentication","fold_authentication",
        "paired_people","state_width","decision"} | set(FLAGS))
    require(a["schema"] == "bran-native-cbc-calibration-aggregate-v1" and a["status"] == "completed")
    require(type(a["paired_people"]) is int and a["paired_people"] == 1928
            and type(a["state_width"]) is int and a["state_width"] == 192)
    require(all(a[k] is v for k,v in FLAGS.items()))
    require(a["fold_authentication"] == p["native_source"]["source"]["authentication"])
    require(type(a["split_authentication"]) is dict and set(a["split_authentication"]) == {"patient_order_sha256","roles_sha256"})
    require(all(type(x) is str and len(x)==64 and all(c in "0123456789abcdef" for c in x) for x in a["split_authentication"].values()))
    require(type(a["results"]) is dict and set(a["results"]) == set(decoding.PATTERNS))
    for routes in a["results"].values():
        require(type(routes) is dict and set(routes) == set(decoding.ROUTES))
        for cell in routes.values():
            require(type(cell) is dict and set(cell) == {"metrics","prediction_support","calibrated_support"})
            metrics.validate_result(cell["metrics"])
            for key in ("prediction_support","calibrated_support"):
                require(type(cell[key]) is dict and set(cell[key]) == set(CBC_FIELDS))
                for coverage in cell[key].values(): check_coverage(coverage)
    d = a["decision"]
    expected = decision(a["results"])
    require(type(d) is dict and set(d) == set(expected) and type(d["checks"]) is dict
            and set(d["checks"]) == set(expected["checks"])
            and all(type(x) is bool for x in d["checks"].values())
            and type(d["further_decoder_study_supported"]) is bool and d["automatic_selection"] is False
            and d == expected)


def compute(p, notify=None):
    import torch
    source = origin.native.source; torch.set_num_threads(2)
    ctx, folds, c0, cm0, eligible, r0, rm, ages, names = source.io.load_context()
    ids = list(map(str,ctx["raw_cohort"].patient_ids)); require(len(ids)==1928)
    roles = splitting.make_roles(ids,folds); score_rows = np.flatnonzero(roles == 1)
    slots = tuple(names.index(x) for x in CBC_FIELDS)
    target = c0[:,slots].copy(); observed = (cm0 & eligible)[:,slots].copy()
    shapes = (len(folds),9)
    cells = {(pattern,route):{"pred":{head:np.full(shapes,np.nan) for head in decoding.HEADS},
        "radius":{head:np.full(shapes,np.nan) for head in decoding.HEADS}, "available":np.zeros(shapes,bool)}
        for pattern in decoding.PATTERNS for route in decoding.ROUTES}
    groups = {g:np.zeros(shapes,bool) for g in ("low","middle","high")}
    radii = np.full((5,3,2,2,9),np.nan)
    screens = {route:np.full((len(folds),26),np.nan) for route in ("both","clinical","retinal")}
    normalizer_hashes = {}
    for fold in range(5):
        if notify: notify("fixed_checkpoint_inference",fold)
        fit,cal,score = splitting.partition(ids,folds,fold)
        heldout = np.flatnonzero(folds == fold)
        _, identity = source.base._inner_context(ctx,fit,fold)
        require(identity == p["native_source"]["source"]["authentication"]["inner_fold_sha256"][fold])
        transform = source.base.FoldTransform(c0,cm0,eligible,r0,rm,ages,fit)
        c,cm,r,age = transform.apply(c0,cm0,eligible,r0,rm,ages)
        model = origin.load_initial(fold,transform,p); before = parameter_digest(model)
        normalizer_hashes["fold"+str(fold)] = digest_json({k:np.asarray(getattr(transform,k)).tolist() for k in origin.NORMALIZERS})
        before_screen = origin.native.kernel.predict_native(model,c[heldout],cm[heldout],r[heldout],rm[heldout],age[heldout])
        for route,x in before_screen.items(): screens[route][heldout] = x
        for j in range(9):
            fit_values = target[fit,j][observed[fit,j]]
            require(len(fit_values)>=20 and np.isfinite(fit_values).all())
            low,high = np.quantile(fit_values,[.1,.9])
            groups["low"][heldout,j] = target[heldout,j] < low
            groups["high"][heldout,j] = target[heldout,j] > high
            groups["middle"][heldout,j] = (target[heldout,j]>=low)&(target[heldout,j]<=high)
        for pi,pattern in enumerate(decoding.PATTERNS):
            for ri,route in enumerate(decoding.ROUTES):
                prediction,support = decoding.infer(model,c[heldout],cm[heldout],r[heldout],rm[heldout],age[heldout],names,
                    transform.clinical_median,transform.clinical_iqr,pattern=pattern,route=route,batch_size=256)
                cell = cells[(pattern,route)]; cell["available"][heldout] = support
                for hi,head in enumerate(decoding.HEADS):
                    cell["pred"][head][heldout] = prediction[head]
                    radius = metrics.fit_radii(target[cal],cell["pred"][head][cal],observed[cal]&cell["available"][cal])
                    radii[fold,pi,ri,hi] = radius
                    cell["radius"][head][score] = radius
        after_screen = origin.native.kernel.predict_native(model,c[heldout],cm[heldout],r[heldout],rm[heldout],age[heldout])
        require(before == parameter_digest(model) and all(np.array_equal(before_screen[x],after_screen[x],equal_nan=True) for x in before_screen))
    if notify: notify("reference_replay")
    baseline = json.loads((origin.native.OUT/"aggregate.json").read_text())["results"]["endpoints"]
    common = np.logical_and.reduce([np.isfinite(x).all(1) for x in screens.values()])
    for j,name in enumerate(p["native_source"]["source"]["endpoint_names"]):
        for route in screens:
            auc = source.base.fold_weighted_auc(ctx["labels_by_source"][name],screens[route][:,j],ctx["observed_by_source"][name]&common,folds)
            require(abs(auc-baseline[name]["arms"]["native_"+route]["auroc"]) <= PARAMETERS["screening_replay_tolerance"])
    previous = json.loads((source.OUT/"aggregate.json").read_text())["retained_cbc_head_whole_panel"]
    cell = cells[("whole_cbc_hidden","both")]
    for j,field in enumerate(CBC_FIELDS):
        valid = observed[:,j]&cell["available"][:,j]
        error = cell["pred"]["native"][valid,j]-target[valid,j]
        for name,value in (("mae",np.abs(error).mean()),("mse",np.square(error).mean())):
            require(np.isclose(value,previous[field][name],rtol=PARAMETERS["cbc_replay_rtol"],atol=PARAMETERS["cbc_replay_atol"]))
    if notify: notify("scoring_aggregate")
    counts = source.ev.paired_counts(folds[score_rows],draws=1000,seed=94701)
    results = {pattern:{} for pattern in decoding.PATTERNS}
    for (pattern,route),cell in cells.items():
        point_support = observed[score_rows]&cell["available"][score_rows]
        calibrated = point_support.copy()
        for head in decoding.HEADS: calibrated &= np.isfinite(cell["radius"][head][score_rows])
        results[pattern][route] = {"metrics":metrics.summarize(target[score_rows],
            {h:cell["pred"][h][score_rows] for h in decoding.HEADS},
            {h:cell["radius"][h][score_rows] for h in decoding.HEADS},calibrated,
            {g:mask[score_rows]&calibrated for g,mask in groups.items()},folds[score_rows],counts),
            "prediction_support":{field:safe_coverage(point_support[:,j]) for j,field in enumerate(CBC_FIELDS)},
            "calibrated_support":{field:safe_coverage(calibrated[:,j]) for j,field in enumerate(CBC_FIELDS)}}
    split_auth = {"patient_order_sha256":digest_json(ids),"roles_sha256":digest_json(roles.tolist())}
    result = {"schema":"bran-native-cbc-calibration-aggregate-v1", "status":"completed", "results":results,
        "split_authentication":split_auth,"fold_authentication":p["native_source"]["source"]["authentication"],
        "paired_people":1928,"state_width":192,"decision":decision(results),**FLAGS}
    validate_result(result,p)
    metadata = {"protocol_sha256":sha(PROTOCOL), "split_authentication":split_auth,
        "checkpoint_sha256":p["native_source"]["checkpoint_sha256"], "normalizer_sha256":normalizer_hashes,
        "radii_order":["fold","pattern","route","head","cbc_field"],"parameters":PARAMETERS}
    return result,radii,metadata


def write_private(radii,metadata):
    require(isinstance(radii,np.ndarray) and radii.shape==(5,3,2,2,9) and radii.dtype==np.float64
            and not np.isinf(radii).any() and (radii[np.isfinite(radii)]>=0).all())
    fd = os.open(PRIVATE/"calibration.npz",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as handle:
        np.savez_compressed(handle,radii=radii,metadata_json=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),np.uint8))
        handle.flush(); os.fsync(handle.fileno())


def publish(result,pin,start):
    # The exclusive success artifact is committed LAST. Earlier errors can leave
    # staging/manifest evidence, but cannot create conflicting terminal states.
    staged=OUT/"aggregate.staged.json"
    exclusive_json(staged,result)
    exclusive_json(OUT/"manifest.json",{"protocol_sha256":pin,
        "aggregate_sha256":sha(staged),"calibration_sha256":sha(PRIVATE/"calibration.npz"),
        "elapsed_seconds":round(time.monotonic()-start,1),"patient_level_output_emitted":False})
    origin.native.source.base._atomic_completed(OUT/"progress.json")
    os.link(staged,OUT/"aggregate.json")


def audit(p,pin):
    require(not (OUT/"failure.json").exists())
    a=json.loads((OUT/"aggregate.json").read_text());validate_result(a,p)
    m=json.loads((OUT/"manifest.json").read_text())
    require(set(m)=={"protocol_sha256","aggregate_sha256","calibration_sha256","elapsed_seconds","patient_level_output_emitted"})
    require(m["protocol_sha256"]==pin and m["aggregate_sha256"]==sha(OUT/"aggregate.json") and m["patient_level_output_emitted"] is False)
    require(type(m["elapsed_seconds"]) in (int,float) and np.isfinite(m["elapsed_seconds"]) and m["elapsed_seconds"]>=0)
    path=PRIVATE/"calibration.npz";require(path.stat().st_mode&0o777==0o600 and sha(path)==m["calibration_sha256"])
    replay,radii,metadata=compute(p)
    require(replay==a)
    with np.load(path,allow_pickle=False) as b:
        require(set(b.files)=={"radii","metadata_json"} and b["radii"].shape==(5,3,2,2,9)
            and b["radii"].dtype==np.float64 and np.array_equal(b["radii"],radii,equal_nan=True))
        require(b["metadata_json"].dtype==np.uint8 and b["metadata_json"].ndim==1)
        require(json.loads(b["metadata_json"].tobytes())==metadata)
    require(sha(path)==m["calibration_sha256"]);validate_protocol(p)
    return {"schema":"bran-native-cbc-calibration-audit-v1","status":"authenticated","protocol_sha256":pin,
        "aggregate_sha256":sha(OUT/"aggregate.json"),"manifest_sha256":sha(OUT/"manifest.json"),
        "calibration_sha256":m["calibration_sha256"],"split_authentication":a["split_authentication"],
        "fold_authentication":a["fold_authentication"],"all_aggregates_and_radii_replayed":True,"patient_level_output_emitted":False}


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    for op in ("prepare","run","audit"):g.add_argument("--"+op,action="store_true")
    parser.add_argument("--protocol-sha256");args=parser.parse_args()
    ok,owned,phase,start=False,False,"protocol",time.monotonic();dest=AUDIT if args.audit else OUT
    with _quiet():
        try:
            if args.prepare:
                require(not OUT.exists() and not AUDIT.exists() and not PRIVATE.exists())
                exclusive_json(PROTOCOL,prepare())
            else:
                require(args.protocol_sha256 and sha(PROTOCOL)==args.protocol_sha256)
                p=json.loads(PROTOCOL.read_text());validate_protocol(p);dest.mkdir();owned=True
                with open("/private/tmp/bran_native_cbc_calibration_v1.lock","a") as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if args.run:
                        PRIVATE.mkdir(mode=0o700)
                        phase="calibration_comparison"
                        result,radii,metadata=compute(p,lambda phase,fold=None:origin.native.source.base._atomic_progress(OUT/"progress.json",phase,fold))
                        write_private(radii,metadata);validate_protocol(p)
                        publish(result,args.protocol_sha256,start)
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

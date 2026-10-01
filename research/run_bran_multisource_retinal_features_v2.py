"""Source-authenticated BRSET TRAIN features in the paired V3 coordinate.

Local-only, FD-quiet, shared-heavy-lock lifecycle. No labels or pixels exported.
Existing unadapted-DINO BRSET features are explicitly NOT reused. This performs
frozen inference only; a completed extraction is not a trained V2 patient model.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
LOCK=Path('/private/tmp/bran_retinal_extraction_v1.lock')
PARAMETERS={'source':'brset','source_split':0,'dimension':384,'batch_size':16,
    'resolution':224,'device':'mps','dtype':'float32','audit_images':24,
    'audit_rtol':1e-5,'audit_atol':1e-5,'model_training_steps':0,
    'feature_rule':'mean_normalized_final_layer_patch_tokens_after_five_prefix_tokens',
    'retinal_pixels_emitted':False,'patient_level_output_emitted':False}
CODE=('run_bran_multisource_retinal_features_v2.py',
    'run_bran_retinal_supervised_adaptation_v1.py','run_bran_retinal_extraction_v1.py',
    'patient_atlas_eye_contracts.py','patient_atlas_contracts.py',
    'run_bran_retinal_content_admission_v1.py','run_bran_retinal_group_readiness_v1.py')


def require(ok):
    if not ok: raise ValueError('bran_v2_retinal_feature_contract_failed')


@contextmanager
def quiet():
    sys.stdout.flush(); sys.stderr.flush()
    saved=(os.dup(1),os.dup(2)); null=os.open(os.devnull,os.O_WRONLY)
    try:
        os.dup2(null,1); os.dup2(null,2); yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0],1); os.dup2(saved[1],2)
        for fd in (*saved,null): os.close(fd)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value,private=False):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def progress(out,phase,completed_batches=0):
    value={'phase':phase,'completed_batches':completed_batches,
           'patient_level_output_emitted':False,'model_training_started':False}
    temp=out/'progress.next.json'; write_json(temp,value); os.replace(temp,out/'progress.json')


def paths(attempt):
    require(type(attempt) is int and 1<=attempt<=99)
    return (ROOT/f'BRAN_MULTISOURCE_RETINAL_FEATURES_V2_ATTEMPT{attempt}',
        ROOT/'private_artifacts'/f'bran_multisource_retinal_features_v2_attempt{attempt}')


class LocalBackend:
    def source(self):
        import numpy as np
        import run_bran_retinal_supervised_adaptation_v1 as source
        import run_bran_retinal_extraction_v1 as eye
        admission=source.a
        audit_pin=sha(admission.AUDIT/'audit.json')
        admission.authenticate_audit(source.ADMISSION_PROTOCOL_SHA256,audit_pin)
        data=source.load_source(); arrays=data['arrays']
        use=np.flatnonzero(arrays['split']==PARAMETERS['source_split'])
        require(len(use)>=24 and len(np.unique(arrays['groups'][use]))>=20)
        # Single hash of full selected membership, never per-person IDs/hashes.
        h=hashlib.sha256()
        for name in ('source_rows','groups','split','ages','raw_sha256','decoded224_sha256'):
            value=np.ascontiguousarray(arrays[name][use]); h.update(name.encode())
            h.update(str(value.dtype).encode()); h.update(value.tobytes())
        pins={'admission_protocol_sha256':source.ADMISSION_PROTOCOL_SHA256,
            'admission_audit_sha256':audit_pin,'membership_sha256':h.hexdigest(),
            'eye_checkpoint_sha256':sha(eye.EXTERNAL['checkpoint']),
            'eye_contract_sha256':sha(eye.EXTERNAL['eye_contract']),
            'images_lower_bound_20':len(use)//20*20,
            'source_local_people_lower_bound_20':len(np.unique(arrays['groups'][use]))//20*20}
        return data,use,pins

    def encoder(self):
        import torch
        import run_bran_retinal_extraction_v1 as eye
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0')
        artifact,tower=eye.load_tower()
        for parameter in tower.parameters(): parameter.requires_grad_(False)
        require(not tower.training and not any(p.requires_grad for p in tower.parameters()))
        self.artifact=artifact; self.tower=tower
        return eye.encoder(tower)

    def infer(self,encoder,data,indices):
        import numpy as np
        import run_bran_retinal_supervised_adaptation_v1 as source
        batch=source.pixel_batch(data,indices).numpy()
        result=encoder(batch)
        require(result.dtype==np.float32 and result.shape==(len(indices),384)
            and np.isfinite(result).all() and (np.abs(result).sum(1)>0).all())
        return result


def specification(pins):
    return {'schema':'bran-multisource-retinal-features-v2',
        'status':'frozen_before_feature_extraction','parameters':PARAMETERS,
        'source':pins,'code_sha256':{n:sha(ROOT/n) for n in CODE},
        'unadapted_dino_cache_reused':False,'v2_model_trained':False}


def authenticate_spec(out,backend):
    require(not (out/'failure.json').exists())
    p=json.loads((out/'protocol.json').read_text())
    data,use,pins=backend.source()
    require(p==specification(pins))
    return p,data,use


def prepare(out,private,backend):
    require(not out.exists() and not private.exists())
    _,_,pins=backend.source()
    out.mkdir(); private.mkdir(mode=0o700,parents=True)
    require(private.stat().st_mode&0o777==0o700)
    write_json(out/'protocol.json',specification(pins))
    progress(out,'prepared')


def extract(out,private,backend):
    import numpy as np
    p,data,use=authenticate_spec(out,backend)
    require(not (out/'aggregate.json').exists() and not (private/'features.npy').exists())
    require(private.is_dir() and not private.is_symlink() and private.stat().st_mode&0o777==0o700)
    progress(out,'frozen_encoder_loading'); encode=backend.encoder()
    target=private/'features.npy'
    # Reserve path before mmap creation; permissions remain private throughout.
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    mapped=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(use),384))
    start=time.monotonic(); batch=PARAMETERS['batch_size']
    for first in range(0,len(use),batch):
        selected=use[first:first+batch]; mapped[first:first+len(selected)]=backend.infer(encode,data,selected)
        if first==0 or (first//batch+1)%25==0:
            mapped.flush();progress(out,'frozen_feature_extraction',first//batch+1)
    mapped.flush();del mapped
    with (private/'grouping.npz').open('xb') as stream:
        os.fchmod(stream.fileno(),0o600)
        # No disease labels; positional group IDs/ages remain strictly private.
        np.savez_compressed(stream,source_rows=data['arrays']['source_rows'][use],
            groups=data['arrays']['groups'][use],ages=data['arrays']['ages'][use])
    # Re-authenticate source and code after inference as well as before it.
    require(authenticate_spec(out,backend)[0]==p)
    value={'schema':'bran-multisource-retinal-features-v2','status':'features_extracted_pending_audit',
        'protocol_sha256':sha(out/'protocol.json'),
        'features_sha256':sha(target),'grouping_sha256':sha(private/'grouping.npz'),
        'dimension':384,'dtype':'float32','elapsed_seconds':round(time.monotonic()-start,3),
        'images_lower_bound_20':p['source']['images_lower_bound_20'],
        'source_local_people_lower_bound_20':p['source']['source_local_people_lower_bound_20'],
        'model_training_started':False,'patient_level_output_emitted':False}
    write_json(out/'aggregate.json',value)
    write_json(out/'manifest.json',{'protocol_sha256':sha(out/'protocol.json'),
        'aggregate_sha256':sha(out/'aggregate.json')})
    progress(out,'extraction_completed_pending_audit')


def audit(out,private,backend):
    import numpy as np
    require(not (out/'audit.json').exists() and not (out/'audit_failure.json').exists())
    p,data,use=authenticate_spec(out,backend)
    value=json.loads((out/'aggregate.json').read_text());m=json.loads((out/'manifest.json').read_text())
    require(set(value)=={'schema','status','protocol_sha256','features_sha256','grouping_sha256',
        'dimension','dtype','elapsed_seconds','images_lower_bound_20','source_local_people_lower_bound_20',
        'model_training_started','patient_level_output_emitted'})
    require(value['schema']=='bran-multisource-retinal-features-v2' and
        value['status']=='features_extracted_pending_audit' and value['patient_level_output_emitted'] is False
        and value['model_training_started'] is False and value['dimension']==384 and value['dtype']=='float32'
        and type(value['elapsed_seconds']) in (float,int) and np.isfinite(value['elapsed_seconds']) and value['elapsed_seconds']>=0)
    require(all(value[k]==p['source'][k] for k in ('images_lower_bound_20','source_local_people_lower_bound_20')))
    require(m=={'protocol_sha256':sha(out/'protocol.json'),'aggregate_sha256':sha(out/'aggregate.json')})
    require(value['protocol_sha256']==sha(out/'protocol.json'))
    for name,key in (('features.npy','features_sha256'),('grouping.npz','grouping_sha256')):
        path=private/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_mode&0o777==0o600 and sha(path)==value[key])
    z=np.load(private/'features.npy',mmap_mode='r',allow_pickle=False)
    require(z.shape==(len(use),384) and z.dtype==np.float32)
    for first in range(0,len(z),256): require(np.isfinite(z[first:first+256]).all())
    with np.load(private/'grouping.npz',allow_pickle=False) as groups:
        require(set(groups.files)=={'source_rows','groups','ages'})
        require(all(np.array_equal(groups[name],data['arrays'][name][use],equal_nan=True) for name in groups.files))
    selected=np.linspace(0,len(use)-1,PARAMETERS['audit_images'],dtype=np.int64)
    encode=backend.encoder()
    for first in range(0,len(selected),PARAMETERS['batch_size']):
        rows=selected[first:first+PARAMETERS['batch_size']]
        replay=backend.infer(encode,data,use[rows])
        require(np.allclose(z[rows],replay,atol=PARAMETERS['audit_atol'],rtol=PARAMETERS['audit_rtol']))
    require(authenticate_spec(out,backend)[0]==p)
    write_json(out/'audit.json',{'schema':'bran-multisource-retinal-features-audit-v2',
        'status':'authenticated','protocol_sha256':sha(out/'protocol.json'),
        'aggregate_sha256':sha(out/'aggregate.json'),'manifest_sha256':sha(out/'manifest.json'),
        'features_sha256':sha(private/'features.npy'),'grouping_sha256':sha(private/'grouping.npz'),
        'selected_image_replay_passed':True,'model_training_started':False,
        'patient_level_output_emitted':False})
    progress(out,'feature_audit_completed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--attempt',type=int,default=1); args=parser.parse_args()
    answer={'status':'execution_failed','action':args.action,'patient_level_output_emitted':False}
    out=None; owned=False; audit_owned=False
    with quiet():
        try:
            out,private=paths(args.attempt)
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                backend=LocalBackend()
                if args.action=='prepare': prepare(out,private,backend)
                elif args.action=='run':
                    require(not (out/'aggregate.json').exists() and not (out/'failure.json').exists())
                    owned=True;extract(out,private,backend)
                else:
                    require(not (out/'audit.json').exists() and not (out/'audit_failure.json').exists())
                    audit_owned=True;audit(out,private,backend)
                answer['status']='completed'
        except Exception:
            if owned and out and out.is_dir() and not (out/'aggregate.json').exists() and not (out/'failure.json').exists():
                write_json(out/'failure.json',{'status':'execution_failed','phase':'feature_extraction',
                    'patient_level_output_emitted':False,'scientific_negative':False})
            if audit_owned and out and out.is_dir() and not (out/'audit.json').exists() and not (out/'audit_failure.json').exists():
                write_json(out/'audit_failure.json',{'status':'execution_failed','phase':'feature_audit',
                    'patient_level_output_emitted':False,'scientific_negative':False})
    print(json.dumps(answer));return 0 if answer['status']=='completed' else 1


if __name__=='__main__': raise SystemExit(main())

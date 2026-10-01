"""FD-quiet reauthentication and V2 age/eligibility binding, no model fitting."""
import argparse
import fcntl
import json
from pathlib import Path

from run_bran_multisource_retinal_features_v2 import LOCK,quiet,sha,write_json
from bran_multisource_clinical_v2 import bridge,safe_summary,eligibility_hash,SOURCES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'BRAN_MULTISOURCE_CLINICAL_BINDING_V2_ATTEMPT1'


def run():
    import run_bran_native_source_qualification_v1 as legacy
    legacy.require(not OUT.exists())
    protocol=json.loads(legacy.PROTOCOL.read_text());legacy.validate_protocol(protocol)
    # Existing terminal/audit metadata must match. No old experiment is rerun.
    from run_bran_multisource_inventory_v2 import verify_legacy_native_metadata
    legacy.require(verify_legacy_native_metadata(ROOT)['status']=='legacy_artifact_links_authenticated')
    results={}
    for source in SOURCES:
        arrays=legacy.load_one(source,protocol['sources'][source])
        pool=bridge(source,arrays);summary=safe_summary(pool)
        legacy.require(summary['status']=='supported_training_pool')
        results[pool.source]={'summary':summary,'legacy_source_receipt':protocol['sources'][source],
            'eligibility_sha256':eligibility_hash(pool)}
        del pool,arrays
    legacy.validate_protocol(protocol)
    result={'schema':'bran-multisource-clinical-binding-v2','status':'sources_reauthenticated_and_v2_eligibility_bound',
        'sources':results,'legacy_protocol_sha256':sha(legacy.PROTOCOL),
        'code_sha256':{name:sha(ROOT/name) for name in ('bran_multisource_clinical_v2.py',
            'run_bran_multisource_clinical_binding_v2.py','bran_multisource_age_v2.py')},
        'training_started':False,'private_arrays_emitted':False,'patient_level_output_emitted':False,
        'source_unit_policy_changed':False,'heldout_sources_used':False}
    OUT.mkdir();write_json(OUT/'aggregate.json',result)
    write_json(OUT/'manifest.json',{'aggregate_sha256':sha(OUT/'aggregate.json'),
        'patient_level_output_emitted':False})
    return result


def main():
    argparse.ArgumentParser().parse_args();ok=False
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);run();ok=True
        except Exception:pass
    print(json.dumps({'status':'sources_reauthenticated_and_v2_eligibility_bound' if ok else 'clinical_binding_failed',
        'training_started':False,'patient_level_output_emitted':False}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())

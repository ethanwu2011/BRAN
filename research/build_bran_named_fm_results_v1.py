"""New publication-safe named-FM package from authenticated aggregates only."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bran_named_fm_experiment_v1_attempt2 as experiment
from build_bran_endpoint_comparison_v2 import SHORT

ROOT=Path(__file__).resolve().parent
DIRECTORY='BRAN_NAMED_FM_RESULTS_V1'
GENERATOR='build_bran_named_fm_results_v1.py'
LABEL_SOURCE='build_bran_endpoint_comparison_v2.py'
PROTOCOL_SHA='7da7d9781d1f9d0f00df4ab12923eb3024c94d63b2c0539df35a5fa63904dd1e'
TERMINAL_SHA='d5af27264ba109117c9e767fd1344dd37d05ff1a461376acf4c1a09982978144'
FILES=('REPORT.md','results.json','01_macro_auroc.png','01_macro_auroc.svg','02_paired_differences.png','02_paired_differences.svg')
ORDER=('bran_both','bran_clinical','bran_retinal','retfound_green','visionfm_last4','dinov3_generic','labrador')
LABEL={'bran_both':'BRAN combined','bran_clinical':'BRAN clinical only','bran_retinal':'BRAN retinal only',
       'retfound_green':'RETFound-Green','visionfm_last4':'VisionFM (last four layers)',
       'dinov3_generic':'DINOv3 (generic)','labrador':'Labrador'}
INPUT={'bran_both':'Retina + full clinical','bran_clinical':'Full clinical',
       'bran_retinal':'Retina','retfound_green':'Retina','visionfm_last4':'Retina','dinov3_generic':'Retina',
       'labrador':'Mapped blood tests'}
require=experiment.require; sha=experiment.sha

def source(root,external=False):
    root=Path(root)
    require(sha(root/experiment.PROTOCOL)==PROTOCOL_SHA and sha(root/experiment.PATHS['success'])==TERMINAL_SHA)
    require(not(root/experiment.PATHS['failure']).exists() and not(root/experiment.PATHS['lock']).exists())
    with experiment.prior.base.paired.base._quiet_sensitive_block():
        p=experiment.validate_protocol(root,check_external=external)
        r=json.loads((root/experiment.PATHS['success']).read_text())
        require(r['protocol_sha256']==PROTOCOL_SHA and len(p['expected_hashes'])==126)
        summary=experiment.validate_report(r,p)
    require(set(r['result']['endpoint_results'])<=set(SHORT))
    return r,summary

def report(r,summary):
    macro=summary['macro_auroc']; deltas=summary['macro_paired_deltas']; family=summary['named_fm_macro_family']
    lines=['# BRAN matched named-foundation-model benchmark','',
        'Completed internal-development screening comparison: 1,928 participants, 26 supported recorded endpoints, five outer folds.',
        '', '## Mean screening AUROC', '',
        '| Model / readout | Inputs, plus age in every arm | Mean AUROC |', '|---|---|---:|']
    for arm in ORDER: lines.append(f"| {LABEL[arm]} | {INPUT[arm]} | {macro[arm]:.4f} |")
    lines+=['','![Macro AUROC](01_macro_auroc.png)','','## Paired comparisons','','Positive differences favor BRAN combined.',
        '', '| BRAN combined minus comparator | AUROC difference | Marginal paired 95% interval | Four-FM one-sided adjusted lower bound |',
        '|---|---:|---|---:|']
    for arm in ORDER[1:]:
        d=deltas['bran_both_minus_'+arm]; lower=family['bonferroni_one_sided_95_lower'].get(arm)
        lines.append(f"| {LABEL[arm]} | {d['mean_endpoint_auroc_difference']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "+
                     (f'{lower:+.4f} |' if lower is not None else 'Not in four-FM family |'))
    lines+=['','![Paired differences](02_paired_differences.png)','',
        f"The prespecified four-named-FM macro comparison gate **{'passes' if family['all_lower_above_zero'] else 'does not pass'}**: "
        'all four Bonferroni-adjusted one-sided lower bounds are above zero.' if family['all_lower_above_zero'] else
        'The prespecified four-named-FM macro comparison gate does not pass.',
        '', 'The combined-versus-full-clinical-state marginal interval includes zero. Do not infer that '
        'retina adds demonstrated benefit beyond full clinical data from the named-FM gate.',
        '', '## Interpretation and fairness', '',
        '- AUROC measures discrimination of supported recorded conditions/history/proxy endpoints, not demonstrated years-ahead disease forecasting.',
        '- Compute AUROC within each outer fold, weight by observed fold sample count, then average equally across the 26 endpoints. The macro plot shows point estimates only; no unsupported macro-AUROC uncertainty is invented.',
        '- All comparisons use the same canonical patients, endpoint masks, five outer/all five inner identities and shared paired bootstrap draws. Retinal models use the same selected images and unweighted all-view pooling; absent retinal evidence stays in the cohort as zero vectors.',
        '- BRAN combined has both modalities and a larger block-profile readout-selection budget. The named encoders are frozen; heads select among four regularization values. These are not equal-information or equal-search-budget comparisons and do not isolate encoder superiority.',
        '- One thousand fold-stratified common-patient bootstrap draws give fixed-fit, reused-development intervals. The four-FM macro gate uses the 1.25th percentile as each one-sided lower bound. Endpoint intervals remain marginal; neither historical development nor the entire outcome atlas has familywise correction.',
        '- RETFound-Green is not the original Nature RETFound model. Labrador is a mapped blood representation, not a longitudinal EHR foundation model. MOTOR and CLMBR were not scored.',
        '- No untouched external cohort, official test set, new disease subtype, treatment effect or clinical replacement claim is established. No automatic promotion of the default model.',
        '', '## All 26 endpoint point estimates', '',
        'Labels summarize the original source constructs; source codes are retained. All detailed marginal intervals and log losses are preserved in results.json.',
        '', '| Recorded endpoint / source | '+ ' | '.join(LABEL[a] for a in ORDER)+' |',
        '|---|'+'---:|'*len(ORDER)]
    for code,e in sorted(r['result']['endpoint_results'].items()):
        lines.append(f"| {SHORT[code]} ({code}) | "+' | '.join(f"{e['arms'][arm]['auroc']:.4f}" for arm in ORDER)+' |')
    lines+=['','## Authentication','',f'- Protocol SHA-256: `{PROTOCOL_SHA}`; 126 bound components.',
        f'- Exclusive success SHA-256: `{TERMINAL_SHA}`.',
        '- Parent technical failure is preserved. All model artifact hashes and both runtimes were re-authenticated at package creation.',
        '- Canonical source/fold identities, image selection/content binding and end-of-run unchanged-source assertions passed. All three reference arms replayed 26 AUROCs and log losses before named-FM scoring.',
        '- No patient images, identifiers, source paths, predictions, embeddings, latent states, model weights or bootstrap draws are included. Figures are deterministic Python/Matplotlib plots from the authenticated aggregates.','']
    return '\n'.join(lines)

def figures(summary,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.spines.left':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(10,5.9)); y=np.arange(len(ORDER))
    for i,arm in enumerate(ORDER):
        value=summary['macro_auroc'][arm]; color='#126B70' if arm=='bran_both' else '#687681'
        ax.scatter(value,i,s=65,color=color,zorder=3); ax.text(value+.0025,i,f'{value:.3f}',va='center',fontsize=11)
    ax.set_yticks(y,[LABEL[a] for a in ORDER]); ax.invert_yaxis(); ax.set_xlim(.50,.76)
    ax.set_xlabel('Mean AUROC across 26 recorded endpoints'); ax.grid(axis='x',color='#E4E8EB',linewidth=.7)
    ax.set_title('Screening: same cohort and folds, different available inputs',loc='left',pad=18,fontweight='bold')
    ax.tick_params(axis='y',length=0,pad=10); fig.subplots_adjust(left=.31,right=.97,top=.87,bottom=.24)
    fig.text(.035,.10,'Point estimates; paired uncertainty is shown separately. All arms include age.',fontsize=10)
    fig.text(.035,.052,'BRAN combined uses retina + full clinical data and a larger readout-selection budget.\nInternal development; RETFound-Green is not original Nature RETFound.',fontsize=9,color='#4D5861')
    for ext in ('png','svg'): fig.savefig(out/f'01_macro_auroc.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6.0)); comparisons=ORDER[1:]
    for i,arm in enumerate(comparisons):
        d=summary['macro_paired_deltas']['bran_both_minus_'+arm]; value=d['mean_endpoint_auroc_difference']; lo,hi=d['ci95']
        color='#687681' if arm.startswith('bran_') else '#126B70'
        ax.plot([lo,hi],[i,i],color=color,lw=2); ax.scatter(value,i,color=color,s=50,zorder=3)
    ax.axvline(0,color='#687681',lw=1,ls='--'); ax.set_yticks(range(len(comparisons)),[LABEL[a] for a in comparisons])
    ax.invert_yaxis(); ax.set_xlim(-.014,.113); ax.set_xlabel('BRAN combined − comparator: mean AUROC difference')
    ax.grid(axis='x',color='#E4E8EB',linewidth=.7); ax.tick_params(axis='y',length=0,pad=10)
    ax.set_title('Paired differences: full-clinical advantage remains uncertain',loc='left',pad=18,fontweight='bold')
    fig.subplots_adjust(left=.31,right=.97,top=.87,bottom=.24)
    fig.text(.035,.105,'Lines: marginal paired 95% intervals. Four named-FM macro comparisons also pass\nthe prespecified one-sided Bonferroni gate; adjusted lower bounds are in the report.',fontsize=10)
    fig.text(.035,.040,'Fixed-fit bootstrap on reused development data. Not an equal-modality comparison or external validation.',fontsize=9,color='#4D5861')
    for ext in ('png','svg'): fig.savefig(out/f'02_paired_differences.{ext}',dpi=220)
    plt.close(fig)

def build(root=ROOT):
    root=Path(root); r,summary=source(root,external=True); out=root/DIRECTORY
    require(not out.exists()); out.mkdir()
    (out/'REPORT.md').write_text(report(r,summary))
    (out/'results.json').write_text(json.dumps(r['result'],indent=2,sort_keys=True,allow_nan=False)+'\n')
    figures(summary,out)
    manifest={'schema':'bran-named-fm-release-v1','protocol_sha256':PROTOCOL_SHA,'terminal_sha256':TERMINAL_SHA,
        'generator_sha256':sha(root/GENERATOR),'label_source_sha256':sha(root/LABEL_SOURCE),
        'external_artifacts_and_runtimes_authenticated_at_build':True,
        'files':{name:sha(out/name) for name in FILES},'patient_content_included':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return audit(root)

def audit(root=ROOT):
    root=Path(root); out=root/DIRECTORY; manifest=json.loads((out/'manifest.json').read_text())
    experiment.exact_keys(manifest,{'schema','protocol_sha256','terminal_sha256','generator_sha256','label_source_sha256',
        'external_artifacts_and_runtimes_authenticated_at_build','files','patient_content_included'})
    require(manifest['schema']=='bran-named-fm-release-v1' and manifest['protocol_sha256']==PROTOCOL_SHA
        and manifest['terminal_sha256']==TERMINAL_SHA and manifest['generator_sha256']==sha(root/GENERATOR)
        and manifest['label_source_sha256']==sha(root/LABEL_SOURCE)
        and manifest['external_artifacts_and_runtimes_authenticated_at_build'] is True and manifest['patient_content_included'] is False)
    r,summary=source(root); experiment.exact_keys(manifest['files'],set(FILES))
    require(set(p.name for p in out.iterdir())==set(FILES)|{'manifest.json'})
    for name,digest in manifest['files'].items(): require(sha(out/name)==digest)
    got=json.loads((out/'results.json').read_text()); experiment.validate_result(got,tuple(r['result']['endpoint_results']))
    require(got==r['result'] and (out/'REPORT.md').read_text()==report(r,summary))
    return {'status':'authenticated_aggregate_release','files':len(FILES),'components_authenticated':126,
        'manifest_sha256':sha(out/'manifest.json'),'exact_source_equality':True,'patient_content_included':False}

if __name__=='__main__':
    try: print(json.dumps(build() if sys.argv[1]=='build' else audit(),sort_keys=True))
    except Exception as exc:
        print(json.dumps({'status':'blocked_without_disclosure','error_class':type(exc).__name__,'contents_emitted':False})); raise SystemExit(1)

"""Publish only authenticated outcome-level aggregates; never load patient data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from audit_bran_anchor_ablation_v2 import audit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'BRAN_ENDPOINT_COMPARISON_V2'
RESULT_SHA = '3e274f9920e272634442cfb90a3e794d4f15c5505a8c30345f98fb375de79c18'
REGISTRY_SHA = '7204a934d180db5af79dbb57bd759e44603977d52f47c0725d8012c8de9f6c21'
SHORT = {
    'mh_a1c': 'Elevated HbA1c', 'mhoccur_ad': 'Dementia',
    'mhoccur_amd': 'Age-related macular degeneration', 'mhoccur_ca': 'Cancer (any type)',
    'mhoccur_circ': 'Circulation problems', 'mhoccur_clsh': 'High cholesterol',
    'mhoccur_cns': 'Other neurological conditions', 'mhoccur_cogn': 'Mild cognitive impairment',
    'mhoccur_crt': 'Cataracts', 'mhoccur_cvdot': 'Other heart issues',
    'mhoccur_ded': 'Dry eye', 'mhoccur_ear': 'Hearing impairment',
    'mhoccur_fall': 'Falls in prior 12 months', 'mhoccur_gi': 'Digestive problems',
    'mhoccur_glc': 'Glaucoma', 'mhoccur_hbp': 'High blood pressure',
    'mhoccur_lbp': 'Low blood pressure', 'mhoccur_mi': 'Heart attack',
    'mhoccur_ms': 'Multiple sclerosis', 'mhoccur_oa': 'Osteoporosis',
    'mhoccur_obs': 'Obesity', 'mhoccur_pd': 'Parkinson disease',
    'mhoccur_pdr': 'Diabetic retinopathy', 'mhoccur_plm': 'Chronic lung problems',
    'mhoccur_ra': 'Arthritis', 'mhoccur_rnl': 'Kidney problems',
    'mhoccur_rvo': 'Retinal vascular occlusion', 'mhoccur_strk': 'Stroke',
    'mhoccur_ua': 'Urinary problems', 'mhterm_dm1': 'Type 1 diabetes',
    'mhterm_dm2': 'Type 2 diabetes', 'mhterm_predm': 'Prediabetes',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_rows():
    checked = audit()
    if checked['status'] != 'authenticated_success' or checked['artifact_sha256'] != RESULT_SHA:
        raise ValueError('paired aggregate authentication failed')
    raw = (ROOT / 'validation_results/BRAN_ANCHOR_ABLATION_V2/SUCCESS.json').read_bytes()
    registry = (ROOT / 'BRAN_BODYWIDE_ENDPOINT_REGISTRY_V1.json').read_bytes()
    if sha(raw) != RESULT_SHA or sha(registry) != REGISTRY_SHA:
        raise ValueError('aggregate or registry hash mismatch')
    endpoint = json.loads(raw)['screening']['endpoint_results']
    # Select row-free endpoint metadata only; never export registry support cells.
    metadata = {e['source_id']: {k: e[k] for k in ('source_id', 'label', 'primary_system')}
                for e in json.loads(registry)['entries']
                if e.get('registry_role') == 'primary_binary_screening_candidate'}
    if len(metadata) != 32 or len(endpoint) != 26 or not set(endpoint) <= set(metadata) <= set(SHORT):
        raise ValueError('endpoint universe mismatch')
    rows = []
    for code, item in endpoint.items():
        arms = item['arms']
        eye, clinical, both = [arms[a]['auroc'] for a in ('v2retinal', 'v2clinical', 'v2both')]
        row = {**metadata[code], 'short_label': SHORT[code],
               'retinal_state_auc': eye, 'clinical_state_auc': clinical,
               'combined_state_auc': both, 'raw_clinical_auc': arms['rawclinical']['auroc'],
               'raw_concatenation_auc': arms['rawconcat']['auroc'],
               'combined_minus_best_single_point': both - max(eye, clinical),
               'best_single_at_point': 'retina' if eye > clinical else ('clinical' if clinical > eye else 'tie'),
               'combined_above_both_points': both > max(eye, clinical),
               'paired_combined_minus_clinical': item['paired_deltas']['v2both-v2clinical'],
               'paired_combined_minus_retinal': item['paired_deltas']['v2both-v2retinal']}
        rows.append(row)
    rows.sort(key=lambda r: (r['primary_system'], r['short_label']))
    return rows, checked


def figure_base(rows, title, subtitle):
    fig = go.Figure()
    fig.update_layout(width=1500, height=1560, paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Arial', size=20, color='#17212B'),
        title=dict(text=title+'<br><span style="font-size:19px;color:#53616C">'+subtitle+'</span>', x=.025, y=.98),
        margin=dict(l=435, r=80, t=180, b=240),
        legend=dict(orientation='h', x=0, y=1.065, font=dict(size=19)),
        yaxis=dict(tickvals=list(range(len(rows))),
                   ticktext=[r['short_label'] for r in rows], range=[len(rows)-.4, -.6],
                   gridcolor='#E9EEF1', zeroline=False),
        xaxis=dict(gridcolor='#E2E7EB', zeroline=False))
    return fig


def write_figures(rows):
    fig = figure_base(rows, 'BRAN V2 | which inputs discriminate each condition?',
        'All 26 supported reported-condition outcomes, grouped by registry system; age is supplied to every route')
    for field, name, color, symbol in [
        ('retinal_state_auc', 'Retinal state', '#167A74', 'circle'),
        ('clinical_state_auc', 'Clinical state', '#C37632', 'square'),
        ('combined_state_auc', 'Combined state', '#6956A5', 'diamond'),
        ('raw_clinical_auc', 'Raw clinical reference', '#929CA5', 'x')]:
        fig.add_trace(go.Scatter(x=[r[field] for r in rows], y=list(range(len(rows))),
            mode='markers', name=name, marker=dict(color=color, symbol=symbol, size=10 if field=='raw_clinical_auc' else 12)))
    low = min(r[k] for r in rows for k in ('retinal_state_auc', 'clinical_state_auc', 'combined_state_auc', 'raw_clinical_auc'))
    fig.update_xaxes(title='AUROC (higher is better)', range=[min(.45, low-.02), 1.0], dtick=.1)
    fig.add_annotation(x=-.43, y=-.14, xref='paper', yref='paper', xanchor='left', yanchor='top', align='left', showarrow=False,
        text='Internal five-fold development evaluation; contemporaneous clinical proxies remain.<br>'
             'Clinical means eligible measurements beyond CBC, not a blood-only arm. Point differences are not causal modality attribution.<br>'
             'This is the supported 26-outcome panel, not a comprehensive body-wide disease atlas.', font=dict(size=18))
    fig.write_image(OUT/'all_26_outcomes.png', scale=1.1)
    fig.write_image(OUT/'all_26_outcomes.svg')

    diff = figure_base(rows, 'BRAN V2 | what does combining inputs add?',
        'Prespecified paired AUROC contrasts; marginal 95% intervals, with every supported outcome retained')
    for key, label, color, offset in [
        ('paired_combined_minus_clinical', 'Combined − clinical state', '#C37632', -.14),
        ('paired_combined_minus_retinal', 'Combined − retinal state', '#167A74', .14)]:
        d = [r[key] for r in rows]
        x = [v['auroc_delta'] for v in d]
        # Percentile intervals need not contain the point estimate: draw endpoints directly.
        for i, v in enumerate(d):
            diff.add_trace(go.Scatter(x=v['ci95'], y=[i+offset, i+offset], mode='lines',
                                     line=dict(color=color, width=2), showlegend=False, hoverinfo='skip'))
        diff.add_trace(go.Scatter(x=x, y=np.arange(len(rows))+offset, mode='markers', name=label,
                                 marker=dict(color=color, size=10)))
    diff.add_vline(x=0, line=dict(color='#79858F', dash='dash', width=1))
    endpoints=[v for r in rows for k in ('paired_combined_minus_clinical','paired_combined_minus_retinal') for v in r[k]['ci95']]
    lower,upper=min(endpoints),max(endpoints);padding=(upper-lower)*.04
    diff.update_xaxes(title='Paired AUROC difference (positive favors combined)', range=[lower-padding,upper+padding])
    diff.add_annotation(x=-.43, y=-.14, xref='paper', yref='paper', xanchor='left', yanchor='top', align='left', showarrow=False,
        text='1,000 shared fold-stratified patient-bootstrap draws; fits remain fixed. No multiplicity or refit-uncertainty adjustment.<br>'
             'Intervals are for the two prespecified comparisons, not for a post-selected best single modality.<br>'
             'Greater discrimination does not identify the disease mechanism, affected organ or a novel subtype.', font=dict(size=18))
    diff.write_image(OUT/'paired_modality_contrasts.png', scale=1.1)
    diff.write_image(OUT/'paired_modality_contrasts.svg')


def main():
    rows, checked = load_rows()
    OUT.mkdir(exist_ok=True)
    counts = {'combined_above_both_points': sum(r['combined_above_both_points'] for r in rows),
              'retina_above_clinical_point': sum(r['best_single_at_point']=='retina' for r in rows),
              'clinical_above_retina_point': sum(r['best_single_at_point']=='clinical' for r in rows)}
    if counts['combined_above_both_points'] != checked['both_above_own_single_views_point_count']['v2']:
        raise ValueError('summary disagreement')
    payload = {'schema':'bran-outcome-comparison-v2', 'aggregate_only':True,
               'source_sha256':RESULT_SHA, 'registry_sha256':REGISTRY_SHA,
               'point_summary':counts, 'rows':rows,
               'claim_limit':'Exploratory supported-condition discrimination; no comprehensive atlas or causal attribution claim.'}
    (OUT/'results.json').write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+'\n')
    lines = ['# BRAN V2: all 26 supported outcomes', '',
        f"Combined state exceeds both of its own single-input point estimates on **{counts['combined_above_both_points']}/26 outcomes**. "
        f"The retinal state exceeds the clinical state on **{counts['retina_above_clinical_point']}/26**; "
        f"the clinical state exceeds the retinal state on **{counts['clinical_above_retina_point']}/26**. "
        'These are descriptive comparisons, not multiplicity-adjusted superiority or disease-mechanism assignments.', '',
        'Every route includes age. Clinical inputs include 43 eligible continuous measurements, not only CBC. '
        'Diagnosis-history slots are excluded, but contemporaneous clinical proxies remain. '
        'Outcomes are reported prevalent conditions, not adjudicated incident disease.', '',
        '## Complete table', '',
        '| Registry system | Reported condition | Retina state AUROC | Clinical state AUROC | Combined state AUROC | Raw clinical AUROC | Combined − best single, point |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        label=r['label'].replace('|','/')
        lines.append(f"| {r['primary_system']} | {label} | {r['retinal_state_auc']:.3f} | {r['clinical_state_auc']:.3f} | {r['combined_state_auc']:.3f} | {r['raw_clinical_auc']:.3f} | {r['combined_minus_best_single_point']:+.3f} |")
    lines += ['', '## How to read the figures', '',
        'The AUROC plot retains all outcomes in registry-system/label order; no favorable-result selection. '
        'The second plot shows the two prespecified paired contrasts with their marginal 95% intervals. '
        'Do not treat overlapping/nonoverlapping individual-arm intervals as a test, or attach these paired intervals '
        'to the data-selected best-single difference in the table.', '',
        'Endpoint AUROC is an observed-count-weighted mean of the five outer-fold AUROCs; the study macro '
        'equally weights all 26 endpoints. Intervals use 1,000 shared fold-stratified patient bootstrap draws '
        'with fitted models fixed. No familywise, external-validation or future-onset claim is supported.', '',
        '## Scope boundary', '',
        'The registry contains 32 primary survey candidates; the authenticated evaluation supports 26. '
        'This release neither adds unsupported outcomes nor publishes small support cells. AI-READI and the wider '
        'project have additional clinical/OMOP concepts; these 32 are not the total disease universe. '
        'A comprehensive body-wide outcome atlas requires additional valid endpoint definitions and support, '
        'not simply treating each recorded procedure or measurement as a new disease label.', '',
        f'Source result SHA-256: `{RESULT_SHA}`.',
        f'Row-free registry SHA-256: `{REGISTRY_SHA}`.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write_figures(rows)
    files = {p.name:sha(p.read_bytes()) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps({'aggregate_only':True,'source_sha256':RESULT_SHA,
         'builder_sha256':sha(Path(__file__).read_bytes()),'files':files}, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'status':'built_aggregate_endpoint_comparison','endpoint_count':len(rows),'point_summary':counts}))


if __name__ == '__main__':
    main()

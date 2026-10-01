"""Disclosure-safe figures and Markdown for authenticated R7 missingness aggregates.

This module consumes only the caller-provided aggregate dictionary. It does not
load source files, arrays, checkpoints, or perform inference.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCHEMA = 'bran-r7-missingness-figure-m2'
PROFILE_ORDER = (
    'available',
    'clinical_drop25',
    'clinical_drop50',
    'clinical_drop75',
    'no_retina',
    'all_clinical_hidden',
    'whole_cbc_hidden',
    'whole_cbc_no_retina',
)
PRIMARY_PATTERN_ORDER = (
    'all_clinical_hidden',
    'clinical_drop25',
    'clinical_drop50',
    'clinical_drop75',
    'no_retina',
)
DESCRIPTIVE_FOREST_ORDER = ('available', 'whole_cbc_hidden', 'whole_cbc_no_retina')
CURVE_ORDER = ('available', 'clinical_drop25', 'clinical_drop50', 'clinical_drop75')
CURVE_PERCENT = {
    'available': 0,
    'clinical_drop25': 25,
    'clinical_drop50': 50,
    'clinical_drop75': 75,
}
ARMS = ('C', 'S', 'V5')
CONTRASTS = ('S_minus_C', 'S_minus_V5')
LABELS = {
    'available': 'Existing observations',
    'clinical_drop25': '25% clinical fields withheld',
    'clinical_drop50': '50% clinical fields withheld',
    'clinical_drop75': '75% clinical fields withheld',
    'no_retina': 'No retinal input',
    'all_clinical_hidden': 'Clinical physiology hidden',
    'whole_cbc_hidden': 'Whole CBC hidden (descriptive route)',
    'whole_cbc_no_retina': 'Whole CBC hidden, no retina (descriptive route)',
}
ARM_COLORS = {'S': '#087f83', 'V5': '#7d8991', 'C': '#c98562'}
OUTPUT_STEM = 'bran_r7_missingness'


def _fail():
    raise ValueError('bran_r7_missingness_figure_m2_invalid') from None


def _exact_keys(value, keys):
    return type(value) is dict and set(value) == set(keys)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _safe_coverage(value):
    if type(value) is not dict or type(value.get('status')) is not str:
        _fail()
    if value['status'] == 'withheld':
        if not _exact_keys(value, ('status',)):
            _fail()
        return {'status': 'withheld'}
    if value['status'] != 'released' or not _exact_keys(value, ('status', 'supported', 'total')):
        _fail()
    supported, total = value['supported'], value['total']
    if type(supported) is not int or type(total) is not int or total < 20:
        _fail()
    complement = total - supported
    if supported < 0 or complement < 0:
        _fail()
    if not (supported == 0 or supported >= 20):
        _fail()
    if not (complement == 0 or complement >= 20):
        _fail()
    return {'status': 'released', 'supported': supported, 'total': total}


def _ci(value, lower, upper):
    if type(value) is not list or len(value) != 2:
        _fail()
    if not all(_finite_number(item) for item in value):
        _fail()
    lo, hi = (float(item) for item in value)
    if lo > hi or lo < lower or hi > upper:
        _fail()
    return [lo, hi]


def _standard_cell(value, *, allow_unsupported=True):
    if type(value) is not dict or type(value.get('status')) is not str:
        _fail()
    if value['status'] == 'unsupported' and allow_unsupported:
        if not _exact_keys(value, ('status',)):
            _fail()
        return {'status': 'unsupported'}
    if value['status'] != 'supported' or not _exact_keys(value, ('status', 'arms', 'contrasts')):
        _fail()
    if not _exact_keys(value['arms'], ARMS) or not _exact_keys(value['contrasts'], CONTRASTS):
        _fail()
    arms = {}
    for role in ARMS:
        arm = value['arms'][role]
        if not _exact_keys(arm, ('auroc', 'ci95')) or not _finite_number(arm['auroc']):
            _fail()
        auroc = float(arm['auroc'])
        if not 0 <= auroc <= 1:
            _fail()
        arms[role] = {'auroc': auroc, 'ci95': _ci(arm['ci95'], 0, 1)}
    contrasts = {}
    for name in CONTRASTS:
        contrast = value['contrasts'][name]
        if not _exact_keys(contrast, ('delta', 'ci95')) or not _finite_number(contrast['delta']):
            _fail()
        delta = float(contrast['delta'])
        if not -1 <= delta <= 1:
            _fail()
        contrasts[name] = {'delta': delta, 'ci95': _ci(contrast['ci95'], -1, 1)}
    for name, left, right in (('S_minus_C', 'S', 'C'), ('S_minus_V5', 'S', 'V5')):
        expected = arms[left]['auroc'] - arms[right]['auroc']
        if abs(contrasts[name]['delta'] - expected) > 1e-9:
            _fail()
    return {'status': 'supported', 'arms': arms, 'contrasts': contrasts}


def _coverage_record(value):
    if not _exact_keys(value, ('arms', 'matched')) or not _exact_keys(value['arms'], ARMS):
        _fail()
    return {
        'arms': {role: _safe_coverage(value['arms'][role]) for role in ARMS},
        'matched': _safe_coverage(value['matched']),
    }


def payload(native: dict) -> dict:
    """Build the compact plotting payload from authenticated native aggregates."""
    if type(native) is not dict or 'stress_profiles' not in native or 'stress_primary' not in native:
        _fail()
    profiles_in = native['stress_profiles']
    primary_in = native['stress_primary']
    if not _exact_keys(profiles_in, PROFILE_ORDER):
        _fail()

    profiles = {}
    for pattern in PROFILE_ORDER:
        source = profiles_in[pattern]
        if not _exact_keys(source, ('complete_26_panel', 'endpoints', 'macro',
                                    'matched_population_coverage', 'prediction_coverage')):
            _fail()
        complete = source['complete_26_panel']
        endpoints = source['endpoints']
        if type(complete) is not bool or type(endpoints) is not dict or len(endpoints) != 26:
            _fail()
        if any(type(name) is not str or not name for name in endpoints):
            _fail()
        endpoint_cells = [_standard_cell(endpoint_cell) for endpoint_cell in endpoints.values()]
        if complete != all(cell['status'] == 'supported' for cell in endpoint_cells):
            _fail()
        macro = source['macro']
        if complete:
            macro = _standard_cell(macro, allow_unsupported=False)
        elif macro is not None:
            _fail()
        coverage_source = source['prediction_coverage']
        if not _exact_keys(coverage_source, ARMS):
            _fail()
        coverage = {
            'arms': {role: _safe_coverage(coverage_source[role]) for role in ARMS},
            'matched': _safe_coverage(source['matched_population_coverage']),
        }
        if pattern == 'all_clinical_hidden' and (
            coverage['matched']['status'] != 'withheld'
            or any(coverage['arms'][role]['status'] != 'withheld' for role in ARMS)
        ):
            _fail()
        profiles[pattern] = {
            'complete_26_panel': complete,
            'macro': macro,
            'coverage': coverage,
        }

    if not _exact_keys(primary_in, ('complete_26_panel', 'coverage', 'endpoint_count',
                                    'patterns', 'status', 'summary')):
        _fail()
    if type(primary_in['complete_26_panel']) is not bool:
        _fail()
    if not primary_in['complete_26_panel']:
        _fail()
    if type(primary_in['endpoint_count']) is not int or primary_in['endpoint_count'] != 26:
        _fail()
    if primary_in['status'] != 'supported':
        _fail()
    if not _exact_keys(primary_in['patterns'], PRIMARY_PATTERN_ORDER):
        _fail()
    primary_pattern_cells = {
        pattern: _standard_cell(primary_in['patterns'][pattern], allow_unsupported=False)
        for pattern in PRIMARY_PATTERN_ORDER
    }
    primary_coverage_in = primary_in['coverage']
    if not _exact_keys(primary_coverage_in, PRIMARY_PATTERN_ORDER):
        _fail()
    primary_coverage = {}
    for pattern in PRIMARY_PATTERN_ORDER:
        coverage = _coverage_record(primary_coverage_in[pattern])
        if coverage != profiles[pattern]['coverage']:
            _fail()
        primary_coverage[pattern] = coverage
    summary = _standard_cell(primary_in['summary'], allow_unsupported=False)

    result = {
        'schema': SCHEMA,
        'profiles': profiles,
        'primary_coverage': primary_coverage,
        'primary_pattern_cells': primary_pattern_cells,
        'primary_summary': summary,
    }
    validate_payload(result)
    return result


def validate_payload(data: dict) -> None:
    """Validate the closed, disclosure-safe payload schema; raise on any defect."""
    if not _exact_keys(data, ('schema', 'profiles', 'primary_coverage',
                              'primary_pattern_cells', 'primary_summary')):
        _fail()
    if data['schema'] != SCHEMA or not _exact_keys(data['profiles'], PROFILE_ORDER):
        _fail()
    for pattern in PROFILE_ORDER:
        profile = data['profiles'][pattern]
        if not _exact_keys(profile, ('complete_26_panel', 'macro', 'coverage')):
            _fail()
        if type(profile['complete_26_panel']) is not bool:
            _fail()
        if profile['complete_26_panel']:
            _standard_cell(profile['macro'], allow_unsupported=False)
        elif profile['macro'] is not None:
            _fail()
        _coverage_record(profile['coverage'])
    if not _exact_keys(data['primary_coverage'], PRIMARY_PATTERN_ORDER):
        _fail()
    for pattern in PRIMARY_PATTERN_ORDER:
        coverage = _coverage_record(data['primary_coverage'][pattern])
        if coverage != data['profiles'][pattern]['coverage']:
            _fail()
    if not _exact_keys(data['primary_pattern_cells'], PRIMARY_PATTERN_ORDER):
        _fail()
    for pattern in PRIMARY_PATTERN_ORDER:
        _standard_cell(data['primary_pattern_cells'][pattern], allow_unsupported=False)
    _standard_cell(data['primary_summary'], allow_unsupported=False)
    if data['profiles']['all_clinical_hidden']['coverage']['matched']['status'] != 'withheld':
        _fail()
    if any(data['profiles']['all_clinical_hidden']['coverage']['arms'][role]['status'] != 'withheld'
           for role in ARMS):
        _fail()


def _coverage_text(value):
    if value['status'] == 'withheld':
        return 'Withheld (privacy threshold)'
    return f"{value['supported']} / {value['total']} released"


def _arm_text(cell, role):
    if cell is None:
        return 'Unsupported'
    arm = cell['arms'][role]
    lo, hi = arm['ci95']
    return f"{arm['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"


def _contrast_text(cell, name):
    if cell is None:
        return 'Unsupported'
    contrast = cell['contrasts'][name]
    lo, hi = contrast['ci95']
    return f"{contrast['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def markdown(data: dict) -> str:
    """Return a disclosure-safe report containing score intervals and all coverage."""
    validate_payload(data)
    lines = [
        '# R7 missingness stress test',
        '',
        'Descriptive presentation of the prespecified completed evaluation using reused development '
        'data and the fixed fit. This report adds no new primary test, fitting, or inference. Arm '
        'scores use absolute bootstrap 95% CIs; contrasts use marginal paired 95% CIs. Intervals are '
        'not multiplicity adjusted. The 0% row means no additional controlled dropout and does not '
        'imply complete fields. Clinical physiology hidden retains typed-age input. Controlled '
        'field dropout differs from natural missingness; no-retina and whole-CBC conditions are '
        'separately labeled descriptive routes.',
        '',
        '## Coverage and abstention',
        '',
        'Values report supported predictions over the full denominator only when disclosure '
        'thresholds permit release. Withheld values remain explicitly suppressed.',
        '',
        '| Condition | Matched population | C | S (R7) | V5 |',
        '|---|---:|---:|---:|---:|',
    ]
    for pattern in PROFILE_ORDER:
        coverage = data['profiles'][pattern]['coverage']
        lines.append(
            f"| {LABELS[pattern]} | {_coverage_text(coverage['matched'])} | "
            f"{_coverage_text(coverage['arms']['C'])} | {_coverage_text(coverage['arms']['S'])} | "
            f"{_coverage_text(coverage['arms']['V5'])} |"
        )

    lines.extend([
        '',
        '## Prespecified primary five-pattern estimates',
        '',
        'Pattern cells come from the joint five-pattern bootstrap; the summary row averages the '
        'prespecified patterns. Arm columns show absolute bootstrap 95% CIs. Contrast columns show '
        'paired marginal 95% CIs.',
        '',
        '| Primary result | C AUROC [bootstrap 95% CI] | S (R7) AUROC [bootstrap 95% CI] | V5 AUROC [bootstrap 95% CI] | S−C [paired 95% CI] | S−V5 [paired 95% CI] |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| **Primary summary** | **{_arm_text(data['primary_summary'], 'C')}** | "
        f"**{_arm_text(data['primary_summary'], 'S')}** | **{_arm_text(data['primary_summary'], 'V5')}** | "
        f"**{_contrast_text(data['primary_summary'], 'S_minus_C')}** | "
        f"**{_contrast_text(data['primary_summary'], 'S_minus_V5')}** |",
    ])
    for pattern in PRIMARY_PATTERN_ORDER:
        cell = data['primary_pattern_cells'][pattern]
        lines.append(
            f"| {LABELS[pattern]} | {_arm_text(cell, 'C')} | {_arm_text(cell, 'S')} | "
            f"{_arm_text(cell, 'V5')} | {_contrast_text(cell, 'S_minus_C')} | "
            f"{_contrast_text(cell, 'S_minus_V5')} |"
        )

    lines.extend([
        '',
        '## Per-profile screening macro (all eight profiles)',
        '',
        'Profile rows below are separate per-profile screening aggregates. Arm columns show absolute '
        'bootstrap 95% CIs; contrast columns show paired marginal 95% CIs. Negative differences '
        'are retained.',
        '',
        '| Condition | Additional clinical-field dropout | C AUROC [bootstrap 95% CI] | S (R7) AUROC [bootstrap 95% CI] | V5 AUROC [bootstrap 95% CI] | S−C [paired 95% CI] | S−V5 [paired 95% CI] |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for pattern in PROFILE_ORDER:
        cell = data['profiles'][pattern]['macro']
        dropout = f"{CURVE_PERCENT[pattern]}%" if pattern in CURVE_PERCENT else '—'
        if pattern == 'available':
            dropout = '0% additional'
        lines.append(
            f"| {LABELS[pattern]} | {dropout} | {_arm_text(cell, 'C')} | "
            f"{_arm_text(cell, 'S')} | {_arm_text(cell, 'V5')} | "
            f"{_contrast_text(cell, 'S_minus_C')} | {_contrast_text(cell, 'S_minus_V5')} |"
        )
    lines.extend([
        '',
        'Unsupported profiles are labeled explicitly. Whole-CBC conditions are descriptive routes; '
        'the primary summary row is the fixed five-pattern summary.',
    ])
    return '\n'.join(lines)


def _supported_macro(profile):
    return profile['macro'] if profile['macro'] is not None else None


def _curve_ylim(data):
    bounds = []
    for pattern in CURVE_ORDER:
        macro = _supported_macro(data['profiles'][pattern])
        if macro is None:
            continue
        for role in ARMS:
            bounds.append(macro['arms'][role]['auroc'])
            bounds.extend(macro['arms'][role]['ci95'])
    if not bounds or (min(bounds) >= 0.5 and max(bounds) <= 0.8):
        return 0.5, 0.8
    lo, hi = min(bounds), max(bounds)
    pad = max((hi - lo) * 0.08, 0.015)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _draw_curve(data):
    fig, axis = plt.subplots(figsize=(8.8, 5.8), constrained_layout=False)
    x_values = [CURVE_PERCENT[pattern] for pattern in CURVE_ORDER]
    for role in ('S', 'V5', 'C'):
        y_values, intervals = [], []
        for pattern in CURVE_ORDER:
            macro = _supported_macro(data['profiles'][pattern])
            if macro is None:
                y_values.append(float('nan'))
                intervals.append(None)
            else:
                arm = macro['arms'][role]
                y_values.append(arm['auroc'])
                intervals.append(arm['ci95'])
        axis.plot(x_values, y_values, color=ARM_COLORS[role], linewidth=1.8, zorder=2)
        for x, y, interval in zip(x_values, y_values, intervals):
            if interval is None:
                continue
            lo, hi = interval
            axis.vlines(x, lo, hi, color=ARM_COLORS[role], linewidth=1.35, zorder=3)
            axis.hlines((lo, hi), x - 1.15, x + 1.15, color=ARM_COLORS[role], linewidth=1.1, zorder=3)
            axis.scatter(x, y, color=ARM_COLORS[role], s=31, edgecolor='white', linewidth=.5, zorder=4)
    axis.set_xlim(-5, 80)
    axis.set_xticks(x_values, [str(value) for value in x_values])
    axis.set_ylim(*_curve_ylim(data))
    axis.set_xlabel('Clinical fields withheld (%)')
    axis.set_ylabel('Macro AUROC (absolute bootstrap 95% CI)')
    axis.grid(axis='y', color='#e4e9eb', linewidth=.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    legend = [
        Line2D([0], [0], color=ARM_COLORS[role], marker='o', linewidth=1.8, label=label)
        for role, label in (('S', 'R7 (S)'), ('V5', 'V5'), ('C', 'C'))
    ]
    axis.legend(handles=legend, frameon=False, ncol=3, loc='upper right')
    fig.suptitle('R7 screening under controlled dropout', x=.10, y=.965, ha='left',
                 fontsize=14, weight='bold')
    fig.text(
        .10, .025,
        'Absolute bootstrap 95% CIs for each arm. 0% means existing observations, not complete fields.\n'
        'Descriptive view of reused development data and fixed fit; controlled dropout differs from natural missingness.',
        ha='left', va='bottom', fontsize=8.2, color='#39474e',
    )
    fig.subplots_adjust(left=.11, right=.98, top=.89, bottom=.21)
    return fig


def _forest_rows(data):
    rows = [('Primary summary', data['primary_summary'], 'S_minus_V5', True)]
    rows.extend(
        (LABELS[pattern], data['primary_pattern_cells'][pattern], 'S_minus_V5', False)
        for pattern in PRIMARY_PATTERN_ORDER
    )
    descriptive_labels = {
        'available': 'Existing observations (descriptive)',
        'whole_cbc_hidden': LABELS['whole_cbc_hidden'],
        'whole_cbc_no_retina': LABELS['whole_cbc_no_retina'],
    }
    rows.extend(
        (descriptive_labels[pattern], data['profiles'][pattern]['macro'], 'S_minus_V5', False)
        for pattern in DESCRIPTIVE_FOREST_ORDER
    )
    rows = [
        (f'{label} (unsupported)' if cell is None else label, cell, contrast_name, is_summary)
        for label, cell, contrast_name, is_summary in rows
    ]
    return rows


def _forest_xlim(rows):
    bounds = []
    for _, cell, contrast_name, _ in rows:
        if cell is not None:
            contrast = cell['contrasts'][contrast_name]
            bounds.extend(contrast['ci95'])
            bounds.append(contrast['delta'])
    extent = max((max(abs(item) for item in bounds) if bounds else 0.05), 0.025)
    pad = max(extent * .12, .008)
    x_limit = min(1.0, extent + pad)
    return -x_limit, x_limit


def _draw_forest(data):
    rows = _forest_rows(data)
    fig, axis = plt.subplots(figsize=(10.8, 7.4), constrained_layout=False)
    if rows:
        axis.axhspan(-.5, .5, color='#e8f3f2', zorder=0)
    axis.axvline(0, color='#35434a', linewidth=1, zorder=1)
    for index, (label, cell, contrast_name, is_summary) in enumerate(rows):
        if cell is None:
            continue
        contrast = cell['contrasts'][contrast_name]
        lo, hi = contrast['ci95']
        delta = contrast['delta']
        color = '#075f64' if is_summary else '#087f83'
        axis.hlines(index, lo, hi, color=color, linewidth=2.0 if is_summary else 1.55, zorder=2)
        axis.vlines((lo, hi), index - .09, index + .09, color=color, linewidth=1.15, zorder=2)
        axis.scatter(delta, index, color=color, marker='D' if is_summary else 'o',
                     s=43 if is_summary else 30, edgecolor='white', linewidth=.5, zorder=3)
    axis.set_yticks(range(len(rows)), [label for label, _, _, _ in rows])
    axis.set_ylim(len(rows) - .5, -.5)
    axis.set_xlim(*_forest_xlim(rows))
    axis.set_xlabel('S (R7) − V5 macro AUROC difference [paired marginal 95% CI]')
    axis.grid(axis='x', color='#e4e9eb', linewidth=.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.suptitle('Paired R7−V5 differences across fixed profiles', x=.08, y=.965, ha='left',
                 fontsize=14, weight='bold')
    fig.text(
        .08, .025,
        'Primary summary highlighted. Typed-age input is retained for the clinical-physiology-hidden profile.\n'
        'Marginal paired 95% CIs, not multiplicity adjusted; prespecified evaluation, reused data and fixed fit.',
        ha='left', va='bottom', fontsize=8.2, color='#39474e',
    )
    fig.subplots_adjust(left=.43, right=.98, top=.89, bottom=.21)
    return fig


def render(data: dict, outputdir) -> dict:
    """Write editable-text SVG, PNG, and PDF plot outputs into outputdir."""
    validate_payload(data)
    style = {
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'svg.fonttype': 'none',
        'svg.hashsalt': SCHEMA,
    }
    directory = Path(outputdir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError('bran_r7_missingness_output_directory_not_empty')
    metadata = {
        'svg': {'Date': None, 'Creator': SCHEMA},
        'png': {'Software': SCHEMA},
        'pdf': {
            'CreationDate': None,
            'ModDate': None,
            'Creator': SCHEMA,
            'Title': 'R7 missingness stress test',
        },
    }
    with plt.rc_context(style):
        figures = {
            'curve': _draw_curve(data),
            'forest': _draw_forest(data),
        }
        outputs = {}
        try:
            for figure_name, figure in figures.items():
                paths = {}
                for extension in ('svg', 'png', 'pdf'):
                    path = directory / f'{OUTPUT_STEM}_{figure_name}.{extension}'
                    figure.savefig(path, format=extension, dpi=220, facecolor='white',
                                   edgecolor='white', transparent=False,
                                   metadata=metadata[extension])
                    paths[extension] = str(path)
                outputs[figure_name] = paths
        finally:
            for figure in figures.values():
                plt.close(figure)
    return outputs

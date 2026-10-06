# BRAN historical research reproduction code

This directory preserves the original Python source for retained R7 training and evaluation and for the analyses mapped below. It does not yet hold the code of most later analyses whose aggregates feed the paper figures. [Source Data and analysis code](source_data_code_coverage.md) maps every Source Data row to its analysis and marks those analyses PLACEHOLDER. The `research/` files are byte-identical copies of the reviewed workspace modules. They are not a runnable public dataset or a fitted model. No patient data, source tables, image files, weights, fold transforms, private caches, receipt JSON, or performance aggregates are included.

The source archive contains 425 Python modules and 5,580,403 source bytes. `research/snapshot_manifest.json` records each original code hash, byte size, local import edge, and entrypoint dependency set. `tools/reviewed_sources.txt` is the explicit hash allowlist. The collector parses source with Python `ast` and never imports a research runner. The original code has not been edited to remove historical paths or authentication gates.

## Levels of reproduction

| Level | What a reader needs | What it shows |
| --- | --- | --- |
| Software | This repository | The model code runs, synthetic tests pass and the research snapshot matches its reviewed hashes |
| Figures | This repository and the pinned figure environment | 9 of the 12 finalized items are redrawn from their Source Data, byte for byte only in the recorded reference environment. Fig. 3 and Extended Data Figs. 3 and 4 are blocked until a new export. Every plotted value can be read in the Source Data |
| Aggregates | Restricted data access and the private workspace | Each Source Data value equals the value in a pinned aggregate file with a recorded SHA-256 |
| Analyses | Restricted data access, the original environment and private artifacts | The aggregates of the analyses mapped below are recomputed from the datasets with the code in `research/`. PLACEHOLDER for the code of the other analyses behind the figures, listed in [Source Data and analysis code](source_data_code_coverage.md) |

## Reproduce the figures from Source Data

The figure level needs no restricted data. From the repository root run

```sh
python3.12 -m venv .venv-figures
.venv-figures/bin/python -m pip install --require-hashes -r paper/figures/requirements-figures.txt
make figures-check PYTHON=.venv-figures/bin/python
```

The command draws Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2 from `paper/source_data` alone and compares the PNG, PDF and JPEG hashes with the reference. It reports Fig. 3 and Extended Data Figs. 3 and 4 as blocked, and it ends with exit code 2 while any item is pending in the reference. [The figure guide](../paper/README.md) explains the steps from aggregates to figures, the reference environment, the verification record and the items that still need a new export.

The step from aggregates to Source Data ran in the private workspace. Each private builder read every plotted value from a pinned aggregate file, checked the value and the file's SHA-256, applied the disclosure rules and wrote the Source Data. Every Source Data row names its aggregate file and SHA-256, so the authors can show the origin of any value on request. The aggregates stay private because they are derived from data under the PhysioNet and AI-READI agreements.

## Verify the code snapshot

Run these commands from the release root. They inspect code only.

```sh
python tools/research_snapshot.py verify
python -m unittest discover -s tests -v
```

The byte check authenticates this snapshot against the reviewed source hashes. It does not authenticate missing restricted data or historical experiment receipts. A fresh public clone cannot reproduce numerical results without those separately governed inputs.

## Entry points and dependency map

The manifest gives every edge. The generated [research index](research_index.md) maps each module to the analyses that reach it, its role and its other versions, and explains the legacy name prefixes. The following counts include each entrypoint and its transitive static local Python imports. Shared modules appear in more than one count.

| Analysis | Historical entrypoint | Local modules |
| --- | --- | --- |
| R7 retained training | `run_bran_robust_clinical_r7.py` | 226 |
| R7 native and historical evaluation | `run_bran_robust_clinical_evaluation_r7.py` | 227 |
| A3 modality atlas | `run_bran_r7_modality_atlas_a3.py` | 233 |
| F1 information-matched comparison | `run_bran_r7_information_matched_f1.py` | 262 |
| M2 MIMIC state | `run_bran_mimic_broad_state_m2.py` | 359 |
| M2 missingness figure renderer | `bran_r7_missingness_figure_m2.py` | 1 |
| U2 completion utility | `run_bran_r7_completion_utility_u2.py` | 236 |
| Q4 quantile analysis | `run_bran_r7_quantile_q4.py` | 245 |
| S4 clinical panel | `run_bran_r7_clinical_s4.py` | 375 |
| S7 expanded clinical panel | `run_bran_expanded_clinical_s7.py` | 378 |
| INSPIRE E1 | `run_bran_r7_inspire_e1.py` | 239 |
| HiRID H6 | `run_bran_hirid_r7_h6.py` | 245 |

The M2 MIMIC state runner prepares a source-specific state bridge. It is not the missingness robustness evaluator. The retained robustness measurements come from `run_bran_robust_clinical_evaluation_r7.py`, while `bran_r7_missingness_figure_m2.py` only renders an approved aggregate.

The retained R7 code sets the bridge screening and CBC coefficients to zero, applies `3 * asinh(z / 3)` only at the encoder input, and leaves teacher and state preservation terms unchanged. Its fit uses 3,000 updates per fold, learning rate `5e-5`, paired batch size 96, and source batch size 128. The runner first authenticates archived matched V6 controls and warm-start V5 checkpoints. This is the implemented recipe, not a new proposed training configuration.

## Historical invocation syntax

These commands show the interfaces in the preserved files. Do not run them against public or synthetic data as though they reproduced the retained results. The runners expect authenticated private source layouts, prior terminals, restricted weights, and exact hashes. Their exclusive output rules can also refuse a second attempt at an existing path.

```sh
python research/run_bran_robust_clinical_r7.py --stage pilot --attempt 1
python research/run_bran_robust_clinical_r7.py --stage fit --attempt 1
python research/run_bran_robust_clinical_evaluation_r7.py --stage native --attempt 1 --fit-attempt 1
python research/run_bran_robust_clinical_evaluation_r7.py --stage historical --attempt 1 --fit-attempt 1

python research/run_bran_r7_modality_atlas_a3.py --attempt 1
python research/run_bran_r7_modality_atlas_a3.py --attempt 1 --audit
python research/run_bran_r7_information_matched_f1.py --attempt 1 --run
python research/run_bran_r7_information_matched_f1.py --attempt 1 --audit
python research/run_bran_r7_information_matched_f1.py --attempt 1 --replay
python research/run_bran_mimic_broad_state_m2.py
python research/run_bran_r7_completion_utility_u2.py
python research/run_bran_r7_completion_utility_u2.py --replay
python research/run_bran_r7_quantile_q4.py pilot
python research/run_bran_r7_quantile_q4.py fit
python research/run_bran_r7_quantile_q4.py evaluate
python research/run_bran_r7_clinical_s4.py
python research/run_bran_r7_clinical_s4.py --audit-only
python research/run_bran_expanded_clinical_s7.py
python research/run_bran_expanded_clinical_s7.py --audit-only
python research/run_bran_r7_inspire_e1.py
python research/run_bran_r7_inspire_e1.py --audit-only
python research/run_bran_hirid_r7_h6.py --attempt 1
python research/run_bran_hirid_r7_h6.py --attempt 1 --audit
```

The M2 missingness file is a renderer, not a command-line runner. Its `render(data, outputdir)` function requires a disclosure-approved aggregate payload and writes editable SVG, PNG, and PDF files. This release includes no such payload or rendered patient-derived figure.

## Restricted inputs and unresolved dependencies

The training and evaluation runners require the original authorized source binding, fold assignments, normalized native transforms, V5 parent checkpoints, archived matched V6 control checkpoints, and the corresponding authenticated pilot and fit receipts. The `bran_multisource_data_v2.py` and `bran_multisource_native_transform_v3.py` source modules are present, but the fitted data and transforms are not. The clinical and external analyses require their own source tables, mappings, permissions, prior terminal receipts, frozen R7 checkpoints, and source-specific adapters. INSPIRE and HiRID are separate restricted routes. No model evaluation result can be derived from this code snapshot alone.

The static import graph also names more data sources than the reported analyses used. The retained R7 training entrypoint reaches modules that reference exploratory sources such as FD3611, TriHemo-MCV, KNHANES, ODIR and Zigong. Do not infer the training or evaluation data of an analysis from its imports. The source list and its PLACEHOLDER roles are in [data, code and model availability](data_and_model_availability.md).

Static imports in the 425-file archive reference third-party packages named `PIL`, `joblib`, `matplotlib`, `numpy`, `pandas`, `plotly`, `pyarrow`, `pydicom`, `safetensors`, `scipy`, `sklearn`, `threadpoolctl`, `timm`, `torch`, and `torchvision`. This is an import inventory, not a version-locked environment. The package root has a smaller install requirement for its reusable model facade. Reproducing the historical pipelines needs the original approved environment and restricted dependency versions.

The AST graph leaves six variable-driven import sites unresolved across four modules. They occur in `audit_bran_september_push_v1.py` at line 50, `bran_agefree_unified_source_v1.py` at line 39, `run_bran_joint_lab_comparison_v1.py` at lines 96 and 155, and `run_bran_native_source_qualification_v1.py` at lines 51 and 68. The imported module names depend on runtime arguments or source-specification dictionaries. The manifest records these unresolved sites, so the static snapshot must not be described as a complete executable closure for those branches.

The code-only scan found no credential literal, hardcoded patient-key comparison, or large numeric literal sequence under its declared AST checks. That scan is not a privacy certification. Historical source paths and protocol constants remain in the unmodified files. The owner approved the MIT license for the code. Trained-weight redistribution is not part of this release.

# BRAN

BRAN learns a reusable patient representation from blood measurements, other eligible clinical measurements, typed age and retinal features. The representation has 192 dimensions and supports disease detection, CBC completion and downstream clinical analyses under incomplete observation.

This repository contains code and the Source Data of the paper figures, not patient data or trained weights. It separates a small usable model interface from the original research implementation. The retained implementation is R7. Historical names remain inside the research code where they identify dependencies used in the study.

## Contents

| Path | Contents |
| --- | --- |
| `src/` | Six unchanged architecture modules and the checked inference interface |
| `schemas/` | The documented field and output contracts |
| `research/` | Original training and evaluation code with dependency and hash records, mapped in [the research index](docs/research_index.md) |
| `paper/source_data/` | Source Data of the paper figures, one workbook and one JSON file per item, with a hash manifest |
| `paper/figures/` | Code that redraws every finalized figure from its Source Data and checks the bytes |
| `examples/` | Generated-input demonstrations |
| `tests/` | Synthetic interface, schema, release, Source Data and figure checks |
| `scripts/verify_release.py` | The file policy check, with core source and Source Data hashes |
| `tools/` | Research snapshot verification and the research index generator |
| `SOURCE_MANIFEST.json` | Source hashes and R7 implementation provenance of release 0.2.0 |

## Implementation

The included implementation preserves the retained R7 architecture. R7 applies `3 * asinh(x / 3)` to sanitized clinical values before both inherited clinical encoder paths. It leaves observation masks, typed-age representation, target heads, and the 192-dimensional state interface unchanged. This paragraph records implementation provenance and does not claim clinical benefit or reproduce a fitted model.

## Quick start

Use PyTorch 2.6 or newer. Continuous integration tests Python 3.11, 3.12 and 3.13 on CPU, and runs the examples against the installed package. The package metadata still allows Python 3.10, which has not been tested. GPU hardware is not required for the examples.

```sh
git clone https://github.com/ethanwu2011/BRAN.git
cd BRAN
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test,figures]'
python -m unittest discover -s tests -v
python examples/synthetic_encode.py
python examples/synthetic_end_to_end.py
```

The examples use generated tensors. The end-to-end example demonstrates model fitting, checkpoint serialization, reload and inference on synthetic inputs. Its small optimization loop is not the study training recipe and does not recreate a pretrained BRAN model.

The model accepts 59 clinical values with observation masks, 384-dimensional retinal features with visibility masks, and a seven-coordinate typed-age input. Forty-three continuous fields are eligible and eleven history slots are disabled. Native heads provide 26 recorded-condition outputs and nine CBC estimates. The shared and private state blocks are architectural components, not established biological compartments.

Completion removes the target value and observation flag before inferring the state. With no physiological input the model abstains, even if age is available. Retinal feature reconstruction is not retinal image generation. Read the [input contract](docs/input_contract.md) and [inference guide](docs/inference.md) before using authorized local data.

## Reproduce the paper figures

Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2 are redrawn from their Source Data with one command. It needs no restricted data.

```sh
python3.12 -m venv .venv-figures
.venv-figures/bin/python -m pip install --require-hashes -r paper/figures/requirements-figures.txt
make figures-check PYTHON=.venv-figures/bin/python
```

The command writes the PNG, PDF and JPEG of every item into a fresh folder and compares their SHA-256 with the reference. Byte identity holds in the reference environment recorded in `paper/figures/REFERENCE_OUTPUTS.json`. Five items still need a new export before release. The [figure guide](paper/README.md) gives the verification record, the reasons and the path from aggregates to figures.

## Reproducing the research

The [research reproduction guide](docs/research_reproduction.md) sets out four levels, from software checks and figures that anyone can run, to aggregates and analyses that need restricted data. It maps the original entrypoints and their dependencies and distinguishes the training recipe, native evaluation, fixed-readout comparison, completion and clinical analyses. Research runners retain their source and fold checks. Do not disable those checks to make an incompatible workspace run.

Exact local package versions of the research environment are listed in `requirements-research.txt`. This is an environment inventory rather than a universal lockfile. No fresh clinical experiment was run to prepare this release.

## Data, code and model availability

No patient records, images, notes, identifiers, embeddings, predictions, fitted transforms or weights are distributed here. The Source Data hold disclosure-checked aggregates only. Keep real inputs and outputs in a separate restricted workspace. Git ignore rules and automated scans supplement manual review but do not prove privacy.

Dataset access is obtained from providers. Trained-parameter distribution has not been authorized. Constructing a model with random weights will not reproduce the paper's results. See [data, code and model availability](docs/data_and_model_availability.md) for the draft Data availability and Code availability statements, the role of each source, what stays private and why, and the release plan. See [data and access](docs/privacy_and_access.md) for the repository privacy rules.

The learned state is a research representation. It is not a calibrated posterior, biological compartment, or individual clinical recommendation. Synthetic tests verify software behavior only.

## Release status

This is a research release, not a clinical product. The code is available under the [MIT license](LICENSE), approved by the owner. The four BioRender icons embedded in the Fig. 1 Source Data are not covered by that license, as noted in `paper/source_data/README.md`. Version 1.0.0 will be tagged and archived at Zenodo with a DOI for the paper, following the release plan in the availability page and the [release checklist](docs/release_checklist.md). Changes since 0.2.0 are listed in the [changelog](CHANGELOG.md).

Core source hashes are recorded in `SOURCE_MANIFEST.json`. Architecture files remain unchanged so implementation provenance can be checked independently of the new user-facing interface. The [validation record](docs/validation.md) describes the checks of release 0.2.0.

## Citation

Cite the software with the metadata in `CITATION.cff`. The archived DOI will be added at release.

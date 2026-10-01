# BRAN

BRAN learns a reusable patient representation from blood measurements, other eligible clinical measurements, typed age and retinal features. The representation has 192 dimensions and supports disease detection, CBC completion and downstream clinical analyses under incomplete observation.

This repository contains code, not patient data or trained weights. It separates a small usable model interface from the original research implementation. The retained implementation is R7. Historical names remain inside the research code where they identify dependencies used in the study.

## Implementation

The included implementation preserves the retained R7 architecture. R7 applies `3 * asinh(x / 3)` to sanitized clinical values before both inherited clinical encoder paths. It leaves observation masks, typed-age representation, target heads, and the 192-dimensional state interface unchanged. This paragraph records implementation provenance and does not claim clinical benefit or reproduce a fitted model.

## Quick start

Use Python 3.10 or newer and PyTorch 2.6 or newer. The development checks use Python 3.12 on CPU. GPU hardware is not required for the examples.

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

## Included files

- `src/` contains six unchanged architecture modules and the checked inference interface.
- `schemas/` contains the documented field and output contracts.
- `research/` contains original training and evaluation code with dependency and hash records.
- `paper/` contains the renderer for locally supplied figure specifications.
- `examples/` contains generated-input demonstrations.
- `tests/` contains synthetic interface, schema and rendering checks.
- `pyproject.toml` and `requirements.txt` describe the installable package and runtime dependency.
- `SOURCE_MANIFEST.json` records source hashes and R7 implementation provenance.
- `scripts/verify_release.py` checks the code-only file policy and core hashes.
- `RIGHTS_STATUS.md` records the licensing status.

## Reproducing the research

There are three separate levels of use.

1. Run synthetic examples and tests without clinical data.
2. Rerender locally supplied paper figure specifications with [the figure tool](paper/README.md).
3. Reproduce the scientific analyses using the original code, authorized datasets, compatible retinal features, fitted preprocessing, checkpoints and source authentication artifacts.

The [research reproduction guide](docs/research_reproduction.md) maps the original entrypoints and their dependencies. It distinguishes the actual training recipe, native evaluation, fixed-readout comparison, completion and clinical analyses. Research runners retain their source and fold checks. Do not disable those checks to make an incompatible workspace run.

Exact local package versions are listed in `requirements-research.txt`. This is an environment inventory rather than a universal lockfile. No fresh clinical experiment was run to prepare this release.

## Data and weights

No patient records, images, notes, identifiers, embeddings, predictions, fitted transforms, weights or research-result tables are distributed here. Keep real inputs and outputs in a separate restricted workspace. Git ignore rules and automated scans supplement manual review but do not prove privacy.

Dataset access is obtained from providers. Trained-parameter distribution has not been authorized for this release. Constructing a model with random weights will not reproduce the paper's results. See [data and access](docs/privacy_and_access.md).

The learned state is a research representation. It is not a calibrated posterior, biological compartment, or individual clinical recommendation. Synthetic tests verify software behavior only.

## Release status

This is a code-only research release, not a clinical product or an independently reproduced result package. The code is available under the [MIT license](LICENSE), approved by the owner. Final manuscript citation metadata, trained-parameter access and a persistent archival identifier are tracked in the [release checklist](docs/release_checklist.md).

Core source hashes are recorded in `SOURCE_MANIFEST.json`. Architecture files remain unchanged so implementation provenance can be checked independently of the new user-facing interface.

The [validation record](docs/validation.md) summarizes the 39 passing tests, isolated installation check, archived source verification and figure-render comparison, with their limits.

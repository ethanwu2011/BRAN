# BRAN reusable representation

BRAN represents an encounter with a 192-dimensional state inferred from typed age, eligible clinical measurements and masks, and visible retinal embeddings. Its state includes shared, retinal-private, and clinical-private coordinates. The API exposes the architecture and input contracts without assuming a field dictionary or distributing fitted parameters.

The source package is for research and software reuse. It contains no patient records, model weights, fold-fitted transforms, source authentication, or clinical evaluation pipeline. Callers must provide their own authorized field map and training-fitted preprocessing. The example uses an explicitly fake index map and generated tensors only.

## Implementation

The included implementation preserves the retained R7 architecture. R7 applies `3 * asinh(x / 3)` to sanitized clinical values before both inherited clinical encoder paths. It leaves observation masks, typed-age representation, target heads, and the 192-dimensional state interface unchanged. This paragraph records implementation provenance and does not claim clinical benefit or reproduce a fitted model.

## Install and test

The package requires Python 3.10 or newer and PyTorch 2.0 or newer. The local synthetic checks passed with Python 3.12.4 and PyTorch 2.12.1.

```sh
python -m pip install .
python -m unittest discover -s tests -v
python examples/synthetic_encode.py
```

The model accepts 59 clinical values with matching observation masks, 384-dimensional retinal embeddings with visibility masks, a seven-feature normalized typed-age matrix, and caller-supplied eligibility and CBC index maps. The architecture has 43 eligible slots among 48 continuous clinical fields, nine CBC slots, eleven disabled binary slots, and a 192-dimensional state. Index values in the example are fake positions chosen for shape checks. They are not a clinical field mapping.

## Included files

- `src/` contains six unmodified model and typed-age modules plus the `bran_r7_release` import facade.
- `examples/` contains one generated-tensor smoke example.
- `tests/` contains CPU-only synthetic contract checks.
- `pyproject.toml` and `requirements.txt` describe the installable package and runtime dependency.
- `SOURCE_MANIFEST.json` records source hashes and R7 implementation provenance.
- `RIGHTS_STATUS.md` records the unresolved licensing status.

## Limits and publication status

This package is not a standalone pretrained model and cannot reproduce retained-cohort predictions or reported results. The corresponding fold-specific weights and transforms are not included, and no permission to redistribute them was established by the local release materials. The source runner, source adapters, field dictionary, and outcome evaluation are also outside this package.

The learned state is a research representation. It is not a calibrated posterior, biological compartment, or individual clinical recommendation. Synthetic tests verify software behavior only.

The code is maintained in the private [BRAN repository](https://github.com/ethanwu2011/BRAN). No license is assigned. Public release and redistribution of trained parameters require separate ownership, permission and privacy review.

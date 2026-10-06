# Changelog

All notable changes to this repository are recorded here. Versions follow the package version in `pyproject.toml`.

## [Unreleased]

### Added

- `paper/source_data/` with the Source Data of Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2, one workbook and one JSON file per item, and `SOURCE_DATA_MANIFEST.json` with the SHA-256, display number and status of every file. Every file has the status final. Extended Data Table 3 belongs to the Figure 6 work and is held back until it is final.
- `paper/figures/` with the shared Nature Medicine style module, the display numbers and public drawing code that redraws the finalized items from their Source Data alone. `python -m paper.figures.build_all` rebuilds the items into a fresh folder and checks the PNG, PDF and JPEG hashes against `REFERENCE_OUTPUTS.json`. All 12 items rebuild byte for byte and match the files exported for submission. `LEGENDS.md` collects every legend, which is the submission legend the private builders take from one module.
- `python -m paper.figures.disclosure`, which checks that every count and printed bound in the Source Data is at least 20, that no table lists people one by one and that no unreviewed proportion is an exact fraction of a hidden count. It scans the draw section of each file as well as its panels and lists any open disclosure item that blocks release. None is open.
- `tools/source_data_coverage.py` and the generated `docs/source_data_code_coverage.md`, which map every Source Data row to the analysis that produced its aggregate and show whether that analysis's code is in `research/`, with a freshness test.
- A hashed lock of the figure environment in `paper/figures/requirements-figures.txt`, a `Makefile` with test, release and figure targets, a CI job that checks the Source Data and fails when an item that is not blocked does not rebuild, and a CI job that rebuilds every item twice with a pinned Chrome for Testing build and compares the hashes.
- Tests of Source Data identity, the disclosure rules, the public code and byte-identical figure rebuilds.
- `docs/data_and_model_availability.md` with draft Data availability and Code availability statements in the form Nature Medicine prints them, the role of each data source, what stays private and why, the PhysioNet and AI-READI terms that restrict sharing and the release plan.
- `tools/research_index.py` and the generated `docs/research_index.md`, a map of the research snapshot by analysis, role and version family, with a freshness test.
- An absolute local path scan in `scripts/verify_release.py`. Paths outside `research/` are rejected. The 59 historical occurrences in 26 research modules are reported and documented in `docs/privacy_and_access.md`.
- Continuous integration on Python 3.11, 3.12 and 3.13, with the examples run against the installed package.

### Changed

- `scripts/verify_release.py` admits Source Data workbooks only when the manifest lists them with a matching SHA-256, and scans their text for credentials and absolute paths.
- README, `paper/README.md`, the research reproduction guide and the release checklist describe the figure layer. `paper/render_figures.py` is kept as the legacy renderer of the 0.2.0 scene format.
- Packaging metadata uses the SPDX license expression, requires setuptools 77 or newer and lists authors, keywords, classifiers and project URLs.
- NumPy is a runtime dependency, so a bare install no longer makes torch report that NumPy is missing.
- The schema provenance records `research/bran_clinical_semantics_v1.py` as bundled and rechecks its hash.

### Fixed

- `tools/research_snapshot.py verify` no longer fails when `research/__pycache__` exists.

## [0.2.0] - 2026-10-01

### Added

- The original research code of retained R7 training and evaluation and of the analyses in its entrypoint map as a 425-module hash-locked snapshot.
- A checked inference interface with target erasure, abstention without physiological input and tensor-only checkpoints.
- The documented clinical input contract, generated-input examples and synthetic tests.
- A renderer for locally supplied figure scenes, MIT licensing and software citation metadata.

The sealed 0.2.0 record is in `docs/release_notes.md`, `docs/validation.md` and `SOURCE_MANIFEST.json`.

[Unreleased]: https://github.com/ethanwu2011/BRAN/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ethanwu2011/BRAN/releases/tag/v0.2.0

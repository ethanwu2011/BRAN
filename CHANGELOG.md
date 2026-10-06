# Changelog

All notable changes to this repository are recorded here. Versions follow the package version in `pyproject.toml`.

## [Unreleased]

### Added

- `docs/data_and_model_availability.md` with draft code, data and model statements, the sources the research code references and the PhysioNet and AI-READI terms that restrict sharing.
- `tools/research_index.py` and the generated `docs/research_index.md`, a map of the research snapshot by analysis, role and version family, with a freshness test.
- An absolute local path scan in `scripts/verify_release.py`. Paths outside `research/` are rejected. The 59 historical occurrences in 26 research modules are reported and documented in `docs/privacy_and_access.md`.
- Continuous integration on Python 3.11, 3.12 and 3.13, with the examples run against the installed package.

### Changed

- Packaging metadata uses the SPDX license expression, requires setuptools 77 or newer and lists authors, keywords, classifiers and project URLs.
- NumPy is a runtime dependency, so a bare install no longer makes torch report that NumPy is missing.
- The schema provenance records `research/bran_clinical_semantics_v1.py` as bundled and rechecks its hash.

### Fixed

- `tools/research_snapshot.py verify` no longer fails when `research/__pycache__` exists.

## [0.2.0] - 2026-10-01

### Added

- The original research code for retained R7 training and the reported analyses as a 425-module hash-locked snapshot with an entrypoint map.
- A checked inference interface with target erasure, abstention without physiological input and tensor-only checkpoints.
- The documented clinical input contract, generated-input examples and synthetic tests.
- A renderer for locally supplied figure scenes, MIT licensing and software citation metadata.

The sealed 0.2.0 record is in `docs/release_notes.md`, `docs/validation.md` and `SOURCE_MANIFEST.json`.

[Unreleased]: https://github.com/ethanwu2011/BRAN/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ethanwu2011/BRAN/releases/tag/v0.2.0

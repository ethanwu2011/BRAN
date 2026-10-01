# Release validation

These checks were completed for software release 0.2.0 on 1 October 2026.

## Executable checks

- All 39 tests passed from a Git tree exported outside the research workspace.
- The package built and installed as a wheel. Both generated-input examples passed using the installed package from outside the source directory.
- The tests covered target erasure, hidden-payload invariance, all-empty abstention, checkpoint bindings, checkpoint resource limits, schema consistency, the original AUROC calculation and synthetic figure rendering.
- Independent scikit-learn calculations agreed with the original AUROC functions on generated inputs, including tied scores and weighted draws.
- All 425 archived research modules parsed and matched their recorded SHA-256 hashes. Verification did not require the original research workspace.
- The portable renderer reproduced all six retained figure PNGs pixel for pixel in a local comparison. The figure inputs and outputs remain outside this repository.

## Distribution checks

The explicit release file list was reviewed and checked for disallowed data and binary formats, symlinks, selected credential patterns and core source hash changes. No findings remained under those checks. Repository history was also reviewed for previously committed data artifacts.

No patient data, trained weights, fitted transforms, research-result tables or real figure specifications are distributed. Automated checks supplement review but do not establish a privacy guarantee for future additions.

## Scope

The isolated checks reused the installed Python 3.12 dependency environment. They were not a fresh operating-system installation or a rerun of the clinical studies. The GitHub workflow separately runs synthetic checks on Linux. Its current status is available in the repository Actions tab.

The archived research implementation retains private artifact dependencies and six variable-driven import sites across four modules. See the research reproduction guide for those limits. Passing this suite does not establish clinical validity, numerical reproduction of published results or permission to distribute trained parameters.

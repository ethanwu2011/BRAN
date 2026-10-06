# Publication release checklist

This checklist distinguishes executable code from permission to distribute research artifacts.

## Code checks

- Verify the unchanged architecture hashes.
- Run the generated-input unit tests and end-to-end demonstration.
- Check the real field schema against code definitions without inspecting participant data.
- Preserve the original training and evaluation methods in the research snapshot.
- Identify missing dependencies rather than substituting a new estimator.
- Scan staged files and review the exact upload allowlist with `python scripts/verify_release.py`.
- Check the installed package from outside its source directory.
- Verify the research snapshot, the research index and the map from Source Data to analysis code with `make release-check`.

## Figure and Source Data checks

- Run `python -m paper.figures.disclosure` and confirm that it exits 0, with no failed rule and no open item.
- Review the reasons in `REVIEWED_RATIOS` of `paper/figures/disclosure.py` for every exact fraction it accepts.
- Run `make figures-check` in the reference environment and confirm that every item is reproduced and none is blocked or pending.
- Regenerate the reference with a Chrome for Testing build pinned by version and SHA-256 through `BROWSER_PATH`, and record the SHA-256 of the archive used by the pinned-browser CI job.
- Run `python tools/source_data_coverage.py --check` and confirm that no analysis behind the figures is still marked PLACEHOLDER, or that the Code availability statement says which are missing.
- Confirm that each legend in `paper/figures/LEGENDS.md` matches the manuscript and uses the display numbers.
- Upload each Source Data workbook with its figure, and each Extended Data item as `Surname_EDfigN.jpg`.

## Author decisions

- MIT license approved by the owner on 1 October 2026.
- Confirm copyright and author metadata before issuing an archival citation.
- Confirm that the BioRender plan allows the four icons in the Fig. 1 Source Data to be redistributed in a public repository, before the first push. If it does not, rewrite the branch to remove them and redraw Fig. 1.
- Decide the licence of the Source Data in this repository (`RIGHTS_STATUS.md`).
- State whether trained parameters can be shared separately and under which conditions.
- Approve the final data and code availability statements in `docs/data_and_model_availability.md`.
- Decide whether to keep Python 3.10 in `requires-python` or raise it to 3.11.
- Archive the exact manuscript code version as release v1.0.0 with a Zenodo DOI, following the release plan.

The synthetic demonstration, code snapshot and figure rebuild do not establish a fresh reproduction of the clinical results. Reproduction on restricted inputs is a separate activity. No real-data study is rerun as part of this packaging task.

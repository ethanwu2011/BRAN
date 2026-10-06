# Source Data

This folder holds the Source Data for Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2 of the BRAN paper. Extended Data Table 3 belongs to the Figure 6 work and is held back until that work is final.

Each item has two files with the same content.

- `<id>_source_data.xlsx` is the Source Data file for the journal, with an About sheet and one sheet per panel.
- `<id>_source_data.json` is the machine-readable copy and the only input of the drawing code in `paper/figures`.

`SOURCE_DATA_MANIFEST.json` lists every file with its display number, status, SHA-256 and size. `python scripts/verify_release.py` fails if a file is missing, unlisted or changed. The status is final for every item except Fig. 3 and Extended Data Figs. 3 and 4. Their Source Data are complete, but their status is pending because their drawing code cannot yet run under the final style module, as `status_reason` explains.

## What a row holds

A row gives the plotted value with its interval, bound, test result, floored count and label, and names the aggregate file it was read from with that file's SHA-256. Those paths are relative to the private project folder. The aggregates are patient-derived summaries that stay under the data use agreements of PhysioNet and AI-READI and are not distributed. The hashes let the authors show which aggregate produced each value. [Source Data and analysis code](../../docs/source_data_code_coverage.md) groups these files by the analysis that wrote them and says whether its code is in `research/`.

Some files also name the private builder that wrote them, under `builder`, with its SHA-256. That path is provenance like the aggregate paths and does not exist in this repository. `paper/figures/PROVENANCE.json` maps each builder to the public drawing code cut from it. The `rebuild_command` field gives the public command, `python -m paper.figures.build_all --only <id>`.

The JSON and workbook files are the files the private builders exported, with two changes. Absolute paths into the private project folder are written relative to that folder, and the private rebuild command is replaced by the public one. The manifest records the SHA-256 of each exported file.

## Disclosure

Every value is an aggregate. Every declared count and every printed lower bound is at least 20. Counts of people in an analysed sample are floored to a multiple of 20 unless the column, a count-type column or the panel title marks them exact, and KNHANES counts are rounded to 10 by their release. Exact counts are cohort, site, study-group and image totals, each at least 20. No row describes one person. No proportion may be an exact fraction of a count that is not printed, unless the disclosure check lists the cell as reviewed with its reason, for example a P value from 1,000 resamples. `python -m paper.figures.disclosure` checks these rules on every file, the draw section included.

## Open disclosure items

None. The three items found on 6 October 2026 were fixed by new exports of their private builders. Fig. 2 now prints the retrieval cohort as a lower bound floored to 20 with its proportions at 3 decimals, Fig. 5 releases the panel g HiRID errors and intervals at 3 decimals, and Extended Data Fig. 2 panel b has a count_type column. `python -m paper.figures.disclosure` exits 0 when no rule fails and no item is open.

## Internal ids

| File id | Display |
| --- | --- |
| `fig1` to `fig5` | Figs. 1 to 5 |
| `ed1`, `ed2`, `ed3` | Extended Data Figs. 1, 2 and 3 |
| `ed3b` | Extended Data Fig. 4 |
| `ed4` | Extended Data Fig. 5 |
| `ed_table1`, `ed_table2` | Extended Data Tables 1 and 2 |

## Licence note

The code in this repository is MIT licensed. The four icons embedded as base64 PNG in `fig1_source_data.json` (`draw.icons`) were drawn with BioRender and are not covered by that licence. PLACEHOLDER for confirmation that the BioRender plan allows these icon files to be redistributed in a public repository. Until it is confirmed the repository must not be made public. If it is not allowed, the icons must be removed from the branch history and Fig. 1 redrawn with openly licensed or self-drawn symbols. `RIGHTS_STATUS.md` records the rights of the Source Data and of third-party assets.

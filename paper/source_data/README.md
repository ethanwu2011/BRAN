# Source Data

This folder holds the Source Data for Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2 of the BRAN paper, and provisional Source Data for Extended Data Table 3.

Each item has two files with the same content.

- `<id>_source_data.xlsx` is the Source Data file for the journal, with an About sheet and one sheet per panel.
- `<id>_source_data.json` is the machine-readable copy and the only input of the drawing code in `paper/figures`.

`SOURCE_DATA_MANIFEST.json` lists every file with its display number, SHA-256 and size. `python scripts/verify_release.py` fails if a file is missing, unlisted or changed.

## What a row holds

A row gives the plotted value with its interval, bound, test result, floored count and label, and names the aggregate file it was read from with that file's SHA-256. Those paths are relative to the private project folder. The aggregates are patient-derived summaries that stay under the data use agreements of PhysioNet and AI-READI and are not distributed. The hashes let the authors show which aggregate produced each value.

## Disclosure

Every value is an aggregate. Every declared count and every printed lower bound is at least 20, counts are floored to a multiple of 20 unless the column says exact, and KNHANES counts are rounded to 10 by their release. No row describes one person. `python -m paper.figures.disclosure` checks these rules on every file.

## Internal ids

| File id | Display |
| --- | --- |
| `fig1` to `fig5` | Figs. 1 to 5 |
| `ed1`, `ed2`, `ed3` | Extended Data Figs. 1, 2 and 3 |
| `ed3b` | Extended Data Fig. 4 |
| `ed4` | Extended Data Fig. 5 |
| `ed_table1`, `ed_table2`, `ed_table3` | Extended Data Tables 1, 2 and 3 |

## Licence note

The code in this repository is MIT licensed. The four icons embedded as base64 PNG in `fig1_source_data.json` (`draw.icons`) were drawn with BioRender and are not covered by that licence. PLACEHOLDER for confirmation of the BioRender publication licence before the repository is made public.

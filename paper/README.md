# Paper figures and Source Data

This folder rebuilds the figures of the BRAN paper from their Source Data. It covers Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2. Fig. 6, Extended Data Figs. 6 and 7 and Extended Data Table 3 are still in preparation and are not rebuilt here.

## Layout

| Path | Contents |
| --- | --- |
| `source_data/` | One Excel workbook and one JSON file per item, `SOURCE_DATA_MANIFEST.json` with the SHA-256 and display number of each file, and a README |
| `figures/draw_<id>.py` | Drawing code of each item |
| `figures/bran_plotly_style.py` | Shared Nature Medicine style, export and Source Data helpers |
| `figures/display_numbers.py` | Display numbers used in every title and cross-reference |
| `figures/ed_canvas.py`, `figures/ed_nm_canvas.py`, `figures/fig3_nm_kit.py`, `figures/fig4_lane_kit.py` | Canvas and layout helpers |
| `figures/build_all.py` | The single command that rebuilds and checks every item |
| `figures/disclosure.py` | The disclosure check of the Source Data |
| `figures/REFERENCE_OUTPUTS.json` | Reference hashes of every PNG, PDF and JPEG, the reference environment and the code hashes |
| `figures/SUBMISSION_EXPORTS.json` | Hashes of the files exported for submission |
| `figures/PROVENANCE.json` | The private builder file behind each public code file |
| `figures/LEGENDS.md` | Every legend, generated from the Source Data |
| `figures/requirements-figures.in`, `figures/requirements-figures.txt` | The exact figure environment and its hashed lock |
| `render_figures.py` | Legacy renderer of the 0.2.0 scene format, kept for its tests |

## Rebuild the figures

The reference environment is macOS on Apple silicon with Python 3.12.4, Google Chrome 154.0.8037.98 and the Arial and Arial Bold files that ship with macOS. `figures/REFERENCE_OUTPUTS.json` records it, with the fonts identified by content hash.

```sh
python3.12 -m venv .venv-figures
.venv-figures/bin/python -m pip install --require-hashes -r paper/figures/requirements-figures.txt
make figures-check PYTHON=.venv-figures/bin/python
```

`make figures-check` runs `python -m paper.figures.build_all`. It draws every item in its own process from its Source Data JSON alone, writes the 600 dpi PNG, the PDF at final size and, for Extended Data items, the 300 ppi RGB JPEG into a new temporary folder, and compares every file hash with `figures/REFERENCE_OUTPUTS.json`. An audit hook stops the drawing process if it opens any other file in the repository. Use `make figures` to keep the files in `paper/output/figures`, which Git ignores.

In another environment the items still build, but the bytes can differ, mostly because of other font files or another Chrome version. `make figures-portable` reports those differences without failing. Set `BRAN_FONT_DIR` to a folder that holds `Arial.ttf` and `Arial Bold.ttf` to choose the font files, and `BROWSER_PATH` to choose the Chrome binary.

More commands follow.

```sh
python -m paper.figures.build_all --list          # internal ids and display numbers
python -m paper.figures.build_all --only fig2     # one item
python -m paper.figures.disclosure                # disclosure check of every Source Data file
python -m paper.figures.build_all --legends       # rewrite figures/LEGENDS.md from the Source Data
```

## From aggregates to figures

1. The analyses ran in the private research workspace on the restricted datasets and wrote aggregate files. Those aggregates hold no patient rows, but they are derived from data under the PhysioNet and AI-READI agreements and stay private.
2. A private builder for each item read every plotted value from pinned aggregate files, checked each value against its file and that file's SHA-256, applied the disclosure rules and wrote the Source Data. Each Source Data row names the aggregate file and its SHA-256.
3. The public drawing code in `figures/` reads only the Source Data JSON. It is cut verbatim from the private builder and keeps the top-level statements that the drawing entry point reaches. `figures/PROVENANCE.json` gives the builder file and its SHA-256 for each public file.
4. `figures/build_all.py` renders each item and checks the bytes.

A reader can repeat step 4 exactly and inspect every value of step 2 in the Source Data without access to the restricted data.

## Verification record, 6 October 2026

`python -m paper.figures.build_all` was run in a fresh virtual environment installed from `figures/requirements-figures.txt` on macOS 26.5.2 with Python 3.12.4 and Chrome 154.0.8037.98. Every item that builds reproduced the reference hashes of its PNG, PDF and JPEG, and each drawing process opened only its own Source Data file in the repository. The reference build compared each item with the file exported for submission.

| Item | Status | Detail |
| --- | --- | --- |
| Fig. 1 | Reproduced | PNG and PDF identical to the submission export |
| Fig. 2 | Reproduced | PNG and PDF identical |
| Fig. 3 | Blocked | See below |
| Fig. 4 | Reproduced | PNG and PDF identical |
| Fig. 5 | Differs from the submission export | See below |
| Extended Data Fig. 1 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Fig. 2 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Fig. 3 | Blocked | See below |
| Extended Data Fig. 4 | Blocked | See below |
| Extended Data Fig. 5 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Table 1 | PNG reproduced, PDF differs | PNG and JPEG identical, see below |
| Extended Data Table 2 | Reproduced | PNG, PDF and JPEG identical |

Five items need a new export by their figure builders before release.

- Fig. 3 and Extended Data Figs. 3 and 4 were exported with an earlier version of the style module. Their layout kit requires panel letters of exactly 8.0 pt, while the final style module stores 8.0025 pt, the nearest size that Chrome keeps in a PDF. Their drawing code therefore stops under the final style module. Their submission PDFs were also printed with the earlier page margins, which shrink all text by 2.65%.
- Fig. 5 was exported with the same earlier style module. Under the final module it builds deterministically, but its text sizes differ by up to 0.05%, so its PNG and PDF differ from the export.
- Extended Data Table 1 reproduces its PNG and JPEG, but its submission PDF was printed with the earlier page margins.

After each new export the public layer is regenerated and `python -m paper.figures.build_all --write-reference` records the new reference. Until then `python -m paper.figures.build_all` ends with exit code 2, and the tests check that every item not reproduced is listed here.

## Nature Medicine format

The drawing code applies the Nature Medicine profile of `figures/bran_plotly_style.py`. Figures are at most 180 by 170 mm and drawn at final size, in Arial at 5 to 7 pt with 8 pt bold lowercase panel letters, with strokes from 0.25 to 1 pt and no title on the canvas. The private export also checked that every PDF embeds its fonts and holds no Type 3 font and no raster below 300 ppi. Extended Data items are also written as RGB JPEG files at 300 ppi, which are renamed `Surname_EDfigN.jpg` at upload. Each legend opens with a bold title and gives each panel by its bold letter, and `figures/LEGENDS.md` lists the legends with their word counts.

## What stays private and why

The restricted datasets, the aggregates computed from them, the private builders that read those aggregates, trained parameters and any individual-level output stay in the restricted workspace. The data use agreements of PhysioNet and AI-READI do not allow redistribution of patient-level data, and PhysioNet asks that models and data derived from its credentialed datasets be shared only through PhysioNet. See [data, code and model availability](../docs/data_and_model_availability.md).

## Legacy renderer

`render_figures.py` replays the drawing primitives of the 0.2.0 figure scene format. It draws from pixel coordinates rather than from data and is kept only so that its tests and the 0.2.0 record stay valid. The paper figures are built by `figures/build_all.py`.

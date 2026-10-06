# Paper figures and Source Data

This folder rebuilds the figures of the BRAN paper from their Source Data. It covers Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2, and all 12 items rebuild byte for byte from their Source Data in the reference environment (see the verification record). Fig. 6, Extended Data Figs. 6 and 7 and Extended Data Table 3 are still in preparation and are not included.

## Layout

| Path | Contents |
| --- | --- |
| `source_data/` | One Excel workbook and one JSON file per item, `SOURCE_DATA_MANIFEST.json` with the SHA-256 and display number of each file, and a README |
| `figures/draw_<id>.py` | Drawing code of each item |
| `figures/bran_plotly_style.py` | Shared Nature Medicine style, export and Source Data helpers |
| `figures/display_numbers.py` | Display numbers used in every title and cross-reference |
| `figures/ed_canvas.py`, `figures/ed_nm_canvas.py`, `figures/fig3_nm_kit.py`, `figures/fig4_lane_kit.py` | Canvas and layout helpers |
| `figures/build_all.py` | The command that rebuilds and checks the items |
| `figures/disclosure.py` | The disclosure check of the Source Data, with its reviewed exceptions and the open items that block release |
| `figures/REFERENCE_OUTPUTS.json` | Reference hashes of every PNG, PDF and JPEG, the reference environment and the code hashes |
| `figures/SUBMISSION_EXPORTS.json` | Hashes of the files exported for submission |
| `figures/PROVENANCE.json` | The private builder file behind each public code file |
| `figures/LEGENDS.md` | Every legend, generated from the Source Data |
| `figures/requirements-figures.in`, `figures/requirements-figures.txt` | The exact figure environment and its hashed lock |
| `render_figures.py` | Legacy renderer of the 0.2.0 scene format, kept for its tests |

## Rebuild the figures

The reference environment is macOS on Apple silicon with Python 3.12.4, Google Chrome 154.0.8037.98 and the Arial and Arial Bold files that ship with macOS. `figures/REFERENCE_OUTPUTS.json` records it, with the fonts identified by content hash. Byte identity holds only in this recorded environment.

```sh
python3.12 -m venv .venv-figures
.venv-figures/bin/python -m pip install --require-hashes -r paper/figures/requirements-figures.txt
make figures-check PYTHON=.venv-figures/bin/python
```

`make figures-check` runs `python -m paper.figures.build_all`. It draws each item in its own process from its Source Data JSON alone, writes the 600 dpi PNG, the PDF at final size and, for Extended Data items, the 300 ppi RGB JPEG into a new temporary folder, and compares every file hash with `figures/REFERENCE_OUTPUTS.json`. An audit hook stops the drawing process if it opens any other file in the repository. Use `make figures` to keep the files in `paper/output/figures`, which Git ignores.

An item that the reference marks blocked is reported as not drawn on every run. Any other build failure ends the run with exit code 1, also with `--allow-pending`, and the closing line says that every item matches only when every item was rebuilt.

In another environment the items that are not blocked should still build, but the bytes differ when the font files or the Chrome build differ. `make figures-portable` reports those differences without failing. `BRAN_FONT_DIR` names a folder that holds `Arial.ttf` and `Arial Bold.ttf`. It sets the files used to measure text and recorded in the environment, but Chrome draws the text with the fonts the system provides, so on Linux install fonts with Arial metrics such as Liberation Sans. `BROWSER_PATH` names the Chrome binary when no Chrome downloaded by choreographer is present.

The reference Chrome is the system Google Chrome, which updates itself, so the recorded build is neither kept nor available as a pinned download. When Chrome reports another version, `build_all` exits 3 and the byte-identical rebuild test is skipped. PLACEHOLDER for a new reference made with a Chrome for Testing build pinned by version and SHA-256 through `BROWSER_PATH`, which readers can download. Until then a reader outside the reference environment can check that each item builds from its Source Data and can compare the drawings, but cannot expect equal bytes. On Linux, the continuous integration workflow rebuilds every item that is not blocked twice with Chrome for Testing 142.0.7444.175 and requires the two builds to be byte identical.

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
3. The public drawing code in `figures/` reads only the Source Data JSON. It is cut verbatim from the private builder and keeps the top-level statements that the drawing entry point reaches, plus a short `render()` adapter. The adapter of Extended Data Table 1 also measures the boxes of its text check in Arial, the font the table is drawn in, which changes the check and not the drawing. `figures/PROVENANCE.json` gives the builder file and its SHA-256 for each public file.
4. `figures/build_all.py` renders each item and checks the bytes.

Without access to the restricted data, a reader can run step 4 and inspect every value of step 2 in the Source Data. Step 4 gives byte-identical files only in the recorded reference environment. Steps 1 and 2 need the restricted data, and the code of step 1 is in `research/` only for the analyses that [Source Data and analysis code](../docs/source_data_code_coverage.md) marks so.

## Verification record, 6 October 2026

`python -m paper.figures.build_all` was run on macOS 26.5.2 with Python 3.12.4, Chrome 154.0.8037.98 and the packages of `figures/requirements-figures.txt` (all 18 pins at the locked versions except tzdata, which pandas reads only for time zones on platforms without a system zone database). All 12 items were rebuilt from their Source Data alone and reproduced the reference hashes of their PNG, PDF and JPEG, and each drawing process opened only its own Source Data file in the repository. Every file is also byte identical to the file exported for submission.

| Item | Status | Detail |
| --- | --- | --- |
| Fig. 1 | Reproduced | PNG and PDF identical to the submission export |
| Fig. 2 | Reproduced | PNG and PDF identical |
| Fig. 3 | Reproduced | PNG and PDF identical |
| Fig. 4 | Reproduced | PNG and PDF identical |
| Fig. 5 | Reproduced | PNG and PDF identical |
| Extended Data Fig. 1 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Fig. 2 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Fig. 3 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Fig. 4 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Fig. 5 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Table 1 | Reproduced | PNG, PDF and JPEG identical |
| Extended Data Table 2 | Reproduced | PNG, PDF and JPEG identical |

Fig. 3 and Extended Data Figs. 3 and 4 had been exported with an earlier style module whose PDF page margins shrank the drawing to 97.35%, and their layout kit required panel letters of exactly 8.0 pt. The kit now accepts the 0.01 px size that Chrome keeps in a PDF (8.0025 pt), its source-data replay writes the PDF with the same zero-margin writer as the export and compares PDF bytes, and all three items, Figs. 4 and 5, Extended Data Fig. 5 and Extended Data Tables 1 and 2 were exported again under the final module. Every PDF is drawn at scale 1, embeds Arial as TrueType and prints text from 5.0 to 7.0 pt with 8.0 pt panel letters.

After any new export the public layer is regenerated and `python -m paper.figures.build_all --write-reference` records the new reference. The tests require this table to hold one row per item with the status that `figures/REFERENCE_OUTPUTS.json` records, and the Source Data manifest to mark exactly the blocked items as pending.

## Nature Medicine format

The drawing code applies the Nature Medicine profile of `figures/bran_plotly_style.py`. Figures are at most 180 by 170 mm and drawn at final size, in Arial at 5 to 7 pt with 8 pt bold lowercase panel letters, with strokes from 0.25 to 1 pt and no title on the canvas. The private export also checked that every PDF embeds its fonts and holds no Type 3 font and no raster below 300 ppi. Extended Data items are also written as RGB JPEG files at 300 ppi, which are renamed `Surname_EDfigN.jpg` at upload. Each legend opens with a bold title and gives each panel by its bold letter, and `figures/LEGENDS.md` lists the legends with their word counts.

## What stays private and why

The restricted datasets, the aggregates computed from them, the private builders that read those aggregates, trained parameters and any individual-level output stay in the restricted workspace. The data use agreements of PhysioNet and AI-READI do not allow redistribution of patient-level data, and PhysioNet asks that models and data derived from its credentialed datasets be shared only through PhysioNet. See [data, code and model availability](../docs/data_and_model_availability.md).

## Legacy renderer

`render_figures.py` replays the drawing primitives of the 0.2.0 figure scene format. It draws from pixel coordinates rather than from data and is kept only so that its tests and the 0.2.0 record stay valid. The paper figures are built by `figures/build_all.py`.

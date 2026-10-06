# Data, code and model availability

This page holds the draft availability statements for the BRAN manuscript in the form Nature Medicine prints them, the sources the study used, what stays private and why, and the release plan. Items marked PLACEHOLDER need an author decision or a check before submission.

## Draft statements for the manuscript

**Data availability.** The datasets analysed in this study are available from their providers to researchers who meet each provider's access conditions, and the authors are not permitted to redistribute them. AI-READI is available through the AI-READI data platform under the AI-READI license (https://aireadi.org). MIMIC-IV version 3.1, MIMIC-III, the eICU Collaborative Research Database version 2.0, HiRID version 1.1.1, the Northwestern ICU database version 0.1.0, INSPIRE and BRSET version 1.0.1 are available through PhysioNet (https://physionet.org) to credentialed users who complete the required training and sign the data use agreement of each project. NHANES public use files are available from the US National Center for Health Statistics (https://www.cdc.gov/nchs/nhanes). KNHANES data are available from the Korea Disease Control and Prevention Agency after registration (https://knhanes.kdca.go.kr). Source data are provided with this paper. PLACEHOLDER for the MIMIC-III and INSPIRE versions, the AI-READI release and the access tier of BRSET, INSPIRE and the Northwestern ICU database, each to be checked on the provider page.

**Code availability.** The BRAN code is available at https://github.com/ethanwu2011/BRAN under the MIT license. It includes the model, the research code of BRAN training and evaluation and of the analyses mapped in the repository, the input specification, synthetic examples and tests, and code that redraws the finalized figures from their Source Data. PLACEHOLDER for the code of the remaining analyses behind the figures, which is not yet in the repository (`docs/source_data_code_coverage.md` lists them), and for the figures that cannot yet be redrawn (Fig. 3 and Extended Data Figs. 3 and 4). The version used for this paper is archived at Zenodo (PLACEHOLDER for the DOI of release v1.0.0). Trained model parameters are not distributed, as explained under Data availability and below.

**Model availability.** Trained BRAN parameters and fitted preprocessing were derived in part from credentialed PhysioNet data and from AI-READI data and are not distributed through GitHub, Hugging Face or Zenodo. PLACEHOLDER for the access route, which for parameters trained on PhysioNet data can only be a PhysioNet credentialed project under the same agreement, and which for AI-READI needs a determination under its license.

## Sources

The import graph of the research code names more sources than the reported analyses used. The roles below come from the Source Data of Figs. 1 to 5, Extended Data Figs. 1 to 5 and Extended Data Tables 1 and 2. Roles in Fig. 6 and its Extended Data are PLACEHOLDER until that work is final.

| Source | Provider and access | Role in Figs. 1 to 5 and their Extended Data |
| --- | --- | --- |
| AI-READI | AI-READI consortium, controlled access under the AI-READI license | Development cohort with paired retinal and clinical data, used for training and out-of-fold evaluation |
| BRSET 1.0.1 | PhysioNet, PLACEHOLDER for the access tier | Retinal-only training source |
| MIMIC-IV 3.1 | PhysioNet, credentialed access | Clinical-only training source |
| eICU Collaborative Research Database 2.0 | PhysioNet, credentialed access | Clinical-only training source |
| Northwestern ICU (NWICU) 0.1.0 | PhysioNet, PLACEHOLDER for the access tier | Clinical-only training source |
| MIMIC-III | PhysioNet, credentialed access | Clinical-only training source |
| NHANES | US National Center for Health Statistics, public use files | Clinical-only training source |
| INSPIRE | PhysioNet, PLACEHOLDER for the access tier | External surgical cohort without retinal images (Fig. 5f) |
| HiRID 1.1.1 | PhysioNet, credentialed access | External intensive care cohort without retinal images (Fig. 5g) |
| KNHANES 2022 to 2024 | Korea Disease Control and Prevention Agency, registration required | External cohort without retinal images, readout fitting (2022 to 2023) and temporal test (2024) in Fig. 5e |
| SICdb 1.0.8, ICU infection data of Zigong Fourth People's Hospital, ODIR, FD3611, TriHemo-MCV | Various | Referenced by research code only, not used in these figures |

## What stays private and why

| Item | Where it stays | Reason |
| --- | --- | --- |
| Patient records, images, notes and identifiers | Provider and the restricted workspace | Provider data use agreements forbid redistribution |
| Embeddings, latent states, predictions and bootstrap draws | Restricted workspace | Individual-level outputs derived from restricted data |
| Aggregate result files read by the figure builders | Restricted workspace | Derived from restricted data, released only through the disclosure-checked Source Data |
| Private figure builders | Restricted workspace | They read the aggregates by path, and their public drawing part is in `paper/figures` |
| Trained parameters and fitted transforms | Restricted workspace | PhysioNet asks that models derived from credentialed data be shared only through PhysioNet, and the AI-READI license covers derivatives and open-weight models |

The repository holds the code, synthetic fixtures and the Source Data. The Source Data hold aggregates only. Every declared count and every printed lower bound is at least 20, and no proportion may be an unreviewed exact fraction of a hidden count. `python -m paper.figures.disclosure` checks those rules. No disclosure item is open (`paper/source_data/README.md`).

## Provider terms that constrain sharing

These summaries were read on 1 October 2026 from the provider pages listed below. They are a reading aid, not legal advice. Check the current terms before release.

- The MIMIC-IV 3.1 project page asks that derived datasets and models be treated as sensitive and shared only through PhysioNet under the same agreement. Weights or fitted transforms trained partly on PhysioNet data therefore cannot be posted on GitHub, Hugging Face or Zenodo.
- The PhysioNet credentialed data use agreement asks users who openly publish results to share the code behind them. This repository is that code release.
- The AI-READI license applies its restrictions to derivative data, treats synthetic data generated from AI-READI as derivative, applies its model restrictions equally to open-weight models and requires an acknowledgement.

Provider pages are https://physionet.org/content/mimiciv/3.1/, https://physionet.org/content/mimiciv/view-dua/3.1/ and https://docs.aireadi.org/docs/3/preliminary/license.

## Release plan

1. Re-export Fig. 3, Fig. 5, Extended Data Figs. 3 and 4 and Extended Data Table 1 with the final style module, regenerate the public figure layer and record the new reference (see `paper/README.md`).
2. Close the open disclosure items by exporting Figs. 2 and 5 and Extended Data Fig. 2 again with the changes listed in `paper/source_data/README.md`, until `python -m paper.figures.disclosure` exits 0.
3. Add the runner code of every analysis that `docs/source_data_code_coverage.md` marks PLACEHOLDER as a new hash-locked snapshot with its entrypoints, after the same code review as `research/`, and widen the Code availability statement only then.
4. Regenerate the figure reference with a Chrome for Testing build pinned by version and SHA-256 through `BROWSER_PATH`, so that readers can obtain the same browser (see `paper/README.md`).
5. Resolve the PLACEHOLDERs on this page, the BioRender licence of the four icons in `paper/source_data/fig1_source_data.json`, and the Python 3.10 support question in `pyproject.toml`. If the licence does not allow redistribution, rewrite the branch to remove the icons and redraw Fig. 1 with openly licensed or self-drawn symbols before anything is pushed.
6. Review and merge the release branch into `main`, then push. Pushing is the author's step.
7. Set the version to 1.0.0 in `pyproject.toml` and `src/bran_r7_release/__init__.py`, and write a 1.0.0 source manifest, because the facade hash in the sealed 0.2.0 `SOURCE_MANIFEST.json` changes with the version string.
8. Turn on the Zenodo integration for the GitHub repository, then publish the GitHub release `v1.0.0`. Zenodo archives the tagged tree and mints a version DOI and a concept DOI.
9. Add the version DOI to `CITATION.cff`, the README and the Code availability statement, and cite the archive in the manuscript.

## Repository rules

- Synthetic fixtures are generated from a random seed. They are never fitted to, sampled from or tuned on real data.
- Trained weights, fitted transforms, embeddings, predictions and bootstrap draws are never committed.
- Released aggregates follow one rule, a minimum of 20 for every count and printed bound. Parts of the historical research code use a minimum cell of 10, and their outputs are released only through Source Data that pass the rule of 20.

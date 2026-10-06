# Data, code and model availability

This page holds the draft availability statements for the BRAN manuscript and the rules this repository follows. It is a draft for the author. Items marked PLACEHOLDER need an author decision before submission.

## Draft statements

**Code.** The BRAN model code, the research source snapshot used for the reported analyses, the input contract, synthetic examples and tests are available at https://github.com/ethanwu2011/BRAN under the MIT license. PLACEHOLDER for the archived version and its persistent identifier.

**Data.** The study used credentialed and controlled-access datasets that cannot be redistributed by the authors. Each dataset is available from its provider to researchers who meet that provider's access conditions (see the table below). No patient-level data are included in this repository.

**Model.** Trained BRAN parameters and fitted preprocessing were derived in part from credentialed PhysioNet data and from AI-READI data. They are not distributed through GitHub, Hugging Face or Zenodo. PLACEHOLDER for the access route, which for parameters trained on PhysioNet data can only be a PhysioNet credentialed project under the same agreement.

## Sources referenced by the research code

The import graph of the retained runners names more sources than the final analyses used. The table lists every source the code references. The role of each source in the reported analyses must come from the methods, not from the code.

| Source | Provider and access | Role in the reported analyses |
| --- | --- | --- |
| AI-READI | AI-READI consortium, controlled access under the AI-READI license | PLACEHOLDER |
| MIMIC-IV 3.1 | PhysioNet, credentialed access | PLACEHOLDER |
| MIMIC-III | PhysioNet, credentialed access | PLACEHOLDER |
| eICU Collaborative Research Database 2.0 | PhysioNet, credentialed access | PLACEHOLDER |
| HiRID 1.1.1 | PhysioNet, credentialed access | PLACEHOLDER |
| Northwestern ICU (NWICU) 0.1.0 | PhysioNet, credentialed access | PLACEHOLDER |
| SICdb 1.0.8 | PhysioNet, credentialed access | PLACEHOLDER |
| INSPIRE 1.4.2 | PhysioNet, credentialed access | PLACEHOLDER |
| ICU infection, Zigong Fourth People's Hospital | PhysioNet, PLACEHOLDER for the access tier | PLACEHOLDER |
| BRSET 1.0.1 | PhysioNet, PLACEHOLDER for the access tier | PLACEHOLDER |
| NHANES | US National Center for Health Statistics, public use files | PLACEHOLDER |
| KNHANES | Korea Disease Control and Prevention Agency, registration required | PLACEHOLDER |
| ODIR | ODIR-2019 challenge data, registration required | PLACEHOLDER |
| FD3611 | PLACEHOLDER | PLACEHOLDER |
| TriHemo-MCV | PLACEHOLDER | PLACEHOLDER |

## Provider terms that constrain sharing

These summaries were read on 1 October 2026 from the provider pages listed below. They are a reading aid, not legal advice. Check the current terms before release.

- The MIMIC-IV 3.1 project page asks that derived datasets and models be treated as sensitive and shared only through PhysioNet under the same agreement. Weights or fitted transforms trained partly on PhysioNet data therefore cannot be posted on GitHub, Hugging Face or Zenodo.
- The PhysioNet credentialed data use agreement asks users who openly publish results to share the code behind them. This repository is that code release.
- The AI-READI license applies its restrictions to derivative data, treats synthetic data generated from AI-READI as derivative, applies its model restrictions equally to open-weight models and requires an acknowledgement.

Provider pages are https://physionet.org/content/mimiciv/3.1/, https://physionet.org/content/mimiciv/view-dua/3.1/ and https://docs.aireadi.org/docs/3/preliminary/license.

## Repository rules

- Synthetic fixtures are generated from a random seed. They are never fitted to, sampled from or tuned on real data.
- Trained weights, fitted transforms, embeddings, predictions and bootstrap draws are never committed.
- The disclosure threshold for any released aggregate is PLACEHOLDER. The research code uses a minimum cell of 10 in some analyses and 20 in others, and one rule should apply to every released number.

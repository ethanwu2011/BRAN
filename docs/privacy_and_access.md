# Data and access

This repository distributes code. It does not distribute patient records, images, identifiers, notes, embeddings, predictions, bootstrap draws, fitted transforms, model weights or research-result tables.

## Running the examples

The included examples generate artificial inputs from a random seed. They do not load or resemble a named participant. Synthetic checks test the software interface and cannot validate clinical performance.

## Reproducing the study

Obtain each dataset through its provider and comply with its access conditions. Use a separate restricted workspace for source files, derived features, fitted parameters and evaluation outputs. Do not place that workspace inside a clone of this repository.

The source snapshot documents the original local data readers and authentication checks. Those readers depend on study-specific data versions, approved mappings and private artifacts. They do not download credentialed data or confer access to it. An original-run authentication failure is not permission to disable a gate.

The public checkpoint interface supports separately authorized model artifacts. No released weights are currently attached. A reader cannot recover the retained results merely by constructing an untrained model.

## Keeping data out of Git

The ignore rules block common data and checkpoint formats. The release checker also rejects tracked binary or data-like files, scans for selected credential patterns and rejects absolute local paths outside `research/`. Neither mechanism proves privacy. Review each proposed addition before staging it, including literals in Python files and text documents.

Never commit access tokens, credential files, individual-level outputs or a saved real-data notebook. Do not paste those contents into issue reports or hosted coding tools. Report problems using source code, generated fixtures and non-sensitive error descriptions.

## Historical absolute paths in research

The release check rejects any absolute path into a personal home folder or an external volume outside `research/`. Inside `research/` it reports them instead. The research snapshot is hash-locked as byte-identical copies of the study code, so those files cannot be edited without breaking `tools/research_snapshot.py verify`. The author decided on 6 October 2026 to keep the 59 occurrences in 26 modules and document them here.

The paths show the local user name, the home folder layout, an external volume, local dataset folder names and versions, a model cache folder, a project virtual environment and one dataset download container identifier. They contain no patient data, credential or result. They are also present in the public v0.2.0 tag. Running `python scripts/verify_release.py` prints the same list.

| Module | Occurrences |
| --- | --- |
| `research/audit_knhanes_metadata_v1.py` | 1 |
| `research/bran_clinical_dictionary_binding_v1.py` | 3 |
| `research/bran_clinical_utility_schema_v1.py` | 6 |
| `research/bran_hirid_source_h3.py` | 1 |
| `research/bran_mimic_prior_disease_dictionary_v1.py` | 1 |
| `research/bran_nhanes_header_extension_v1.py` | 1 |
| `research/bran_odir_schema_discovery_v1.py` | 2 |
| `research/bran_six_source_schema_preflight_v1.py` | 5 |
| `research/build_bran_knhanes_phase9_binding_v1.py` | 1 |
| `research/build_bran_screening_finish_figure_v1.py` | 2 |
| `research/organize_knhanes_downloads_v1.py` | 1 |
| `research/run_bran_eicu_discovery_admission_e1.py` | 1 |
| `research/run_bran_eicu_discovery_admission_e2.py` | 1 |
| `research/run_bran_inspire_source_qualification_v1.py` | 1 |
| `research/run_bran_mimic_joint_labs_v1.py` | 1 |
| `research/run_bran_mimic_landmark_linkage_v1.py` | 1 |
| `research/run_bran_mimiciii_intake_v2.py` | 1 |
| `research/run_bran_multisource_inventory_v2.py` | 7 |
| `research/run_bran_nhanes_joint_labs_v1.py` | 1 |
| `research/run_bran_observed_cbc_pool_v1.py` | 3 |
| `research/run_bran_odir_content_qualification_v1.py` | 1 |
| `research/run_bran_retinal_content_admission_v1.py` | 3 |
| `research/run_bran_retinal_extraction_v1.py` | 2 |
| `research/run_bran_retinal_group_readiness_v1.py` | 1 |
| `research/run_bran_retinal_supervised_adaptation_v1.py` | 1 |
| `research/run_bran_source_linkage_audit_v1.py` | 10 |

No new file may add such a path. Data roots for new code belong in environment variables or command line arguments.

## Availability statement draft

Code for BRAN is available in the BRAN repository. The repository includes model definitions, a synthetic demonstration, input specifications, the research source snapshot and figure-rendering code. Clinical datasets are obtained from their respective providers subject to access conditions. Patient-derived artifacts are not distributed through GitHub. Access to trained parameters and fitted preprocessing requires a separate determination. PLACEHOLDER for the final version identifier and author-approved access procedure.

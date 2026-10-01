# Data and access

This repository distributes code. It does not distribute patient records, images, identifiers, notes, embeddings, predictions, bootstrap draws, fitted transforms, model weights or research-result tables.

## Running the examples

The included examples generate artificial inputs from a random seed. They do not load or resemble a named participant. Synthetic checks test the software interface and cannot validate clinical performance.

## Reproducing the study

Obtain each dataset through its provider and comply with its access conditions. Use a separate restricted workspace for source files, derived features, fitted parameters and evaluation outputs. Do not place that workspace inside a clone of this repository.

The source snapshot documents the original local data readers and authentication checks. Those readers depend on study-specific data versions, approved mappings and private artifacts. They do not download credentialed data or confer access to it. An original-run authentication failure is not permission to disable a gate.

The public checkpoint interface supports separately authorized model artifacts. No released weights are currently attached. A reader cannot recover the retained results merely by constructing an untrained model.

## Keeping data out of Git

The ignore rules block common data and checkpoint formats. The release checker also rejects tracked binary or data-like files and scans for selected credential patterns. Neither mechanism proves privacy. Review each proposed addition before staging it, including literals in Python files and text documents.

Never commit access tokens, credential files, individual-level outputs or a saved real-data notebook. Do not paste those contents into issue reports or hosted coding tools. Report problems using source code, generated fixtures and non-sensitive error descriptions.

## Availability statement draft

Code for BRAN is available in the BRAN repository. The repository includes model definitions, a synthetic demonstration, input specifications, the research source snapshot and figure-rendering code. Clinical datasets are obtained from their respective providers subject to access conditions. Patient-derived artifacts are not distributed through GitHub. Access to trained parameters and fitted preprocessing requires a separate determination. PLACEHOLDER for the final version identifier and author-approved access procedure.

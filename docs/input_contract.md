# Clinical input contract

This note describes the public model interface used by this release. It does not establish that a source supplies a field, assay, unit, specimen, visit, linkage, or eligible observation.

## Clinical container

The documented container has 59 positions. Positions 0 through 47 are continuous registry fields. Positions 48 through 58 are condition and history labels. The V2 model validates 59-column values and observation masks, accepts 43 unique eligible continuous indices in the range 0 through 47, and requires nine CBC indices to be contained in the eligible set. The 11 history positions are disabled for this V2 input path.

The accompanying schema records the ordered registry names and the eligible positions reported by the public input-schema appendix. That appendix attributes the mask to a separate five-fold configuration receipt. The allowlisted model code validates a caller-supplied mask but does not contain the concrete 43-index list or the complete checkpoint-to-field binding. This release has not independently recreated that checkpoint binding.

Only the nine CBC adapter units are enumerated here. Those units are canonical interface units. They do not qualify any source dictionary, assay, specimen, or conversion. The remaining field unit values are left unset because the allowlisted contract does not independently specify them.

## CBC target interface

The native CBC head has nine outputs. The canonical adapter order is hematocrit, hemoglobin, mean corpuscular hemoglobin, mean corpuscular hemoglobin concentration, mean corpuscular volume, platelets, red blood cells, red-cell distribution width, and white blood cells. The public appendix maps those outputs to continuous container positions 17, 19, 22, 23, 24, 26, 29, 30, and 37. The model code requires nine CBC indices but does not hard-code these positions. The schema labels the position mapping as documented metadata rather than a mapping recovered from a checkpoint in this release lane.

The adapter units are percent, grams per deciliter, picograms, grams per deciliter, femtoliters, thousands per microliter, millions per microliter, percent, and thousands per microliter in the same order. RDW-CV percent is not interchangeable with RDW-SD femtoliters. Source admission and unit conversion require independent evidence.

## Typed age

The age input has seven coordinates. The first three are normalized active value, lower bound, and upper bound. The final four coordinates are a one-hot kind indicator in the order reported, interval, right-censored, and unknown. Original age values and bounds use years. The normalized numeric coordinates have no clinical unit. The model validates shape `[batch, 7]` and the typed-coordinate invariants.

## Retinal input

The documented retinal interface has shape `[batch, retinal_count, retinal_feature_dim]` and a Boolean visibility mask shaped `[batch, retinal_count]`. The public appendix reports an adapted retinal feature dimension of 384. The allowlisted model reads this dimension from its configuration rather than defining a constant. This release does not inspect a checkpoint or establish the feature coordinate order, image provenance, source availability, or pixel-level meaning.

## Provenance and limits

The metadata file records hashes for four code references and two public-ready notes used to prepare this contract. The two code files present in this release are rechecked by the test suite. Hashes for references not included in the release were verified in the original workspace only. The portable test validates those reference identities and digest formats without requiring the original workspace. The patient-atlas field-contract builder was inspected as code but not run. No patient artifacts, source rows, private manifests, model checkpoints, latent arrays, or predictions were opened.

The appendix documents the 48 registry names, the 43 eligible positions, the 11 history labels, the CBC mapping, age shape, and retinal shape. The code directly verifies tensor widths, caller-supplied eligibility constraints, nine-output CBC width, typed-age validation, and dynamic retinal feature width. Exact checkpoint-order bindings remain an explicit limitation of this release schema. No units are inferred for the 39 non-CBC continuous fields.

# Checkpoint loading and inference

This interface operates on a new tensor-only checkpoint format named `bran-r7-source-checkpoint-v1`. It does not read historical private fit checkpoints and must not be described as a byte-for-byte historical checkpoint format. The checkpoint stores the model configuration, tensor state, and digests for the configuration, weights, field map, and fold transform.

## Bindings

Build a field-map binding from the caller's exact 59 feature names and positional indices. The checkpoint stores the map digest and index lists but not the feature names. The example uses fabricated names and indices. They are only shape fixtures.

Build a transform binding from the already fitted fold transform. The expected object follows the existing `FoldTransformV2` fields for clinical median and IQR, retinal mean and scale, age mean and scale, eligibility, held-out fold, and fold identity digests. The helper fingerprints those values in the existing V3 transform order. It does not fit, alter, or serialize transform statistics.

The loader requires caller-pinned checkpoint, field-map, and transform bindings. Pass the expected checkpoint binding again to every `infer` call. A digest detects mismatched bytes but is not a digital signature or source authentication. Keep the expected bindings in a trusted project record. `LoadedR7Checkpoint` is an in-process convenience object, not a security capability. Python callers can construct or mutate objects, so this interface does not defend against a hostile caller in the same process. Supply the same transform object to request original-unit CBC output. Without it the API returns standardized CBC estimates only.

Before deserialization the loader accepts only the PyTorch ZIP serialization format and caps the checkpoint at 64 MiB, the archive at 256 members, each uncompressed member at 64 MiB, and all uncompressed members at 128 MiB. It also caps state dictionaries at 128 tensors and 5 million total scalar elements before hashing. These ceilings are above the included MLP architecture and are resource safeguards, not a guarantee against every denial-of-service condition in the Python runtime.

## Inputs and outputs

`infer` accepts clinical `[batch, 59]`, clinical-observed `[batch, 59]`, retinal `[batch, images, 384]`, retinal-visible `[batch, images]`, and normalized typed age `[batch, 7]` tensors. It also requires the caller-pinned `expected_binding` used at load time. Inputs must already use the bound fold-standardized coordinate frame. This package does not transform raw source arrays.

Observed eligible clinical values and visible retinal vectors must be finite. Masked clinical values and invisible retinal vectors are ignored. Typed-age features must follow the seven-column normalized age contract. Tensor device and floating dtype must match the loaded model.

The result contains the posterior state mean and log variance, the sigmoid probabilities from the frozen 26-output screening head, and the nine linear CBC-head outputs in standardized units. When the supplied transform still matches its binding, CBC outputs are mapped back with the transform's feature-specific IQR and median. The API does not infer a unit label beyond the caller's transform.

Abstained rows retain state outputs and receive NaN for screening and CBC outputs. Set `erase_cbc_targets=True` to call the model's actual `erase_cbc_for_completion` helper before state inference. That operation zeros the nine CBC values and clears their observation flags. The returned completion target mask records which of those fields were originally observed and eligible.

## Example

The small example trains only a single synthetic screening head for three updates, saves it, reloads it with caller-pinned bindings, runs native and completion inference, and checks the all-empty abstention path. It prints only aggregate tensor shapes and synthetic support totals.

```sh
PYTHONPATH=src python examples/synthetic_end_to_end.py
```

The example is not a training recipe, retained model, clinical result, or permission to redistribute fitted study weights. Its tensors, feature names, indices, transform values, and pseudo-target are generated only for software demonstration.

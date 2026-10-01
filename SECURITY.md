# Reporting a problem

Use a minimal generated example for a public issue. Do not attach patient records, images, identifiers, embeddings, predictions, access credentials or private model artifacts.

For a security or privacy concern, contact the repository owner through an established private channel before disclosing exploit details. No institutional security contact has been assigned to this repository.

The public inference helper reads its own restricted checkpoint format with PyTorch's `weights_only` loader. That reduces arbitrary object deserialization but does not make unknown checkpoints trustworthy. Use separately authenticated artifacts from a source you trust.

The historical research snapshot preserves original loading and execution behavior. It is not a service exposed to untrusted inputs. Some original runners load local research artifacts through pickle-compatible formats. Do not use them on downloaded or untrusted files. Run restricted-data research only inside an appropriately controlled environment.

No example or continuous-integration task requires a clinical dataset or a credential. No model output from this repository is approved for clinical care.

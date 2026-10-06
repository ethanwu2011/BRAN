# Contributing

Open a small change with a generated fixture and a focused test. Keep model interface changes separate from scientific-method changes.

The retained architecture modules and historical research snapshot identify the code used in the study. Do not silently rewrite them while cleaning names or formatting. Record any intentional scientific change as a new method version with its own evaluation.

Before proposing a change, run

```sh
python -m unittest discover -s tests -v
python scripts/verify_release.py --working-tree
python tools/research_snapshot.py verify
python tools/research_index.py --check
```

The snapshot check confirms that `research/` is still byte-identical to the reviewed study code. The index check confirms that `docs/research_index.md` matches the snapshot manifest. Run `python tools/research_index.py` to regenerate the index rather than editing it. Record user-visible changes in `CHANGELOG.md`.

Review the files staged for commit. Do not add real data, fitted parameters, figure-result specifications, rendered research figures or private logs. A passing automated scan is not a substitute for reviewing what is being shared.

# Contributing

Open a small change with a generated fixture and a focused test. Keep model interface changes separate from scientific-method changes.

The retained architecture modules and historical research snapshot identify the code used in the study. Do not silently rewrite them while cleaning names or formatting. Record any intentional scientific change as a new method version with its own evaluation.

Before proposing a change, run

```sh
python -m unittest discover -s tests -v
python scripts/verify_release.py --working-tree
python tools/research_snapshot.py verify
python tools/research_index.py --check
python tools/source_data_coverage.py --check
python -m paper.figures.disclosure --allow-open
```

The snapshot check confirms that `research/` is still byte-identical to the reviewed study code. The index check confirms that `docs/research_index.md` matches the snapshot manifest. Run `python tools/research_index.py` to regenerate the index rather than editing it. The coverage check confirms that `docs/source_data_code_coverage.md` maps every Source Data row to an analysis, and `python tools/source_data_coverage.py` regenerates it. The disclosure check fails on any new finding and lists the open items that still block release. Record user-visible changes in `CHANGELOG.md`.

Review the files staged for commit. Do not add real data, fitted parameters, figure-result specifications, rendered research figures or private logs. A passing automated scan is not a substitute for reviewing what is being shared.

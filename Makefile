# Common checks. Run from the repository root. PYTHON can point to any interpreter, for example
#   make figures-check PYTHON=.venv-figures/bin/python
PYTHON ?= python
FIGURE_OUT ?= paper/output/figures

.PHONY: test release-check figures-env figures figures-check figures-portable source-data-check legends all-checks

test:
	$(PYTHON) -m unittest discover -s tests -v

release-check:
	$(PYTHON) scripts/verify_release.py
	$(PYTHON) tools/research_snapshot.py verify
	$(PYTHON) tools/research_index.py --check
	$(PYTHON) tools/source_data_coverage.py --check

# A separate environment with the exact figure stack of the reference build.
figures-env:
	$(PYTHON) -m venv .venv-figures
	.venv-figures/bin/python -m pip install --require-hashes -r paper/figures/requirements-figures.txt

# Exits 2 while open disclosure items remain. They block release (paper/source_data/README.md).
source-data-check:
	$(PYTHON) -m paper.figures.disclosure

legends:
	$(PYTHON) -m paper.figures.build_all --legends

# Rebuild every item from Source Data into a fresh folder and require byte-identical outputs. Exits 2 while items are
# pending in the reference, blocked items included.
figures-check:
	$(PYTHON) -m paper.figures.build_all

# Keep the rebuilt files. FIGURE_OUT must not exist yet. Blocked items are reported and not drawn.
figures:
	$(PYTHON) -m paper.figures.build_all --out $(FIGURE_OUT) --allow-pending

# Outside the reference environment, report byte differences without failing on them.
figures-portable:
	$(PYTHON) -m paper.figures.build_all --portable --allow-pending

all-checks: test release-check source-data-check figures-check

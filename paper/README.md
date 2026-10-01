# Paper figure renderer

This code-only package contains the renderer, not the scientific figure scene or its results. The real scene, aggregate values, source records and rendered figures remain outside this repository. The tests generate a synthetic scene in memory and write only temporary mock figures.

## Render the authorized local handoff

Use the approved local scene at `BRAN_SPICEMIX_FIGURES_2026-10-01_v1/build/figure_scene.json` from the workspace root. Keep the input and generated output local. Do not copy the scene, any aggregate tables, or rendered results into this repository or a public code release.

Install Matplotlib in the local environment, then run

```sh
python paper/render_figures.py \
  --scene /path/to/authorized/figure_scene.json \
  --output /path/to/local/figure-output
```

The renderer expects a six-figure JSON scene in the approved drawing-spec format. It writes one PNG and one SVG per figure. SVG files are editable vector renderings. To use another approved scene, pass its local path with `--scene` and choose a local output directory with `--output`.

## Figure and evidence map

The caller-supplied scene defines the six figure panels and carries each figure's source references. It also includes the approved source basenames and SHA-256 values in its `sources` and `extra_sources` fields. The scene is the frozen aggregate drawing specification, not a portable analytic input. This repository deliberately does not reproduce or distribute those values.

The figure sequence covers a reusable patient representation, complementary disease information, incomplete observation, completion of nine blood measurements, variation within recorded diseases and reuse across clinical settings. Use the source references embedded in the local scene to map each figure to its disclosure-safe aggregate reports.

## Scope

This tool rerenders a supplied frozen scene. It does not rerun statistical analysis, training or evaluation, and it does not establish scientific reproducibility by itself. Scientific reproduction requires the approved aggregate inputs and analysis procedures. Those materials remain under the owner's access controls.

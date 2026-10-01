"""Build an aggregate-only stopping-point screening figure; no new model fitting."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bran-publication-mpl-v1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "bran-screening-finish-figure-v1"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "BRAN_SCREENING_FINISH_FIGURE_V1"
SOURCE = ROOT / "BRAN_INTERNAL_RESEARCH_V1_ATTEMPT2"
SOURCE_MANIFEST_SHA256 = "b7fbe7ea8325b56d716e9e425d6e22d8cefbdd554fa09e4ac82610ec5f9ca2aa"
IMPLEMENTATION = (
    "build_bran_screening_finish_figure_v1.py",
    "test_bran_screening_finish_figure_v1.py",
)
RELEASE_FILES = (
    "01_screening_finish.png",
    "01_screening_finish.svg",
    "README.md",
    "RESULTS.md",
    "METHODS.md",
    "LIMITATIONS.md",
    "REPRODUCE.md",
    "evidence.json",
)
ARM_ORDER = (
    ("bran_both", "BRAN combined\n(blood + retinal)"),
    ("bran_clinical", "BRAN full clinical"),
    ("bran_retinal", "BRAN retinal"),
    ("raw_blood", "Raw blood"),
    ("retfound_green", "RETFound (Green)"),
    ("visionfm_last4", "VisionFM (last 4)"),
    ("dinov3_generic", "DINOv3 generic"),
    ("labrador", "Labrador"),
)
DELTA_KEYS = {
    "bran_clinical": "bran_both_minus_bran_clinical",
    "bran_retinal": "bran_both_minus_bran_retinal",
    "raw_blood": "innovation_selected_minus_raw_blood",
    "retfound_green": "bran_both_minus_retfound_green",
    "visionfm_last4": "bran_both_minus_visionfm_last4",
    "dinov3_generic": "bran_both_minus_dinov3_generic",
    "labrador": "bran_both_minus_labrador",
}


def require(condition, message="release verification failed"):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_screening(root=ROOT):
    """Pin the source manifest before parsing it, then authenticate evidence."""
    root = Path(root)
    manifest_path = root / SOURCE.name / "manifest.json"
    require(sha256_file(manifest_path) == SOURCE_MANIFEST_SHA256,
            "approved source manifest pin does not match")
    manifest = json.loads(manifest_path.read_text())
    evidence_path = root / SOURCE.name / "evidence.json"
    require(sha256_file(evidence_path) == manifest["files"]["evidence.json"],
            "approved source evidence hash does not match manifest")
    evidence = json.loads(evidence_path.read_text())
    screening = evidence["screening"]
    # Select only the approved aggregate screening macro/paired evidence.
    selected = {
        "named_fm_macro_auroc": screening["named_fm_macro_auroc"],
        "named_fm_macro_deltas": screening["named_fm_macro_deltas"],
        "innovation_macro_auroc": {"raw_blood": screening["innovation_macro_auroc"]["raw_blood"]},
        "innovation_macro_deltas": {
            "innovation_selected_minus_raw_blood": screening[
                "innovation_macro_deltas"]["innovation_selected_minus_raw_blood"]},
        "named_fm_primary_family": screening["named_fm_primary_family"],
    }
    expected_arms = {name for name, _ in ARM_ORDER if name != "raw_blood"}
    require(set(selected["named_fm_macro_auroc"]) == expected_arms)
    require(set(selected["named_fm_macro_deltas"]) == {
        key for arm, key in DELTA_KEYS.items() if arm != "raw_blood"})
    require(set(selected["named_fm_primary_family"]["comparators"]) ==
            set(selected["named_fm_primary_family"]["bonferroni_one_sided_95_lower"]))
    return {
        "source": {
            "package": SOURCE.name,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "evidence_sha256": manifest["files"]["evidence.json"],
            "selection": "evidence.screening aggregate macro AUROC and paired fields only",
        },
        "screening": selected,
    }


def figure(data):
    screening = data["screening"]
    macro = dict(screening["named_fm_macro_auroc"])
    macro["raw_blood"] = screening["innovation_macro_auroc"]["raw_blood"]
    deltas = dict(screening["named_fm_macro_deltas"])
    deltas[DELTA_KEYS["raw_blood"]] = screening["innovation_macro_deltas"][DELTA_KEYS["raw_blood"]]
    adjusted = screening["named_fm_primary_family"]["bonferroni_one_sided_95_lower"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.9, 7.5), gridspec_kw={"width_ratios": [1.04, 1.38]})
    fig.patch.set_facecolor("white")
    positions = list(range(len(ARM_ORDER) - 1, -1, -1))
    labels = [label for _, label in ARM_ORDER]
    colors = {
        "bran_both": "#0B5D5E", "bran_clinical": "#3B8C88", "bran_retinal": "#74B6AE",
        "raw_blood": "#9D6A28", "retfound_green": "#5F6E7D", "visionfm_last4": "#778595",
        "dinov3_generic": "#8B97A5", "labrador": "#A1ABB5",
    }
    for y, (arm, _) in zip(positions, ARM_ORDER):
        left.hlines(y, 0.58, macro[arm], color="#D8DEE3", lw=1.6, zorder=1)
        left.plot(macro[arm], y, "o", ms=8.4, color=colors[arm], mec="white", mew=1.2, zorder=2)
        left.text(macro[arm] + .0014, y, f"{macro[arm]:.3f}", va="center", ha="left", fontsize=9.2,
                  color="#263238")
    left.set(xlim=(.58, .715), ylim=(-.7, 7.85), yticks=positions, yticklabels=labels,
             xlabel="Mean endpoint macro AUROC", title="A  Observed screening performance")
    left.grid(axis="x", color="#E6EAED", lw=.9)
    left.tick_params(axis="y", length=0, pad=8, labelsize=10)
    left.set_axisbelow(True)
    left.text(.58, -.12, "Points are estimates only; macro-AUROC confidence intervals are not shown or inferred.",
              transform=left.get_xaxis_transform(), fontsize=8.3, color="#53616C", va="top")

    right.axvline(0, color="#4C5A65", lw=1.2, zorder=0)
    for y, (arm, _) in zip(positions, ARM_ORDER):
        if arm == "bran_both":
            right.text(.104, y, "reference", ha="right", va="center", fontsize=9, color="#53616C")
            continue
        value = deltas[DELTA_KEYS[arm]]
        lower, upper = value["ci95"]
        mean = value["mean_endpoint_auroc_difference"]
        right.hlines(y, lower, upper, color=colors[arm], lw=2.4, zorder=1)
        right.plot(mean, y, "o", ms=7.7, color=colors[arm], mec="white", mew=1.1, zorder=2)
    right.set(xlim=(-.012, .108), ylim=(-.7, 7.85), yticks=positions, yticklabels=[""] * len(positions),
              xlabel="Combined BRAN minus comparator (paired AUROC difference)",
              title="B  Paired endpoint comparison")
    right.grid(axis="x", color="#E6EAED", lw=.9)
    right.tick_params(axis="y", length=0)
    right.set_axisbelow(True)
    right.text(.0, -.12, "Whiskers are paired 95% intervals. Full-clinical interval includes zero.",
               transform=right.get_xaxis_transform(), fontsize=8.3, color="#53616C", va="top")
    right.text(.0, -.205,
               "Four-FM family-adjusted one-sided L95 lower bounds: "
               f"Green {adjusted['retfound_green']:+.3f}  •  VisionFM {adjusted['visionfm_last4']:+.3f}\n"
               f"DINOv3 {adjusted['dinov3_generic']:+.3f}  •  Labrador {adjusted['labrador']:+.3f}",
               transform=right.get_xaxis_transform(), fontsize=8.0, color="#53616C", va="top")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0B5D5E", markeredgecolor="white",
               markersize=8, label="Mean paired difference"),
        Line2D([0], [0], color="#5F6E7D", lw=2.4, label="Paired 95% interval"),
    ]
    right.legend(handles=legend, loc="upper left", frameon=False, fontsize=8.6, handlelength=1.8)
    fig.suptitle("BRAN finite stopping-point screening: observed endpoint evidence", x=.5, y=.975,
                 fontsize=15.2, fontweight="bold", color="#17222B")
    fig.text(.5, .015,
             "Comparisons have unequal information/readout budgets and reuse development data. “RETFound (Green)” is a checkpoint variant, not original RETFound.\n"
             "This screening package provides no biological or clinical superiority guarantee.",
             ha="center", va="bottom", fontsize=8.65, color="#43515B")
    fig.subplots_adjust(left=.215, right=.975, top=.89, bottom=.265, wspace=.25)
    return fig


def layout_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    require(fig.get_figwidth() >= 13, "figure width is too small")
    for ax in fig.axes:
        require(ax.get_position().width > .22, "panel is too narrow")
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            box = text.get_window_extent(renderer)
            require(box.width < fig.bbox.width * .22, "axis label likely overlaps")
        legend = ax.get_legend()
        if legend is not None:
            legend_box = legend.get_window_extent(renderer)
            for text in ax.texts:
                require(not legend_box.overlaps(text.get_window_extent(renderer)),
                        "legend overlaps annotation")
    footer_boxes = [text.get_window_extent(renderer) for text in fig.texts]
    for footer in footer_boxes:
        for ax in fig.axes:
            require(not footer.overlaps(ax.get_window_extent(renderer)), "footer overlaps panel")


def text_files(data):
    source = data["source"]
    return {
        "README.md": """# BRAN screening finish figure V1\n\nA finite stopping-point, aggregate-only screening figure. It contrasts the seven named foundation-model panel arms plus raw-blood comparator, with observed macro-AUROC points and paired combined-minus-comparator intervals. It is descriptive screening evidence, not a clinical-performance claim.\n\nSource pin: `BRAN_INTERNAL_RESEARCH_V1_ATTEMPT2/manifest.json` SHA-256 `{manifest}`. The build authenticates that manifest before parsing it, then verifies `evidence.json` against the manifest before selecting only `evidence.screening` macro/paired fields.\n""".format(manifest=source["manifest_sha256"]),
        "RESULTS.md": """# Caption\n\n**Finite stopping-point screening evidence.** Left: observed endpoint mean macro AUROC for all seven named-FM panel arms and raw blood; confidence intervals for those point estimates are intentionally not invented or displayed. Right: combined BRAN minus each available comparator using paired 95% intervals. The full-clinical comparison includes zero. Four named-FM comparisons also report their family-adjusted one-sided 95% lower bounds as text, distinct from the plotted marginal paired intervals.\n\nReadout and information budgets differ across arms, and this package reuses development data. “RETFound (Green)” is a checkpoint variant, not original RETFound. These results do not establish biological or clinical superiority.\n""",
        "METHODS.md": """# Methods\n\nNo model was fit, tuned, or re-evaluated. The script first pins the approved aggregate release manifest, authenticates its `evidence.json` file hash, and selects only the screening macro-AUROC, paired-difference, raw-blood, and named-FM family fields needed for this figure. The plotted intervals are the authenticated paired 95% intervals; no macro-AUROC confidence intervals are calculated or inferred.\n""",
        "LIMITATIONS.md": """# Interpretation limits\n\nThis is a finite stopping-point screening comparison. Model arms have unequal information/readout budgets and development data are reused. The Green RETFound checkpoint is not original RETFound. A paired interval including zero for the full-clinical comparison means this figure does not support a definitive difference there. Nothing in this release guarantees biological validity, clinical utility, transportability, or superiority.\n""",
        "REPRODUCE.md": """# Reproduce\n\n```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/bran-publication-mpl-v1 /Users/ethanwu/brset-oculomics/.venv/bin/python build_bran_screening_finish_figure_v1.py build\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/bran-publication-mpl-v1 /Users/ethanwu/brset-oculomics/.venv/bin/python -m unittest -v test_bran_screening_finish_figure_v1\n```\n\nThe audit validates source pins, selected evidence equality, generated-file hashes, implementation hashes, and the closed release file list.\n""",
    }


def build(root=ROOT):
    root = Path(root)
    data = source_screening(root)
    out = root / OUT.name
    out.mkdir(exist_ok=True)
    fig = figure(data)
    try:
        layout_check(fig)
        fig.savefig(out / "01_screening_finish.png", dpi=220, facecolor="white")
        fig.savefig(out / "01_screening_finish.svg", facecolor="white", metadata={"Date": None})
    finally:
        plt.close(fig)
    for name, contents in text_files(data).items():
        (out / name).write_text(contents)
    (out / "evidence.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = {name: sha256_file(out / name) for name in RELEASE_FILES}
    manifest = {
        "schema": "bran-screening-finish-figure-v1",
        "automatic_promotion": False,
        "new_model_fits": 0,
        "private_artifacts_opened": False,
        "source": data["source"],
        "files": files,
        "implementation": {name: sha256_file(root / name) for name in IMPLEMENTATION},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"status": "built", "manifest_sha256": sha256_file(out / "manifest.json"), "output": str(out)}


def audit(root=ROOT):
    root = Path(root)
    source = source_screening(root)
    out = root / OUT.name
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["source"] == source["source"], "source provenance changed")
    release_evidence = json.loads((out / "evidence.json").read_text())
    require(release_evidence == source, "release evidence differs from authenticated selection")
    require(set(path.name for path in out.iterdir()) == set(RELEASE_FILES) | {"manifest.json"},
            "release has missing or untracked files")
    for name, digest in manifest["files"].items():
        require(name in RELEASE_FILES and sha256_file(out / name) == digest, "release file hash mismatch")
    require(set(manifest["implementation"]) == set(IMPLEMENTATION), "implementation list changed")
    for name, digest in manifest["implementation"].items():
        require(sha256_file(root / name) == digest, "implementation hash mismatch")
    return {"status": "audited", "manifest_sha256": sha256_file(manifest_path), "output": str(out)}


if __name__ == "__main__":
    try:
        command = sys.argv[1]
        require(command in {"build", "audit"}, "expected build or audit")
        print(json.dumps({"build": build, "audit": audit}[command](), sort_keys=True))
    except Exception:
        print(json.dumps({"status": "blocked_without_disclosure"}))
        raise SystemExit(1)

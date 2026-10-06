"""Rebuild the BRAN paper figures from Source Data and check them against the committed reference hashes.

Run from the repository root.

  python -m paper.figures.build_all                 rebuild every figure into a fresh temporary folder and check it
  python -m paper.figures.build_all --out DIR       rebuild into DIR, which must be empty or absent
  python -m paper.figures.build_all --only fig2     rebuild one figure (ids are listed by --list)
  python -m paper.figures.build_all --portable      outside the reference environment, report hash differences
                                                    without failing on them
  python -m paper.figures.build_all --allow-pending exit 0 when the only problems are items the reference marks
                                                    pending, blocked items included
  python -m paper.figures.build_all --legends       rewrite paper/figures/LEGENDS.md from the Source Data
  python -m paper.figures.build_all --write-reference
                                                    record the hashes of this build as the reference (maintainers)

Each figure is drawn in its own Python process from one file, paper/source_data/<id>_source_data.json. An audit hook
in that process refuses to open any other file inside the repository except the drawing code in this folder and the
output folder, so the drawing step cannot read anything but its Source Data. The process writes the 600 dpi PNG, the
PDF at final size and, for Extended Data items, the 300 ppi RGB JPEG.

The check compares SHA-256 hashes of the PNG, the PDF and the JPEG with REFERENCE_OUTPUTS.json. Byte identity needs
the reference environment recorded there (Python, plotly, Kaleido, Chrome, Pillow and the Arial font files), and it
holds only there. In another environment every item that is not blocked should still build, but the bytes can
differ, and --portable reports that difference without failing.

Exit codes are 0 when every item matches, 1 when an item fails to build or a hash differs, 2 when the only problems
are items marked pending in the reference, and 3 when the environment differs from the reference outside --portable.
Items that the reference marks blocked cannot be drawn under the final style module. Their build failure is reported
on every run but counts as pending, so --allow-pending accepts it. Any other build failure exits 1, with or without
--allow-pending. The closing line states how many items were rebuilt, and it says that every item matches only when
every item was rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

FIGDIR = Path(__file__).resolve().parent
REPO = FIGDIR.parents[1]
SOURCE_DATA = REPO / "paper" / "source_data"
REFERENCE = FIGDIR / "REFERENCE_OUTPUTS.json"
EXPORTS = FIGDIR / "SUBMISSION_EXPORTS.json"
LEGENDS = FIGDIR / "LEGENDS.md"
PACKAGES = ("plotly", "kaleido", "choreographer", "logistro", "orjson", "numpy", "pandas", "openpyxl", "pillow")

# Internal id, drawing module and the way its Extended Data JPEG is made. Order follows the display numbers.
FIGURES = {
    "fig1": None, "fig2": None, "fig3": None, "fig4": None, "fig5": None,
    "ed1": "render_jpeg", "ed2": "render_jpeg", "ed3": "box_average", "ed3b": "box_average", "ed4": "render_jpeg",
    "ed_table1": "render_jpeg", "ed_table2": "render_jpeg",
}


# Display items of the Figure 6 work, which is not yet final. They have no Source Data or drawing code here.
NOT_YET_FINAL = ("fig6", "fig6A", "ed8", "ed_table3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def display(fid: str) -> str:
    sys.path.insert(0, str(FIGDIR))
    try:
        import display_numbers
        return display_numbers.label(fid)
    finally:
        sys.path.remove(str(FIGDIR))


def slug(label: str) -> str:
    """File stem from a display label, for example ExtendedDataFig4 for Extended Data Fig. 4."""
    return re.sub(r"[^A-Za-z0-9]", "", label)


# ------------------------------------------------------------------ child process, one figure
def guard_reads(allowed_files: set[Path], allowed_dirs: list[Path]) -> list[str]:
    """Refuse to open any file inside the repository other than the allowed files, Python files of this folder and
    anything under the allowed folders. Files outside the repository (Python itself, fonts, Chrome) are not limited,
    except an interpreter or virtual environment kept inside the repository, which is exempt."""
    repo = os.path.realpath(REPO)
    files = {os.path.realpath(p) for p in allowed_files}
    dirs = [os.path.realpath(d) for d in allowed_dirs]
    exempt = [os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix}]
    exempt += [os.path.realpath(p) for p in sys.path if p and "site-packages" in p]
    if os.environ.get("BRAN_FONT_DIR"):
        exempt.append(os.path.realpath(os.environ["BRAN_FONT_DIR"]))
    figdir = os.path.realpath(FIGDIR)
    opened: list[str] = []

    def inside(p: str, d: str) -> bool:
        return p == d or p.startswith(d + os.sep)

    def hook(event, args):
        if event != "open" or not args or args[0] is None or isinstance(args[0], int):
            return
        p = os.path.realpath(os.fsdecode(args[0]))
        if not inside(p, repo) or any(inside(p, e) for e in exempt):
            return
        code = os.path.dirname(p) == figdir and p.endswith(".py")
        cache = os.path.basename(os.path.dirname(p)) == "__pycache__" and inside(p, figdir)
        if p in files or code or cache or any(inside(p, d) for d in dirs):
            opened.append(os.path.relpath(p, repo))
            return
        raise PermissionError(f"the drawing step may not read {p}")

    sys.addaudithook(hook)
    return opened


def child(fid: str, out_dir: Path) -> int:
    """Draw one figure from its Source Data alone and write its outputs. Prints one RESULT line of JSON."""
    source = SOURCE_DATA / f"{fid}_source_data.json"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(tempfile.mkdtemp(prefix="bran_figure_cwd_"))   # Kaleido opens its browser profile by relative name
    sys.path.insert(0, str(FIGDIR))
    opened = guard_reads({source}, [out_dir])
    payload = json.loads(source.read_text(encoding="utf-8"))
    module = importlib.import_module(f"draw_{fid}")
    st = sys.modules["bran_plotly_style"]
    assert module.st is st and st.PROFILE["name"] == st.NM_PROFILE, "drawing code must use the Nature Medicine profile"
    pages = module.render(payload)
    label = display(fid)
    result = {"id": fid, "display": label, "pages": {}}
    for i, (stem, fig, width, height) in enumerate(pages):
        name = slug(label) + ("" if len(pages) == 1 else f"_page{i + 1}")
        png, pdf = out_dir / f"{name}.png", out_dir / f"{name}.pdf"
        st.render_png(fig, png, width, height)
        st.write_pdf(fig, pdf, width, height)
        page = {"stem": stem, "size_px": [width, height], "png": png.name, "png_sha256": sha256(png),
                "pdf": pdf.name, "pdf_sha256": sha256(pdf)}
        method = FIGURES[fid]
        if method:
            jpg = out_dir / f"{name}_300ppi.jpg"
            if method == "render_jpeg":
                st.render_jpeg(fig, jpg, width, height)
            else:
                box_average_jpeg(png, jpg)
            page.update({"jpeg": jpg.name, "jpeg_sha256": sha256(jpg), "jpeg_method": method})
        result["pages"][stem] = page
    result["files_opened_in_repository"] = sorted(set(p for p in opened if not p.endswith((".py", ".pyc"))))
    print("RESULT " + json.dumps(result))
    return 0


def box_average_jpeg(png: Path, out: Path) -> None:
    """300 ppi RGB JPEG from the 600 dpi PNG by exact 2 x 2 box averaging, as the Fig. 3 builders write it."""
    from PIL import Image
    with Image.open(png) as im:
        rgb = im.convert("RGB")
    half = rgb.reduce(2)
    half.save(out, "JPEG", quality=95, subsampling=0, optimize=False, dpi=(300, 300))


# ------------------------------------------------------------------ parent process
def environment() -> dict:
    """What decides the output bytes. Fonts are identified by content hash, not by path."""
    env = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
           "os_release": platform.platform(), "packages": {}}
    for name in PACKAGES:
        try:
            env["packages"][name] = md.version(name)
        except md.PackageNotFoundError:
            env["packages"][name] = None
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import bran_plotly_style as st; "
            "print(json.dumps({'arial': st.NM_FONT_FILES[False], 'arial_bold': st.NM_FONT_FILES[True]}))")
    try:
        r = subprocess.run([sys.executable, "-c", code, str(FIGDIR)], capture_output=True, text=True, timeout=120)
        fonts = json.loads(r.stdout.strip().splitlines()[-1])
        env["fonts"] = {k: (sha256(Path(v)) if os.path.isfile(v) else None) for k, v in fonts.items()}
    except Exception as e:  # noqa: BLE001
        env["fonts"] = {"error": type(e).__name__}
    try:
        from choreographer.browsers.chromium import Chromium
        chrome = os.environ.get("BROWSER_PATH") or Chromium.find_browser(skip_local=False)
        out = subprocess.run([chrome, "--version"], capture_output=True, text=True, timeout=120).stdout.strip()
        env["chrome"] = out or None
    except Exception as e:  # noqa: BLE001
        env["chrome"] = f"unknown ({type(e).__name__})"
    return env


STRICT_FIELDS = ("python", "system", "machine", "packages", "fonts", "chrome")


def environment_differences(env: dict, ref: dict) -> list[str]:
    return [f"{k}: {env.get(k)} here, {ref.get(k)} in the reference" for k in STRICT_FIELDS if env.get(k) != ref.get(k)]


def code_hashes() -> dict:
    return {p.name: sha256(p) for p in sorted(FIGDIR.glob("*.py"))}


def run_child(fid: str, out_dir: Path) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "2")
    r = subprocess.run([sys.executable, "-m", "paper.figures.build_all", "--child", fid, "--out", str(out_dir)],
                       cwd=REPO, capture_output=True, text=True, env=env)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not lines:
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"id": fid, "error": tail[0][:400]}
    return json.loads(lines[-1][len("RESULT "):])


def compare(result: dict, ref: dict | None) -> list[str]:
    if ref is None or not ref.get("pages"):
        return []
    out = []
    if set(result["pages"]) != set(ref["pages"]):
        return [f"pages {sorted(result['pages'])} against {sorted(ref['pages'])}"]
    for stem, page in result["pages"].items():
        want = ref["pages"][stem]
        if page["size_px"] != want["size_px"]:
            out.append(f"{stem} size {page['size_px']} against {want['size_px']}")
        for kind in ("png", "pdf", "jpeg"):
            if want.get(f"{kind}_sha256") != page.get(f"{kind}_sha256"):
                out.append(f"{stem} {kind.upper()} hash differs")
    return out


def status_from_exports(result: dict, exports: dict) -> tuple[str, dict]:
    """reproduced when every PNG, PDF and JPEG equals the file the private builder exported for submission, and
    png_reproduced_other_files_differ when only the PNG does. The export names its JPEG by upload name, so any JPEG
    of the item counts."""
    files = exports["figures"][result["id"]]["files"]
    jpegs = {v for k, v in files.items() if k.endswith(".jpg")}
    detail = {}
    for stem, page in result["pages"].items():
        detail[f"{stem}.png"] = page["png_sha256"] == files.get(f"{stem}.png")
        detail[f"{stem}.pdf"] = page["pdf_sha256"] == files.get(f"{stem}.pdf")
        if page.get("jpeg_sha256"):
            detail[f"{stem}.jpeg"] = page["jpeg_sha256"] in jpegs
    if not all(v for k, v in detail.items() if k.endswith(".png")):
        return "differs_from_submission_export", detail
    if not all(detail.values()):
        return "png_reproduced_other_files_differ", detail
    return "reproduced", detail


def legends_text(ids: list[str]) -> str:
    """The legends of the given items in display order, as they stand in their Source Data."""
    later = [display(f) for f in NOT_YET_FINAL]
    lines = ["# Figure legends", "",
             "Generated by `python -m paper.figures.build_all --legends` from the legend field of each Source Data "
             "file. Do not edit by hand. " + ", ".join(later[:-1]) + " and " + later[-1] + " belong to work that is "
             "not yet final and are not included. Each legend opens with its bold title and gives each panel by its "
             "bold letter.", ""]
    for fid in ids:
        payload = json.loads((SOURCE_DATA / f"{fid}_source_data.json").read_text(encoding="utf-8"))
        legend = payload["legend"].strip()
        words = len(re.sub(r"\*\*|__", "", legend).split())
        lines += [f"## {payload['display']}", "", legend, "", f"Words in this legend, {words}.", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_legends(ids: list[str]) -> Path:
    LEGENDS.write_text(legends_text(ids), encoding="utf-8")
    return LEGENDS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="output folder, empty or absent (default a new temporary folder)")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), help="figure ids to rebuild")
    ap.add_argument("--portable", action="store_true", help="report, but do not fail on, hash differences when "
                                                             "the environment is not the reference environment")
    ap.add_argument("--allow-pending", action="store_true", help="exit 0 when the only problems are pending figures")
    ap.add_argument("--write-reference", action="store_true", help="record this build as the reference")
    ap.add_argument("--legends", action="store_true", help="rewrite LEGENDS.md from the Source Data and stop")
    ap.add_argument("--list", action="store_true", help="list figure ids and display numbers and stop")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(a.child, a.out)
    if a.list:
        for fid in FIGURES:
            print(f"{fid:10s} {display(fid)}")
        return 0
    if a.legends:
        print("wrote", write_legends(list(FIGURES)).relative_to(REPO))
        return 0

    if a.write_reference and a.only:
        ap.error("--write-reference needs a full build, without --only")
    ids = a.only or list(FIGURES)
    out = (a.out or Path(tempfile.mkdtemp(prefix="bran_figures_"))).resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.is_file() else None
    exports = json.loads(EXPORTS.read_text(encoding="utf-8"))
    env = environment()
    env_diff = environment_differences(env, reference["environment"]) if reference else ["no reference recorded"]
    code_diff = sorted(k for k, v in code_hashes().items()
                       if reference and reference["code"].get(k) != v) if reference else []

    print(f"Rebuilding {len(ids)} items into {out}")
    results = {}
    for fid in ids:
        res = run_child(fid, out)
        results[fid] = res
        if "error" in res:
            blocked = reference and reference["figures"].get(fid, {}).get("status") == "blocked"
            print(f"  {fid:10s} {display(fid):24s} build failed  {res['error']}"
                  + ("  [blocked in the reference]" if blocked else ""))
            continue
        ref = reference["figures"].get(fid) if reference else None
        problems = compare(res, ref)
        status = ref.get("status") if ref else None
        mark = "matches reference" if ref and ref.get("pages") and not problems else (
            "; ".join(problems) if problems else "no reference")
        print(f"  {fid:10s} {display(fid):24s} {mark}" + (f"  [{status}]" if status and status != "reproduced"
                                                          else ""))

    if a.write_reference:
        figs = {}
        for fid in FIGURES:
            res = results.get(fid) or {"id": fid, "error": "not built in this run"}
            entry = {"display": display(fid), "source_data": f"paper/source_data/{fid}_source_data.json"}
            if "error" in res:
                entry.update({"status": "blocked", "pages": {}, "reason": res["error"]})
            else:
                status, detail = status_from_exports(res, exports)
                entry.update({"status": status, "pages": {s: {k: v for k, v in p.items()}
                                                          for s, p in res["pages"].items()},
                              "matches_submission_export": detail,
                              "files_opened_in_repository": res["files_opened_in_repository"]})
            figs[fid] = entry
        old = reference or {}
        rec = {"schema": "bran-paper-figure-reference-v1",
               "description": "Hashes of the outputs that python -m paper.figures.build_all writes in the reference "
                              "environment. status is reproduced when the rebuilt PNG, PDF and JPEG equal the files "
                              "exported for submission (SUBMISSION_EXPORTS.json), png_reproduced_other_files_differ "
                              "when only the PNG does, differs_from_submission_export when the rebuild is "
                              "deterministic but the export was made with an earlier style module, and blocked when "
                              "the drawing code cannot run under the final style module.",
               "environment": env, "code": code_hashes(),
               "source_data_manifest_sha256": sha256(SOURCE_DATA / "SOURCE_DATA_MANIFEST.json"),
               "figures": figs, "notes": old.get("notes", [])}
        REFERENCE.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("wrote", REFERENCE.relative_to(REPO))
        return 0

    print(f"Outputs in {out}")
    code, lines = summarize(ids, results, reference, env_diff, code_diff, a.portable, a.allow_pending)
    for line in lines:
        print(line)
    return code


def summarize(ids: list[str], results: dict, reference: dict | None, env_diff: list[str], code_diff: list[str],
              portable: bool, allow_pending: bool) -> tuple[int, list[str]]:
    """Exit code and report lines of a rebuild. A build failure of an item the reference marks blocked is pending.
    Any other build failure exits 1, also under --allow-pending. The closing line never says that every item matches
    unless every requested item was rebuilt and matched."""
    figs = reference["figures"] if reference else {}

    def status(f):
        return figs.get(f, {}).get("status")

    errored = [f for f in ids if "error" in results[f]]
    blocked = [f for f in errored if status(f) == "blocked"]
    failed = [f for f in errored if f not in blocked]
    unblocked = [f for f in ids if status(f) == "blocked" and "error" not in results[f]]
    mismatched = [f for f in ids if "error" not in results[f] and compare(results[f], figs.get(f))]
    pending = [f for f in ids if reference and status(f) != "reproduced"]
    built = [f for f in ids if "error" not in results[f]]
    lines = []
    if code_diff:
        lines.append("Drawing code differs from the reference in " + ", ".join(code_diff))
    if env_diff:
        lines.append("Environment differs from the reference")
        lines += ["  " + d for d in env_diff]
    if pending:
        lines.append("Pending in the reference: " + ", ".join(f"{display(f)} ({status(f)})" for f in pending))
    if blocked:
        lines.append("Not drawn, because the reference marks them blocked: " + ", ".join(display(f) for f in blocked))
    if unblocked:
        lines.append("Built although the reference marks them blocked, record a new reference: "
                     + ", ".join(display(f) for f in unblocked))
    if failed:
        lines.append("Build failed: " + ", ".join(display(f) for f in failed))
        return 1, lines
    if mismatched:
        if env_diff and portable:
            lines.append(f"{len(mismatched)} items differ in bytes, expected outside the reference environment "
                         "(--portable)")
        elif env_diff:
            lines.append(f"{len(mismatched)} items differ in bytes and the environment is not the reference "
                         "environment")
            return 3, lines
        else:
            lines.append("Hash mismatch in the reference environment: " + ", ".join(display(f) for f in mismatched))
            return 1, lines
    if len(built) < len(ids):
        closing = (f"Incomplete. {len(built)} of {len(ids)} items were rebuilt, and {len(ids) - len(built)} could "
                   "not be drawn.")
    elif mismatched:
        closing = (f"All {len(ids)} items were rebuilt, and {len(mismatched)} differ in bytes from the reference.")
    else:
        closing = f"All {len(ids)} items were rebuilt and match the reference."
    if pending and not allow_pending:
        return 2, lines + [closing]
    return 0, lines + [closing]


if __name__ == "__main__":
    sys.exit(main())

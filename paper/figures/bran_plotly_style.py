"""Shared style and export helpers for the locked BRAN figures.

Every figure is drawn by a Python script with Plotly and exported by Kaleido to PDF, SVG
and a 600 dpi PNG. Slides only hold the exported image, so no data mark can be dragged
or edited by hand. Each export writes a manifest with the SHA-256 of every output, of the
builder script and of every aggregate source it read, so any change is detectable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# Palette from the 15 September register. Colour encodes modality or comparator class only.
EYE = "#177E89"
BLOOD = "#C05A47"
SHARED = "#514B7C"
FM = "#B8860B"
GREY = "#6B7280"
LIGHT = "#D9DDE2"
INK = "#17212B"
SLATE = "#4A5563"

# Nature full-page width is 183 mm. Plotly sizes are in CSS pixels at 96 per inch.
MM = 96 / 25.4
FULL_W = round(183 * MM)          # about 692 px
FULL_H = round(170 * MM)          # default figure height, adjustable per figure

# Plotly font sizes are CSS pixels at 96 per inch. At the 183 mm export width these give the
# printed point sizes title 12, letter 10, heading 8, label 7 and tick 6.5 (px = pt * 96 / 72).
PT = 96 / 72
SIZES = {"title": 12 * PT, "letter": 10 * PT, "heading": 8 * PT, "label": 7 * PT, "tick": 6.5 * PT}


def template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=SIZES["label"], color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(showline=True, linecolor=INK, linewidth=0.75, ticks="outside", ticklen=3,
                   tickwidth=0.75, tickcolor=INK, showgrid=False, zeroline=False,
                   tickfont=dict(size=SIZES["tick"]), title=dict(font=dict(size=SIZES["label"]))),
        yaxis=dict(showline=True, linecolor=INK, linewidth=0.75, ticks="outside", ticklen=3,
                   tickwidth=0.75, tickcolor=INK, showgrid=False, zeroline=False,
                   tickfont=dict(size=SIZES["tick"]), title=dict(font=dict(size=SIZES["label"]))),
    )
    return t


pio.templates["bran"] = template()
pio.templates.default = "bran"


def panel_letter(fig: go.Figure, x: float, y: float, letter: str, heading: str = "") -> None:
    """Panel letter and short noun-phrase heading as one annotation in paper coordinates.

    Under the Nature Medicine profile the span sizes are written to two decimals, the precision Chrome keeps in a
    PDF, so the letter prints at 8.0 pt and not 8.03 pt (6 October 2026, b). The house profile keeps one decimal."""
    d = 2 if PROFILE["name"] == NM_PROFILE else 1
    text = f"<span style='font-size:{SIZES['letter']:.{d}f}px'><b>{letter}</b></span>"
    if heading:
        text += f"&nbsp;&nbsp;&nbsp;<span style='font-size:{SIZES['heading']:.{d}f}px'><b>{heading}</b></span>"
    fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=text, showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=INK))


def figure_title(fig: go.Figure, number: str, title: str) -> None:
    if PROFILE["name"] == NM_PROFILE:      # Nature Medicine puts the figure title in the legend, not on the canvas
        return
    fig.add_annotation(x=0, y=1.0, xref="paper", yref="paper", text=f"<b>Figure {number}. {title}</b>",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=SIZES["title"], color=INK))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(fig: go.Figure, out_stem: Path, builder: Path, sources: list[Path], legend: str,
           width: int | None = None, height: int | None = None, extra: dict | None = None,
           jpeg: Path | None = None) -> dict:
    """Write PDF, SVG and 600 dpi PNG, then a manifest binding outputs, builder and sources.

    width and height default to FULL_W and FULL_H at call time (6 October 2026, so that the Nature Medicine profile
    can change them). Under that profile the figure, the PDF and the legend are checked before the manifest is
    written, and extra holds further manifest fields such as methods notes and source data hashes.

    6 October 2026 (b). The PDF is written by write_pdf(), which under the Nature Medicine profile prints at final
    size and with fixed dates, the manifest records the software environment, and jpeg, when given, adds the
    Extended Data deliverable (RGB JPEG at 300 ppi, at most 10 MB) written by render_jpeg()."""
    width = FULL_W if width is None else width
    height = FULL_H if height is None else height
    nm = PROFILE["name"] == NM_PROFILE
    if nm:
        problems = nm_figure_problems(fig, width, height)
        assert not problems, "Nature Medicine figure check failed\n  " + "\n  ".join(problems)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, svg, png = (out_stem.with_suffix(s) for s in (".pdf", ".svg", ".png"))
    write_pdf(fig, pdf, width, height)
    fig.write_image(str(svg), width=width, height=height)
    render_png(fig, png, width, height)
    outs = [pdf, svg, png]
    if jpeg is not None:
        render_jpeg(fig, Path(jpeg), width, height)
        outs.append(Path(jpeg))
    manifest = {
        "schema": "bran-locked-figure-v1",
        "builder": {"path": str(builder), "sha256": sha256(builder)},
        "style_module_sha256": sha256(Path(__file__)),
        "outputs": {p.name: sha256(p) for p in outs},
        "sources": {str(s): sha256(s) for s in sources},
        "size_px": [width, height],
        "legend": legend,
    }
    if nm:
        pdf_report = nm_pdf_report(pdf)
        assert not pdf_report["problems"], "Nature Medicine PDF check failed " + "; ".join(pdf_report["problems"])
        manifest["profile"] = {"name": NM_PROFILE, "size_mm": [round(width / MM, 1), round(height / MM, 1)],
                               "font": FONT, "sizes_pt": dict(NM_PT),
                               "sizes_px": {k: SIZES[k] for k in NM_PT}, "pdf": pdf_report,
                               "legend_words": legend_words(legend)}
        if jpeg is not None:
            jrep = nm_jpeg_report(Path(jpeg), width, height)
            assert not jrep["problems"], "Extended Data JPEG check failed " + "; ".join(jrep["problems"])
            manifest["profile"]["jpeg"] = jrep
        manifest["environment"] = environment(pdf)
    if extra:
        assert not set(extra) & set(manifest), set(extra) & set(manifest)
        manifest.update(extra)
    out_stem.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def slide_deck(pngs: list[Path], legends: list[str], out: Path) -> Path:
    """One 16:9 slide per locked figure. The image is a single picture, notes hold the legend."""
    from pptx import Presentation
    from pptx.util import Inches
    from PIL import Image

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for png, legend in zip(pngs, legends):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        w, h = Image.open(png).size
        max_w, max_h = 12.9, 7.2
        scale = min(max_w / (w / 600), max_h / (h / 600))
        pw, ph = w / 600 * scale, h / 600 * scale
        s.shapes.add_picture(str(png), Inches((13.333 - pw) / 2), Inches((7.5 - ph) / 2), Inches(pw), Inches(ph))
        s.notes_slide.notes_text_frame.text = legend
    prs.save(out)
    return out


def render_png(fig: go.Figure, path: Path, width: int, height: int) -> Path:
    """The 600 dpi PNG exactly as export() writes it, so a rebuild can be compared byte for byte."""
    fig.write_image(str(path), width=width, height=height, scale=600 / 96)
    return Path(path)


# ===================================================================== Nature Medicine profile, 6 October 2026
#
# Added by the Figure 1 lane, the one lane allowed to edit this module, from the Nature Medicine figure checklist of
# 6 October 2026 (nature.com/nm submission guidelines and the Nature research figure guide, retrieved that day).
# The profile is opt-in and backward compatible. Nothing above changes its behaviour unless a builder calls
# use_nature_medicine(), so a builder that never calls it renders exactly as before. To opt in, call
# st.use_nature_medicine() straight after importing this module and before copying any constant from it. The call
# changes these module values in place.
#   FONT     "Arial, Helvetica, sans-serif". One sans serif font, Arial or Helvetica, in every figure.
#   SIZES    label 7 pt, tick 6 pt, small 5 pt, panel letter 8 pt bold, heading 7 pt bold. Text is 5 to 7 pt at
#            final size, so the canvas must be drawn at its printed size (FULL_W wide).
#   FULL_W   180 mm (680 px), the widest Nature Medicine figure. FULL_H 169.6 mm (641 px, 642 px until 6 October
#            2026 b), the tallest canvas whose PDF page stays within 170 mm. Both are maxima.
#   EYE, BLOOD, FM   Wong (Okabe-Ito) blue 0072B2, vermillion D55E00 and orange E69F00. BRAN purple 514B7C and the
#            baseline grey 6B7280 stay. Under simulated protanopia the house teal 177E89 and grey 6B7280 differ by
#            3.0 CIELAB units, while the closest pair of the profile differs by 13.9 (cvd_report()).
#   figure_title()   draws nothing, because the figure title belongs in the legend.
#   export()  refuses a figure larger than 180 by 170 mm, a text size outside 5 to 7 pt (8 pt for a lone bold panel
#            letter), a stroke outside 0.25 to 1 pt, gridlines, coloured text or a canvas title, and a PDF with an
#            unembedded or Type 3 font or a raster below 300 ppi. It records the profile and legend word count.
# Helpers for Source Data (one sheet per panel plus a machine-readable JSON), the disclosure check and the
# source-data-only rebuild are at the end. They read nothing themselves.
#
# Revision of 6 October 2026 (b), Figure 1 lane, after an independent check of the exported PDFs.
#   write_pdf()   kaleido 1.3 prints every PDF with 0.1 inch page margins and shrinks the drawing to 97.35%, so 5 pt
#            text printed at 4.86 pt and 8 pt letters at 7.8 pt. Under the profile the page is now printed with zero
#            margins, so the drawing is at final size, and the PDF creation and modification dates are set to
#            NM_PDF_DATE, which makes the PDF byte-for-byte reproducible. The house profile writes PDFs as before.
#   SIZES    Chrome stores PDF font sizes truncated to 0.01 CSS px, so 5 pt (6.6667 px) printed as 4.995 pt. Under
#            the profile each size is now the nearest 0.01 px value inside 5 to 7 pt (small 6.67 px = 5.0025 pt,
#            tick 8 px = 6 pt, label and heading 9.33 px = 6.9975 pt, letter 10.67 px = 8.0025 pt). Builders that
#            already use the profile draw text up to 0.05% larger or smaller and must re-export.
#   nm_pdf_report()   now also reads every text span of the PDF with PyMuPDF and fails on a size outside 5 to 7 pt
#            (8 pt for a lone bold lowercase panel letter), a font other than Arial, or a drawing not at scale 1.
#   environment()   package versions, Chrome path and version, Arial file hashes, poppler version and OS, written
#            into every manifest under the profile. Font files and poppler tools are found from BRAN_FONT_DIR,
#            BRAN_POPPLER_BIN or PATH before the macOS defaults.
#   render_jpeg()   the Extended Data deliverable, RGB JPEG at 300 ppi and at most 10 MB, named Surname_EDfigN.jpg.
#   write_source_data()   rows may carry further source column pairs (any column ending in source with a matching
#            _sha256 column), verify_sources=True re-hashes every named file, and the workbook is written with
#            fixed timestamps so that it is byte-for-byte reproducible.

NM_PROFILE = "nature-medicine-2026-10-06"
PROFILE = {"name": "bran-house-2026-10-01"}
HOUSE = {"FONT": FONT, "EYE": EYE, "BLOOD": BLOOD, "SHARED": SHARED, "FM": FM, "GREY": GREY, "LIGHT": LIGHT,
         "INK": INK, "SLATE": SLATE, "FULL_W": FULL_W, "FULL_H": FULL_H, "SIZES": dict(SIZES)}

WONG = {"black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
        "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7"}
NM_FONT = "Arial, Helvetica, sans-serif"
_FONT_DIRS = (os.environ.get("BRAN_FONT_DIR"),)   # public copy, the system folders are searched by Pillow


def _find_font(names: tuple[str, ...], default: str) -> str:
    """Public copy. A file in BRAN_FONT_DIR wins, then the first match of Pillow's own search of the system font
    folders. When nothing matches, the bare file name is returned and the first text measurement raises."""
    for d in _FONT_DIRS:
        for n in names:
            if d and os.path.isfile(os.path.join(d, n)):
                return os.path.join(d, n)
    from PIL import ImageFont
    for n in names:
        try:
            return ImageFont.truetype(n, 10).path
        except OSError:
            continue
    return default


NM_FONT_FILES = {False: _find_font(("Arial.ttf",), "Arial.ttf"),
                 True: _find_font(("Arial Bold.ttf", "Arial_Bold.ttf", "arialbd.ttf"), "Arial Bold.ttf")}
HOUSE_FONT_FILE = _find_font(("HelveticaNeue.ttc",), "HelveticaNeue.ttc")
NM_MAX_W_MM, NM_MAX_H_MM, NM_ONE_COL_MM = 180.0, 170.0, 89.0
NM_W = math.floor(NM_MAX_W_MM * MM)          # 680 px, 179.9 mm
NM_MAX_H = math.floor(NM_MAX_H_MM * MM) - 1  # 641 px, 169.6 mm. Chrome rounds the PDF page height up to a multiple
#                                              of 0.96 pt, so a 642 px canvas printed a 170.01 mm page (6 October 2026, b)
NM_PT = {"title": 7, "letter": 8, "heading": 7, "label": 7, "tick": 6, "small": 5}
NM_TEXT_PT = (5.0, 7.0)
NM_LETTER_PT = 8.0
NM_LINE_PT = (0.25, 1.0)
NM_COLOURS = {"EYE": WONG["blue"], "BLOOD": WONG["vermillion"], "FM": WONG["orange"], "SHARED": SHARED,
              "GREY": GREY}
NM_TEXT_COLOURS = {INK, SLATE, GREY, "#000000", "black"}
NM_PDF_DATE = "D:20261006000000+00'00'"       # same length as the date Chrome writes, so no offset moves
NM_PRINT_PARAMS = {"printBackground": True, "marginTop": 0, "marginBottom": 0, "marginLeft": 0, "marginRight": 0,
                   "preferCSSPageSize": True, "pageRanges": "1"}
NM_ED_PPI = 300
NM_ED_MAX_BYTES = 10 * 1024 * 1024
NM_SURNAME = "Wu"     # Extended Data file names follow Surname_EDfig1.jpg. Confirm the first author before upload.


def nm_px(pt: float) -> float:
    """CSS px for a text size in pt, rounded to the 0.01 px Chrome keeps in a PDF and kept inside 5 to 7 pt (a size
    above 7 pt, the panel letter, rounds up so that it prints at no less than its nominal size)."""
    px = pt * PT
    up = math.ceil(px * 100 - 1e-6) / 100
    down = math.floor(px * 100 + 1e-6) / 100
    if pt <= NM_TEXT_PT[1] and up / PT > NM_TEXT_PT[1] + 1e-9:
        return down
    return up


def use_nature_medicine() -> None:
    """Switch this module to the Nature Medicine profile. Idempotent. See the block comment above."""
    g = globals()
    g["FONT"] = NM_FONT
    g.update(NM_COLOURS)
    g["FULL_W"], g["FULL_H"] = NM_W, NM_MAX_H
    SIZES.clear()
    SIZES.update({k: nm_px(v) for k, v in NM_PT.items()})
    t = template()
    t.layout.font.family = NM_FONT
    t.layout.font.size = SIZES["label"]
    for ax in ("xaxis", "yaxis"):
        t.layout[ax].tickfont.size = SIZES["tick"]
        t.layout[ax].title.font.size = SIZES["label"]
    pio.templates["bran"] = t
    pio.templates.default = "bran"
    PROFILE["name"] = NM_PROFILE


def text_w(s: str, size_px: float, bold: bool = False) -> float:
    """Advance width in px of one line of plain text at size_px, in the profile's font (Arial under the Nature
    Medicine profile, Helvetica Neue otherwise). HTML tags are ignored."""
    from PIL import ImageFont
    key = (PROFILE["name"], bold)
    face = _FACES.get(key)
    if face is None:
        if PROFILE["name"] == NM_PROFILE:
            face = ImageFont.truetype(NM_FONT_FILES[bold], 1000)
        else:
            face = ImageFont.truetype(HOUSE_FONT_FILE, 1000, index=1 if bold else 0)
        _FACES[key] = face
    plain = re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ")
    return face.getlength(plain) / 1000 * size_px


_FACES: dict = {}


def legend_words(legend: str) -> int:
    """Words in a legend as a reader counts them, markdown emphasis removed."""
    return len(re.sub(r"\*\*|__", "", legend).split())


# ------------------------------------------------------------------ checks on a drawn figure
def _span_sizes(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"font-size:\s*([0-9.]+)px", text or "")]


def nm_figure_problems(fig: go.Figure, width: int, height: int) -> list[str]:
    """Every Nature Medicine canvas rule this module can check on the figure object. Returns a list of problems."""
    out = []
    if width > NM_W or height > NM_MAX_H:
        out.append(f"canvas {width / MM:.1f} x {height / MM:.1f} mm exceeds 180 x 170 mm")
    tol = 0.06                     # px, panel_letter() writes span sizes to one decimal
    lo, hi = NM_TEXT_PT[0] * PT - tol, NM_TEXT_PT[1] * PT + tol
    letter_px = NM_LETTER_PT * PT
    base = fig.layout.font.size or (fig.layout.template.layout.font.size if fig.layout.template else None)
    fam_ok = {None, NM_FONT}

    def size_ok(px, txt):
        if lo <= px <= hi:
            return True
        plain = re.sub(r"<[^>]+>", "", txt or "").replace("&nbsp;", " ").strip()
        return abs(px - letter_px) < tol and re.fullmatch(r"[a-z]", plain) is not None

    for a in fig.layout.annotations:
        txt = a.text or ""
        # a title is a figure number followed by a full stop or a bar and words, not a cross-reference cell
        if re.search(r"^\s*(Figure|Fig\.|Extended\s+Data\s+Fig\.)\s*\d+\s*[.|]\s*\w", re.sub(r"<[^>]+>", "", txt)):
            out.append(f"canvas title {txt[:40]!r}")
        spans = re.findall(r"<span style='font-size:\s*([0-9.]+)px'>(.*?)</span>", txt)
        if spans:
            for px, inner in spans:
                if not size_ok(float(px), inner):
                    out.append(f"text size {float(px) / PT:.2f} pt in {inner[:40]!r}")
        else:
            px = a.font.size if a.font.size is not None else base
            if px is None or not size_ok(px, txt):
                out.append(f"text size {None if px is None else round(px / PT, 2)} pt in {txt[:40]!r}")
        if a.font.family not in fam_ok:
            out.append(f"font {a.font.family!r} in {txt[:30]!r}")
        col = (a.font.color or INK)
        if col not in NM_TEXT_COLOURS:
            out.append(f"coloured text {col} in {txt[:30]!r}")
        if a.showarrow:
            out.append(f"annotation arrow on {txt[:30]!r}")
    lay = fig.layout.to_plotly_json()
    if lay.get("title", {}).get("text"):
        out.append("layout title set")
    if lay.get("font", {}).get("family") not in (None, NM_FONT):
        out.append(f"layout font {lay['font'].get('family')!r}")
    lw_lo, lw_hi = NM_LINE_PT[0] * PT - 1e-6, NM_LINE_PT[1] * PT + 1e-6

    def stroke(w, what):
        for x in (w if isinstance(w, (list, tuple)) else [w]):      # per-point widths come as a sequence
            if x is not None and x > 0 and not (lw_lo <= x <= lw_hi):
                out.append(f"stroke {x / PT:.2f} pt on {what}")

    for k, ax in lay.items():
        if not k.startswith(("xaxis", "yaxis")):
            continue
        if ax.get("showgrid"):
            out.append(f"gridlines on {k}")
        if ax.get("showline"):
            stroke(ax.get("linewidth", 1), f"{k} line")
        if ax.get("ticks") in ("outside", "inside"):
            stroke(ax.get("tickwidth", 1), f"{k} ticks")
        for part in ("tickfont", "title"):
            f = ax.get(part, {})
            f = f.get("font", {}) if part == "title" else f
            if "size" in f and not (lo <= f["size"] <= hi):
                out.append(f"{k} {part} size {f['size'] / PT:.2f} pt")
            if f.get("color") not in (None,) and f.get("color") not in NM_TEXT_COLOURS:
                out.append(f"{k} {part} colour {f.get('color')}")
    for i, sh in enumerate(fig.layout.shapes):
        if sh.line is not None:
            stroke(sh.line.width if sh.line.width is not None else 2, f"shape {i} ({sh.type})")
    for tr in fig.data:
        name = tr.name or tr.type
        if tr.type == "scatter":
            mode = tr.mode or "lines+markers"
            if "lines" in mode:
                stroke(tr.line.width if tr.line.width is not None else 2, f"trace {name} line")
            if "markers" in mode and tr.marker.line is not None:
                stroke(tr.marker.line.width, f"trace {name} marker outline")
            for eb in (tr.error_x, tr.error_y):
                if eb is not None and eb.visible is not False and (eb.array is not None or eb.value is not None):
                    stroke(eb.thickness if eb.thickness is not None else 2, f"trace {name} error bar")
        elif tr.type == "bar" and tr.marker.line is not None:
            stroke(tr.marker.line.width, f"trace {name} bar outline")
    return out


def poppler_tool(name: str) -> str:
    """Path of a poppler command line tool, from BRAN_POPPLER_BIN, then PATH (public copy)."""
    d = os.environ.get("BRAN_POPPLER_BIN")
    if d and os.path.isfile(os.path.join(d, name)):
        return os.path.join(d, name)
    p = shutil.which(name)
    if p:
        return p
    raise FileNotFoundError(f"poppler tool {name} not found, set BRAN_POPPLER_BIN")


def write_pdf(fig: go.Figure, path: Path, width: int, height: int) -> Path:
    """The PDF exactly as export() writes it, so a rebuild can be compared byte for byte.

    House profile, fig.write_image as before. Nature Medicine profile, kaleido's Page.printToPDF call is replaced for
    this one write by the same call with zero page margins (NM_PRINT_PARAMS), so Chrome draws the figure at scale 1 on
    a page of width x height CSS px, and the CreationDate and ModDate strings are then set to NM_PDF_DATE in place
    (same byte length, so the cross-reference offsets stay valid). Nothing else in the file is changed."""
    path = Path(path)
    if PROFILE["name"] != NM_PROFILE:
        fig.write_image(str(path), width=width, height=height)
        return path
    import inspect
    import kaleido._kaleido_tab._devtools_utils as dt
    orig = dt.print_pdf
    assert inspect.iscoroutinefunction(orig) and callable(getattr(dt, "_raise_error", None)), \
        "kaleido internals changed, check write_pdf against the installed kaleido"

    async def print_pdf_final_size(tab):
        response = await tab.send_command("Page.printToPDF", params=dict(NM_PRINT_PARAMS))
        dt._raise_error(response)
        return response.get("result", {}).get("data")

    dt.print_pdf = print_pdf_final_size
    try:
        fig.write_image(str(path), width=width, height=height)
    finally:
        dt.print_pdf = orig
    raw = path.read_bytes()
    stamp = ("(" + NM_PDF_DATE + ")").encode()
    new, n = re.subn(rb"\((D:\d{14}[^)]*)\)", lambda m: stamp if len(m.group(0)) == len(stamp) else m.group(0), raw)
    dates = re.findall(rb"/(?:CreationDate|ModDate) (\([^)]*\))", new)
    assert len(new) == len(raw) and len(dates) == 2 and all(d == stamp for d in dates), \
        f"PDF dates not normalised in {path.name}"
    path.write_bytes(new)
    return path


def render_jpeg(fig: go.Figure, path: Path, width: int, height: int, ppi: int = NM_ED_PPI) -> Path:
    """Extended Data deliverable, an RGB JPEG at ppi (300) on white, quality 95 without chroma subsampling, with the
    resolution written into the file. Rendered by kaleido at ppi / 96 and encoded by Pillow, both deterministic."""
    import io
    from PIL import Image
    raw = fig.to_image(format="png", width=width, height=height, scale=ppi / 96)
    im = Image.open(io.BytesIO(raw))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(path), format="JPEG", quality=95, subsampling=0, dpi=(ppi, ppi), optimize=False, progressive=False)
    return path


def nm_jpeg_report(path: Path, width: int, height: int) -> dict:
    """Extended Data file rules, RGB, 300 ppi or less, 10 MB or less, at most 180 x 170 mm."""
    from PIL import Image
    im = Image.open(path)
    dpi = im.info.get("dpi", (None, None))
    size = Path(path).stat().st_size
    rep = {"file": Path(path).name, "mode": im.mode, "px": list(im.size), "ppi": [round(float(v), 2) for v in dpi]
           if dpi[0] else None, "bytes": size, "mm": [round(im.size[0] / NM_ED_PPI * 25.4, 1),
                                                     round(im.size[1] / NM_ED_PPI * 25.4, 1)], "problems": []}
    if im.mode != "RGB":
        rep["problems"].append(f"JPEG mode {im.mode}")
    if not dpi[0] or round(float(dpi[0])) > NM_ED_PPI or round(float(dpi[1])) > NM_ED_PPI:
        rep["problems"].append(f"JPEG resolution {dpi}")
    if size > NM_ED_MAX_BYTES:
        rep["problems"].append(f"JPEG {size} bytes")
    if rep["mm"][0] > NM_MAX_W_MM + 0.05 or rep["mm"][1] > NM_MAX_H_MM + 0.05:
        rep["problems"].append(f"JPEG {rep['mm']} mm")
    if not re.fullmatch(r"[A-Z][A-Za-z\-]*_EDfig\d+\.jpg", Path(path).name):
        rep["problems"].append(f"file name {Path(path).name}")
    return rep


def ed_jpeg_name(display: str) -> str:
    """Surname_EDfigN.jpg for a display label such as 'Extended Data Fig. 2'."""
    m = re.fullmatch(r"Extended Data Fig\. (\d+)", display)
    assert m, display
    return f"{NM_SURNAME}_EDfig{m.group(1)}.jpg"


def environment(pdf: Path | None = None) -> dict:
    """The software that determines the exported bytes, for the manifest. Reads no project file except the PDF
    named, whose producer string it records."""
    import importlib.metadata as md
    import platform
    pk = {}
    for name in ("plotly", "kaleido", "choreographer", "logistro", "orjson", "numpy", "pillow", "openpyxl",
                 "pymupdf"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    env = {"python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(),
           "machine": platform.machine(), "packages": pk,
           "pip_pins": [f"{k}=={v}" for k, v in pk.items() if v]}
    try:
        from plotly.offline import get_plotlyjs_version
        env["plotly_js"] = get_plotlyjs_version()
    except Exception as e:  # noqa: BLE001
        env["plotly_js"] = f"unknown ({type(e).__name__})"
    try:
        from choreographer.browsers.chromium import Chromium
        chrome = os.environ.get("BROWSER_PATH") or Chromium.find_browser(skip_local=False)
        ver = subprocess.run([chrome, "--version"], capture_output=True, text=True, timeout=120).stdout.strip()
        env["chrome"] = {"path": chrome, "version": ver, "browser_path_env": os.environ.get("BROWSER_PATH")}
    except Exception as e:  # noqa: BLE001
        env["chrome"] = {"path": None, "version": f"unknown ({type(e).__name__})"}
    env["fonts"] = {p: (sha256(Path(p)) if os.path.isfile(p) else None) for p in sorted(set(NM_FONT_FILES.values()))}
    try:
        out = subprocess.run([poppler_tool("pdffonts"), "-v"], capture_output=True, text=True)
        env["poppler"] = (out.stdout + out.stderr).splitlines()[0].strip()
    except Exception as e:  # noqa: BLE001
        env["poppler"] = f"unknown ({type(e).__name__})"
    if pdf is not None:
        import fitz
        meta = fitz.open(stream=Path(pdf).read_bytes(), filetype="pdf").metadata
        env["pdf_writer"] = {"creator": meta.get("creator"), "producer": meta.get("producer")}
    env["pdf_print"] = {"params": dict(NM_PRINT_PARAMS), "dates": NM_PDF_DATE,
                        "note": "kaleido print_pdf replaced during write_pdf only"}
    return env


def nm_pdf_report(pdf: Path) -> dict:
    """Fonts and rasters of an exported PDF, from poppler's pdffonts, pdfimages and pdfinfo, and since 6 October 2026
    (b) the drawing scale, the size and font of every text span (PyMuPDF) and the page count."""
    rep = {"fonts": [], "images": [], "page_mm": None, "problems": []}
    txt = subprocess.run([poppler_tool("pdffonts"), str(pdf)], capture_output=True, text=True, check=True).stdout
    for ln in txt.splitlines()[2:]:
        m = re.match(r"^(\S+)\s+(.+?)\s{2,}(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", ln)
        if not m:
            rep["problems"].append(f"unparsed pdffonts line {ln!r}")
            continue
        name, ftype, _, emb, sub, _ = m.groups()
        rep["fonts"].append({"name": name, "type": ftype.strip(), "embedded": emb == "yes"})
        if "Type 3" in ftype:
            rep["problems"].append(f"Type 3 font {name}")
        if emb != "yes":
            rep["problems"].append(f"font not embedded {name}")
    txt = subprocess.run([poppler_tool("pdfimages"), "-list", str(pdf)], capture_output=True, text=True,
                         check=True).stdout
    for ln in txt.splitlines()[2:]:
        c = ln.split()
        if len(c) >= 14 and c[2] in ("image", "smask", "mask"):
            xppi, yppi = float(c[12]), float(c[13])
            rep["images"].append({"type": c[2], "px": [int(c[3]), int(c[4])], "ppi": [xppi, yppi]})
            if c[2] == "image" and min(xppi, yppi) < 300:
                rep["problems"].append(f"raster at {min(xppi, yppi):.0f} ppi")
    txt = subprocess.run([poppler_tool("pdfinfo"), str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", txt)
    if m:
        w, h = float(m.group(1)) * 25.4 / 72, float(m.group(2)) * 25.4 / 72
        rep["page_mm"] = [round(w, 1), round(h, 1)]
        rep["page_mm_exact"] = [round(w, 3), round(h, 3)]
        if w > NM_MAX_W_MM + 1e-6 or h > NM_MAX_H_MM + 1e-6:
            rep["problems"].append(f"PDF page {w:.3f} x {h:.3f} mm")
    else:
        rep["problems"].append("no PDF page size")
    rep.update(nm_pdf_text(pdf, rep["problems"]))
    return rep


def nm_pdf_text(pdf: Path, problems: list[str]) -> dict:
    """Drawing scale and text spans of a one-page PDF written by write_pdf() under the profile.

    Chrome opens the page content with a flip to 1/300 inch units and a second transform from CSS px. At final size
    that second transform is 3.125 0 0 3.125 0 0, so 1 CSS px is 0.75 pt. Every span with visible text must be Arial at
    5 to 7 pt with no tolerance beyond float noise, or a lone bold lowercase letter within the 0.01 px storage step
    (0.0075 pt) of 8 pt."""
    import fitz
    doc = fitz.open(stream=Path(pdf).read_bytes(), filetype="pdf")
    out = {"pages": doc.page_count, "drawn_scale": None, "drawn_offset_pt": None, "text_pt": None}
    if doc.page_count != 1:
        problems.append(f"PDF has {doc.page_count} pages")
    page = doc[0]
    head = page.read_contents()[:400]
    m = re.match(rb"\s*([0-9.]+) 0 0 -([0-9.]+) 0 [0-9.]+ cm\s+q\s+([0-9.]+) 0 0 ([0-9.]+) ([0-9.]+) ([0-9.]+) cm", head)
    if m:
        unit = float(m.group(1))
        scale = unit * float(m.group(3)) / 0.75
        out["drawn_scale"] = round(scale, 5)
        out["drawn_offset_pt"] = [round(unit * float(m.group(5)), 3), round(unit * float(m.group(6)), 3)]
        if abs(scale - 1) > 1e-4 or max(out["drawn_offset_pt"]) > 1e-3:
            problems.append(f"PDF drawing at scale {scale:.4f} with offset {out['drawn_offset_pt']} pt")
    else:
        problems.append("PDF drawing transform not recognised")
    sizes: dict = {}
    fonts: set = set()
    letters = 0
    noise, tol = 1e-4, 0.0075 + 1e-4
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].replace(" ", " ").strip()
                if not t:
                    continue
                sz, font = float(s["size"]), s["font"]
                fonts.add(font)
                sizes[round(sz, 4)] = sizes.get(round(sz, 4), 0) + 1
                if "Arial" not in font:
                    problems.append(f"PDF font {font} in {t[:30]!r}")
                bold = "Bold" in font or bool(s["flags"] & 16)
                if NM_TEXT_PT[0] - noise <= sz <= NM_TEXT_PT[1] + noise:
                    continue
                if abs(sz - NM_LETTER_PT) <= tol and re.fullmatch(r"[a-z]", t) and bold:
                    letters += 1
                    continue
                problems.append(f"PDF text {sz:.4f} pt in {t[:30]!r}")
    if sizes:
        body = [k for k in sizes if k < NM_LETTER_PT - 0.5]
        out["text_pt"] = {"min": min(body) if body else None, "max": max(body) if body else None,
                          "panel_letters_8pt": letters, "spans_by_size": {str(k): v for k, v in sorted(sizes.items())},
                          "fonts": sorted(fonts)}
    else:
        problems.append("PDF has no text spans")
    return out


def cvd_report(colours: dict | None = None) -> dict:
    """Smallest CIELAB distance between any two class colours with normal vision and under simulated protanopia,
    deuteranopia and tritanopia (Machado, Oliveira and Fernandes 2009, severity 1)."""
    import itertools
    import numpy as np
    cols = colours or {k: globals()[k] for k in ("EYE", "BLOOD", "SHARED", "FM", "GREY")}
    mats = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
            "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
            "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}

    def lin(c):
        c = np.asarray(c, float) / 255
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def lab(l_rgb):
        x = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ np.clip(l_rgb, 0, 1)
        f = lambda t: np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)  # noqa: E731
        fx, fy, fz = f(x / np.array([0.95047, 1.0, 1.08883]))
        return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])

    rgb = {k: lin([int(h[i:i + 2], 16) for i in (1, 3, 5)]) for k, h in cols.items()}
    rep = {}
    for view in ("normal", "protan", "deutan", "tritan"):
        m = np.eye(3) if view == "normal" else np.array(mats[view])
        d, a, b = min((float(np.linalg.norm(lab(m @ rgb[a]) - lab(m @ rgb[b]))), a, b)
                      for a, b in itertools.combinations(cols, 2))
        rep[view] = {"min_delta_e": round(d, 1), "pair": [a, b]}
    return rep


# ------------------------------------------------------------------ Source Data and the source-data-only rebuild
COUNT_FLOOR = 20
_ID_COLUMNS = re.compile(r"(^|_)(id|ids|patient|patients_id|person_id|subject|subject_id|row_id|mrn|seqn|stay_id|"
                         r"hadm_id|subject_ids|image_id|file_id)$", re.I)


def disclosure_problems(payload: dict, max_rows: int = 400) -> list[str]:
    """Checks a Source Data payload before it is written. Every column a builder declares as a count, and every
    lower bound printed as "≥n" inside a text cell, must be at least 20. No column may look like a person or record
    identifier, and no panel may hold more than max_rows rows, so per-person values cannot slip in."""
    out = []
    for key, panel in payload["panels"].items():
        cols = panel["columns"]
        rows = panel["rows"]
        if len(rows) > max_rows:
            out.append(f"panel {key} has {len(rows)} rows")
        for c in cols:
            if _ID_COLUMNS.search(c):
                out.append(f"panel {key} column {c!r} looks like an identifier")
        counts = [cols.index(c) for c in panel.get("count_columns", [])]
        floored = [cols.index(c) for c in panel.get("floored_columns", [])]
        for r in rows:
            if len(r) != len(cols):
                out.append(f"panel {key} row length {len(r)} for {len(cols)} columns")
                continue
            for i in counts:
                v = r[i]
                if v is None:
                    continue
                if not isinstance(v, (int, float)) or v < COUNT_FLOOR:
                    out.append(f"panel {key} count {cols[i]}={v!r} below {COUNT_FLOOR}")
            for i in floored:
                v = r[i]
                if v is not None and (int(v) != v or int(v) % COUNT_FLOOR):
                    out.append(f"panel {key} floored count {cols[i]}={v!r} is not a multiple of {COUNT_FLOOR}")
            for v in r:
                if isinstance(v, str):
                    for m in re.findall(r"≥\s?([0-9][0-9,]*)", v):
                        if int(m.replace(",", "")) < COUNT_FLOOR:
                            out.append(f"panel {key} prints a lower bound below {COUNT_FLOOR} in {v[:40]!r}")
    return out


def source_pairs(columns: list[str]) -> list[tuple[int, int]]:
    """Index pairs of every provenance column (a name ending in source) and its _sha256 column."""
    return [(columns.index(c), columns.index(c + "_sha256")) for c in columns
            if c.endswith("source") and c + "_sha256" in columns]


SD_DATE = (2026, 10, 6, 0, 0, 0)       # fixed workbook and zip timestamps, so the workbook is reproducible


def _normalise_xlsx(path: Path) -> None:
    """Rewrite an openpyxl workbook with fixed zip entry times and fixed created and modified properties."""
    import zipfile
    src = Path(path).read_bytes()
    import io
    zin = zipfile.ZipFile(io.BytesIO(src))
    buf = io.BytesIO()
    stamp = "%04d-%02d-%02dT%02d:%02d:%02dZ" % SD_DATE
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:)",
                              lambda m: m.group(1) + stamp.encode() + m.group(2), data)
            zi = zipfile.ZipInfo(info.filename, date_time=SD_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            zout.writestr(zi, data)
    Path(path).write_bytes(buf.getvalue())


def write_source_data(payload: dict, json_path: Path, xlsx_path: Path, verify_sources: bool = False) -> dict:
    """Write the machine-readable JSON (the drawing input) and the Excel workbook (one sheet per panel) from one
    payload. payload holds figure, display, title, legend, panels {key: {title, columns, rows, count_columns,
    floored_columns, notes}} and draw (everything the drawing step needs). Every panel row carries its source path
    and SHA-256 in columns named source and source_sha256. Returns the SHA-256 of both files.

    6 October 2026 (b). A row read from more than one file names each further file in its own column pair, a column
    ending in source followed by the same name with _sha256 (for example encoder_source, encoder_source_sha256), and
    a pair may be empty only as a whole. verify_sources=True re-hashes every named file and requires equality. The
    workbook is written with fixed timestamps (SD_DATE), so that the same payload gives the same bytes."""
    from openpyxl import Workbook
    from openpyxl.styles import Font
    problems = disclosure_problems(payload)
    assert not problems, "Source Data disclosure check failed\n  " + "\n  ".join(problems)
    hashes: dict = {}
    for key, panel in payload["panels"].items():
        assert panel["columns"][-2:] == ["source", "source_sha256"], key
        pairs = source_pairs(panel["columns"])
        for r in panel["rows"]:
            assert isinstance(r[-2], str) and r[-2] and re.fullmatch(r"[0-9a-f]{64}", r[-1] or ""), (key, r[-2:])
            for i, j in pairs:
                if r[i] is None:
                    assert r[j] is None, (key, panel["columns"][i], r[j])
                    continue
                assert isinstance(r[i], str) and r[i] and re.fullmatch(r"[0-9a-f]{64}", r[j] or ""), (key, r[i], r[j])
                if verify_sources:
                    if r[i] not in hashes:
                        hashes[r[i]] = sha256(Path(r[i]))
                    assert hashes[r[i]] == r[j], f"Source Data {key} names {r[i]} with a stale SHA-256"
    text = json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False)
    assert json.loads(text) == payload, "Source Data payload does not round-trip through JSON"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(text + "\n")
    wb = Workbook()
    ws = wb.active
    ws.title = "About"
    about = [("Figure", payload["display"]), ("Title", payload["title"]),
             ("Contents", "One sheet per panel with exactly the plotted values, intervals, floored counts and labels. "
                          "Each row names the aggregate file it was read from and that file's SHA-256, with one "
                          "further column pair for each further file a row was read from."),
             ("Disclosure", "Counts are lower bounds rounded down to a multiple of 20 unless the column says exact. "
                            "No cell below 20 and no per-person value is present."),
             ("Machine-readable copy", json_path.name), ("Legend", payload["legend"])]
    for k, v in about:
        ws.append([k, v])
    ws["A1"].font = Font(bold=True)
    for key, panel in payload["panels"].items():
        sh = wb.create_sheet(title=f"{payload['display_short']} {key}"[:31])
        sh.append([panel["title"]])
        sh["A1"].font = Font(bold=True)
        for n in panel.get("notes", []):
            sh.append([n])
        sh.append([])
        sh.append(panel["columns"])
        for c in sh[sh.max_row]:
            c.font = Font(bold=True)
        for r in panel["rows"]:
            sh.append([("" if v is None else v) for v in r])
    for sh in wb.worksheets:
        for col in sh.columns:
            sh.column_dimensions[col[0].column_letter].width = 18
    wb.properties.creator = "BRAN locked figure builder"
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(xlsx_path))
    _normalise_xlsx(xlsx_path)
    return {"json": {"path": str(json_path), "sha256": sha256(json_path)},
            "xlsx": {"path": str(xlsx_path), "sha256": sha256(xlsx_path)}}


PRIVATE_ROOTS = (Path(__file__).resolve().parents[2],)   # public copy, the repository root


def guard_reads(allowed: list[Path], code_dirs: list[Path]) -> list[str]:
    """For the source-data-only rebuild. From this call on, opening any file under the project tree or an external
    volume raises, except the allowed files and Python source files in code_dirs. Returns the live list of the
    project files that were opened, for the build log."""
    ok = {os.path.realpath(p) for p in allowed}
    dirs = [os.path.realpath(d) for d in code_dirs]
    roots = [os.path.realpath(r) for r in PRIVATE_ROOTS]
    opened: list[str] = []

    def hook(event, args):
        if event != "open" or not args or isinstance(args[0], int) or args[0] is None:
            return
        p = os.path.realpath(os.fsdecode(args[0]))
        if not any(p == r or p.startswith(r + os.sep) for r in roots):
            return
        code = (p.endswith(".py") and os.path.dirname(p) in dirs) or (
            p.endswith(".pyc") and os.path.basename(os.path.dirname(p)) == "__pycache__"
            and os.path.dirname(os.path.dirname(p)) in dirs)
        if p in ok or code:
            opened.append(p)
            return
        raise PermissionError(f"source-data-only rebuild may not read {p}")

    sys.addaudithook(hook)
    return opened


def same_bytes(a: Path, b: Path) -> bool:
    return Path(a).read_bytes() == Path(b).read_bytes()


def rebuild_outputs(fig: go.Figure, width: int, height: int, out_dir: Path, stem: str, compare: dict) -> dict:
    """For the source-data-only rebuild. Writes the PNG, and the PDF and JPEG when compare names a locked file for
    them, into out_dir with the functions export() uses, and compares each with the locked file byte for byte.
    compare maps 'png', 'pdf' and 'jpeg' to locked paths. Returns {kind: {path, identical}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for kind in ("png", "pdf", "jpeg"):
        ref = compare.get(kind)
        if ref is None and kind != "png":
            continue
        out = out_dir / f"{stem}_rebuild.{'jpg' if kind == 'jpeg' else kind}"
        {"png": render_png, "pdf": write_pdf, "jpeg": render_jpeg}[kind](fig, out, width, height)
        res[kind] = {"path": str(out), "identical": ref is not None and same_bytes(out, ref)}
    return res

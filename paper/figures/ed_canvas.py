"""Pixel-grid canvas for the locked Extended Data builders (ED3, ED4 and ED Table 1).

A Canvas is one Plotly figure with zero margins, so one CSS pixel on the canvas is one pixel of
the 183 mm export width and paper coordinates follow directly from pixel positions measured from
the top-left corner. Panels are composed from axis domains. Data marks are Plotly traces on those
axes. Rules, reference lines and table rules are layout shapes. All text is drawn as annotations
whose ink boxes are recorded, so overlaps and text outside the canvas stop the build.

Only the sizes in bran_plotly_style.SIZES and the text colours INK and SLATE are accepted. Hover is
disabled, so the exported PDF, SVG and PNG are the only products and nothing can be dragged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import plotly.graph_objects as go  # noqa: E402
from PIL import ImageFont  # noqa: E402

import bran_plotly_style as st  # noqa: E402

FONT_FILE = "HelveticaNeue.ttc"   # public copy, found by Pillow's search of the system font folders
TEXT_COLOURS = {st.INK, st.SLATE}
WHITE = "#FFFFFF"
MUTED = "#9AA3AD"     # dashed reference lines and PLACEHOLDER frames
DASH = "3px,2px"      # fine dash for reference lines and PLACEHOLDER frames
MINUS = "−"
_FACES: dict[int, ImageFont.FreeTypeFont] = {}


def text_w(s: str, size: float, bold: bool = False) -> float:
    """Rendered width in pixels of a single-line string in Helvetica Neue."""
    idx = 1 if bold else 0
    if idx not in _FACES:
        _FACES[idx] = ImageFont.truetype(FONT_FILE, 1000, index=idx)
    plain = re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ")
    return _FACES[idx].getlength(plain) / 1000 * size


NBSP = "\u00a0"


def wrap(s: str, width: float, size: float, bold: bool = False, first: float | None = None,
         keep: tuple[str, ...] = ()) -> list[str]:
    """Greedy word wrap to a pixel width. Words break only at plain spaces, so phrases in keep, joined by
    non-breaking spaces, stay on one line. first is a narrower width for the first line."""
    for k in keep:
        s = s.replace(k, k.replace(" ", NBSP))
    lines, cur = [], ""
    for word in s.split(" "):
        lim = first if (first is not None and not lines) else width
        trial = word if not cur else cur + " " + word
        if text_w(trial, size, bold) <= lim or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    for i, ln in enumerate(lines):
        lim = first if (first is not None and i == 0) else width
        assert text_w(ln, size, bold) <= lim + 0.5, (ln, text_w(ln, size, bold), lim)
    return lines


def signed(v: float, nd: int) -> str:
    if abs(v) < 0.5 * 10 ** -nd:
        return "0"
    return (MINUS if v < 0 else "+") + f"{abs(v):.{nd}f}"


def plain(v: float, nd: int) -> str:
    if abs(v) < 0.5 * 10 ** -nd:
        return "0"
    return (MINUS if v < 0 else "") + f"{abs(v):.{nd}f}"


class Ax:
    """One Plotly axis pair with a known pixel box and data range."""

    def __init__(self, cv: "Canvas", n: int, box, xr, yr):
        self.cv, self.n = cv, n
        self.x0, self.x1, self.y0, self.y1 = box
        self.xr, self.yr = xr, yr
        self.xref = "x" if n == 1 else f"x{n}"
        self.yref = "y" if n == 1 else f"y{n}"

    def X(self, v: float) -> float:
        a, b = self.xr
        return self.x0 + (v - a) / (b - a) * (self.x1 - self.x0)

    def Y(self, v: float) -> float:
        """Pixel y of data value v. The y range is given top to bottom as (top value, bottom value)."""
        a, b = self.yr
        return self.y0 + (v - a) / (b - a) * (self.y1 - self.y0)

    def inside_x(self, *vals: float) -> None:
        lo, hi = sorted(self.xr)
        for v in vals:
            assert lo < v < hi, (self.n, v, self.xr)

    def inside_y(self, *vals: float) -> None:
        lo, hi = sorted(self.yr)
        for v in vals:
            assert lo < v < hi, (self.n, v, self.yr)

    # data marks
    def seg(self, xs, ys, color, width=1.0, dash=None):
        self.cv._seg(self, color, width, dash, xs, ys)

    def hci(self, y, est, lo, hi, color, hollow=False, size=5.0, width=1.0, caps=False):
        """Point estimate and horizontal interval at data row y."""
        self.inside_x(lo, hi, est)
        self.seg([lo, hi], [y, y], color, width)
        if caps:
            h = (self.yr[1] - self.yr[0]) / (self.y1 - self.y0) * 2.2
            self.seg([lo, lo], [y - h, y + h], color, 0.8)
            self.seg([hi, hi], [y - h, y + h], color, 0.8)
        self.dot(est, y, color, hollow=hollow, size=size)

    def vci(self, x, est, lo, hi, color, hollow=False, size=5.0, width=1.0):
        self.inside_y(lo, hi, est)
        self.seg([x, x], [lo, hi], color, width)
        self.dot(x, est, color, hollow=hollow, size=size)

    def dot(self, x, y, color, hollow=False, size=5.0, symbol="circle"):
        self.cv._dot(self, x, y, color, hollow, size, symbol)


class Canvas:
    def __init__(self, width: int, height: int):
        self.W, self.H = width, height
        self.fig = go.Figure()
        self.fig.update_layout(template="bran", width=width, height=height, margin=dict(l=0, r=0, t=0, b=0),
                               showlegend=False, hovermode=False, dragmode=False,
                               paper_bgcolor="white", plot_bgcolor="white")
        self.n = 0
        self.boxes: list[tuple[float, float, float, float, str]] = []
        self.texts: list[tuple[str, float, str, bool]] = []
        self._segs: dict = {}
        self._dots: dict = {}

    # coordinates
    def fx(self, x: float) -> float:
        return x / self.W

    def fy(self, y: float) -> float:
        return 1 - y / self.H

    # axes
    def axis(self, box, xr, yr, xticks=None, xtext=None, yticks=None, ytext=None, xline=True, yline=False,
             xside="bottom", yside="left"):
        """Axis pair on a pixel box (x0, x1, y0, y1). yr runs from the top value to the bottom value."""
        x0, x1, y0, y1 = box
        assert 0 <= x0 < x1 <= self.W and 0 <= y0 < y1 <= self.H, box
        self.n += 1
        n = self.n
        ax = Ax(self, n, box, xr, yr)
        common = dict(showgrid=False, zeroline=False, ticks="outside", ticklen=3, tickwidth=0.75, tickcolor=st.INK,
                      linecolor=st.INK, linewidth=0.75, tickfont=dict(size=st.SIZES["tick"], color=st.SLATE,
                                                                      family=st.FONT),
                      fixedrange=True, layer="above traces")
        xa = dict(common, domain=[self.fx(x0), self.fx(x1)], range=list(xr), anchor=ax.yref, side=xside,
                  showline=xline, showticklabels=xticks is not None)
        if xticks is None:
            xa.update(ticks="")
        else:
            xa.update(tickmode="array", tickvals=list(xticks), ticktext=list(xtext))
        ya = dict(common, domain=[self.fy(y1), self.fy(y0)], range=[yr[1], yr[0]], anchor=ax.xref, side=yside,
                  showline=yline, showticklabels=yticks is not None)
        if yticks is None:
            ya.update(ticks="")
        else:
            ya.update(tickmode="array", tickvals=list(yticks), ticktext=list(ytext))
        self.fig.layout["xaxis" if n == 1 else f"xaxis{n}"] = xa
        self.fig.layout["yaxis" if n == 1 else f"yaxis{n}"] = ya
        if xticks is not None:  # tick label ink boxes under a bottom axis
            for v, t in zip(xticks, xtext):
                w = text_w(t, st.SIZES["tick"])
                xx = ax.X(v)
                yy = y1 + 3 + 2 if xside == "bottom" else y0 - 5 - st.SIZES["tick"]
                self.boxes.append((xx - w / 2, yy, xx + w / 2, yy + st.SIZES["tick"] * 0.95, t))
        if yticks is not None:
            for v, t in zip(yticks, ytext):
                w = text_w(t, st.SIZES["tick"])
                yy = ax.Y(v)
                self.boxes.append((x0 - 5 - w, yy - st.SIZES["tick"] * 0.45, x0 - 5, yy + st.SIZES["tick"] * 0.45, t))
        return ax

    # batched data traces
    def _seg(self, ax, color, width, dash, xs, ys):
        k = (ax.n, color, width, dash)
        X, Y = self._segs.setdefault(k, ([], []))
        X.extend(list(xs) + [None])
        Y.extend(list(ys) + [None])

    def _dot(self, ax, x, y, color, hollow, size, symbol):
        k = (ax.n, color, hollow, size, symbol)
        X, Y = self._dots.setdefault(k, ([], []))
        X.append(x)
        Y.append(y)

    def flush(self):
        for (n, color, width, dash), (X, Y) in self._segs.items():
            self.fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", xaxis="x" if n == 1 else f"x{n}",
                                          yaxis="y" if n == 1 else f"y{n}", hoverinfo="skip", cliponaxis=False,
                                          line=dict(color=color, width=width, dash=dash or "solid")))
        for (n, color, hollow, size, symbol), (X, Y) in self._dots.items():
            self.fig.add_trace(go.Scatter(x=X, y=Y, mode="markers", xaxis="x" if n == 1 else f"x{n}",
                                          yaxis="y" if n == 1 else f"y{n}", hoverinfo="skip", cliponaxis=False,
                                          marker=dict(symbol=symbol, size=size, color=WHITE if hollow else color,
                                                      line=dict(color=color, width=1.0 if hollow else 0))))
        self._segs, self._dots = {}, {}

    # paper-space marks
    def line(self, x0, y0, x1, y1, color=st.INK, width=0.75, dash=None):
        self.fig.add_shape(type="line", xref="paper", yref="paper", x0=self.fx(x0), y0=self.fy(y0),
                           x1=self.fx(x1), y1=self.fy(y1), line=dict(color=color, width=width, dash=dash or "solid"),
                           layer="above")

    def rect(self, x0, y0, x1, y1, fill=None, edge=None, width=0.5, dash=None):
        self.fig.add_shape(type="rect", xref="paper", yref="paper", x0=self.fx(x0), y0=self.fy(y1),
                           x1=self.fx(x1), y1=self.fy(y0), fillcolor=fill or "rgba(0,0,0,0)",
                           line=dict(color=edge or "rgba(0,0,0,0)", width=width if edge else 0, dash=dash or "solid"),
                           layer="below" if fill and not edge else "above")

    def text(self, x, y, s, size="label", color=st.INK, bold=False, xa="left", ya="middle", angle=0, check=True):
        """One line of text. x, y in canvas pixels, anchored by xa and ya. Returns the ink width."""
        assert size in st.SIZES, size
        assert color in TEXT_COLOURS, (color, s)
        for bad in (";", "—"):
            assert bad not in s, s
        assert ":" not in s, s
        px = st.SIZES[size]
        w = text_w(s, px, bold)
        h = px * 1.0
        if angle == 0 and check:
            x0 = x if xa == "left" else (x - w if xa == "right" else x - w / 2)
            y0 = y - h / 2 if ya == "middle" else (y - h if ya == "bottom" else y)
            self.boxes.append((x0, y0, x0 + w, y0 + h, s))
        self.texts.append((s, px, color, bold))
        body = f"<b>{s}</b>" if bold else s
        self.fig.add_annotation(x=self.fx(x), y=self.fy(y), xref="paper", yref="paper", text=body, showarrow=False,
                                xanchor=xa, yanchor=ya, textangle=angle, borderpad=0,
                                font=dict(size=px, color=color, family=st.FONT))
        return w

    def heading(self, x, y, letter, name):
        """Panel letter and noun-phrase heading sharing one text-box bottom at y (one baseline)."""
        assert 1 <= len(name.split()) <= 4 and name[0].isupper() and not name.endswith("."), name
        self.text(x, y, letter, size="letter", bold=True, ya="bottom")
        self.text(x + 13, y - 0.6, name, size="heading", bold=True, ya="bottom")

    def title(self, s):
        self.text(0, 2, s, size="title", bold=True, ya="top")

    def placeholder(self, x0, y0, x1, y1):
        self.rect(x0, y0, x1, y1, edge=MUTED, width=0.6, dash=DASH)
        self.text((x0 + x1) / 2, (y0 + y1) / 2, "PLACEHOLDER", size="tick", color=st.SLATE, xa="center")

    # checks
    def check(self, tag: str, tol: float = 0.4) -> int:
        hits = 0
        for i, a in enumerate(self.boxes):
            if a[0] < -0.5 or a[2] > self.W + 0.5 or a[1] < -0.5 or a[3] > self.H + 0.5:
                print(f"OUTSIDE {tag} {a}")
                hits += 1
            for b in self.boxes[i + 1:]:
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > tol and oy > tol:
                    hits += 1
                    print(f"OVERLAP {tag} {a[4]!r} x {b[4]!r} ({ox:.1f} by {oy:.1f} px)")
        print(f"{tag} text check, {len(self.boxes)} boxes, {hits} problems")
        return hits


PREVIEW = Path(__import__("tempfile").gettempdir()) / "bran_figure_preview"   # public copy


def preview(fig, name: str, width: int, height: int) -> Path:
    """Draft PNG in the session scratchpad for visual review when a check fails. Never a deliverable."""
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = PREVIEW / f"{name}_draft.png"
    fig.write_image(str(out), width=width, height=height, scale=3)
    return out


def legend_ok(txt: str) -> None:
    """Prose rules for a legend string. URLs and file paths are exempt from the colon rule."""
    for bad in (";", "—"):
        assert bad not in txt, bad
    stripped = re.sub(r"(https?\S+|/\S+)", "", txt)
    assert ":" not in stripped, [ln for ln in stripped.splitlines() if ":" in ln][:2]

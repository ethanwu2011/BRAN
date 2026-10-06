"""Pixel-grid canvas for Extended Data Figs. 1 and 2 under the Nature Medicine profile.

Written 6 October 2026 for build_ed1.py and build_ed2.py only. It follows ed_canvas.py, which other builders still
use unchanged, with three differences. Text is measured in Arial (bran_plotly_style.text_w under the Nature Medicine
profile), any text size from 5 to 7 pt can be used (panel letters 8 pt bold), and every stroke is checked to lie
between 0.25 and 1 pt. One canvas pixel is one CSS pixel of the 180 mm export, with zero margins, so paper
coordinates follow directly from pixel positions measured from the top-left corner.

Text is drawn as annotations whose ink boxes are recorded, so overlaps and text outside the canvas stop the build.
Only ink, slate and grey text is accepted. Hover is off, so the exported PDF, SVG and PNG are the only products.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import plotly.graph_objects as go  # noqa: E402

import bran_plotly_style as st  # noqa: E402

assert st.PROFILE["name"] == st.NM_PROFILE, "call st.use_nature_medicine() before importing ed_nm_canvas"

WHITE = "#FFFFFF"
MUTED = "#9AA3AD"      # dashed reference lines and PLACEHOLDER frames
DASH = "3px,2px"
MINUS = "−"
TEXT_COLOURS = {st.INK, st.SLATE, st.GREY}
PT = st.PT
LW_MAX = 1.0 * PT      # 1 pt
LW_MIN = 0.25 * PT


def px_of(size) -> float:
    """Text size in px from a SIZES key or a number of points between 5 and 7 (8 for a panel letter). A number of
    points is rounded by st.nm_px to the 0.01 px step Chrome keeps in a PDF (6 October 2026, b)."""
    if isinstance(size, str):
        return st.SIZES[size]
    assert 5.0 <= float(size) <= 7.0, size
    return st.nm_px(float(size))


def text_w(s: str, size, bold: bool = False) -> float:
    return st.text_w(s, px_of(size), bold)


def fmt(v: float, d: int, sign: bool = False) -> str:
    s = f"{abs(float(v)):.{d}f}"
    if float(v) < 0 and float(s) != 0:
        return MINUS + s
    if float(v) < 0:
        return MINUS + s
    return ("+" + s) if sign else s


def ci(lo: float, hi: float, d: int, sign: bool = False) -> str:
    return f"[{fmt(lo, d, sign)}, {fmt(hi, d, sign)}]"


def wrap(s: str, width: float, size, bold: bool = False, keep: tuple[str, ...] = ()) -> list[str]:
    """Greedy word wrap to a pixel width. Phrases in keep never break."""
    nb = " "
    for k in keep:
        s = s.replace(k, k.replace(" ", nb))
    lines, cur = [], ""
    for word in s.split(" "):
        trial = word if not cur else cur + " " + word
        if text_w(trial, size, bold) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    for ln in lines:
        assert text_w(ln, size, bold) <= width + 0.5, (ln, width)
    return lines


class Ax:
    """One Plotly axis pair with a known pixel box and data range."""

    def __init__(self, cv: "Canvas", n: int, box, xr, yr):
        self.cv, self.n = cv, n
        self.x0, self.x1, self.y0, self.y1 = box
        self.xr, self.yr = xr, yr

    def X(self, v: float) -> float:
        a, b = self.xr
        return self.x0 + (v - a) / (b - a) * (self.x1 - self.x0)

    def Y(self, v: float) -> float:
        a, b = self.yr
        return self.y0 + (v - a) / (b - a) * (self.y1 - self.y0)

    def inside_x(self, *vals: float) -> None:
        lo, hi = sorted(self.xr)
        for v in vals:
            assert lo < v < hi, (self.n, v, self.xr)

    def seg(self, xs, ys, color, width=1.0, dash=None):
        self.cv._seg(self, color, width, dash, xs, ys)

    def hci(self, y, est, lo, hi, color, hollow=False, size=4.6, width=1.0):
        self.inside_x(lo, hi, est)
        self.seg([lo, hi], [y, y], color, width)
        self.dot(est, y, color, hollow=hollow, size=size)

    def dot(self, x, y, color, hollow=False, size=4.6, symbol="circle"):
        self.cv._dot(self, x, y, color, hollow, size, symbol)


class Canvas:
    def __init__(self, width: int, height: int):
        assert width <= st.NM_W and height <= st.NM_MAX_H, (width, height)
        self.W, self.H = width, height
        self.fig = go.Figure()
        self.fig.update_layout(template="bran", width=width, height=height, margin=dict(l=0, r=0, t=0, b=0),
                               showlegend=False, hovermode=False, dragmode=False, paper_bgcolor="white",
                               plot_bgcolor="white")
        self.n = 0
        self.boxes: list[tuple[float, float, float, float, str]] = []
        self.texts: list[tuple[str, float, str, bool]] = []
        self._segs: dict = {}
        self._dots: dict = {}

    def fx(self, x: float) -> float:
        return x / self.W

    def fy(self, y: float) -> float:
        return 1 - y / self.H

    def axis(self, box, xr, yr, xticks=None, xtext=None, xline=True, tick_size="small"):
        x0, x1, y0, y1 = box
        assert 0 <= x0 < x1 <= self.W and 0 <= y0 < y1 <= self.H, box
        self.n += 1
        n = self.n
        ax = Ax(self, n, box, xr, yr)
        xref, yref = ("x", "y") if n == 1 else (f"x{n}", f"y{n}")
        common = dict(showgrid=False, zeroline=False, ticks="outside", ticklen=2.5, tickwidth=0.75, tickcolor=st.INK,
                      linecolor=st.INK, linewidth=0.75, tickfont=dict(size=px_of(tick_size), color=st.SLATE,
                                                                      family=st.FONT),
                      fixedrange=True, layer="above traces")
        xa = dict(common, domain=[self.fx(x0), self.fx(x1)], range=list(xr), anchor=yref, side="bottom",
                  showline=xline, showticklabels=xticks is not None)
        if xticks is None:
            xa.update(ticks="")
        else:
            xa.update(tickmode="array", tickvals=list(xticks), ticktext=list(xtext))
        ya = dict(common, domain=[self.fy(y1), self.fy(y0)], range=[yr[1], yr[0]], anchor=xref, showline=False,
                  showticklabels=False, ticks="")
        self.fig.layout["xaxis" if n == 1 else f"xaxis{n}"] = xa
        self.fig.layout["yaxis" if n == 1 else f"yaxis{n}"] = ya
        if xticks is not None:
            h = px_of(tick_size)
            for v, t in zip(xticks, xtext):
                w = text_w(t, tick_size)
                xx = ax.X(v)
                self.boxes.append((xx - w / 2, y1 + 4, xx + w / 2, y1 + 4 + h * 0.95, t))
        return ax

    def _seg(self, ax, color, width, dash, xs, ys):
        assert LW_MIN - 1e-6 <= width <= LW_MAX + 1e-6, width
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
                                                      opacity=1, line=dict(color=color, width=1.0 if hollow else 0))))
        self._segs, self._dots = {}, {}

    def line(self, x0, y0, x1, y1, color=None, width=0.75, dash=None):
        assert LW_MIN - 1e-6 <= width <= LW_MAX + 1e-6, width
        self.fig.add_shape(type="line", xref="paper", yref="paper", x0=self.fx(x0), y0=self.fy(y0), x1=self.fx(x1),
                           y1=self.fy(y1), line=dict(color=color or st.INK, width=width, dash=dash or "solid"),
                           layer="above")

    def rect(self, x0, y0, x1, y1, fill=None, edge=None, width=0.6, dash=None):
        if edge:
            assert LW_MIN - 1e-6 <= width <= LW_MAX + 1e-6, width
        self.fig.add_shape(type="rect", xref="paper", yref="paper", x0=self.fx(x0), y0=self.fy(y1), x1=self.fx(x1),
                           y1=self.fy(y0), fillcolor=fill or "rgba(0,0,0,0)",
                           line=dict(color=edge or "rgba(0,0,0,0)", width=width if edge else 0, dash=dash or "solid"),
                           layer="below" if fill and not edge else "above")

    def text(self, x, y, s, size="label", color=None, bold=False, xa="left", ya="middle", check=True):
        color = color or st.INK
        assert color in TEXT_COLOURS, (color, s)
        for bad in (";", "—", ":"):
            assert bad not in s, s
        px = px_of(size)
        w = text_w(s, size, bold)
        h = px * 1.0
        if check:
            x0 = x if xa == "left" else (x - w if xa == "right" else x - w / 2)
            y0 = y - h / 2 if ya == "middle" else (y - h if ya == "bottom" else y)
            self.boxes.append((x0, y0, x0 + w, y0 + h, s))
        self.texts.append((s, px, color, bold))
        body = f"<b>{s}</b>" if bold else s
        self.fig.add_annotation(x=self.fx(x), y=self.fy(y), xref="paper", yref="paper", text=body, showarrow=False,
                                xanchor=xa, yanchor=ya, borderpad=0, font=dict(size=px, color=color, family=st.FONT))
        return w

    def heading(self, x, y, letter, name):
        """Panel letter 8 pt bold and a short noun-phrase heading 7 pt bold, sharing one text-box bottom at y."""
        assert 1 <= len(name.split()) <= 5 and name[0].isupper() and not name.endswith("."), name
        self.text(x, y, letter, size="letter", bold=True, ya="bottom")
        self.text(x + 10, y - 0.3, name, size="heading", bold=True, ya="bottom")

    def placeholder(self, x0, y0, x1, y1, size="small"):
        self.rect(x0, y0, x1, y1, edge=MUTED, width=0.6, dash=DASH)
        self.text((x0 + x1) / 2, (y0 + y1) / 2, "PLACEHOLDER", size=size, color=st.SLATE, xa="center")

    def check(self, tag: str, tol: float = 0.3) -> list[str]:
        out = []
        for i, a in enumerate(self.boxes):
            if a[0] < -0.5 or a[2] > self.W + 0.5 or a[1] < -0.5 or a[3] > self.H + 0.5:
                out.append(f"OUTSIDE {tag} {a}")
            for b in self.boxes[i + 1:]:
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > tol and oy > tol:
                    out.append(f"OVERLAP {tag} {a[4]!r} x {b[4]!r} ({ox:.1f} by {oy:.1f} px)")
        return out

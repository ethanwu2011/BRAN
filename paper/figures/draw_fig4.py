"""Drawing code for Fig. 4.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_fig4.py
(SHA-256 99866f4bc79297550a308535048b43ae61b6d203cce7a42ef153d61010033f27).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import re
import bran_plotly_style as st
st.use_nature_medicine()
import plotly.graph_objects as go
FIG_ID = "fig4"
FIELDS = ["hemoglobin", "hct", "rbc", "mcv", "mch", "mchc", "rdw", "wbc", "plt"]
LABEL = {"hemoglobin": "Haemoglobin", "hct": "Haematocrit", "rbc": "Red cells", "mcv": "MCV", "mch": "MCH",
         "mchc": "MCHC", "rdw": "RDW", "wbc": "White cells", "plt": "Platelets"}
SHORT = {"hemoglobin": "Hb", "hct": "Hct", "rbc": "RBC", "mcv": "MCV", "mch": "MCH", "mchc": "MCHC", "rdw": "RDW",
         "wbc": "WBC", "plt": "Plt"}
CONTEXTS = [("single_target_hidden", "single_target_no_retina"), ("red_cell_hidden", "red_cell_no_retina"),
            ("whole_cbc_hidden", "whole_cbc_no_retina")]
GROUP_LABEL = ["One value", "Red-cell family", "Whole panel"]
INC_LIM = (-6.0, 3.0)
C_LIM = (0.90, 1.06)
MINUS = "−"
HEADINGS = {"a": "Target erasure", "b": "Masking contexts", "c": "Retina and age only",
            "d": "Error against reference intervals", "e": "Referral in development",
            "f": "Uncertainty and selective use", "g": "Haemoglobin below 12 g/dL", "h": "Official test split"}
EYE, BLOOD, SHARED, GREY, LIGHT, INK, SLATE = st.EYE, st.BLOOD, st.SHARED, st.GREY, st.LIGHT, st.INK, st.SLATE
assert (EYE, BLOOD) == ("#0072B2", "#D55E00"), "Nature Medicine profile colours expected"
WHITE = "#FFFFFF"
FAINT = "#D9DDE2"
RAMP_END = (0x8F, 0x99, 0xA3)
TINT_BLOOD = "#%02X%02X%02X" % tuple(round(255 + (int(BLOOD[i:i + 2], 16) - 255) * 0.2) for i in (1, 3, 5))
SZ = st.SIZES
TEXT_COLOURS = {INK, SLATE}
W_PX = st.FULL_W
SIDE_PX = 4
T_PX = 2
W_MM = (W_PX - 2 * SIDE_PX) / st.MM
PAPER_MM = 167.3
H_PX = round(PAPER_MM * st.MM) + T_PX
PH_MM = (H_PX - T_PX) / st.MM
assert H_PX <= st.FULL_H, (H_PX, st.FULL_H)
MS = 4.4
MS_TICK = 5.6
LW = 1.0
AXW = 0.75
DASH = "3px,2px"


def fx(x):
    return x / W_MM


def fy(y):
    return 1.0 - y / PH_MM


def tw(s, size=SZ["label"], bold=False):
    """Rendered width in mm of the widest line of text at a Plotly px size, in the profile font."""
    lines = [re.sub(r"<[^>]+>", "", ln).replace("&nbsp;", " ") for ln in s.split("<br>")]
    return max(st.text_w(ln, size, bold) for ln in lines) / st.MM


def th(s, size=SZ["label"]):
    return (s.count("<br>") + 1) * size * 1.2 / st.MM


def interval_class(lo, hi):
    if hi < 0:
        return "lower"
    if lo > 0:
        return "higher"
    return "zero"


def grid_fill(r):
    """Neutral ramp, white at ratio 1.0 to a mid grey at 0.35, light enough for black numbers (as build_fig4_v5)."""
    t = max(0.0, min(1.0, (1.0 - r) / 0.65))
    return "#%02X%02X%02X" % tuple(round(255 + (e - 255) * t) for e in RAMP_END)


def signed(v):
    return "0" if abs(v) < 1e-12 else f"{v:+.0f}".replace("-", MINUS)


class Draw:
    """One figure, its axis registry and the ink boxes of every text, for the overlap audit."""

    def __init__(self):
        self.fig = go.Figure()
        self.boxes = []
        self.n_ax = 0
        self.canvas = self.new_axes(0, W_MM, 0, PH_MM, xline=False)

    # axes
    def new_axes(self, x0, x1, y0, y1, xrange=None, yrange=None, xticks=None, xtext=None, yticks=None, ytext=None,
                 xline=True, yline=False):
        self.n_ax += 1
        i = self.n_ax
        s = "" if i == 1 else str(i)
        xa = dict(domain=[fx(x0), fx(x1)], range=xrange or [x0, x1], anchor=f"y{s}", fixedrange=True,
                  showline=xline, linewidth=AXW, linecolor=INK, showgrid=False, zeroline=False,
                  ticks="outside" if xticks else "", ticklen=3, tickwidth=AXW, tickcolor=INK,
                  tickvals=xticks or [], ticktext=xtext or [], showticklabels=bool(xticks),
                  tickfont=dict(size=SZ["tick"], color=INK), visible=xline or bool(xticks))
        ya = dict(domain=[fy(y1), fy(y0)], range=yrange or [y1, y0], anchor=f"x{s}", fixedrange=True,
                  showline=yline, linewidth=AXW, linecolor=INK, showgrid=False, zeroline=False,
                  ticks="outside" if yticks else "", ticklen=3, tickwidth=AXW, tickcolor=INK,
                  tickvals=yticks or [], ticktext=ytext or [], showticklabels=bool(yticks),
                  tickfont=dict(size=SZ["tick"], color=INK), visible=yline or bool(yticks))
        self.fig.layout[f"xaxis{s}"] = xa
        self.fig.layout[f"yaxis{s}"] = ya
        ax = dict(x=f"x{s}", y=f"y{s}", box=(x0, x1, y0, y1), xr=xa["range"], yr=ya["range"])
        for t, lab in zip(xticks or [], xtext or []):
            xm = self.px(ax, t)
            self.boxes.append((xm - tw(lab, SZ["tick"]) / 2, y1 + 1.3, xm + tw(lab, SZ["tick"]) / 2,
                               y1 + 1.3 + th(lab, SZ["tick"]), f"tick {lab}"))
        for t, lab in zip(yticks or [], ytext or []):
            ym = self.py(ax, t)
            self.boxes.append((x0 - 1.3 - tw(lab, SZ["tick"]), ym - th(lab, SZ["tick"]) / 2, x0 - 1.3,
                               ym + th(lab, SZ["tick"]) / 2, f"tick {lab}"))
        return ax

    @staticmethod
    def px(ax, v):
        x0, x1 = ax["box"][:2]
        a, b = ax["xr"]
        return x0 + (v - a) / (b - a) * (x1 - x0)

    @staticmethod
    def py(ax, v):
        y0, y1 = ax["box"][2:]
        lo, hi = ax["yr"]
        return y1 + (v - lo) / (hi - lo) * (y0 - y1)

    # text
    def text(self, x, y, s, size=SZ["label"], color=INK, align="left", valign="middle", bold=False, angle=0):
        assert size in SZ.values(), (size, s)
        assert color in TEXT_COLOURS or (s == "PLACEHOLDER" and color == GREY), (color, s)
        assert s == "PLACEHOLDER" or "PLACEHOLDER" not in s, s
        plain = re.sub(r"<[^>]+>", "", s).replace("&nbsp;", " ")
        for bad in (";", ":", "—"):
            assert bad not in plain, s
        assert not re.search(r"[A-Za-z]\.(\s|$)", plain), f"sentence on canvas {s!r}"
        assert sum(any(c.isalpha() for c in t) for t in plain.replace("<br>", " ").split()) <= 5, s
        self.fig.add_annotation(x=x, y=y, xref=self.canvas["x"], yref=self.canvas["y"], text=f"<b>{s}</b>" if bold else s,
                                showarrow=False, xanchor=align, yanchor=valign, textangle=angle,
                                font=dict(size=size, color=color), borderpad=0,
                                align=align if align != "center" else "center")
        w, h = tw(s, size, bold), th(s, size)
        if angle:
            w, h = h, w
        x0 = {"left": x, "center": x - w / 2, "right": x - w}[align]
        y0 = {"top": y, "middle": y - h / 2, "bottom": y - h}[valign]
        self.boxes.append((x0, y0, x0 + w, y0 + h, plain))
        return self.boxes[-1]

    def heading(self, x, y, letter):
        name = HEADINGS[letter]
        assert 1 <= len(name.split()) <= 4 and "?" not in name, name
        st.panel_letter(self.fig, fx(x), fy(y), letter, name)
        w = tw(letter, SZ["letter"], True) + tw("   ", SZ["heading"]) + tw(name, SZ["heading"], True)
        self.boxes.append((x, y - SZ["letter"] * 1.2 / st.MM, x + w, y, f"heading {letter}"))

    def xtitle(self, ax, s, dy=4.4, x=None):
        x0, x1, _, y1 = ax["box"]
        return self.text((x0 + x1) / 2 if x is None else x, y1 + dy, s, align="center", valign="top")

    # marks
    def rect(self, x0, y0, x1, y1, fill=None, edge=None, width=AXW, dash=None, ax=None):
        ax = ax or self.canvas
        self.fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref=ax["x"], yref=ax["y"], layer="above",
                           fillcolor=fill or "rgba(0,0,0,0)",
                           line=dict(color=edge or "rgba(0,0,0,0)", width=width if edge else 0, dash=dash))

    def hline(self, x0, x1, y, color=INK, width=AXW, dash=None):
        self.fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, xref=self.canvas["x"], yref=self.canvas["y"],
                           layer="above", line=dict(color=color, width=width, dash=dash))

    def vref(self, ax, x, color=INK, dash=None, width=AXW):
        self.fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref=ax["x"], yref=f"{ax['y']} domain",
                           layer="below", line=dict(color=color, width=width, dash=dash))

    def href(self, ax, y, color=INK, dash=None, width=AXW):
        self.fig.add_shape(type="line", x0=0, x1=1, y0=y, y1=y, xref=f"{ax['x']} domain", yref=ax["y"],
                           layer="below", line=dict(color=color, width=width, dash=dash))

    def dot(self, x, y, color, hollow=False, size=MS, ax=None, name=None, symbol="circle"):
        ax = ax or self.canvas
        self.fig.add_trace(go.Scatter(x=[x], y=[y], xaxis=ax["x"], yaxis=ax["y"], mode="markers", hoverinfo="skip",
                                      cliponaxis=False, showlegend=False, name=name,
                                      marker=dict(size=size, color=WHITE if hollow else color, symbol=symbol,
                                                  line=dict(color=color, width=1.0 if hollow else 0))))

    def hci(self, ax, y, est, lo, hi, color, hollow=False, name=None, symbol="circle", size=MS):
        assert lo <= est <= hi, (est, lo, hi)
        assert ax["xr"][0] < lo and hi < ax["xr"][1], ("interval outside axis", lo, hi, ax["xr"])
        self.fig.add_trace(go.Scatter(x=[est], y=[y], xaxis=ax["x"], yaxis=ax["y"], mode="markers", hoverinfo="skip",
                                      cliponaxis=False, showlegend=False, name=name,
                                      error_x=dict(type="data", symmetric=False, array=[hi - est],
                                                   arrayminus=[est - lo], color=color, thickness=LW, width=0),
                                      marker=dict(size=size, color=WHITE if hollow else color, symbol=symbol,
                                                  line=dict(color=color, width=1.0 if hollow else 0))))

    def vci(self, ax, x, est, lo, hi, color, hollow=False):
        assert lo <= est <= hi, (est, lo, hi)
        assert min(ax["yr"]) < lo and hi < max(ax["yr"]), ("interval outside axis", lo, hi, ax["yr"])
        self.fig.add_trace(go.Scatter(x=[x], y=[est], xaxis=ax["x"], yaxis=ax["y"], mode="markers", hoverinfo="skip",
                                      cliponaxis=False, showlegend=False,
                                      error_y=dict(type="data", symmetric=False, array=[hi - est],
                                                   arrayminus=[est - lo], color=color, thickness=LW, width=0),
                                      marker=dict(size=MS, color=WHITE if hollow else color,
                                                  line=dict(color=color, width=1.0 if hollow else 0))))

    def key_dot(self, x, y, color, label, hollow=False, size=SZ["label"]):
        self.dot(x + 0.9, y, color, hollow=hollow)
        self.text(x + 2.3, y, label, size=size)
        return x + 2.3 + tw(label, size) + 2.6

    def key_tick(self, x, y, label, size=SZ["label"]):
        self.fig.add_trace(go.Scatter(x=[x + 0.9], y=[y], xaxis=self.canvas["x"], yaxis=self.canvas["y"],
                                      mode="markers", hoverinfo="skip", cliponaxis=False, showlegend=False,
                                      marker=dict(symbol="line-ns", size=MS_TICK, line=dict(color=GREY, width=1.2))))
        self.text(x + 2.3, y, label, size=size)
        return x + 2.3 + tw(label, size) + 2.6

    def placeholder(self, x, y, w, h=3.0):
        """A thin grey dashed rectangle with the single word PLACEHOLDER."""
        self.rect(x, y - h / 2, x + w, y + h / 2, edge=GREY, width=0.6, dash="2px,1.5px")
        self.text(x + w / 2, y, "PLACEHOLDER", size=SZ["tick"], color=GREY, align="center")

    def bracket(self, ax, lo, hi, y, color, cap, width):
        self.fig.add_trace(go.Scatter(x=[lo, lo, None, lo, hi, None, hi, hi],
                                      y=[y - cap, y + cap, None, y, y, None, y - cap, y + cap],
                                      xaxis=ax["x"], yaxis=ax["y"], mode="lines", hoverinfo="skip",
                                      showlegend=False, cliponaxis=False, line=dict(color=color, width=width)))

    def band(self, ax, lo, hi, y0, y1):
        self.fig.add_shape(type="rect", x0=lo, x1=hi, y0=y0, y1=y1, xref=ax["x"], yref=ax["y"], layer="below",
                           fillcolor=LIGHT, line=dict(width=0))


A_HEAD = 3.7


def panel_a(D):
    D.heading(0, A_HEAD, "a")
    ts, tg = 2.8, 0.6
    ty = 5.0
    mid = ty + ts / 2
    fy0 = ty + ts + 0.6
    ly = 11.4

    def tile(x, kind):
        if kind == "eye":
            D.rect(x, ty, x + ts, ty + ts, fill=EYE)
        elif kind == "other":
            D.rect(x, ty, x + ts, ty + ts, fill=TINT_BLOOD, edge=BLOOD, width=0.6)
        elif kind == "cbc":
            D.rect(x, ty, x + ts, ty + ts, fill=BLOOD)
        elif kind == "erased":
            D.rect(x, ty, x + ts, ty + ts, fill=WHITE, edge=BLOOD, width=0.8, dash="2px,1.2px")
        elif kind == "estimate":
            D.rect(x, ty, x + ts, ty + ts, edge=BLOOD, width=1.0)
            D.dot(x + ts / 2, mid, BLOOD, size=3.8)

    def flag(x, erased):
        if erased:
            D.rect(x, fy0, x + ts, fy0 + 0.9, fill=WHITE, edge=BLOOD, width=0.6, dash="2px,1.2px")
        else:
            D.rect(x, fy0, x + ts, fy0 + 0.9, fill=SLATE)

    def arrow(x0, x1):
        D.hline(x0, x1 - 1.0, mid, color=INK, width=0.9)
        D.fig.add_shape(type="path", xref=D.canvas["x"], yref=D.canvas["y"], layer="above", fillcolor=INK,
                        line=dict(width=0),
                        path=f"M {x1 - 1.3},{mid - 0.7} L {x1},{mid} L {x1 - 1.3},{mid + 0.7} Z")

    before = ["eye", "other", "other", "cbc", "cbc", "cbc"]
    after = ["eye", "other", "other", "erased", "erased", "erased"]
    gw = len(before) * (ts + tg) - tg
    x1 = 1.0
    for i, k in enumerate(before):
        tile(x1 + i * (ts + tg), k)
        flag(x1 + i * (ts + tg), False)
    D.text(x1 + gw / 2, ly, "Observed", align="center")
    arrow(x1 + gw + 1.3, x1 + gw + 5.4)
    x2 = x1 + gw + 6.8
    for i, k in enumerate(after):
        tile(x2 + i * (ts + tg), k)
        flag(x2 + i * (ts + tg), k == "erased")
    D.text(x2 + gw / 2, ly, "Target erased", align="center")
    arrow(x2 + gw + 1.3, x2 + gw + 5.4)
    bh = 5.0
    x3 = x2 + gw + 6.8
    w3 = tw("BRAN encoder") + 4.4
    D.rect(x3, mid - bh / 2, x3 + w3, mid + bh / 2, fill=WHITE, edge=SHARED, width=1.0)
    D.text(x3 + w3 / 2, mid, "BRAN encoder", align="center")
    arrow(x3 + w3 + 1.3, x3 + w3 + 5.4)
    x4 = x3 + w3 + 6.8
    w4 = tw("Blood count head") + 4.4
    D.rect(x4, mid - bh / 2, x4 + w4, mid + bh / 2, fill=WHITE, edge=SHARED, width=1.0)
    D.text(x4 + w4 / 2, mid, "Blood count head", align="center")
    D.text((x3 + x4 + w4) / 2, ly, "Frozen", align="center")
    arrow(x4 + w4 + 1.3, x4 + w4 + 5.4)
    x5 = x4 + w4 + 6.8
    for i in range(3):
        tile(x5 + i * (ts + tg), "estimate")
    e5 = x5 + 3 * (ts + tg) - tg
    D.text((x5 + e5) / 2, ly, "Scored", align="center")

    col1 = [("eye", "Retina"), ("cbc", "Blood count"), ("flag", "Flag")]
    col2 = [("other", "Other measurement"), ("erased", "Erased")]
    ks = 2.4
    w1 = ks + 1.3 + max(tw(t) for _, t in col1)
    w2 = ks + 1.3 + max(tw(t) for _, t in col2)
    kx2 = W_MM - 1.0 - w2
    kx1 = kx2 - 4.0 - w1
    assert kx1 > e5 + 6.0, ("panel a key collides with the schematic", kx1, e5)
    for kx, col in ((kx1, col1), (kx2, col2)):
        for j, (kind, label) in enumerate(col):
            yc = ty + 0.9 + j * 2.9
            if kind == "flag":
                D.rect(kx, yc - 0.45, kx + ks, yc + 0.45, fill=SLATE)
            elif kind == "eye":
                D.rect(kx, yc - ks / 2, kx + ks, yc + ks / 2, fill=EYE)
            elif kind == "cbc":
                D.rect(kx, yc - ks / 2, kx + ks, yc + ks / 2, fill=BLOOD)
            elif kind == "other":
                D.rect(kx, yc - ks / 2, kx + ks, yc + ks / 2, fill=TINT_BLOOD, edge=BLOOD, width=0.6)
            elif kind == "erased":
                D.rect(kx, yc - ks / 2, kx + ks, yc + ks / 2, fill=WHITE, edge=BLOOD, width=0.8, dash="2px,1.2px")
            D.text(kx + ks + 1.3, yc, label)
    return ly + th("Observed") / 2


B_HEAD = 17.0
H1, H2 = 19.8, 23.2
ROW0, RH, GAP_FAMILY = 25.2, 3.8, 1.2


def row_top(i):
    return ROW0 + i * RH + (GAP_FAMILY if i >= 6 else 0.0)


def row_mid(i):
    return row_top(i) + RH / 2


GRID_BOTTOM = row_top(8) + RH
BC_AX_TOP, BC_AX_BOTTOM = ROW0 - 0.5, GRID_BOTTOM + 0.5
LAB_R = 15.2
CELL0, CW, CGAP, GGAP = 16.3, 8.8, 0.5, 1.5


def cell_x(g, k):
    return CELL0 + g * (2 * CW + GGAP) + k * CW


GRID_RIGHT = cell_x(2, 1) + CW - CGAP
IX0, IX1 = GRID_RIGHT + 4.6, GRID_RIGHT + 25.6


def panel_b(D, V):
    b = V["b"]
    D.heading(0, B_HEAD, "b")
    for g, lab in enumerate(GROUP_LABEL):
        gx0, gx1 = cell_x(g, 0), cell_x(g, 1) + CW - CGAP
        D.text((gx0 + gx1) / 2, H1, lab, size=SZ["tick"], align="center")
        D.hline(gx0 + 0.3, gx1 - 0.3, H1 + 1.7, color=GREY, width=0.6)
        D.text(cell_x(g, 0) + (CW - CGAP) / 2, H2, "with", size=SZ["tick"], align="center", color=SLATE)
        D.text(cell_x(g, 1) + (CW - CGAP) / 2, H2, "without", size=SZ["tick"], align="center", color=SLATE)
    D.text(LAB_R, H2, "Retina", size=SZ["tick"], align="right", color=SLATE)
    outlined = {tuple(x) for x in b["outlined"]}
    n_out = 0
    for i, f in enumerate(FIELDS):
        y0, y1 = row_top(i) + CGAP / 2, row_top(i) + RH - CGAP / 2
        D.text(LAB_R, row_mid(i), LABEL[f], align="right")
        for g, (ctx_r, ctx_n) in enumerate(CONTEXTS):
            for k, ctx in enumerate((ctx_r, ctx_n)):
                x0 = cell_x(g, k)
                x1 = x0 + CW - CGAP
                r = b["ratio"][ctx][f]
                if r is None:
                    D.rect(x0, y0, x1, y1, fill=WHITE, edge=FAINT, width=0.75)
                    D.text((x0 + x1) / 2, row_mid(i), "–", size=SZ["tick"], color=SLATE, align="center")
                    continue
                D.rect(x0, y0, x1, y1, fill=grid_fill(r))
                if (ctx, f) in outlined:
                    D.rect(x0 + 0.3, y0 + 0.3, x1 - 0.3, y1 - 0.3, edge=INK, width=0.9)
                    n_out += 1
                D.text((x0 + x1) / 2, row_mid(i), f"{r:.2f}", size=SZ["tick"], align="center")
    assert n_out == len(outlined) == 15, n_out

    # Key under the grid. The colour bar is drawn as vector steps of the exact grid_fill ramp.
    ky = GRID_BOTTOM + 3.6
    D.text(0, ky, "Error / median", size=SZ["tick"])
    lo_r, hi_r = 0.35, 1.0
    assert grid_fill(lo_r) == "#8F99A3" and grid_fill(hi_r) == "#FFFFFF"
    bx0 = tw("Error / median", SZ["tick"]) + 1.4
    bx1 = bx0 + 17.0
    kax = D.new_axes(bx0, bx1, ky - 1.0, ky + 1.0, xrange=[lo_r, hi_r], yrange=[0, 1],
                     xticks=[0.4, 0.7, 1.0], xtext=["0.4", "0.7", "1.0"], xline=False)
    nstep = 60
    for j in range(nstep):
        a_, b_ = lo_r + (hi_r - lo_r) * j / nstep, lo_r + (hi_r - lo_r) * (j + 1) / nstep
        D.rect(a_, 0, b_ + (hi_r - lo_r) / nstep * 0.02, 1, fill=grid_fill((a_ + b_) / 2), ax=kax)
    D.rect(bx0, ky - 1.0, bx1, ky + 1.0, edge=FAINT, width=0.5)
    kx = bx1 + 4.2
    D.rect(kx, ky - 1.1, kx + 3.6, ky + 1.1, fill=WHITE, edge=FAINT, width=0.75)
    D.text(kx + 1.8, ky, "–", size=SZ["tick"], color=SLATE, align="center")
    D.text(kx + 4.6, ky, "Not a target", size=SZ["tick"])
    ky2 = ky + 3.3
    D.rect(kx, ky2 - 1.1, kx + 3.6, ky2 + 1.1, fill=grid_fill(0.85))
    D.rect(kx + 0.3, ky2 - 0.8, kx + 3.3, ky2 + 0.8, edge=INK, width=0.9)
    D.text(kx + 4.6, ky2, "Retina raised error", size=SZ["tick"])
    assert kx + 4.6 + tw("Retina raised error", SZ["tick"]) < IX0 - 2.0, "grid key runs into the increment axis"

    # Paired retinal increment with the whole panel hidden, with retina minus without.
    D.text((IX0 + IX1) / 2, H1, "Retina increment", size=SZ["tick"], align="center")
    D.hline(IX0 + 0.3, IX1 - 0.3, H1 + 1.7, color=GREY, width=0.6)
    ticks = [-4, -2, 0, 2]
    iax = D.new_axes(IX0, IX1, BC_AX_TOP, BC_AX_BOTTOM, xrange=list(INC_LIM), xticks=ticks,
                     xtext=[signed(t) for t in ticks])
    D.vref(iax, 0.0, color=INK)
    for i, f in enumerate(FIELDS):
        est, lo, hi = b["inc_pct"][f]
        D.hci(iax, row_mid(i), est, lo, hi, EYE, hollow=(interval_class(lo, hi) == "zero"), name=f"b inc {f}")
    D.xtitle(iax, "Change (% of<br>median error)")
    return dict(iax=iax, key_bottom=ky2 + th("Retina raised error", SZ["tick"]) / 2)


C_X = IX1 + 5.6
CAX0, CAX1 = C_X + 9.6, W_MM - 1.5


def panel_c(D, V):
    c = V["c"]
    D.heading(C_X, B_HEAD, "c")
    kx = D.key_tick(CAX0, H1, "Age only")
    D.key_dot(kx, H1, EYE, "Retina and age")
    kx = D.key_dot(CAX0, H2, INK, "Excludes 0")
    D.key_dot(kx, H2, INK, "Includes 0", hollow=True)
    ticks = [0.90, 0.95, 1.00, 1.05]
    cax = D.new_axes(CAX0, CAX1, BC_AX_TOP, BC_AX_BOTTOM, xrange=list(C_LIM), xticks=ticks,
                     xtext=[f"{t:.2f}" for t in ticks])
    D.vref(cax, 1.0, color=GREY, dash=DASH)
    for i, f in enumerate(FIELDS):
        y = row_mid(i)
        q = c[f]
        D.text(CAX0 - 1.8, y, SHORT[f], align="right")
        D.fig.add_trace(go.Scatter(x=[q["r_iv"]], y=[y], xaxis=cax["x"], yaxis=cax["y"], mode="markers",
                                   hoverinfo="skip", cliponaxis=False, showlegend=False, name=f"c age {f}",
                                   marker=dict(symbol="line-ns", size=MS_TICK, line=dict(color=GREY, width=1.2))))
        # The line reaches the grey tick exactly when the paired interval includes zero.
        assert (q["lo"] <= q["r_iv"] <= q["hi"]) == q["zero"], f
        D.hci(cax, y, q["r_iii"], q["lo"], q["hi"], EYE, hollow=q["zero"], name=f"c eye {f}")
    D.xtitle(cax, "Error relative to training median")
    yph = BC_AX_BOTTOM + 10.6
    D.placeholder(CAX0, yph, tw("PLACEHOLDER", SZ["tick"]) + 1.8)
    D.text(CAX0 + tw("PLACEHOLDER", SZ["tick"]) + 3.6, yph, "RETFound-Green and age")
    return dict(cax=cax, bottom=yph + 1.5)


DE_HEAD = 77.4
D_RIGHT = 80.6
E_LEFT = D_RIGHT + 5.2
REF_SETTINGS = [("bran_i_cbc_hidden_with_retina", "BRAN with retina", SHARED),
                ("bran_iii_retina_and_age_only", "Retina and age", EYE),
                ("training_median", "Training median", GREY)]
RD_LAB_R, RS0 = 7.8, 9.4
RD_K1, RD_K2 = 80.6, 84.0
RD_ROW0, RD_P, RD_GAP = 85.8, 3.3, 1.0
BR_DY, BR_CAP, BR_W = 0.62, 0.22, 1.1
BR_LANE = BR_DY + BR_CAP
STRIP_IN, STRIP_OUT = 0.98, 1.38
SEX_GLYPH_W = 2.8
MAE_DP = 3


def rd_mid(i):
    return RD_ROW0 + i * RD_P + (RD_GAP if i >= 6 else 0.0) + RD_P / 2


RD_BOTTOM = rd_mid(8) + RD_P / 2


def unit_html(u):
    """Units with a superscript exponent. Plotly sets a superscript at 70% of the line, which would print the exponent
    of 6 pt units at 4.2 pt, below the Nature Medicine minimum of 5 pt, so the exponent carries the profile's 5 pt
    size explicitly."""
    return re.sub(r"10\^(\d)", rf'10<sup><span style="font-size:{SZ["small"]:.2f}px">\1</span></sup>', u)


def panel_d(D, V):
    dd = V["d"]
    ref, mae = dd["ref"], dd["mae"]
    D.heading(0, DE_HEAD, "d")
    T = SZ["tick"]
    units = {f: unit_html(ref[f]["unit"]) for f in FIELDS}
    x_unit = D_RIGHT - max(tw(u, T) for u in units.values())
    x_mae = x_unit - 1.2
    x_glyph1 = x_mae - tw("MAE", T) - 0.9
    x_glyph0 = x_glyph1 - 2.8
    x_male = x_glyph0 - 1.4
    sex_txt = {f: {s: f"{a}–{b}" for s, a, b in ref[f]["bands"]} for f in FIELDS}
    x_fem = x_male - max(tw(t["male"], T) for t in sex_txt.values() if "male" in t) - 2.0
    x_fem_left = x_fem - max(tw(t["female"], T) for t in sex_txt.values() if "female" in t)
    RS1 = x_fem_left - 2.0
    assert RS1 - RS0 > 26.0, ("strip too narrow", RS0, RS1)
    x_both = (x_fem_left + x_male) / 2
    for t in sex_txt.values():
        if "both" in t:
            assert x_both - tw(t["both"], T) / 2 > RS1 + 1.0, t

    kx = RS0
    for _, lab, col in REF_SETTINGS:
        D.bracket(D.canvas, kx, kx + 2.8, RD_K1, col, BR_CAP, BR_W)
        D.text(kx + 3.8, RD_K1, lab, size=T)
        kx += 3.8 + tw(lab, T) + 2.6
    assert kx - 2.6 < D_RIGHT, ("panel d key runs into panel e", kx)
    D.rect(RS0, RD_K2 - 0.9, RS0 + 2.8, RD_K2 + 0.9, fill=LIGHT)
    D.text(RS0 + 3.8, RD_K2, "Adult interval", size=T)
    D.text(x_fem, RD_K2, "Female", size=T, color=SLATE, align="right")
    D.text(x_male, RD_K2, "Male", size=T, color=SLATE, align="right")
    glyph_x = {}
    for s, xr, lab in (("female", x_fem, "Female"), ("male", x_male, "Male")):
        g1 = xr - tw(lab, T) - 0.8
        g0 = g1 - SEX_GLYPH_W
        top, bottom = (RD_K2 - STRIP_OUT, RD_K2 - STRIP_IN), (RD_K2 + STRIP_IN, RD_K2 + STRIP_OUT)
        filled, empty = (top, bottom) if s == "female" else (bottom, top)
        D.rect(g0, filled[0], g1, filled[1], fill=LIGHT)
        D.rect(g0, empty[0], g1, empty[1], edge=FAINT, width=0.5)
        glyph_x[s] = (g0, g1)
    assert RS0 + 3.8 + tw("Adult interval", T) + 1.2 < glyph_x["female"][0], "female glyph meets the band key"
    assert x_fem + 0.8 < glyph_x["male"][0], "male glyph meets the Female header"
    D.bracket(D.canvas, x_glyph0, x_glyph1, RD_K2, SHARED, BR_CAP, BR_W)
    D.text(x_mae, RD_K2, "MAE", size=T, color=SLATE, align="right")

    n_word, drawn, slots = 0, {}, []
    for i, f in enumerate(FIELDS):
        y = rd_mid(i)
        r = ref[f]
        D.text(RD_LAB_R, y, SHORT[f], align="right")
        pad = 0.08 * r["span"]
        ax = D.new_axes(RS0, RS1, y - RD_P / 2, y + RD_P / 2, xrange=[r["lo"] - pad, r["hi"] + pad], xline=False)
        for s, a, b in r["bands"]:
            y0, y1 = {"both": (y - STRIP_OUT, y + STRIP_OUT), "female": (y - STRIP_OUT, y - STRIP_IN),
                      "male": (y + STRIP_IN, y + STRIP_OUT)}[s]
            D.band(ax, float(a), float(b), y0, y1)
        for j, (k, _, col) in enumerate(REF_SETTINGS):
            v = mae[k][f]
            yj = y + (j - 1) * BR_DY
            if v is None:
                D.rect(r["lo"], yj - BR_CAP, r["hi"], yj + BR_CAP, edge=GREY, width=0.6, dash="2px,1.5px", ax=ax)
                slots.append([k, f])
                continue
            lo, hi = r["mid"] - v[0], r["mid"] + v[0]
            assert ax["xr"][0] < lo and hi < ax["xr"][1], ("bracket outside the row axis", f, k)
            D.bracket(ax, lo, hi, yj, col, BR_CAP, BR_W)
            drawn[(k, f)] = (lo, hi, yj)
        t = sex_txt[f]
        if "both" in t:
            D.text(x_both, y, t["both"], size=T, align="center")
        else:
            D.text(x_fem, y, t["female"], size=T, align="right")
            D.text(x_male, y, t["male"], size=T, align="right")
        v = mae[REF_SETTINGS[0][0]][f]
        if v is None:
            assert D_RIGHT - tw("PLACEHOLDER", T) > x_male + 1.0
            D.text(D_RIGHT, y, "PLACEHOLDER", size=T, color=GREY, align="right")
            n_word += 1
        else:
            D.text(x_mae, y, f"{v[0]:.{MAE_DP}f}", size=T, align="right")
            D.text(x_unit, y, units[f], size=T)
    assert sorted(slots) == sorted(dd["missing"]), (slots, dd["missing"])
    half_line = BR_W / st.MM / 2
    assert BR_DY - BR_CAP > BR_CAP
    assert STRIP_IN - max(BR_LANE, BR_DY + half_line) >= 0.12, "a bracket reaches a sex strip"
    assert STRIP_IN - BR_LANE < RD_P - 2 * STRIP_OUT and STRIP_OUT < RD_P / 2 - 0.25
    for (k, f), (lo, hi, yj) in drawn.items():
        yc = rd_mid(FIELDS.index(f))
        assert abs(yj - yc) + BR_CAP <= BR_LANE + 1e-9, (k, f)
        if len(ref[f]["bands"]) == 2:
            for s0, s1 in ((yc - STRIP_OUT, yc - STRIP_IN), (yc + STRIP_IN, yc + STRIP_OUT)):
                assert yj + BR_CAP < s0 or yj - BR_CAP > s1, ("bracket inside a sex strip", k, f)
    assert RD_BOTTOM + 2.5 < FG_HEAD - SZ["letter"] * 1.2 / st.MM, ("panel d runs into the f heading", RD_BOTTOM)
    return dict(n_word=n_word, slots=slots)


RIDGE_DASH, RIDGE_DOT = "3px,1.6px", "0.9px,1.4px"
E_FLAG = [("eye_plus_remaining", "BRAN with retina", SHARED, None, 1.3),
          ("remaining_no_eye", "BRAN without retina", BLOOD, None, 1.3),
          ("eye_and_age", "Retina and age", EYE, None, 1.3),
          ("ridge_remaining_plus_retinal_pca64", "Ridge, clinical, retina, age", GREY, RIDGE_DASH, 1.1),
          ("ridge_retinal_pca64_and_age", "Ridge, retina and age", GREY, RIDGE_DOT, 1.1)]
E_ROW_Y = [83.9, 87.9, 91.9, 96.5, 100.5]
E_TOP, E_BOTTOM = 81.6, 104.0
DC_X, DC_Y = (0.045, 0.31), (-0.024, 0.062)
FOUND_LIM = (12.0, 45.0)
FOUND_W = 13.0
E_LINE_X = (0.2, 3.4)
E_LAB_DX = 4.4
T0 = 0.10


def panel_e(D, V):
    ev = V["e"]
    E_X = E_LEFT
    D.heading(E_X, DE_HEAD, "e")
    lab_x = E_X + E_LAB_DX
    f0 = lab_x + max(tw(lab) for _, lab, *_ in E_FLAG) + 1.4
    f1 = f0 + FOUND_W
    c1 = W_MM - 1.5
    c0 = f1 + 1.0 + tw("0.06", SZ["tick"]) + 1.3 + 0.9
    assert c1 - c0 > 31.0, ("decision curve too narrow", c0, c1)
    for (k, lab, col, dash, wd), y in zip(E_FLAG, E_ROW_Y):
        D.hline(E_X + E_LINE_X[0], E_X + E_LINE_X[1], y, color=col, width=wd, dash=dash)
        D.text(lab_x, y, lab)
    fax = D.new_axes(f0, f1, E_TOP, E_BOTTOM, xrange=list(FOUND_LIM), xticks=[20, 30, 40], xtext=["20", "30", "40"])
    for (k, lab, col, *_), y in zip(E_FLAG, E_ROW_Y):
        est, lo, hi = ev["found"][k]
        D.hci(fax, y, est, lo, hi, col, name=f"e found {k}")
    D.xtitle(fax, "Found per 1,000<br>at 90% specificity")
    yt = [0.0, 0.02, 0.04, 0.06]
    cax = D.new_axes(c0, c1, E_TOP, E_BOTTOM, xrange=list(DC_X), yrange=list(DC_Y),
                     xticks=[0.05, 0.10, 0.20, 0.30], xtext=["0.05", "0.10", "0.20", "0.30"],
                     yticks=yt, ytext=["0" if t == 0 else f"{t:.2f}" for t in yt], yline=True)
    D.href(cax, 0.0, color=INK)
    D.vref(cax, T0, color=LIGHT, width=1.2)
    curves = {}
    for k, lab, col, dash, wd in E_FLAG[::-1]:
        pts = ev["curve"][k]
        assert all(DC_Y[0] < v < DC_Y[1] for _, v in pts), k
        curves[k] = pts
        D.fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], xaxis=cax["x"], yaxis=cax["y"],
                                   mode="lines", hoverinfo="skip", showlegend=False, cliponaxis=True,
                                   name=f"e curve {k}", line=dict(color=col, width=wd, dash=dash or "solid")))
    ta = ev["treat_all"]
    curves["treat_all"] = ta
    D.fig.add_trace(go.Scatter(x=[p[0] for p in ta], y=[p[1] for p in ta], xaxis=cax["x"], yaxis=cax["y"],
                               mode="lines", hoverinfo="skip", showlegend=False, cliponaxis=True,
                               name="e curve treat all", line=dict(color=INK, width=0.9)))
    for k, _, col, *_ in E_FLAG[:2]:
        D.dot(T0, ev["dot"][k], col, size=3.8, ax=cax, name=f"e dot {k}")
    boxes = {"treat all": D.text(D.px(cax, 0.118), D.py(cax, -0.0175), "Treat all", size=SZ["tick"]),
             "treat none": D.text(c1 - 0.4, D.py(cax, -0.0072), "Treat none", size=SZ["tick"], align="right")}
    assert boxes["treat none"][1] > D.py(cax, 0.0) + 0.2, "Treat none label meets the zero line"
    assert boxes["treat all"][3] < E_BOTTOM - 0.2 and boxes["treat all"][2] < boxes["treat none"][0] - 1.0
    for name, b in boxes.items():
        assert b[2] < c1 - 0.2 and b[0] > c0 + 0.2, ("label leaves the decision-curve axis", name)
        for k, pts in curves.items():
            assert not seg_hits(cax, pts, b), ("curve crosses a label", name, k)
    D.text(c0 - 1.3 - tw("0.06", SZ["tick"]), E_TOP - 2.7, "Net benefit")
    D.xtitle(cax, "Risk threshold")
    return dict(cax=cax, fax=fax, x=E_X, bottom=E_BOTTOM + 4.4 + th("Found per 1,000<br>at 90% specificity"))


def seg_hits(ax, pts, box, pad=0.3):
    """True when the polyline pts (data units on ax), clipped to the axis box, enters the mm box."""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    bx0, bx1, by0, by1 = ax["box"]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        for s in range(41):
            u = s / 40
            xm, ym = Draw.px(ax, xa + u * (xb - xa)), Draw.py(ax, ya + u * (yb - ya))
            if not (bx0 <= xm <= bx1 and by0 <= ym <= by1):
                continue
            if x0 < xm < x1 and y0 < ym < y1:
                return True
    return False


FG_HEAD = 122.6
F_TOP, F_BOTTOM = 125.4, 156.4


def panel_f(D, V):
    fv = V["f"]
    D.heading(0, FG_HEAD, "f")
    sel = fv["sel"]
    a, s80, s60, r80, r60 = (sel[k] for k in ("all", "select80", "select60", "random80", "random60"))
    assert all(0.76 < p[2] and p[3] < 1.02 for p in sel.values())
    l0, l1 = 9.8, 29.4
    yt = [0.80, 0.85, 0.90, 0.95, 1.00]
    lax = D.new_axes(l0, l1, F_TOP, F_BOTTOM, xrange=[55, 105], yrange=[0.76, 1.045], xticks=[60, 80, 100],
                     xtext=["60", "80", "100"], yticks=yt, ytext=[f"{t:.2f}" for t in yt], yline=True)
    off = 2.2
    pct = 100.0
    D.fig.add_trace(go.Scatter(x=[a[0] * pct, r80[0] * pct + off, r60[0] * pct + off], y=[a[1], r80[1], r60[1]],
                               xaxis=lax["x"], yaxis=lax["y"], mode="lines", hoverinfo="skip", showlegend=False,
                               line=dict(color=GREY, width=0.9, dash=DASH), name="f random line"))
    D.fig.add_trace(go.Scatter(x=[a[0] * pct, s80[0] * pct, s60[0] * pct], y=[a[1], s80[1], s60[1]], xaxis=lax["x"],
                               yaxis=lax["y"], mode="lines", line=dict(color=SHARED, width=1.3), hoverinfo="skip",
                               showlegend=False, name="f narrowest line"))
    for p in (r80, r60):
        D.vci(lax, p[0] * pct + off, p[1], p[2], p[3], GREY, hollow=True)
    for p in (a, s80, s60):
        D.vci(lax, p[0] * pct, p[1], p[2], p[3], SHARED)
    D.xtitle(lax, "Share kept (%)")
    D.text(1.6, (F_TOP + F_BOTTOM) / 2, "Hb MAE (g/dL)", align="center", angle=-90)
    rb = D.text(D.px(lax, r60[0] * pct + off) - 0.8, D.py(lax, r60[3]) - 0.9, "Random", size=SZ["tick"],
                valign="bottom")
    assert rb[0] > l0 + 0.6, "Random label crosses the y axis"
    nb = D.text(D.px(lax, s60[0] * pct) + 1.4, D.py(lax, 0.779), "Narrowest kept", size=SZ["tick"])
    assert nb[1] > D.py(lax, s80[2]) + 0.3, "label meets the 81% interval"
    assert nb[2] < l1 - 0.2, "Narrowest kept leaves the axis"

    m_lab_r = 37.6
    m0, m1 = m_lab_r + 1.8, m_lab_r + 22.6
    pitch, fam = 3.25, 1.0
    y_rows = [F_TOP + 2.0 + i * pitch + (fam if i >= 6 else 0.0) for i in range(9)]
    assert y_rows[-1] + 1.5 < F_BOTTOM
    mt = [0.85, 0.90, 0.95]
    max_ = D.new_axes(m0, m1, F_TOP, F_BOTTOM, xrange=[0.84, 0.96], xticks=mt, xtext=[f"{t:.2f}" for t in mt])
    D.vref(max_, 0.90, color=GREY, dash=DASH)
    for f, y in zip(FIELDS, y_rows):
        nn, q = fv["strip"][f]["native"], fv["strip"][f]["quantile"]
        D.text(m_lab_r, y, SHORT[f], size=SZ["tick"], align="right")
        D.hci(max_, y - 0.65, *nn, SHARED, hollow=True, size=4.0)
        D.hci(max_, y + 0.65, *q, SHARED, size=4.0)
    D.xtitle(max_, "Interval coverage")

    r_lab_r = m1 + 4.4 + tw("Without retina")
    r0, r1 = r_lab_r + 1.8, G_X - 4.6
    assert r1 - r0 > 15.0, (r0, r1)
    kx = r_lab_r - tw("Without retina")
    D.key_dot(kx, F_TOP + 0.4, SHARED, "Native intervals", hollow=True)
    D.key_dot(kx, F_TOP + 3.8, SHARED, "Quantile attachment")
    rt = F_TOP + 11.6
    D.text((kx + r1) / 2, rt - 2.0, "Hb below 12 g/dL", align="center", valign="bottom")
    rticks = [0.2, 0.6, 1.0]
    rax = D.new_axes(r0, r1, rt, F_BOTTOM, xrange=[0.2, 1.0], xticks=rticks, xtext=[f"{t:.1f}" for t in rticks])
    D.vref(rax, 0.90, color=GREY, dash=DASH)
    rows = [("whole_cbc_hidden", SHARED, "With retina"), ("whole_cbc_no_retina", BLOOD, "Without retina")]
    ry = [rt + (F_BOTTOM - rt) * 0.30, rt + (F_BOTTOM - rt) * 0.70]
    for (ctx, col, lab), y in zip(rows, ry):
        nn, q = fv["lowcov"][ctx]["native"], fv["lowcov"][ctx]["quantile"]
        assert q[2] < 0.85 and 0.2 < nn[1]
        D.text(r_lab_r, y, lab, align="right")
        D.hci(rax, y - 0.85, *nn, col, hollow=True)
        D.hci(rax, y + 0.85, *q, col)
    D.xtitle(rax, "Interval coverage")


G_ROWS = [("native_with_retina", "With retina", SHARED, False),
          ("native_no_retina", "Without retina", BLOOD, False),
          ("median", "Training median", GREY, True)]
G_X = 105.0
H_X = 144.6


def panel_g(D, V):
    gv = V["g"]
    D.heading(G_X, FG_HEAD, "g")
    lab_r = G_X + tw("Training median") + 0.6
    g0, g1 = lab_r + 1.8, H_X - 4.0
    assert g1 - g0 > 14.0, (g0, g1)
    rows = [F_TOP + 4.0 + i * 10.0 for i in range(3)]
    ticks = [2.0, 2.4, 2.8]
    ax = D.new_axes(g0, g1, F_TOP, F_BOTTOM, xrange=[1.9, 2.85], xticks=ticks, xtext=[f"{t:.1f}" for t in ticks])
    for (k, lab, col, hol), y in zip(G_ROWS, rows):
        D.text(lab_r, y, lab, align="right")
        est, lo, hi = gv["lowarm"][k]
        D.hci(ax, y, est, lo, hi, col, hollow=hol, name=f"g {k}")
    D.xtitle(ax, "Hb MAE (g/dL)")


H_HB_LIM, H_NB_LIM = (-0.11, 0.01), (-0.04, 0.035)
H_ROWS_HB = [("dev", "Development", "circle", 8.0), ("test", "Test, five-fold", "diamond", 5.6),
             ("test_f0", "Test, fold 0†", "diamond", 5.6)]
H_ROWS_NB = [("dev", "Development", "circle", 8.0), ("test_deployed", "Test, fold 0", "diamond", 5.6),
             ("test_fivefold", "Test, five-fold†", "diamond", 5.6)]
H_PITCH = 2.8
H_HB_TITLE = "Hb MAE<br>difference (g/dL)"
H_NB_TITLE = "Net benefit<br>difference at 0.10"


def panel_h(D, V):
    hv = V["h"]
    D.heading(H_X, FG_HEAD, "h")
    T = SZ["tick"]
    D.text(H_X, F_TOP + 0.6, "With minus without retina")
    lab_r = H_X + max(tw(lab, T) for _, lab, *_ in H_ROWS_HB + H_ROWS_NB) + 0.4
    a0, a1 = lab_r + 1.8, W_MM - 1.5
    assert a1 - a0 > 15.0, (a0, a1)
    span = H_PITCH * (len(H_ROWS_HB) - 1) + 3.0
    top0 = F_TOP + 2.6
    top1 = top0 + span
    hax = D.new_axes(a0, a1, top0, top1, xrange=list(H_HB_LIM), xticks=[-0.1, -0.05, 0.0],
                     xtext=[f"{MINUS}0.1", f"{MINUS}0.05", "0"])
    D.vref(hax, 0.0, color=INK)
    for j, (k, lab, sym, size) in enumerate(H_ROWS_HB):
        y = top0 + 1.5 + j * H_PITCH
        est, lo, hi = hv["hb"][k]
        D.text(lab_r, y, lab, size=T, align="right")
        D.hci(hax, y, est, lo, hi, EYE, hollow=(interval_class(lo, hi) == "zero"), name=f"h hb {k}", symbol=sym,
              size=size * 0.62 if sym == "circle" else size)
    hx = min((a0 + a1) / 2, W_MM - tw(H_HB_TITLE) / 2 - 0.1)
    tb = D.xtitle(hax, H_HB_TITLE, dy=4.0, x=hx)
    bot1 = F_BOTTOM
    bot0 = bot1 - span
    assert tb[3] + 1.0 < bot0, ("h upper axis title meets the lower axis", tb[3], bot0)
    nax = D.new_axes(a0, a1, bot0, bot1, xrange=list(H_NB_LIM), xticks=[-0.02, 0.0, 0.02],
                     xtext=[f"{MINUS}0.02", "0", "+0.02"])
    D.vref(nax, 0.0, color=INK)
    for j, (k, lab, sym, size) in enumerate(H_ROWS_NB):
        y = bot0 + 1.5 + j * H_PITCH
        est, lo, hi = hv["nb"][k]
        D.text(lab_r, y, lab, size=T, align="right")
        D.hci(nax, y, est, lo, hi, EYE, hollow=(interval_class(lo, hi) == "zero"), name=f"h nb {k}", symbol=sym,
              size=size * 0.62 if sym == "circle" else size)
    nx = min((a0 + a1) / 2, W_MM - tw(H_NB_TITLE) / 2 - 0.1)
    D.xtitle(nax, H_NB_TITLE, x=nx)
    for x_, t_ in ((hx, H_HB_TITLE), (nx, H_NB_TITLE)):
        assert abs(x_ - (a0 + a1) / 2) < 1.0, ("h axis title drifts from its axis", t_, x_)
    return dict(hax=hax, nax=nax)


def audit(D, n_placeholder):
    anns = list(D.fig.layout.annotations)
    plain = [re.sub(r"<[^>]+>", " ", a.text).replace("&nbsp;", " ") for a in anns]
    blob = "\n".join(plain)
    for a in anns:
        if "font-size" in a.text:
            continue
        assert a.font.size in SZ.values(), (a.text, a.font.size)
        assert a.font.color in TEXT_COLOURS or (a.text == "PLACEHOLDER" and a.font.color == GREY), (a.text, a.font.color)
    ridge = [p for p in plain if "idge" in p]
    assert ridge == [lab for k, lab, *_ in E_FLAG if k.startswith("ridge")], ridge
    for m in re.finditer(r"RETFound", blob):
        assert blob[m.end():m.end() + 6] == "-Green", "Name RETFound-Green, never RETFound alone"
    assert [p for p in plain if "PLACEHOLDER" in p] == ["PLACEHOLDER"] * (1 + n_placeholder)
    letters = re.findall(r"<b>([a-h])</b></span>&nbsp;", "\n".join(a.text for a in anns))
    assert letters == list("abcdefgh"), letters
    assert not any(re.search(r"(Figure|Fig\.)\s*\d", p) for p in plain), "no title or figure number on the canvas"
    for ax_name in [k for k in D.fig.layout if k.startswith(("xaxis", "yaxis"))]:
        assert not D.fig.layout[ax_name].showgrid, ax_name
    for tr in D.fig.data:
        assert tr.hoverinfo == "skip", tr.type
        assert tr.type == "scatter" and tr.showlegend is False, tr.type
    for b in D.boxes:
        assert b[0] >= -0.05 and b[2] <= W_MM + 0.05 and b[1] >= -T_PX / st.MM and b[3] <= PH_MM + 0.05, \
            ("off canvas", b)
    for i in range(len(D.boxes)):
        for j in range(i + 1, len(D.boxes)):
            a, b = D.boxes[i], D.boxes[j]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            assert not (ox > 0.15 and oy > 0.15), ("text overlap", a, b)
    return blob


def draw(V):
    """The whole figure from the drawing input alone. Returns (figure, width px, height px)."""
    D = Draw()
    D.fig.update_layout(width=W_PX, height=H_PX, margin=dict(l=SIDE_PX, r=SIDE_PX, t=T_PX, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="white", showlegend=False, hovermode=False,
                        dragmode=False)
    a_bottom = panel_a(D)
    assert a_bottom < B_HEAD - SZ["letter"] * 1.2 / st.MM - 0.6, a_bottom
    pb = panel_b(D, V)
    pc = panel_c(D, V)
    assert max(pb["key_bottom"], pc["bottom"]) < DE_HEAD - SZ["letter"] * 1.2 / st.MM - 0.6
    pd = panel_d(D, V)
    pe = panel_e(D, V)
    assert pe["bottom"] < FG_HEAD - SZ["letter"] * 1.2 / st.MM - 0.6, pe["bottom"]
    panel_f(D, V)
    panel_g(D, V)
    panel_h(D, V)
    bottom = max(b[3] for b in D.boxes)
    assert bottom <= PH_MM - 0.2, ("paper too short", bottom, PH_MM)
    assert PH_MM - bottom < 1.5, ("paper taller than its content", bottom, PH_MM)
    D.blob = audit(D, pd["n_word"])
    D.parts = dict(e=pe, h=None)
    return D


def draw_pages(V):
    D = draw(V)
    return [(FIG_ID, D.fig, W_PX, H_PX)]


def render(payload):
    return draw_pages(payload['draw'])

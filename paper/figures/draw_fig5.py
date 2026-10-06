"""Drawing code for Fig. 5.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_fig5.py
(SHA-256 70631e41a7041a2744eeb2c073affceb586dc90b150c39299690a7a82afa321c).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import math
import re
import plotly.graph_objects as go
import bran_plotly_style as st
st.use_nature_medicine()
import display_numbers as dn
FIG_ID = "fig5"
DISPLAY = dn.label(FIG_ID)
MINUS = "−"
FS = st.SIZES
INK, GREY = st.INK, st.GREY
TEXT_COLOURS = {INK, GREY}
COLOUR = {"shared": st.SHARED, "blood": st.BLOOD, "eye": st.EYE, "grey": st.GREY, "ink": st.INK}
MARK_COLOURS = set(COLOUR.values())
YES, PART, NO, NA = "yes", "partial", "no", "not applicable"
CELLS = ["blood", "other", "retina", "enc", "scored"]
ARMS = ["BRAN", "GBM", "MICE_LR", "MEDIAN_LR", "LATE_FUSION"]
COMPS = ARMS[1:]
NAME = {"BRAN": "BRAN", "GBM": "Gradient boosting", "MICE_LR": "MICE", "MEDIAN_LR": "Median imputation",
        "LATE_FUSION": "Late fusion"}
SYMBOL = {"BRAN": "circle", "GBM": "square", "MICE_LR": "triangle-up", "MEDIAN_LR": "diamond", "LATE_FUSION": "cross"}
E_SHAPE = {"bran_state_logistic": "BRAN", "raw_gbm_masktrained": "GBM", "raw_logistic_masktrained": "MEDIAN_LR"}
E_NAME = {"bran_state_logistic": "BRAN", "raw_gbm_masktrained": "Gradient boosting",
          "raw_logistic_masktrained": "Raw logistic"}


def sg(x, nd=3):
    """Signed estimate, a negative zero printed as +0.000 as in the source reports."""
    t = f"{x:+.{nd}f}"
    if t == "-0." + "0" * nd:
        t = "+0." + "0" * nd
    return t.replace("-", MINUS)


def ci(lo, hi, nd=3):
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]".replace("-", MINUS)


def sgc(t):
    return f"{sg(t[0])} {ci(t[1], t[2])}"


E_ROW_LE, E_ROW_HB = 86, 151
ML, MR, MT, MB = 4, 4, 4, 4
W = st.FULL_W
PW = W - ML - MR
ROW_A, ROW_B, ROW_C, ROW_D = 0, 152, 304, 492
PH = ROW_D + 140
H = PH + MT + MB
assert W <= st.NM_W and H <= st.NM_MAX_H, (W, H)
HEAD = 11
LINE_GAP = 1.25
SIZE_BRAN, SIZE_BASE = 6.4, 5.2
SIZE_ADJ = {"circle": 1.0, "square": 0.9, "triangle-up": 1.1, "diamond": 1.15, "cross": 1.05}
HIDDEN = dict(visible=False, showline=False, ticks="", showticklabels=False)


def strip(s):
    return re.sub(r"<[^>]+>", " ", s).replace("&nbsp;", " ")


def msize(arm):
    return (SIZE_BRAN if arm == "BRAN" else SIZE_BASE) * SIZE_ADJ[SYMBOL[arm]]


def mcolour(arm):
    return st.SHARED if arm == "BRAN" else GREY


def tickfmt(nd):
    def f(t):
        if abs(t) < 1e-12:
            return "0"
        return f"{t:.{nd}f}".replace("-", MINUS)
    return f


class Canvas:
    """One Plotly figure on a pixel grid. Axis 1 is a hidden canvas in pixel units, origin top left. Every text ink
    box is recorded for the overlap audit."""

    def __init__(self):
        self.fig = go.Figure()
        self.fig.update_layout(template="bran", width=W, height=H, margin=dict(l=ML, r=MR, t=MT, b=MB),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="white", hovermode=False,
                               dragmode=False, showlegend=False)
        self.AX: dict[str, str] = {}
        self.TXT: list[tuple[str, str]] = []
        self.BOXES: list[tuple[float, float, float, float, str]] = []
        self.axes("canvas", (0, 0, PW, PH), (0, PW), (PH, 0), x=HIDDEN, y=HIDDEN)

    # geometry
    @staticmethod
    def px(x):
        return x / PW

    @staticmethod
    def py(y):
        return 1 - y / PH

    def axes(self, key, box, xr, yr, x=None, y=None):
        n = len(self.AX) + 1
        sx = "" if n == 1 else str(n)
        x0, y0, x1, y1 = box
        assert -0.5 <= x0 < x1 <= PW + 0.5 and -0.5 <= y0 < y1 <= PH + 0.5, (key, box)
        xa = dict(domain=[self.px(x0), self.px(x1)], range=list(xr), anchor=f"y{sx}", fixedrange=True)
        ya = dict(domain=[self.py(y1), self.py(y0)], range=list(yr), anchor=f"x{sx}", fixedrange=True)
        xa.update(x or {})
        ya.update(y or {})
        self.fig.update_layout({f"xaxis{sx}": xa, f"yaxis{sx}": ya})
        self.AX[key] = sx
        return sx

    # text
    def _ink_box(self, x, y, s, size, xanchor, yanchor, bold=False):
        lines = s.split("<br>")
        px_ = FS[size]
        w = max(st.text_w(ln, px_, bold) for ln in lines)
        h = px_ * (1.0 + LINE_GAP * (len(lines) - 1))
        x0 = x if xanchor == "left" else (x - w if xanchor == "right" else x - w / 2)
        y0 = y - h / 2 if yanchor == "middle" else (y - h if yanchor == "bottom" else y)
        return (x0, y0, x0 + w, y0 + h, strip(s))

    def text(self, x, y, s, size="tick", color=INK, xanchor="left", yanchor="middle", kind="label", ref="canvas",
             bold=False):
        assert color in TEXT_COLOURS, (color, s)
        assert size in ("label", "tick", "small"), size
        plain = strip(s)
        assert not re.search(r"[A-Za-z]\.(\s|$)", plain), f"sentence on canvas {plain!r}"
        assert sum(any(c.isalpha() for c in t) for t in plain.split()) <= 6, plain
        if "PLACEHOLDER" in s:
            assert kind == "placeholder" and s == "PLACEHOLDER", s
        self.TXT.append((kind, plain))
        if ref == "canvas":
            xref, yref = "x", "y"
            self.BOXES.append(self._ink_box(x, y, s, size, xanchor, yanchor, bold))
        else:
            xref, yref = ref
        body = f"<b>{s}</b>" if bold else s
        self.fig.add_annotation(x=x, y=y, xref=xref, yref=yref, text=body, showarrow=False, xanchor=xanchor,
                                yanchor=yanchor, align=xanchor, font=dict(family=st.FONT, size=FS[size], color=color),
                                borderpad=0)

    def heading(self, x, y, letter, name):
        assert 1 <= len(name.split()) <= 4, name
        self.TXT.append(("letter", letter))
        self.TXT.append(("heading", name))
        st.panel_letter(self.fig, self.px(x), self.py(y + HEAD), letter, name)
        w = st.text_w(letter, FS["letter"], True) + st.text_w(" " * 3, FS["heading"]) + \
            st.text_w(name, FS["heading"], True)
        self.BOXES.append((x, y + HEAD - FS["letter"], x + w, y + HEAD, f"{letter} {name}"))

    def axis_title(self, s, standoff=3):
        self.TXT.append(("axis", s))
        return dict(text=s, standoff=standoff, font=dict(size=FS["label"], color=INK, family=st.FONT))

    def xtitle_box(self, box, s, tick_lines=1):
        """Ink box of a Plotly x-axis title under box, for the overlap audit. Measured on the 20261006a render, the
        title ink of a standoff-2 title sits 16 to 25 px below the axis line with one line of tick labels and one
        tick-label line (11 px) lower for each further line."""
        x0, _, x1, y1 = box
        lines = s.split("<br>")
        w = max(st.text_w(ln, FS["label"]) for ln in lines)
        top = y1 + 15 + 11 * (tick_lines - 1)
        h = 10 + FS["label"] * LINE_GAP * (len(lines) - 1)
        self.BOXES.append(((x0 + x1) / 2 - w / 2, top, (x0 + x1) / 2 + w / 2, top + h, strip(s)))

    def tick_boxes(self, box, xr, vals, labels, side="bottom", yr=None):
        """Ink boxes of axis tick labels so that the overlap audit sees them."""
        x0, y0, x1, y1 = box
        if side == "bottom":
            for v, lab in zip(vals, labels):
                cx = x0 + (v - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
                self.BOXES.append(self._ink_box(cx, y1 + 5, lab, "tick", "center", "top"))
        else:
            for v, lab in zip(vals, labels):
                cy = y1 - (v - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
                self.BOXES.append(self._ink_box(x0 - 5, cy, lab, "tick", "right", "middle"))

    def placeholder(self, cx, cy, w, h):
        self.fig.add_shape(type="rect", xref="x", yref="y", x0=cx - w / 2, x1=cx + w / 2, y0=cy - h / 2, y1=cy + h / 2,
                           line=dict(color=GREY, width=0.6, dash="2.5px,1.5px"), fillcolor="rgba(0,0,0,0)")
        self.text(cx, cy, "PLACEHOLDER", size="tick", color=GREY, xanchor="center", kind="placeholder")

    # marks
    def scatter(self, sx, x, y, symbol="circle", size=6, color=INK, line=None, mode="markers", err_x=None,
                err_y=None, open_=False):
        assert color in MARK_COLOURS, color
        m = dict(symbol=symbol if not open_ else symbol + "-open", size=size, color=color,
                 line=dict(width=1.0 if open_ else 0, color=color))
        tr = go.Scatter(x=list(x), y=list(y), mode=mode, marker=m, xaxis=f"x{sx}", yaxis=f"y{sx}",
                        hoverinfo="skip", cliponaxis=False, showlegend=False)
        if line:
            tr.line = line
        if err_x:
            lo, hi = err_x
            tr.error_x = dict(type="data", symmetric=False, array=[h - v for h, v in zip(hi, x)],
                              arrayminus=[v - lo_ for lo_, v in zip(lo, x)], thickness=0.9, width=0, color=color)
        if err_y:
            lo, hi = err_y
            tr.error_y = dict(type="data", symmetric=False, array=[h - v for h, v in zip(hi, y)],
                              arrayminus=[v - lo_ for lo_, v in zip(lo, y)], thickness=0.9, width=0, color=color)
        self.fig.add_trace(tr)
        return tr

    def vline(self, sx, x, color=GREY, dash="dash", width=0.6, y0=None, y1=None):
        if y0 is None:
            self.fig.add_shape(type="line", xref=f"x{sx}", yref=f"y{sx} domain", x0=x, x1=x, y0=0, y1=1,
                               line=dict(color=color, width=width, dash=dash), layer="below")
        else:
            self.fig.add_shape(type="line", xref=f"x{sx}", yref=f"y{sx}", x0=x, x1=x, y0=y0, y1=y1,
                               line=dict(color=color, width=width, dash=dash), layer="below")

    def hline(self, sx, y, color=GREY, dash="dash", width=0.6):
        self.fig.add_shape(type="line", xref=f"x{sx} domain", yref=f"y{sx}", x0=0, x1=1, y0=y, y1=y,
                           line=dict(color=color, width=width, dash=dash), layer="below")

    def rule(self, x0, x1, y, width=0.6):
        self.fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=INK, width=width))

    def half_dot(self, x, y, r, colour):
        pts = [(x + (r - 0.5) * math.cos(t), y + (r - 0.5) * math.sin(t))
               for t in [math.pi / 2 + j * math.pi / 24 for j in range(25)]]
        path = "M " + " L ".join(f"{a:.3f},{b:.3f}" for a, b in pts) + " Z"
        self.fig.add_shape(type="path", path=path, xref="x", yref="y", fillcolor=colour, line=dict(width=0))

    def dash_mark(self, x, y):
        self.fig.add_shape(type="line", xref="x", yref="y", x0=x - 3.2, x1=x + 3.2, y0=y, y1=y,
                           line=dict(color=GREY, width=1.0))


def axis_style(**kw):
    """Explicit axis style at the Nature Medicine sizes, so nothing depends on template defaults."""
    d = dict(showline=True, linecolor=INK, linewidth=0.75, ticks="outside", ticklen=2.5, tickwidth=0.75,
             tickcolor=INK, showgrid=False, zeroline=False,
             tickfont=dict(size=FS["tick"], color=INK, family=st.FONT))
    d.update(kw)
    return d


def ticks(vals, nd):
    return dict(tickmode="array", tickvals=list(vals), ticktext=[tickfmt(nd)(v) for v in vals])


def panel_a(C, A):
    C.heading(0, ROW_A, "a", "Cohort inputs and exposure")
    name_x = 1
    cols = {"blood": 104, "other": 136, "retina": 168, "enc": 210, "scored": 248}
    people_r = 334
    colour = {"blood": st.BLOOD, "other": st.BLOOD, "retina": st.EYE, "enc": INK, "scored": INK}
    gy = ROW_A + 26
    for name, a, b in (("BRAN input", "blood", "retina"), ("History", "enc", "scored")):
        x0, x1 = cols[a] - 14, cols[b] + 14
        C.text((x0 + x1) / 2, gy, name, xanchor="center", yanchor="bottom", kind="col_head")
        C.rule(x0, x1, gy + 2)
    heads = {"blood": "Blood<br>count", "other": "Other<br>clinical", "retina": "Retina",
             "enc": "Encoder<br>training", "scored": "Scored<br>before"}
    hy = ROW_A + 49
    for k, s in heads.items():
        C.text(cols[k], hy, s, xanchor="center", yanchor="bottom", kind="col_head")
    C.text(people_r, hy, "People", xanchor="right", yanchor="bottom", kind="col_head")
    R = 3.8
    ys, y = [], ROW_A + 57
    for i, r in enumerate(A["rows"]):
        ys.append(y)
        y += 12 + (4 if i == 1 else 0)
    for r, y in zip(A["rows"], ys):
        C.text(name_x, y, r["cohort"], kind="category")
        for k in CELLS:
            state, c = r["cells"][k], colour[k]
            x = cols[k]
            if state == YES:
                C.scatter(C.AX["canvas"], [x], [y], size=2 * R, color=c)
            elif state == PART:
                C.scatter(C.AX["canvas"], [x], [y], size=2 * R - 1, color=c, open_=True)
                C.half_dot(x, y, R, c)
            elif state == NO:
                C.scatter(C.AX["canvas"], [x], [y], size=2.8, color=GREY)
            else:
                assert state == NA
                C.dash_mark(x, y)
        if r["people"] is None:
            w = st.text_w("PLACEHOLDER", FS["tick"]) + 6
            C.placeholder(people_r - w / 2, y, w, 10.5)
        else:
            C.text(people_r, y, r["people"], xanchor="right", kind="number")
    ky = ys[-1] + 15
    items = [("Yes", YES), ("Partial", PART), ("No", NO), ("Not applicable", NA)]
    for (lab, state), kx in zip(items, (name_x + 3, 44, 92, 126)):
        if state == YES:
            C.scatter(C.AX["canvas"], [kx], [ky], size=2 * R, color=INK)
        elif state == PART:
            C.scatter(C.AX["canvas"], [kx], [ky], size=2 * R - 1, color=INK, open_=True)
            C.half_dot(kx, ky, R, INK)
        elif state == NO:
            C.scatter(C.AX["canvas"], [kx], [ky], size=2.8, color=GREY)
        else:
            C.dash_mark(kx, ky)
        C.text(kx + 7, ky, lab, kind="key")
    return ky + 5


B_YLIM, B_YTICKS = (0.58, 0.72), [0.60, 0.64, 0.68, 0.72]
C_YLIM, C_YTICKS = (0.60, 0.72), [0.60, 0.64, 0.68, 0.72]


def inside(lim, *vals):
    for v in vals:
        assert lim[0] < v < lim[1], (lim, v)


def panel_b(C, Bd):
    x0p = 350
    C.heading(x0p, ROW_A, "b", "Field removal")
    box = (388, ROW_A + 22, PW - 3, ROW_A + 108)
    xr = (-9, 84)
    pct = Bd["pct"]
    sx = C.axes("b", box, xr, B_YLIM,
                x=axis_style(**ticks(pct, 0), title=C.axis_title("Clinical fields withheld (%)", 2)),
                y=axis_style(**ticks(B_YTICKS, 2), title=C.axis_title("Macro AUROC", 3)))
    C.tick_boxes(box, xr, pct, [tickfmt(0)(p) for p in pct])
    C.tick_boxes(box, xr, B_YTICKS, [tickfmt(2)(v) for v in B_YTICKS], side="left", yr=B_YLIM)
    C.xtitle_box(box, "Clinical fields withheld (%)")
    V = Bd["values"]
    for arm in ARMS:
        inside(B_YLIM, *[t[0] for t in V[arm]])
    inside(B_YLIM, *[t[1] for t in V["BRAN"]], *[t[2] for t in V["BRAN"]])
    dodge = {"BRAN": -2.6, "GBM": -1.3, "MICE_LR": 0.0, "MEDIAN_LR": 1.3, "LATE_FUSION": 2.6}
    for arm in COMPS:
        C.scatter(sx, [p + dodge[arm] for p in pct], [t[0] for t in V[arm]], symbol=SYMBOL[arm], size=msize(arm),
                  color=GREY, mode="lines+markers", line=dict(color=GREY, width=0.6))
    C.scatter(sx, [p + dodge["BRAN"] for p in pct], [t[0] for t in V["BRAN"]], size=msize("BRAN"), color=st.SHARED,
              mode="lines+markers", line=dict(color=st.SHARED, width=1.1),
              err_y=([t[1] for t in V["BRAN"]], [t[2] for t in V["BRAN"]]))
    # key in the empty band under every drawn value, two rows of three
    lowest = min(t[0] for arm in ARMS for t in V[arm])
    rows = [0.6195, 0.6005]
    unit = (B_YLIM[1] - B_YLIM[0]) / (box[3] - box[1])             # data units per px
    assert rows[0] + (FS["tick"] / 2 + 3.5) * unit < lowest, (rows, lowest)
    assert rows[1] - (FS["tick"] / 2 + 1.0) * unit > B_YLIM[0]
    layout = [("BRAN", -6, rows[0]), ("GBM", 26, rows[0]), ("MICE_LR", 58, rows[0]),
              ("MEDIAN_LR", -6, rows[1]), ("LATE_FUSION", 26, rows[1])]
    xu = (xr[1] - xr[0]) / (box[2] - box[0])
    for arm, x, y in layout:
        C.scatter(sx, [x], [y], symbol=SYMBOL[arm], size=msize(arm), color=mcolour(arm))
        C.text(x + 4 * xu, y, NAME[arm], kind="key", ref=(f"x{sx}", f"y{sx}"))
        cx = box[0] + (x + 4 * xu - xr[0]) / xu
        cy = box[3] - (y - B_YLIM[0]) / unit
        C.BOXES.append(C._ink_box(cx, cy, NAME[arm], "tick", "left", "middle"))
    # nothing drawn above a key entry may sit inside the key band
    for arm in ARMS:
        for p, t in zip(pct, V[arm]):
            if p + dodge[arm] >= -6:
                assert t[0] - (msize(arm) / 2) * unit > rows[0] + (FS["tick"] / 2) * unit, (arm, p)


def panel_c(C, Cd):
    C.heading(0, ROW_B, "c", "Block removal")
    prof = Cd["profiles"]
    names = [lab for _, lab in prof]
    box = (36, ROW_B + 22, 232, ROW_B + 96)
    xr = (-0.5, len(prof) - 0.5)
    sx = C.axes("c", box, xr, C_YLIM,
                x=axis_style(tickmode="array", tickvals=list(range(len(prof))), ticktext=names, ticks="",
                             tickangle=0, automargin=False),
                y=axis_style(**ticks(C_YTICKS, 2), title=C.axis_title("Macro AUROC", 3)))
    for n in names:
        C.TXT.append(("category", strip(n)))
    C.tick_boxes(box, xr, list(range(len(prof))), names)
    C.tick_boxes(box, xr, C_YTICKS, [tickfmt(2)(v) for v in C_YTICKS], side="left", yr=C_YLIM)
    C.hline(sx, Cd["reference"], color=GREY, dash="dash", width=0.6)
    V = Cd["values"]
    for arm in ARMS:
        inside(C_YLIM, *[t[0] for t in V[arm]])
    inside(C_YLIM, *[t[1] for t in V["BRAN"]], *[t[2] for t in V["BRAN"]])
    dodge = {"BRAN": -0.26, "GBM": -0.13, "MICE_LR": 0.0, "MEDIAN_LR": 0.13, "LATE_FUSION": 0.26}
    for arm in COMPS:
        C.scatter(sx, [i + dodge[arm] for i in range(len(prof))], [t[0] for t in V[arm]], symbol=SYMBOL[arm],
                  size=msize(arm), color=GREY)
    C.scatter(sx, [i + dodge["BRAN"] for i in range(len(prof))], [t[0] for t in V["BRAN"]], size=msize("BRAN"),
              color=st.SHARED, err_y=([t[1] for t in V["BRAN"]], [t[2] for t in V["BRAN"]]))


D_ABS, D_CON = (0.64, 0.69), (-0.02, 0.04)


def panel_d(C, Dd):
    x0 = 246
    C.heading(x0, ROW_B, "d", "Missing-data baselines")
    top, rows0, pitch = ROW_B + 49, ROW_B + 58, 13
    rows_y = [rows0 + i * pitch for i in range(len(ARMS))]
    n = len(ARMS)
    yr = (n - 0.5, -0.5)

    def ybox(xa, xb):
        return (xa, rows_y[0] - pitch / 2, xb, rows_y[-1] + pitch / 2 + 1)

    lab = C.axes("d_lab", ybox(x0, x0 + 9), (0, 1), yr, x=HIDDEN, y=HIDDEN)
    for i, arm in enumerate(ARMS):
        C.scatter(lab, [0.5], [i], symbol=SYMBOL[arm], size=msize(arm), color=mcolour(arm))
        C.text(x0 + 11, rows_y[i], NAME[arm], kind="category")
    ax0, ax1 = 330, 384
    box = ybox(ax0, ax1)
    sa = C.axes("d_abs", box, D_ABS, yr, x=axis_style(**ticks([0.64, 0.66, 0.68], 2)), y=HIDDEN)
    C.tick_boxes(box, D_ABS, [0.64, 0.66, 0.68], ["0.64", "0.66", "0.68"])
    for i, arm in enumerate(ARMS):
        e, lo, hi = Dd["abs"][arm]
        inside(D_ABS, lo, hi)
        C.scatter(sa, [e], [i], symbol=SYMBOL[arm], size=msize(arm), color=mcolour(arm), err_x=([lo], [hi]))
    C.text((ax0 + ax1) / 2, top - 1, "Five-pattern<br>mean", xanchor="center", yanchor="bottom", kind="col_head")
    cw, gap, ggap, c0 = 38, 5, 10, 398
    cxs, x = [], c0
    for i in range(len(Dd["cols"])):
        cxs.append(x)
        x += cw + (ggap if i in (2, 4) else gap)
    assert cxs[-1] + cw <= PW - 1, cxs[-1] + cw
    for (k, name), cx in zip(Dd["cols"], cxs):
        box = ybox(cx, cx + cw)
        sc = C.axes(f"d_{k}", box, D_CON, yr, x=axis_style(**ticks([0.0, 0.03], 2)), y=HIDDEN)
        C.tick_boxes(box, D_CON, [0.0, 0.03], ["0", "0.03"])
        C.vline(sc, 0.0)
        for i, comp in enumerate(COMPS, start=1):
            e, lo, hi = Dd["con"][comp][[c for c, _ in Dd["cols"]].index(k)]
            inside(D_CON, lo, hi)
            C.scatter(sc, [e], [i], size=5.2, color=st.SHARED, err_x=([lo], [hi]), open_=[k, comp] in Dd["hollow"])
        C.text(cx + cw / 2, top - 1, name, xanchor="center", yanchor="bottom", kind="col_head")
    gx0, gx1 = cxs[0], cxs[2] + cw
    C.text((gx0 + gx1) / 2, ROW_B + 27, "Fields withheld", xanchor="center", yanchor="bottom", kind="col_head")
    C.rule(gx0, gx1, ROW_B + 29)
    yb = rows_y[-1] + pitch / 2 + 1 + 15
    C.text((ax0 + ax1) / 2, yb, "AUROC", size="label", xanchor="center", yanchor="top", kind="axis")
    C.text((cxs[0] + cxs[-1] + cw) / 2, yb, "BRAN minus comparator (AUROC)", size="label", xanchor="center",
           yanchor="top", kind="axis")


E_YLIM, E_YTICKS = (0.68, 0.76), [0.68, 0.70, 0.72, 0.74, 0.76]
E_XR = (-0.55, 5.9)
E_CON = (-0.02, 0.03)
E_HB = (-0.025, 0.075)
E_LAB_X, E_AX0, E_AX1, E_EST_R, E_DEC_X = 300, 408, 504, 610, 618
E_ZERO, E_MARGIN = "3px,2px", "1px,1.6px"


def e_shape(arm):
    a = E_SHAPE[arm]
    return SYMBOL[a], msize(a), mcolour(a)


def panel_e(C, E):
    C.heading(0, ROW_C, "e", "KNHANES 2024 temporal test")
    cols = E["cols"]
    x0, x1, top, bot = 36, 284, ROW_C + 32, ROW_C + 100
    box = (x0, top, x1, bot)
    sx = C.axes("e_pat", box, E_XR, E_YLIM,
                x=axis_style(tickmode="array", tickvals=[c for _, _, c in cols], ticktext=[n for _, n, _ in cols],
                             ticks="", tickangle=0, automargin=False, title=C.axis_title("Missing-data pattern", 2)),
                y=axis_style(**ticks(E_YTICKS, 2), title=C.axis_title("Macro AUROC", 3)))
    for _, n, _ in cols:
        C.TXT.append(("category", strip(n)))
    C.tick_boxes(box, E_XR, [c for _, _, c in cols], [n for _, n, _ in cols])
    C.xtitle_box(box, "Missing-data pattern", tick_lines=max(n.count("<br>") + 1 for _, n, _ in cols))
    C.tick_boxes(box, E_XR, E_YTICKS, [tickfmt(2)(v) for v in E_YTICKS], side="left", yr=E_YLIM)
    X = lambda v: x0 + (v - E_XR[0]) / (E_XR[1] - E_XR[0]) * (x1 - x0)  # noqa: E731
    for name, a, b in (("Primary patterns", 0.0, 3.0), ("Other patterns", 4.35, 5.35)):
        ga, gb = X(a) - 15, X(b) + 15
        C.text((ga + gb) / 2, top - 5, name, xanchor="center", yanchor="bottom", kind="col_head")
        C.rule(ga, gb, top - 3)
    dodge = {"bran_state_logistic": -0.22, "raw_gbm_masktrained": 0.0, "raw_logistic_masktrained": 0.22}
    V = E["values"]
    lo_all = min(min(t[0], t[1]) for arm in E["arms"] for t in V[arm])
    hi_all = max(max(t[0], t[2]) for arm in E["arms"] for t in V[arm])
    inside(E_YLIM, lo_all, hi_all)
    for arm in E["arms"][1:]:
        sym, size, col = e_shape(arm)
        C.scatter(sx, [c + dodge[arm] for _, _, c in cols], [t[0] for t in V[arm]], symbol=sym, size=size, color=col)
    arm = "bran_state_logistic"
    C.scatter(sx, [c + dodge[arm] for _, _, c in cols], [t[0] for t in V[arm]], size=msize("BRAN"), color=st.SHARED,
              err_y=([t[1] for t in V[arm]], [t[2] for t in V[arm]]))
    ky = 0.6885
    unit = (E_YLIM[1] - E_YLIM[0]) / (bot - top)
    assert ky + (FS["tick"] / 2 + 2) * unit < lo_all and ky - (FS["tick"] / 2 + 1) * unit > E_YLIM[0], (ky, lo_all)
    xu = (E_XR[1] - E_XR[0]) / (x1 - x0)
    for arm, kx in zip(E["arms"], (-0.42, 0.55, 2.8)):
        sym, size, col = e_shape(arm)
        C.scatter(sx, [kx], [ky], symbol=sym, size=size, color=col)
        C.text(kx + 4 * xu, ky, E_NAME[arm], kind="key", ref=(f"x{sx}", f"y{sx}"))
        C.BOXES.append(C._ink_box(x0 + (kx + 4 * xu - E_XR[0]) / xu, bot - (ky - E_YLIM[0]) / unit, E_NAME[arm],
                                  "tick", "left", "middle"))
    # right, BRAN minus comparator on the three-pattern mean, then the key secondaries
    con_x = lambda v, r=E_CON: E_AX0 + (v - r[0]) / (r[1] - r[0]) * (E_AX1 - E_AX0)  # noqa: E731
    C.text(E_LAB_X, ROW_C + 20, "Primary contrast, three-pattern mean", size="label", yanchor="bottom",
           kind="subtitle")
    hy = ROW_C + 29
    C.text(con_x(0) - 3, hy, "Favours comparator", xanchor="right", kind="col_head")
    C.text(con_x(0) + 3, hy, "Favours BRAN", kind="col_head")
    C.text(E_EST_R, hy, "Estimate (95% CI)", xanchor="right", kind="col_head")
    C.text(E_DEC_X, hy, "Decision", kind="col_head")
    ya = [ROW_C + 41, ROW_C + 53]
    yle = ROW_C + E_ROW_LE
    ytop, ybot = ya[0] - 7, yle + 8
    box = (E_AX0, ytop, E_AX1, ybot)
    tv = [-0.01, 0.0, 0.01, 0.02]
    tt = [f"{MINUS}0.01<br>margin", "0", "0.01", "0.02"]
    sa = C.axes("e_con", box, E_CON, (ybot, ytop), x=axis_style(tickmode="array", tickvals=tv, ticktext=tt), y=HIDDEN)
    C.TXT.append(("tick", "margin"))
    C.tick_boxes(box, E_CON, tv, tt)
    # axis title with its unit, drawn as canvas text under the two-line tick labels (ink ends 25 px below the axis)
    C.text((E_AX0 + E_AX1) / 2, ybot + 27, "BRAN minus comparator (AUROC)", size="label", xanchor="center",
           yanchor="top", kind="axis")
    C.vline(sa, 0.0, dash=E_ZERO, y0=ybot, y1=ytop)
    C.vline(sa, E["margin"], dash=E_MARGIN, width=0.9, y0=ya[1] + 6, y1=ytop)
    for (comp, t, verdict), y in zip(E["primary"], ya):
        sym, size, col = e_shape(comp)
        C.scatter(C.AX["canvas"], [E_LAB_X + 3], [y], symbol=sym, size=size, color=col)
        C.text(E_LAB_X + 10, y, E_NAME[comp], kind="category")
        inside(E_CON, t[1], t[2])
        C.scatter(sa, [t[0]], [y], size=5.2, color=st.SHARED, err_x=([t[1]], [t[2]]))
        C.text(E_EST_R, y, sgc(t), xanchor="right", kind="number")
        C.text(E_DEC_X, y, verdict, kind="verdict")
    C.text(E_LAB_X, ROW_C + 73, "Key secondaries", size="label", yanchor="bottom", kind="subtitle")
    t = E["le"]
    sym, size, col = e_shape("raw_logistic_masktrained")
    C.scatter(C.AX["canvas"], [E_LAB_X + 3], [yle], symbol=sym, size=size, color=col)
    C.text(E_LAB_X + 10, yle, f"Label efficiency,<br>{E['le_size']} labels", kind="category")
    inside(E_CON, t[1], t[2])
    assert t[1] < 0
    C.scatter(sa, [t[0]], [yle], size=5.2, color=st.SHARED, err_x=([t[1]], [t[2]]))
    C.text(E_EST_R, yle, sgc(t), xanchor="right", kind="number")
    C.text(E_DEC_X, yle, "Not met", kind="verdict")
    # haemoglobin error, lower is better, so the direction header is reversed and printed again
    yh = ROW_C + E_ROW_HB
    hb_x = lambda v: con_x(v, E_HB)  # noqa: E731
    C.text(hb_x(0) - 3, yh - 11, "Favours BRAN", xanchor="right", kind="col_head")
    C.text(hb_x(0) + 3, yh - 11, "Favours comparator", kind="col_head")
    t = E["hb"]
    box = (E_AX0, yh - 5, E_AX1, yh + 5)
    tv, tt = [0.0, 0.05], ["0", "0.05 margin"]
    sh = C.axes("e_hb", box, E_HB, (yh + 5, yh - 5), x=axis_style(tickmode="array", tickvals=tv, ticktext=tt),
                y=HIDDEN)
    C.TXT.append(("tick", "margin"))
    C.tick_boxes(box, E_HB, tv, tt)
    C.text((E_AX0 + E_AX1) / 2, yh + 5 + 15, "BRAN minus comparator (g/dL)", size="label", xanchor="center",
           yanchor="top", kind="axis")
    C.vline(sh, 0.0, dash=E_ZERO)
    C.vline(sh, E["hb_margin"], dash=E_MARGIN, width=0.9)
    sym, size, col = e_shape("raw_gbm_masktrained")
    C.scatter(C.AX["canvas"], [E_LAB_X + 3], [yh], symbol=sym, size=size, color=col)
    C.text(E_LAB_X + 10, yh, "Haemoglobin error (g/dL)", kind="category")
    inside(E_HB, t[1], t[2])
    assert t[2] > E["hb_margin"]
    C.scatter(sh, [t[0]], [yh], size=5.2, color=st.SHARED, err_x=([t[1]], [t[2]]))
    C.text(E_EST_R, yh, sgc(t), xanchor="right", kind="number")
    C.text(E_DEC_X, yh, "Not met", kind="verdict")


F_CON_TITLE = "BRAN + context minus<br>comparator (AUROC)"


def panel_f(C, Fd):
    C.heading(0, ROW_D, "f", "Surgical cohort")
    rows_y = [ROW_D + 27 + i * 13 for i in range(3)]
    yr = (2.5, -0.5)

    def box(a, b):
        return (a, rows_y[0] - 7, b, rows_y[-1] + 8)

    bx = box(108, 240)
    sa = C.axes("f_abs", bx, (0.80, 1.00), yr,
                x=axis_style(**ticks([0.80, 0.85, 0.90, 0.95, 1.00], 2), title=C.axis_title("AUROC", 2)), y=HIDDEN)
    C.tick_boxes(bx, (0.80, 1.00), [0.80, 0.85, 0.90, 0.95, 1.00], ["0.80", "0.85", "0.90", "0.95", "1.00"])
    C.xtitle_box(bx, "AUROC")
    for i, (lab, key, t) in enumerate(Fd["arms"]):
        inside((0.80, 1.00), t[1], t[2])
        C.text(102, rows_y[i], lab, xanchor="right", kind="category")
        C.scatter(sa, [t[0]], [i], size=msize("BRAN"), color=COLOUR[key], err_x=([t[1]], [t[2]]))
    bx = box(266, 396)
    sc = C.axes("f_con", bx, (-0.02, 0.14), yr,
                x=axis_style(**ticks([0.0, 0.05, 0.10], 2),
                             title=C.axis_title(F_CON_TITLE, 2)),
                y=HIDDEN)
    C.tick_boxes(bx, (-0.02, 0.14), [0.0, 0.05, 0.10], ["0", "0.05", "0.10"])
    C.xtitle_box(bx, F_CON_TITLE)
    C.vline(sc, 0.0)
    row = {"vs context": 0, "vs blood tests + context": 1}
    for lab, t in Fd["con"]:
        inside((-0.02, 0.14), t[1], t[2])
        C.scatter(sc, [t[0]], [row[lab]], size=msize("BRAN"), color=st.SHARED, err_x=([t[1]], [t[2]]))
    assert Fd["con"][1][1][1] < 0


def panel_g(C, Gd):
    x0 = 410
    C.heading(x0, ROW_D, "g", "ICU haemoglobin")
    pitch = 11
    y = ROW_D + 24
    rows, glabels = [], []
    for g, items in Gd["groups"]:
        glabels.append((g, y))
        y += pitch
        for it in items:
            rows.append((it, y))
            y += pitch
        y += 2
    top, bot = rows[0][1] - pitch / 2, rows[-1][1] + pitch / 2
    bx = (534, top, PW - 4, bot)
    sx = C.axes("g", bx, (0.0, 4.0), (bot, top),
                x=axis_style(**ticks([0, 1, 2, 3, 4], 0), title=C.axis_title("Mean absolute error (g/dL)", 2)),
                y=HIDDEN)
    C.tick_boxes(bx, (0.0, 4.0), [0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"])
    C.xtitle_box(bx, "Mean absolute error (g/dL)")
    for g, gy in glabels:
        C.text(x0 + 1, gy, g, kind="category")
    for (lab, key, t), ry in rows:
        inside((0.0, 4.0), t[1], t[2])
        C.text(528, ry, lab, xanchor="right", kind="category")
        C.scatter(sx, [t[0]], [ry], size=msize("BRAN"), color=COLOUR[key], err_x=([t[1]], [t[2]]))
        if lab == "BRAN" and ry == rows[1][1]:
            s = f"Bias +{Gd['bias']:.2f} g/dL"
            C.text(t[1] - 0.09, ry, s, xanchor="right", kind="effect", ref=(f"x{sx}", f"y{sx}"))
            C.BOXES.append(C._ink_box(bx[0] + (t[1] - 0.09) / 4.0 * (bx[2] - bx[0]), ry, s, "tick", "right", "middle"))
    assert all(r[0][2][2] < rows[1][0][2][1] for r in rows[2:])


def audit(C):
    for a in C.fig.layout.annotations:
        plain = strip(a.text)
        for ch in (";", ":", "—"):
            assert ch not in plain, (ch, plain)
    for tr in C.fig.data:
        assert tr.hoverinfo == "skip" and tr.mode is not None
        assert tr.marker.color in MARK_COLOURS, tr.marker.color
    letters = [t for k, t in C.TXT if k == "letter"]
    assert letters == list("abcdefg"), letters
    hits = []
    for i, a in enumerate(C.BOXES):
        if a[0] < -ML - 0.5 or a[2] > PW + MR + 0.5 or a[1] < -MT - 0.5 or a[3] > PH + MB + 0.5:
            hits.append(("outside", a))
        for b in C.BOXES[i + 1:]:
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox > 0.4 and oy > 0.4:
                hits.append(("overlap", a[4], b[4], round(ox, 1), round(oy, 1)))
    assert not hits, hits
    problems = st.nm_figure_problems(C.fig, W, H)
    assert not problems, problems


def draw(S):
    """Draw the figure from the Source Data payload alone."""
    assert S["figure"] == FIG_ID and S["display"] == DISPLAY, (S["figure"], S["display"])
    dr = S["draw"]
    C = Canvas()
    a_bottom = panel_a(C, dr["a"])
    assert a_bottom < ROW_B - 4, a_bottom
    panel_b(C, dr["b"])
    panel_c(C, dr["c"])
    panel_d(C, dr["d"])
    panel_e(C, dr["e"])
    panel_f(C, dr["f"])
    panel_g(C, dr["g"])
    audit(C)
    return C


def render(payload):
    C = draw(payload)
    return [(FIG_ID, C.fig, W, H)]

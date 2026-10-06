"""Drawing code for Fig. 1.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_fig1.py
(SHA-256 f55063418c8fb3fd1c032f893183d121fa15437fdbdb4cb54918458b940c1daa).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import math
import re
import bran_plotly_style as st
st.use_nature_medicine()
import plotly.graph_objects as go
FIG_ID = "fig1"
SZ = st.SIZES
PX = st.MM
PT = st.PT
WHITE = "#FFFFFF"
ML = MR = MT = MB = 4
W = st.FULL_W
IW = (W - ML - MR) / PX
H = st.FULL_H
IH = (H - MT - MB) / PX
A_X1 = 55.0
R_X0 = 58.5
C_TOP = 78.0
D_TOP = 114.0
PANELS = {                    # x0, y0, x1, y1 in mm, one invisible axis pair per panel
    "a": (0.0, 0.0, A_X1, D_TOP),
    "b": (R_X0, 0.0, IW, C_TOP),
    "c": (R_X0, C_TOP, IW, D_TOP),
    "d": (0.0, D_TOP, IW, IH),
}
AXIS = {"a": "", "b": "2", "c": "3", "d": "4"}
HEAD_DY = 4.2


def need(cond, msg):
    if not cond:
        raise SystemExit(f"CHECK FAILED  {msg}")


def mix(hex7: str, w: float) -> str:
    """Blend a colour toward white by fraction w (flat tint, no transparency)."""
    r, g, b = (int(hex7[i:i + 2], 16) for i in (1, 3, 5))
    f = lambda c: round(c + (255 - c) * w)  # noqa: E731
    return "#%02X%02X%02X" % (f(r), f(g), f(b))


def text_box(s, size, bold=False):
    """Text box in mm from Arial metrics, line height 1.3 em as Plotly draws it."""
    lines = s.split("\n")
    w = max(st.text_w(ln, SZ[size], bold) for ln in lines) / PX
    h = len(lines) * 1.3 * SZ[size] / PX
    return w, h


class Canvas:
    def __init__(self):
        self.fig = go.Figure()
        self.ref = "a"
        self.texts = []           # (plain, size_key, colour, kind, bbox mm)
        self.frames = []          # boxes whose edges text must not cross
        self.icons = []
        lay = dict(width=W, height=H, margin=dict(l=ML, r=MR, t=MT, b=MB, pad=0), showlegend=False,
                   hovermode=False, dragmode=False, paper_bgcolor="white", plot_bgcolor="white")
        for p, (x0, y0, x1, y1) in PANELS.items():
            n = AXIS[p]
            need(0 <= x0 < x1 <= IW + 1e-6 and 0 <= y0 < y1 <= IH + 1e-6, f"panel {p} inside canvas")
            lay[f"xaxis{n}"] = dict(domain=[x0 / IW, x1 / IW], range=[x0, x1], visible=False, fixedrange=True,
                                    anchor=f"y{n}", showgrid=False, zeroline=False)
            lay[f"yaxis{n}"] = dict(domain=[1 - y1 / IH, 1 - y0 / IH], range=[IH - y1, IH - y0], visible=False,
                                    fixedrange=True, anchor=f"x{n}", showgrid=False, zeroline=False)
        self.fig.update_layout(**lay)

    @property
    def xr(self):
        return "x" + AXIS[self.ref]

    @property
    def yr(self):
        return "y" + AXIS[self.ref]

    def in_panel(self, x, y, what):
        x0, y0, x1, y1 = PANELS[self.ref]
        need(x0 - 0.01 <= x <= x1 + 0.01 and y0 - 0.01 <= y <= y1 + 0.01, f"{what} at {x:.1f},{y:.1f} in panel {self.ref}")

    def rect(self, x0, y0, x1, y1, fill=None, edge=None, width_pt=0.75, dash=None, frame=False):
        if frame:
            self.frames.append((x0, y0, x1, y1))
        self.in_panel(x0, y0, "rect")
        self.in_panel(x1, y1, "rect")
        self.fig.add_shape(type="rect", xref=self.xr, yref=self.yr, x0=x0, x1=x1, y0=IH - y1, y1=IH - y0,
                           fillcolor=fill or "rgba(0,0,0,0)",
                           line=dict(color=edge or "rgba(0,0,0,0)", width=width_pt * PT if edge else 0,
                                     dash=dash or "solid"), layer="above")

    def line(self, x0, y0, x1, y1, colour, width_pt=0.75, dash=None):
        self.in_panel(x0, y0, "line")
        self.in_panel(x1, y1, "line")
        self.fig.add_shape(type="line", xref=self.xr, yref=self.yr, x0=x0, x1=x1, y0=IH - y0, y1=IH - y1,
                           line=dict(color=colour, width=width_pt * PT, dash=dash or "solid"), layer="above")

    def poly(self, pts, fill):
        path = "M " + " L ".join(f"{x:.3f},{IH - y:.3f}" for x, y in pts) + " Z"
        self.fig.add_shape(type="path", xref=self.xr, yref=self.yr, path=path, fillcolor=fill,
                           line=dict(width=0), layer="above")

    def dot(self, cx, cy, d, fill):
        self.in_panel(cx, cy, "dot")
        r = d / 2
        self.fig.add_shape(type="circle", xref=self.xr, yref=self.yr, x0=cx - r, x1=cx + r, y0=IH - cy - r,
                           y1=IH - cy + r, fillcolor=fill, line=dict(width=0), layer="above")

    def half_disc(self, cx, cy, d, side, fill):
        r = d / 2
        a0, a1 = (90, 270) if side == "left" else (-90, 90)
        pts = [(cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a)))
               for a in [a0 + (a1 - a0) * i / 72 for i in range(73)]]
        self.poly(pts, fill)

    def arrow(self, x0, y0, x1, y1, colour, width_pt=0.75, head=1.1, dash=None):
        """Line with a filled triangular head that ends exactly at (x1, y1)."""
        L = math.hypot(x1 - x0, y1 - y0)
        need(L > head * 1.2, f"arrow too short at {x0:.1f},{y0:.1f}")
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        bx, by = x1 - ux * head, y1 - uy * head
        self.line(x0, y0, bx + ux * 0.15, by + uy * 0.15, colour, width_pt, dash)
        hw = head * 0.42
        self.poly([(x1, y1), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)], colour)

    def text(self, x, y, s, size="label", colour=st.INK, xa="left", ya="middle", bold=False, kind="label"):
        need(size in SZ, f"font size key {size}")
        self.in_panel(x, y, f"text anchor {s!r}")
        html = "<br>".join(f"<b>{ln}</b>" if bold else ln for ln in s.split("\n"))
        self.fig.add_annotation(x=x, y=IH - y, xref=self.xr, yref=self.yr, text=html, showarrow=False,
                                xanchor=xa, yanchor=ya, align={"left": "left", "center": "center", "right": "right"}[xa],
                                font=dict(size=SZ[size], color=colour, family=st.FONT), borderpad=0,
                                captureevents=False)
        w, h = text_box(s, size, bold)
        x0 = {"left": x, "center": x - w / 2, "right": x - w}[xa]
        y0 = {"top": y, "middle": y - h / 2, "bottom": y - h}[ya]
        self.texts.append((s, size, colour, kind, (x0, y0, x0 + w, y0 + h)))
        return w

    def icon(self, ic, cx, cy, bw, bh, rel):
        self.icons.append(rel)
        iw, ih = ic["px"]
        s = min(bw / iw, bh / ih)
        w, h = iw * s, ih * s
        self.in_panel(cx - w / 2, cy - h / 2, "icon")
        self.in_panel(cx + w / 2, cy + h / 2, "icon")
        self.fig.add_layout_image(source="data:image/png;base64," + ic["png_base64"], xref=self.xr, yref=self.yr,
                                  x=cx, y=IH - cy, sizex=w, sizey=h, xanchor="center", yanchor="middle",
                                  sizing="contain", layer="above")
        return w, h

    def squares(self, pts, side, colour):
        n = AXIS[self.ref]
        self.fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[IH - p[1] for p in pts], mode="markers",
                                      xaxis=f"x{n}", yaxis=f"y{n}", hoverinfo="skip", cliponaxis=False,
                                      marker=dict(symbol="square", size=side * PX, color=colour,
                                                  line=dict(width=0))))

    def heading(self, x, top, letter, heading):
        y = top + HEAD_DY
        st.panel_letter(self.fig, x / IW, 1 - y / IH, letter, heading)
        lw, lh = text_box(letter, "letter", True)
        hw, _ = text_box(heading, "heading", True)
        gap = st.text_w("&nbsp;&nbsp;&nbsp;".replace("&nbsp;", " "), SZ["heading"], True) / PX
        self.texts.append((letter, "letter", st.INK, "letter", (x, y - lh, x + lw, y)))
        self.texts.append((heading, "heading", st.INK, "heading", (x + lw + gap, y - lh, x + lw + gap + hw, y)))
        return y


class Ctx:
    """Payload accessors for draw()."""

    def __init__(self, p):
        self.d = p["draw"]
        c = self.d["colours"]
        self.EYE, self.SHARED, self.BLOOD = c["eye"], c["shared"], c["blood"]
        self.INK, self.SLATE, self.GREY, self.LIGHT, self.MUTED = c["ink"], c["slate"], c["grey"], c["light"], c["muted"]
        self.block_colour = {"eye": self.EYE, "shared": self.SHARED, "blood": self.BLOOD}
        b = self.d["block"]
        self.block_label = {"eye": f"{b} retinal-associated", "shared": f"{b} shared", "blood": f"{b} clinical-associated"}

    def icon(self, rel):
        return self.d["icons"][rel]


def empty_tile(c, K, x, y, s):
    """Missing value. White tile, faint edge and one slash."""
    c.rect(x, y, x + s, y + s, fill=WHITE, edge=K.MUTED, width_pt=0.5)
    c.line(x + 0.3, y + s - 0.3, x + s - 0.3, y + 0.3, colour=K.MUTED, width_pt=0.5)


def block_of(k):
    return "eye" if k < 64 else ("shared" if k < 128 else "blood")


def mosaic(c, K, ox, oy, width, pale=None, gap_frac=0.13):
    """The 192 tile silhouette of build_fig1_v5.py, 64 tiles per block."""
    k = width / 20.0
    side = K.d["pitch"] * k * (1 - gap_frac)
    groups, spans = {}, {}
    for idx, (cx, cy) in enumerate(K.d["cells"]):
        b = block_of(idx)
        col = K.block_colour[b]
        col = col if b != pale else mix(col, 0.80)
        x, y = ox + cx * k, oy + cy * k
        groups.setdefault(col, []).append((x, y))
        lo, hi = spans.get(b, (y, y))
        spans[b] = (min(lo, y - side / 2), max(hi, y + side / 2))
    need(sum(len(p) for p in groups.values()) == 192, "mosaic has 192 tiles")
    for col, pts in groups.items():
        c.squares(pts, side, col)
    return spans


def box(c, K, x0, x1, ym, h, lines, edge, tag=None, dash=None):
    """Plain module box, square corners, white fill, thin coloured edge, optional grey layer tag."""
    c.rect(x0, ym - h / 2, x1, ym + h / 2, fill=WHITE, edge=edge, width_pt=0.75, frame=True, dash=dash)
    rows = [(s, K.INK, "label") for s in lines] + ([(tag, K.SLATE, "tag")] if tag else [])
    lh = 1.3 * SZ["label"] / PX
    for i, (s, col, kind) in enumerate(rows):
        c.text((x0 + x1) / 2, ym + (i - (len(rows) - 1) / 2) * lh, s, colour=col, xa="center", kind=kind)


def panel_a(c, K):
    c.ref = "a"
    x0, x1 = 0.3, A_X1 - 0.3
    c.heading(0.0, 0.0, "a", "Training cohorts")
    e4 = K.d["e4"]
    nmax = max(r["people"] for r in e4.values())
    DMAX = 20.0

    def diam(n):
        return DMAX * math.sqrt(n / nmax)

    lanes = [(6.6, 23.6), (25.2, 38.4), (40.0, D_TOP - 0.4)]
    for (ya, yb) in lanes:
        c.rect(x0, ya, x1, yb, edge=K.LIGHT, width_pt=0.75, frame=True)
    ret, tube = K.icon("symbols/retina.png"), K.icon("symbols/tube.png")
    # retinal only
    ya, yb = lanes[0]
    c.text(2.2, ya + 3.0, "Retinal only", bold=True)
    c.icon(ret, x1 - 3.6, ya + 3.4, 4.6, 4.6, "symbols/retina.png")
    cy = ya + 10.6
    c.dot(9.0, cy, diam(e4["brset"]["people"]), K.EYE)
    c.text(17.0, cy - 3.2, "BRSET")
    c.text(17.0, cy, f"≥{e4['brset']['people']:,} people", colour=K.SLATE, kind="number")
    c.text(17.0, cy + 3.2, f"≥{e4['brset']['images']:,} photographs", colour=K.SLATE, kind="number")
    # paired
    ya, yb = lanes[1]
    c.text(2.2, ya + 3.0, "Paired", bold=True)
    c.icon(ret, x1 - 9.0, ya + 3.4, 4.6, 4.6, "symbols/retina.png")
    c.icon(tube, x1 - 3.6, ya + 3.4, 4.6, 4.6, "symbols/tube.png")
    cy = ya + 8.6
    d = diam(e4["aireadi"]["people"])
    c.half_disc(9.0, cy, d, "left", K.EYE)
    c.half_disc(9.0, cy, d, "right", K.BLOOD)
    c.text(17.0, cy - 1.6, "AI-READI")
    c.text(17.0, cy + 1.6, f"≥{e4['aireadi']['people']:,} people", colour=K.SLATE, kind="number")
    # clinical only
    ya, yb = lanes[2]
    c.text(2.2, ya + 3.0, "Clinical only", bold=True)
    c.icon(tube, x1 - 3.6, ya + 3.4, 4.6, 4.6, "symbols/tube.png")
    rows = [[("eicu", "eICU"), ("mimiciv", "MIMIC-IV")],
            [("nwicu", "NWICU"), ("nhanes_exposed", "NHANES"), ("mimiciii", "MIMIC-III")]]
    gap = 3.0
    name_h = 6.6
    row_d = [max(diam(e4[k]["people"]) for k, _ in row) for row in rows]
    top = ya + 6.0
    free = yb - top - sum(row_d) - 2 * name_h
    vgap = free / 2.4
    ytop = top + 0.3 * vgap
    for row, dmax_row in zip(rows, row_d):
        ds = [diam(e4[k]["people"]) for k, _ in row]
        width = sum(ds) + gap * (len(ds) - 1)
        xx = x0 + (x1 - x0 - width) / 2
        cy = ytop + name_h + dmax_row / 2
        for (key, name), dd in zip(row, ds):
            cx = xx + dd / 2
            c.text(cx, cy - dd / 2 - 4.6, name, xa="center")
            c.text(cx, cy - dd / 2 - 1.5, f"≥{e4[key]['people']:,}", colour=K.SLATE, xa="center", kind="number")
            c.dot(cx, cy, dd, K.BLOOD)
            xx += dd + gap
        need(xx - gap <= x1 - 1.0 and xx - gap - width >= x0 + 1.0, f"clinical row fits lane {row}")
        ytop = cy + dmax_row / 2 + vgap
    need(ytop - vgap < yb - 1.5, "clinical circles inside lane")


def age_glyph(c, K, state, cx, cy, w=6.0):
    xa, xb = cx - w / 2, cx + w / 2
    if state == "unknown":
        empty_tile(c, K, cx - 1.4, cy - 1.4, 2.8)
        return
    c.line(xa, cy, xb, cy, colour=K.LIGHT, width_pt=0.9)
    if state == "reported":
        c.dot(cx + 0.4, cy, 1.9, K.INK)
    elif state == "interval":
        a, b = cx - 1.5, cx + 1.9
        c.rect(a, cy - 0.45, b, cy + 0.45, fill=K.INK)
        for xx in (a, b):
            c.line(xx, cy - 1.0, xx, cy + 1.0, colour=K.INK, width_pt=0.9)
    elif state == "censored":
        a = cx - 0.6
        c.line(a, cy - 1.0, a, cy + 1.0, colour=K.INK, width_pt=0.9)
        c.arrow(a, cy, xb + 0.5, cy, colour=K.INK, width_pt=1.0, head=1.5)


def panel_b(c, K):
    c.ref = "b"
    X = R_X0
    c.heading(X, 0.0, "b", "Patient state")
    d = K.d
    person, ret = K.icon("people/person_5.png"), K.icon("symbols/retina.png")
    pw, ph = c.icon(person, X + 7.0, 26.5, 13.0, 34.0, "people/person_5.png")
    need(ph > 32, "person icon height")
    c.icon(ret, X + 26.0, 14.6, 8.4, 8.4, "symbols/retina.png")
    c.text(X + 26.0, 21.2, "retinal photo", colour=K.SLATE, xa="center")
    c.icon(K.icon("symbols/tube.png"), X + 23.6, 34.0, 7.4, 7.4, "symbols/tube.png")
    c.icon(K.icon("symbols/cuff.png"), X + 28.4, 34.2, 6.6, 6.6, "symbols/cuff.png")
    c.text(X + 26.0, 39.8, "blood, vitals", colour=K.SLATE, xa="center")
    c.arrow(X + 13.6, 21.4, X + 19.6, 17.4, colour=K.EYE)
    c.arrow(X + 13.6, 28.6, X + 18.8, 32.2, colour=K.BLOOD)

    ox, oy, width = X + 46.0, 7.4, 16.0
    c.arrow(X + 31.4, 14.4, ox + 3.6, 13.0, colour=K.EYE, width_pt=1.0)
    c.arrow(X + 32.6, 34.0, ox - 0.5, 36.0, colour=K.BLOOD, width_pt=1.0)
    spans = mosaic(c, K, ox, oy, width)
    c.text(ox + width / 2, oy + 2 * width + 2.6, f"{d['state_dim']}-coordinate state", xa="center", kind="number")
    bx = ox + width + 1.6
    for b in ("eye", "shared", "blood"):
        lo, hi = spans[b]
        c.line(bx, lo + 0.25, bx, hi - 0.25, colour=K.block_colour[b], width_pt=1.0)
        c.text(bx + 1.4, (lo + hi) / 2, K.block_label[b], kind="number")

    # observation strip
    top = 49.0
    x_lab, x_t = X, X + 21.0
    s, pitch, row_h = 2.9, 3.7, 4.4
    c.text(x_lab, top, "Observation patterns†", colour=K.SLATE)
    kx = X + 28.6
    c.rect(kx, top - s / 2, kx + s, top + s / 2, fill=K.SLATE)
    c.text(kx + s + 1.0, top, "observed", colour=K.SLATE, kind="key")
    kx2 = kx + s + 12.6
    empty_tile(c, K, kx2, top - s / 2, s)
    c.text(kx2 + s + 1.0, top, "missing", colour=K.SLATE, kind="key")
    y0 = top + 5.0
    n = len(d["strip_rows"][0][2])
    for r, (lab, col, pat) in enumerate(d["strip_rows"]):
        y = y0 + r * row_h
        c.text(x_lab, y, lab)
        for i, bit in enumerate(pat):
            x = x_t + i * pitch
            if bit == "1":
                c.rect(x, y - s / 2, x + s, y + s / 2, fill=K.block_colour[col])
            else:
                empty_tile(c, K, x, y - s / 2, s)
    x_end = x_t + (n - 1) * pitch + s
    bxs = x_end + 1.4
    c.line(bxs, y0 - s / 2, bxs, y0 + row_h + s / 2, colour=K.BLOOD, width_pt=1.0)
    c.text(bxs + 1.3, y0 + row_h / 2, f"{d['n_clin']} values\nwith masks", kind="number")
    y_r = y0 + 2 * row_h
    c.line(bxs, y_r - s / 2, bxs, y_r + s / 2, colour=K.EYE, width_pt=1.0)
    c.text(bxs + 1.3, y_r, f"{d['ret_dim']} features", kind="number")
    y_a = y0 + 3 * row_h + 0.8
    c.text(x_lab, y_a, "Typed age")
    step = 11.6
    for k, stt in enumerate(d["age_states"]):
        cx = x_t + 2.6 + k * step
        age_glyph(c, K, stt, cx, y_a)
        c.text(cx, y_a + 3.6, stt, colour=K.SLATE, xa="center")

    # three small states, absent block pale
    modes = [("retina only", "blood"), ("clinical only", "eye"), ("both", None)]
    mw = 6.0
    for (lab, pale), cx in zip(modes, (X + 85.0, X + 100.0, X + 114.0)):
        mosaic(c, K, cx - mw / 2, y0 - 3.4, mw, pale=pale, gap_frac=0.16)
        c.text(cx, y0 - 3.4 + 2 * mw + 2.4, lab, colour=K.SLATE, xa="center")


def panel_c(c, K):
    c.ref = "c"
    X = R_X0
    d = K.d
    hy = c.heading(X, C_TOP, "c", "Downstream uses")
    xl = X + 8.6
    rows_y = [hy + 3.0, hy + 13.2, hy + 23.4]
    hub = (X + 2.6, rows_y[1])
    for yy in rows_y:
        c.arrow(hub[0] + 1.4, hub[1], xl - 0.9, yy, colour=K.SHARED)
    c.rect(hub[0] - 1.3, hub[1] - 1.3, hub[0] + 1.3, hub[1] + 1.3, fill=K.SHARED)
    ptr = d["pointers"]

    def head(y, s, pointer):
        w = c.text(xl, y, s, kind="number")
        c.text(xl + w + 1.0, y, f"({pointer})", colour=K.SLATE, kind="key")

    y = rows_y[0]
    head(y, f"{d['n_cond']} conditions", ptr["cond"])
    s, p = 3.0, 3.66
    for i in range(d["n_cond"]):
        col = K.EYE if i < d["n_ocular"] else mix(K.SHARED, 0.45)
        c.rect(xl + i * p, y + 1.9, xl + i * p + s, y + 1.9 + s, fill=col)
    c.text(xl + (d["n_ocular"] * p - (p - s)) / 2, y + 6.5, "eye", colour=K.SLATE, xa="center")
    end_cond = xl + (d["n_cond"] - 1) * p + s

    y = rows_y[1]
    head(y, f"{d['n_cbc']} blood count values", ptr["cbc"])
    yb = y + 4.9
    c.icon(K.icon("symbols/retina.png"), xl + 2.2, yb, 4.6, 4.6, "symbols/retina.png")
    c.arrow(xl + 4.9, yb, xl + 8.2, yb, colour=K.EYE)
    cw, ch, gx = 8.6, 4.0, 9.5
    bx0 = xl + 9.1
    for i, lab in enumerate(d["cbc"]):
        xx = bx0 + i * gx
        c.rect(xx, yb - ch / 2, xx + cw, yb + ch / 2, fill=WHITE, edge=K.BLOOD, width_pt=0.75, frame=True)
        c.text(xx + cw / 2, yb, lab, xa="center")
    end_cbc = bx0 + (d["n_cbc"] - 1) * gx + cw
    need(abs(end_cbc - end_cond) < 1.0, f"condition row and blood count row end together {end_cond:.1f} {end_cbc:.1f}")

    # Clinical discovery. Two ordinary boxes that name the two Figure 6 findings. No number is drawn.
    y = rows_y[2]
    head(y, "Clinical discovery", ptr["discovery"])
    yb = y + 4.9
    pad = 1.5
    xx = xl
    for lab in d["discovery"]:
        need(not re.search(r"\d", lab), "discovery label carries no number")
        w_lab, _ = text_box(lab, "label")
        c.rect(xx, yb - ch / 2, xx + pad + w_lab + pad, yb + ch / 2, fill=WHITE, edge=K.BLOOD, width_pt=0.75,
               frame=True)
        c.text(xx + pad, yb, lab)
        xx += pad + w_lab + pad + 2.0
    need(xx < end_cbc, "discovery boxes stay inside the row")


def tile_bar(c, K, x, y, n, s, pitch, col, pattern=None):
    for i in range(n):
        xx = x + i * pitch
        if pattern is None or pattern[i] == "1":
            c.rect(xx, y - s / 2, xx + s, y + s / 2, fill=col)
        else:
            empty_tile(c, K, xx, y - s / 2, s)
    return x + (n - 1) * pitch + s


def panel_d(c, K):
    c.ref = "d"
    d = K.d
    hy = c.heading(0.0, D_TOP, "d", "Architecture")
    y_r, y_m, y_c = hy + 14.0, hy + 26.0, hy + 38.0
    bh = 10.4
    lh = 1.3 * SZ["label"] / PX

    for yy, name, sub in ((y_r, "Retinal only", "BRSET"), (y_m, "Paired", "AI-READI"),
                          (y_c, "Clinical only", f"{d['n_clin_sources']} sources")):
        c.text(0.0, yy, name)
        c.text(0.0, yy + lh, sub, colour=K.SLATE, kind="number")
    xj = 14.6
    w_ret, _ = text_box("Retinal only", "label")
    w_par, _ = text_box("Paired", "label")
    w_cli, _ = text_box("Clinical only", "label")
    c.line(w_ret + 0.7, y_r, xj + 1.0, y_r, colour=K.EYE)
    c.line(w_par + 0.7, y_m, xj, y_m, colour=K.INK)
    c.line(xj, y_r, xj, y_c, colour=K.INK)
    c.dot(xj, y_r, 1.05, K.INK)
    c.dot(xj, y_c, 1.05, K.INK)

    c.icon(K.icon("symbols/retina.png"), 18.6, y_r, 4.6, 4.6, "symbols/retina.png")
    c.text(18.6, y_r - 4.7, "Photographs", colour=K.SLATE, xa="center")
    dx0, dx1 = 26.3, 43.1
    c.arrow(21.1, y_r, dx0, y_r, colour=K.EYE)
    box(c, K, dx0, dx1, y_r, bh, ["Adapted", d["tower_name"].replace("adapted-", "")], edge=K.EYE, tag="frozen")
    bar_x, n_t, s_t, p_t = 45.9, 5, 2.0, 2.4
    c.arrow(dx1, y_r, bar_x - 0.3, y_r, colour=K.EYE)
    bar_end = tile_bar(c, K, bar_x, y_r, n_t, s_t, p_t, K.EYE)
    c.text((bar_x + bar_end) / 2, y_r + 3.3, f"{d['ret_dim']} features", colour=K.SLATE, xa="center", kind="number")
    ex0, ex1 = 59.9, 81.1
    c.arrow(bar_end + 0.3, y_r, ex0, y_r, colour=K.EYE)
    box(c, K, ex0, ex1, y_r, bh, ["Retinal projection", "and pooling"], edge=K.EYE, tag="MLP")

    c.line(w_cli + 0.7, y_c, xj, y_c, colour=K.BLOOD)
    c.arrow(xj, y_c, bar_x - 0.3, y_c, colour=K.BLOOD)
    bar_end_c = tile_bar(c, K, bar_x, y_c, n_t, s_t, p_t, K.BLOOD, pattern="11010")
    c.text((bar_x + bar_end_c) / 2, y_c - 4.3, f"{d['n_clin']} values\nwith masks", colour=K.SLATE, xa="center",
           kind="number")
    c.arrow(bar_end_c + 0.3, y_c, ex0, y_c, colour=K.BLOOD)
    box(c, K, ex0, ex1, y_c, bh, ["Clinical encoder"], edge=K.BLOOD, tag="MLP")

    ax0, ax1, ah = 63.7, 77.3, 5.0
    box(c, K, ax0, ax1, y_m, ah, ["Typed age"], edge=K.SLATE)
    axc = (ax0 + ax1) / 2
    c.arrow(axc, y_m - ah / 2, axc, y_r + bh / 2, colour=K.SLATE)
    c.arrow(axc, y_m + ah / 2, axc, y_c - bh / 2, colour=K.SLATE)

    px0, px1 = 83.8, 96.2
    box(c, K, px0, px1, y_m, bh, ["Shared", "posterior"], edge=K.SHARED, tag="MLP")
    pxc = (px0 + px1) / 2
    col_x, col_w = 98.8, 4.4
    c.arrow(ex1, y_r, col_x, y_r, colour=K.EYE)
    c.arrow(ex1, y_c, col_x, y_c, colour=K.BLOOD)
    c.arrow(pxc, y_r, pxc, y_m - bh / 2, colour=K.EYE)
    c.arrow(pxc, y_c, pxc, y_m + bh / 2, colour=K.BLOOD)
    c.dot(pxc, y_r, 0.95, K.EYE)
    c.dot(pxc, y_c, 0.95, K.BLOOD)
    c.arrow(px1, y_m, col_x, y_m, colour=K.SHARED)

    tiles_per_block = 4
    need(d["block"] % tiles_per_block == 0, "tile size divides block")
    th = (y_m - y_r) / tiles_per_block
    col_top = y_m - 1.5 * tiles_per_block * th
    for bi, col in enumerate((K.EYE, K.SHARED, K.BLOOD)):
        for t in range(tiles_per_block):
            yy = col_top + (bi * tiles_per_block + t) * th
            c.rect(col_x, yy, col_x + col_w, yy + th, fill=col, edge=K.INK, width_pt=0.5)
    col_bot = col_top + 3 * tiles_per_block * th
    need(abs(col_top + 0.5 * tiles_per_block * th - y_r) < 0.01, "retinal row meets retinal block")
    need(abs(col_top + 2.5 * tiles_per_block * th - y_c) < 0.01, "clinical row meets clinical block")
    c.text(col_x + col_w / 2, col_top - 2.4, f"{d['state_dim']}-coordinate posterior mean", xa="center",
           kind="number")
    lx = col_x + col_w + 1.4
    c.text(lx, y_r, K.block_label["eye"], kind="number")
    c.text(lx, y_m - 2.2, K.block_label["shared"], kind="number")
    c.text(lx, y_c, K.block_label["blood"], kind="number")

    # outputs, native linear heads, auxiliary MLP decoders and separately fitted readouts
    bus_x, ob0, ob1, obh = 130.0, 131.8, 156.4, 4.2
    out = dict((k, lab) for k, lab in d["outputs"])
    groups = [("Native linear heads", ["cond", "cbc"], None), ("Auxiliary MLP decoders", ["clin_rec", "ret_rec"], None),
              ("Frozen-state reuse", ["readout"], "dash")]
    # training source table, icon column heads in the colours of panel a, one dot where a source supplies the objective
    cols = [("paired", K.INK), ("clinical", K.BLOOD), ("retinal", K.EYE)]
    cpitch = 5.1
    cx0 = 161.4
    need(cx0 - 2.0 > ob1 + 0.6, "table clears the output boxes")
    c.text(cx0 + cpitch, hy - 1.5, "Training source", xa="center", ya="bottom")
    icon_y = hy + 2.6
    ret, tube = K.icon("symbols/retina.png"), K.icon("symbols/tube.png")
    c.icon(ret, cx0 - 1.05, icon_y, 2.6, 2.6, "symbols/retina.png")
    c.icon(tube, cx0 + 1.05, icon_y, 2.6, 2.6, "symbols/tube.png")
    c.icon(tube, cx0 + cpitch, icon_y, 2.6, 2.6, "symbols/tube.png")
    c.icon(ret, cx0 + 2 * cpitch, icon_y, 2.6, 2.6, "symbols/retina.png")
    rule_y = icon_y + 2.2
    c.line(cx0 - 2.6, rule_y, cx0 + 2 * cpitch + 1.8, rule_y, colour=K.SLATE, width_pt=0.5)
    y = rule_y + 1.4
    row_y = {}
    for gname, keys, dash in groups:
        c.text(ob0, y + 1.2, gname, colour=K.SLATE)
        y += 3.1
        for key in keys:
            cy = y + obh / 2
            row_y[key] = cy
            c.rect(ob0, y, ob1, y + obh, fill=WHITE, edge=K.SLATE, width_pt=0.75, frame=True, dash=dash)
            lab = out[key]
            c.text((ob0 + ob1) / 2, cy, lab, xa="center", kind="number" if lab[0].isdigit() else "label")
            c.arrow(bus_x, cy, ob0, cy, colour=K.SHARED, dash=dash)
            y += obh + 1.1
        y += 1.3
    need(max(row_y.values()) + obh / 2 <= IH - 0.3, "outputs inside the page")
    c.line(col_x + col_w, y_m, bus_x, y_m, colour=K.SHARED)
    c.line(bus_x, min(row_y.values()), bus_x, max(row_y.values()), colour=K.SHARED)
    for key, cy in row_y.items():
        for i, (src, col) in enumerate(cols):
            if src in d["train_rows"][key]:
                c.dot(cx0 + i * cpitch, cy, 1.7, col)
    return col_bot, row_y


def draw(payload):
    K = Ctx(payload)
    c = Canvas()
    panel_a(c, K)
    panel_b(c, K)
    panel_c(c, K)
    col_bot, _ = panel_d(c, K)
    need(col_bot <= IH - 0.5, "panel d column inside the page")
    return c


def render(payload):
    c = draw(payload)
    return [(FIG_ID, c.fig, W, H)]

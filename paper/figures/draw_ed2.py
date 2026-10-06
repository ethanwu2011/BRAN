"""Drawing code for Extended Data Fig. 2.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed2.py
(SHA-256 3536458358e5faa4819135a83f59d71188ec9973a7ee35096a3b17596528bd28).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import math
import bran_plotly_style as st
st.use_nature_medicine()
import ed_nm_canvas as E
FIG_ID = "ed2"
INK, SLATE, GREY, LIGHT = st.INK, st.SLATE, st.GREY, st.LIGHT
W = st.FULL_W


def need(cond, msg):
    if not cond:
        raise SystemExit(f"CHECK FAILED  {msg}")


def ge(n: int) -> str:
    """A floored lower bound as printed on the canvas."""
    return f"≥{int(n):,}"


T_PT = 6.0
B_PT = 5.5
LP = {T_PT: 8.8, B_PT: 8.1}
D_LP = 8.1
PAD = 2.4
A_X0, A_X1 = 0.0, 318.0
B_X0, B_X1 = 334.0, float(W)
KEEP = ("Fig. 6", "ED Fig. 1", "ED Fig. 6", "ED Fig. 7", "1,542 or 1,543", "385 or 386", "(42,558 images)",
        "aged 19 to 79", "aged 40 to 79", "aged 65 to 79", "aged 40 and older", "August 2026", "5 October 2026")


class Paper:
    def __init__(self, cv):
        self.ax = cv.axis((0.0, float(W), 0.0, float(cv.H)), (0.0, float(W)), (0.0, float(cv.H)), xline=False)

    def seg(self, xs, ys, width=0.75):
        self.ax.seg(xs, ys, INK, width)

    def arrow_down(self, x, y0, y1):
        self.ax.seg([x, x], [y0, y1 - 2.6], INK, 0.75)
        self.ax.dot(x, y1 - 2.2, INK, size=4.4, symbol="triangle-down")


def box(cv, x0, y0, w, title, detail):
    laid = [(ln, T_PT, True, INK) for ln in E.wrap(title, w - 2 * PAD - 3, T_PT, bold=True, keep=KEEP)]
    laid += [(ln, B_PT, False, SLATE) for ln in E.wrap(detail, w - 2 * PAD - 3, B_PT, keep=KEEP)]
    h = 2 * PAD + sum(LP[s] for _, s, _, _ in laid)
    y = y0 + PAD
    for ln, size, bold, colour in laid:
        cv.text(x0 + PAD + 1.5, y + LP[size] / 2, ln, size=size, bold=bold, color=colour)
        y += LP[size]
    cv.rect(x0, y0, x0 + w, y0 + h, edge=INK, width=0.6)
    return x0, y0, x0 + w, y0 + h


def panel_a(cv, p, top):
    cv.heading(A_X0, top + 11, "a", "AI-READI participant flow")
    P = Paper(cv)
    f = {r[0]: r for r in p["draw"]["flow"]}
    cx = (A_X0 + A_X1) / 2
    y = top + 16
    b1 = box(cv, 40.0, y, 238.0, f["release"][1], f["release"][2])
    lw, rw = 170.0, 138.0
    lx, rx = A_X0 + 1.0, A_X1 - rw
    yb = b1[3] + 16.0
    b2 = box(cv, lx, yb, lw, f["development"][1], f["development"][2])
    b3 = box(cv, rx, yb, rw, f["official"][1], f["official"][2])
    ymid = b1[3] + 8.0
    P.seg([cx, cx], [b1[3], ymid])
    lcx, rcx = (b2[0] + b2[2]) / 2, (b3[0] + b3[2]) / 2
    P.seg([lcx, rcx], [ymid, ymid])
    P.arrow_down(lcx, ymid, b2[1])
    P.arrow_down(rcx, ymid, b3[1])
    b4 = box(cv, lx, b2[3] + 13.0, lw, f["folds"][1], f["folds"][2])
    P.arrow_down(lcx, b2[3], b4[1])
    b5 = box(cv, lx, b4[3] + 13.0, lw, f["fold0"][1], f["fold0"][2])
    P.arrow_down(lcx, b4[3], b5[1])
    b6 = box(cv, rx, b3[3] + 13.0, rw, f["bran_scoring"][1], f["bran_scoring"][2])
    P.arrow_down(rcx, b3[3], b6[1])
    return max(b5[3], b6[3])


def table_b(cv, p, top):
    cv.heading(B_X0, top + 11, "b", "Development characteristics")
    rows = p["draw"]["table1"]
    x_lab, x_dev, x_rel = B_X0, B_X0 + 200.0, B_X1 - 1.0
    pitch = 7.8
    y = top + 16
    cv.line(B_X0, y, B_X1, y, width=1.0)
    y += PAD + pitch / 2
    dev_n = p["panels"]["b"]["title"].split("(")[1].split(")")[0]
    rel_n = p["panels"]["b"]["title"].split("(")[2].split(")")[0]
    cv.text(x_lab, y, "Characteristic", size=B_PT, bold=True)
    cv.text(x_dev, y, f"Development, {dev_n}", size=B_PT, bold=True, xa="right")
    cv.text(x_rel, y, f"Local release, {rel_n}", size=B_PT, bold=True, xa="right")
    y += pitch / 2 + PAD
    cv.line(B_X0, y, B_X1, y, width=0.5)
    y += 1.0
    bw = E.text_w("PLACEHOLDER", "small") + 8
    cur, sg_top, sg_bot = None, None, None
    for g, lab, dev, rel, *_ in rows:
        if g != cur:
            cur = g
            y += pitch / 2 + 0.6
            cv.text(x_lab, y, g, size=B_PT, bold=True)
            y += pitch / 2
        y += pitch / 2
        cv.text(x_lab + 5, y, lab, size=B_PT)
        if g == "Study group":
            sg_top = y - pitch / 2 if sg_top is None else sg_top
            sg_bot = y + pitch / 2
        else:
            for x, v in ((x_dev, dev), (x_rel, rel)):
                if v == "PLACEHOLDER":
                    cv.placeholder(x - bw, y - 3.6, x, y + 3.6)
                else:
                    cv.text(x, y, v, size=B_PT, xa="right", color=SLATE if v == "Not counted" else INK)
        if g == "Study group":
            cv.text(x_rel, y, rel, size=B_PT, xa="right")
        y += pitch / 2
    cv.placeholder(x_dev - bw, sg_top + 1.0, x_dev, sg_bot - 1.0)
    y += PAD
    cv.line(B_X0, y, B_X1, y, width=1.0)
    return y


C_PITCH = 7.7
C_LABW, C_AXW = 92.0, 100.0
C_MAX = 60.0


def panel_c(cv, p, y0):
    cv.heading(0.0, y0 + 11, "c", "Self-reported conditions")
    top = y0 + 15
    rows = p["draw"]["conditions"]
    blocks = p["draw"]["blocks"]
    lists = [(0.0, [blocks[0], blocks[2]]), (345.0, [blocks[1]])]
    bottoms, bars = [], []
    for x0, bl in lists:
        n_slots = sum(1 + len(b) for _, b in bl)
        a0, a1 = x0 + C_LABW + 5.0, x0 + C_LABW + 5.0 + C_AXW
        y_axis = top + n_slots * C_PITCH + 1.5
        ax = cv.axis((a0, a1, top, y_axis), (0.0, C_MAX), (top, y_axis), xticks=[0, 20, 40, 60],
                     xtext=["0", "20", "40", "60"])
        y = top
        for name, _ in bl:
            cv.text(x0, y + C_PITCH / 2, name, size=B_PT, bold=True)
            y += C_PITCH
            mem = [r for r in rows if r[0] == name]
            ax.seg([0.0, 0.0], [y, y + len(mem) * C_PITCH], INK, 0.75)
            for r in mem:
                ym = y + C_PITCH / 2
                cv.text(x0 + C_LABW, ym, r[2], size=B_PT, xa="right")
                ax.inside_x(r[3])
                cv.rect(ax.X(0.0), ym - 2.6, ax.X(r[3]), ym + 2.6, fill=GREY, edge=GREY, width=0.4)
                bars.append((r[1], r[3]))
                cv.text(a1 + 5.0, ym, f"{r[3]:.1f} ({ge(r[4])} of {ge(r[5])})", size=B_PT)
                y += C_PITCH
        cv.text((a0 + a1) / 2, y_axis + 15.0, "Prevalence among labelled (%)", size=B_PT, xa="center")
        bottoms.append(y_axis + 20.0)
    return max(bottoms), bars


D_COLS = [("Cohort", 62.0), ("Analysis", 150.0), ("Encoder", 94.0), ("People", 196.0), ("Deaths", 76.0),
          ("Shown in", 60.0)]
D_GUT = 8.4
D_MUTED = ("Cross-sectional", "Not studied", "Not estimated", "Insufficient support, no estimate")


def panel_d(cv, p, y0):
    assert abs(sum(w for _, w in D_COLS) + D_GUT * 5 - W) < 1e-6, sum(w for _, w in D_COLS) + D_GUT * 5
    cv.heading(0.0, y0 + 11, "d", "External cohorts")
    pitch = D_LP
    xs, x = [], 0.0
    for _, w in D_COLS:
        xs.append((x, w))
        x += w + D_GUT
    y = y0 + 16
    cv.line(0, y, W, y, width=1.0)
    y += PAD + pitch / 2
    for (h, _), (x0, _) in zip(D_COLS, xs):
        cv.text(x0, y, h, size=B_PT, bold=True)
    y += pitch / 2 + PAD
    cv.line(0, y, W, y, width=0.5)
    bw = E.text_w("PLACEHOLDER", "small") + 8
    for gname, items in p["draw"]["external"]:
        y += 1.5
        cv.text(0, y + pitch / 2, gname, size=B_PT, bold=True)
        y += pitch
        for cells in items:
            laid = []
            for i, ((_, w), t) in enumerate(zip(xs, cells)):
                laid.append(None if t is None else E.wrap(t, w - (5.0 if i == 0 else 0.0), B_PT, keep=KEEP))
            n = max(1 if v is None else len(v) for v in laid)
            yy = y + 1.5 + pitch / 2
            for i, ((x0, _), lines, t) in enumerate(zip(xs, laid, cells)):
                if lines is None:
                    cv.placeholder(x0, yy - 3.6, x0 + bw, yy + 3.6)
                    continue
                for j, ln in enumerate(lines):
                    cv.text(x0 + (5.0 if i == 0 else 0.0), yy + j * pitch, ln, size=B_PT,
                            color=SLATE if t in D_MUTED else INK)
            y += n * pitch + 2.5
    y += 1.0
    cv.line(0, y, W, y, width=1.0)
    return y


def draw(payload):
    H = st.FULL_H
    for _ in range(2):
        cv = E.Canvas(W, H)
        ya = panel_a(cv, payload, 0.0)
        yb = table_b(cv, payload, 0.0)
        yc1, bars = panel_c(cv, payload, max(ya, yb) + 4.0)
        yd1 = panel_d(cv, payload, yc1 + 2.0)
        need_h = int(math.ceil(yd1 + 2.0))
        if need_h == H:
            break
        need(need_h <= st.FULL_H, f"ED2 needs {need_h} px ({need_h / st.MM:.1f} mm), more than 170 mm")
        H = need_h
    cv.flush()
    return cv, H, bars


def render(payload):
    cv, H, _ = draw(payload)
    return [(FIG_ID, cv.fig, W, H)]

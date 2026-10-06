"""Drawing code for Extended Data Fig. 1.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed1.py
(SHA-256 7889bfca30ef5b9ed006307ca46255d43a74c098487ec61a7b9f29fb9b13e254).

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
FIG_ID = "ed1"
MINUS = E.MINUS
fmt, ci = E.fmt, E.ci


def need(cond, msg):
    if not cond:
        raise SystemExit(f"CHECK FAILED  {msg}")


BODY = 5.5
HEAD_PT = 5.5
RP = 9.0
GP = 11.0
DOT = 4.2
W = st.FULL_W


def col_layout(rows):
    """Column positions from measured text widths, so the table fits 180 mm."""
    g = 4.0
    w_use = max(E.text_w(r["use"], BODY) + (6 if r["sub"] else 0) for r in rows if r["use"])
    w_b = max([E.text_w(r["bran"], BODY) for r in rows if r["bran"]] + [E.text_w("BRAN (95% CI)", BODY)])
    w_cn = max(E.text_w(r["comp_name"], BODY) for r in rows)
    w_c = max(E.text_w(r["comp"], BODY) for r in rows if r["comp"])
    w_d = max([E.text_w(r["diff"], BODY) for r in rows if r["diff"]] + [E.text_w(r["note"], BODY) for r in rows if r["note"]])
    x = 0.0
    X = {}
    X["use"] = x
    x += w_use + g
    X["bran_r"] = x + w_b
    x += w_b + g + 1
    X["cname"] = x
    x += w_cn + g
    X["comp_r"] = x + w_c
    x += w_c + g + 1
    X["diff_r"] = x + w_d
    x += w_d + g + 1
    X["ax0"] = x
    X["hw"] = W - E.text_w("±0.015", "small") - 1
    X["ax1"] = X["hw"] - 4
    need(X["ax1"] - X["ax0"] >= E.text_w("comparator", "small") + E.text_w("BRAN", "small") + 4,
         f"forest axis too narrow {X['ax1'] - X['ax0']:.1f} px")
    return X


def draw_a(cv, p, y_top):
    d = p["draw"]
    rows = d["rows"]
    C = d["colours"]
    X = col_layout(rows)
    cv.heading(0, y_top + 12, "a", "Uses against comparators")
    # key, in the figure rather than in the legend
    kx, ky = 250.0, y_top + 7
    kax = cv.axis((kx, W, ky - 4, ky + 4), (kx, W), (ky - 4, ky + 4), xline=False)
    for lab, col, hollow in (("Foundation model", C["fm"], False), ("Raw input or baseline", C["raw"], False),
                             ("BRAN without the eye or before compression", C["bran"], False),
                             ("Interval includes 0", C["raw"], True)):
        kax.dot(kx + 2.5, ky, col, hollow=hollow, size=DOT)
        kx += 7
        kx += cv.text(kx, ky, lab, size="small") + 9
    need(kx - 9 <= W, f"key fits {kx}")
    top = y_top + 16
    cv.line(0, top, W, top, width=1.0)
    h1, h2 = top + 5.6, top + 13.0
    for x, s1, s2, xa in ((X["use"], "Use", "and measure", "left"), (X["bran_r"], "BRAN", "(95% CI)", "right"),
                          (X["cname"], "Strongest or primary", "comparator", "left"),
                          (X["comp_r"], "Comparator", "(95% CI)", "right"),
                          (X["diff_r"], "BRAN minus comparator", "(95% CI)", "right")):
        cv.text(x, h1, s1, size=HEAD_PT, bold=True, xa=xa)
        cv.text(x, h2, s2, size=HEAD_PT, bold=True, xa=xa)
    xz = (X["ax0"] + X["ax1"]) / 2
    for yy, a_, b_ in ((h1, "Favours", "Favours"), (h2, "comparator", "BRAN")):
        cv.text(X["ax0"], yy, a_, size="small")
        cv.text(X["ax1"], yy, b_, size="small", xa="right")
    y = top + 18.5
    cv.line(0, y, W, y, width=0.5)
    y += 2
    names = {k: (n, dd) for k, n, dd in d["levels"]}
    cur = None
    drawn = []
    for r in rows:
        if r["level"] != cur:
            cur = r["level"]
            y += GP / 2
            n, desc = names[cur]
            w = cv.text(0, y, n, size=BODY, bold=True)
            cv.text(w + 5, y, desc, size="small", color=st.SLATE)
            y += GP / 2
        y += RP / 2
        if r["use"]:
            cv.text(6 if r["sub"] else 0, y, r["use"], size=BODY, color=st.SLATE if r["sub"] else st.INK)
        if r["bran"]:
            cv.text(X["bran_r"], y, r["bran"], size=BODY, xa="right")
        cv.text(X["cname"], y, r["comp_name"], size=BODY)
        if r["comp"]:
            cv.text(X["comp_r"], y, r["comp"], size=BODY, xa="right", color=st.SLATE)
        if r["status"] == "paired":
            cv.text(X["diff_r"], y, r["diff"], size=BODY, xa="right")
            h = r["half"]
            xr = (-h, h) if r["better"] == "higher" else (h, -h)
            ax = cv.axis((X["ax0"], X["ax1"], y - 3.6, y + 3.6), xr, (y - 3.6, y + 3.6), xline=False)
            cv.line(ax.X(0.0), y - RP / 2 + 0.6, ax.X(0.0), y + RP / 2 - 0.6, color=E.MUTED, width=0.5, dash=E.DASH)
            # the line's own axis, from minus to plus its half-range, with an outward tick at each end
            yb_ = y + 3.0
            cv.line(X["ax0"], yb_, X["ax1"], yb_, color=E.MUTED, width=0.5)
            for xe in (X["ax0"], X["ax1"]):
                cv.line(xe, yb_, xe, yb_ + 1.3, color=E.MUTED, width=0.5)
            e_, lo, hi = r["diff_values"]
            ax.hci(y, e_, lo, hi, C[r["cls"]], hollow=lo <= 0 <= hi, size=DOT, width=1.0)
            cv.text(X["hw"], y, "±" + (f"{h:.3f}".rstrip("0").rstrip(".")), size="small", color=st.SLATE)
            drawn.append((ax.n, e_, lo, hi))
        elif r["status"] == "none":
            cv.text(X["diff_r"], y, r["note"], size=BODY, xa="right", color=st.SLATE)
        else:
            cv.placeholder(X["diff_r"] - E.text_w("PLACEHOLDER", "small") - 10, y - RP / 2 + 0.1, X["ax1"],
                           y + RP / 2 - 0.1)
        y += RP / 2
    y += 2
    cv.line(0, y, W, y, width=1.0)
    return y, drawn


PB_H = 52.0


def draw_b(cv, p, y0):
    d = p["draw"]["pb"]
    C = p["draw"]["colours"]
    arms = [("both", "Paired", C["shared"]), ("blood_only", "Blood arm", C["blood"]), ("eye_only", "Eye arm", C["eye"])]
    cv.heading(0, y0 + 10, "b", "Predecessor official test")
    top, bot = y0 + 16, y0 + 16 + PB_H
    drawn = []
    ax = cv.axis((70, 210, top, bot), (0.85, 1.40), (top, bot), xticks=[0.9, 1.0, 1.1, 1.2, 1.3],
                 xtext=["0.9", "1.0", "1.1", "1.2", "1.3"])
    for j, (k, lab, col) in enumerate(arms):
        yy = top + 9 + j * 17
        ax.hci(yy, *d["arms"][k], col, size=DOT)
        cv.text(64, yy, lab, size="small", xa="right")
        drawn.append((ax.n, *d["arms"][k]))
    cv.text(140, bot + 17, "Normalized loss, lower is better", size="small", xa="center")
    bx = cv.axis((330, 450, top, bot), (-0.04, 0.12), (top, bot), xticks=[-0.04, 0.0, 0.04, 0.08],
                 xtext=[f"{MINUS}0.04", "0", "0.04", "0.08"])
    cv.line(bx.X(0.0), top, bx.X(0.0), bot, color=E.MUTED, width=0.5, dash=E.DASH)
    cv.text(492, top + 1, "Holm P", size="small", xa="right", bold=True)
    for j, (k, lab, col) in enumerate((("blood_minus_both", "Blood arm minus paired", C["blood"]),
                                        ("eye_minus_both", "Eye arm minus paired", C["eye"]))):
        yy = top + 17 + j * 17
        t = d["con"][k]["t"]
        bx.hci(yy, *t, col, hollow=t[1] <= 0 <= t[2], size=DOT)
        cv.text(324, yy, lab, size="small", xa="right")
        cv.text(492, yy, fmt(d["con"][k]["p"], 3), size="small", xa="right")
        drawn.append((bx.n, *t))
    cv.text(390, bot + 17, "Loss difference, positive favours paired", size="small", xa="center")
    # one row per target, the three arms on separate sub-rows so that no two markers overlap
    tx = cv.axis((560, 676, top, bot), (0.75, 1.35), (top, bot), xticks=[0.8, 1.0, 1.2], xtext=["0.8", "1.0", "1.2"])
    pitch, dy, size = 10.5, 3.3, 2.6
    need(dy >= size + 0.5 and 2 * dy + size <= pitch - 1.0, "per-target markers separate")
    for j, t in enumerate(d["targets"]):
        yy = top + 5 + j * pitch
        need(yy + dy + size / 2 <= bot, "per-target rows inside the axis")
        for off, (k, _, col) in zip((-dy, 0.0, dy), arms):
            tx.dot(d["per"][k][t], yy + off, col, size=size)
        cv.text(554, yy, d["short"][t], size="small", xa="right")
    cv.text(618, bot + 17, "Normalized loss by target", size="small", xa="center")
    kx, ky = 506, y0 + 7
    kax = cv.axis((kx, W, ky - 4, ky + 4), (kx, W), (ky - 4, ky + 4), xline=False)
    for _, lab, col in arms:
        kax.dot(kx + 2.5, ky, col, size=DOT)
        kx += 7
        kx += cv.text(kx, ky, lab, size="small") + 8
    need(kx - 8 <= W, "b key fits")
    return bot + 24, drawn


def draw(payload):
    probe = E.Canvas(W, st.FULL_H)
    ya, _ = draw_a(probe, payload, 0)
    H = int(math.ceil(ya + 2 + 16 + PB_H + 24))
    need(H <= st.FULL_H, f"height {H} px ({H / st.MM:.1f} mm) exceeds 170 mm")
    cv = E.Canvas(W, H)
    ya, drawn_a = draw_a(cv, payload, 0)
    yb, drawn_b = draw_b(cv, payload, ya + 2)
    need(yb <= H + 0.5, f"panel b inside {yb} {H}")
    cv.flush()
    return cv, H, drawn_a + drawn_b


def render(payload):
    cv, H, _ = draw(payload)
    return [(FIG_ID, cv.fig, W, H)]

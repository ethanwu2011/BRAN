"""Drawing code for Extended Data Fig. 4.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed3b.py
(SHA-256 38500125c2c9bd1823a4dbc30e64b62084bd220643ae231d02f4bb97432f5519).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import bran_plotly_style as st
import ed_canvas as E
import fig3_nm_kit as kit
kit.apply_nm_sizes()
NAME = "ed3b"
W = kit.NM_W
INK, SLATE = st.INK, st.SLATE
MINUS = E.MINUS
NAT = "bran_native_heads"
LAB_EYE = "labrador_clinical_retinal_age"
LAB_EYE_LINES = ("Labrador + clinical +", "BRAN eye features + age")
MS_DOT = 4.0
signed, plain = E.signed, E.plain


def colour(role):
    return kit.colour(role)


def sig(t):
    return 1 if t[1] > 0 else (-1 if t[2] < 0 else 0)


LAB_GAP = 8.2
F_HEAD = 13
F_PITCH = 8.2
F_LAB_ABOVE, F_LAB_BELOW = 3.0, 1.5
F_LAB_W, F_COL_W, F_GAP = 104.0, 98.0, 10.0
F_AUC_LIM = (0.56, 0.72)
F_DIF_LIM = (-0.02, 0.085)
F_BLOCK_X = (0.0, 344.0)
F_FIX_SYMBOL, F_FIX_SIZE = "diamond", 4.0


def two_line_row(y, pitch, above, below):
    h = above + pitch + LAB_GAP + below
    return y + above + (pitch + LAB_GAP) / 2, h


def f_rows(groups, top):
    y, heads, rows = top, [], {}
    for name, arms in groups:
        heads.append((name, y + F_PITCH / 2))
        y += F_PITCH
        for k, _, _ in arms:
            if k == LAB_EYE:
                rows[k], p = two_line_row(y, F_PITCH, F_LAB_ABOVE, F_LAB_BELOW)
            else:
                rows[k], p = y + F_PITCH / 2, F_PITCH
            y += p
    return heads, rows, y


F_TOP = F_HEAD + 6.0
F_BOT_ROWS = F_TOP + 4 * F_PITCH + 18 * F_PITCH - 4 * F_PITCH
F_BOT = F_BOT_ROWS + 2 + 26
GP_HEAD = F_BOT + 16
G_BOX = (40.0, 250.0, GP_HEAD + 22, GP_HEAD + 112)
G_XR = (-0.3, 2.3)
G_YR = (0.71, 0.61)
G_LABX = 290.0
G_SWATCH = G_LABX - 5.0
G_NUM_R = 452.0
G_BND_R = [512.0, 566.0, 620.0]
READOUTS = ["Fixed C = 1", "Shared C", "Per-arm tuned"]
E_HALF, E_LAB_CLEAR = 4.2, 2.0
R3 = G_BOX[3] + 40
C_LAB_R = 110.0
C_AX = [(116.0, 216.0), (232.0, 332.0)]
C_AUC_LIM = [(0.60, 0.66), (0.66, 0.75)]
C_TEST_LIM = [(-0.01, 0.025), (-0.01, 0.04)]
C_ROW1 = R3 + 21.0
C_AX1_BOT = C_ROW1 + 4 * F_PITCH + 5
C_HEAD2 = C_AX1_BOT + 30
C_ROW2 = C_HEAD2 + 11
C_TEST_DY = 9.0
C_AX2_BOT = C_ROW2 + 3 * C_TEST_DY + 5
D_X0 = 350.0
D_LAB_R = 470.0
D_AX = (478.0, 640.0)
D_LIM = (-0.016, 0.010)
D_TOP = R3 + 20.0
D_ARMS = [("bran_retinal_state", "BRAN retinal state", "eye"), ("bran_retinal_state_age", "BRAN retinal state + age", "eye"),
          ("bran_combined_state", "BRAN combined state", "shared")]
D_MACROS = [("all_26", "26 conditions"), ("eye_4", "Four eye conditions"), ("non_eye_22", "22 other conditions")]
D_GATES = [("g1_clinical_only_bit_identity_aireadi", "Clinical-only route bit-identical"),
           ("g2_frozen_tensor_digests", "Frozen tensors unchanged"),
           ("g3_pilot_and_exact_checkpoint_reload", "Checkpoints reload exactly"),
           ("g4a_eye_alone_paired_interval_above_zero", "Eye alone better than R7"),
           ("g4b_combined_native_macro_within_0_005_of_r7", "Native combined within 0.005"),
           ("g6_clinical_only_teacher_preservation_zero", "Clinical-only teacher preserved")]


def d_layout():
    y, heads, rows = D_TOP, [], {}
    for arm, lab, _ in D_ARMS:
        heads.append((lab, y))
        for m, _ in D_MACROS:
            y += F_PITCH
            rows[(arm, m)] = y
        y += F_PITCH + 1
    native = y + 1
    return heads, rows, native


D_HEADS, D_ROWS, D_NATIVE = d_layout()
D_BOT = D_NATIVE + 6
D_GATE_TOP = D_BOT + 40
H = int(D_GATE_TOP + 9 * len(D_GATES) + 16)
assert H <= kit.NM_H_MAX, H


def f_key(cv, y):
    items = [("dot", "Per-arm tuned", SLATE), ("diamond", "Fixed C = 1", SLATE), ("ring", "Interval spans 0", SLATE),
             ("tick", "Prespecified bound", INK), ("tick", "All-arm bound", st.GREY)]
    widths = [7 + E.text_w(t, st.SIZES["tick"]) for _, t, _ in items]
    x = W - 4 - (sum(widths) + 8 * (len(items) - 1))
    kax = cv.axis((x - 5, W - 4, y - 5, y + 5), (x - 5, W - 4), (y - 5, y + 5), xline=False)
    for (kind, t, c), w in zip(items, widths):
        if kind == "tick":
            cv.line(x + 2, y - 3.0, x + 2, y + 3.0, color=c, width=0.9)
        elif kind == "diamond":
            kax.dot(x + 2, y, c, hollow=True, size=F_FIX_SIZE, symbol=F_FIX_SYMBOL)
        else:
            kax.dot(x + 2, y, c, hollow=(kind == "ring"), size=MS_DOT)
        cv.text(x + 7, y, t, size="tick")
        x += w + 8


def row_label(cv, x, y, key, lab):
    if key != LAB_EYE:
        cv.text(x, y, lab, size="tick", xa="right")
        return
    for ln, dy in zip(LAB_EYE_LINES, (-LAB_GAP / 2, LAB_GAP / 2)):
        cv.text(x, y + dy, ln, size="tick", xa="right")


def draw_a(cv, V, X):
    kit.heading(cv, 0, F_HEAD, "a", "Equal tuning")
    f_key(cv, F_HEAD - 4)
    left = [(n, [tuple(x) for x in ar]) for n, ar in X["a_left"]]
    right = [(n, [tuple(x) for x in ar]) for n, ar in X["a_right"]]
    layouts = [f_rows(g, F_TOP) for g in (left, right)]
    bot = max(lay[2] for lay in layouts) + 2
    value_axes = []
    bran = V["a.auc.bran_combined_state"]["estimate"]
    for x0, groups, (heads, rows, _) in zip(F_BLOCK_X, (left, right), layouts):
        lab_r = x0 + F_LAB_W
        a0 = lab_r + 5
        d0 = a0 + F_COL_W + F_GAP
        lad = cv.axis((a0, a0 + F_COL_W, F_TOP, bot), F_AUC_LIM, (F_TOP, bot), xticks=[0.58, 0.62, 0.66, 0.70],
                      xtext=["0.58", "0.62", "0.66", "0.70"])
        fst = cv.axis((d0, d0 + F_COL_W, F_TOP, bot), F_DIF_LIM, (F_TOP, bot), xticks=[0.0, 0.04, 0.08],
                      xtext=[plain(t, 2) for t in (0.0, 0.04, 0.08)])
        value_axes += [lad.n, fst.n]
        lad.seg([bran, bran], [F_TOP, bot], st.SHARED, width=0.7)
        z = fst.X(0)
        cv.line(z, F_TOP, z, bot, color=E.MUTED, width=0.5, dash="dot")
        for name, yy in heads:
            cv.text(x0, yy, name, size="tick", color=SLATE)
        for name, arms in groups:
            for key, lab, role in arms:
                c = colour(role)
                yy = rows[key]
                row_label(cv, lab_r, yy, key, lab)
                t = kit.tri(V[f"a.auc.{key}"])
                if key == NAT:
                    d = kit.tri(V[f"a.diff.{key}"])
                    for ax_, v, hol in ((lad, t, False), (fst, d, sig(d) == 0)):
                        ax_.inside_x(*v)
                        ax_.seg([v[1], v[2]], [yy, yy], c, width=0.8)
                        ax_.dot(v[0], yy, c, hollow=hol, size=MS_DOT, symbol="square")
                    continue
                if f"a.fixed.{key}" in V:
                    fxv = V[f"a.fixed.{key}"]["estimate"]
                    lad.inside_x(fxv)
                    lad.dot(fxv, yy, c, hollow=True, size=F_FIX_SIZE, symbol=F_FIX_SYMBOL)
                lad.hci(yy, t[0], t[1], t[2], c, size=MS_DOT, width=0.8)
                if key == "bran_combined_state":
                    cv.text(z + 3.5, yy, "Reference", size="tick", color=SLATE)
                    continue
                r = V[f"a.diff.{key}"]
                d = kit.tri(r)
                fst.hci(yy, d[0], d[1], d[2], c, hollow=(sig(d) == 0), size=MS_DOT, width=0.8)
                fst.inside_x(r["bound"])
                pre = r["bound_kind"].startswith("prespecified")
                fst.seg([r["bound"], r["bound"]], [yy - (3.0 if pre else 2.5), yy + (3.0 if pre else 2.5)],
                        INK if pre else st.GREY, width=0.9)
        cv.text(a0 + F_COL_W / 2, bot + 21, "Tuned macro AUROC", xa="center")
        cv.text(d0 + F_COL_W / 2, bot + 21, f"Combined {MINUS} arm", xa="center")
    assert abs(bot + 26 - F_BOT) < 1e-9, (bot, F_BOT)
    return value_axes


def spread_h(ys, half, lo, hi):
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos, hs = [ys[i] for i in order], [half[i] for i in order]
    top, bot = lo - E_HALF + hs[0], hi + E_HALF - hs[-1]
    for _ in range(400):
        moved = False
        for j in range(1, len(pos)):
            need = hs[j - 1] + hs[j]
            if pos[j] - pos[j - 1] < need - 1e-9:
                mm = (pos[j] + pos[j - 1]) / 2
                pos[j - 1], pos[j] = mm - need / 2, mm + need / 2
                moved = True
        if pos[0] < top:
            pos = [p + top - pos[0] for p in pos]
        if pos[-1] > bot:
            pos = [p - (pos[-1] - bot) for p in pos]
        if not moved:
            break
    assert all(b - a >= hs[j] + hs[j + 1] - 1e-6 for j, (a, b) in enumerate(zip(pos, pos[1:]))), pos
    out = [0.0] * len(ys)
    for j, i in enumerate(order):
        out[i] = pos[j]
    return out


def draw_b(cv, V, X):
    kit.heading(cv, 0, GP_HEAD, "b", "Fixed versus tuned readouts")
    x0, x1, y0, y1 = G_BOX
    yt = [0.62, 0.64, 0.66, 0.68, 0.70]
    ax = cv.axis(G_BOX, G_XR, G_YR, xticks=[0, 1, 2], xtext=READOUTS, yticks=yt, ytext=[plain(t, 2) for t in yt],
                 yline=True)
    cv.text(5, (y0 + y1) / 2, "Macro AUROC", xa="center", angle=-90, check=False)
    assert E.text_w("Macro AUROC", st.SIZES["label"]) < y1 - y0
    vals, ends = {}, {}
    curves = [tuple(cc) for cc in X["b_curves"]]
    for key, lab, role, wd, dash, sym, off in curves:
        pts = [V[f"b.{rd}.{key}"]["estimate"] for rd in ("fixed", "shared", "tuned")]
        ax.inside_y(*pts)
        xs = [i + off for i in range(3)]
        c = colour(role)
        ax.seg(xs, pts, c, width=wd, dash=dash)
        for xx, v in zip(xs, pts):
            ax.dot(xx, v, c, size=5.0 if key == "bran_combined_state" else 4.2, symbol=sym)
        vals[key] = pts
        ends[key] = ax.Y(pts[2])
    lead0 = ax.X(2 + max(cc[6] for cc in curves)) + 4
    assert lead0 < G_SWATCH - 10, (lead0, G_SWATCH)
    keys = [cc[0] for cc in curves]
    half = [E_HALF + (LAB_GAP / 2 + E_LAB_CLEAR if k == LAB_EYE else 0.0) for k in keys]
    slots = spread_h([ends[k] for k in keys], half, y0 + 3, y1 - 5)
    hy = y0 - 5
    cv.text(G_NUM_R, hy, f"Tuned {MINUS} fixed", size="tick", color=SLATE, xa="right")
    for r, t in zip(G_BND_R, ("Fixed", "Shared", "Tuned")):
        cv.text(r, hy, t, size="tick", color=SLATE, xa="right")
    cv.text((G_BND_R[0] - 26 + G_BND_R[-1]) / 2, hy - 9.5, "Prespecified bound", size="tick", color=SLATE, xa="center")
    cv.line(G_BND_R[0] - 26, hy - 4.8, G_BND_R[-1], hy - 4.8, color=SLATE, width=0.5)
    sw = cv.axis((G_SWATCH - 5, G_LABX, y0 - 7, y1 + 7), (G_SWATCH - 5, G_LABX), (y0 - 7, y1 + 7), xline=False)
    for (key, lab, role, wd, dash, sym, off), ys in zip(curves, slots):
        c = colour(role)
        cv.line(lead0, ends[key], G_SWATCH - 3.5, ys, color=c, width=0.6, dash=dash)
        sw.dot(G_SWATCH, ys, c, size=5.0 if key == "bran_combined_state" else 4.2, symbol=sym)
        if key == LAB_EYE:
            for ln, dy in zip(LAB_EYE_LINES, (-LAB_GAP / 2, LAB_GAP / 2)):
                cv.text(G_LABX + 3, ys + dy, ln, size="tick")
        else:
            cv.text(G_LABX + 3, ys, lab, size="tick")
        cv.text(G_NUM_R, ys, signed(vals[key][2] - vals[key][0], 3), size="tick", xa="right")
        if key == "bran_combined_state":
            cv.text(G_BND_R[-1], ys, "Reference", size="tick", color=SLATE, xa="right")
            continue
        for r, rd in zip(G_BND_R, ("fixed", "shared", "tuned")):
            cv.text(r, ys, signed(V[f"b.{rd}.{key}"]["bound"], 4), size="tick", xa="right")
    cv.text((x0 + x1) / 2, y1 + 20, "Readout", xa="center")
    return ax


def draw_c(cv, V, X):
    kit.heading(cv, 0, R3, "c", "Where BRAN loses eye information")
    arms = [tuple(r) for r in X["c_arms"]]
    for (ax0, ax1), title in zip(C_AX, ("All 26 conditions", "Four eye conditions")):
        cv.text((ax0 + ax1) / 2, R3 + 11, title, size="tick", color=SLATE, xa="center")
    axes1 = []
    for (ax0, ax1), lim, macro, ticks in zip(C_AX, C_AUC_LIM, ("all_26", "eye_4"),
                                             ([0.60, 0.62, 0.64], [0.68, 0.70, 0.72, 0.74])):
        assert all(lim[0] <= t < lim[1] for t in ticks)
        ax = cv.axis((ax0, ax1, C_ROW1 - 5, C_AX1_BOT), lim, (C_ROW1 - 5, C_AX1_BOT), xticks=ticks,
                     xtext=[f"{t:.2f}" for t in ticks])
        axes1.append(ax)
        for i, (k, lab, role) in enumerate(arms):
            y = C_ROW1 + i * F_PITCH
            t = kit.tri(V[f"c.auc.{macro}.{k}"])
            ax.hci(y, t[0], t[1], t[2], colour(role), size=MS_DOT, width=0.8)
    for i, (k, lab, role) in enumerate(arms):
        cv.text(C_LAB_R, C_ROW1 + i * F_PITCH, lab, size="tick", xa="right")
    cv.text((C_AX[0][0] + C_AX[1][1]) / 2, C_AX1_BOT + 21, "Eye-alone macro AUROC", xa="center")
    cv.text(0, C_HEAD2, "Prespecified decision family", bold=True, color=SLATE)
    tests = [tuple(r) for r in X["c_tests"]]
    margin = X["c_margin"]
    axes2 = []
    for (ax0, ax1), lim, macro in zip(C_AX, C_TEST_LIM, ("all_26", "eye_4")):
        ticks = [0.0, 0.01, 0.02] if lim[1] < 0.03 else [0.0, 0.02, 0.04]
        ticks = [t for t in ticks if lim[0] <= t < lim[1]]
        ax = cv.axis((ax0, ax1, C_ROW2 - 5, C_AX2_BOT), lim, (C_ROW2 - 5, C_AX2_BOT), xticks=ticks,
                     xtext=[plain(t, 2) for t in ticks])
        axes2.append(ax)
        cv.line(ax.X(0), C_ROW2 - 5, ax.X(0), C_AX2_BOT, color=E.MUTED, width=0.5, dash="dot")
        cv.line(ax.X(margin), C_ROW2 - 5, ax.X(margin), C_AX2_BOT, color=INK, width=0.5, dash=E.DASH)
        for i, (name, lab) in enumerate(tests):
            y = C_ROW2 + i * C_TEST_DY
            r = V[f"c.test.{name}.{macro}"]
            t = kit.tri(r)
            established = r["status"] == "loss_established"
            assert established == (t[1] > 0 and t[0] >= margin)
            ax.hci(y, t[0], t[1], t[2], st.EYE if name == "compression" else st.FM, hollow=not established,
                   size=MS_DOT, width=0.8)
    cv.text(C_AX[0][0] + (C_AX[0][1] - C_AX[0][0]) * (margin - C_TEST_LIM[0][0]) / (C_TEST_LIM[0][1] -
            C_TEST_LIM[0][0]), C_HEAD2, "Margin", size="tick", color=SLATE, xa="center")
    for i, (name, lab) in enumerate(tests):
        cv.text(C_LAB_R, C_ROW2 + i * C_TEST_DY, lab, size="tick", xa="right")
    cv.text((C_AX[0][0] + C_AX[1][1]) / 2, C_AX2_BOT + 21, "Difference (AUROC), filled when a loss is established",
            xa="center")
    return axes1, axes2


def draw_d(cv, V, X):
    kit.heading(cv, D_X0, R3, "d", "R8 retinal-route retrain")
    cv.text((D_AX[0] + D_AX[1]) / 2, R3 + 11, f"R8 {MINUS} R7 (AUROC)", size="tick", color=SLATE, xa="center")
    ax = cv.axis((D_AX[0], D_AX[1], D_TOP - 4, D_BOT), D_LIM, (D_TOP - 4, D_BOT), xticks=[-0.015, -0.01, -0.005, 0.0,
                                                                                         0.005, 0.01],
                 xtext=[plain(t, 3) for t in (-0.015, -0.01, -0.005, 0.0, 0.005, 0.01)])
    cv.line(ax.X(0), D_TOP - 4, ax.X(0), D_BOT, color=E.MUTED, width=0.5, dash="dot")
    for lab, y in D_HEADS:
        cv.text(D_X0, y, lab, size="tick", color=SLATE)
    for arm, lab, role in D_ARMS:
        for m, mlab in D_MACROS:
            y = D_ROWS[(arm, m)]
            t = kit.tri(V[f"d.diff.{arm}.{m}"])
            cv.text(D_LAB_R, y, mlab + (" (gate g4a)" if (arm, m) == ("bran_retinal_state", "all_26") else ""),
                    size="tick", xa="right")
            ax.hci(y, t[0], t[1], t[2], colour(role), hollow=(sig(t) == 0), size=MS_DOT, width=0.8)
    r = V["d.native"]
    cv.text(D_LAB_R, D_NATIVE, "Native combined (gate g4b)", size="tick", xa="right")
    ax.inside_x(r["estimate"], r["bound"])
    ax.dot(r["estimate"], D_NATIVE, st.SHARED, size=MS_DOT, symbol="square")
    cv.line(ax.X(r["bound"]), D_NATIVE - 4, ax.X(r["bound"]), D_NATIVE + 4, color=INK, width=0.9)
    cv.text(ax.X(r["bound"]) + 3, D_NATIVE, "Tolerance", size="tick", color=SLATE)
    cv.text((D_AX[0] + D_AX[1]) / 2, D_BOT + 21, "Open markers, interval spans 0", xa="center")
    cv.text(D_X0, D_GATE_TOP - 9, "Acceptance gates", bold=True, color=SLATE)
    for i, (g, lab) in enumerate(D_GATES):
        y = D_GATE_TOP + 9 * i
        rr = V[f"d.gate.{g}"]
        cv.text(D_LAB_R, y, lab, size="tick", xa="right")
        cv.text(D_LAB_R + 8, y, "Pass" if rr["result"] == "pass" else "Fail", size="tick", bold=rr["result"] == "fail")
    cv.text(D_LAB_R + 40, D_GATE_TOP + 9 * 2.5, X["d_decision"], size="tick")
    return ax


def draw(sd: dict):
    assert sd["figure_id"] == NAME and sd["canvas"]["width_px"] == W and sd["canvas"]["height_px"] == H
    V, X = kit.values(sd), sd["draw"]
    cv = E.Canvas(W, H)
    va = draw_a(cv, V, X)
    gb = draw_b(cv, V, X)
    c1, c2 = draw_c(cv, V, X)
    dax = draw_d(cv, V, X)
    cv.flush()
    if cv.check(NAME):
        print("draft", E.preview(cv.fig, NAME, W, H))
        raise SystemExit("text check failed")
    letters = [t for t, px, _, b in cv.texts if b and px == st.SIZES["letter"]]
    assert letters == list("abcd"), letters
    assert sum(t == "PLACEHOLDER" for t, *_ in cv.texts) == 0
    cv.axes_for_check = ({n: ("x", set()) for n in va} | {gb.n: ("y", set())} |
                         {a.n: ("x", set()) for a in c1 + c2} | {dax.n: ("x", set())})
    return cv


def render(payload):
    kit.apply_nm_sizes()
    cv = draw(payload)
    return [(NAME, cv.fig, payload['canvas']['width_px'], payload['canvas']['height_px'])]

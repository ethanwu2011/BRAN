"""Drawing code for Extended Data Fig. 3.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed3.py
(SHA-256 0577cbc6b206755c90111cd514c80238169b8bfc31bf521f4343146e736c2652).

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
NAME = "ed3"
W = kit.NM_W
HASH, PILCROW, SECT = "#", "¶", "§"
SMALL = "small_lane"
INK, SLATE = st.INK, st.SLATE
MINUS = E.MINUS
LAB_EYE = "labrador_clinical_retinal_age"
LAB_EYE_LINES = ("Labrador + clinical +", "BRAN eye features + age")
FAINT = "#D9DDE2"
MS_DOT, BIG_DOT = 4.0, 4.4
signed, plain = E.signed, E.plain


def colour(role: str) -> str:
    """Colour roles recorded in the source data, resolved against the style module's profile at draw time."""
    return kit.colour(role)


def sig(t):
    return 1 if t[1] > 0 else (-1 if t[2] < 0 else 0)


def shade(v, vmax=0.10):
    """Diverging fill from the comparator grey (arm higher) through white to BRAN purple (BRAN combined higher), the
    rule of build_ed3_v5.shade written against the style module's colours."""
    t = max(-1.0, min(1.0, v / vmax))
    hexc = st.SHARED if t >= 0 else st.GREY
    end = tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    a = abs(t)
    return "#%02X%02X%02X" % tuple(round(255 + (end[i] - 255) * a) for i in range(3))


HEAD1 = 13
LAB_R = 84
STRIP = (90.0, 222.0, 21.0, 49.0)
STRIP_LIM = (0.61, 0.72)
STRIP_ROWS = {"clinical": 26.0, "both": 34.6, "retinal": 43.2}
GRID_TOP = 88.0
HDR = ROW = 7.2
FX = {"add_retina": (90.0, 150.0), "add_clinical": (160.0, 220.0)}
F_LIM = (-0.06, 0.27)


def condition_rows(blocks):
    y, heads, rows, spans = GRID_TOP, [], {}, []
    for name, eps in blocks:
        heads.append((name, y + HDR / 2))
        y += HDR
        y0 = y
        for ep in eps:
            rows[ep] = y + ROW / 2
            y += ROW
        spans.append((y0, y))
    return heads, rows, spans


ROW_BOT = GRID_TOP + HDR * 3 + ROW * 26
MEAN_Y = (ROW_BOT + 6.0, ROW_BOT + 13.2)
FAX_BOT = ROW_BOT + 18.0
GX0, CW, GGAP = 262.0, 29.5, 4.5
G_GAP_AFTER = {1, 4, 7, 9}
VMAX = 0.10
HEAD2 = 354
C_TOP = HEAD2 + 40.0
C_PITCH = 8.4
C_GX = [5.0 + 10.5 * i for i in range(6)]
C_LAB_R = 156.0
LAD = (162.0, 244.0, (0.55, 0.70))
FST = (256.0, 350.0, (-0.03, 0.125))
LAB_GAP = 8.2
C_LAB_ABOVE, C_LAB_BELOW = 3.0, 1.5
D_X0 = 362.0
D_LAB_R = 410.0
D_AX = [(416.0, 494.0), (506.0, 584.0), (596.0, 674.0)]
D_LIM = (-0.03, 0.08)
D_ROWS = {"fixed": HEAD2 + 23.0, "logistic": HEAD2 + 32.4, "mlp": HEAD2 + 40.8, "extra_trees": HEAD2 + 49.2}
D_BOT = HEAD2 + 54.0
EYE_TOP, EYE_DY = HEAD2 + 90.0, 8.4
EYE_LIM = (0.598, 0.642)
EYE_AX = (416.0, 520.0)
EYE_DX = (548.0, 674.0)
EYE_DLIM = (-0.04, 0.0)
E_HEAD = HEAD2 + 156.0
E_BOX = (400.0, 516.0, E_HEAD + 10.0, E_HEAD + 84.0)
E_YLIM = (0.07, -0.025)
E_LABX = 528.0
E_HALF, E_LAB_CLEAR = 4.2, 2.0
UP_OFF = [-3.0, -1.5, 0.0, 1.5, 3.0]
LO_OFF = [-2.0, 0.0, 2.0]
H = int(E_BOX[3] + 36)
assert H <= kit.NM_H_MAX, H


def two_line_row(y, pitch, above, below):
    h = above + pitch + LAB_GAP + below
    return y + above + (pitch + LAB_GAP) / 2, h


def row_label(cv, x, y, key, lab):
    if key != LAB_EYE:
        cv.text(x, y, lab, size="tick", xa="right")
        return
    for ln, dy in zip(LAB_EYE_LINES, (-LAB_GAP / 2, LAB_GAP / 2)):
        cv.text(x, y + dy, ln, size="tick", xa="right")


def draw_a(cv, V, X):
    T = lambda rid: kit.tri(V[rid])  # noqa: E731
    kit.heading(cv, 0, HEAD1, "a", "Native route gains")
    blocks = X["blocks"]
    heads, rows, spans = condition_rows(blocks)
    ax = cv.axis(STRIP, STRIP_LIM, (STRIP[2], STRIP[3]), xticks=[0.62, 0.66, 0.70], xtext=["0.62", "0.66", "0.70"])
    for k in ("clinical", "both", "retinal"):
        r = V[f"a.route.{k}"]
        t = kit.tri(r)
        ax.hci(STRIP_ROWS[k], t[0], t[1], t[2], colour(r["role"]), size=BIG_DOT, width=0.9)
        cv.text(LAB_R, STRIP_ROWS[k], r["label"], size="tick", xa="right")
    cv.text((STRIP[0] + STRIP[1]) / 2, STRIP[3] + 20, "Macro AUROC", xa="center")
    axes = [ax]
    for key, title in (("add_retina", "Retina added"), ("add_clinical", "Clinical added")):
        x0, x1 = FX[key]
        fax = cv.axis((x0, x1, GRID_TOP, FAX_BOT), F_LIM, (GRID_TOP, FAX_BOT), xticks=[0.0, 0.1, 0.2],
                      xtext=["0", "0.1", "0.2"])
        axes.append(fax)
        cv.text((x0 + x1) / 2, GRID_TOP - 7, title, xa="center")
        z = fax.X(0)
        for (y0, y1) in spans:
            cv.line(z, y0, z, y1, color=E.MUTED, width=0.5, dash="dot")
        c = st.EYE if key == "add_retina" else st.BLOOD
        for ep, y in rows.items():
            t = T(f"a.{key}.{ep}")
            fax.hci(y, t[0], t[1], t[2], c, hollow=(sig(t) == 0), size=MS_DOT, width=0.8)
    cv.text((FX["add_retina"][0] + FX["add_clinical"][1]) / 2, FAX_BOT + 20, "Paired AUROC difference", xa="center")
    for name, y in heads:
        cv.text(0, y, name, size=SMALL, color=SLATE)
    for ep, y in rows.items():
        cv.text(LAB_R, y, X["short"][ep] + (" " + HASH if ep in X["overlap"] else ""), size=SMALL, xa="right")
    for name, y in zip(("Eye mean", "Other 22 mean"), MEAN_Y):
        cv.text(LAB_R, y, name, size=SMALL, xa="right")
    return axes


def grid_x(n):
    xs, x = [], GX0
    for i in range(n):
        xs.append(x)
        x += CW + (GGAP if i in G_GAP_AFTER else 0)
    return xs


def draw_b(cv, V, X):
    kit.heading(cv, GX0 - 4, HEAD1, "b", "Fixed-readout per-condition comparison")
    cols = X["b_cols"]
    heads, rows, _ = condition_rows(X["blocks"])
    xs = grid_x(len(cols))
    right = xs[-1] + CW
    assert right <= W - 4, right
    gax = cv.axis((GX0, right, GRID_TOP, MEAN_Y[1] + 5), (GX0, right), (GRID_TOP, MEAN_Y[1] + 5), xline=False)
    two = {LAB_EYE: ("Labrador + clinical +", "BRAN eye features")}
    for (key, lab, _), xc in zip(cols, xs):
        cx = xc + CW / 2
        head = "<br>".join(two[key]) if key in two else lab
        assert key not in two or " ".join(two[key]) == lab
        cv.text(cx + 2.5 + (3.5 if key in two else 0), GRID_TOP - 1, head, size=SMALL, xa="right", ya="bottom",
                angle=45, check=False)
        for ep, y in rows.items():
            r = V[f"b.cell.{key}.{ep}"]
            t = kit.tri(r)
            g = sig(t)
            assert r["outcome"] == {1: "win", 0: "tie", -1: "loss"}[g]
            cv.rect(xc + 1, y - ROW / 2 + 0.7, xc + CW - 1, y + ROW / 2 - 0.7, fill=shade(t[0]))
            dark = abs(t[0]) / VMAX > 0.55
            if g == 1:
                gax.dot(cx, y, E.WHITE if dark else INK, size=2.8)
            elif g == -1:
                gax.dot(cx, y, INK, hollow=True, size=5.4)
        for nm, yc in (("eye", MEAN_Y[0]), ("other", MEAN_Y[1])):
            cv.rect(xc + 1, yc - 3.5, xc + CW - 1, yc + 3.5, fill=shade(V[f"b.mean.{nm}.{key}"]["estimate"]))
    pw = E.text_w("PLACEHOLDER", st.SIZES["tick"]) + 10
    cv.placeholder(GX0, MEAN_Y[1] + 7, GX0 + pw, MEAN_Y[1] + 17)
    ky = MEAN_Y[1] + 12
    kax = cv.axis((GX0, right, ky - 5, ky + 5), (GX0, right), (ky - 5, ky + 5), xline=False)
    cb_w, steps = 60.0, 11
    x_loss = right - E.text_w("Loss", st.SIZES["tick"])
    cv.text(x_loss, ky, "Loss", size="tick")
    kax.dot(x_loss - 5, ky, INK, hollow=True, size=5.4)
    x_win = x_loss - 16 - E.text_w("Win", st.SIZES["tick"])
    cv.text(x_win, ky, "Win", size="tick")
    kax.dot(x_win - 4.5, ky, INK, size=2.8)
    cb_x1 = x_win - 18
    cb_x0 = cb_x1 - cb_w
    for i in range(steps):
        v = -VMAX + 2 * VMAX * (i + 0.5) / steps
        cv.rect(cb_x0 + i * cb_w / steps, ky - 6, cb_x0 + (i + 1) * cb_w / steps, ky - 1.5, fill=shade(v))
    for v, lab in ((-VMAX, "≤" + MINUS + "0.1"), (0.0, "0"), (VMAX, "≥+0.1")):
        xx = cb_x0 + (v + VMAX) / (2 * VMAX) * cb_w
        cv.line(xx, ky - 1.5, xx, ky + 0.5, color=INK, width=0.5)
        cv.text(xx, ky + 4.5, lab, size="tick", color=SLATE, xa="center")
    cv.text(cb_x0 - 4, ky - 3.8, "Combined minus arm", size="tick", xa="right")


def draw_c(cv, V, X):
    kit.heading(cv, 0, HEAD2, "c", "Fixed readout, 14 arms")
    y = C_TOP
    yrow, heads = {}, []
    for name, arms in X["c_rows"]:
        heads.append((name, y + C_PITCH / 2))
        y += C_PITCH
        for key, lab, role in arms:
            if key == LAB_EYE:
                yrow[key], h = two_line_row(y, C_PITCH, C_LAB_ABOVE, C_LAB_BELOW)
            else:
                yrow[key], h = y + C_PITCH / 2, C_PITCH
            y += h
    bot = y + 2
    lad = cv.axis((LAD[0], LAD[1], C_TOP, bot), LAD[2], (C_TOP, bot), xticks=[0.55, 0.60, 0.65, 0.70],
                  xtext=["0.55", "0.60", "0.65", "0.70"])
    fst = cv.axis((FST[0], FST[1], C_TOP, bot), FST[2], (C_TOP, bot), xticks=[0.0, 0.04, 0.08, 0.12],
                  xtext=[plain(t, 2) for t in (0.0, 0.04, 0.08, 0.12)])
    gax = cv.axis((0.0, C_GX[-1] + 7, C_TOP, bot), (0.0, C_GX[-1] + 7), (C_TOP, bot), xline=False)
    for i, h in enumerate(X["glyph_head"]):
        cv.text(C_GX[i] - 2.5, C_TOP - 1, h, size="tick", color=SLATE, xa="left", ya="bottom", angle=-45, check=False)
    bl = V["c.auc.blood_age"]["estimate"]
    ref = V["c.auc.bran_combined_state"]["estimate"]
    cv.line(lad.X(bl), C_TOP, lad.X(bl), bot, color=st.GREY, width=0.6, dash=E.DASH)
    cv.line(lad.X(ref), C_TOP, lad.X(ref), bot, color=st.SHARED, width=0.7)
    cv.line(fst.X(0), yrow["bran_combined_state"] + 5, fst.X(0), bot, color=E.MUTED, width=0.5, dash="dot")
    for name, yy in heads:
        cv.text(0, yy, name, size="tick", color=SLATE)
    for name, arms in X["c_rows"]:
        for key, lab, role in arms:
            c = colour(role)
            yy = yrow[key]
            r = V[f"c.auc.{key}"]
            for i, on in enumerate(r["glyph"]):
                if on == "1":
                    gax.dot(C_GX[i], yy, SLATE, size=4.0)
                else:
                    gax.dot(C_GX[i], yy, FAINT, size=1.8)
            row_label(cv, C_LAB_R, yy, key, lab)
            a = kit.tri(r)
            lad.hci(yy, a[0], a[1], a[2], c, size=MS_DOT, width=0.8)
            if key == "bran_combined_state":
                cv.text(fst.X(0) + 3.5, yy, "Reference", size="tick", color=SLATE)
                continue
            dr = V[f"c.diff.{key}"]
            d = kit.tri(dr)
            fst.hci(yy, d[0], d[1], d[2], c, hollow=(sig(d) == 0), size=MS_DOT, width=0.8)
            if dr["bound"] is not None:
                fst.inside_x(dr["bound"])
                cv.line(fst.X(dr["bound"]), yy - 3.0, fst.X(dr["bound"]), yy + 3.0, color=INK, width=0.9)
    cv.text((LAD[0] + LAD[1]) / 2, bot + 21, "Macro AUROC", xa="center")
    cv.text((FST[0] + FST[1]) / 2, bot + 21, f"Combined {MINUS} arm", xa="center")
    cx = lad.X(min(V[f"c.auc.{k}"]["ci_low"] for k in X["concat5"])) - 4
    ys = [yrow[k] for k in X["concat5"]]
    cv.line(cx, min(ys), cx, max(ys), color=INK, width=0.5)
    for yy in ys:
        cv.line(cx, yy, cx + 2.2, yy, color=INK, width=0.5)
    ky = C_TOP - 16
    items = [("dot", "Interval excludes 0"), ("ring", "Interval spans 0"), ("tick", "Bonferroni bound")]
    widths = [7 + E.text_w(t, st.SIZES["tick"]) for _, t in items]
    x = FST[1] - (sum(widths) + 8 * (len(items) - 1))
    kax = cv.axis((x - 5, FST[1], ky - 5, ky + 5), (x - 5, FST[1]), (ky - 5, ky + 5), xline=False)
    for (kind, t), w in zip(items, widths):
        if kind == "tick":
            cv.line(x + 2, ky - 3.0, x + 2, ky + 3.0, color=INK, width=0.9)
        else:
            kax.dot(x + 2, ky, INK, hollow=(kind == "ring"), size=MS_DOT)
        cv.text(x + 7, ky, t, size="tick")
        x += w + 8
    return {"bot": bot + 26, "lad": lad, "fst": fst}


def draw_d(cv, V, X):
    kit.heading(cv, D_X0, HEAD2, "d", "Other readouts")
    axes = []
    for (x0, x1), (fk, mk, title, role) in zip(D_AX, X["d_cols"]):
        ax = cv.axis((x0, x1, D_ROWS["fixed"] - 6, D_BOT), D_LIM, (D_ROWS["fixed"] - 6, D_BOT),
                     xticks=[0.0, 0.05], xtext=["0", "0.05"])
        axes.append(ax)
        cv.text((x0 + x1) / 2, D_ROWS["fixed"] - 12, title, size="tick", xa="center")
        z = ax.X(0)
        cv.line(z, D_ROWS["fixed"] - 5, z, D_BOT, color=E.MUTED, width=0.5, dash="dot")
    cv.text(D_LAB_R, D_ROWS["fixed"], "Fixed C = 1", size="tick", xa="right")
    for ax, (fk, mk, title, role) in zip(axes, X["d_cols"]):
        d = kit.tri(V[f"d.fixed.{fk}"])
        ax.hci(D_ROWS["fixed"], d[0], d[1], d[2], colour(role), hollow=sig(d) == 0, size=MS_DOT, width=0.8, caps=True)
    for fam, lab in X["d_rows"]:
        y = D_ROWS[fam]
        cv.text(D_LAB_R, y, lab + PILCROW, size="tick", xa="right")
        for ax, (fk, mk, title, role) in zip(axes, X["d_cols"]):
            d = kit.tri(V[f"d.ms.{fam}.{fk}"])
            ax.hci(y, d[0], d[1], d[2], colour(role), hollow=sig(d) == 0, size=MS_DOT, width=0.8, caps=True)
    cv.text((D_AX[0][0] + D_AX[-1][1]) / 2, D_BOT + 21, "Combined minus arm (AUROC)", xa="center")
    e_bot = EYE_TOP + EYE_DY * 3 + 5
    ex = cv.axis((EYE_AX[0], EYE_AX[1], EYE_TOP - 5, e_bot), EYE_LIM, (EYE_TOP - 5, e_bot),
                 xticks=[0.60, 0.62, 0.64], xtext=["0.60", "0.62", "0.64"])
    for i, (k, lab, role) in enumerate(X["d_eye_rows"]):
        y = EYE_TOP + i * EYE_DY
        v = V[f"d.nfm.{k}"]["estimate"]
        ex.inside_x(v)
        ex.dot(v, y, colour(role), size=BIG_DOT)
        cv.text(D_LAB_R, y, lab, size="tick", xa="right")
    cv.text((EYE_AX[0] + EYE_AX[1]) / 2, e_bot + 21, "Eye-alone AUROC" + SECT, xa="center")
    dx = cv.axis((EYE_DX[0], EYE_DX[1], EYE_TOP - 5, e_bot), EYE_DLIM, (EYE_TOP - 5, e_bot),
                 xticks=[-0.04, -0.02, 0.0], xtext=[plain(t, 2) for t in (-0.04, -0.02, 0.0)])
    z = dx.X(0.0)
    cv.line(z, EYE_TOP - 5, z, e_bot, color=E.MUTED, width=0.5, dash="dot")
    for i, (k, lab, role) in enumerate(X["d_eye_rows"]):
        if k == "bran_retinal_state":
            continue
        y = EYE_TOP + i * EYE_DY
        r = V[f"d.eye.{k}"]
        t, b = kit.tri(r), r["bound"]
        dx.inside_x(b)
        dx.hci(y, t[0], t[1], t[2], colour(role), hollow=(sig(t) == 0), size=MS_DOT, width=0.8)
        cv.line(dx.X(b), y - 2.6, dx.X(b), y + 2.6, color=INK, width=0.9)
    cv.text((EYE_DX[0] + EYE_DX[1]) / 2, e_bot + 21, "Retinal minus arm (AUROC)", xa="center")
    return {"eye_ax": dx, "bot": e_bot + 25}


def spread_h(ys, half, lo, hi):
    """Label centres near ys in input order, neighbours at least half[i] + half[j] apart, inside [lo, hi]."""
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


def draw_e(cv, V, X):
    kit.heading(cv, D_X0, E_HEAD, "e", "Fixed-readout label budget")
    fracs = X["fracs"]
    x0, x1, y0, y1 = E_BOX
    yt = [-0.02, 0.0, 0.02, 0.04, 0.06]
    ax = cv.axis(E_BOX, (12.0, 108.0), E_YLIM, xticks=fracs, xtext=[str(f) for f in fracs], yticks=yt,
                 ytext=[signed(t, 2) for t in yt], yline=True)
    cv.line(x0, ax.Y(0), x1, ax.Y(0), color=E.MUTED, width=0.5, dash="dot")
    cv.text(D_X0 + 4, (y0 + y1) / 2, f"Combined {MINUS} arm", size="tick", xa="center", angle=-90, check=False)
    assert E.text_w(f"Combined {MINUS} arm", st.SIZES["tick"]) < y1 - y0
    lines = [(k, lab, colour(role), heavy, False, o) for (k, lab, role, heavy), o in zip(X["e_up"], UP_OFF)]
    lines += [(k, lab, colour(role), False, dash, o) for (k, lab, role, dash), o in zip(X["e_lo"], LO_OFF)]
    ends = {}
    unit = (x1 - x0) / (108.0 - 12.0)
    for key, lab, c, heavy, dash, o in lines:
        pts = [kit.tri(V[f"e.{key}.{f}"]) for f in fracs]
        xs = [f + o / unit for f in fracs]
        ax.seg(xs, [p[0] for p in pts], c, width=1.0 if heavy else 0.6, dash=E.DASH if dash else None)
        for xx, p, f in zip(xs, pts, fracs):
            ax.vci(xx, p[0], p[1], p[2], c, hollow=sig(p) == 0, size=BIG_DOT if heavy else 3.4,
                   width=0.8 if heavy else 0.55)
            b = V[f"e.{key}.{f}"]["bound"]
            if b is not None:
                ax.inside_y(b)
                cv.line(ax.X(xx) - 1.8, ax.Y(b), ax.X(xx) + 1.8, ax.Y(b), color=INK, width=0.6)
        ends[key] = (ax.X(xs[-1]), ax.Y(pts[-1][0]))
    keys = [r[0] for r in lines]
    two = ("Labrador + clinical +", "BRAN eye features")
    half = [E_HALF + (LAB_GAP / 2 + E_LAB_CLEAR if k == LAB_EYE else 0.0) for k in keys]
    slots = spread_h([ends[k][1] for k in keys], half, y0 - 2, y1 + 2)
    assert max(E_LABX + E.text_w(t, st.SIZES["tick"]) for t in two) <= W - 1
    for (key, lab, c, heavy, dash, o), ys in zip(lines, slots):
        xa, ya = ends[key]
        cv.line(xa + 2.5, ya, E_LABX - 2, ys, color=E.MUTED, width=0.35)
        if key == LAB_EYE:
            assert lab == " ".join(two)
            for ln, dy in zip(two, (-LAB_GAP / 2, LAB_GAP / 2)):
                cv.text(E_LABX, ys + dy, ln, size="tick")
        else:
            cv.text(E_LABX, ys, lab, size="tick")
    yn = y1 + 18
    cv.text(x0 - 4, yn, "Fold n ≥", size="tick", color=SLATE, xa="right")
    for f in fracs:
        cv.text(ax.X(f), yn, f"{V[f'e.n.{f}']['n_exact']:,}", size="tick", color=SLATE, xa="center")
    cv.text((x0 + x1) / 2, yn + 12, "Readout labels (%)", xa="center")
    return {"bot": yn + 17}


def draw(sd: dict):
    assert sd["figure_id"] == NAME and sd["canvas"]["width_px"] == W and sd["canvas"]["height_px"] == H
    V, X = kit.values(sd), sd["draw"]
    cv = E.Canvas(W, H)
    draw_a(cv, V, X)
    draw_b(cv, V, X)
    c = draw_c(cv, V, X)
    d = draw_d(cv, V, X)
    e = draw_e(cv, V, X)
    cv.flush()
    assert c["bot"] < H and d["bot"] < E_HEAD - 8 and e["bot"] < H, (c["bot"], d["bot"], e["bot"], H)
    if cv.check(NAME):
        print("draft", E.preview(cv.fig, NAME, W, H))
        raise SystemExit("text check failed")
    letters = [t for t, px, _, b in cv.texts if b and px == st.SIZES["letter"]]
    assert letters == list("abcde"), letters
    assert sum(t == "PLACEHOLDER" for t, *_ in cv.texts) == 1   # the mean-row intervals of b
    cv.axes_for_check = {d["eye_ax"].n: ("x", set())}
    return cv


def render(payload):
    kit.apply_nm_sizes()
    cv = draw(payload)
    return [(NAME, cv.fig, payload['canvas']['width_px'], payload['canvas']['height_px'])]

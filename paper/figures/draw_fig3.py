"""Drawing code for Fig. 3.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_fig3.py
(SHA-256 5417f9816de4fd8257eec707c2f0e3d76c8188e105dd2ce73ca349bc6e1e81cb).

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
import plotly.graph_objects as go
kit.apply_nm_sizes()
NAME = "fig3"
INK, SLATE = st.INK, st.SLATE
MINUS = E.MINUS
DAG, DDAG = "†", "‡"
BOTH_SYMBOL = "triangle-up"
NATIVE_SYMBOL = "square"
OT_SYMBOL = "diamond"
DOT, BOTH_DOT, OT_DOT = 4.4, 5.6, 5.2
NAT = "bran_native_heads"
LAB_EYE = "labrador_clinical_retinal_age"
LAB_EYE_LINES = ("Labrador + clinical +", "BRAN eye features + age")
EYE3 = ["retfound_green_age", "visionfm_last4_age", "dinov3_generic_age"]
SINGLE = [("retfound_green_age", "RETFound-Green + age", "Eye"), ("dinov3_generic_age", "DINOv3 + age", "Eye"),
          ("visionfm_last4_age", "VisionFM + age", "Eye"), ("labrador_clinical_age", "Labrador + age", "Blood")]
BOTH = [("retfound_green_clinical_age", "RETFound-Green + clinical + age"),
        ("dinov3_generic_clinical_age", "DINOv3 + clinical + age"),
        ("visionfm_last4_clinical_age", "VisionFM + clinical + age"),
        (LAB_EYE, "Labrador + clinical + BRAN eye features + age")]
SINGLE4 = [k for k, _, _ in SINGLE]
PRIMARY4 = [k for k, _ in BOTH]
PCA3 = [("retfound_green_pca64_clinical_age", "RETFound-Green"), ("dinov3_generic_pca64_clinical_age", "DINOv3"),
        ("visionfm_last4_pca64_clinical_age", "VisionFM")]
PCA3K = [k for k, _ in PCA3]
FAMILY = [("RETFound-Green", ("RETFound-", "Green"), "retfound_green_age", "retfound_green_clinical_age"),
          ("DINOv3", ("DINOv3",), "dinov3_generic_age", "dinov3_generic_clinical_age"),
          ("VisionFM", ("VisionFM",), "visionfm_last4_age", "visionfm_last4_clinical_age"),
          ("Labrador", ("Labrador",), "labrador_clinical_age", LAB_EYE)]
D_LAB = {k: lab for k, lab, _ in SINGLE} | dict(BOTH)
D_SPEC = [("With their own modality", SINGLE4), ("With both modalities", PRIMARY4)]
GROUPS = [("Eye", "eye", ("Eye (4)",)), ("Cardiometabolic and vascular", "cardiometabolic_vascular",
                                         ("Cardiometabolic", "and vascular (12)")),
          ("Other", "other", ("Other (10)",))]
FM_NAME = {"retfound_green_age": "RETFound-Green", "dinov3_generic_age": "DINOv3", "visionfm_last4_age": "VisionFM",
           "labrador_clinical_age": "Labrador", "retfound_green_clinical_age": "RETFound-Green",
           "dinov3_generic_clinical_age": "DINOv3", "visionfm_last4_clinical_age": "VisionFM",
           LAB_EYE: "Labrador"} | {k: f"{lab} PCA-64" for k, lab in PCA3}
OT_ROWS = [("raw_clinical_age", "Raw clinical + age"), ("retfound_green_clinical_age", "RETFound-Green + clinical + age"),
           ("dinov3_generic_clinical_age", "DINOv3 + clinical + age"),
           ("visionfm_last4_clinical_age", "VisionFM + clinical + age"),
           (LAB_EYE, "Labrador + clinical + BRAN eye features + age")]
W = kit.NM_W
ML = 4
RIGHT = W - 4
LAB_R = 146
LAD = (152.0, 224.0, (0.60, 0.74))
DIF = (234.0, 336.0, (-0.04, 0.08))
LINE_GAP = 9.6
SUB_C = 2.5
TICK_H = 2.2
REF_X = DIF[0] + (0.0111 - DIF[2][0]) / (DIF[2][1] - DIF[2][0]) * (DIF[1] - DIF[0]) + 5
HEAD1 = 13
COLHEAD1, DIR1 = 23, 31.5
A_NAT, A_BRAN, A_EYE = 43.0, 53.5, 64.0
A_SUB = 76.5
A_OFF4 = (-6.0, -2.0, 2.0, 6.0)
A_OFF2 = (-2.25, 2.25)


def a_rows():
    y, out = A_SUB + 15 - 18.5, {}
    for k in SINGLE4:
        y += 18.5 if k in EYE3 else 15
        out[k] = y
    return out


A_ROW = a_rows()
A_BOT = A_ROW["labrador_clinical_age"] + 7.5
A_KEY_Y = A_BOT + 28
B_X0 = 346
B_BOX = (376.0, 588.0, 28.0, float(A_BOT))
B_XR = (-0.5, 4.7)
B_YR = (0.705, 0.615)
FAM_X = [0.0, 1.0, 2.0, 3.0]
BRAN_X = {"bran_clinical_state": 3.95, "bran_combined_state": 4.15, "bran_retinal_state": 4.35}
HEAD2 = 199
COLHEAD2, DIR2 = HEAD2 + 10, HEAD2 + 18.5
C_NAT, C_BRAN = HEAD2 + 29, HEAD2 + 39.5
C_SUB = HEAD2 + 51.5
C_PITCH, C_LAB_EXTRA = 12.0, 4.0
OT_PITCH = 11.5


def c_rows():
    y, out = C_SUB, {}
    for k in PRIMARY4:
        y += C_PITCH + (C_LAB_EXTRA if k == LAB_EYE else 0)
        out[k] = y
    sep = y + 10
    out["raw_clinical_age"] = sep + 9
    pca_head = out["raw_clinical_age"] + 12.5
    y = pca_head
    for k in PCA3K:
        y += 11.5
        out[k] = y
    ot_sep = y + 9
    ot_head = ot_sep + 9.5
    ot = {"native": ot_head + 12}
    y = ot["native"]
    for k, _ in OT_ROWS:
        y += OT_PITCH + (C_LAB_EXTRA if k == LAB_EYE else 0)
        ot[k] = y
    return out, sep, pca_head, ot_sep, ot_head, ot


C_ROW, C_SEP, C_PCA_SUB, OT_SEP, OT_HEAD, OT_ROW = c_rows()
C_DEV_BOT = C_ROW[PCA3K[-1]] + 5
C_BOT = OT_ROW[LAB_EYE] + 8
STRIP_TITLE_Y = C_BOT + 28
STRIP = (34.0, 330.0, STRIP_TITLE_Y + 9, STRIP_TITLE_Y + 74)
STRIP_YR = (0.77, 0.58)
STRIP_XR = (-0.55, 2.55)
STRIP_OFF = {"bran_retinal_state": -0.40, "bran_clinical_state": -0.24, "bran_combined_state": -0.08, NAT: 0.08,
             "single": 0.24, "both": 0.40}
STRIP_KEY_X = 360.0
STRIP_NAME_Y = (STRIP[3] + 27, STRIP[3] + 35.4)
H = int(STRIP_NAME_Y[1] + 7)
assert H <= kit.NM_H_MAX, H
D_KEY_Y = HEAD2 + 11
D_STEP, D_HDR, D_FIRST, D_CELL_H = 11, 14, 11, 7.6
D_LAB_EXTRA, D_LAB_AFTER = 5, 3
D_LAB_R = 486
D_BAR = (492.0, float(RIGHT))
D_UNIT = (D_BAR[1] - D_BAR[0]) / 26


def row2_layout():
    y = D_KEY_Y
    heads, ys = [], {}
    for name, keys in D_SPEC:
        y += D_HDR
        heads.append((name, y))
        y += D_FIRST - D_STEP
        for k in keys:
            y += D_STEP + (D_LAB_EXTRA if k == LAB_EYE else 0)
            ys[k] = y
    rule = y + D_STEP / 2 + 1.5 + D_LAB_AFTER
    return heads, ys, rule, [rule + 9, rule + 20, rule + 31]


HEADS2, DY, D_RULE, D_TOT = row2_layout()
D_AX_Y = D_TOT[-1] + 10
assert D_AX_Y + 26 < STRIP[2] - 4, (D_AX_Y, STRIP)


def mid(span):
    return (span[0] + span[1]) / 2


def sig(t):
    """+1 if the interval excludes zero in BRAN's favour, -1 if against, 0 if it spans zero."""
    return 1 if t[1] > 0 else (-1 if t[2] < 0 else 0)


def row_label(cv, x, y, key, lab, size="label"):
    if key != LAB_EYE:
        cv.text(x, y, lab, size=size, xa="right")
        return
    for ln, dy in zip(LAB_EYE_LINES, (-LINE_GAP / 2, LINE_GAP / 2)):
        cv.text(x, y + dy, ln, size=size, xa="right")


def key_row(cv, x_right, y, items, x_left=None):
    """Marker key on one line, right-aligned at x_right or left-aligned at x_left. items are (symbol, hollow, colour,
    size, text)."""
    widths = [8 + E.text_w(t, st.SIZES["tick"]) for *_, t in items]
    span = sum(widths) + 9 * (len(items) - 1)
    x = x_left if x_left is not None else x_right - span
    kax = cv.axis((x - 5, x + span + 1, y - 5, y + 5), (x - 5, x + span + 1), (y - 5, y + 5), xline=False)
    for (sym, hollow, col, size, t), w in zip(items, widths):
        if sym == "tick":
            cv.line(x + 2, y - 3.5, x + 2, y + 3.5, color=col, width=0.9)
        else:
            kax.dot(x + 2, y, col, hollow=hollow, size=size, symbol=sym)
        cv.text(x + 8, y, t, size="tick", color=SLATE)
        x += w + 9
    return x


def ladder_and_forest(cv, top, bot, colhead, dirline, stop=None):
    cv.text(mid(LAD), colhead, "Macro AUROC", xa="center")
    cv.text(mid(DIF), colhead, f"BRAN {MINUS} arm (AUROC)", xa="center")
    lt = [0.62, 0.66, 0.70]
    lad = cv.axis((LAD[0], LAD[1], top, bot), LAD[2], (top, bot), xticks=lt, xtext=[E.plain(t, 2) for t in lt])
    ft = [-0.04, 0.0, 0.04, 0.08]
    dif = cv.axis((DIF[0], DIF[1], top, bot), DIF[2], (top, bot), xticks=ft, xtext=[E.plain(t, 2) for t in ft])
    z = dif.X(0.0)
    cv.text(z - 3, dirline, "Favours arm", size="tick", color=SLATE, xa="right")
    cv.text(z + 3, dirline, "Favours BRAN", size="tick", color=SLATE)
    cv.line(z, dirline + 5, z, stop or bot, color=E.MUTED, width=0.5, dash=E.DASH)
    assert abs(REF_X - (dif.X(0.0111) + 5)) < 1e-9
    return lad, dif, z


def ladder_mark(lad, y, t, color, symbol="circle"):
    lad.inside_x(*t)
    lad.seg([t[1], t[2]], [y, y], color, width=0.9)
    lad.dot(t[0], y, color, size=DOT, symbol=symbol)


def diff_mark(dif, y, t, color, symbol="circle", size=DOT):
    dif.inside_x(*t)
    dif.seg([t[1], t[2]], [y, y], color, width=0.9)
    dif.dot(t[0], y, color, hollow=(sig(t) == 0), size=size, symbol=symbol)


def bran_rows(cv, lad, dif, rows, comb_minus_native):
    for y, inp, lab, t, col, sym in rows:
        if inp:
            cv.text(ML, y, inp, size="tick", color=SLATE)
        cv.text(LAB_R, y, lab, xa="right")
        ladder_mark(lad, y, t, col, sym)
        cv.text(REF_X, y, "Reference", size="tick", color=SLATE)
        if sym == NATIVE_SYMBOL and comb_minus_native is not None:
            assert comb_minus_native[2] < 0
            diff_mark(dif, y, comb_minus_native, st.SHARED)


def draw_a(cv, V):
    T = lambda rid: kit.tri(V[rid])  # noqa: E731
    kit.heading(cv, ML, HEAD1, "a", "BRAN versus foundation models")
    cv.text(ML, COLHEAD1, "Input", size="tick", color=SLATE)
    lad, dif, z = ladder_and_forest(cv, A_NAT - 7, A_BOT, COLHEAD1, DIR1)
    assert ML + E.text_w("Eye + clinical", st.SIZES["tick"]) + 5 < LAB_R - E.text_w("BRAN native heads" + DDAG,
                                                                                   st.SIZES["label"])
    bran = T("a.combined")
    bran_rows(cv, lad, dif, [(A_NAT, "Eye + clinical", "BRAN native heads" + DDAG, T("a.native"), st.SHARED,
                              NATIVE_SYMBOL),
                             (A_BRAN, "Eye + clinical", "BRAN combined" + DAG, bran, st.SHARED, "circle"),
                             (A_EYE, "Eye", "BRAN eye only" + DAG, T("a.eye_route"), st.EYE, "circle")],
              T("a.combined_minus_native"))
    lad.seg([bran[0], bran[0]], [A_BRAN + 4, A_BOT], st.SHARED, width=0.5)
    cv.text(ML, A_SUB, "Foundation models with their own modality", bold=True, color=SLATE)
    for key, lab, tag in SINGLE:
        y = A_ROW[key]
        assert V[f"a.auc.{key}"]["input"] == tag
        cv.text(ML, y, tag, size="tick", color=SLATE)
        cv.text(LAB_R, y, lab, xa="right")
        ladder_mark(lad, y, T(f"a.auc.{key}"), st.FM)
        if key in EYE3:
            o = A_OFF4
            diff_mark(dif, y + o[0], T(f"a.native_minus.{key}"), st.SHARED, NATIVE_SYMBOL)
            diff_mark(dif, y + o[1], T(f"a.combined_minus.{key}"), st.SHARED)
            diff_mark(dif, y + o[2], T(f"a.eye_dev.{key}"), st.EYE)
            diff_mark(dif, y + o[3], T(f"a.eye_official.{key}"), st.EYE, OT_SYMBOL, OT_DOT)
        else:
            o = A_OFF2
            diff_mark(dif, y + o[0], T(f"a.native_minus.{key}"), st.SHARED, NATIVE_SYMBOL)
            diff_mark(dif, y + o[1], T(f"a.combined_minus.{key}"), st.SHARED)
    # Marker key for a and c
    key_row(cv, None, A_KEY_Y, [(NATIVE_SYMBOL, False, st.SHARED, DOT, "Native heads" + DDAG),
                                ("circle", False, st.SHARED, DOT, "BRAN combined" + DAG),
                                ("circle", False, st.EYE, DOT, "BRAN eye only" + DAG),
                                (OT_SYMBOL, False, st.EYE, OT_DOT, "Eye only, official test")], x_left=ML + 4)
    return {"lad": lad, "dif": dif}


def draw_b(cv, V):
    E_ = lambda rid: V[rid]["estimate"]  # noqa: E731
    kit.heading(cv, B_X0, HEAD1, "b", "Bridging the gap")
    key_row(cv, RIGHT, HEAD1 - 4, [("circle", False, SLATE, DOT, "One modality"),
                                   (BOTH_SYMBOL, False, SLATE, BOTH_DOT, "Both modalities")])
    x0, x1, y0, y1 = B_BOX
    yt = [0.62, 0.64, 0.66, 0.68, 0.70]
    ax = cv.axis(B_BOX, B_XR, B_YR, yticks=yt, ytext=[E.plain(t, 2) for t in yt], yline=True)
    cv.text(B_X0 + 4, (y0 + y1) / 2, "Tuned macro AUROC", xa="center", angle=-90, check=False)
    assert E.text_w("Tuned macro AUROC", st.SIZES["label"]) < y1 - y0
    bran, raw = E_("b.combined"), E_("b.raw")
    ends = (B_XR[0] + 0.05, B_XR[1] - 0.05)
    ax.inside_y(bran, raw)
    ax.seg(list(ends), [bran, bran], st.SHARED, width=0.7)
    ax.seg(list(ends), [raw, raw], st.GREY, width=0.7, dash=E.DASH)
    for v, col, dash, lab, dy in ((bran, st.SHARED, None, "BRAN combined", -5.5),
                                  (raw, st.GREY, E.DASH, "Raw clinical + age", 5.5)):
        yl = ax.Y(v) + dy
        cv.line(x1 + 3, yl, x1 + 12, yl, color=col, width=0.7, dash=dash)
        cv.text(x1 + 15, yl, lab, size="tick")
    for (name, lines, ks, kb), x in zip(FAMILY, FAM_X):
        s, e = E_(f"b.single.{ks}"), E_(f"b.both.{kb}")
        ax.inside_y(s, e)
        ax.seg([x, x], [s, e], st.FM, width=0.9)
        ax.dot(x, s, st.FM, size=DOT)
        ax.dot(x, e, st.FM, size=BOTH_DOT, symbol=BOTH_SYMBOL)
        for i, ln in enumerate(lines):
            cv.text(ax.X(x), y1 + 8 + 8.4 * i, ln, size="tick", xa="center")
    v = {"bran_retinal_state": E_("b.route.bran_retinal_state"), "bran_clinical_state": E_("b.route.bran_clinical_state"),
         "bran_combined_state": bran}
    xc = BRAN_X["bran_combined_state"]
    for k, col in (("bran_clinical_state", st.BLOOD), ("bran_retinal_state", st.EYE)):
        ax.inside_y(v[k])
        ax.seg([BRAN_X[k], xc], [v[k], v["bran_combined_state"]], col, width=0.9)
        ax.dot(BRAN_X[k], v[k], col, size=DOT)
    ax.dot(xc, bran, st.SHARED, size=BOTH_DOT, symbol=BOTH_SYMBOL)
    cv.text(ax.X(BRAN_X["bran_clinical_state"]), ax.Y(v["bran_clinical_state"]) - 7.5, "Clinical only", size="tick",
            xa="center")
    cv.text(ax.X(BRAN_X["bran_retinal_state"]) - 4.5, ax.Y(v["bran_retinal_state"]), "Eye only", size="tick", xa="right")
    cv.text(ax.X(xc), y1 + 8, "BRAN", size="tick", xa="center")
    cv.text(ax.X(FAM_X[3]), y1 + 16.4, "+ raw clinical", size="tick", color=SLATE, xa="center")
    cv.text((x0 + x1) / 2, y1 + 28, "Each line adds the other data source", xa="center")
    return {"ax": ax}


def pair_row(dif, y, nat_t, nat_b, comb_t, comb_b):
    for yy, t, b, sym in ((y - SUB_C, nat_t, nat_b, NATIVE_SYMBOL), (y + SUB_C, comb_t, comb_b, "circle")):
        diff_mark(dif, yy, t, st.SHARED, sym)
        if b is not None:
            dif.inside_x(b)
            dif.seg([b, b], [yy - TICK_H, yy + TICK_H], INK, width=0.9)


def draw_c(cv, V, X, a_axes):
    T = lambda rid: kit.tri(V[rid])  # noqa: E731
    kit.heading(cv, ML, HEAD2, "c", "Gap with both modalities")
    key_row(cv, DIF[1], HEAD2 - 4, [("circle", True, SLATE, DOT, "Interval spans 0"),
                                    ("tick", False, INK, 0, "Adjusted bound")])
    lad, dif, z = ladder_and_forest(cv, C_NAT - 7, C_BOT, COLHEAD2, DIR2, stop=OT_SEP - 1)
    cv.line(z, OT_HEAD + 6, z, C_BOT, color=E.MUTED, width=0.5, dash=E.DASH)   # zero line of the official block
    bran = T("c.combined")
    bran_rows(cv, lad, dif, [(C_NAT, None, "BRAN native heads" + DDAG, T("c.native"), st.SHARED, NATIVE_SYMBOL),
                             (C_BRAN, None, "BRAN combined" + DAG, bran, st.SHARED, "circle")],
              T("c.combined_minus_native"))
    lad.seg([bran[0], bran[0]], [C_BRAN + 4, C_DEV_BOT], st.SHARED, width=0.5)
    cv.text(ML, C_SUB, "Foundation models with both modalities", bold=True, color=SLATE)
    for key, lab in BOTH + [("raw_clinical_age", "Raw clinical + age")]:
        y = C_ROW[key]
        row_label(cv, LAB_R, y, key, lab)
        ladder_mark(lad, y, T(f"c.auc.{key}"), st.GREY if key == "raw_clinical_age" else st.FM)
        nr, cr = V[f"c.native_minus.{key}"], V[f"c.combined_minus.{key}"]
        pair_row(dif, y, kit.tri(nr), nr["bound"], kit.tri(cr), cr["bound"])
    cv.line(ML, C_SEP, DIF[1], C_SEP, color=E.MUTED, width=0.5, dash=E.DASH)
    cv.text(ML, C_PCA_SUB, "PCA-64 + clinical + age", bold=True, color=SLATE)
    for key, lab in PCA3:
        y = C_ROW[key]
        cv.text(LAB_R, y, lab, xa="right")
        ladder_mark(lad, y, T(f"c.auc.{key}"), st.FM)
        pair_row(dif, y, T(f"c.native_minus.{key}"), None, T(f"c.combined_minus.{key}"), None)
    # Official-test block
    cv.line(ML, OT_SEP, DIF[1], OT_SEP, color=INK, width=0.5)
    n_ot = X["official_people_floor20"]
    cv.text(ML, OT_HEAD, "Official test, a previously accessed paired check", bold=True, color=SLATE)
    cv.text(DIF[1], OT_HEAD, f"n ≥ {n_ot}, 24 conditions", size="tick", color=SLATE, xa="right")
    cv.text(LAB_R, OT_ROW["native"], "BRAN native heads" + DDAG, xa="right")
    ladder_mark(lad, OT_ROW["native"], T("c.ot.native"), st.SHARED, NATIVE_SYMBOL)
    cv.text(REF_X, OT_ROW["native"], "Reference", size="tick", color=SLATE)
    for key, lab in OT_ROWS:
        y = OT_ROW[key]
        row_label(cv, LAB_R, y, key, lab)
        ladder_mark(lad, y, T(f"c.ot.auc.{key}"), st.GREY if key == "raw_clinical_age" else st.FM)
        r = V[f"c.ot.native_minus.{key}"]
        diff_mark(dif, y, kit.tri(r), st.SHARED, NATIVE_SYMBOL)
        dif.inside_x(r["bound"])
        dif.seg([r["bound"], r["bound"]], [y - TICK_H, y + TICK_H], INK, width=0.9)
    assert (a_axes["lad"].X(0.66), a_axes["dif"].X(0.04)) == (lad.X(0.66), dif.X(0.04))
    strip_ax = draw_strip(cv, V, X)
    return {"lad": lad, "dif": dif, "strip": strip_ax}


def draw_strip(cv, V, X):
    x0, x1, y0, y1 = STRIP
    cv.text(ML, STRIP_TITLE_Y, "Condition groups", bold=True, color=SLATE)
    yt = [0.60, 0.65, 0.70, 0.75]
    ax = cv.axis(STRIP, STRIP_XR, STRIP_YR, yticks=yt, ytext=[E.plain(t, 2) for t in yt], yline=True)
    cv.text(ML + 4, (y0 + y1) / 2, "Macro AUROC", xa="center", angle=-90, check=False)
    assert E.text_w("Macro AUROC", st.SIZES["label"]) < y1 - y0
    single, best = X["strip"]["single"], X["strip"]["best"]
    for gi, (name, g, labs) in enumerate(GROUPS):
        for role, col, sym, size in (("bran_retinal_state", st.EYE, "circle", DOT),
                                     ("bran_clinical_state", st.BLOOD, "circle", DOT),
                                     ("bran_combined_state", st.SHARED, "circle", DOT),
                                     (NAT, st.SHARED, NATIVE_SYMBOL, DOT),
                                     ("single", st.FM, "circle", DOT), ("both", st.FM, BOTH_SYMBOL, BOTH_DOT)):
            r = V[f"c.group.{g}.{role}"]
            assert r["arm"] == {"single": single[name], "both": best[name]}.get(role, role)
            t = kit.tri(r)
            ax.inside_y(t[1], t[2])
            ax.seg([gi + STRIP_OFF[role], gi + STRIP_OFF[role]], [t[1], t[2]], col, width=0.8)
            ax.dot(gi + STRIP_OFF[role], t[0], col, size=size, symbol=sym)
        for i, ln in enumerate(labs):
            cv.text(ax.X(gi), y1 + 8 + 8.4 * i, ln, size="tick", xa="center")
        for k, yy in zip((single[name], best[name]), STRIP_NAME_Y):
            cv.text(ax.X(gi), yy, FM_NAME[k], size="tick", color=SLATE, xa="center")
    nax = cv.axis((ML, ML + 12, STRIP_NAME_Y[0] - 5, STRIP_NAME_Y[1] + 5), (ML, ML + 12),
                  (STRIP_NAME_Y[0] - 5, STRIP_NAME_Y[1] + 5), xline=False)
    nax.dot(ML + 6, STRIP_NAME_Y[0], st.FM, size=DOT)
    nax.dot(ML + 6, STRIP_NAME_Y[1], st.FM, size=BOTH_DOT, symbol=BOTH_SYMBOL)
    kx = STRIP_KEY_X
    kax = cv.axis((kx - 4, kx + 8, y0, y1), (kx - 4, kx + 8), (y0, y1), xline=False)
    items = [(st.EYE, "circle", DOT, "BRAN eye only"), (st.BLOOD, "circle", DOT, "BRAN clinical only"),
             (st.SHARED, "circle", DOT, "BRAN combined"), (st.SHARED, NATIVE_SYMBOL, DOT, "BRAN native heads"),
             (st.FM, "circle", DOT, "Best FM, own modality"), (st.FM, BOTH_SYMBOL, BOTH_DOT, "Best FM, both modalities")]
    for i, (col, sym, size, t) in enumerate(items):
        yy = y0 + 5 + 9.5 * i
        kax.dot(kx + 2, yy, col, size=size, symbol=sym)
        cv.text(kx + 8, yy, t, size="tick", color=SLATE)
    cv.text(kx, y0 + 5 + 9.5 * len(items) + 2, "Bars, marginal 95% intervals", size="tick", color=SLATE)
    return ax


LOSS = st.FM


def draw_d(cv, V):
    kit.heading(cv, B_X0, HEAD2, "d", "Per-condition comparison")
    x = D_BAR[0]
    for name, fill in (("Win", st.SHARED), ("Tie", st.LIGHT), ("Loss", LOSS)):
        cv.rect(x, D_KEY_Y - 3, x + 6, D_KEY_Y + 3, fill=fill)
        cv.text(x + 9, D_KEY_Y, name, color=SLATE, size="tick")
        x += 9 + E.text_w(name, st.SIZES["tick"]) + 10
    cv.text(RIGHT, HEAD2 - 4, "Per-arm tuned readout", size="tick", color=SLATE, xa="right")
    for name, y in HEADS2:
        cv.text(B_X0, y, name, bold=True, color=SLATE)
    for _, keys in D_SPEC:
        for k in keys:
            r = V[f"d.tally.{k}"]
            w, t = r["wins"], r["ties"]
            assert w + t + r["losses"] == 26
            y = DY[k]
            row_label(cv, D_LAB_R, y, k, D_LAB[k])
            for i in range(26):
                fill = st.SHARED if i < w else (st.LIGHT if i < w + t else LOSS)
                xa = D_BAR[0] + i * D_UNIT + 0.08 * D_UNIT
                cv.rect(xa, y - D_CELL_H / 2, xa + 0.84 * D_UNIT, y + D_CELL_H / 2, fill=fill)
    cv.line(D_BAR[0], D_RULE, D_BAR[1], D_RULE, color=INK, width=0.5)
    for rid, y, col in (("d.total.own", D_TOT[0], INK), ("d.total.both", D_TOT[1], INK),
                        ("d.total.fixed", D_TOT[2], SLATE)):
        r = V[rid]
        cv.text(D_LAB_R, y, r["label"], color=col, xa="right")
        cv.text(D_BAR[0], y, f"{r['wins']} wins, {r['losses']} {'loss' if r['losses'] == 1 else 'losses'} of "
                             f"{r['contrasts']}", color=col)
    dt = [0, 5, 10, 15, 20, 25]
    dax = cv.axis((D_BAR[0], D_BAR[1], D_KEY_Y + 7, D_AX_Y), (0.0, 26.0), (D_KEY_Y + 7, D_AX_Y), xticks=dt,
                  xtext=[str(t) for t in dt])
    cv.fig.add_trace(go.Scatter(x=[0.0], y=[D_AX_Y], xaxis=dax.xref, yaxis=dax.yref, mode="markers",
                                marker=dict(size=1, opacity=0), hoverinfo="skip", cliponaxis=False))
    cv.text(mid(D_BAR), D_AX_Y + 19, "Conditions (n)", xa="center")


def draw(sd: dict):
    """The whole canvas from the JSON form of the source-data record."""
    assert sd["figure_id"] == NAME and sd["canvas"]["width_px"] == W and sd["canvas"]["height_px"] == H
    V = kit.values(sd)
    X = sd["draw"]
    cv = E.Canvas(W, H)
    a_axes = draw_a(cv, V)
    b_axes = draw_b(cv, V)
    c_axes = draw_c(cv, V, X, a_axes)
    draw_d(cv, V)
    cv.flush()
    if cv.check(NAME):
        print("draft", E.preview(cv.fig, NAME, W, H))
        raise SystemExit("text check failed")
    letters = [t for t, px, _, bo in cv.texts if bo and px == st.SIZES["letter"]]
    assert letters == list("abcd"), letters
    assert sum(t == "PLACEHOLDER" for t, *_ in cv.texts) == 0
    cv.axes_for_check = {a_axes["lad"].n: ("x", set()), a_axes["dif"].n: ("x", set()), b_axes["ax"].n: ("y", set()),
                         c_axes["lad"].n: ("x", set()), c_axes["dif"].n: ("x", set()), c_axes["strip"].n: ("y", set())}
    return cv


def render(payload):
    kit.apply_nm_sizes()
    cv = draw(payload)
    return [(NAME, cv.fig, payload['canvas']['width_px'], payload['canvas']['height_px'])]

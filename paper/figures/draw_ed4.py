"""Drawing code for Extended Data Fig. 5.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed4.py
(SHA-256 40dfc7211ef3dd0d4067dc7bffd3f77c6475d03984a9a07daf61b680e3b52732).

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
import fig4_lane_kit as K
import ed_canvas as E
FIG_ID = "ed4"
INK, SLATE, GREY, BLOOD, SHARED, EYE = st.INK, st.SLATE, st.GREY, st.BLOOD, st.SHARED, st.EYE
HEX = {"6B7280": GREY, "C05A47": BLOOD, "514B7C": SHARED, "17212B": INK, "4A5563": SLATE, "177E89": EYE}
W = st.FULL_W
DOT = 5.0
FIELDS = ["hemoglobin", "hct", "rbc", "mcv", "mch", "mchc", "rdw", "wbc", "plt"]
LABEL = {"hemoglobin": "Haemoglobin", "hct": "Haematocrit", "rbc": "Red cells", "mcv": "MCV", "mch": "MCH",
         "mchc": "MCHC", "rdw": "RDW", "wbc": "White cells", "plt": "Platelets"}
V5_AUC_ARMS = [["median", "Training median", "6B7280", True], ["ridge", "Ridge", "6B7280", False],
               ["native_no_retina", "BRAN", "C05A47", False], ["native_with_retina", "BRAN with retina", "514B7C", False],
               ["observed_cbc", "Observed CBC", "17212B", False], ["noncbc_only", "No CBC input", "4A5563", True]]
V5_F_PAIRS = [["bran_i_cbc_hidden_with_retina", "BRAN with retina", "514B7C", "ridge_noncbc_pca64_age",
               "Ridge, clinical, retina, age", "i_minus_ridge_noncbc_pca64_age", "With retina minus Ridge"],
              ["bran_ii_cbc_hidden_no_retina", "BRAN", "C05A47", "ridge_noncbc_age", "Ridge, clinical and age",
               "ii_minus_ridge_noncbc_age", "BRAN minus Ridge"],
              ["bran_iii_retina_and_age_only", "Retina plus age", "177E89", "ridge_pca64_age", "Ridge, retina and age",
               "iii_minus_ridge_pca64_age", "Retina plus age minus Ridge"]]
RENAME = {"BRAN": "BRAN without retina", "Retina plus age": "Retina and age",
          "BRAN minus Ridge": "Without retina minus Ridge", "Retina plus age minus Ridge": "Retina and age minus Ridge"}
AUC_ARMS = [[k, RENAME.get(t, t), c, h] for k, t, c, h in V5_AUC_ARMS]
F_PAIRS = [[a, RENAME.get(b, b), c, d, e, f, RENAME.get(g, g)] for a, b, c, d, e, f, g in V5_F_PAIRS]
BE_LIM = (-9.0, 5.0)
C_LIM, CD_LIM = (1.8, 2.8), (-0.5, 0.35)
D_LIM, DD_LIM = (0.660, 0.700), (-0.004, 0.010)
F_LIM, FD_LIM = (0.60, 0.85), (-0.05, 0.05)
LOGISTIC, NORMAL = "logistic_recalibration", "gaussian_residual"
DY = 26.0
COLS = [(0.0, 206.0), (218.0, 424.0), (436.0, W - 8.0)]
LABW = [104.0, 52.0, 110.0]
HEAD = {1: 40.0 - DY, 2: 252.0 - DY, 3: 464.0 - DY}
KEY = {1: 52.0 - DY, 2: 264.0 - DY}
Y0 = {1: 62.0 - DY, 2: 274.0 - DY, 3: 480.0 - DY}
RH, GAP = 15.0, 5.0
B = {r: Y0[r] + 9 * RH + GAP for r in (1, 2)}


def ax_x(c):
    return COLS[c][0] + LABW[c] + 6


def row_mid(i, r):
    return Y0[r] + i * RH + (GAP if i >= 6 else 0.0) + RH / 2


def labels(cv, c, rows):
    for y, t in rows:
        cv.text(COLS[c][0] + LABW[c], y, t, size="tick", xa="right")


def key(cv, c, r, items):
    x = COLS[c][0] + 2
    y = KEY[r]
    kax = cv.axis((COLS[c][0], COLS[c][1], y - 5, y + 5), (COLS[c][0], COLS[c][1]), (y - 5, y + 5), xline=False)
    for kind, color, t in items:
        kax.dot(x + 2.5, y, color, hollow=(kind == "ring"), size=DOT)
        x += 8
        x += cv.text(x, y, t, size="tick") + 9
    assert x - 9 <= COLS[c][1] + 0.5, (x, items)


def zero(cv, ax, y0, y1, color=INK):
    cv.line(ax.X(0), y0, ax.X(0), y1, color=color, width=0.6)


def be_ticks():
    t = [-8, -6, -4, -2, 0, 2, 4]
    return t, [E.signed(v, 0) for v in t]


def panel_a(cv, V):
    c, r = 0, 1
    cv.heading(COLS[c][0], HEAD[r], "a", "Whole-panel completion")
    key(cv, c, r, [("dot", GREY, "Ridge"), ("dot", BLOOD, "BRAN without retina"), ("dot", SHARED, "BRAN with retina")])
    ax = cv.axis((ax_x(c), COLS[c][1], Y0[r], B[r]), (0.80, 1.01), (Y0[r], B[r]),
                 xticks=[0.80, 0.85, 0.90, 0.95, 1.00], xtext=["0.80", "0.85", "0.90", "0.95", "1.00"])
    cv.line(ax.X(1.0), Y0[r], ax.X(1.0), B[r], color=GREY, width=0.75, dash=E.DASH)
    for i, f in enumerate(FIELDS):
        y = row_mid(i, r)
        for k, dy, color in (("ridge", -4.0, GREY), ("native_no_retina", 0.0, BLOOD), ("native_with_retina", 4.0, SHARED)):
            v = V["a"][f][k]
            ax.inside_x(v)
            ax.dot(v, y + dy, color, size=DOT)
    labels(cv, c, [(row_mid(i, r), LABEL[f]) for i, f in enumerate(FIELDS)])
    cv.text((ax.x0 + ax.x1) / 2, B[r] + 20, "MAE relative to training median", size="label", xa="center")


def be_axis(cv, r):
    t, tt = be_ticks()
    ax = cv.axis((ax_x(1), COLS[1][1], Y0[r], B[r]), BE_LIM, (Y0[r], B[r]), xticks=t, xtext=tt)
    zero(cv, ax, Y0[r], B[r])
    return ax


def panel_b(cv, V):
    c, r = 1, 1
    cv.heading(COLS[c][0], HEAD[r], "b", "Per-value Ridge contrast")
    ax = be_axis(cv, r)
    for i, f in enumerate(FIELDS):
        ax.hci(row_mid(i, r), *V["b"][f], BLOOD, size=DOT)
    labels(cv, c, [(row_mid(i, r), LABEL[f]) for i, f in enumerate(FIELDS)])
    cv.text((ax.x0 + ax.x1) / 2, B[r] + 20, "BRAN minus Ridge MAE (% of median MAE)", size="label", xa="center")


def panel_c(cv, V):
    c, r = 2, 1
    cv.heading(COLS[c][0], HEAD[r], "c", "Haemoglobin below 12 g/dL")
    ua = Y0[r] + 58
    ax = cv.axis((ax_x(c), COLS[c][1], Y0[r], ua), C_LIM, (Y0[r], ua), xticks=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                 xtext=["1.8", "2.0", "2.2", "2.4", "2.6", "2.8"])
    rows = [("median", "Training median", GREY, True), ("ridge", "Ridge", GREY, False),
            ("native_no_retina", "BRAN without retina", BLOOD, False),
            ("native_with_retina", "BRAN with retina", SHARED, False)]
    ys = []
    for i, (k, t, color, hol) in enumerate(rows):
        y = Y0[r] + 8 + i * 14
        ax.hci(y, *V["c"]["lowarm"][k], color, hollow=hol, size=DOT)
        ys.append((y, t))
    cv.text((ax.x0 + ax.x1) / 2, ua + 20, "MAE (g/dL)", size="label", xa="center")
    bx = cv.axis((ax_x(c), COLS[c][1], B[r] - 44, B[r]), CD_LIM, (B[r] - 44, B[r]), xticks=[-0.4, -0.2, 0.0, 0.2],
                 xtext=[E.plain(v, 1) for v in (-0.4, -0.2, 0.0, 0.2)])
    zero(cv, bx, B[r] - 44, B[r])
    for i, (k, t) in enumerate((("low_ridge", "Without retina minus Ridge"),
                                ("low_median", "Without retina minus median"))):
        y = B[r] - 32 + i * 15
        bx.hci(y, *V["c"][k], BLOOD, size=DOT)
        ys.append((y, t))
    labels(cv, c, ys)
    cv.text((bx.x0 + bx.x1) / 2, B[r] + 20, "Paired MAE difference (g/dL)", size="label", xa="center")


def panel_d(cv, V):
    c, r = 0, 2
    cv.heading(COLS[c][0], HEAD[r], "d", "Downstream readout")
    ua = Y0[r] + 68
    ax = cv.axis((ax_x(c), COLS[c][1], Y0[r], ua), D_LIM, (Y0[r], ua), xticks=[0.66, 0.67, 0.68, 0.69, 0.70],
                 xtext=["0.66", "0.67", "0.68", "0.69", "0.70"])
    ys = []
    for i, (k, t, color, hol) in enumerate(AUC_ARMS):
        y = Y0[r] + 6 + i * 11
        ax.hci(y, *V["d"]["auc"][k], HEX[color], hollow=hol, size=DOT)
        ys.append((y, t))
    cv.text((ax.x0 + ax.x1) / 2, ua + 20, "Macro AUROC", size="label", xa="center")
    bx = cv.axis((ax_x(c), COLS[c][1], B[r] - 44, B[r]), DD_LIM, (B[r] - 44, B[r]),
                 xticks=[-0.004, 0.0, 0.004, 0.008], xtext=[E.plain(v, 3) for v in (-0.004, 0.0, 0.004, 0.008)])
    zero(cv, bx, B[r] - 44, B[r])
    for i, (k, t) in enumerate((("native_minus_ridge", "Without retina minus Ridge"),
                                ("native_minus_median", "Without retina minus median"))):
        y = B[r] - 32 + i * 15
        bx.hci(y, *V["d"]["actr"][k], BLOOD, size=DOT)
        ys.append((y, t))
    labels(cv, c, ys)
    cv.text((bx.x0 + bx.x1) / 2, B[r] + 20, "Paired AUROC difference", size="label", xa="center")


def panel_e(cv, V):
    c, r = 1, 2
    cv.heading(COLS[c][0], HEAD[r], "e", "Ridge on retinal features")
    key(cv, c, r, [("dot", EYE, "Retina and age only"), ("dot", SHARED, "Panel hidden, with retina")])
    ax = be_axis(cv, r)
    for i, f in enumerate(FIELDS):
        y = row_mid(i, r)
        for p, dy, color in ((V["e"][f]["eye"], -3.0, EYE), (V["e"][f]["wr"], 3.0, SHARED)):
            ax.hci(y + dy, *p, color, size=4.4)
    labels(cv, c, [(row_mid(i, r), LABEL[f]) for i, f in enumerate(FIELDS)])
    cv.text((ax.x0 + ax.x1) / 2, B[r] + 20, "BRAN minus Ridge MAE (% of median MAE)", size="label", xa="center")


def panel_f(cv, V):
    c, r = 2, 2
    cv.heading(COLS[c][0], HEAD[r], "f", "Low haemoglobin detection")
    ua = Y0[r] + 67
    ax = cv.axis((ax_x(c), COLS[c][1], Y0[r], ua), F_LIM, (Y0[r], ua), xticks=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
                 xtext=["0.60", "0.65", "0.70", "0.75", "0.80", "0.85"])
    ys = []
    for p, (k_b, lab_b, color, k_r, lab_r, _, _) in enumerate(F_PAIRS):
        for j, (k, t, cc) in enumerate(((k_b, lab_b, HEX[color]), (k_r, lab_r, GREY))):
            y = Y0[r] + 6 + p * 22 + j * 10
            ax.hci(y, *V["f"]["auc"][k], cc, size=DOT)
            ys.append((y, t))
    cv.text((ax.x0 + ax.x1) / 2, ua + 20, "AUROC", size="label", xa="center")
    bx = cv.axis((ax_x(c), COLS[c][1], B[r] - 46, B[r]), FD_LIM, (B[r] - 46, B[r]),
                 xticks=[-0.04, 0.0, 0.04], xtext=[E.plain(v, 2) for v in (-0.04, 0.0, 0.04)])
    zero(cv, bx, B[r] - 46, B[r])
    for i, (_, _, color, _, _, k_c, lab_c) in enumerate(F_PAIRS):
        y = B[r] - 36 + i * 12
        bx.hci(y, *V["f"]["dcon"][k_c], HEX[color], size=DOT)
        ys.append((y, lab_c))
    labels(cv, c, ys)
    cv.text((bx.x0 + bx.x1) / 2, B[r] + 20, "Paired AUROC difference", size="label", xa="center")


G_ARMS = [("observed_cbc", "Observed CBC"), ("native_no_retina", "BRAN without retina"),
          ("median", "Training median"), ("noncbc_only", "No CBC input")]
G_DY = [7.0, 19.0, 31.0, 45.0]
G_DIFF = (228.0, 316.0)
G_DLIM = (-0.002, 0.009)


def panel_g(cv, V):
    r = 3
    cv.heading(COLS[0][0], HEAD[r], "g", "Downstream readout, four arms")
    arms = {k: (HEX[color], hol) for k, _, color, hol in AUC_ARMS}
    ua = Y0[r] + 52
    ax = cv.axis((ax_x(0), COLS[0][1], Y0[r], ua), D_LIM, (Y0[r], ua), xticks=[0.66, 0.67, 0.68, 0.69, 0.70],
                 xtext=["0.66", "0.67", "0.68", "0.69", "0.70"])
    ys = []
    for (k, t), dy in zip(G_ARMS, G_DY):
        y = Y0[r] + dy
        color, hol = arms[k]
        ax.hci(y, *V["g"]["auc"][k], color, hollow=hol, size=DOT)
        ys.append((y, t))
    labels(cv, 0, ys)
    cv.text((ax.x0 + ax.x1) / 2, ua + 20, "Macro AUROC", size="label", xa="center")
    t = [0.0, 0.004, 0.008]
    bx = cv.axis((G_DIFF[0], G_DIFF[1], Y0[r], ua), G_DLIM, (Y0[r], ua), xticks=t, xtext=[E.plain(v, 3) for v in t])
    zero(cv, bx, Y0[r], ua)
    bx.hci(Y0[r] + G_DY[1], *V["g"]["dmm"], BLOOD, size=DOT)
    cv.text((bx.x0 + bx.x1) / 2, Y0[r] - 4, "Without retina minus median", size="tick", xa="center", ya="bottom")
    cv.text((bx.x0 + bx.x1) / 2, ua + 20, "Paired AUROC difference", size="label", xa="center")


HI_ARMS = [("eye_plus_remaining", "BRAN with retina", SHARED), ("remaining_no_eye", "BRAN without retina", BLOOD),
           ("eye_and_age", "Retina and age", EYE), ("age_only", "Age only", GREY),
           ("ridge_remaining_plus_retinal_pca64", "Ridge, clinical, retina, age", GREY),
           ("ridge_remaining_no_eye", "Ridge, clinical and age", GREY),
           ("ridge_retinal_pca64_and_age", "Ridge, retina and age", GREY),
           ("ridge_age_and_predicted_sex", "Ridge, age and predicted sex", GREY)]
OFF_SCALE = ("age_only", "ridge_age_and_predicted_sex")
H_X0 = 344.0
H_KEY = HEAD[3] + 12.0
H_ROW0, H_RH, H_GAP = 486.0 - DY, 12.5, 4.0
SLOPE_LIM, OE_LIM = (0.50, 1.35), (0.65, 1.32)
SQUARE = 4.6
H_MODELS = ((LOGISTIC, -2.6, "circle"), (NORMAL, 2.6, "square"))


def h_row(i):
    return H_ROW0 + i * H_RH + (H_GAP if i >= 4 else 0.0) + H_RH / 2


H_BOTTOM = H_ROW0 + 8 * H_RH + H_GAP + 2.0
H = int(math.ceil(H_BOTTOM + 20 + st.SIZES["label"] + 6.0))
assert H <= st.FULL_H, ("taller than 170 mm", H)


def hci_sym(ax, y, v, color, symbol):
    ax.inside_x(*v)
    ax.seg([v[1], v[2]], [y, y], color, 1.0)
    ax.dot(v[0], y, color, size=DOT if symbol == "circle" else SQUARE, symbol=symbol)


def panel_h(cv, V):
    cv.heading(H_X0, HEAD[3], "h", "Calibration, low haemoglobin")
    labw = max(K.tw_px(lab, st.SIZES["tick"]) for _, lab, _ in HI_ARMS)
    lab_r = H_X0 + labw
    a0 = lab_r + 8.0
    a1 = a0 + 96.0
    b0, b1 = a1 + 16.0, W - 8.0
    assert b1 - b0 > 90.0, (b0, b1)
    kax = cv.axis((H_X0, b1, H_KEY - 5, H_KEY + 5), (H_X0, b1), (H_KEY - 5, H_KEY + 5), xline=False)
    x = H_X0 + 2
    for symbol, t in (("circle", "Logistic map"), ("square", "Normal residual map")):
        kax.dot(x + 2.5, H_KEY, INK, size=DOT if symbol == "circle" else SQUARE, symbol=symbol)
        x += 8
        x += cv.text(x, H_KEY, t, size="tick") + 12
    assert x - 12 <= b1
    sax = cv.axis((a0, a1, H_ROW0, H_BOTTOM), SLOPE_LIM, (H_ROW0, H_BOTTOM), xticks=[0.6, 0.8, 1.0, 1.2],
                  xtext=["0.6", "0.8", "1.0", "1.2"])
    oax = cv.axis((b0, b1, H_ROW0, H_BOTTOM), OE_LIM, (H_ROW0, H_BOTTOM), xticks=[0.7, 0.9, 1.1, 1.3],
                  xtext=["0.7", "0.9", "1.1", "1.3"])
    for ax in (sax, oax):
        cv.line(ax.X(1.0), H_ROW0, ax.X(1.0), H_BOTTOM, color=GREY, width=0.75, dash=E.DASH)
    for i, (k, lab, col) in enumerate(HI_ARMS):
        y = h_row(i)
        cv.text(lab_r, y, lab, size="tick", xa="right")
        for model, dy, symbol in H_MODELS:
            s, oe = V["h"][f"{k}|{model}"]
            hci_sym(oax, y + dy, oe, col, symbol)
            if k in OFF_SCALE:
                assert not (SLOPE_LIM[0] < s[0] < SLOPE_LIM[1]), (k, model, s)
            else:
                hci_sym(sax, y + dy, s, col, symbol)
        if k in OFF_SCALE:
            w = cv.text(a0 + 5.0, y, "Off scale", size="tick", color=SLATE)
            assert a0 + 5.0 + w < sax.X(1.0) - 3.0
    cv.text((a0 + a1) / 2, H_BOTTOM + 20, "Calibration slope", size="label", xa="center")
    cv.text((b0 + b1) / 2, H_BOTTOM + 20, "Observed over expected", size="label", xa="center")
    return dict(sax=sax, oax=oax, lab_r=lab_r, a0=a0, bottom=H_BOTTOM + 20 + st.SIZES["label"])


def draw(V):
    cv = K.NMCanvas(W, H)
    for f in (panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g):
        f(cv, V)
    hp = panel_h(cv, V)
    assert hp["bottom"] + 4 <= H, (hp["bottom"], H)
    cv.flush()
    if cv.check(FIG_ID):
        print("draft", E.preview(cv.fig, "ed4", W, H))
        raise SystemExit("text check failed")
    letters = [t for t, px, _, b in cv.texts if b and px == st.SIZES["letter"]]
    assert letters == list("abcdefgh"), letters
    assert not any("PLACEHOLDER" in t for t, *_ in cv.texts)
    return cv, hp


def draw_pages(V):
    cv, _ = draw(V)
    return [(FIG_ID, cv.fig, W, H)]


def render(payload):
    return draw_pages(payload['draw'])

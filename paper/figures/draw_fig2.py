"""Drawing code for Fig. 2.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_fig2.py
(SHA-256 7e9aafef7d54a2c4ae50796ffb9f7384fd4c27ae66ce9da1ab840c7004af3edf).

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
import plotly.graph_objects as go
FIG_ID = "fig2"
MINUS = "−"
DAGGER = "†"
W = st.FULL_W
ML, MR, MT, MB = 6, 6, 6, 4
PWP = W - ML - MR
LP0, LP1 = 124, 266
BX0, BX1 = 324, 664
S_PX = BX1 - BX0
B_LET = 286
P = 12
GG = 4
BH = 7.5
HEAD_TO_KEY, HEAD_TO_ROW = 12, 28
HEAD_TO_ROW_C = 40
N_DY = -1.5
DASH = "2px,1.5px"
GAP = 16
HEAD_H = 12
TITLE_DY = 22
END_DY = 29
SZ = st.SIZES
ERR_W = 1.3
LINE_MAX = 1.33


def tw(s, size):
    return st.text_w(s, SZ[size])


def need(cond, msg):
    if not cond:
        raise SystemExit(f"CHECK FAILED  {msg}")


class Canvas:
    def __init__(self, height):
        self.H = height
        self.PH = height - MT - MB
        self.fig = go.Figure()
        self.ax = {}
        self.texts = []

    def X(self, px):
        return px / PWP

    def Y(self, py):
        return 1 - py / self.PH

    def text(self, px, py, s, size="label", color=None, xa="left", ya="middle", angle=0, bold=False):
        color = color or st.INK
        need(size in SZ, size)
        need(color in (st.INK, st.SLATE, st.GREY), color)
        body = f"<b>{s}</b>" if bold else s
        self.fig.add_annotation(x=self.X(px), y=self.Y(py), xref="paper", yref="paper", text=body,
                                showarrow=False, xanchor=xa, yanchor=ya, textangle=angle, borderpad=0,
                                font=dict(size=SZ[size], color=color))
        w = max(st.text_w(ln, SZ[size], bold) for ln in s.split("<br>"))
        h = SZ[size] * 1.2 * len(s.split("<br>"))
        if angle:
            w, h = h, w
        x0 = px if xa == "left" else px - w if xa == "right" else px - w / 2
        y0 = py - h if ya == "bottom" else py if ya == "top" else py - h / 2
        self.texts.append((s, x0, y0, x0 + w, y0 + h))
        return w

    def line(self, x0, y0, x1, y1, color, width=0.75, dash=None):
        self.fig.add_shape(type="line", xref="paper", yref="paper", x0=self.X(x0), y0=self.Y(y0),
                           x1=self.X(x1), y1=self.Y(y1), line=dict(color=color, width=width, dash=dash))

    def rect(self, x0, y0, x1, y1, fill=None, edge=None, width=0.0, dash=None):
        self.fig.add_shape(type="rect", xref="paper", yref="paper", x0=self.X(x0), y0=self.Y(y0),
                           x1=self.X(x1), y1=self.Y(y1), fillcolor=fill or "rgba(0,0,0,0)",
                           line=dict(color=edge or "rgba(0,0,0,0)", width=width, dash=dash))

    def axes(self, n, px0, px1, py_top, py_bottom, xrange, yrange, xline=True, yline=True, xticks=None,
             yticks=None):
        xa, ya = ("xaxis", "yaxis") if n == 1 else (f"xaxis{n}", f"yaxis{n}")
        xr, yr = ("x", "y") if n == 1 else (f"x{n}", f"y{n}")
        yr_ = [py_bottom, py_top] if yrange is None else list(yrange)
        xd = dict(domain=[self.X(px0), self.X(px1)], range=list(xrange), anchor=yr, showline=xline,
                  automargin=False, fixedrange=True)
        yd = dict(domain=[self.Y(py_bottom), self.Y(py_top)], range=yr_, anchor=xr, showline=yline,
                  automargin=False, fixedrange=True)
        if xticks is None:
            xd.update(showticklabels=False, ticks="")
        else:
            xd.update(tickvals=list(xticks[0]), ticktext=list(xticks[1]))
        if yticks is None:
            yd.update(showticklabels=False, ticks="")
        else:
            yd.update(tickvals=list(yticks[0]), ticktext=list(yticks[1]))
        self.fig.update_layout({xa: xd, ya: yd})
        self.ax[n] = dict(x=xr, y=yr, px=(px0, px1), py=(py_top, py_bottom), xr=tuple(xrange), yr=tuple(yr_))
        return xr, yr

    def to_px(self, n, xd, yd):
        a = self.ax[n]
        (x0, x1), (yt, yb) = a["px"], a["py"]
        (r0, r1), (q0, q1) = a["xr"], a["yr"]
        return x0 + (xd - r0) / (r1 - r0) * (x1 - x0), yb - (yd - q0) / (q1 - q0) * (yb - yt)

    def key_swatch(self, px, cy, color, label):
        self.rect(px, cy - 3.5, px + 7, cy + 3.5, fill=color)
        w = self.text(px + 10, cy, label)
        return px + 10 + w

    def key_glyph(self, px, cy, color, label, d):
        self.line(px, cy, px + 14, cy, color=color, width=ERR_W)
        self.fig.add_shape(type="circle", xref="paper", yref="paper", x0=self.X(px + 7 - d / 2),
                           x1=self.X(px + 7 + d / 2), y0=self.Y(cy + d / 2), y1=self.Y(cy - d / 2),
                           fillcolor=color, line=dict(width=0))
        w = self.text(px + 18, cy, label)
        return px + 18 + w

    def heading(self, px, hy, letter, heading):
        st.panel_letter(self.fig, self.X(px), self.Y(hy), letter, heading)
        w = tw(letter, "letter") + st.text_w("   ", SZ["label"]) + st.text_w(heading, SZ["heading"], True)
        self.texts.append((f"{letter} {heading}", px, hy - HEAD_H, px + w, hy))

    def xtitle(self, n, py_axis, label):
        a = self.ax[n]
        self.text((a["px"][0] + a["px"][1]) / 2, py_axis + TITLE_DY, label, xa="center")


def layout(n_rows_c, n_sel):
    L = {}
    hy = HEAD_H + 2
    L["a_head"] = hy
    r0 = hy + HEAD_TO_ROW
    L["a_last"] = r0 + 9 * P + 4 * GG
    L["a_axis"] = L["a_last"] + P / 2 + 2
    a_end = L["a_axis"] + END_DY
    L["c_head"] = a_end + GAP + HEAD_H
    L["c_axis"] = L["c_head"] + HEAD_TO_ROW_C + (n_rows_c - 1) * P + P / 2 + 2
    c_end = L["c_axis"] + END_DY
    L["d_head"] = c_end + GAP + HEAD_H
    L["d_axis"] = L["d_head"] + HEAD_TO_ROW + (n_sel - 1) * P + P / 2 + 2
    L["paper_h"] = L["d_axis"] + END_DY + 2
    L["b_top"] = r0 - P / 2
    L["b_axis"] = L["b_top"] + S_PX
    L["s_top"] = L["b_axis"] + END_DY + 16
    L["s_axis"] = L["d_axis"]
    return L


def rows_of(panel):
    cols = panel["columns"]
    return [dict(zip(cols, r)) for r in panel["rows"]]


def draw_a(C, p, L, K):
    hy = L["a_head"]
    C.heading(0, hy, "a", K["head"]["a"])
    C.text(LP1, hy + N_DY, f"n ≥ {p['draw']['n_paired']:,}", xa="right", ya="bottom")
    ky = hy + HEAD_TO_KEY
    kx = C.key_swatch(0, ky, K["shared"], "BRAN shared space" + DAGGER)
    C.key_swatch(kx + 12, ky, K["grey"], "Comparator")
    rows = rows_of(p["panels"]["a"])
    vals = [r["value"] for r in rows if r["value"] is not None]
    lim = (0.0, 0.13)
    need(max(vals) < lim[1], "a values inside the axis")
    top = hy + HEAD_TO_ROW - P / 2
    xr, yr = C.axes(1, LP0, LP1, top, L["a_axis"], lim, None, xticks=((0, 0.05, 0.10), ("0", "0.05", "0.10")))
    cy = hy + HEAD_TO_ROW
    cur = 0
    for r in rows:
        if r["key"] == "retained":
            break
        if r["group"] != cur:
            cy += GG
            cur = r["group"]
        main = r["series"] == "BRAN shared space"
        ind = 0 if main else 8
        need(tw(r["row"], "label") + ind + 6 <= LP0, r["row"])
        C.text(ind, cy, r["row"], color=st.INK if main else st.SLATE)
        C.fig.add_trace(go.Bar(x=[r["value"]], y=[cy], width=BH, orientation="h", base=0,
                               marker=dict(color=K["shared"] if main else K["grey"], line=dict(width=0)),
                               customdata=[r["key"]], name=f"a:{r['key']}", xaxis=xr, yaxis=yr, hoverinfo="skip"))
        cy += P
    cy += GG
    need(abs(cy - L["a_last"]) < 1e-9, "a rows end at the placeholder row")
    C.text(0, cy, "Retained model")
    w = tw("PLACEHOLDER", "tick") + 10
    C.rect(LP0 + 4, cy - 5.5, LP0 + 4 + w, cy + 5.5, edge=K["ref"], width=0.6, dash=DASH)
    C.text(LP0 + 4 + w / 2, cy, "PLACEHOLDER", size="tick", color=st.GREY, xa="center")
    C.xtitle(1, L["a_axis"], "Proportion or MRR")


def draw_c(C, p, L, K):
    hy = L["c_head"]
    C.heading(0, hy, "c", K["head"]["c"])
    C.text(LP1, hy + N_DY, f"n ≥ {p['draw']['n_floor']:,}", xa="right", ya="bottom")
    ky = hy + HEAD_TO_KEY
    kx = C.key_glyph(0, ky, K["eye"], "Against age only", d=5.0)
    C.key_glyph(kx + 12, ky, K["grey"], "Against training median", d=4.2)
    lim = (-9.0, 6.0)
    ticks = (-8, -4, 0, 4)
    labs = tuple("0" if t == 0 else (MINUS if t < 0 else "+") + str(abs(t)) for t in ticks)
    top = hy + HEAD_TO_ROW_C - P / 2
    xr, yr = C.axes(3, LP0, LP1, top, L["c_axis"], lim, None, yline=False, xticks=(ticks, labs))
    C.fig.add_shape(type="line", xref=xr, yref=yr, x0=0, x1=0, y0=top, y1=L["c_axis"], layer="below",
                    line=dict(color=K["ref"], width=0.75, dash="dash"))
    zx, _ = C.to_px(3, 0.0, 0.0)
    dy = hy + HEAD_TO_ROW_C - 14
    C.text(zx - 4, dy, "Lower error", size="tick", color=st.SLATE, xa="right")
    C.text(zx + 4, dy, "Higher error", size="tick", color=st.SLATE)
    need(zx - 4 - tw("Lower error", "tick") >= LP0 and zx + 4 + tw("Higher error", "tick") <= LP1 + 2, "c header")
    OFF = 2.8
    rows = rows_of(p["panels"]["c"])
    series = (("vs_age_only_pct", "open_age_only", K["eye"], 5.0, -OFF, "iv"),
              ("vs_median_pct", "open_median", K["grey"], 4.2, OFF, "tm"))
    for sk, ok, color, dd, off, tag in series:
        for hollow in (False, True):
            xs, ys, lo, hi, keys = [], [], [], [], []
            for i, r in enumerate(rows):
                if r[ok] != hollow:
                    continue
                e, l_, h_ = r[sk], r[sk + "_low"], r[sk + "_high"]
                need(lim[0] < l_ and h_ < lim[1], f"c inside axis {r['field_key']}")
                xs.append(e)
                ys.append(hy + HEAD_TO_ROW_C + i * P + off)
                lo.append(e - l_)
                hi.append(h_ - e)
                keys.append(r["field_key"])
            if not xs:
                continue
            C.fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", customdata=keys, name=f"c:{tag}:{'open' if hollow else 'filled'}",
                error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=color, thickness=ERR_W,
                             width=0),
                marker=dict(symbol="circle", size=dd, opacity=1, color="white" if hollow else color,
                            line=dict(color=color, width=1.0 if hollow else 0)),
                xaxis=xr, yaxis=yr, hoverinfo="skip", cliponaxis=False))
    for i, r in enumerate(rows):
        C.text(0, hy + HEAD_TO_ROW_C + i * P, r["field"])
    C.xtitle(3, L["c_axis"], "MAE difference (% of training-median MAE)")


def draw_d(C, p, L, K):
    hy = L["d_head"]
    C.heading(0, hy, "d", K["head"]["d"])
    ky = hy + HEAD_TO_KEY
    kx = 0
    cols = {"shared_derived_pct": K["shared"], "retinal_residual_pct": K["eye"], "clinical_residual_pct": K["cardio"]}
    for (key, col), name in zip(cols.items(), p["draw"]["block_names"]):
        kx = C.key_swatch(kx, ky, col, name) + 10
    need(kx - 10 <= LP1 + 2, f"d key fits {kx}")
    top = hy + HEAD_TO_ROW - P / 2
    xr, yr = C.axes(4, LP0, LP1, top, L["d_axis"], (0, 100), None,
                    xticks=((0, 25, 50, 75, 100), ("0", "25", "50", "75", "100")))
    rows = rows_of(p["panels"]["d"])
    keys = list(cols)
    for j, key in enumerate(keys):
        xs, base, ys, codes = [], [], [], []
        for i, r in enumerate(rows):
            xs.append(r[key])
            base.append(sum(r[k] for k in keys[:j]))
            ys.append(hy + HEAD_TO_ROW + i * P)
            codes.append(r["condition_code"])
        C.fig.add_trace(go.Bar(x=xs, base=base, y=ys, width=BH, orientation="h", customdata=codes, name=f"d:{key}",
                               marker=dict(color=cols[key], line=dict(width=0)), xaxis=xr, yaxis=yr,
                               hoverinfo="skip"))
    for i, r in enumerate(rows):
        C.text(0, hy + HEAD_TO_ROW + i * P, r["condition"])
    C.xtitle(4, L["d_axis"], "Share of mean absolute contribution (%)")


def draw_b(C, p, L, K):
    d = p["draw"]
    pts = rows_of(p["panels"]["b"])
    G_REF, D_MAX, D_MIN = d["g_ref"], 22.0, 3.6

    def diam(g):
        return max(D_MIN, D_MAX * math.sqrt(abs(g) / G_REF))

    need(max(abs(q["added_by_retina"]) for q in pts) <= G_REF + 1e-9, "largest gain inside the size scale")
    hy = L["a_head"]
    C.heading(B_LET, hy, "b", K["head"]["b"])
    lim = (0.50, 0.92)
    ticks = (0.5, 0.6, 0.7, 0.8, 0.9)
    tl = tuple(f"{t:.1f}" for t in ticks)
    top, bot = L["b_top"], L["b_axis"]
    xr, yr = C.axes(2, BX0, BX1, top, bot, lim, lim, xticks=(ticks, tl), yticks=(ticks, tl))
    C.fig.add_shape(type="line", xref=xr, yref=yr, x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], layer="below",
                    line=dict(color=K["ref"], width=0.75, dash="dash"))
    C.xtitle(2, bot, "Clinical-only AUROC")
    C.text(BX0 - 30, (top + bot) / 2, "Retinal-only AUROC", xa="center", angle=-90)
    gcol = {"eye": K["eye"], "cardio": K["cardio"], "other": K["other"]}

    def scatter(sub, name, hollow):
        sub = sorted(sub, key=lambda q: -diam(q["added_by_retina"]))
        if not sub:
            return
        cols = [gcol[q["group"]] for q in sub]
        C.fig.add_trace(go.Scatter(
            x=[q["clinical_only_auroc"] for q in sub], y=[q["retinal_only_auroc"] for q in sub], mode="markers",
            customdata=[q["condition_code"] for q in sub], name=name,
            marker=dict(symbol="circle", size=[diam(q["added_by_retina"]) for q in sub], opacity=1,
                        color=["white"] * len(sub) if hollow else cols,
                        line=dict(color=cols if hollow else "white", width=1.2 if hollow else 0.6)),
            xaxis=xr, yaxis=yr, hoverinfo="skip", cliponaxis=False))

    scatter([q for q in pts if not q["open_marker"]], "b:filled", False)
    scatter([q for q in pts if q["open_marker"]], "b:open", True)
    by = {q["condition_code"]: q for q in pts}
    for code, (name, where, dx) in d["lab"].items():
        q = by[code]
        cx, cy = C.to_px(2, q["clinical_only_auroc"], q["retinal_only_auroc"])
        ox, oy = d["nudge"].get(code, (0, 0))
        cx, cy = cx + ox, cy + oy
        r = diam(q["added_by_retina"]) / 2
        if where == "r":
            C.text(cx + r + 4, cy, name)
        elif where == "l":
            C.text(cx - r - 4, cy, name, xa="right")
        elif where == "t":
            C.text(cx + dx * 57, cy - r - 2, name, xa="center", ya="bottom")
        elif where == "tr":
            C.text(cx + dx * 57 + r, cy - r - 2, name, xa="right", ya="bottom")
        elif where == "tl":
            C.text(BX0 + 4 + ox, cy - r - 2, name, ya="bottom")
    # size key in the empty upper-left corner of the scatter
    kx0, ky0 = BX0 + 10, top + 10
    C.text(kx0, ky0, "AUROC added by retina")
    base = ky0 + 32
    xx = kx0 + 2
    kxs, kys, kds, kfill, kw = [], [], [], [], []
    for gv in (0.01, 0.05, 0.10, -0.02):
        dd = diam(gv)
        cxk = xx + dd / 2
        kxs.append(cxk)
        kys.append(base - dd / 2)
        kds.append(dd)
        kfill.append("white" if gv < 0 else K["faint"])
        kw.append(1.2 if gv < 0 else 0.6)
        lab = f"{MINUS}{abs(gv):.2f}" if gv < 0 else f"+{gv:.2f}"
        C.text(cxk, base + 3, lab, size="tick", xa="center", ya="top")
        xx = cxk + dd / 2 + 20
    xd = [lim[0] + (x - BX0) / S_PX * (lim[1] - lim[0]) for x in kxs]
    yd = [lim[1] - (y - top) / S_PX * (lim[1] - lim[0]) for y in kys]
    C.fig.add_trace(go.Scatter(x=xd, y=yd, mode="markers", name="b:size_key",
                               marker=dict(symbol="circle", size=kds, color=kfill, opacity=1,
                                           line=dict(color=K["ref"], width=kw)),
                               xaxis=xr, yaxis=yr, hoverinfo="skip", cliponaxis=False))
    key_box = (kx0 - 4, ky0 - 8, xx, base + 16)
    for q in pts:
        cx, cy = C.to_px(2, q["clinical_only_auroc"], q["retinal_only_auroc"])
        r = diam(q["added_by_retina"]) / 2
        need(not (key_box[0] < cx + r and cx - r < key_box[2] and key_box[1] < cy + r and cy - r < key_box[3]),
             f"size key covers {q['condition']}")

    # strip of all 26 gains by group, with the group mean and its paired 95% interval
    slim = (-0.03, 0.16)
    s_top, s_ax = L["s_top"], L["s_axis"]
    names = {"eye": "Eye<br>conditions", "cardio": "Cardiometabolic<br>and vascular", "other": "Other"}
    colx = {"eye": 0.5, "cardio": 1.5, "other": 2.5}
    sxr, syr = C.axes(5, BX0, BX1, s_top, s_ax, (0, 3), slim,
                      xticks=(tuple(colx.values()), tuple(names[k] for k in colx)),
                      yticks=((0.0, 0.05, 0.10, 0.15), ("0", "0.05", "0.10", "0.15")))
    C.fig.update_layout(xaxis5=dict(tickfont=dict(size=SZ["label"])))
    C.text(BX0 - 30, (s_top + s_ax) / 2, "AUROC added by retina", xa="center", angle=-90)
    C.fig.add_shape(type="line", xref=sxr, yref=syr, x0=0, x1=3, y0=0, y1=0, layer="below",
                    line=dict(color=K["ref"], width=0.75, dash="dash"))
    col_px = S_PX / 3
    DS = 5.2
    grp = {r["group"]: r for r in rows_of(p["panels"]["b strip"])}
    placed_all = []
    for k, cxd in colx.items():
        members = sorted([q for q in pts if q["group"] == k], key=lambda q: q["added_by_retina"])
        g = grp[k]
        half = 26 / col_px
        C.fig.add_trace(go.Scatter(x=[cxd - half, cxd + half], y=[g["macro_difference"]] * 2, mode="lines",
                                   customdata=[k, k], name=f"b:mean:{k}", line=dict(color=K["ink"], width=LINE_MAX),
                                   xaxis=sxr, yaxis=syr, hoverinfo="skip"))
        ci_x = cxd + (half + 8 / col_px)
        C.fig.add_trace(go.Scatter(x=[ci_x, ci_x], y=[g["ci95_low"], g["ci95_high"]], mode="lines",
                                   customdata=[k, k], name=f"b:ci:{k}", line=dict(color=K["ink"], width=LINE_MAX),
                                   xaxis=sxr, yaxis=syr, hoverinfo="skip"))
        C.fig.add_trace(go.Scatter(x=[ci_x], y=[g["macro_difference"]], mode="markers", customdata=[k],
                                   name=f"b:ci_mark:{k}", marker=dict(symbol="diamond", size=5.4, color=K["ink"],
                                                                     line=dict(width=0), opacity=1),
                                   xaxis=sxr, yaxis=syr, hoverinfo="skip", cliponaxis=False))
        cxp, _ = C.to_px(5, cxd, 0)
        placed = []
        for q in members:
            _, yy = C.to_px(5, cxd, q["added_by_retina"])
            for off in (0.0, -6.5, 6.5, -13, 13, -19.5, 19.5, -26, 26):
                if all(math.hypot(off - o, yy - qq) >= DS + 0.6 for o, qq in placed):
                    break
            else:
                raise SystemExit(f"strip crowding {k}")
            need(abs(off) + DS / 2 < col_px / 2 - 20, "strip inside its column")
            placed.append((off, yy))
            placed_all.append((q, cxp + off, yy))
        xs = [cxd + o / col_px for o, _ in placed]
        for hollow in (False, True):
            idx = [i for i, q in enumerate(members) if q["open_marker"] == hollow]
            if not idx:
                continue
            C.fig.add_trace(go.Scatter(
                x=[xs[i] for i in idx], y=[members[i]["added_by_retina"] for i in idx], mode="markers",
                customdata=[members[i]["condition_code"] for i in idx],
                name=f"b:strip:{k}:{'open' if hollow else 'filled'}",
                marker=dict(symbol="circle", size=DS, opacity=1, color="white" if hollow else gcol[k],
                            line=dict(color=gcol[k], width=1.0 if hollow else 0)),
                xaxis=sxr, yaxis=syr, hoverinfo="skip", cliponaxis=False))
    # key in the empty upper part of the Other column
    gx, gy = C.to_px(5, 1.96, 0.148)
    C.line(gx, gy, gx + 14, gy, color=K["ink"], width=LINE_MAX)
    C.line(gx + 18, gy - 5, gx + 18, gy + 5, color=K["ink"], width=LINE_MAX)
    C.fig.add_shape(type="path", xref="paper", yref="paper",
                    path=f"M {C.X(gx + 18)},{C.Y(gy - 2.7)} L {C.X(gx + 20.7)},{C.Y(gy)} L {C.X(gx + 18)},{C.Y(gy + 2.7)} "
                         f"L {C.X(gx + 15.3)},{C.Y(gy)} Z", fillcolor=K["ink"], line=dict(width=0))
    w = C.text(gx + 24, gy, "Group mean, 95% CI")
    gbox = (gx - 4, gy - 8, gx + 24 + w + 2, gy + 8)
    need(gbox[2] <= BX1, f"group key inside the strip {gbox}")
    for q, x, y in placed_all:
        need(not (gbox[0] < x + DS / 2 and x - DS / 2 < gbox[2] and gbox[1] < y + DS / 2 and y - DS / 2 < gbox[3]),
             f"group key covers {q['condition']}")
    for k, g in grp.items():
        cxd = colx[k]
        top_px = C.to_px(5, cxd, g["ci95_high"])[1]
        need(not (gbox[1] < top_px < gbox[3] and gbox[0] < C.to_px(5, cxd, 0)[0] + 40), "group key covers an interval")


def draw(payload):
    rc, rd = len(payload["panels"]["c"]["rows"]), len(payload["panels"]["d"]["rows"])
    L = layout(rc, rd)
    H = int(math.ceil(L["paper_h"] + MT + MB))
    C = Canvas(H)
    C.PH = H - MT - MB
    fig = C.fig
    fig.update_layout(template="bran", width=W, height=H, margin=dict(l=ML, r=MR, t=MT, b=MB),
                      barmode="overlay", bargap=0, showlegend=False, hovermode=False, dragmode=False)
    K = payload["draw"]["colours"] | {"head": payload["draw"]["head"]}
    draw_a(C, payload, L, K)
    draw_b(C, payload, L, K)
    draw_c(C, payload, L, K)
    draw_d(C, payload, L, K)
    return C, H, L


def render(payload):
    C, H, _ = draw(payload)
    return [(FIG_ID, C.fig, W, H)]

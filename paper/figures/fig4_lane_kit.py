"""Helpers of the fig4_lane_kit lane kit used by the drawing code.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
fig4_lane_kit.py
(SHA-256 b5d52fa699889586a98b3b12da37315160bbe36ed4f3a80cb9b58646bd256a67).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import re
import bran_plotly_style as st


def tw_px(s: str, size_px: float, bold: bool = False) -> float:
    return st.text_w(s, size_px, bold)


def wrap(s: str, width: float, size_px: float, bold: bool = False) -> list[str]:
    """Greedy word wrap to a pixel width in the profile font."""
    lines, cur = [], ""
    for word in s.split(" "):
        trial = word if not cur else cur + " " + word
        if tw_px(trial, size_px, bold) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    for ln in lines:
        assert tw_px(ln, size_px, bold) <= width + 0.5, (ln, width)
    return lines


import ed_canvas as E


class NMCanvas(E.Canvas):
    """ed_canvas.Canvas with text measured in the profile font. Sizes are the names of st.SIZES."""

    def text(self, x, y, s, size="label", color=None, bold=False, xa="left", ya="middle", angle=0, check=True):
        color = st.INK if color is None else color
        assert size in st.SIZES, size
        assert color in (st.INK, st.SLATE), (color, s)
        for bad in (";", "—", ":"):
            assert bad not in s, s
        px = st.SIZES[size]
        w = tw_px(s, px, bold)
        h = px * 1.0
        if angle == 0 and check:
            x0 = x if xa == "left" else (x - w if xa == "right" else x - w / 2)
            y0 = y - h / 2 if ya == "middle" else (y - h if ya == "bottom" else y)
            self.boxes.append((x0, y0, x0 + w, y0 + h, s))
        self.texts.append((s, px, color, bold))
        body = f"<b>{s}</b>" if bold else s
        self.fig.add_annotation(x=self.fx(x), y=self.fy(y), xref="paper", yref="paper", text=body, showarrow=False,
                                xanchor=xa, yanchor=ya, textangle=angle, borderpad=0,
                                font=dict(size=px, color=color, family=st.FONT))
        return w

    def heading(self, x, y, letter, name):
        """Panel letter at 8 pt and a noun-phrase heading at 7 pt on one text-box bottom."""
        assert re.fullmatch(r"[a-z]", letter), letter
        assert 1 <= len(name.split()) <= 4 and name[0].isupper() and not name.endswith("."), name
        w = self.text(x, y, letter, size="letter", bold=True, ya="bottom")
        self.text(x + w + 5.0, y - 0.4, name, size="heading", bold=True, ya="bottom")

    def axis(self, box, xr, yr, xticks=None, xtext=None, yticks=None, ytext=None, xline=True, yline=False,
             xside="bottom", yside="left"):
        """ed_canvas.Canvas.axis, with the tick-label ink boxes measured in the profile font."""
        n0 = len(self.boxes)
        ax = super().axis(box, xr, yr, xticks=xticks, xtext=xtext, yticks=yticks, ytext=ytext, xline=xline,
                          yline=yline, xside=xside, yside=yside)
        added = self.boxes[n0:]
        nx = len(xticks) if xticks is not None else 0
        fresh = []
        for i, b in enumerate(added):
            w = tw_px(b[4], st.SIZES["tick"])
            if i < nx:                                  # centred under or over the tick
                cx = (b[0] + b[2]) / 2
                fresh.append((cx - w / 2, b[1], cx + w / 2, b[3], b[4]))
            else:                                       # right-aligned left of the axis
                fresh.append((b[2] - w, b[1], b[2], b[3], b[4]))
        self.boxes[n0:] = fresh
        return ax

    def wrap_note(self, s: str, width: float, size: str = "tick") -> list[str]:
        return wrap(s, width, st.SIZES[size])


TP = {"row": 10.9, "group": 13.8, "line": 10.8, "note": 10.6}
T_PAD, T_HEAVY, T_LIGHT = 4.0, 1.1, 0.65
HEADING_H = 17.0
PAGE_BOTTOM = 5.0


def grid(rows, heads, aligns, width, size_px, label_pad=12.0, min_gap=8.0):
    """Column anchors for a computed grid. Every column is as wide as its widest header or cell, and the gaps between
    columns are equal. Returns (columns [(x, anchor)], widths, right edges, gap)."""
    lab_w = max(tw_px(lab, size_px, bold=(kind == "group")) for kind, lab, _ in rows if kind not in ("group",))
    widths = []
    for j, h in enumerate(heads):
        col = [r[2][j] for r in rows if r[0] in ("row", "lab") and len(r[2]) > j and r[2][j]]
        if j == len(heads) - 1:      # the second value of a span row sits in the last column
            col += [r[2][1] for r in rows if r[0] == "span" and len(r[2]) > 1 and r[2][1]]
        widths.append(max([tw_px(h, size_px, bold=True)] + [tw_px(c, size_px) for c in col]))
    gap = (width - lab_w - label_pad - sum(widths)) / max(1, len(widths) - 1)
    assert gap >= min_gap, ("columns do not fit", gap, widths, lab_w)
    rights, x = [], lab_w + label_pad
    for w in widths:
        x += w
        rights.append(x)
        x += gap
    cols = [[xr if a == "right" else xr - w, a] for xr, w, a in zip(rights, widths, aligns)]
    return cols, widths, rights, gap


def table_items(blocks, width, size_px):
    """Flat list of items with their heights from blocks. A block is a dict with optional letter and name (a part
    heading), optional sub (a bold line), header {spans, heads}, rows [(kind, label, values)], cols, span_x and note."""
    items = []
    line = TP["line"]
    for bi, b in enumerate(blocks):
        if b.get("gap"):
            items.append({"k": "gap", "h": b["gap"]})
        if b.get("letter"):
            items.append({"k": "heading", "h": HEADING_H, "letter": b["letter"], "name": b["name"], "block": bi})
        if b.get("sub"):
            items.append({"k": "sub", "h": line + 3.0, "text": b["sub"], "block": bi})
        hd = b["header"]
        hh = T_PAD + line + T_PAD + 2.0 + ((line + 3.0) if hd["spans"] else 0.0)
        items.append({"k": "header", "h": hh, "block": bi})
        n = len(b["rows"])
        for ri, r in enumerate(b["rows"]):
            items.append({"k": "row", "h": TP["group"] if r[0] == "group" else TP["row"], "block": bi, "i": ri,
                          "last": n - 1 - ri})
        lines = wrap(b["note"], width, size_px) if b.get("note") else []
        items.append({"k": "close", "h": 3.0 + 4.0 + len(lines) * TP["note"], "lines": lines, "block": bi})
    return items


def paginate(items, max_h):
    """Items onto pages no taller than max_h. The start of a block is never left without its header and three rows, the
    last two rows of a block stay with its closing rule and note, and a block that breaks continues under a line
    "<letter>, continued" with its column heads repeated. Returns [[item, y], ...] per page and each page height."""
    pages, cur, y = [], [], 0.0
    limit = max_h - PAGE_BOTTOM
    cont_h = TP["line"] + 3.0

    def need(j):
        it = items[j]
        starts = it["k"] in ("heading", "sub") or (it["k"] == "header" and (j == 0 or items[j - 1]["k"] not in
                                                                             ("heading", "sub")))
        if starts:
            k, h = j, 0.0
            rows = 0
            while k < len(items) and rows < 3 and items[k]["k"] != "close":
                h += items[k]["h"]
                rows += items[k]["k"] == "row"
                k += 1
            return h
        if it["k"] == "row" and it["last"] < 2:
            k, h = j, 0.0
            while items[k]["k"] != "close":
                h += items[k]["h"]
                k += 1
            return h + items[k]["h"]
        return it["h"]

    last_letter = {}
    for j, it in enumerate(items):
        if it["k"] == "heading":
            last_letter[it["block"]] = it["letter"]
        if it["k"] == "gap" and not cur:
            continue
        if y + need(j) > limit and cur:
            pages.append((cur, y))
            cur, y = [], 0.0
            if it["k"] in ("row", "close", "header"):
                blk = it["block"]
                letter = next((v for b, v in sorted(last_letter.items()) if b <= blk), None)
                cur.append([{"k": "sub", "h": cont_h, "text": f"{letter}, continued" if letter else "Continued",
                             "block": blk}, y])
                y += cont_h
                if it["k"] != "header":
                    hd = next(x for x in items if x["k"] == "header" and x["block"] == blk)
                    cur.append([dict(hd), y])
                    y += hd["h"]
        if it["k"] == "gap" and not cur:
            continue
        cur.append([it, y])
        y += it["h"]
    if cur:
        pages.append((cur, y))
    for pg, h in pages:
        assert h <= limit + 1e-9, ("page too tall", h, limit)
    return pages


def draw_table_pages(blocks, width, max_h, size="label", check_tag="table"):
    """Draw every page of a paginated table. Returns [(canvas, height px, drawn [(text, x, y)])]."""
    size_px = st.SIZES[size]
    items = table_items(blocks, width, size_px)
    out = []
    for pg, h_used in paginate(items, max_h):
        H = int(round(h_used + PAGE_BOTTOM))
        assert H <= max_h, (H, max_h)
        cv = NMCanvas(width, H)
        drawn = []

        def T(x, y, s, **kw):
            w = cv.text(x, y, s, **kw)
            drawn.append((s, x, y))
            return w

        for it, y in pg:
            k = it["k"]
            b = blocks[it["block"]] if "block" in it else None
            if k == "heading":
                cv.heading(0, y + 13.0, it["letter"], it["name"])
                drawn.append((it["letter"], 0, y + 13.0))
            elif k == "sub":
                T(0, y + TP["line"] / 2 + 1.0, it["text"], size=size, bold=True)
            elif k == "header":
                hd = b["header"]
                cv.line(0, y, width, y, width=T_HEAVY)
                yy = y + T_PAD + TP["line"] / 2
                if hd["spans"]:
                    for t, x0, x1 in hd["spans"]:
                        T((x0 + x1) / 2, yy, t, size=size, bold=True, xa="center")
                        cv.line(x0, yy + TP["line"] / 2 + 1.0, x1, yy + TP["line"] / 2 + 1.0, width=T_LIGHT)
                    yy += TP["line"] + 3.0
                for t, x, xa in hd["heads"]:
                    T(x, yy, t, size=size, bold=True, xa=xa)
                cv.line(0, yy + TP["line"] / 2 + T_PAD, width, yy + TP["line"] / 2 + T_PAD, width=T_LIGHT)
            elif k == "row":
                kind, lab, vals = b["rows"][it["i"]]
                if kind == "group":
                    yy = y + TP["group"] / 2 + 1.5
                    w = T(0, yy, lab, size=size, bold=True)
                    if vals:
                        T(w + 6.0, yy, vals[0], size=size, color=st.SLATE)
                    continue
                yy = y + TP["row"] / 2
                colour = st.SLATE if kind == "lab" else st.INK
                T(0, yy, lab, size=size, color=colour)
                if kind == "span":
                    T(b["span_x"], yy, vals[0], size=size, color=st.SLATE)
                    if len(vals) > 1 and vals[1]:
                        x, xa = b["cols"][-1]
                        T(x, yy, vals[1], size=size, xa=xa)
                else:
                    assert len(vals) == len(b["cols"]), (lab, vals)
                    for v, (x, xa) in zip(vals, b["cols"]):
                        if v:
                            T(x, yy, v, size=size, color=colour, xa=xa)
            elif k == "close":
                cv.line(0, y + 3.0, width, y + 3.0, width=T_HEAVY)
                for j2, ln in enumerate(it["lines"]):
                    T(0, y + 7.0 + TP["note"] / 2 + j2 * TP["note"], ln, size=size)
        cv.flush()
        if cv.check(check_tag):
            raise SystemExit(f"text check failed on {check_tag}")
        out.append((cv, H, drawn))
    return out

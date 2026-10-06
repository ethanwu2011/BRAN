"""Drawing code for Extended Data Table 2.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed_table2_merged.py
(SHA-256 c415d20b3c5810eb78ca50a9f5a7d9f50840db2d6e35c4ce3fd8b10301583667).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import bran_plotly_style as st
st.use_nature_medicine()
import fig4_lane_kit as K
FIG_ID = "ed_table2"
W = st.FULL_W
MAX_H = st.FULL_H
SIZE = "label"
OP_HEAD = ["Sensitivity", "Specificity", "PPV", "NPV", "Referred", "Found", "Missed"]
SPAN_HEAD = "Per 1,000 evaluated"
B_HEAD = ["Estimate [95% CI]", "Criterion", "Reading"]


def blocks_of(V):
    """Booktabs blocks of the two parts, with column anchors computed in the profile font."""
    px = st.SIZES[SIZE]
    a, b = V["a"], V["b"]
    cols_a, widths_a, rights_a, _ = K.grid(a["rows"], OP_HEAD, ["right"] * 7, W, px)
    span0 = rights_a[4] - widths_a[4]
    blk_a = {"letter": "a", "name": "Operating points",
             "header": {"spans": [[SPAN_HEAD, span0, rights_a[6]]],
                        "heads": [["Setting", 0, "left"], *[[h, x, al] for h, (x, al) in zip(OP_HEAD, cols_a)]]},
             "rows": a["rows"], "cols": cols_a, "span_x": rights_a[0] - widths_a[0], "note": a["note"]}
    cols_b, widths_b, rights_b, _ = K.grid(b["rows"], B_HEAD, ["right", "left", "left"], W - 4.0, px)
    blk_b = {"gap": 6.0, "letter": "b", "name": "Referral results",
             "header": {"spans": [], "heads": [["Contrast", 0, "left"], *[[h, x, al] for h, (x, al) in
                                                                          zip(B_HEAD, cols_b)]]},
             "rows": b["rows"], "cols": cols_b, "span_x": rights_b[0] - widths_b[0], "note": b["note"]}
    return [blk_a, blk_b]


def draw_pages(V):
    blocks = blocks_of(V)
    pages = K.draw_table_pages(blocks, W, MAX_H, size=SIZE, check_tag=FIG_ID)
    assert len(pages) == 1, ("Extended Data Table 2 must fit one page", len(pages))
    cv, H, _ = pages[0]
    letters = [t for t, px_, _, bold in cv.texts if bold and px_ == st.SIZES["letter"]]
    assert letters == ["a", "b"], letters
    return [(FIG_ID, cv.fig, W, H)]


def render(payload):
    return draw_pages(payload['draw'])

"""Drawing code for Extended Data Table 1.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
build_ed_table1.py
(SHA-256 8b2a9e889f35416c3def3e2f2a7de5b2ab23ca67568889d7832b4eb60c1b39ad).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import re
import bran_plotly_style as st
st.use_nature_medicine()
import display_numbers as dn
import ed_canvas as E
TABLE_ID = "ed_table1"
DISPLAY = dn.label(TABLE_ID)
FIG5 = dn.label("fig5")
W = st.FULL_W
SIZE, NOTE_SIZE = "label", "tick"
PX = st.SIZES[SIZE]
LINE = 11.6
PAD = 4.0
HEAVY, LIGHT = 1.1, 0.65
GUT = 9.0
CW = {"pop": 110.0, "inp": 128.0, "ev": 162.0, "exp": 112.0, "hist": 132.0}
NOTE_LINE = 9.6
NOTE_MARK = "*"
assert abs(sum(CW.values()) + GUT * 4 - W) < 1e-9, sum(CW.values()) + GUT * 4
KEEP = (f"{FIG5}b to 5d", f"{FIG5}e", f"{FIG5}f", f"{FIG5}g", "not an external test", "fold 0 fit", "out of fold",
        "earlier checkpoint", "age unknown", "KNHANES 2022 to 2023", "5 patterns", "(3 primary)",
        f"7 diagnoses{NOTE_MARK}", "10,620 adults", "5,600 adults", "1,928 adults", "≥17,660 survivors",
        "support PLACEHOLDER")


def wrap(s, width, size, keep=()):
    """Greedy word wrap with the advance widths of the profile font. Phrases in keep stay on one line."""
    for k in keep:
        s = s.replace(k, k.replace(" ", E.NBSP))
    # A figure reference stays with the word before it, so no wrapped line starts with "Fig." (which the Nature
    # Medicine check would read as a canvas title).
    s = re.sub(r" (Fig\.)", E.NBSP + r"\1", s)
    lines, cur = [], ""
    for word in s.split(" "):
        trial = word if not cur else cur + " " + word
        if st.text_w(trial, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    for ln in lines:
        assert st.text_w(ln, size) <= width + 0.5, (ln, width)
    return lines


def layout(dr):
    keys = [k for k, _ in dr["heads"]]
    assert keys == list(CW), keys
    xs, x = {}, 0.0
    for k in keys:
        xs[k] = x
        x += CW[k] + GUT
    laid = []
    for r in dr["rows"]:
        cells = {k: wrap(t, CW[k], PX, keep=KEEP) for k, t in zip(keys, r)}
        laid.append((r, cells, max(len(v) for v in cells.values())))
    note = wrap(dr["note"], W, st.SIZES[NOTE_SIZE])
    top = 1.0
    head_h = LINE + 2 * PAD
    group_h = LINE + PAD + 2.0
    body = sum(n * LINE + 2 * PAD for _, _, n in laid) + len(dr["groups"]) * group_h + 2.0
    H = int(round(top + head_h + body + 6 + 4 + len(note) * NOTE_LINE + 4))
    return keys, xs, laid, note, top, head_h, group_h, H


def draw(S):
    assert S["figure"] == TABLE_ID and S["display"] == DISPLAY
    dr = S["draw"]
    keys, xs, laid, note, top, head_h, group_h, H = layout(dr)
    assert H <= st.NM_MAX_H, H
    cv = E.Canvas(W, H)
    cv.line(0, top, W, top, width=HEAVY)
    y = top + PAD + LINE / 2
    for k, h in dr["heads"]:
        cv.text(xs[k], y, h, size=SIZE, bold=True)
    y = top + head_h
    cv.line(0, y, W, y, width=LIGHT)
    y += 2.0
    it = iter(laid)
    for gname, n in dr["groups"]:
        cv.text(0, y + PAD + LINE / 2 + 1, gname, size=SIZE, bold=True)
        y += group_h
        for _ in range(n):
            r, cells, nl = next(it)
            yy = y + PAD + LINE / 2
            for k in keys:
                colour = st.SLATE if cells[k][0].startswith("Not evaluated") else st.INK
                for j, ln in enumerate(cells[k]):
                    cv.text(xs[k], yy + j * LINE, ln, size=SIZE, color=colour)
            y += nl * LINE + 2 * PAD
    y += 1.0
    cv.line(0, y, W, y, width=HEAVY)
    y += 4.0
    for j, ln in enumerate(note):
        cv.text(0, y + NOTE_LINE / 2 + j * NOTE_LINE, ln, size=NOTE_SIZE)
    y += len(note) * NOTE_LINE
    assert y < H + 0.5, (y, H)
    cv.flush()
    assert cv.check(TABLE_ID) == 0, "text check failed"
    problems = st.nm_figure_problems(cv.fig, W, H)
    assert not problems, problems
    return cv, H


def render(payload):
    # Overlap boxes are measured in the profile font (Arial), the font the table is drawn in.
    E.text_w = st.text_w
    cv, H = draw(payload)
    return [(TABLE_ID, cv.fig, W, H)]

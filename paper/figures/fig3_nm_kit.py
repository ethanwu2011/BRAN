"""Helpers of the fig3_nm_kit lane kit used by the drawing code.

Public drawing code for the BRAN paper, generated on 6 October 2026 from the private builder
fig3_nm_kit.py
(SHA-256 86b2cd78090227ee3d79e5028824c57fe653aef653c61bd6d897013ae4804282).

It holds, verbatim and in their original order, only the top-level statements that the drawing entry point reaches,
plus render(), which returns [(stem, figure, width px, height px)] for each page. The code that read the pinned
aggregates, asserted every plotted value against them and wrote the Source Data stays in the private builder, and
changes are made there before this copy is regenerated. Rebuild and verify every figure with
python -m paper.figures.build_all from the repository root.
"""

from __future__ import annotations

import bran_plotly_style as st
import ed_canvas as E
NM_W = st.NM_W
NM_H_MAX = st.NM_MAX_H
SMALL_LANE_PT = 5.2


def apply_nm_sizes() -> None:
    """Opt in to the shared Nature Medicine profile for this process, add the lane's 5.2 pt small size and measure
    text in the profile's font."""
    st.use_nature_medicine()
    st.SIZES["small_lane"] = SMALL_LANE_PT * st.PT
    E.text_w = st.text_w
    for k, v in st.SIZES.items():
        pt = v / st.PT
        assert (5.0 <= pt <= 7.0) or (k == "letter" and pt == 8.0), (k, pt)
    assert st.PROFILE["name"] == st.NM_PROFILE and st.FONT == st.NM_FONT


def heading(cv, x: float, y: float, letter: str, name: str = "") -> None:
    """Bold lowercase panel letter at 8 pt and an optional short bold heading at 7 pt on one text bottom."""
    assert len(letter) == 1 and letter.islower(), letter
    cv.text(x, y, letter, size="letter", bold=True, ya="bottom")
    if name:
        assert 1 <= len(name.split()) <= 6 and name[0].isupper() and not name.endswith("."), name
        cv.text(x + 9.5, y - 0.4, name, size="heading", bold=True, ya="bottom")


def tint(hexc: str, f: float) -> str:
    """Mix a colour toward white (f > 0) or black (f < 0), for the light and dark members of one class colour."""
    rgb = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
    tgt = 255 if f > 0 else 0
    return "#%02X%02X%02X" % tuple(round(c + (tgt - c) * abs(f)) for c in rgb)


def colour(role: str) -> str:
    """Colour roles recorded in the source data, resolved against the style module at draw time. The three gold
    roles are tints of the profile's foundation-model colour."""
    base = {"eye": st.EYE, "blood": st.BLOOD, "shared": st.SHARED, "fm": st.FM, "grey": st.GREY, "slate": st.SLATE,
            "ink": st.INK}
    if role in base:
        return base[role]
    return {"gold_light": tint(st.FM, 0.40), "gold_dark": tint(st.FM, -0.45), "gold_mid": tint(st.FM, -0.20)}[role]


def values(sd: dict) -> dict:
    """Rows by key, as dicts, from the JSON form of a record."""
    out = {}
    for p in sd["panels"].values():
        cols = p["columns"]
        for r in p["rows"]:
            d = dict(zip(cols, r))
            out[d["key"]] = d
    return out


def tri(row: dict) -> tuple:
    return row["estimate"], row["ci_low"], row["ci_high"]

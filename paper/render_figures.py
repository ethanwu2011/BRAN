"""Render the frozen, disclosure-safe BRAN paper figure scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle


HERE = Path(__file__).resolve().parent


def _draw(scene: dict, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    width, height = scene["width"], scene["height"]
    rendered = []
    for figure in scene["figures"]:
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=150, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.axis("off")
        for item in figure["items"]:
            kind = item["kind"]
            if kind == "text":
                x = item["x"]
                if item["align"] == "center":
                    x += item["w"] / 2
                elif item["align"] == "right":
                    x += item["w"]
                ax.text(
                    x, item["y"], item["text"], fontsize=item["size"] * 0.72,
                    color=item["color"], fontfamily=scene["font"],
                    weight="bold" if item["bold"] else "normal", ha=item["align"], va="top",
                    linespacing=1.15, style="italic" if item["italic"] else "normal",
                )
            elif kind == "line":
                ax.plot(
                    [item["x"], item["x2"]], [item["y"], item["y2"]],
                    color=item["color"], lw=item["width"] * 0.72,
                    ls=(0, (4, 3)) if item["dash"] else "-",
                )
            elif kind in ("rect", "ellipse"):
                cls = Rectangle if kind == "rect" else Ellipse
                args = ((item["x"], item["y"]), item["w"], item["h"])
                if kind == "ellipse":
                    args = ((item["x"] + item["w"] / 2, item["y"] + item["h"] / 2), item["w"], item["h"])
                ax.add_patch(cls(
                    *args, facecolor=item["fill"], edgecolor=item["stroke"],
                    lw=item["width"] * 0.72, ls=(0, (4, 3)) if item.get("dash") else "-",
                ))
            elif kind == "polygon":
                ax.add_patch(Polygon(item["points"], closed=True, facecolor=item["fill"], edgecolor=item["stroke"]))
            elif kind == "table":
                y = item["y"]
                for row_index, row in enumerate(item["rows"]):
                    x = item["x"]
                    for column, cell in enumerate(row):
                        color = item["colors"].get(str(column), "#202020")
                        ax.text(x + 5, y + 7, str(cell), fontfamily=scene["font"],
                                fontsize=item["size"] * 0.72, va="top", color=color,
                                weight="bold" if row_index == 0 else "normal")
                        x += item["widths"][column]
                    if row_index == 0:
                        ax.plot([item["x"], item["x"] + item["w"]],
                                [y + item["rh"] - 2] * 2, color="#202020", lw=0.6)
                    y += item["rh"]
                ax.plot([item["x"], item["x"] + item["w"]], [y] * 2, color="#DFDFDF", lw=0.5)
            else:
                plt.close(fig)
                raise ValueError(f"Unsupported scene item kind {kind!r}")
        for extension in ("png", "svg"):
            path = output / f"figure_{figure['number']}.{extension}"
            fig.savefig(path, dpi=150, facecolor="white")
            rendered.append(path)
        plt.close(fig)
    return rendered


def render_scene(scene: dict, output: Path | str) -> list[Path]:
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": [scene.get("font", "Arial"), "DejaVu Sans"]})
    if len(scene.get("figures", [])) != 6:
        raise ValueError("Expected a six-figure scene")
    source_ids = set(scene.get("sources", {})) | set(scene.get("extra_sources", {}))
    for figure in scene["figures"]:
        unknown = set(figure["sources"]) - source_ids
        if unknown:
            raise ValueError(f"Figure {figure['number']} has unknown source IDs {sorted(unknown)}")
    return _draw(scene, Path(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path, help="caller-supplied six-figure JSON scene")
    parser.add_argument("--output", required=True, type=Path, help="directory for six PNG and six SVG files")
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    paths = render_scene(scene, args.output)
    print(f"Rendered {len(paths) // 2} figures to {args.output}")


if __name__ == "__main__":
    main()

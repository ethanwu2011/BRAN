import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
sys.path.insert(0, str(PAPER))
import render_figures


def synthetic_scene():
    items = [
        {"kind": "text", "x": 40, "y": 35, "w": 500, "h": 30,
         "text": "Synthetic illustration only", "size": 20, "color": "#202020",
         "bold": True, "align": "left", "italic": False},
        {"kind": "line", "x": 70, "y": 130, "x2": 240, "y2": 130,
         "color": "#367E91", "width": 2, "dash": False},
        {"kind": "rect", "x": 80, "y": 160, "w": 110, "h": 45,
         "fill": "#F4F4F4", "stroke": "#367E91", "width": 1, "dash": False},
        {"kind": "ellipse", "x": 230, "y": 160, "w": 40, "h": 40,
         "fill": "#FFFFFF", "stroke": "#9980AE", "width": 1, "dash": False},
        {"kind": "polygon", "points": [[300, 160], [330, 190], [300, 220]],
         "fill": "#D5A25A", "stroke": "#D5A25A"},
        {"kind": "table", "x": 80, "y": 250, "w": 220, "h": 56,
         "widths": [110, 110], "rh": 28,
         "rows": [["Mock field", "Mock value"], ["Synthetic", "Example"]],
         "size": 13, "colors": {}},
    ]
    return {
        "width": 1200,
        "height": 900,
        "font": "Arial",
        "sources": {"synthetic_fixture": {"basename": "generated-in-test", "sha256": "0" * 64}},
        "extra_sources": {},
        "figures": [
            {"number": number, "title": f"Synthetic demo {number}",
             "sources": ["synthetic_fixture"], "items": items}
            for number in range(1, 7)
        ],
    }


class PaperFigureTests(unittest.TestCase):
    def test_synthetic_scene_has_six_figures_with_resolved_source_bindings(self):
        scene = synthetic_scene()
        self.assertEqual(len(scene["figures"]), 6)
        self.assertEqual([figure["number"] for figure in scene["figures"]], list(range(1, 7)))
        source_ids = set(scene["sources"])
        for figure in scene["figures"]:
            self.assertTrue(figure["sources"])
            self.assertLessEqual(set(figure["sources"]), source_ids)

    def test_renderer_code_only_package_has_no_scene_or_result_manifest(self):
        self.assertFalse((PAPER / "figure_scene.json").exists())
        self.assertFalse((PAPER / "evidence_manifest.json").exists())
        self.assertFalse((PAPER / "rendered").exists())

    def test_renderer_writes_six_synthetic_png_and_svg_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = render_figures.render_scene(synthetic_scene(), output)
            self.assertEqual(len(paths), 12)
            self.assertEqual({path.suffix for path in paths}, {".png", ".svg"})
            self.assertEqual(len(list(output.glob("figure_*.png"))), 6)
            self.assertEqual(len(list(output.glob("figure_*.svg"))), 6)
            for index in range(1, 7):
                image = mpimg.imread(output / f"figure_{index}.png")
                self.assertEqual(image.shape[:2], (1350, 1800))
                svg = (output / f"figure_{index}.svg").read_text(encoding="utf-8")
                self.assertIn("<svg", svg[:500])
                self.assertIn('width="864pt"', svg[:1000])

    def test_renderer_rejects_unbound_figure_source(self):
        scene = synthetic_scene()
        scene["figures"][0]["sources"] = ["unbound_source"]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unknown source IDs"):
                render_figures.render_scene(scene, temporary)


if __name__ == "__main__":
    unittest.main()

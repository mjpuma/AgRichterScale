"""The figure redraw gate is recorded, and Figures 1 and 2 carry no machine path."""
import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "docs" / "FIGURE_REDRAW_GATE.md"
ABSOLUTE = re.compile(r"""['"](/Users/|/home/|[A-Za-z]:\\)""")


class TestFigureRedrawGate(unittest.TestCase):
    def test_figure_scripts_have_no_absolute_user_path(self):
        for name in ("fig1_agrichter_scale.py", "fig2_hp_envelopes.py"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIsNone(ABSOLUTE.search(text), name)
            self.assertIn("REPO_ROOT = Path(__file__).resolve().parent.parent", text)
            self.assertIn("root_dir=REPO_ROOT", text)
            self.assertIn("REPO_ROOT / 'ancillary' / 'DisruptionCountry.xls'", text)
            self.assertIn("results_dir = REPO_ROOT / 'results'", text)

    def test_gate_lists_six_conditions_and_stays_closed(self):
        text = GATE.read_text(encoding="utf-8")
        self.assertIn("**Status: closed.**", text)
        rows = re.findall(r"^\| ([1-9]) \|", text, flags=re.MULTILINE)
        self.assertEqual(rows, ["1", "2", "3", "4", "5", "6"])
        for phrase in (
            "faostat-footprint-fix",
            "faostat-quantity-label",
            "implicit loss fraction of 1",
            "no failure count",
            "36.4",
            "18–22 percent",
            "VT 4,175 million",
            "5.2 million tonnes",
        ):
            self.assertIn(phrase, text)

    def test_registry_is_untouched(self):
        with (ROOT / "ancillary" / "historical_event_registry.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 27)


if __name__ == "__main__":
    unittest.main()

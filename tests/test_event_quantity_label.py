"""The plotted event quantity is labeled, and observed losses stay empty."""
import csv
import unittest
from pathlib import Path

from agrichter.analysis.event_quantity import QUANTITY, build_rows
from agrichter.core.constants import HISTORICAL_EVENTS

ROOT = Path(__file__).resolve().parents[1]
NOT_PLOTTED = {
    "SovietFamine1932", "UkraineWar2022", "SudanConflict2023", "SouthSudanConflict",
}


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestEventQuantityLabel(unittest.TestCase):
    def test_label_covers_the_registry_and_leaves_losses_empty(self):
        rows = build_rows()
        self.assertEqual(rows, read("ancillary/event_quantity_label_v1.csv"))
        registry = [row["event_id"] for row in read("ancillary/historical_event_registry.csv")]
        self.assertEqual([row["event_id"] for row in rows], registry)
        plotted = [row["event_id"] for row in rows if row["on_current_figures"] == "yes"]
        self.assertEqual(plotted, list(HISTORICAL_EVENTS))
        self.assertEqual(
            {row["event_id"] for row in rows if row["on_current_figures"] == "no"},
            NOT_PLOTTED)
        for row in rows:
            self.assertEqual(row["observed_affected_area_ha"], "")
            self.assertEqual(row["observed_crop_loss_fraction"], "")
            self.assertEqual(row["historical_baseline"], "")
            self.assertEqual(row["figures_redrawn"], "no")
            if row["on_current_figures"] == "yes":
                self.assertEqual(row["quantity"], QUANTITY)
                self.assertEqual(row["geography_year"], "2020")
                self.assertEqual(row["loss_fraction_applied"], "1")
            else:
                self.assertEqual(row["quantity"], "")


if __name__ == "__main__":
    unittest.main()

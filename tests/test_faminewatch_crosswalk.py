"""The faminewatch crosswalk covers the registry and does not confirm crop loss."""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestFaminewatchCrosswalk(unittest.TestCase):
    def test_one_row_per_registry_case_and_no_crop_loss(self):
        registry = [row["event_id"] for row in read("ancillary/historical_event_registry.csv")]
        cross = read("ancillary/faminewatch_registry_crosswalk_v1.csv")
        self.assertEqual([row["event_id"] for row in cross], registry)
        self.assertTrue(all(row["crop_loss_from_this_record"] == "no" for row in cross))
        absent = {row["event_id"] for row in cross if row["faminewatch_ids"] == ""}
        self.assertEqual(absent, {
            "ENSO2015_2016", "Solomon", "Vanuatu", "EastTimor", "Haiti",
            "SierraLeone", "Liberia", "Laos", "Syria", "UkraineWar2022"})


if __name__ == "__main__":
    unittest.main()

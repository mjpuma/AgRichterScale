"""Checks on the committed pre-1961 archival reconstruction. No network."""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestArchivalReconstruction(unittest.TestCase):
    def test_coverage_matches_pre_1961_registry_cases(self):
        registry = [row for row in read("ancillary/historical_event_registry.csv")
                    if int(row["year_start"]) < 1961]
        coverage = read("ancillary/archival_reconstruction_coverage_v1.csv")
        self.assertEqual([row["event_id"] for row in coverage],
                         [row["event_id"] for row in registry])
        self.assertEqual(len(coverage), 9)

    def test_china_series_is_national_and_keeps_procurement_separate(self):
        rows = [row for row in read("ancillary/archival_reconstruction_v1.csv")
                if row["event_id"] == "ChineseFamine1960"]
        output = {row["year"]: row["value"] for row in rows if row["variable"] == "output"}
        self.assertEqual(set(output), {str(year) for year in range(1954, 1967)})
        self.assertEqual(output["1958"], "200.00")
        self.assertEqual(output["1960"], "143.50")
        self.assertEqual(output["1966"], "214.00")
        procurement = {row["year"]: row["value"] for row in rows if row["variable"] == "procurement"}
        self.assertEqual(procurement["1959"], "64.12")
        self.assertNotEqual(procurement["1959"], output["1959"])
        self.assertTrue(all(row["geography"] == "China, national" for row in rows))

    def test_soviet_1921_keeps_competing_evaluations_and_regions(self):
        rows = [row for row in read("ancillary/archival_reconstruction_v1.csv")
                if row["event_id"] == "SovietFamine1921" and row["year"] == "1921"]
        values = {row["value"] for row in rows}
        self.assertIn("23.46", values)
        self.assertIn("27.67", values)
        self.assertIn("35.2", values)
        regional = {row["geography"]: row["value"] for row in rows if row["variable"] == "output_source_3a"}
        self.assertEqual(regional["Southern Producer Region"], "12.51")
        self.assertEqual(regional["USSR less SCR"], "27.67")
        self.assertNotIn("0", values)

    def test_potato_stays_off_the_cereal_axis_and_gaps_stay_empty(self):
        rows = [row for row in read("ancillary/archival_reconstruction_v1.csv")
                if row["event_id"] == "PotatoFamine"]
        self.assertTrue(all(row["crop"] == "potato" for row in rows))
        years = {row["year"] for row in rows}
        self.assertNotIn("1849", years)
        self.assertNotIn("1852", years)
        planted = {row["year"]: row["value"] for row in rows if row["variable"] == "area_planted"}
        self.assertEqual(planted["1847"], "89000")
        self.assertEqual(planted["1845"], "829875")

    def test_unreopened_cases_are_not_given_invented_rows(self):
        observed = {row["event_id"] for row in read("ancillary/archival_reconstruction_v1.csv")}
        self.assertEqual(observed, {"ChineseFamine1960", "SovietFamine1921", "PotatoFamine"})
        with (ROOT / "ancillary/archival_reconstruction_v1.csv").open(encoding="utf-8") as handle:
            header = handle.readline()
        self.assertNotIn("flag_10pct", header)
        self.assertNotIn("shortfall", header)


if __name__ == "__main__":
    unittest.main()

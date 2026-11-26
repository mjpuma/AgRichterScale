"""Checks on the committed FAOSTAT observation links. No snapshot required."""
import csv
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "links", ROOT / "scripts" / "link_faostat_observations.py")
links = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(links)


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestObservationLinks(unittest.TestCase):
    def test_registry_ids_are_unchanged_and_complete(self):
        registry = [row["event_id"] for row in read("ancillary/historical_event_registry.csv")]
        cases = read("ancillary/faostat_event_links_v1.csv")
        self.assertEqual([row["event_id"] for row in cases], registry)
        self.assertEqual([row[0] for row in links.CASES], registry)
        self.assertEqual(len(registry), 27)

    def test_pre_1961_noncereal_and_unnamed_geographies_have_no_row(self):
        cases = {row["event_id"]: row for row in read("ancillary/faostat_event_links_v1.csv")}
        no_row = {
            "GreatFamine", "Laki1783", "NoSummer", "PotatoFamine", "Drought18761878",
            "SovietFamine1921", "DustBowl", "SahelDrought2010", "ENSO2015_2016",
            "SovietFamine1932",
        }
        self.assertEqual({e for e, row in cases.items() if row["link_status"] == "no_row"}, no_row)
        for event_id in no_row:
            self.assertEqual(cases[event_id]["area_code"], "")
            self.assertEqual(cases[event_id]["n_with_production_value"], "0")

    def test_area_codes_follow_the_year_in_force(self):
        cases = {row["event_id"]: row for row in read("ancillary/faostat_event_links_v1.csv")}
        self.assertEqual(cases["ChineseFamine1960"]["area_code"], "41")
        self.assertEqual(cases["ChineseFamine1960"]["year_start"], "1961")
        self.assertEqual(cases["Ethiopia"]["area_code"], "62")
        self.assertEqual(cases["SudanConflict2023"]["area_code"], "276")
        self.assertEqual(cases["SouthSudanConflict"]["area_code"], "277")
        self.assertEqual(cases["UkraineWar2022"]["area_code"], "230")
        self.assertEqual(cases["Yemen"]["area_code"], "249")
        detail = read("ancillary/faostat_event_link_rows_v1.csv")
        used = {row["area_code"] for row in detail}
        for blocked in {"228", "206", "238", "351", "231", "104", "99"}:
            self.assertNotIn(blocked, used)
        china_years = {row["year"] for row in detail if row["event_id"] == "ChineseFamine1960"}
        self.assertEqual(china_years, {"1961"})
        self.assertFalse(any(row["event_id"] == "ENSO2015_2016" for row in detail))

    def test_detail_file_has_no_anomaly_fields(self):
        with (ROOT / "ancillary/faostat_event_link_rows_v1.csv").open(encoding="utf-8") as f:
            header = f.readline()
        self.assertNotIn("flag_10pct", header)
        self.assertNotIn("shortfall", header)
        self.assertNotIn("baseline", header)
        detail = read("ancillary/faostat_event_link_rows_v1.csv")
        cereals = [row for row in detail if row["crop"] == "cereals_primary"]
        self.assertTrue(cereals)
        self.assertTrue(all(row["presence"] == "production_value" for row in cereals))


if __name__ == "__main__":
    unittest.main()

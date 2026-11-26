"""Footprint inventory matches the workbooks and leaves losses empty."""
import csv
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from agrichter.data.events import EventsProcessor

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_event_footprints", ROOT / "scripts" / "audit_event_footprints.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestEventFootprints(unittest.TestCase):
    def test_inventory_keeps_every_workbook_place_and_no_loss(self):
        rows = audit.build_rows()
        committed = read("ancillary/event_footprints_v1.csv")
        self.assertEqual(rows, committed)
        self.assertEqual(len(rows), 225)
        self.assertTrue(all(
            row["observed_affected_area_ha"] == ""
            and row["observed_crop_loss_fraction"] == ""
            and row["mapped_2020_harvest_area_ha"] == ""
            for row in rows))
        events = {row["event_id"] for row in rows}
        self.assertEqual(len(events), 23)
        for missing in ("SovietFamine1932", "UkraineWar2022", "SudanConflict2023", "SouthSudanConflict"):
            self.assertNotIn(missing, events)

    def test_parser_drops_the_first_country_only_on_multicountry_sheets(self):
        processor = EventsProcessor.__new__(EventsProcessor)
        book = pd.ExcelFile(ROOT / "ancillary" / "DisruptionCountry.xls")
        try:
            famine = processor._process_country_sheet(pd.read_excel(book, sheet_name="GreatFamine"))
            self.assertNotIn("Belgium", famine["countries"])
            dust = processor._process_country_sheet(pd.read_excel(book, sheet_name="DustBowl"))
            self.assertEqual(dust["countries"], ["United States"])
        finally:
            book.close()
        dropped = {row["event_id"]: row["workbook_name"]
                   for row in read("ancillary/event_footprints_v1.csv")
                   if row["parser_status"] == "dropped_by_parser"}
        self.assertEqual(dropped["ENSO2015_2016"], "Australia")
        self.assertEqual(dropped["SahelDrought2010"], "Burina Faso")
        self.assertEqual(len(dropped), 7)

    def test_enso_and_nosummer_share_one_state_list(self):
        rows = read("ancillary/event_footprints_v1.csv")
        shared = [row for row in rows if row["calculator_binding"] == "receives_shared_state_list"]
        self.assertEqual({row["event_id"] for row in shared}, {"NoSummer", "ENSO2015_2016"})
        australia = [row for row in rows if row["event_id"] == "ENSO2015_2016"
                     and row["workbook_name"] == "Australia"]
        self.assertEqual(len(australia), 1)
        self.assertEqual(australia[0]["state_flag"], "1")
        self.assertEqual(australia[0]["parser_status"], "dropped_by_parser")


if __name__ == "__main__":
    unittest.main()

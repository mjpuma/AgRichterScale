"""Footprint inventory matches the workbooks and leaves losses empty."""
import csv
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from agrichter.analysis.event_calculator import EventCalculator, subdivisions_for_country
from agrichter.data.events import EventsProcessor

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_event_footprints", ROOT / "scripts" / "audit_event_footprints.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calculator():
    config = Mock()
    config.crop_type = "wheat"
    config.get_crop_indices.return_value = [1]
    grid = Mock()
    grid.is_loaded.return_value = True
    mapper = Mock()
    mapper.country_codes_mapping = {}
    return EventCalculator(config, grid, mapper)


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
        self.assertFalse(any(row["parser_status"] == "dropped_by_parser" for row in rows))
        self.assertFalse(any(row["calculator_binding"] == "receives_shared_state_list" for row in rows))

    def test_parser_keeps_the_first_country_on_every_sheet(self):
        processor = EventsProcessor.__new__(EventsProcessor)
        book = pd.ExcelFile(ROOT / "ancillary" / "DisruptionCountry.xls")
        state_book = pd.ExcelFile(ROOT / "ancillary" / "DisruptionStateProvince.xls")
        try:
            famine = processor._process_country_sheet(pd.read_excel(book, sheet_name="GreatFamine"))
            self.assertEqual(famine["countries"][0], "Belgium")
            dust = processor._process_country_sheet(pd.read_excel(book, sheet_name="DustBowl"))
            self.assertEqual(dust["countries"], ["United States"])
            enso = processor._process_country_sheet(pd.read_excel(book, sheet_name="ENSO2015_2016"))
            self.assertEqual(enso["countries"][0], "Australia")
            self.assertEqual(enso["state_flags"][0], 1.0)
            states = processor._process_state_sheet(pd.read_excel(state_book, sheet_name="ENSO2015_2016"))
        finally:
            book.close()
            state_book.close()
        event = {"state_records": states["state_records"], "state_codes": states["state_codes"]}
        australia = subdivisions_for_country(event, 14)
        united_states = subdivisions_for_country(event, 240)
        self.assertTrue(australia)
        self.assertTrue(united_states)
        self.assertFalse(set(australia) & set(united_states))
        self.assertEqual(subdivisions_for_country({"state_codes": ["Colorado"]}, 240), ["Colorado"])

    def test_state_records_are_not_shared_and_unmapped_is_not_a_harvest(self):
        calc = calculator()
        seen = {}

        def fake(country_code, state_codes):
            seen[float(country_code)] = list(state_codes)
            return 1.0, 1.0, 1

        calc.calculate_state_level_loss = fake
        result = calc.calculate_single_event("demo", {
            "country_codes": [14.0, 240.0],
            "state_flags": [1, 1],
            "state_records": [
                {"country_code": 14.0, "iso_3166_2": "AU.NSW", "name": "New South Wales"},
                {"country_code": 240.0, "iso_3166_2": "US.TX", "name": "Texas"},
            ],
        })
        self.assertEqual(seen[14.0], ["New South Wales"])
        self.assertEqual(seen[240.0], ["Texas"])
        self.assertEqual(result["mapping_failures"], [])

        seen.clear()
        missed = calc.calculate_single_event("demo", {
            "country_codes": [32.0],
            "state_flags": [1],
            "state_records": [
                {"country_code": 14.0, "iso_3166_2": "AU.NSW", "name": "New South Wales"},
            ],
        })
        self.assertEqual(missed["harvest_area_loss_ha"], 0.0)
        self.assertEqual(missed["mapping_failures"][0]["reason"], "no_subdivisions_for_country")
        self.assertEqual(seen, {})

        unmapped = calculator()
        unmapped.spatial_mapper.get_fips_from_country_code.return_value = None
        unmapped.spatial_mapper.get_country_name_from_code.return_value = None
        self.assertEqual(unmapped.calculate_country_level_loss(99999), (0.0, 0.0, 0))
        self.assertEqual(unmapped.mapping_failures[0]["reason"], "no_fips")

        measured = calculator()
        cells = pd.DataFrame({"id": [1]})
        measured.spatial_mapper.get_fips_from_country_code.return_value = "US"
        measured.grid_manager.get_grid_cells_by_iso3.return_value = (cells, cells)
        measured.grid_manager.get_crop_harvest_area.return_value = 0.0
        measured.grid_manager.get_crop_production.return_value = 0.0
        measured.config.get_unit_conversions.return_value = {"grams_per_metric_ton": 1_000_000}
        measured.config.get_caloric_content.return_value = 1
        harvest, production, count = measured.calculate_country_level_loss(240)
        self.assertEqual((harvest, production, count), (0.0, 0.0, 1))
        self.assertEqual(measured.mapping_failures, [])


if __name__ == "__main__":
    unittest.main()

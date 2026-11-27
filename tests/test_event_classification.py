"""Citation status, footprint flags, the crop-evidence rule, and ENSO overlap."""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "ancillary" / name).open(newline="", encoding="utf-8") as handle:
        return {row["event_id"]: row for row in csv.DictReader(handle)} if name != "event_overlap_v1.csv" else list(csv.DictReader(handle))


class TestEventClassification(unittest.TestCase):
    def test_citations_are_not_invented(self):
        rows = read("event_citations_v1.csv")
        self.assertEqual(len(rows), 27)
        opened = {k for k, r in rows.items() if r["citation_status"] == "catalog_reference_opened"}
        self.assertEqual(opened, {"Drought18761878", "MillenniumDrought"})
        self.assertIn("[not checked]", rows["ChineseFamine1960"]["catalog_references_with_status"])
        self.assertIn("GIEWS Country Brief: Yemen", rows["Yemen"]["sources_opened_in_repo"])
        self.assertIn("Lin and Yang", rows["ChineseFamine1960"]["sources_opened_in_repo"])
        self.assertIn("[named, not opened]", rows["NorthKorea1990s"]["catalog_references_with_status"])
        self.assertEqual({k for k, r in rows.items() if r["citation_status"] == "no_source_opened"}, {"Solomon"})
        self.assertIn("identical to", rows["Laos"]["notes"])

    def test_footprint_flags_and_empty_observed_fields(self):
        rows = read("event_footprint_uncertainty_v1.csv")
        self.assertEqual(len(rows), 27)
        for row in rows.values():
            self.assertEqual(row["observed_affected_area_ha"], "")
            self.assertEqual(row["observed_crop_loss_fraction"], "")
        self.assertIn("footprint_wider_than_registry_region", rows["PotatoFamine"]["uncertainty_flags"])
        self.assertIn("workbook_place_differs_from_registry_region", rows["SovietFamine1921"]["uncertainty_flags"])
        self.assertNotIn("workbook_place_differs", rows["Laos"]["uncertainty_flags"])
        self.assertEqual(rows["UkraineWar2022"]["footprint_basis"], "no_workbook_sheet")
        self.assertEqual(sum(r["on_current_figures"] == "yes" for r in rows.values()), 23)

    def test_only_grain_quantities_count_as_crop_disruption(self):
        rows = read("event_evidence_class_v1.csv")
        counted = {k for k, r in rows.items() if r["counts_as_crop_disruption_evidence"] == "yes"}
        self.assertEqual(counted, {"SovietFamine1921", "ChineseFamine1960", "Ethiopia", "Syria", "Yemen", "Haiti",
                                   "Laos", "EastTimor", "SierraLeone", "Liberia", "NorthKorea1990s",
                                   "MillenniumDrought", "UkraineWar2022"})
        for event_id in ("Bangladesh", "Laki1783"):
            self.assertEqual(rows[event_id]["evidence_class"], "no_usable_crop_quantity")
        self.assertEqual(rows["SahelDrought2010"]["evidence_class"], "production_context_not_attributed")
        for event_id in ("PotatoFamine", "Vanuatu", "SudanConflict2023", "Drought18761878"):
            self.assertEqual(rows[event_id]["documentary_disposition"], "confirmed")
            self.assertEqual(rows[event_id]["counts_as_crop_disruption_evidence"], "no")
        self.assertEqual(rows["ENSO2015_2016"]["evidence_class"], "production_context_not_attributed")
        self.assertEqual(rows["Solomon"]["evidence_class"], "no_crop_source_opened")

    def test_enso_overlap(self):
        rows = read("event_overlap_v1.csv")
        pairs = {(r["event_a"], r["event_b"]): r for r in rows}
        self.assertEqual(set(pairs), {("ENSO2015_2016", "EastTimor"), ("ENSO2015_2016", "Laos"), ("ENSO2015_2016", "Solomon")})
        self.assertEqual(pairs[("ENSO2015_2016", "Laos")]["shared_gdam_codes"], "123")
        self.assertEqual(pairs[("ENSO2015_2016", "Solomon")]["shared_gdam_codes"], "")
        self.assertTrue(all("do_not_sum" in r["rule"] or "keep_separate" in r["rule"] for r in rows))

    def test_primary_source_readings(self):
        with (ROOT / "ancillary" / "primary_source_readings_v1.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(r["adopted_as_event_loss"] == "no" for r in rows))
        opened = {r["event_id"] for r in rows if r["source_status"] == "opened"}
        self.assertEqual(opened, {"SahelDrought2010", "Ethiopia", "Bangladesh", "Syria", "Yemen", "Haiti", "Laos",
                                  "EastTimor", "SierraLeone", "Liberia", "Laki1783", "NorthKorea1990s",
                                  "MillenniumDrought", "ENSO2015_2016", "UkraineWar2022"})
        self.assertEqual({r["reading_id"]: r["source_status"] for r in rows}["prk-goodkind"], "blocked")
        for r in rows:
            if r["source_status"] != "opened":
                self.assertEqual(r["quantity"], "")
        by_id = {r["reading_id"]: r for r in rows}
        self.assertEqual(by_id["yem-giews-2017"]["quantity"], "480,000")
        self.assertEqual(by_id["laki-nhess-2024"]["quantity"], "")

    def test_registry_untouched(self):
        with (ROOT / "ancillary" / "historical_event_registry.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 27)


if __name__ == "__main__":
    unittest.main()

"""Synthetic tests for screening logic, no FAOSTAT network dependency."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "screen", Path(__file__).resolve().parents[1] / "scripts" / "screen_faostat_disruptions.py")
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


class TestScreening(unittest.TestCase):
    def test_trend_and_no_leakage(self):
        history = {y: 100 * 1.03**(y-1961) for y in range(1961, 1980)}
        expected, n, _, _ = s.estimate(history, 1971)
        self.assertEqual(n, 10)
        self.assertAlmostEqual(expected, 100 * 1.03**10)
        history[1971] = 1
        history[1972] = 1000000
        self.assertEqual(s.estimate(history, 1971)[0], expected)

    def test_early_missing_and_recent(self):
        self.assertIsNone(s.estimate({1961: 100}, 1962)[0])
        self.assertIsNone(s.estimate({y: 100 for y in range(1961, 1968)}, 1971)[0])
        self.assertIsNone(s.estimate({y: 100 for y in range(1961, 1971)}, 1971, [1968])[0])

    def test_signed_and_zero(self):
        for value, expected, flags in [(90, 100, [True, True, False]),
                                       (95, 100, [True, False, False]),
                                       (80, 100, [True, True, True]),
                                       (110, 100, [False, False, False]),
                                       (0, 100, [True, True, True]),
                                       (None, 100, [None, None, None])]:
            delta, frac, shortfall, result = s.screen(value, expected)
            self.assertEqual(list(result.values()), flags)
            if value is not None:
                self.assertEqual(delta, value-expected)
                self.assertEqual(shortfall, max(0, expected-value))

    def test_panel_missing_flag_and_aggregate(self):
        entities = {1: "Example", 351: "China", 5000: "World"}
        active = {a: set(range(1961, 1972)) for a in entities}
        cells = {}
        for y in range(1961, 1972):
            cells[1, 15, y, "production"] = dict(value=100 if y < 1971 else 0,
                                                    unit="t", flag="A", note="")
        panel = list(s.build_panel(entities, active, cells, 1971, {}))
        self.assertEqual({r["area_code"] for r in panel}, {1})
        row = next(r for r in panel if r["year"] == 1971 and r["item_code"] == 15)
        self.assertTrue(row["flag_20pct"])
        cells[1, 15, 1971, "production"]["flag"] = "M"
        row = next(r for r in s.build_panel(entities, active, cells, 1971, {})
                   if r["year"] == 1971 and r["item_code"] == 15)
        self.assertEqual(row["status"], "missing_production")
        self.assertIsNone(row["flag_10pct"])
        self.assertEqual(row["production_value"], 0)

    def test_parse_synthetic_archive(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"qcl.zip"
            header = "Area Code,Area,Item Code,Item,Element,Year,Unit,Value,Flag,Note\n"
            data = "".join(f"1,Example,F{i},Crop,Production,1971,t,100,A,\n" for i in s.CROPS)
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("Production_All_Data_(Normalized).csv", header+data)
            entities, active, cells, member = s.parse(path, 1971)
            self.assertEqual(len(cells), 4)
            self.assertEqual(active[1], {1971})
            self.assertEqual(cells[1, 1717, 1971, "production"]["value"], 100)


if __name__ == "__main__":
    unittest.main()

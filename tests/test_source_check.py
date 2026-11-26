"""Opened crop sentences stay out of the event-loss column."""
import csv
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decline(reading):
    pairs = re.findall(r"(192\d|1930|1934) (\d+)", reading)
    values = {year: int(amount) for year, amount in pairs}
    baseline_years = [str(year) for year in range(1921, 1931)]
    baseline = [values[year] for year in baseline_years]
    return 100.0 * (1 - values["1934"] / (sum(baseline) / len(baseline)))


class TestSourceCheck(unittest.TestCase):
    def test_nothing_is_adopted_as_an_event_loss(self):
        rows = read("ancillary/source_check_v1.csv")
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["adopted_as_event_loss"] == "no" for row in rows))
        self.assertNotIn("flag_10pct", rows[0])
        closed = {row["check_id"] for row in rows if row["source_status"] == "not_opened"}
        self.assertEqual(closed, {"gf-jordan", "pf-bourke", "nk-goodkind", "ns-stommel"})
        by_id = {row["check_id"]: row for row in rows}
        self.assertEqual(by_id["db-wheat"]["place"], "United States")
        self.assertEqual(by_id["db-corn"]["place"], "United States")
        self.assertAlmostEqual(decline(by_id["db-wheat"]["reading"]), 36.37, places=2)
        self.assertAlmostEqual(decline(by_id["db-corn"]["reading"]), 47.80, places=2)
        self.assertIn("does not say production fell 86 percent", by_id["pf-cso"]["reading"])
        self.assertIn("656,014 ha", by_id["pf-cso"]["reading"])
        self.assertIn("Not cereal tonnes", by_id["vu-pdna"]["reading"])
        self.assertIn("National production", by_id["sd-cfsam"]["reading"])
        self.assertIn("not 30 percent of grain crops lost", by_id["nk-lee"]["reading"])


if __name__ == "__main__":
    unittest.main()

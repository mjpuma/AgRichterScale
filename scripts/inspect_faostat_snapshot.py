#!/usr/bin/env python3
"""Inspect an existing FAOSTAT snapshot before choosing the end-year cutoff."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from screen_faostat_disruptions import CROPS, METRICS, code, open_source, sha256


def inspect(path, encoding):
    counts = Counter()
    flags = Counter()
    labels = defaultdict(set)
    areas = defaultdict(set)
    units = defaultdict(set)
    with open_source(path, encoding) as (f, member):
        reader = csv.DictReader(f)
        required = {"Area Code", "Area", "Item Code", "Item", "Element",
                    "Year", "Unit", "Value", "Flag"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected QCL header: {reader.fieldnames}")
        for row in reader:
            item = code(row["Item Code"])
            metric = METRICS.get(row["Element"])
            if item not in CROPS or metric is None:
                continue
            y, a = int(row["Year"]), code(row["Area Code"])
            labels[item].add(row["Item"])
            units[item, metric].add(row["Unit"])
            counts[y, item, metric] += 1
            flags[item, metric, row["Flag"]] += 1
            if metric == "production" and row["Value"].strip() and row["Flag"] != "M":
                areas[y, item].add(a)
    if not counts:
        raise ValueError("No selected QCL rows found")
    years = sorted({y for y, i, m in counts})
    return {"filename": path.name, "sha256": sha256(path), "member": member,
            "encoding": encoding, "year_min": min(years), "year_max": max(years),
            "item_labels": {i: sorted(v) for i,v in labels.items()},
            "units": [{"item": i, "metric": m, "units": sorted(v)}
                      for (i,m),v in sorted(units.items())],
            "flags": [{"item": i, "metric": m, "flag": f, "rows": n}
                      for (i,m,f),n in sorted(flags.items())],
            "annual_coverage": [
                {"year": y, "crop": name,
                 "production_rows": counts[y,i,"production"],
                 "nonmissing_production_area_codes_including_aggregates": len(areas[y,i]),
                 "harvested_area_rows": counts[y,i,"harvested_area"],
                 "yield_rows": counts[y,i,"yield"]}
                for y in years for i,name in CROPS.items()],
            "warning": "Maximum year is not a verified complete-year cutoff; counts include aggregates. Review coverage, units, labels and release provenance."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, nargs="?")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--out", type=Path, default=Path("results/faostat_snapshot_audit.json"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "FoodShockCascadeModel" / "inputs"
    candidates = [sibling/"faostat_bulk_cache"/"production.zip",
                  sibling/"Production_Crops_Livestock_E_All_Data_(Normalized).csv"]
    path = args.source or next((p for p in candidates if p.is_file()), None)
    if path is None:
        p.error("No sibling repository snapshot found; supply original ZIP or CSV path")
    audit = inspect(path, args.encoding)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
        f.write("\n")
    print(f"Source: {path}")
    print(f"Coverage: {audit['year_min']}–{audit['year_max']}; see {args.out}")
    print(audit["warning"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the event points that Figures 1 and 2 plot, using the agrichter code of one tree.

Run it once on a checkout of the pre-fix parser and once on this repository, against
the same SPAM 2020 files, to see which points the parser fix moved. The values are
SPAM 2020 harvested area and production inside the listed places, counted in full
(implicit loss fraction 1). They are not observed losses.

    python scripts/compare_event_points.py --tree /path/to/old/checkout --out before.csv
    python scripts/compare_event_points.py --tree . --out after.csv --failures failures.csv
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

CROPS = ["allgrain", "wheat", "maize", "rice"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, help="checkout whose agrichter package is used")
    parser.add_argument("--spam-root", default=None, help="folder holding the SPAM 2020 directories (default: this repository). Ancillary tables always come from this repository")
    parser.add_argument("--out", required=True)
    parser.add_argument("--failures", default=None)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    repo = Path(__file__).resolve().parent.parent
    spam_root = Path(args.spam_root).resolve() if args.spam_root else repo
    sys.path.insert(0, str(tree))
    logging.basicConfig(level=logging.WARNING)

    import pandas as pd
    from agrichter.core.config import Config
    from agrichter.data.grid_manager import GridDataManager
    from agrichter.data.spatial_mapper import SpatialMapper
    from agrichter.data.events import EventsProcessor
    from agrichter.analysis.event_calculator import EventCalculator

    country_sheets = pd.read_excel(repo / "ancillary" / "DisruptionCountry.xls", sheet_name=None, engine="xlrd")
    state_sheets = pd.read_excel(repo / "ancillary" / "DisruptionStateProvince.xls", sheet_name=None, engine="xlrd")

    rows, failures = [], []
    for crop in CROPS:
        config = Config(crop_type=crop, root_dir=repo, use_dynamic_thresholds=True)
        # The old tree has no SPAM root option, so point only the SPAM files elsewhere.
        for key, folder in (("production", "production"), ("harvest_area", "harvested_area"), ("yield", "yield")):
            name = config.data_files[key].name
            config.data_files[key] = spam_root / f"spam2020V2r0_global_{folder}" / f"spam2020V2r0_global_{folder}" / name
        grid = GridDataManager(config)
        grid.load_spam_data()
        events = EventsProcessor(config).process_event_sheets({"country": country_sheets, "state": state_sheets})
        mapper = SpatialMapper(config, grid)
        mapper.load_country_codes_mapping()
        calc = EventCalculator(config, grid, mapper)
        df = calc.calculate_all_events(events)
        for _, r in df.iterrows():
            name = r["event_name"]
            ev = events.get(name, {})
            stored = getattr(calc, "event_results", {}).get(name, {})
            event_failures = stored.get("mapping_failures", None)
            rows.append({
                "crop": crop,
                "event_name": name,
                "country_codes": ";".join(str(int(c)) for c in ev.get("country_codes", [])),
                "n_countries": len(ev.get("country_codes", [])),
                "harvest_area_2020_ha": r["harvest_area_loss_ha"],
                "production_2020_kcal": r["production_loss_kcal"],
                "magnitude": r["magnitude"],
                "grid_cells_count": r["grid_cells_count"],
                "mapping_failures": "" if event_failures is None else len(event_failures),
            })
            for f in event_failures or []:
                failures.append({"crop": crop, "event_name": name, **f})

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.failures:
        with open(args.failures, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["crop", "event_name", "country_code", "reason"])
            writer.writeheader()
            writer.writerows(failures)


if __name__ == "__main__":
    main()

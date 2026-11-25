#!/usr/bin/env python3
"""Prospective FAOSTAT QCL country/crop/year screening, not event attribution."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

URL = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
CROPS = {15: "wheat", 27: "rice_paddy", 56: "maize", 1717: "cereals_primary"}
METRICS = {"Production": "production", "Production Quantity": "production",
           "Area harvested": "harvested_area", "Yield": "yield"}
THRESHOLDS = (5, 10, 20)


def code(s):
    return int(str(s).strip().lstrip("Ff").strip("'"))


def area_class(a):
    # Provisional FAO code rule; membership audit required before inference.
    return "aggregate" if a >= 5000 or a == 351 else "reporting_entity"


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(f)


def _hash(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def acquire(path):
    """Never replace a snapshot; download to a temporary file and verify ZIP."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(URL, timeout=120) as src, tmp.open("wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt FAOSTAT ZIP")
        tmp.rename(path)
    finally:
        tmp.unlink(missing_ok=True)


def parse(path, end_year, encoding="utf-8-sig"):
    """Preserve all selected source rows, including aggregate areas and missing values."""
    entities, active, cells = {}, defaultdict(set), {}
    with zipfile.ZipFile(path) as z:
        candidates = [n for n in z.namelist() if n.lower().endswith(".csv")
                      and "all_data" in n.lower() and "normalized" in n.lower()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one normalized data CSV, got {candidates}")
        member = candidates[0]
        with z.open(member) as binary, io.TextIOWrapper(binary, encoding=encoding, errors="strict", newline="") as f:
            # Some legacy releases use Latin-1; fail rather than corrupt labels.
            reader = csv.DictReader(f)
            required = {"Area Code", "Area", "Item Code", "Item", "Element",
                        "Year", "Unit", "Value", "Flag"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected QCL header: {reader.fieldnames}")
            for row in reader:
                a, year = code(row["Area Code"]), int(row["Year"])
                if not 1961 <= year <= end_year:
                    continue
                label = row["Area"]
                if a in entities and entities[a] != label:
                    raise ValueError(f"Area code {a} has conflicting labels")
                entities[a] = label
                active[a].add(year)
                item = code(row["Item Code"])
                metric = METRICS.get(row["Element"])
                if item not in CROPS or metric is None:
                    continue
                key = a, item, year, metric
                if key in cells:
                    raise ValueError(f"Duplicate QCL observation: {key}")
                raw = row["Value"].strip()
                value = float(raw) if raw else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f"Invalid nonnegative QCL value: {key}")
                if metric == "production" and row["Unit"] != "t":
                    raise ValueError(f"Unexpected production unit: {key}, {row['Unit']}")
                if metric == "harvested_area" and row["Unit"] != "ha":
                    raise ValueError(f"Unexpected area unit: {key}, {row['Unit']}")
                cells[key] = {"value": value, "unit": row["Unit"], "flag": row["Flag"],
                              "note": row.get("Note", ""), "item_label": row["Item"],
                              "element_code": row.get("Element Code", "")}
    if not cells or {k[1] for k in cells} != set(CROPS):
        raise ValueError("Selected item coverage incomplete; check FAOSTAT item codes/release")
    return entities, active, cells, member


def estimate(history, year, breaks=()):
    """Only earlier calendar years; break year is first year of new regime."""
    last_break = max((b for b in breaks if b <= year), default=1961)
    def window(n, positive=False):
        return [(t, v) for t, v in sorted(history.items())
                if max(year-n, last_break) <= t < year and v is not None
                and (v > 0 if positive else v >= 0)]
    def adequate(rows, minimum):
        return len(rows) >= minimum and rows[-1][0] >= year-2
    ten = window(10, True)
    expected = None
    if adequate(ten, 7):
        x = [t-year for t, _ in ten]
        y = [math.log(v) for _, v in ten]
        mx, my = statistics.mean(x), statistics.mean(y)
        slope = sum((a-mx)*(b-my) for a, b in zip(x, y)) / sum((a-mx)**2 for a in x)
        expected = math.exp(my-slope*mx)
    def med(n, minimum):
        rows = window(n)
        if not adequate(rows, minimum):
            return None
        v = statistics.median(v for _, v in rows)
        return v if v > 0 else None
    return expected, len(ten), med(5, 4), med(10, 7)


def screen(value, expected):
    if value is None or expected is None or expected <= 0:
        return None, None, None, {f"flag_{t}pct": None for t in THRESHOLDS}
    delta = value - expected
    frac = delta / expected
    return delta, frac, max(0, -delta), {
        f"flag_{t}pct": frac <= -t/100 + 1e-12 for t in THRESHOLDS}


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(first))
        w.writeheader()
        w.writerow(first)
        w.writerows(rows)


def read_breaks(path):
    result = defaultdict(list)
    if path:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row["reason"].strip():
                    raise ValueError("Every series break requires a reason")
                result[int(row["area_code"]), int(row["item_code"])].append(int(row["break_year"]))
    return result


def build_panel(entities, active, cells, end_year, breaks):
    for a, label in sorted(entities.items()):
        if area_class(a) != "reporting_entity":
            continue
        for item, crop in CROPS.items():
            history = {y: cells.get((a, item, y, "production"), {}).get("value")
                       for y in range(1961, end_year+1)}
            # Missing-flag rows never count as measured production.
            for y in history:
                if cells.get((a, item, y, "production"), {}).get("flag") == "M":
                    history[y] = None
            for y, value in history.items():
                expected, n, median5, median10 = estimate(history, y, breaks.get((a,item), ()))
                delta, frac, shortfall, flags = screen(value, expected)
                status = ("no_area_reporting_row" if y not in active[a] else
                          "missing_production" if value is None else
                          "insufficient_baseline" if expected is None else "assessable")
                row = {"area_code": a, "area": label, "item_code": item, "crop": crop,
                       "year": y, "status": status}
                for metric in ("production", "harvested_area", "yield"):
                    cell = cells.get((a, item, y, metric), {})
                    for k in ("value", "unit", "flag", "note"):
                        row[f"{metric}_{k}"] = cell.get(k)
                row.update(production_used_t=value, baseline_t=expected, baseline_n=n,
                           signed_anomaly_t=delta, signed_anomaly_fraction=frac,
                           shortfall_t=shortfall, **flags)
                for label2, base in (("median5", median5), ("median10", median10)):
                    _, altfrac, _, altflags = screen(value, base)
                    row[f"{label2}_baseline_t"] = base
                    row[f"{label2}_anomaly_fraction"] = altfrac
                    for k, flag in altflags.items():
                        row[f"{label2}_{k}"] = flag
                yield row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--end-year", type=int, required=True, help="Explicit reviewed release cutoff")
    p.add_argument("--release-note", required=True, help="Release edition/date and basis for cutoff")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding; use latin-1 only if verified for the release")
    p.add_argument("--breaks", type=Path, help="CSV area_code,item_code,break_year,reason")
    args = p.parse_args()
    if args.end_year < 1961:
        p.error("end-year must be >=1961")
    if args.download:
        acquire(args.archive)
    entities, active, cells, member = parse(args.archive, args.end_year, args.encoding)
    breaks = read_breaks(args.breaks)
    panel = list(build_panel(entities, active, cells, args.end_year, breaks))
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out/"source_rows.csv", (
        dict(area_code=a, area=entities[a], area_class=area_class(a), item_code=i,
             crop=CROPS[i], year=y, metric=m, **cell)
        for (a,i,y,m), cell in sorted(cells.items())))
    write_csv(args.out/"area_audit.csv", (
        dict(area_code=a, area=label, provisional_class=area_class(a),
             first_qcl_year=min(active[a]), last_qcl_year=max(active[a]),
             qcl_reporting_years=len(active[a]))
        for a,label in sorted(entities.items())))
    write_csv(args.out/"production_panel.csv", panel)
    candidates = [r for r in panel if r["flag_10pct"] is True]
    with (args.out/"candidates_10pct.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(panel[0]))
        w.writeheader()
        w.writerows(candidates)
    coverage = defaultdict(Counter)
    for row in panel:
        c = coverage[row["year"], row["crop"]]
        c[row["status"]] += 1
        for t in THRESHOLDS:
            c[f"flag_{t}pct"] += int(row[f"flag_{t}pct"] is True)
    write_csv(args.out/"coverage.csv", (
        {"year": y, "crop": crop, **{k: c[k] for k in
          ["no_area_reporting_row", "missing_production", "insufficient_baseline",
           "assessable", "flag_5pct", "flag_10pct", "flag_20pct"]}}
        for (y,crop),c in sorted(coverage.items())))
    manifest = {"protocol_version": "1.0", "source": URL, "archive": args.archive.name,
                "archive_sha256": sha256(args.archive), "archive_member": member, "encoding": args.encoding,
                "release_note": args.release_note, "start_year": 1961,
                "end_year": args.end_year, "processed_utc": datetime.now(timezone.utc).isoformat(),
                "breaks_sha256": sha256(args.breaks) if args.breaks else None,
                "area_classification": "provisional; review area_audit before inference",
                "boundary_audit": "not certified; supplied break file applied" if args.breaks else "pending",
                "counts": dict(Counter(r["status"] for r in panel)),
                "source_rows": len(cells), "panel_rows": len(panel),
                "candidates_10pct": len(candidates),
                "thresholds_pct": THRESHOLDS,
                "scientific_status": "screening only, not a validated event sample"}
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

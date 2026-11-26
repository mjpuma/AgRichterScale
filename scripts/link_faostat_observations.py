#!/usr/bin/env python3
"""Link the 27 registry cases to FAOSTAT area-crop-year production rows.

Reads the local source extract. Does not read anomaly flags, baselines, or
the 10% candidate file, and does not add or remove registry rows.
"""
import argparse
import csv
from pathlib import Path

CROPS = ((15, "wheat"), (27, "rice_paddy"), (56, "maize"), (1717, "cereals_primary"))
SNAPSHOT_LAST = 2024

# Registry order. area_code is the code in force during the linked years.
# None means this pass assigns no FAOSTAT row.
CASES = [
    ("GreatFamine", None, None, "no_row",
     "1315-1317 is before the QCL extract, and Europe is not one reporting area. No later European series is attached."),
    ("Laki1783", None, None, "no_row",
     "1783-1784 is before the QCL extract. Iceland code 99 is not used to fill those years, and Iceland/Europe is not one reporting area."),
    ("NoSummer", None, None, "no_row",
     "1816 is before the QCL extract, and Global is not a reporting area."),
    ("PotatoFamine", None, None, "no_row",
     "1845-1852 is before the QCL extract. The documented crop is potato, which is not item 15, 27, 56, or 1717. Ireland code 104 is not attached."),
    ("Drought18761878", None, None, "no_row",
     "1876-1878 is before the QCL extract, and Asia/Africa is not one reporting area."),
    ("SovietFamine1921", None, None, "no_row",
     "1921-1922 is before the QCL extract. USSR code 228 begins in 1961 and is not used to fill 1921-1922."),
    ("ChineseFamine1960", 41, (1961, 1961), "partial_1961_only",
     "1959 and 1960 have no QCL row. 1961 is on China, mainland code 41. China code 351 is an aggregate and is not used. The 1961 rows are national production, not the 1959-1960 decline."),
    ("DustBowl", None, None, "no_row",
     "1930-1936 is before the QCL extract. The Great Plains is not a reporting area, and United States code 231 is not attached."),
    ("SahelDrought2010", None, None, "no_row",
     "2010-2011 is inside the extract, and no FAOSTAT area is named Sahel. This pass does not assign a country set."),
    ("MillenniumDrought", 10, (2000, 2009), "reporting_area_rows",
     "Australia code 10 for 2000-2009. These are national production rows for the registry label. They are not a drought-footprint loss."),
    ("NorthKorea1990s", 116, (1994, 1998), "reporting_area_rows",
     "Democratic People's Republic of Korea code 116 for 1994-1998. National production rows, not a measured flood or policy loss."),
    ("ENSO2015_2016", None, None, "no_row",
     "Global tropics is not a reporting area. Solomon, East Timor, and Laos stay on their own rows and are not attached to this parent."),
    ("Solomon", 25, (2015, 2016), "reporting_area_rows",
     "Solomon Islands code 25 for 2015-2016. These rows are not added to the ENSO parent."),
    ("Vanuatu", 155, (2015, 2015), "reporting_area_rows",
     "Vanuatu code 155 for 2015. The documented impact is a monetary crop-sector total, not these cereal tonnes. The cereal rows are the country-year series only."),
    ("EastTimor", 176, (2015, 2016), "reporting_area_rows",
     "Timor-Leste code 176 for 2015-2016. The code already exists for those years. These rows are not added to the ENSO parent."),
    ("Haiti", 93, (2010, 2016), "reporting_area_rows",
     "Haiti code 93 for 2010-2016. The registry case combines an earthquake and a hurricane, and these national rows do not separate them."),
    ("SierraLeone", 197, (1991, 2002), "reporting_area_rows",
     "Sierra Leone code 197 for 1991-2002. A civil war is not itself a crop loss. These are the national production rows for those years."),
    ("Liberia", 123, (1989, 2003), "reporting_area_rows",
     "Liberia code 123 for 1989-2003. A civil war is not itself a crop loss. These are the national production rows for those years."),
    ("Yemen", 249, (2015, 2024), "reporting_area_rows",
     "Yemen code 249 for 2015-2024. The code does not change in 1990. The registry end is open, so the link stops at the snapshot year 2024. There is no 2025 row."),
    ("Ethiopia", 62, (1983, 1985), "reporting_area_rows",
     "Ethiopia PDR code 62 for 1983-1985. Ethiopia code 238 and Eritrea begin in 1993 and are not used."),
    ("Laos", 120, (2015, 2016), "reporting_area_rows",
     "Lao People's Democratic Republic code 120 for 2015-2016. These rows are not added to the ENSO parent."),
    ("Bangladesh", 16, (1974, 1974), "reporting_area_rows",
     "Bangladesh code 16 for 1974. The code is the reporting series from 1961, including the years before independence."),
    ("Syria", 212, (2011, 2024), "reporting_area_rows",
     "Syrian Arab Republic code 212 for 2011-2024. The registry end is open, so the link stops at the snapshot year 2024. There is no 2025 row."),
    ("SovietFamine1932", None, None, "no_row",
     "1932-1933 is before the QCL extract. USSR code 228 begins in 1961 and is not used. This case is not SovietFamine1921."),
    ("UkraineWar2022", 230, (2022, 2024), "reporting_area_rows",
     "Ukraine code 230 for 2022-2024. The code begins in 1992, so USSR code 228 is not used. The registry end is open and the snapshot has no 2025 row. The 2025 figure in the search log is not a FAOSTAT row."),
    ("SudanConflict2023", 276, (2023, 2024), "reporting_area_rows",
     "Sudan code 276 for 2023-2024. Sudan (former) code 206 ends in 2011 and is not used. The snapshot has no 2025 row. The 2025 figure in the search log is not a FAOSTAT row."),
    ("SouthSudanConflict", 277, (2013, 2024), "reporting_area_rows",
     "South Sudan code 277 for 2013-2024. The code begins in 2012, so the event window is on the successor code. Sudan (former) code 206 is not used. The snapshot has no 2025 row."),
]

EXPECTED_LABEL = {
    10: "Australia",
    16: "Bangladesh",
    25: "Solomon Islands",
    41: "China, mainland",
    62: "Ethiopia PDR",
    93: "Haiti",
    116: "Democratic People's Republic of Korea",
    120: "Lao People's Democratic Republic",
    123: "Liberia",
    155: "Vanuatu",
    176: "Timor-Leste",
    197: "Sierra Leone",
    212: "Syrian Arab Republic",
    230: "Ukraine",
    249: "Yemen",
    276: "Sudan",
    277: "South Sudan",
}


def load_production(path):
    """Production element only. Anomaly columns are not in this extract."""
    found = {}
    labels = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"area_code", "area", "item_code", "crop", "year", "metric", "value", "unit", "flag"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"Unexpected source header: {reader.fieldnames}")
        forbidden = [name for name in reader.fieldnames if "flag_" in name or name.startswith("median")]
        if forbidden:
            raise SystemExit(f"Refusing a file that already carries anomaly fields: {forbidden}")
        for row in reader:
            if row["metric"] != "production":
                continue
            area = int(row["area_code"])
            item = int(row["item_code"])
            year = int(row["year"])
            if area not in labels:
                labels[area] = row["area"]
            elif labels[area] != row["area"]:
                raise SystemExit(f"Area {area} has two labels")
            key = area, item, year
            if key in found:
                raise SystemExit(f"Duplicate production row: {key}")
            if row["unit"] not in {"", "t"}:
                raise SystemExit(f"Unexpected unit for {key}: {row['unit']}")
            found[key] = row
    return found, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()
    if "candidates" in args.source.name:
        raise SystemExit("Refusing to read the candidate flag file")
    found, labels = load_production(args.source)
    for code, label in EXPECTED_LABEL.items():
        if labels.get(code) != label:
            raise SystemExit(f"Area {code} is {labels.get(code)!r}, expected {label!r}")

    case_rows = []
    detail = []
    for event_id, area, window, status, reason in CASES:
        if status == "no_row":
            if area is not None or window is not None:
                raise SystemExit(f"{event_id} is no_row and must not name an area")
            case_rows.append({
                "event_id": event_id,
                "link_status": "no_row",
                "area_code": "",
                "area": "",
                "year_start": "",
                "year_end": "",
                "n_crop_years": 0,
                "n_with_production_value": 0,
                "n_qcl_missing": 0,
                "n_no_production_row": 0,
                "reason": reason,
            })
            continue
        y0, y1 = window
        if y0 < 1961 or y1 > SNAPSHOT_LAST or y0 > y1:
            raise SystemExit(f"{event_id} window {window} is outside the snapshot")
        if labels[area] != EXPECTED_LABEL[area]:
            raise SystemExit(event_id)
        n_value = n_missing = n_absent = 0
        for year in range(y0, y1 + 1):
            for item, crop in CROPS:
                src = found.get((area, item, year))
                if src is None:
                    n_absent += 1
                    presence, value, flag = "no_production_row", "", ""
                elif src["value"] == "":
                    if src["flag"] != "M":
                        raise SystemExit(f"{event_id} {area} {item} {year} has an empty value without flag M")
                    n_missing += 1
                    presence, value, flag = "qcl_missing", "", "M"
                else:
                    n_value += 1
                    presence, value, flag = "production_value", src["value"], src["flag"]
                detail.append({
                    "event_id": event_id,
                    "link_status": status,
                    "area_code": area,
                    "area": labels[area],
                    "item_code": item,
                    "crop": crop,
                    "year": year,
                    "production_t": value,
                    "production_unit": "t" if value != "" else "",
                    "qcl_flag": flag,
                    "presence": presence,
                    "role": "national_or_territorial_production_not_an_event_loss",
                })
        case_rows.append({
            "event_id": event_id,
            "link_status": status,
            "area_code": area,
            "area": labels[area],
            "year_start": y0,
            "year_end": y1,
            "n_crop_years": (y1 - y0 + 1) * len(CROPS),
            "n_with_production_value": n_value,
            "n_qcl_missing": n_missing,
            "n_no_production_row": n_absent,
            "reason": reason,
        })

    write(args.cases, case_rows)
    write(args.rows, detail)
    print(f"cases {len(case_rows)} detail {len(detail)}")


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inventory the legacy disruption workbooks without rerunning SPAM.

The country parser drops the first data row when a sheet has more than one
country. This inventory keeps every workbook row and records that drop.
Observed loss fields are left empty.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

from agrichter.data.events import EventsProcessor

ROOT = Path(__file__).resolve().parents[1]
COUNTRY_XLS = ROOT / "ancillary" / "DisruptionCountry.xls"
STATE_XLS = ROOT / "ancillary" / "DisruptionStateProvince.xls"
CONVERT_XLS = ROOT / "ancillary" / "CountryCode_Convert.xls"
REGISTRY = ROOT / "ancillary" / "historical_event_registry.csv"

FIELDS = [
    "event_id", "place_level", "source_workbook", "source_sheet",
    "workbook_name", "gdam_code", "gdam_name_in_convert_table", "code_check",
    "state_flag", "iso_3166_2", "parser_status", "calculator_binding",
    "registry_year_start", "registry_year_end", "crop",
    "observed_affected_area_ha", "observed_crop_loss_fraction",
    "mapped_2020_harvest_area_ha", "spatial_basis", "notes",
]
SPATIAL_BASIS = "Workbook place on the GDAM code in the sheet. SPAM 2020 exposure was not recalculated."


def gdam_table():
    frame = pd.read_excel(CONVERT_XLS, header=None).iloc[1:]
    by_name = {}
    by_code = {}
    for _, row in frame.iterrows():
        name = str(row.iloc[0]).strip()
        code = pd.to_numeric(row.iloc[7], errors="coerce")
        if pd.isna(code):
            continue
        code = int(code)
        by_name[name.lower()] = (name, code)
        by_code.setdefault(code, name)
    return by_name, by_code


def code_check(workbook_name, code, by_name, by_code):
    key = str(workbook_name).strip().lower()
    named = by_name.get(key)
    canonical = by_code.get(int(code))
    if named and named[1] == int(code):
        return canonical, "name_and_code_match"
    if canonical:
        return canonical, "code_matches_different_name"
    return "", "code_not_in_gdam_column"


def registry_years():
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        return {row["event_id"]: (row["year_start"], row["year_end"])
                for row in csv.DictReader(handle)}


def blank(event_id, years, **kwargs):
    row = {field: "" for field in FIELDS}
    row.update(event_id=event_id, registry_year_start=years[0], registry_year_end=years[1],
               spatial_basis=SPATIAL_BASIS, crop="")
    row.update(kwargs)
    return {key: "" if value is None else str(value) for key, value in row.items()}


def build_rows():
    by_name, by_code = gdam_table()
    years = registry_years()
    processor = EventsProcessor.__new__(EventsProcessor)
    country_book = pd.ExcelFile(COUNTRY_XLS)
    state_book = pd.ExcelFile(STATE_XLS)
    if set(country_book.sheet_names) != set(state_book.sheet_names):
        raise SystemExit("Country and state workbooks do not name the same events")
    rows = []
    for event_id in country_book.sheet_names:
        raw = pd.read_excel(country_book, sheet_name=event_id, header=None).iloc[1:]
        loaded = pd.read_excel(country_book, sheet_name=event_id)
        kept = processor._process_country_sheet(loaded)["countries"]
        flags = []
        countries = []
        for _, item in raw.iterrows():
            if pd.isna(item.iloc[0]):
                continue
            name = str(item.iloc[0])
            code = int(item.iloc[1])
            flag = int(item.iloc[2])
            countries.append((name, code, flag))
            flags.append(flag)
        n_flagged = sum(flag == 1 for _, _, flag in countries)
        for index, (name, code, flag) in enumerate(countries):
            canonical, check = code_check(name, code, by_name, by_code)
            if len(countries) > 1 and index == 0:
                status = "dropped_by_parser"
            else:
                status = "kept_by_parser"
            if name not in kept and status != "dropped_by_parser":
                raise SystemExit(f"{event_id} parser result does not match the first-row rule for {name}")
            if flag == 1 and n_flagged > 1:
                binding = "receives_shared_state_list"
            elif flag == 1:
                binding = "state_list_for_this_country_only"
            else:
                binding = "country_level_total"
            note = ""
            if status == "dropped_by_parser":
                note = "The loaded sheet has more than one country, and the parser skips its first data row."
            if flag == 1 and status == "dropped_by_parser":
                note += " This country is also marked for state processing, so its own row never reaches the calculator."
            rows.append(blank(
                event_id, years[event_id], place_level="country",
                source_workbook="ancillary/DisruptionCountry.xls", source_sheet=event_id,
                workbook_name=name.strip(), gdam_code=code,
                gdam_name_in_convert_table=canonical, code_check=check,
                state_flag=flag, parser_status=status, calculator_binding=binding,
                notes=note.strip()))
        state_raw = pd.read_excel(state_book, sheet_name=event_id, header=None).iloc[1:]
        for _, item in state_raw.iterrows():
            if pd.isna(item.iloc[2]):
                continue
            iso = "" if pd.isna(item.iloc[0]) else str(item.iloc[0])
            code = int(item.iloc[1])
            name = str(item.iloc[2]).strip()
            canonical = by_code.get(code, "")
            check = "code_found_in_gdam_column" if canonical else "code_not_in_gdam_column"
            binding = ("name_enters_shared_list" if n_flagged > 1 else "applied_with_its_country_only")
            rows.append(blank(
                event_id, years[event_id], place_level="subdivision",
                source_workbook="ancillary/DisruptionStateProvince.xls", source_sheet=event_id,
                workbook_name=name, gdam_code=code, gdam_name_in_convert_table=canonical,
                code_check=check, iso_3166_2=iso, parser_status="kept_name_only",
                calculator_binding=binding,
                notes="The parser stores the subdivision name and drops its country code. The calculator gives the whole name list to every state-flagged country."))
    country_book.close()
    state_book.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-event citation status, footprint uncertainty, crop-evidence class, and event overlap.

Reads only files already in the repository. It opens no web page, adds no citation,
fills no observed field, and does not edit ancillary/historical_event_registry.csv.

Writes:
    ancillary/event_citations_v1.csv
    ancillary/event_footprint_uncertainty_v1.csv
    ancillary/event_evidence_class_v1.csv
    ancillary/event_overlap_v1.csv
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANC = ROOT / "ancillary"
FAILURES = ROOT / "results" / "redraw_2026-11" / "mapping_failures_after.csv"


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def years(row):
    start = int(row["year_start"])
    end = 2024 if row["year_end"] == "ongoing" else int(row["year_end"])
    return start, end


# Crop evidence that an opened page in this repository supports, by event. Each entry
# restates a reading already recorded in the documentary log, the source check, the
# archival reconstruction, or the evidence seed. Nothing here was read for the first time.
# measure: what the opened number is. attribution: whether the page ties it to the event.
CROP_EVIDENCE = {
    "GreatFamine": ("yield_one_estate_group", "secondary",
                    "Schuh 2020 reports Titow's Winchester yields 36 and 45 percent below average for 1314/15 and 1315/16. Titow and Jordan were not opened."),
    "Laki1783": ("qualitative", "secondary",
                 "Wieners and Halfdanarson 2024 (NHESS): haze damaged Icelandic grass; records show no evidence of harvest failure in west and central Europe in 1783. No crop quantity."),
    "NoSummer": ("qualitative", "secondary",
                 "Veale and Endfield describe 1816 in the UK qualitatively. Jefferson expected the Atlantic-states corn crop below one-third of ordinary. Stommel and Stommel were not opened."),
    "PotatoFamine": ("potato_area", "reported",
                     "CSO prints potato hectares, 656,014 in 1846 and 89,000 in 1847. Potato is not grain, and area is not production. Bourke 1960 was not opened."),
    "Drought18761878": ("district_crop_share", "secondary",
                        "Davis 2001 Table 1.2 prints percent of crop saved by Madras district, 6 to 37. Tanjore is one district in the documentary log. The Famine Commission page was not opened."),
    "SovietFamine1921": ("grain_reconstruction", "reconstructed",
                         "Wheatcroft 1980 gives three USSR-less-SCR grain evaluations for 1920-1923. His area is not the full Soviet Union and not the Volga."),
    "ChineseFamine1960": ("grain_national_series", "reported",
                          "Lin and Yang 2000 Table 1 national grain output 1954-1966, 200.00 in 1958 and 143.50 million tons in 1960. Policy and weather are not separated."),
    "DustBowl": ("grain_national_series", "not_attributed",
                 "NASS national 1934 wheat and corn are 36.4 and 47.8 percent below their 1921-1930 means. United States, not the Great Plains. The decadal wheat model is a counterfactual."),
    "SahelDrought2010": ("regional_production_report", "not_attributed",
                         "FAO Information Note 1, 6 February 2012: CILSS Sahel cereals 16.6 million tonnes in 2011/12, 25 percent below the 2010/11 record and 2 percent below five years. The aggregate is not the ten workbook countries."),
    "MillenniumDrought": ("yield_gap_counterfactual", "secondary",
                          "van Dijk 2013: dryland wheat yields 18-22 percent below average conditions, while production rose on more area. Not a production loss."),
    "NorthKorea1990s": ("contested_claim", "reported",
                        "Noland, Robinson and Wang 1999 record a government claim of 1.9 million tons of grain lost and say outside estimates were far lower. Goodkind and West were not opened."),
    "ENSO2015_2016": ("regional_production_report", "not_attributed",
                      "NHESS 2020 and an FAO outlook report southern Africa cereal and maize shortfalls. Regional, and not the whole ENSO footprint."),
    "Solomon": ("none", "none", "No crop source was opened. Searches found only qualitative 2015-2016 drought reports."),
    "Vanuatu": ("crop_damage_money", "reported",
                "Post-disaster needs assessment: crop damage and losses VT 4,175 million. Vatu, mixing food gardens and cash crops. Kava is 58 percent of losses."),
    "EastTimor": ("grain_national_series", "reported",
                  "FAO GIEWS brief, 14 December 2016: 2016 rice 60,000 tonnes, 6 percent below 2015 and 30 percent below average; maize 60,000 tonnes, 8 and 21 percent below; El Nino dry weather named."),
    "Haiti": ("grain_national_series", "reported",
              "FAO newsroom for the FAO/WFP CFSAM, 22 September 2010: 2010 cereals forecast 503,600 tonnes, 9 percent below 2009. Earthquake and late rains are named together."),
    "SierraLeone": ("grain_national_series", "reported",
                    "FAO GIEWS special report, 4 February 2000: 1999 paddy 248,220 tonnes, about 45 percent of pre-war (1990, 543,700 tonnes). Provisional estimates; war and displacement named."),
    "Liberia": ("grain_national_series", "reported",
                "FAO GIEWS missions: 1995 paddy at most 23 percent of pre-war (highly tentative); 2000 paddy 144,000 tonnes against 259,000 in 1988. No official crop data since 1989."),
    "Yemen": ("grain_national_series", "reported",
              "FAO GIEWS brief, 10 February 2017: 2016 cereals about 480,000 tonnes, 11 percent below 2015 and 37 percent below the five-year average. Conflict and July 2016 floods are named. FamineWatch IPC phases are not crop data."),
    "Ethiopia": ("grain_national_series", "reported",
                 "USDA ERS RS-85-10, July 1985: 1984 food grain 20 to 30 percent below normal, over 5 million tons. Rainfall, pests, government policy and civil war are named together."),
    "Laos": ("grain_national_series", "reported",
             "FAO GIEWS brief, 24 May 2016: 2015 rice 3.2 million tonnes, 4 percent below the 2014 record; maize 1.1 million tonnes, 5 percent below; El Nino rains named."),
    "Bangladesh": ("grain_change_source_not_named", "secondary",
                   "Dyson 2021: the end-1974 aman crop was 700,000 tons below 1973, tied to the floods. The sentence names no data source."),
    "Syria": ("grain_national_series", "reported",
              "UN News for FAO, May 2014: 2014 wheat 1.97 million tonnes, 52 percent below the 2001-2011 average. FAO 2018: 1.2 million tons. Drought and conflict are named together."),
    "SovietFamine1932": ("statement_not_reread", "none",
                         "Harvest statements of about 56 and 50 million tonnes were not re-read. Ukraine and Kazakhstan were not separated."),
    "UkraineWar2022": ("national_production_report", "not_attributed",
                       "FAO GIEWS and an FAO enterprise survey report national cereal and cultivated-area figures. Not attributed to the war alone."),
    "SudanConflict2023": ("national_production_report", "not_attributed",
                          "FAO CFSAM 2025 national cereals about 5.2 million tonnes, 22 percent below 2024. The mission cites inputs and rainfall as well as conflict."),
    "SouthSudanConflict": ("national_production_report", "not_attributed",
                           "The 2025 CFSAM smallholder cereal result in hand is an increase."),
}

# Workbook names that are the same country as the registry label.
ALIASES = {"Timor-Leste": "East Timor", "Lao People's Democratic Republic": "Laos"}

CONFLICT_WORDS = ("war", "conflict", "policy", "coercion", "displacement")
ENVIRONMENT_WORDS = ("weather", "drought", "volcan", "disease", "blight", "flood", "cyclone",
                     "earthquake", "hurricane", "niño", "nino", "environmental", "degradation")


def driver_class(text):
    text = text.lower()
    conflict = any(word in text for word in CONFLICT_WORDS)
    environment = any(word in text for word in ENVIRONMENT_WORDS)
    if conflict and environment:
        return "compound"
    if conflict:
        return "conflict_or_policy"
    if environment:
        return "environmental"
    return "unclassified"


def evidence_class(measure, attribution):
    """The rule in docs/EVENT_CLASSIFICATION.md, in code."""
    if measure in ("grain_national_series", "grain_reconstruction") and attribution in ("reported", "reconstructed"):
        return "grain_quantity_opened"
    if measure in ("potato_area", "district_crop_share", "yield_one_estate_group", "crop_damage_money"):
        return "crop_measure_opened_not_grain_tonnes"
    if measure in ("grain_national_series", "regional_production_report", "national_production_report"):
        return "production_context_not_attributed"
    if measure in ("qualitative", "yield_gap_counterfactual", "contested_claim", "statement_not_reread",
                   "grain_change_source_not_named"):
        return "no_usable_crop_quantity"
    return "no_crop_source_opened"


def surname_and_year(reference):
    name = re.match(r"\s*([^\s,(]+)", reference)
    year = re.search(r"\((\d{4})\)", reference)
    return (name.group(1).lower() if name else ""), (year.group(1) if year else "")


def cites(titles, reference):
    """A catalog reference matches an opened title only on first author and year together."""
    name, year = surname_and_year(reference)
    return bool(name and year) and any(name in t.lower() and year in t for t in titles)


def main():
    registry = read(ANC / "historical_event_registry.csv")
    catalog = {row["event_name"]: row for row in read(ANC / "food_disruptions.csv")}
    log = {row["event_id"]: row for row in read(ANC / "documentary_search_log_v1.csv")}
    checks = read(ANC / "source_check_v1.csv")
    archival = read(ANC / "archival_reconstruction_v1.csv")
    seed = read(ANC / "historical_loss_evidence_seed.csv")
    footprints = read(ANC / "event_footprints_v1.csv")
    failures = read(FAILURES) if FAILURES.exists() else []
    assert len(registry) == 27 and set(CROP_EVIDENCE) == {row["event_id"] for row in registry}

    opened = defaultdict(set)
    not_opened = defaultdict(set)
    for row in checks:
        (opened if row["source_status"] == "opened" else not_opened)[row["event_id"]].add(row["source_title"])
    for row in archival:
        opened[row["event_id"]].add(row["source_title"])
    for row in seed:
        opened[row["event_id"]].add(row["source_title"])
    readings = ANC / "primary_source_readings_v1.csv"
    for row in (read(readings) if readings.exists() else []):
        if row["source_status"] == "opened":
            opened[row["event_id"]].add(f"{row['source_title']} ({row['date_as_printed']})")
    for event_id, row in log.items():
        if row["measurement_type"] != "not_opened" and row["source_title"]:
            opened[event_id].add(row["source_title"])

    shared_refs = defaultdict(list)
    for name, row in catalog.items():
        shared_refs[row["key_references"]].append(name)

    citations = []
    for reg in registry:
        event_id = reg["event_id"]
        refs = [r.strip() for r in catalog.get(event_id, {}).get("key_references", "").split(";") if r.strip()]
        opened_titles = sorted(opened[event_id])
        ref_status = []
        for ref in refs:
            if cites(opened_titles, ref):
                ref_status.append(f"{ref} [opened]")
            elif cites(not_opened[event_id], ref):
                ref_status.append(f"{ref} [named, not opened]")
            else:
                ref_status.append(f"{ref} [not checked]")
        n_opened_refs = sum(s.endswith("[opened]") for s in ref_status)
        if n_opened_refs:
            status = "catalog_reference_opened"
        elif opened_titles:
            status = "other_source_opened"
        else:
            status = "no_source_opened"
        notes = []
        if not refs:
            notes.append("Extension with no workbook sheet and no catalog reference list.")
        key = catalog.get(event_id, {}).get("key_references", "")
        if key and len(shared_refs[key]) > 1:
            others = ", ".join(n for n in shared_refs[key] if n != event_id)
            notes.append(f"Catalog reference list is identical to {others}.")
        if event_id == "Laos":
            notes.append("The catalog references are Pacific El Niño reports; Laos is in mainland Southeast Asia.")
        citations.append({
            "event_id": event_id,
            "catalog_references_with_status": "; ".join(ref_status),
            "catalog_references_opened": n_opened_refs,
            "sources_opened_in_repo": "; ".join(opened_titles),
            "citation_status": status,
            "record": log.get(event_id, {}).get("repository_extract", ""),
            "notes": " ".join(notes),
        })

    places = defaultdict(list)
    for row in footprints:
        places[row["event_id"]].append(row)
    allgrain_failures = defaultdict(int)
    for row in failures:
        if row["crop"] == "allgrain":
            allgrain_failures[row["event_name"]] += 1
    uncertainty = []
    for reg in registry:
        event_id = reg["event_id"]
        rows = places.get(event_id, [])
        countries = [r for r in rows if r["place_level"] == "country"]
        subdivisions = [r for r in rows if r["place_level"] != "country"]
        start, _ = years(reg)
        flags = []
        if not rows:
            basis = "no_workbook_sheet"
        elif len(countries) > 1:
            basis = "multi_country"
        elif subdivisions:
            basis = "one_country_subdivisions"
        else:
            basis = "one_whole_country"
        if rows:
            if start < 1900:
                flags.append("pre_1900_event_on_2020_geography")
            if len(countries) > 10:
                flags.append("more_than_10_countries")
            single_region = reg["region_label"].strip()
            names = {r["workbook_name"].strip() for r in countries}
            if len(countries) > 1 and "/" not in single_region and single_region not in (
                    "Europe", "Global", "Global tropics", "Sahel region", "Asia/Africa", "Iceland/Europe"):
                flags.append("footprint_wider_than_registry_region")
            names |= {ALIASES.get(n, n) for n in names}
            if basis == "one_whole_country" and single_region not in names and not any(
                    single_region.lower() in n.lower() or n.lower() in single_region.lower() for n in names):
                flags.append("workbook_place_differs_from_registry_region")
            if basis == "one_whole_country":
                flags.append("whole_country_no_subdivision")
            if allgrain_failures.get(event_id):
                flags.append("allgrain_mapping_failure")
            if any(r["code_check"] == "code_matches_different_name" for r in rows):
                flags.append("workbook_label_differs_from_convert_name_code_matches")
        uncertainty.append({
            "event_id": event_id,
            "on_current_figures": "yes" if rows else "no",
            "registry_region": reg["region_label"],
            "registry_years": f"{reg['year_start']}-{reg['year_end']}",
            "footprint_basis": basis,
            "n_countries": len(countries),
            "n_subdivisions": len(subdivisions),
            "allgrain_mapping_failures": allgrain_failures.get(event_id, 0) if rows else "",
            "years_before_2020": 2020 - start,
            "uncertainty_flags": ";".join(flags),
            "n_flags": len(flags),
            "observed_affected_area_ha": "",
            "observed_crop_loss_fraction": "",
        })

    classes = []
    for reg in registry:
        event_id = reg["event_id"]
        measure, attribution, note = CROP_EVIDENCE[event_id]
        cls = evidence_class(measure, attribution)
        classes.append({
            "event_id": event_id,
            "driver_class": driver_class(reg["driver_description"]),
            "documentary_disposition": log[event_id]["disposition"],
            "crop_measure_opened": measure,
            "attribution": attribution,
            "evidence_class": cls,
            "counts_as_crop_disruption_evidence": "yes" if cls == "grain_quantity_opened" else "no",
            "basis": note,
        })

    country_codes = defaultdict(set)
    subdivision_codes = defaultdict(set)
    for row in footprints:
        if row["place_level"] == "country":
            country_codes[row["event_id"]].add(row["gdam_code"])
        else:
            subdivision_codes[row["event_id"]].add(row["iso_3166_2"])
    by_id = {row["event_id"]: row for row in registry}
    overlaps = []
    ids = sorted(country_codes)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = country_codes[a] & country_codes[b]
            if not shared:
                continue
            a0, a1 = years(by_id[a])
            b0, b1 = years(by_id[b])
            if max(a0, b0) > min(a1, b1):
                continue
            names = sorted({r["workbook_name"].strip() for r in footprints
                            if r["event_id"] == a and r["place_level"] == "country" and r["gdam_code"] in shared})
            group_a, group_b = by_id[a]["overlap_group"], by_id[b]["overlap_group"]
            parent_child = bool(group_a and group_b and group_a.split("_")[:3] == group_b.split("_")[:3])
            overlaps.append({
                "event_a": a,
                "event_b": b,
                "shared_years": f"{max(a0, b0)}-{min(a1, b1)}",
                "shared_gdam_codes": ";".join(sorted(shared, key=float)),
                "shared_places": "; ".join(names),
                "registry_overlap_groups": f"{group_a or '-'} / {group_b or '-'}",
                "rule": "parent_contains_subevent_do_not_sum" if parent_child else "same_place_same_years_do_not_sum",
            })

    # Registry subevents whose place is not on the parent's sheet still share its years.
    listed = {(o["event_a"], o["event_b"]) for o in overlaps}
    groups = defaultdict(list)
    for row in registry:
        if row["overlap_group"]:
            groups["_".join(row["overlap_group"].split("_")[:3])].append(row)
    for members in groups.values():
        parents = [r for r in members if r["overlap_group"].endswith("PARENT")]
        for parent in parents:
            for child in members:
                if child is parent or (parent["event_id"], child["event_id"]) in listed:
                    continue
                a0, a1 = years(parent)
                b0, b1 = years(child)
                overlaps.append({
                    "event_a": parent["event_id"],
                    "event_b": child["event_id"],
                    "shared_years": f"{max(a0, b0)}-{min(a1, b1)}",
                    "shared_gdam_codes": "",
                    "shared_places": "",
                    "registry_overlap_groups": f"{parent['overlap_group']} / {child['overlap_group']}",
                    "rule": "registry_subevent_place_not_on_parent_sheet_keep_separate",
                })

    write(ANC / "event_citations_v1.csv", citations, list(citations[0]))
    write(ANC / "event_footprint_uncertainty_v1.csv", uncertainty, list(uncertainty[0]))
    write(ANC / "event_evidence_class_v1.csv", classes, list(classes[0]))
    write(ANC / "event_overlap_v1.csv", overlaps, list(overlaps[0]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the pre-1961 archival reconstruction from tables re-read in this pass.

These rows are not FAOSTAT observations and are not appended to the QCL panel.
Missing years are left missing. Procurement, trade, and balance-sheet shares
are not production losses.
"""
import argparse
import csv
from pathlib import Path

# Lin and Yang 2000, Economic Journal, Table 1.
# Upstream volumes cited in that table were not opened.
CHINA = [
    # year, output_mt, procurement_mt, quota_over_output_pct, net_export_mt, per_capita_kg
    (1954, "169.52", "50.89", "30.02", "1.7", "278.5"),
    (1955, "183.74", "47.54", "25.87", "2.1", "295.5"),
    (1956, "192.75", "40.22", "20.87", "2.5", "302.8"),
    (1957, "195.05", "45.97", "23.57", "1.9", "298.7"),
    (1958, "200.00", "51.83", "25.92", "2.7", "299.0"),
    (1959, "170.00", "64.12", "37.71", "4.2", "246.7"),
    (1960, "143.50", "46.54", "32.43", "2.7", "212.7"),
    (1961, "147.50", "36.55", "24.78", "-4.5", "230.8"),
    (1962, "160.00", "32.42", "20.26", "-3.9", "243.6"),
    (1963, "170.00", "36.99", "21.76", "-4.5", "252.3"),
    (1964, "187.50", "40.14", "21.41", "-4.7", "272.6"),
    (1965, "194.53", "39.22", "20.16", "-4.0", "273.7"),
    (1966, "214.00", "41.42", "19.35", "-3.6", "291.9"),
]
CHINA_SOURCE = "Lin and Yang 2000, Table 1"
CHINA_LOCATION = "Economic Journal 110, Table 1, re-read from the journal PDF. Upstream: State Statistical Bureau 1990, pp. 12 and 26; Ministry of Agriculture Planning Bureau 1984, p. 434. Those volumes were not opened."

# Wheatcroft 1980, page labeled 478. Million tons. USSR less SCR.
SOVIET_NATIONAL = [
    # year, no_correction, half_point, july_1923
    (1920, "29.00", "34.58", "41.9"),
    (1921, "23.46", "27.67", "35.2"),
    (1922, "30.51", "36.23", "45.7"),
    (1923, "29.24", "35.89", "48.5"),
]
# Wheatcroft 1980, page labeled 474, source block 3a. Million tons.
SOVIET_REGIONAL = {
    1920: {"Northern Consumer Region": "3.79", "Southern Producer Region": "16.23",
           "Central Producer Region": "5.99", "EPR": "8.11", "USSR less SCR": "34.11"},
    1921: {"Northern Consumer Region": "4.56", "Southern Producer Region": "12.51",
           "Central Producer Region": "5.01", "EPR": "5.59", "USSR less SCR": "27.67"},
    1922: {"Northern Consumer Region": "5.00", "Southern Producer Region": "15.89",
           "Central Producer Region": "8.67", "EPR": "6.67", "USSR less SCR": "36.23"},
}
WHEAT_SOURCE = "Wheatcroft 1980 thesis"
WHEAT_NATIONAL_LOC = "Page labeled 478. Comparison of early TsSU evaluations. Geography is USSR less SCR, Wheatcroft's smaller area, not the full Soviet Union and not the Volga alone."
WHEAT_REGIONAL_LOC = "Page labeled 474, source block 3a. SCR is blank in this block and is not filled with zero. The 1920 total is 34.11, which is not the 34.58 half-point figure on page 478."

# CSO restatement of Bourke. Hectares as printed. 1844 and 1845 are retrospective.
IRELAND_AREA = [
    (1844, "785308", "retrospective_area", "pre_event", "Bourke via CSO gives 785,308 ha. Collected retrospectively."),
    (1845, "829875", "retrospective_area", "event_window", "Bourke via CSO gives 829,875 ha. Collected retrospectively."),
    (1846, "656014", "reported_area", "event_window", "Bourke via CSO gives 656,014 ha from the 1846 constabulary survey analysis."),
    (1847, "89000", "reported_area", "event_window", "CSO states 89,000 ha after the 1846 blight and a lack of seed."),
    (1848, "258000", "reported_area", "event_window", "CSO states recovery to 258,000 ha."),
    (1859, "374000", "reported_area", "after_event_window", "CSO states 374,000 ha, the peak of the state-level series. This year is after the registry window."),
]
CSO_SOURCE = "Ireland CSO, Farming Since the Famine, Potatoes section"
CSO_LOCATION = "Bourke 1959 as restated by the CSO. The CSO prints hectares. Annual potato area for 1849-1852 was not on the pages read. Potato is not a cereal."

FIELDS = ["event_id", "year", "year_role", "geography", "crop", "variable", "value", "unit",
          "measurement_class", "source_title", "source_location", "note"]


def china_role(year):
    if year < 1959:
        return "pre_event"
    if year <= 1961:
        return "event_window"
    return "after_event_window"


def soviet_role(year):
    if year < 1921:
        return "pre_event"
    if year <= 1922:
        return "event_window"
    return "after_event_window"


def row(**kwargs):
    return {key: kwargs.get(key, "") for key in FIELDS}


def observations():
    rows = []
    for year, output, procurement, ratio, trade, per_capita in CHINA:
        role = china_role(year)
        common = dict(event_id="ChineseFamine1960", year=year, year_role=role,
                      geography="China, national", crop="grain",
                      source_title=CHINA_SOURCE, source_location=CHINA_LOCATION,
                      measurement_class="reported_series_via_secondary")
        rows.append(row(**common, variable="output", value=output, unit="million_tons",
                        note="National grain output. Not a causal loss, and not the FAOSTAT cereals series."))
        rows.append(row(**common, variable="procurement", value=procurement, unit="million_tons",
                        note="State grain procurement. A separate variable from output."))
        rows.append(row(**common, variable="procurement_share_of_output", value=ratio, unit="percent",
                        note="Quota divided by output, as printed. Not a production loss."))
        rows.append(row(**common, variable="net_export", value=trade, unit="million_tons",
                        note="Negative values are net imports, as printed with a minus sign."))
        rows.append(row(**common, variable="per_capita_grain", value=per_capita, unit="kg_per_year",
                        note="Per capita grain. Not a production loss."))
    for year, plain, half, later in SOVIET_NATIONAL:
        role = soviet_role(year)
        for variable, value, klass in (
            ("output_no_correction", plain, "early_statistical_evaluation"),
            ("output_half_point_correction", half, "early_statistical_evaluation"),
            ("output_july_1923_evaluation", later, "early_statistical_evaluation"),
        ):
            rows.append(row(
                event_id="SovietFamine1921", year=year, year_role=role,
                geography="USSR less SCR", crop="grain", variable=variable, value=value,
                unit="million_tons", measurement_class=klass, source_title=WHEAT_SOURCE,
                source_location=WHEAT_NATIONAL_LOC,
                note="One of three evaluations of the same harvest. Not upgraded to a single official total."))
    for year, regions in SOVIET_REGIONAL.items():
        for geography, value in regions.items():
            rows.append(row(
                event_id="SovietFamine1921", year=year, year_role=soviet_role(year),
                geography=geography, crop="grain", variable="output_source_3a", value=value,
                unit="million_tons", measurement_class="regional_evaluation",
                source_title=WHEAT_SOURCE, source_location=WHEAT_REGIONAL_LOC,
                note="Regional piece of source block 3a. Not a Volga-only loss, and not the July 1923 national total."))
    for year, value, klass, role, note in IRELAND_AREA:
        rows.append(row(
            event_id="PotatoFamine", year=year, year_role=role, geography="Ireland",
            crop="potato", variable="area_planted", value=value, unit="ha",
            measurement_class=klass, source_title=CSO_SOURCE, source_location=CSO_LOCATION,
            note=note))
    rows.append(row(
        event_id="PotatoFamine", year="pre_famine", year_role="context",
        geography="Ireland", crop="potato", variable="production_total", value="15",
        unit="million_tons_as_printed", measurement_class="balance_sheet",
        source_title=CSO_SOURCE, source_location=CSO_LOCATION,
        note="Bourke's pre-famine potato balance sheet, as restated by the CSO. Not a famine-year loss and not converted to metric tonnes."))
    for variable, value, label in (
        ("use_share_human_food", "47", "human food"),
        ("use_share_animal_feed", "33", "animal feed"),
        ("use_share_seed", "13", "seed"),
        ("use_share_export", "2", "exports"),
        ("use_share_waste_or_balancing", "5", "losses or waste"),
    ):
        rows.append(row(
            event_id="PotatoFamine", year="pre_famine", year_role="context",
            geography="Ireland", crop="potato", variable=variable, value=value, unit="percent",
            measurement_class="balance_sheet", source_title=CSO_SOURCE, source_location=CSO_LOCATION,
            note=f"Share of the pre-famine 15 million tons assigned to {label}. The waste share is a balance-sheet residual, not a blight loss."))
    return rows


COVERAGE = [
    ("GreatFamine", "outside_this_frame", "No source collection was opened. The target frame is the four collections named in the famine inventory."),
    ("Laki1783", "outside_this_frame", "No source collection was opened."),
    ("NoSummer", "outside_this_frame", "The qualitative 1816 note stays in the evidence seed. No harvest series was reconstructed."),
    ("PotatoFamine", "extracted_partial", "Potato area for 1844-1848 and 1859, plus a pre-famine balance sheet. Annual area for 1849-1852, yield, and oats were not on the pages read. Potato stays off the cereal axis."),
    ("Drought18761878", "not_reopened", "District outturn returns were not assembled. The GIPE file was blocked on retrieval. The Tanjore and Coimbatore percentages stay in the famine inventory and were not copied here. The Gazetteer area is famine exposure, not harvested area."),
    ("SovietFamine1921", "extracted_partial", "Three competing USSR-less-SCR evaluations for 1920-1923, and source block 3a by region for 1920-1922. Not a single official total, and not a Volga series. Years 1918-1919 were not extracted."),
    ("ChineseFamine1960", "extracted_partial", "National grain output, procurement, procurement share, net export, and per capita grain for 1954-1966. Provincial output was not in Table 1. 1961 is not the onset."),
    ("DustBowl", "outside_this_frame", "No Great Plains harvest series was reconstructed. The national 1934 wheat level and the decadal model stay in the evidence seed."),
    ("SovietFamine1932", "not_reopened", "The approximate 56 and 50 million tonne harvest statements were not re-read and are not copied into this table. Ukraine and Kazakhstan were not separated. This case is not SovietFamine1921."),
]


def write(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    found = observations()
    write(args.observations, FIELDS, found)
    write(args.coverage, ["event_id", "coverage", "note"], [
        {"event_id": event_id, "coverage": coverage, "note": note}
        for event_id, coverage, note in COVERAGE])
    print(f"observations {len(found)} coverage {len(COVERAGE)}")


if __name__ == "__main__":
    main()

# Event citations, footprint uncertainty, crop-evidence rule, and overlap (November 27, 2026)

`scripts/audit_event_classification.py` writes four tables from files already in the repository. No web page was opened, no citation was added, no observed field was filled, and `ancillary/historical_event_registry.csv` was not edited. The figures do not change: they plot 2020 exposure for the 23 workbook events whatever class an event has.

## 1. Citations (`ancillary/event_citations_v1.csv`)

Each of the 23 workbook events has three catalog references in `ancillary/food_disruptions.csv`. They were compiled with the catalog and were never checked as a set. The table marks each reference:

- `[opened]` only when an opened page in this repository has the same first author and year. Two of 69 qualify: Davis (2001) for Drought18761878 and van Dijk et al. (2013) for MillenniumDrought.
- `[named, not opened]` when the source check names the work but could not open it.
- `[not checked]` for the rest.

| Citation status | Events |
|---|---|
| A catalog reference was opened | 2 |
| Only other sources were opened (source check, archival reconstruction, evidence seed, documentary log) | 13 |
| No source was opened | 12 |

Solomon, EastTimor, and Laos have the same three references word for word. They are Pacific El Niño reports, and Laos is not in the Pacific. These lists are not per-event primary citations. Filling them needs pages that are opened and read, one event at a time.

## 2. Footprint uncertainty (`ancillary/event_footprint_uncertainty_v1.csv`)

These flags are categorical and come from the workbook inventory and the redraw's mapping failures. There is no numeric error bar, because no observed footprint exists to compare against. `observed_affected_area_ha` and `observed_crop_loss_fraction` stay empty.

| Flag | Meaning | Events |
|---|---|---|
| `pre_1900_event_on_2020_geography` | Footprint drawn on present borders and SPAM 2020 cropland | GreatFamine, Laki1783, NoSummer, PotatoFamine, Drought18761878 |
| `more_than_10_countries` | Whole-country totals across many countries | the five above plus ENSO2015_2016 |
| `footprint_wider_than_registry_region` | The registry says Ireland, the sheet lists 14 countries | PotatoFamine |
| `workbook_place_differs_from_registry_region` | The registry says Soviet Union, the sheet lists the Russian Federation | SovietFamine1921 |
| `whole_country_no_subdivision` | All cropland in the country is counted | 14 single-country events |
| `allgrain_mapping_failure` | A listed place adds zero because it has no FIPS mapping | GreatFamine, Laki1783, NoSummer, ENSO2015_2016 |
| `workbook_label_differs_from_convert_name_code_matches` | A name difference only. The GDAM code matches, so this is not a mapping failure | NoSummer, DustBowl, SahelDrought2010, NorthKorea1990s, ENSO2015_2016 |

MillenniumDrought (one country, seven subdivisions) is the only workbook event with no flag. That says nothing about its loss, only that its footprint is not obviously too wide.

## 3. Rule separating crop disruption from famine and conflict (`ancillary/event_evidence_class_v1.csv`)

The driver class comes from the registry driver text: environmental, conflict or policy, or compound. It describes the cause, not the crop outcome. The evidence class depends only on what an opened page measures:

1. **`grain_quantity_opened`**: an opened page reports or reconstructs a grain quantity for the event's place and years. Only SovietFamine1921 (Wheatcroft) and ChineseFamine1960 (Lin and Yang) qualify. Neither page separates the weather, policy, or conflict share.
2. **`crop_measure_opened_not_grain_tonnes`**: a crop measure exists but is not grain tonnes. The cases are potato area (PotatoFamine), district percent saved (Drought18761878), one estate group's yields (GreatFamine), and vatu (Vanuatu).
3. **`production_context_not_attributed`**: a national or regional production figure that the source does not tie to the event alone. The cases are DustBowl, ENSO2015_2016, UkraineWar2022, SudanConflict2023, and SouthSudanConflict.
4. **`no_usable_crop_quantity`**: a qualitative account (NoSummer), a yield gap against a counterfactual (MillenniumDrought), a contested government claim (NorthKorea1990s), or harvest statements not re-read (SovietFamine1932).
5. **`no_crop_source_opened`**: the other 12.

An event counts as crop-disruption evidence only in class 1. A famine label, a mortality toll, an IPC phase, a conflict, or a FAOSTAT production value does not move an event up a class. A documentary "confirmed" disposition does not move it either, because PotatoFamine, Drought18761878, Vanuatu, and SudanConflict2023 are confirmed but fall in classes 2 and 3. Any future observed-loss layer may draw only on class 1, and only after the attribution is argued from a source. No event is removed from the figures by this rule.

## 4. Overlap (`ancillary/event_overlap_v1.csv`)

Every pair of events that shares a GDAM country code and at least one registry year is listed. Only ENSO pairs qualify:

| Parent | Subevent | Shared place | Rule |
|---|---|---|---|
| ENSO2015_2016 | EastTimor | East Timor (67), 2015–2016 | Parent contains the subevent. Do not sum. |
| ENSO2015_2016 | Laos | Laos (123), 2015–2016 | Parent contains the subevent. Do not sum. |
| ENSO2015_2016 | Solomon | none. The Solomon Islands are not on the ENSO sheet | Registry subevent only. Keep separate, and do not add Solomon to the ENSO footprint |

The three subevents stay separate points, as the documentary log requires. A total across events that includes ENSO2015_2016 counts East Timor and Laos twice if it also includes their own points. Vanuatu 2015 shares the year but not a place, so it is not an overlap. MillenniumDrought and ENSO2015_2016 both use Australia, but the registry years do not meet (2000–2009 and 2015–2016).

## Rerun

```bash
python scripts/audit_event_classification.py
```

The tables depend on `results/redraw_2026-11/mapping_failures_after.csv` for the failure counts. `tests/test_event_classification.py` checks the counts and rules above.

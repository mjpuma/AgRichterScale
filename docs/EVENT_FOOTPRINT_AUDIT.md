# Legacy event footprint audit

*Workbook inspection, 2026-11-26. Parser and calculator binding updated the same day. SPAM 2020 was not rerun. Figures 1–4 were not redrawn. Observed affected area and observed loss fraction are empty.*

The two workbooks are `ancillary/DisruptionCountry.xls` and `ancillary/DisruptionStateProvince.xls`. Each has the same 23 sheet names, in the same set as `HISTORICAL_EVENTS`. The four registry cases added later, SovietFamine1932, UkraineWar2022, SudanConflict2023, and SouthSudanConflict, have no sheet.

The inventory is `ancillary/event_footprints_v1.csv`, written by `scripts/audit_event_footprints.py`. It has 225 places: 166 countries and 59 subdivisions. Every numeric code in those sheets is a GDAM code in `ancillary/CountryCode_Convert.xls`. Ten country labels use a different spelling from that table. The code still matches.

## Every workbook country is kept

`EventsProcessor._process_country_sheet` uses the rows `read_excel` returns after the header. It no longer skips the first data row of a multi-country sheet. These seven countries now reach the calculator:

| Event | Country | GDAM | State flag |
|---|---|---:|---:|
| GreatFamine | Belgium | 22 | 0 |
| Laki1783 | Austria | 15 | 0 |
| NoSummer | Andorra | 6 | 0 |
| PotatoFamine | Austria | 15 | 0 |
| Drought18761878 | Algeria | 4 | 0 |
| SahelDrought2010 | Burina Faso | 37 | 0 |
| ENSO2015_2016 | Australia | 14 | 1 |

The Sahel sheet spells Burkina Faso as Burina Faso. The code 37 is Burkina Faso in the convert table. Australia is state-flagged, and its own subdivisions are the ones applied to it.

## Each subdivision stays with its country

The state sheet stores an ISO 3166-2 code, a GDAM country code, and a subdivision name. The processor keeps all three on `state_records`. `EventCalculator.calculate_single_event` and `get_event_grid_cells` pass a country only the names whose country code matches. A flagged country with no matching subdivision is a mapping failure. It is not replaced by the whole country, and it is not given another country's names.

Five events have subdivisions:

| Event | Subdivisions | Binding |
|---|---|---|
| Laki1783 | 9, all India | India only |
| DustBowl | 8, all United States | United States only |
| MillenniumDrought | 7, all Australia | Australia only |
| NoSummer | 6 Canada and 9 United States | Each country receives its own names |
| ENSO2015_2016 | 1 Australia, 6 Brazil, 3 India, 10 United States | Each country, including Australia, receives its own names |

A caller that still passes only a flat `state_codes` list, with no `state_records`, keeps that list. An empty list passed straight to `calculate_state_level_loss` still falls back to the country total, which is the older direct-call behavior.

## An unmapped place is recorded, and the numeric triple stays three zeros

`calculate_country_level_loss` and `calculate_state_level_loss` still return `0.0, 0.0, 0` when a place cannot be mapped. That return is unchanged so existing callers keep working. The failure is also appended to `mapping_failures` with a reason: `no_fips`, `grid_query_failed`, `no_grid_cells`, `state_query_failed`, `no_matching_subdivisions`, `no_state_production_cells`, or `no_subdivisions_for_country`. `calculate_single_event` copies that list onto its result.

Cells that exist and sum to zero do not create a mapping failure. That zero is a measured empty harvest. A zero that carries a mapping failure is not. This pass did not rerun SPAM, so the inventory does not say which of the 225 places would fail a FIPS lookup.

## Names that differ, codes that match

United States is United States of America, code 240. North Korea is Democratic People's Republic of Korea, code 166. On the ENSO sheet, Brunei, East Timor, Laos, Panamá, and Venezuela match Brunei Darussalam, Timor-Leste, Lao People's Democratic Republic, Panama, and Venezuela (Bolivarian Republic of).

## What this inventory does not do

It does not fill observed affected area or observed loss fraction. It does not recalculate 2020 exposure. It does not redraw Figures 1–4. The figure files on disk still come from the previous parser. The next figure run will keep these seven countries and will bind each subdivision to its own country.

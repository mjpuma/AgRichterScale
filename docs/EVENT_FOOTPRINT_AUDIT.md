# Legacy event footprint audit

*Workbook inspection, 2026-11-26. SPAM 2020 was not rerun. Figures 1–4 were not changed. Observed affected area and observed loss fraction are empty.*

The two workbooks are `ancillary/DisruptionCountry.xls` and `ancillary/DisruptionStateProvince.xls`. Each has the same 23 sheet names, in the same set as `HISTORICAL_EVENTS`. The four registry cases added later, SovietFamine1932, UkraineWar2022, SudanConflict2023, and SouthSudanConflict, have no sheet.

The inventory is `ancillary/event_footprints_v1.csv`, written by `scripts/audit_event_footprints.py`. It has 225 places: 166 countries and 59 subdivisions. Every numeric code in those sheets is a GDAM code in `ancillary/CountryCode_Convert.xls`. Ten country labels use a different spelling from that table. The code still matches.

## The country parser drops the first country when there is more than one

`EventsProcessor._process_country_sheet` skips the first row of a sheet that `read_excel` has already given a header, but only when that sheet has more than one data row. A one-country sheet is kept whole. Seven countries are dropped:

| Event | Dropped country | GDAM | State flag |
|---|---|---:|---:|
| GreatFamine | Belgium | 22 | 0 |
| Laki1783 | Austria | 15 | 0 |
| NoSummer | Andorra | 6 | 0 |
| PotatoFamine | Austria | 15 | 0 |
| Drought18761878 | Algeria | 4 | 0 |
| SahelDrought2010 | Burina Faso | 37 | 0 |
| ENSO2015_2016 | Australia | 14 | 1 |

The Sahel sheet spells Burkina Faso as Burina Faso. The code 37 is Burkina Faso in the convert table. Australia is the only dropped country that is also marked for state processing, so the calculator never receives Australia as a country.

## State lists are one flat list per event

The state workbook does tie each subdivision to a country code. The parser keeps the subdivision name and drops that country code. `EventCalculator.calculate_single_event` then passes the whole name list to every country whose state flag is 1.

Five events have subdivisions:

| Event | Subdivisions | How the calculator uses them |
|---|---|---|
| Laki1783 | 9, all India | One flagged country, India |
| DustBowl | 8, all United States | One flagged country |
| MillenniumDrought | 7, all Australia | One flagged country |
| NoSummer | 6 Canada and 9 United States | Both flagged countries receive all 15 names |
| ENSO2015_2016 | 1 Australia, 6 Brazil, 3 India, 10 United States | Brazil, India, and the United States each receive all 20 names. Australia is the dropped country above |

A state name that does not belong to the country being processed fails the name filter. The failure is not stored. It contributes nothing.

## An unmapped place is added as zero

`calculate_country_level_loss` returns `0.0, 0.0, 0` when the GDAM code has no FIPS match, when the grid query fails, or when the country has no grid cells. `calculate_state_level_loss` returns the same triple when the state list matches no cells. `calculate_single_event` adds those zeros into the event total. The result has no count of failed mappings. This pass did not rerun SPAM, so it does not say which of these 225 places would come back as zero. It does say that a zero in the current output cannot be told apart from a true empty harvest.

## Names that differ, codes that match

United States is United States of America, code 240. North Korea is Democratic People's Republic of Korea, code 166. On the ENSO sheet, Brunei, East Timor, Laos, Panamá, and Venezuela match Brunei Darussalam, Timor-Leste, Lao People's Democratic Republic, Panama, and Venezuela (Bolivarian Republic of).

## What this inventory does not do

It does not fill observed affected area or observed loss fraction. It does not recalculate 2020 exposure. It does not change the parser. The dropped countries remain dropped until that code is changed and the figures are drawn again.

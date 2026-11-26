# AgRichter historical-event audit (preliminary, November 25, 2026)

**Scope:** Code and metadata review. Workbook sheet contents were read on 2026-11-26 and are inventoried in [EVENT_FOOTPRINT_AUDIT.md](EVENT_FOOTPRINT_AUDIT.md). SPAM 2020 was not rerun, and the figures were not changed. This audit does not certify individual historical loss estimates.

## Key result: distinguish potential exposure from observed losses

The operational event definitions are read from `ancillary/DisruptionCountry.xls` and `ancillary/DisruptionStateProvince.xls` by `scripts/fig1_agrichter_scale.py` and `scripts/fig4_risk_probability.py`. `EventsProcessor` translates workbook sheets into country and state lists. `EventCalculator` then totals *all* selected crops' SPAM **2020** harvested area and production within the selected countries/states; it does not apply an event-specific affected-area mask, crop loss fraction, or historical yield observation. The event magnitude is `log10(aggregate harvested area in km²)`. The supplemental methods separately state a complete-loss assumption for theoretical H–P envelopes.

Consequently, the current event outputs labeled `harvest_area_loss_ha` and `production_loss_kcal` are **contemporary agricultural exposure under an implicit complete-disruption counterfactual**, **not verified observed historical losses**. Do not label plotted event values as measured historical losses without additional event-specific evidence. This interpretation matters particularly for famines, which may reflect access and policy rather than total crop failure.

## Audit findings and next actions

| Priority | Finding | Code/data location | Action |
|---|---|---|---|
| Critical | All selected regional harvested area and production are counted as disrupted/lost, without an event-specific crop-loss fraction | `agrichter/analysis/event_calculator.py` (country and state calculations) | Distinguish region-wide 2020 exposure from observed affected area and observed production loss; relabel existing plots or supply documented masks and loss fractions |
| Critical | Event metadata lacks per-event source citations and footprint uncertainty | `ancillary/event_attributes.csv` (23 events) | Record primary references, dates, crop-specific geography, and inclusion criteria |
| High | State entries form a flat event-wide list and are reused when processing each country flagged at state level | `agrichter/data/events.py`, `agrichter/analysis/event_calculator.py` | Preserve country–subdivision association; test multi-country events |
| High | Country-sheet parser skips the first data row when the sheet has more than one country. Seven countries are dropped, including Australia on the ENSO sheet | `agrichter/data/events.py` `_process_country_sheet`; inventory in [EVENT_FOOTPRINT_AUDIT.md](EVENT_FOOTPRINT_AUDIT.md) | Keep the workbook row in the footprint inventory. Do not change the parser until the figures are redrawn |
| High | Unmapped regions can be returned as zero impact rather than an explicit mapping failure | `EventCalculator.calculate_country_level_loss` / state-level equivalent | Export mapping failures, distinguish missing from true zero |
| High | Events from 1315 onward are mapped onto SPAM 2020 agricultural geography | `Config._setup_paths`, historical workbooks | Explain the counterfactual and its historical limitations |
| High | Included events mix crop disruptions, conflict, policy famine, natural disasters, and multiyear emergencies | `event_attributes.csv` | Define agriculturally disrupted-area criteria and avoid conflating famine with crop loss |
| Medium | Main Figure 1 script uses a machine-specific absolute path | `scripts/fig1_agrichter_scale.py` | Make repository-relative for reproducibility |
| Medium | Event periods and possible overlap are not explicitly resolved | `event_attributes.csv` | Clarify ENSO 2015–16 versus its regional events and multiyear crises |

## Inventory (all require documentary verification)

**Historical / pre-1900:** GreatFamine, Laki1783, NoSummer, PotatoFamine, Drought18761878.

**1900–1999:** SovietFamine1921, ChineseFamine1960, DustBowl, NorthKorea1990s, Ethiopia, Bangladesh, SierraLeone, Liberia.

**2000 onward:** SahelDrought2010, MillenniumDrought, ENSO2015_2016, Solomon, Vanuatu, EastTimor, Haiti, Yemen, Laos, Syria.

The 23 identifiers in `HISTORICAL_EVENTS` match the 23 rows in `event_attributes.csv`; an older design document refers to 21. Neither the spreadsheet footprint rows nor their source citations have been independently inspected here.

## Proposed reproducible event-level audit dataset

Extract an explicit row per event and affected country/subdivision, with columns:

`event_id,source_workbook,source_sheet,country_code,country_name,subdivision_code,subdivision_name,first_crop_year,last_crop_year,crop,mapped_2020_harvest_area_ha,observed_affected_area_ha,observed_crop_loss_fraction,spatial_basis,primary_source,uncertainty,mapping_status,inclusion_decision`

Keep *observed* area and loss fraction **missing**, not 0 or 1, until sourced.

## Next steps

1. Parse and inspect every sheet of the two Excel workbooks, reconcile country/state assignments, and export a complete footprint inventory.
2. Fix and test any row-skipping, grouping, or zero-as-missing behavior found.
3. The source check of the Dust Bowl, 1816, Vanuatu 2015, and the Sudan 2025 cereal assessment is in [SOURCE_CHECK.md](SOURCE_CHECK.md). None of those readings is an event loss fraction. Figures were not redrawn.
4. Recalculate event exposure/impact with verified footprints or explicitly label the 2020-based counterfactuals.
5. Revisit Figures 1–4 only after event meaning and mapping are validated.

**Unverified in this audit:** External SPAM inputs, rerun outputs, and event-specific historical crop losses. Workbook rows are now listed in [EVENT_FOOTPRINT_AUDIT.md](EVENT_FOOTPRINT_AUDIT.md). Observed loss fields in that inventory are empty.

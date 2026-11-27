# Figures 1–4 redraw (November 27, 2026)

Figures 1–4 were redrawn from main after PR 10 (parser fix), PR 12 (quantity label) and PR 13 (redraw gate). Every replaced image is kept beside the new one with a `_pre_fix` suffix. Observed losses were not filled and are not plotted. SPAM was not rerun; the same SPAM 2020 V2r0 files were read.

## Caption text

> Event points are SPAM 2020 harvested area and production of the selected crops (wheat, maize, rice, or all grain) inside the places listed for each event in `ancillary/DisruptionCountry.xls` and `ancillary/DisruptionStateProvince.xls`. A country flagged at country level contributes its whole area. A country flagged at state level contributes only its own listed subdivisions. All of that harvest is counted as disrupted, an implicit loss fraction of 1. The points are 2020 exposure under complete disruption, not observed historical losses. A place that could not be mapped adds zero to its event and is recorded as a mapping failure; that zero is not a measured harvest. The 23 workbook events are plotted; the four registry extensions without a workbook sheet are not.

## What changed in the point set

`scripts/compare_event_points.py` computed the 23 points for each crop twice against the same SPAM files: once with the `agrichter` package from main before PR 10 (commit 4438249) and once with current main. Both tables are in `results/redraw_2026-11/`. 16 events have identical points. Seven multi-country events gain a country, and six of them move, because the parser no longer drops the first data country on the sheet:

| Event | Country restored | All grain ΔM_D | All grain production ratio | Largest crop change |
|---|---|---|---|---|
| GreatFamine | Belgium | +0.004 | 1.013 | maize +0.007 |
| Laki1783 | Austria | +0.001 | 1.005 | maize +0.002 |
| NoSummer | Andorra | 0 | 1.000 | none; Andorra has no FIPS mapping and is a recorded failure |
| PotatoFamine | Austria | +0.013 | 1.032 | maize +0.032 |
| Drought18761878 | Algeria | +0.004 | 1.003 | wheat +0.010 |
| SahelDrought2010 | Burkina Faso (workbook label "Burina Faso ") | +0.034 | 1.079 | maize +0.047 |
| ENSO2015_2016 | Australia | +0.035 | 1.026 | wheat +0.175, production ×1.24 |

The ENSO wheat point moves most because Australia enters as a whole country. Rebinding subdivisions to their own country did not move any plotted total for these files. No point moved down.

## Mapping failures in the new draw

`results/redraw_2026-11/mapping_failures_after.csv` lists every place that added a zero. For all grain there are 7, all `no_fips` (codes 110, 245, 6, 148, 196 and 80 across GreatFamine, Laki1783, NoSummer and ENSO2015_2016). Single crops add `no_grid_cells` (no SPAM cells for that crop in the country) and, for rice, `no_matching_subdivisions` for China, the United States and Australia. The plotted totals do not show these; the CSV does.

## Axis and legend wording

| Where | Before | After |
|---|---|---|
| Figure 1 y-axis | Harvest Area Disrupted (km²) | SPAM 2020 harvested area disrupted at loss fraction 1 (km²) |
| Figures 2 and 3 y-axis | Production Loss (kcal) | SPAM 2020 production disrupted at loss fraction 1 (kcal) |
| `fig4_risk_probability.py` legend | Historical Events (Model-Mapped); Historical Obs. Range | Event footprints, SPAM 2020 exposure (model-mapped); Magnitude range of event footprints |

The H–P envelopes share the Figure 2 and 3 axis; they are also complete-disruption totals, so the wording covers both.

## What was and was not redrawn

- **Figure 1** (`figure1_agrichter_scale.*` and four individual panels): redrawn. Points moved as above.
- **Figure 2** (`figure2_hp_envelopes.*` and four individual panels): redrawn. The script logged a lower-envelope harvested-area conservation mismatch for all four crops, from 0.02 to 0.04 percent (all grain 7,027,154 versus 7,029,937 km²). The envelope code is unchanged by PR 10 and PR 12; this was not investigated here.
- **Figure 3** (`figure3_country_envelopes.png` and ten individual panels): redrawn. It plots country envelopes and thresholds, not event points, so only the axis wording changed. `figure3_country_EnvelopesMap.png` and `results/fig3_components/` have no generating script in the repository and were not redrawn.
- **Figure 4** (`fig4_components/figure4_risk_v4.png` from `fig4_overhaul.py`): redrawn. It plots no event points. The threshold labels overlap, as in the old file. `figure4_risk.png` has no generating script and was not redrawn. `fig4_risk_probability.py` produced images 577,747 pixels tall, apparently from a runaway label placement; they were discarded and are not in this change.

## Reproducing

SPAM 2020 is not tracked. `Config` reads it from the repository root, or from `AGRICHTER_SPAM_ROOT` if set; ancillary tables always come from the repository. This draw used `AGRICHTER_SPAM_ROOT` pointing at a separate local copy of SPAM 2020 V2r0 and ran each script from the repository root.

## Second draw after merging the July fixes (November 27, 2026)

Six commits from 14–15 July 2026 were never pushed to main. They fixed FIPS code swaps, the all-grain basket, per-crop kcal, float64 conservation, and USDA units and the threshold window, and they added `scripts/figS5_fractional_loss.py` and `scripts/tableS2_verification.py`. They were merged with PR 10's parser and mapping-failure record kept. Figures 1–4 and S5 were then redrawn, and Table S2 was rerun, from the merged code.

- **Thresholds.** All-grain 1-month, 3-month, and Total Stocks are 7.83, 23.50, and 27.95 ×10¹⁴ kcal, matching the manuscript.
- **Table S2** (`results/tableS2_verification.csv`). Every global crop and every country envelope passes all 5 invariants. Endpoint residuals are at most 1.7×10⁻¹⁵. Events inside the envelope band: all grains 23/23, wheat 16/16, maize 21/21, rice 20/20. The 0.02–0.04 percent lower-envelope conservation mismatch from the first November draw does not recur. It came from the missing float64 fix.
- **Points.** `results/redraw_2026-11/event_points_july_merge.csv` holds the new points, and `mapping_failures_july_merge.csv` the new failures. Against the first November draw, all-grain magnitudes move by at most 0.035. ENSO2015_2016 all-grain production rises 24 percent. Rice changes most, because the FIPS swaps had mapped some European places to the wrong country: PotatoFamine, NoSummer, and GreatFamine rice fall by more than 95 percent, and DustBowl rice rises tenfold. Maize Solomon now has no grid cells.
- **All-grain exposure against the thresholds.** Drought18761878 is 2.14× the 3-month buffer and 1.80× Total Stocks. Laki1783 is 1.74× and 1.46×. ChineseFamine1960 is 0.93× the 3-month buffer, above the 1-month line only. At 50 percent intensity, only Drought18761878 stays above the 3-month buffer, at 1.07×.

# What the plotted event numbers are

*Labeled 2026-11-26. SPAM 2020 was not rerun. Figures 1–4 were not redrawn. Observed area and observed loss fraction stay empty.*

`harvest_area_loss_ha` and `production_loss_kcal` are the names in `EventCalculator`. For each historical event the calculator adds up the SPAM 2020 harvested area and production of the crops selected for that run, inside the places it received, and keeps the whole sum. No event-specific loss fraction is applied. The implicit fraction is 1. Magnitude is the base-10 log of that harvested area in square kilometres.

That quantity is 2020 exposure under a complete-disruption counterfactual. It is not an observed historical loss. A matched crop, place, year, and baseline would be required before a historical loss could replace it. The source check did not supply that match.

The label for every registry case is `ancillary/event_quantity_label_v1.csv`, written from `agrichter/analysis/event_quantity.py`. The 23 identifiers in `HISTORICAL_EVENTS` are the points on the current figures. SovietFamine1932, UkraineWar2022, SudanConflict2023, and SouthSudanConflict have no disruption sheet, so they have no plotted point and no 2020 total in this table.

The label does not recompute the numbers in the figure files. Those files remain as they were drawn. The next drawing is a separate step.

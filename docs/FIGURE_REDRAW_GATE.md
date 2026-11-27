# Figure redraw gate (November 26, 2026)

**Status: closed.** Figures 1–4 have not been redrawn. The image files under `results/` were drawn with the parser on main, which drops the first data country on a multi-country sheet, and their axes still say loss. This page records what has to be true before anyone redraws them. It does not start a redraw, rerun SPAM, fill an observed field, or edit `ancillary/historical_event_registry.csv`.

## The six conditions

All six must hold in the same draw.

| # | Condition | State on main, 2026-11-26 |
|---|---|---|
| 1 | PR 10 (`faostat-footprint-fix`) is merged, or the redraw is explicitly from the pre-fix parser and the caption says which. | Open. Main still skips the first data country and gives every state-flagged country the whole state name list. |
| 2 | PR 12 (`faostat-quantity-label`) is merged, or the axis and caption text are changed in the same draw so the points are named as 2020 exposure under complete disruption. | Open. The plotted columns are still named `harvest_area_loss_ha` and `production_loss_kcal`. |
| 3 | The caption states the crop set, the place rule, the year 2020, and the implicit loss fraction of 1. | Not written. |
| 4 | Observed loss stays off the figure. | Holds. Nothing in [SOURCE_CHECK.md](SOURCE_CHECK.md) is adopted as an event loss. |
| 5 | The note says that unmapped places on current main enter the total as zero with no failure count. After PR 10, a zero with a mapping failure is not a measured harvest. | Holds as a statement, below. |
| 6 | Figure scripts use repository-relative paths. | Done for `scripts/fig1_agrichter_scale.py` and `scripts/fig2_hp_envelopes.py` in this change. Not checked by running them. |

## Condition 4: what must not be plotted as the event loss

These numbers were read in the source check. None of them is the quantity on the vertical axis.

- The Dust Bowl NASS percents, 36.4 for wheat and 47.8 for corn. They are national, from one NASS edition, against a 1921–1930 mean.
- The Millennium Drought 18–22 percent. It is a dryland wheat yield gap against average conditions, and the same paragraph says production rose.
- The Davis percent-of-crop-saved shares for Madras, or their complement.
- The Vanuatu total of VT 4,175 million. It is money.
- The Sudan 2025 national cereals of about 5.2 million tonnes. It is national production, not an event loss.

## Condition 5: zeros on main

On main, `EventCalculator.calculate_country_level_loss` and `calculate_state_level_loss` return `0.0, 0.0, 0` for an unmapped FIPS code, an empty grid, or an unmatched state, and that triple is added into the event total. Main records no failure for it. A failed lookup and a real empty harvest therefore give the same number. PR 10 adds a `mapping_failures` reason but still adds the zero, so a result table that shows only the total still hides the failure.

## Condition 6: paths

Before this change, Figures 1 and 2 built `Config(root_dir='/Users/mjp38/GitHub/AgRichter2025')` while reading the workbooks and writing `results/` relative to the working directory. A run from another checkout read SPAM from one tree and the workbooks from another. Both scripts now resolve the SPAM root, the two workbooks, and `results/` from the script's own location. The SPAM 2020 directories are not tracked and are not present in this clean checkout, so a run here needs them placed at the repository root first; the SPAM-dependent unit tests skip for the same reason. Figures 3, 4, S1, S2, and S3 still use `root_dir='.'` and `Path('results')`, so they work only when run from the repository root. That is a working-directory rule, not a machine path. A green test suite does not show that any figure is reproducible.

## If the user says "redraw"

Produce the new figure from merged code, keep the old files, and write a short note of what changed in the point set: the seven previously dropped countries (GreatFamine Belgium, Laki1783 Austria, NoSummer Andorra, PotatoFamine Austria, Drought18761878 Algeria, SahelDrought2010 Burkina Faso, labelled "Burina Faso " in the workbook, ENSO2015_2016 Australia), country-bound subdivisions, and the axis wording. Do not fill observed losses in that draw.

## Check

`tests/test_figure_redraw_gate.py` checks that Figures 1 and 2 hold no absolute user path, that this page lists six conditions and names both open PRs, and that the registry still has 27 rows. It does not run a figure script.

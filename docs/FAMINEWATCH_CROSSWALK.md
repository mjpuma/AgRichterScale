# FamineWatch crosswalk, 27 registry cases

*Checked 2026-11-25 against [mjpuma/faminewatch-datasets](https://github.com/mjpuma/faminewatch-datasets) commit `d2f14b52`, file `data/faminewatch_events.csv`, 721 rows. The registry was not edited. None of these records is entered as a crop loss.*

FamineWatch v0.2 is a famine chronology: famine-year presence, food-crisis narratives, and excess mortality. The European rows from Ljungqvist et al. (2024) are a region-year flag. They do not give severity or harvested area. The Qing REACHES file is a count of historical records, not production. A famine record, a death toll, or an IPC classification is not a crop loss.

Beta compilation sentences that mention a production percent are leads. In this pass they were not opened back to the cited page, so they stay out of the archival observation table. Where the same comparison was already re-read, the re-read table remains the record.

The row-by-row result is `ancillary/faminewatch_registry_crosswalk_v1.csv`. Every registry case is there once. `crop_loss_from_this_record` is `no` on all 27.

## Crop sentences that are not yet a reconstruction

These are in the compilation text. They were not re-read from the cited source.

- **GreatFamine.** FW-0206 says grain production fell 12–44 percent depending on location. Ljungqvist’s 1315 rows are famine-year flags for Central Europe, France, Great Britain, Ireland, the Low Countries, and the Nordic countries.
- **PotatoFamine.** FW-0353 says potato production fell roughly 86 percent in 1847. The planted-area series already extracted from the CSO page is the reconstruction. The 86 percent is not added to it. FW-0354 is an excess-mortality figure of 1,000,000.
- **Drought18761878.** FW-0367 says 63–94 percent of crops were lost in the Madras Presidency in 1877, and that the Algeria grain harvest fell 50 percent. FW-0366 is an India mortality range. These are not district outturn returns.
- **DustBowl.** FW-0415 says 1934 United States wheat fell 36 percent and corn 48 percent relative to the 1921–1930 average. That is a national comparison, not a Great Plains series, and USDA (2015) was not opened.
- **MillenniumDrought.** FW-0463 says wheat yields fell 18–22 percent from July 2002 to June 2009. The registry window is 2000–2009. The sentence is a yield range, not production.
- **NorthKorea1990s.** FW-0459 says 30 percent of grain crops were lost in 1995. The percent was not re-read. FW-0458 is a mortality range.
- **NoSummer.** FW-0341 says poor harvest and famine, and that parts of New England lost almost all of the maize crop.
- **Laki1783.** FW-0323 says poor harvest and famine, without a crop total.

## One sentence that matches a table already re-read

**ChineseFamine1960.** FW-0429 says grain production fell 28 percent relative to 1958. Lin and Yang’s table, already transcribed, has 143.50 million tons in 1960 against 200.00 in 1958. The faminewatch sentence is that comparison, rounded. It does not add the 1954–1966 series. FW-0428 is a separate mortality range of 15–33 million and is not a production figure.

## Famine or mortality only

- **SovietFamine1921.** FW-0405 is the name of the famine. FW-0406 is an excess mortality of 9,000,000. Neither has a harvest tonnage. The Wheatcroft evaluations remain the reconstruction.
- **SovietFamine1932.** FW-0411 is a mortality record and says the 1931–32 harvests were poor, without tonnes. FW-0412 and FW-0413 are separate mortality rows for Russia with Kazakhstan and for Ukraine. They are not added to FW-0411, and they are not the 1921 series.
- **SahelDrought2010.** FW-0467 is titled Sahel Famine for 2009–2010 and gives no crop figure. The registry years are 2010–2011.
- **Ethiopia.** FW-0450 and FW-0451 are mortality figures for 1983–1985.
- **Bangladesh.** FW-0439 is a mortality range. FW-0440 names a harvest shortfall and gives no quantity.
- **Yemen.** FW-0470 is an IPC food-insecurity record. No area was classified in Famine.
- **SudanConflict2023.** FW-0473 records IPC Famine in Zamzam, camps near El Fasher, El Fasher, and Kadugli in 2024–2025. That is not a national cereal figure.
- **SouthSudanConflict.** FW-0471 is a 2017 declaration affecting 100,000 people. The registry window starts in 2013. There is no cereal total.

## No row in this file

ENSO2015_2016, Solomon, Vanuatu, EastTimor, Haiti, SierraLeone, Liberia, Laos, Syria for 2011 onward, and UkraineWar2022.

Duplicate compilation and OWID rows for the same famine are not added together.

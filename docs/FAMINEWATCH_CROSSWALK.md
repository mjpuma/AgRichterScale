# FamineWatch crosswalk, 27 registry cases

*Checked 2026-11-25 against [mjpuma/faminewatch-datasets](https://github.com/mjpuma/faminewatch-datasets) commit `d2f14b52`, file `data/faminewatch_events.csv`, 721 rows. The registry was not edited. None of these records is entered as a crop loss.*

FamineWatch v0.2 is a famine chronology: famine-year presence, food-crisis narratives, and excess mortality. The European rows from Ljungqvist et al. (2024) are a region-year flag. They do not give severity or harvested area. The Qing REACHES file is a count of historical records, not production. A famine record, a death toll, or an IPC classification is not a crop loss.

Beta compilation sentences that mention a production percent were leads. The pages that could be opened are recorded in [SOURCE_CHECK.md](SOURCE_CHECK.md). They stay out of the archival observation table. Where the same comparison was already re-read, the re-read table remains the record.

The row-by-row result is `ancillary/faminewatch_registry_crosswalk_v1.csv`. Every registry case is there once. `crop_loss_from_this_record` is `no` on all 27.

## Crop sentences checked against the cited page

The compilation sentences were opened where a page could be read. The reading is [SOURCE_CHECK.md](SOURCE_CHECK.md). None of it is entered as an event loss, and none of it is appended to the archival table.

- **GreatFamine.** Jordan (1996) was not opened, so the 12–44 percent sentence stays unverified. A different page, Schuh (2020), reports Titow’s Winchester yields at 36 percent below average in 1314/15 and 45 percent below average in 1315/16. That is one estate series, not the compilation range.
- **PotatoFamine.** The CSO page, re-read, prints potato area, including 656,014 ha in 1846 and 89,000 ha in 1847. It does not say production fell 86 percent. Bourke (1960) was not retrieved. The area series remains the reconstruction.
- **Drought18761878.** Davis (2001), Table 1.2, prints the share of the crop saved in ten Madras districts, from 6 percent in Bellary and Kurnool to 37 percent in Tinnevelly. The compilation’s 63–94 percent lost is the complement of that saved range. Davis does not print it. The same book says half the Algeria grain harvest was lost in 1877. The 1878 Famine Commission page cited under the table was not opened.
- **DustBowl.** Henson (2012) and USDA (2015) were not opened. The NASS April 2023 track record, one national series, puts 1934 wheat 36.4 percent and corn harvested for grain 47.8 percent below the 1921–1930 mean. Those rounded percents match the compilation. The place is the United States, not the Great Plains.
- **MillenniumDrought.** van Dijk et al. (2013) say wheat yields in July 2002–June 2009 were 18–22 percent below yields under average conditions, and that production volume rose because a smaller yield decline was offset by more area. BOM (2015) was not opened.
- **NorthKorea1990s.** Goodkind and West (2001) was not opened. Noland, Robinson, and Wang (1999) record a government announcement of 1.9 million tons of grain lost, and say outside estimates of the damage were far lower. A later paraphrase calls 1.9 million tons about 30 percent of the grain needed to feed the population. That is not “30 percent of grain crops lost.”
- **NoSummer.** Stommel and Stommel (1983) was not opened, so “almost 100 percent” of New England maize was not confirmed. Jefferson’s letter of 8 September 1816, as quoted by Monticello, says the corn crop through the Atlantic states will probably be less than one-third of an ordinary one.
- **Laki1783.** FW-0323 says poor harvest and famine, without a crop total. No crop total was opened.

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

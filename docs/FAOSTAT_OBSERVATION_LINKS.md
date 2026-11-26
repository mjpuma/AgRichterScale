# FAOSTAT observation links, 27 registry cases

*Pass of 2026-11-25. The registry file was not edited. The 10% candidate file was not opened. No case was added or removed.*

A link names the FAOSTAT reporting area that existed during the registry years, and the production rows for wheat (item 15), rice paddy (item 27), maize (item 56), and cereals primary (item 1717). The extract is the local QCL snapshot already audited in [FAOSTAT_SNAPSHOT_FINDINGS.md](FAOSTAT_SNAPSHOT_FINDINGS.md), ZIP SHA-256 `c5835418c18f9322e7decbd6800f93a216eaae3cdfa31acb08f0518c0c6d6853`, through 2024. Open-ended cases stop at 2024. There is no 2025 row in this snapshot.

A production value on a linked row is national or territorial production. It is not an event loss, and it was not chosen because it crossed a threshold. Cereals primary is a separate item. It is not added to wheat, rice, or maize.

The case table is `ancillary/faostat_event_links_v1.csv`. The crop-year rows are `ancillary/faostat_event_link_rows_v1.csv`. Both are written by `scripts/link_faostat_observations.py`.

## Count

| Link | Cases |
|---|---:|
| No FAOSTAT row | 10 |
| 1961 rows only; 1959 and 1960 have no row | 1 |
| Rows on the reporting area in force | 16 |
| Total | 27 |

The 16 linked cases and the 1961 China remainder contain 408 crop-years. Of those, 308 have a production value, 16 are present with FAO flag M and no value, and 84 have no production row for that item. Cereals primary has a production value in every linked case-year. Among the 308 values, the FAO quality flag is A on 230, X on 47, E on 22, and I on 9. That is the source mark, not a shortfall flag.

## No row

These ten cases have no production row in the link file.

- **GreatFamine.** 1315–1317 is before the extract, and Europe is not one reporting area.
- **Laki1783.** 1783–1784 is before the extract. Iceland code 99 is not used to fill those years.
- **NoSummer.** 1816 is before the extract, and Global is not a reporting area.
- **PotatoFamine.** 1845–1852 is before the extract. The documented crop is potato, which is outside items 15, 27, 56, and 1717. Ireland code 104 is not attached.
- **Drought18761878.** 1876–1878 is before the extract, and Asia/Africa is not one reporting area.
- **SovietFamine1921.** 1921–1922 is before the extract. USSR code 228 begins in 1961 and is not used to fill those years.
- **DustBowl.** 1930–1936 is before the extract. The Great Plains is not a reporting area, and United States code 231 is not attached.
- **SahelDrought2010.** 2010–2011 is inside the extract. No FAOSTAT area is named Sahel, and this pass does not assign a country set.
- **ENSO2015_2016.** Global tropics is not a reporting area. The Solomon Islands, Timor-Leste, and Lao rows stay on those cases.
- **SovietFamine1932.** 1932–1933 is before the extract. USSR code 228 is not used. This case is not SovietFamine1921.

## China, 1959–1961

**ChineseFamine1960** has no row for 1959 or 1960. The four 1961 production rows are on China, mainland, code 41. China code 351 is an aggregate and is not used. Those 1961 rows are national production in the last registry year. They are not the 1959–1960 decline, and they are not the grain series in the famine inventory.

## Reporting area in force

| Case | Code | Area | Years | Values | Flag M | No item row |
|---|---:|---|---|---:|---:|---:|
| MillenniumDrought | 10 | Australia | 2000–2009 | 40 | 0 | 0 |
| NorthKorea1990s | 116 | Democratic People's Republic of Korea | 1994–1998 | 20 | 0 | 0 |
| Solomon | 25 | Solomon Islands | 2015–2016 | 4 | 0 | 4 |
| Vanuatu | 155 | Vanuatu | 2015 | 2 | 0 | 2 |
| EastTimor | 176 | Timor-Leste | 2015–2016 | 6 | 0 | 2 |
| Haiti | 93 | Haiti | 2010–2016 | 21 | 0 | 7 |
| SierraLeone | 197 | Sierra Leone | 1991–2002 | 36 | 0 | 12 |
| Liberia | 123 | Liberia | 1989–2003 | 30 | 0 | 30 |
| Yemen | 249 | Yemen | 2015–2024 | 30 | 0 | 10 |
| Ethiopia | 62 | Ethiopia PDR | 1983–1985 | 9 | 0 | 3 |
| Laos | 120 | Lao People's Democratic Republic | 2015–2016 | 6 | 0 | 2 |
| Bangladesh | 16 | Bangladesh | 1974 | 4 | 0 | 0 |
| Syria | 212 | Syrian Arab Republic | 2011–2024 | 42 | 14 | 0 |
| UkraineWar2022 | 230 | Ukraine | 2022–2024 | 12 | 0 | 0 |
| SudanConflict2023 | 276 | Sudan | 2023–2024 | 8 | 0 | 0 |
| SouthSudanConflict | 277 | South Sudan | 2013–2024 | 34 | 2 | 12 |

Ethiopia 1983–1985 stays on Ethiopia PDR, code 62. Ethiopia code 238 begins in 1993 and is not used. Sudan from 2023 stays on code 276. Sudan (former) code 206 ends in 2011 and is not used. South Sudan from 2013 stays on code 277, which begins in 2012. Ukraine from 2022 stays on code 230, which begins in 1992. USSR code 228 is not used. Yemen stays on code 249. Bangladesh 1974 stays on code 16, the same series that runs from 1961. Timor-Leste code 176 already covers 2015–2016.

Yemen, Syria, Ukraine, Sudan, and South Sudan are open in the registry. The link ends at 2024. Figures for 2025 in the documentary search log are not rows in this snapshot.

Vanuatu's cereal rows are the country-year series. The documented impact remains the monetary crop-sector total, not these tonnes. Haiti's rows do not separate the earthquake from the hurricane. A war in Sierra Leone, Liberia, Yemen, Syria, Ukraine, Sudan, or South Sudan is not itself the production number.

## Where an item has no row

These gaps are absences of that item, not losses.

- Wheat has no production row for Solomon Islands 2015–2016, Timor-Leste 2015–2016, Haiti 2010–2016, Sierra Leone 1991–2002, Liberia 1989–2003, Lao People's Democratic Republic 2015–2016, Vanuatu 2015, or South Sudan 2013–2024.
- Maize has no production row for Solomon Islands 2015–2016 or Liberia 1989–2003.
- Rice has no production row for Vanuatu 2015, Ethiopia PDR 1983–1985, or Yemen 2015–2024.
- Rice in the Syrian Arab Republic is present with flag M and no value for every year 2011–2024. Rice in South Sudan is present with flag M for 2013 and 2014, and has a value from 2015 through 2024.

## What this pass does not do

It does not read the 10% flag list or sort flags into events. It does not change `ancillary/historical_event_registry.csv`, the envelopes, or the figures. It does not reconstruct the pre-1961 cases from archival sources. It does not treat a national total as the footprint of the Great Plains, the Sahel, or the global tropics.

# FAOSTAT QCL snapshot verification and screening run

Run on commit `c3cbe9d` (2026-11-25), Python 3.10.5, standard library only. The historical event registry, evidence inventory, H–P envelopes, and figures were not changed. Full generated tables stay local under `outputs/faostat_historical_screen_v1/` (gitignored). Compact copies committed here are the inspector audit, the area audit, annual coverage, and the manifest.

## Snapshot used

The inspector found the sibling FoodShockCascadeModel cache and read it in place. No FAOSTAT file was downloaded.

| | |
|---|---|
| Archive | `FoodShockCascadeModel/inputs/faostat_bulk_cache/production.zip` |
| ZIP SHA-256 | `c5835418c18f9322e7decbd6800f93a216eaae3cdfa31acb08f0518c0c6d6853` |
| Member | `Production_Crops_Livestock_E_All_Data_(Normalized).csv` |
| Member SHA-256 | `c4ff713e4b8e0f30eadf5a37632ede4faf95bb8e60c12becd16ef793b0cbac01` |
| Member size | 545,005,770 bytes |
| ZIP member timestamp | 2025-12-23 17:21:22 |
| Local cache mtime | 2026-11-19 13:33:31 |
| Encoding | strict `utf-8-sig` (full member decoded) |

The extracted CSV at `FoodShockCascadeModel/inputs/Production_Crops_Livestock_E_All_Data_(Normalized).csv` is byte-identical to the ZIP member. FoodTradeNetwork and sheaf-model do not hold this QCL production bulk. The FoodShockCascadeModel reader accepts Latin-1; this file did not need that fallback.

## Temporal coverage and cutoff

Selected wheat, rice, maize, and cereals-primary rows exist in every year from 1961 through 2024. There are no 2025 rows. World totals (area code 5000, element 5510, flag A) show that 2024 is populated at the same scale as 2023:

| Item | 2023 (t) | 2024 (t) |
|---|---:|---:|
| Wheat (15) | 794,567,417 | 798,481,711 |
| Rice (27) | 804,744,456 | 820,223,278 |
| Maize (56) | 1,238,582,561 | 1,218,205,574 |
| Cereals, primary (1717) | 3,126,272,362 | 3,133,222,741 |

Inspector counts of non-missing production area codes, including aggregates, move by at most one code from 2023 to 2024 (wheat 155 to 155, rice 170 to 169, maize 204 to 203, cereals primary 213 to 212). Panel assessable reporting entities likewise move by at most one (wheat 121, rice 115 to 114, maize 161 to 160, cereals primary 178 to 177).

Cutoff **2024** is the last year in the file and is not a stub year relative to 2022–2023. FAO can still revise 2024. Area-code counts are not a certificate of territorial continuity.

## Commodity definitions and units

Each selected series uses one element code. The parser would have stopped on a duplicate key.

| Item code | Source label | CPC | Production | Area harvested | Yield |
|---|---|---|---|---|---|
| 15 | Wheat | 0111 | 5510, t | 5312, ha | 5412, kg/ha |
| 27 | Rice | 0113 | 5510, t | 5312, ha | 5412, kg/ha |
| 56 | Maize (corn) | 0112 | 5510, t | 5312, ha | 5412, kg/ha |
| 1717 | Cereals, primary | F1717 | 5510, t | 5312, ha | 5412, kg/ha |

The bulk labels item 27 **Rice**, not "Rice, paddy". World 2023 production of 805 Mt and yield of 4,764 kg/ha identify paddy rice. Milled rice is a smaller quantity and is not this item. The pipeline name `rice_paddy` matches that magnitude.

Yield in this bulk is **kg/ha**. China, mainland wheat in 2024 is internally consistent: 140,100,000 t / 23,587,400 ha = 5,939.6 kg/ha, equal to the stored yield.

Cereals, primary overlaps the component crops. In 2023 the world cereals-primary total exceeds wheat + rice + maize by 288.4 Mt, which is the other cereals in item 1717. Item 1717 must not be added to items 15, 27, or 56, and it is not established as the AgRichter allgrain basket.

## Data-quality flags

The bulk flag table defines A official, E estimated, I imputed by a receiving agency, M missing (data cannot exist), and X figure from an external organization. Selected production rows: A 36,501, E 5,124, X 1,150, I 804, M 304.

The screener removes flag M from the baseline and from the target used for flagging, and it keeps the original source value on the row. Flags A, E, I, and X all remain eligible. Of the 10% shortfall flags, official values are the majority and estimated or external values are still included:

| Crop | A | E | I | X | 10% flags |
|---|---:|---:|---:|---:|---:|
| Wheat | 1,762 | 84 | 55 | 108 | 2,009 |
| Rice | 1,745 | 93 | 44 | 68 | 1,950 |
| Maize | 2,250 | 154 | 67 | 158 | 2,629 |
| Cereals, primary | 1,985 | 442 | 0 | 32 | 2,459 |

## Area classification

The selected window contains 244 area codes: 209 provisional reporting entities and 35 provisional aggregates. Aggregates excluded from the panel are World, the continental and subregional groups, China (351), European Union (27), and the LDC, LLDC, SIDS, LIFDC, and NFIDC groups. China, mainland (41) stays in the panel. China (351) in 2024 reports 653.2 Mt of cereals primary against 651.4 Mt for China, mainland, so the two series overlap and must not be summed.

The later boundary review found no overlapping years between predecessor and successor codes. See [FAOSTAT_BOUNDARY_AUDIT.md](FAOSTAT_BOUNDARY_AUDIT.md).

## Screening result

The panel has 53,504 rows (209 entities × 4 crops × 64 years). Status counts: no-area-reporting 5,480, missing production 12,726, insufficient baseline 4,994, assessable 30,304. The primary 10% trend flag marks 9,047 assessable rows (29.9%). Share of assessable rows flagged, by crop:

| Crop | ≥5% | ≥10% | ≥20% |
|---|---:|---:|---:|
| Wheat | 0.414 | 0.327 | 0.204 |
| Rice | 0.402 | 0.307 | 0.180 |
| Maize | 0.395 | 0.313 | 0.192 |
| Cereals, primary | 0.366 | 0.262 | 0.140 |

2024 10% rates (wheat 0.281, rice 0.254, maize 0.306, cereals primary 0.243) sit in the same range as 2022–2023. Assessable zeros are retained (wheat 10, rice 17, maize 6, cereals primary 7).

A 10-year log-linear trend therefore flags ordinary year-to-year movement in roughly three of ten assessable country-crop-years. Those rows are a screening list. They are not independent events, causes, loss fractions, or return periods, and they do not revise the H–P envelopes.

## Tests

`python3 -m unittest discover -s tests -p 'test_screen_faostat_disruptions.py' -v` ran 5 tests in 0.003s: OK. The tests check the 3% trend recovery, absence of look-ahead, early and short baselines, inclusive 5/10/20% thresholds, zero production, missing flag M, exclusion of China (351) and World, and parsing of a synthetic normalized ZIP. The log is `results/faostat_screening_tests.txt`.

## Still open

Territorial code successions are reviewed in the boundary audit. The break file clears phantom baselines after a predecessor code ends and does not change assessable flags. An official-only (flag A) rerun has not been produced as a separate panel; the flag composition above is a count on the mixed-flag panel. Event linkage, the documentary search log, and any comparison with envelope losses remain undone. No registry row was added from these flags.

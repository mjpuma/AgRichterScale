# Official-only production screen

Rerun of the 2024 QCL screen on the same FoodShockCascadeModel snapshot, SHA-256 `c5835418c18f9322e7decbd6800f93a216eaae3cdfa31acb08f0518c0c6d6853`, with the reviewed break file. The only change is which production flags may enter a baseline or a target.

Flag A (official) is used. Flags E (estimated), I (imputed by a receiving agency), and X (external organization) are set aside, in the same way flag M was already set aside. The source value and the source flag stay on the row. A set-aside row is `nonofficial_excluded`, not `missing_production`. Harvested area and yield flags are not used for this rule. The historical event registry, H–P envelopes, and figures were not changed.

`--official-only` is off by default, so the mixed-flag screen is unchanged. Six synthetic tests passed, including one that an estimated, imputed, or external target is not flagged and that an estimated year does not enter the official baseline.

## What changes

| | Mixed flags | Official only |
|---|---:|---:|
| Assessable rows | 30,304 | 23,343 |
| Non-official, set aside | 0 | 5,910 |
| Insufficient baseline | 4,994 | 6,045 |
| Missing production | 12,726 | 12,726 |
| 10% flags | 9,047 | 7,081 |
| 10% flags / assessable | 0.299 | 0.303 |

The assessable count falls by 6,961. Of those, 5,910 are E, I, or X targets. The other 1,051 are official observations whose ten-year window no longer has seven usable official years.

Of the 9,047 mixed-flag 10% flags:

| Change | Rows |
|---|---:|
| Still flagged on an official target and an official baseline | 7,012 |
| Dropped because the target is E, I, or X | 1,305 |
| Dropped because the official window is no longer long enough | 680 |
| Dropped because the official baseline moved and the row is still assessable | 50 |
| Newly flagged because the official baseline moved | 69 |

Every retained 10% flag has production flag A. The rate among rows that remain assessable is almost the same crop by crop: wheat 0.327 to 0.330, rice 0.307 to 0.311, maize 0.313 to 0.320, cereals primary 0.262 to 0.262. Setting aside estimated values removes rows. It does not show that official series are quieter.

## 2024 is thinner when only official values count

Among reporting rows that have a production value, the share that is E, I, or X rises toward the end of the file:

| Crop | 2019 | 2024 |
|---|---:|---:|
| Wheat | 0.121 | 0.325 |
| Rice | 0.074 | 0.319 |
| Maize | 0.137 | 0.402 |
| Cereals, primary | 0.106 | 0.382 |

In 2024 the non-A counts are wheat 40 of 123, rice 43 of 135, maize 68 of 169, and cereals primary 68 of 178, of which 61 are estimated. Official 10% rates that year are wheat 0.346 (28 of 81), rice 0.197 (14 of 71), maize 0.304 (28 of 92), and cereals primary 0.257 (28 of 109). The mixed-flag screen had treated those estimated 2024 values as measured. The cutoff stays 2024 because the year is present and the world totals are complete, with the narrower official subset now explicit.

The full panel stays local in `outputs/faostat_historical_screen_v1_official/`. Compact counts are in `results/faostat_screening_v1_official/`. Event linkage is still open. No registry row was added.

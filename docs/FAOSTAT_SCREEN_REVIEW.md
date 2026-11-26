# FAOSTAT production-screen review

*Human check of the 2024 QCL screen, 2026-11-25. Counts below are from the local panels `outputs/faostat_historical_screen_v1_breaks` (all retained flags) and `outputs/faostat_historical_screen_v1_official` (flag A only). Those files are gitignored. This page is a check, not an event catalog and not a figure.*

## What a row is

Each row is one FAO area, one crop item, and one year. The primary rule fits a log-linear trend to the preceding ten years of positive production, with at least seven years and one of them in the previous two years, and flags a harvest at least 10% below that trend. The sensitivities replace the trend with the median of the preceding five or ten years. The percent denominator is that baseline, in tonnes. Cereals, primary includes wheat, rice, maize, and other cereals. Those rows overlap. Do not add them.

A 10% flag is a screening observation. It is not a cause, not a mapped disaster footprint, and not an H–P envelope loss. The registry was not changed.

## Baseline check

The denominator is the number of rows that the named baseline can assess. Official-only removes estimated, imputed, and external production from both the baseline and the target. Flag M was already out of both screens.

| Baseline | All retained flags | Official flag A only |
|---|---:|---:|
| 10-year trend | 9,047 / 30,304 (29.9%) | 7,081 / 23,343 (30.3%) |
| 5-year median | 7,117 / 32,235 (22.1%) | 5,500 / 25,220 (21.8%) |
| 10-year median | 6,613 / 30,399 (21.8%) | 4,864 / 23,401 (20.8%) |

The official-only screen drops rows. It does not make the remaining rows quieter: the trend rate stays at about 30%, and both medians stay near 21%. The median flags fewer rows than the trend because a level has to fall, not merely fall behind a rising trend.

On the 23,343 official rows the trend can assess, the 10-year median is also available. Both rules flag 3,161 rows. The trend alone flags 3,920. The median alone flags 1,670.

## Size check

Size is the baseline in tonnes, on the official-only screen. The rate is 10% flags divided by rows in that size class.

| Baseline size | 10-year trend | 5-year median | 10-year median |
|---|---:|---:|---:|
| Under 10,000 t | 940 / 2,495 (37.7%) | 1,022 / 2,957 (34.6%) | 993 / 2,692 (36.9%) |
| 10,000 to 100,000 t | 1,298 / 3,430 (37.8%) | 1,099 / 3,869 (28.4%) | 1,045 / 3,615 (28.9%) |
| 100,000 to 1 million t | 2,101 / 6,713 (31.3%) | 1,570 / 7,410 (21.2%) | 1,349 / 6,865 (19.7%) |
| 1 million t and over | 2,742 / 10,705 (25.6%) | 1,809 / 10,984 (16.5%) | 1,477 / 10,229 (14.4%) |

Small baselines are flagged more often, especially by a median. Large baselines are flagged less often by a median than by the trend: 14.4% against 25.6% above 1 million tonnes. The all-flags trend has the same shape (31.0%, 36.3%, 31.3%, 25.6% across the four classes).

Absolute tonnes reverse the ranking. On the official trend flags:

- 19 flags have a baseline under 10 t. Their shortfalls sum to 47 t.
- 274 flags have a baseline under 1,000 t. Their shortfalls sum to 42,375 t.
- 940 flags have a baseline under 10,000 t. Their shortfalls sum to 1.19 million t.
- 2,742 flags have a baseline of 1 million t or more. Their shortfalls sum to 7,091 million t.

One national maize row, below, is 118 million t. That single row is about one hundred times the combined shortfall of every official flag whose baseline is under 10,000 t.

## Large shortfalls

Official trend flags. Observed and baseline are tonnes. The fraction is (observed − baseline) / baseline. Cereals, primary in these years contains the maize row. Do not add the two.

- **United States, maize, 1983.** Observed 106,030,000 t, baseline 224,369,687 t, fraction −0.527, shortfall 118,339,687 t, flag A. The 5-year and 10-year medians flag it as well. Cereals, primary for the same country and year is observed 207,657,604 t, baseline 344,792,190 t, fraction −0.398, shortfall 137,134,586 t. That cereals row includes the maize. This is a national total against its own history, not a mapped footprint.
- **United States, maize, 1988.** Observed 125,194,000 t, baseline 189,593,353 t, fraction −0.340, shortfall 64,399,353 t, flag A. Cereals, primary the same year has shortfall 99,006,211 t and fraction −0.324.
- **United States, maize, 2012.** Observed 273,192,400 t, baseline 347,217,421 t, fraction −0.213, shortfall 74,025,021 t, flag A. The percent is milder than 1983. The tonnes are still large because the baseline is large. Cereals, primary the same year has fraction −0.172 and shortfall 74,142,809 t, almost the same tonnes as the maize row.
- **USSR, cereals primary, 1975.** Observed 125,503,000 t, baseline 192,875,473 t, fraction −0.349, shortfall 67,372,473 t, flag A. The USSR code runs through 1991, so this year is inside that code. It is not a successor-state series.
- **China, mainland, cereals primary, 2000.** Observed 405,223,840 t, baseline 469,270,065 t, fraction −0.136, shortfall 64,046,225 t, flag A. The 5-year median flags it. The 10-year median does not. This is a dip below a rising trend, not a drop to a low level. The mixed-flag baseline is 464,259,089 t and the fraction is −0.127, so removing non-official years changes the baseline by about 5 million t and does not change the flag.

## Small denominators

Official trend flags whose baseline is under 2 t. The fraction is −1.00 because the observed harvest is 0 t. The shortfall is the baseline itself.

- **Jamaica, rice, 2021.** Observed 0 t, baseline 0.14 t, shortfall 0.14 t, flag A.
- **Jamaica, rice, 2020.** Observed 0 t, baseline 0.39 t, shortfall 0.39 t, flag A.
- **China, Hong Kong SAR, rice, 1992.** Observed 0 t, baseline 1.18 t, shortfall 1.18 t, flag A. Cereals, primary for the same place and year repeats the same tonnes. It is the same harvest counted again under the aggregate item.
- **Mauritius, rice, 1994.** Observed 0 t, baseline 1.75 t, shortfall 1.75 t, flag A.

These percentages are not comparable to a −0.53 on 224 million t of United States maize. A baseline under 1,000 t should stay in the screen, so the small outcomes are not deleted, and should stay out of any magnitude comparison.

## Check result

The trend flags about three in ten assessable rows, and the medians flag about one in five. Official values do not change those rates. Small baselines produce high percentages and negligible tonnes. Large baselines produce the tonnes, and a trend flags them more often than a 10-year median. The next use of these rows is a documentary link, one registry case at a time, not a sort of the flag list.

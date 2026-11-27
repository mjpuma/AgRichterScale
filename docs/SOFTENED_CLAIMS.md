# Softened loss statements (November 27, 2026)

This pass looked for any place where the repository stated a compilation loss percent, or a crop-failure claim, as fact when no opened page supports it. `ancillary/historical_event_registry.csv` has no loss percents and was not edited.

## The named compilation percents

All six already appear in the repository only as quoted, unverified compilation claims. They live in `ancillary/source_check_v1.csv`, `ancillary/faminewatch_registry_crosswalk_v1.csv`, `docs/SOURCE_CHECK.md`, and `docs/FAMINEWATCH_CROSSWALK.md`. Nothing was rewritten, and each is listed here with the wording to use in the paper.

| Claim in the compilation | Use instead |
|---|---|
| Ireland, potato production fell about 86 percent in 1847 | Potato area planted fell from 656,014 ha in 1846 to 89,000 ha in 1847 (CSO, after Bourke). A production decline of 86 percent is reported in a compilation, but the primary source, Bourke (1960), was not verified. |
| North Korea, 30 percent of grain crops lost in 1995 | Reported as 30 percent in a compilation citing Goodkind and West (2001); the primary source was not verified. FAO/WFP put 1997 maize at 1.14 million tons, 51 percent below 1996. |
| Great Famine, grain production fell 12–44 percent | Reported as 12–44 percent in a compilation citing Jordan (1996); not verified. Schuh (2020), citing Titow, reports Winchester estate yields 36 and 45 percent below average in 1314/15 and 1315/16. |
| New England maize, almost 100 percent lost in 1816 | Severe losses are reported. Jefferson (8 September 1816) expected the Atlantic-states corn crop to be under one-third of an ordinary one. The compilation's "almost 100 percent" (Stommel and Stommel 1983) was not verified. |
| Madras, 63–94 percent of crops lost in 1877 | Davis (2001, Table 1.2) prints the share of the crop saved in ten Madras districts, 6 to 37 percent. The 63–94 percent loss is the complement of that table, and the primary Famine Commission page was not opened. |
| Dust Bowl, wheat −36 and corn −48 percent in the Great Plains | National, not Great Plains. NASS (April 2023) puts 1934 United States wheat 36.4 percent and corn for grain 47.8 percent below their 1921–1930 means. |

## Sentences rewritten in this pass

| File | Before | After |
|---|---|---|
| `docs/MAJOR_FAMINE_LOSS_SOURCES.md`, `docs/DOCUMENTARY_SEARCH_LOG.md`, `ancillary/documentary_search_log_v1.csv` | Tanjore total harvest 25 percent below normal | Reported as well below normal. The GIPE file did not open when re-retrieved, so no percent is adopted. |
| same | Coimbatore crops not above one-quarter of average | Reported via a secondary copy of the district manual as very poor. No fraction is adopted. |
| `docs/MAJOR_FAMINE_LOSS_SOURCES.md`, `docs/DOCUMENTARY_SEARCH_LOG.md` | Wheatcroft: 34.1 Mt in 1920, 27.7 Mt in 1921, a decline of about 18.8 percent | Three competing evaluations. Source block 3a gives 34.11 and 27.67 Mt, and the 1920 half-point is 34.58 Mt. No single percent is adopted. |
| `docs/MAJOR_FAMINE_LOSS_SOURCES.md`, `docs/DOCUMENTARY_SEARCH_LOG.md`, `ancillary/documentary_search_log_v1.csv` | 1932 USSR harvest about 56 Mt (Davies and Wheatcroft) and 50 Mt (Tauger) | Reported in secondary discussion. Neither figure was re-read from its primary page. |
| `docs/HISTORICAL_LOSSES_METHOD.md` | SADC 2015–16 cereals 10.2 percent below average (bold) | Same figure, with a note that the SADC report itself was not opened. |
| `ancillary/event_attributes.csv`, `ancillary/food_disruptions.csv` (notes) | GreatFamine: "Crop failure from excessive rain and cold" | Cold and wet years. Winchester estate yields are reported below average (Schuh 2020 citing Titow). The scale of the grain loss is not verified. |
| same | Laki1783: "Laki volcanic eruption caused crop failures" | Laki eruption. The haze damaged grass in Iceland, and Wieners and Hálfdanarson (2024) find no evidence of harvest failure in west and central Europe in 1783. |
| same | PotatoFamine: "Potato blight destroyed primary food crop" | Potato blight. Potato area fell from 656,014 ha in 1846 to 89,000 ha in 1847 (CSO, after Bourke). The production loss is not verified. |

Numbers that an opened page supports, such as Lin and Yang's 1958–1961 series and the FAO, USDA, and ABS figures in [PRIMARY_SOURCE_PASS.md](PRIMARY_SOURCE_PASS.md), were kept with their stated scope and baseline.

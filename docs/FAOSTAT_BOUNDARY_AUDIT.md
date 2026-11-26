# FAOSTAT territorial continuity audit

Reviewed on the same QCL snapshot as the 2026-11-25 screen: FoodShockCascadeModel `inputs/faostat_bulk_cache/production.zip`, SHA-256 `c5835418c18f9322e7decbd6800f93a216eaae3cdfa31acb08f0518c0c6d6853`, cutoff 2024. The historical event registry, H–P envelopes, and figures were not changed.

The break file is `ancillary/faostat_qcl_breaks_v1.csv` (148 rows, 36 area codes, four crops, SHA-256 `dabf559b646958599d0c5a73677de3b33695412a4d47cb664692267b7a0ed807`). The row-level review is `ancillary/faostat_area_continuity_v1.csv`. A break year is the first year of the new regime. The screener does not let a baseline cross it.

## What the codes already do

Every partial reporting series in this extract is a contiguous block. There are no internal holes in QCL presence. The historical handoffs are different area codes with no shared year:

| Handoff | Last year of the old code | First year of the new codes | Old cereals, t | New cereals, t |
|---|---:|---:|---:|---:|
| USSR to 15 successor republics | 1991 | 1992 | 155,959,908 | 189,190,770 |
| Czechoslovakia to Czechia and Slovakia | 1992 | 1993 | 10,197,895 | 9,636,539 |
| Yugoslav SFR to five republics | 1991 | 1992 | 19,262,686 | 11,451,183 |
| Serbia and Montenegro to Serbia and Montenegro | 2005 | 2006 | 9,533,832 | 8,431,029 |
| Ethiopia PDR to Ethiopia and Eritrea | 1992 | 1993 | 5,034,885 | 5,381,675 |
| Sudan (former) to Sudan and South Sudan | 2011 | 2012 | 5,608,710 | 3,880,000 |
| Belgium-Luxembourg to Belgium and Luxembourg | 1999 | 2000 | 2,445,489 | 2,665,930 |

The USSR and Yugoslavia gaps are harvest variation, not a missing republic. Soviet cereals in 1991 were a poor year, and the 1992 republic sum is larger. Yugoslav cereals were 14.2 Mt in 1990 and 19.3 Mt in 1991; the five 1992 republics sum to 11.5 Mt during the wars. Czechia plus Slovakia, Ethiopia plus Eritrea, and Belgium plus Luxembourg land close to the last predecessor year. Sudan's two successors in 2012 are below the 2011 former-Sudan total, and 2010 was already 3.6 Mt, so the gap is not evidence of a third missing code.

South Sudan's independence date is 9 July 2011. This bulk keeps Sudan (former) through 2011 and starts the two successor codes in 2012. The break year is 2012. The 2011 observation stays on the former code.

Successor codes have no pre-break rows, so their own early years were already unassessable until seven positive years accumulated. Ukraine, the Russian Federation, Sudan, and South Sudan do not inherit a baseline from the predecessor code. That matters for later event linkage: a 2022 Ukraine flag uses the post-1991 Ukraine series only.

## Where a break changes the panel

The rerun with the break file leaves status counts and the 9,047 ten-percent flags unchanged. Coverage, the area audit, and `candidates_10pct.csv` are byte-identical to the no-break run.

The file does change 240 panel rows. All 240 are `no_area_reporting_row` on a predecessor code in the ten years after that code ends: USSR, Yugoslav SFR, and Sudan (former) for all four crops, and Belgium-Luxembourg, Czechoslovakia, Ethiopia PDR, and Serbia and Montenegro for wheat, maize, and cereals primary. Without a break, the rolling window still looked back at the dead series and stored a baseline for a code that no longer reported. The break clears that phantom baseline. It does not change a flag, a status, or an assessable row. On 48 of the 240 rows the stored baseline value itself was removed. On the rest only the baseline count changed, because the window was already too short to qualify.

## Reviewed codes with no break

These codes run through the political date, and the cereal level does not show a new territory being added or removed.

- Germany (79), 1961–2024. Wheat is 13.8–15.6 Mt and cereals about 37 Mt through 1985–1990, then 16.6 and 39.3 Mt in 1991. That is the combined FRG and GDR scale, not a West German series that jumps at unification. This bulk has no separate FRG or GDR code.
- Yemen (249), 1961–2024. Cereals stay near 0.8 Mt across 1990 and return there in 1992 after a one-year drop in 1991. No separate northern or southern code exists. The 1991 drop stays inside one regime.
- Viet Nam (237), 1961–2024. Cereals go from 10.6 Mt in 1975 to 12.2 Mt in 1976. North and South are not separate codes.
- Pakistan (165) and Bangladesh (16) both run from 1961. Pakistan stays near 11–13 Mt across 1971. Bangladesh is already separate before independence.
- Indonesia (101) and Timor-Leste (176) both run from 1961 through 2024. Timor-Leste cereals are about 0.04–0.15 Mt. Indonesia does not step in 1975 or 2002.
- France (68) falls from 61.8 Mt in 2006 to 59.6 Mt in 2007. French Guiana and Réunion stop after 2006 at about 0.02 Mt. Guadeloupe and Martinique have no selected cereal total in those years. France does not absorb the overseas departments.
- China, mainland (41) is 443–456 Mt in 1996–1999. Hong Kong and Macao stay separate codes. The 1997 and 1999 handovers are not merges into the mainland cereal series.
- Malaysia around 1963, Tanzania around 1964, Namibia around 1990, and the Netherlands around 1986 and 2010 show no cereal step. Namibia remains under 0.12 Mt. South Africa's 1992 drop is the regional drought, not a transfer of Namibia.

Palestine reports from 1991 through 2022 and then stops. Nicaragua is missing 2024 only. Marshall Islands and Micronesia (Federated States of) start in 1991 with no predecessor code in this extract. Those are reporting limits, not new regimes, so they are not in the break file.

## What this does not settle

Kosovo has no area code here, so Serbia from 2006 is not tested for whether it includes Kosovo. Yemen cannot be split into its pre-1990 parts. Germany cannot be read as West Germany. Belgium-Luxembourg remains one screened entity through 1999 because the code rule only excludes codes at or above 5000 and China (351). The European Union (27) aggregate is present from 1961 and is already excluded from the country panel; that series is a current-membership construct.

The screener's manifest field `boundary_audit` still says `not certified; supplied break file applied`. That string means a break file was supplied. It does not mean every statistical discontinuity has been closed. Small-series level swings were not used to invent breaks: many minor rice or maize series move by an order of magnitude without a border change.

The official-only rerun is in [FAOSTAT_OFFICIAL_ONLY.md](FAOSTAT_OFFICIAL_ONLY.md). Event linkage is still open. No registry row was added from these flags.

The full rerun panel stays local in `outputs/faostat_historical_screen_v1_breaks/`. The committed manifest is `results/faostat_screening_v1_breaks/manifest.json`.

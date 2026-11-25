# Systematic historical agricultural disruption evidence protocol

**Version 0.1 — 2026-11-25.** This is a prospective screening and extraction protocol, not a claim that the full universe of historical disruptions has been ascertained. It preserves the pre-existing AgRichter catalog and records additions transparently.

## Research questions

1. For a specified crop and region, what is the **potential** production exposed to a complete-disruption end member at a given area magnitude?
2. What is the **observed or historically reconstructed** production or harvested-area anomaly during particular agricultural crises?
3. Under which documented circumstances can observed crop losses be compared quantitatively with a SPAM-2020-based potential exposure estimate?
4. How do nonproduction mechanisms (procurement, displacement, markets and food access) distinguish famine and conflict outcomes from their production effects?

We assess disruptions **regardless of initiating driver**: drought, flood, temperature extremes, crop disease, war, coercive procurement, agricultural policy, displacement and combinations thereof.

## Define the event universe *before* testing against observed losses

**Stage A — fixed legacy catalog.** Retain **all 23** event IDs in `ancillary/event_attributes.csv`, including famines and modern environmental or conflict crises. Every entry remains in `ancillary/historical_event_registry.csv` whether or not a quantitative crop loss is available. Preserve the original event IDs; do not delete observations that complicate the end-member interpretation.

**Stage B — declared additions.** The registry currently adds four candidates: Soviet famine 1932–33 (distinct from 1921–22), Ukraine war from 2022, Sudan conflict from 2023, and South Sudan conflict (starting with 2013, but to be segmented by locality and episode). The addition provenance and date are explicit. They are **not** purported to be an exhaustive conflict census.

**Stage C — systematic expansion (not yet executed).** Establish a reproducible candidate universe by screening:
- FAO GIEWS Country Briefs and annual Crop Prospects / CFSAM reports for every country, **2022–2026**, for conflict-linked crop production, harvested area, or agricultural access disruptions;
- peer-reviewed famine and historical agricultural harvest compilations for *all events in the historical period being studied*, with explicit spatial and temporal scope;
- country-year cereals (FAOSTAT, national statistical services, USDA PSD where consistent) to identify major production anomalies irrespective of known famines, then compare the resulting list with the named events.

**Important:** selecting only Ukraine, Sudan and South Sudan is a *pilot*, not an unbiased sample of contemporary conflicts. Before claiming systematic coverage, record *all candidate countries/events returned by the screening rule*, including those with no usable agriculture data or positive production anomalies. Prefer documenting screened sources and dates, counts included/excluded and reasons, and overlap between named crises.

## Inclusion and exclusion

**Eligible event candidates:** a dated and geographically identified episode with credible documentary indication of changed crop production, harvested area, abandonment, or severe access constraints to agricultural land, **or** a member of the fixed legacy catalog (retained even if impact measurement cannot be identified).

**Observation admissibility:** crop and region identified; event year or crop season identified; unit, observed value and data type clear; authoritative primary or peer-reviewed secondary source cited; baseline specified for anomalies. Production **levels** are admissible data, but are **not** themselves losses. Crop price, famine mortality, food-aid need, procurement and economic damage are separate outcomes, not substitutes for crop tonnage losses.

**Exclusion from quantitative comparison** (not from the registry): no quantified crop measure; missing geographic unit; conflicting crop aggregation that cannot be reconciled; forecast misrepresented as harvest; historical area that cannot be mapped; unresolvable overlapping multiyear series; missing comparable baseline.

Do not require a negative production anomaly for inclusion. Positive national harvests during serious local conflict are critical evidence about scale mismatch, heterogeneity and food access.

## Definitions and measurements

- `P_obs(c,r,t)`: observed production level (metric tons), by crop c, region r and crop year t.
- `P_baseline(c,r,t)`: explicitly estimated non-event/reference production, with method and uncertainty.
- Signed anomaly: `(P_obs - P_baseline) / P_baseline`. **Do not clip positive anomalies.**
- Potential shortfall: `max(0, P_baseline - P_obs)` in tons. Interpret as a **baseline-relative shortfall**, not necessarily a *causally attributable* loss from the named shock.
- Attribution: requires separate identification design or documented estimate, not automatic inference from temporal coincidence.
- End-member exposure: SPAM-2020 production in a declared potential-disruption geographic mask under 100% loss, *not* 2020 production that literally existed during a historical event.
- Area magnitude: `log10(disrupted harvested area / km²)`. Flag whether this uses *measured affected cultivated area*, *affected harvested area*, or *full contemporary crop area within selected administrative units*. Do not mix without separate labels.

For each observation store the original reported level, source URL and page/table, measured population/crop/region, reference window, confidence interval/range if supplied, and whether the value is observational, reconstructed, survey, remotely sensed, statistical model, or illustrative. Convert to kilocalories only with a crop-specific, explicit nutritional conversion; **potato losses are outside the current all-grain model and must be marked separately**.

## Avoid double counting

- Parent events (e.g. ENSO 2015–16) and children (Solomon Islands, East Timor, Laos) share an overlap group, not independent annual observations.
- A multiyear famine is one event with many crop-region-year observations; avoid treating each row as an independent crisis.
- Sudan and South Sudan are distinct post-2011 sovereign countries and should have independently defined spatial series.
- Prefer a single authoritative observation per crop-region-year for primary plots; preserve other estimates for sensitivity and disagreement checks.
- When a modern conflict spans several years, separate annual crop outputs from ongoing damage and localized access restrictions.

## Bias controls and reporting

1. Freeze this protocol and version the inventory before filling in historical outcomes. Log additions and revisions with reasons.
2. Retain all 23 legacy events and every systematically discovered new candidate, including zero/positive anomalies and missing quantitative data.
3. Report an evidence availability matrix: full compatible crop-year series, partial numeric evidence, qualitative only, or not yet screened.
4. Do not interpret the selected illustrative case studies as a representative sample without showing the entire screened denominator.
5. Quantitative overlays must show measurement type, historical baseline, region/crop matching, uncertainty and potential-versus-realized semantics.
6. Run `python scripts/validate_historical_registry.py` to verify foreign keys, required fields and legacy catalog completeness; it cannot validate the truth of historical source content.

## Current files and status

- `ancillary/event_attributes.csv`: fixed legacy metadata (23).
- `ancillary/historical_event_registry.csv`: 27 screened-for-inclusion candidates (23 legacy + four declared additions); **none has been certified as quantitatively matched**.
- `ancillary/historical_loss_evidence_seed.csv`: preliminary extracted source claims, now including Ukraine, Sudan and South Sudan; rows are evidence staging records with direct-source caveats, **not yet harmonized data for the main model**.
- `docs/HISTORICAL_EVENT_AUDIT.md`: concerns about geography, mappings, and historical actual-loss interpretation.
- `docs/MAJOR_FAMINE_LOSS_SOURCES.md`: historical Soviet, Chinese, Indian and Irish source inventory.

## Modern source starting points

- Ukraine, FAO GIEWS country brief: https://www.fao.org/giews/countrybrief/country.jsp?code=UA
- Ukraine FAO survey of agricultural enterprises (2024): https://www.fao.org/countryprofiles/news-archive/detail-news/en/c/1742727/
- Ukraine remote-sensing cropland damage working paper: https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099726309252542219/
- Sudan, FAO GIEWS: https://www.fao.org/giews/countrybrief/country.jsp?code=SDN
- South Sudan, FAO GIEWS: https://www.fao.org/giews/countrybrief/country.jsp?code=SSD&lang=en
- South Sudan 2025 FAO/WFP CFSAM release: https://mafs.gov.ss/mafs-and-un-partners-release-cfsam-report-showing-improved-agricultural-production-amid-persistent-food-insecurity-in-south-sudan/

**Immediate next step:** Screen the complete 2022–26 FAO GIEWS/CFSAM country universe and extract primary crop-year series for the already cataloged famines, with country-region-crop-year as the observational unit. The current four additions are candidates, not a claim of exhaustive modern coverage.

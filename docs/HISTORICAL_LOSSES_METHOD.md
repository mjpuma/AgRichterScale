# AgRichter: historical loss observations alongside potential disruption envelopes

**Research note and implementation protocol — November 25, 2026**

## Scientific position

Keep the H–P envelopes and the 100% disruption scenario as deliberate **end members**. They answer: *Given the spatial arrangement of present-day crop production, how much harvest area and potential production could be affected by a disruption of a specified geographical extent?* They do **not** assert that historical events destroyed all production in designated regions. Upper and lower envelopes represent spatial allocation extremes, not empirical prediction intervals or realized historical event losses.

Add an **independent historical-loss evidence layer**. This answers a different question: *How much crop production or harvested area was demonstrably lost in specific historical events, according to available records?* Actual outcomes should be compared with modeled exposure only when crop set, geography, duration, units, and baseline are harmonized. A 2020 SPAM production counterfactual is not the original event year's counterfactual yield.

These two lenses complement rather than replace one another. Historic observations can establish plausibility, scale and event heterogeneity, but must not be described as validating extreme bounds if their denominators or affected geographies differ.

## Parallel to the recent El Niño–famine study

Esmaili, Puma and colleagues, *El Niño amplified food insecurity in early modern Europe from 1500 to 1800* (Nature Food, 2026), DOI [10.1038/s43016-026-01460-8](https://www.nature.com/articles/s43016-026-01460-8), combine event chronologies and heterogeneous evidence (tithe records, yield-to-seed ratios, and quantity-based local harvest observations), examine distinct mechanisms, and disclose the uneven spatial coverage and uncertainty of historical records. AgRichter should use the same **transparent multi-source principle**, but not directly transfer its ENSO attribution models to AgRichter's disruption-intensity question.

## Data architecture

- **Layer A: theoretical potential.** SPAM 2020 gridded crop production and harvested area, upper/lower H–P envelopes, explicit 100% production-disruption assumption for affected cells. Output is *potential production at risk* or *exposed area*.
- **Layer B: historical evidence.** Event-year actual harvest area, production, yield, losses or crop abandonment from primary statistical reports or peer-reviewed reconstructions. Store raw measurements, units, original sources, geographic scale, time span, baseline and uncertainty. Do not impute missing observed loss as zero.
- **Layer C: case-specific comparison.** Only after matching crop, geographic region, period, commodity aggregation and a documented counterfactual, show the observed **production shortfall** against a corresponding potential-loss construction. The ratio `observed_shortfall / potential_disruption` is **not** interpretable otherwise.
- **Context-only evidence.** Famine mortality, displacement, food prices, monetary damages, and accounts of crop failure can support impact narratives but cannot substitute for crop production tonnage or calories.

The seed evidence table is `ancillary/historical_loss_evidence_seed.csv`. It is a **literature-extraction staging file**, not a finalized model input or a collection of directly comparable values.

### Suggested per-event records

For each event and **crop-region-year**: `event_id`, `event_year`, `source_type`, `region_id`, `crop`, `observed_production`, `production_unit`, `baseline_production`, `baseline_method`, `observed_harvested_area`, `cropland_abandonment`, `estimated_production_shortfall`, `loss_assessment_type`, `source_url`, `page_table`, `uncertainty`, `availability_status`, `mapping_notes`. Event-group assignment and multiyear duplication must be tracked separately.

If complete annual national data exist, retain the full time series rather than extract a single percentage from secondary literature. For historic cases with only local records, keep local observations **local**; do not assign them to an entire country or famine footprint.

## Concrete evidence already located

These are heterogeneous **observations or published estimates**, deliberately *not* directly combined:

1. **Dust Bowl.** A peer-reviewed study estimates mean *modeled climate-driven winter wheat production loss* of **14% across 1931–1940** in the US winter wheat belt, with approximately three-quarters from crop abandonment. It is a decadal model estimate, not an observed one-year loss. [Science Advances paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11290491/). USDA archival reports separately provide **observed 1934 national wheat production of 526,393 thousand bushels**; this is a production datum, *not* a Dust Bowl loss and not confined to the Dust Bowl geography. [USDA historical statistics](https://esmis.nal.usda.gov/sites/default/release-files/vh53wv748/pn89d9043/k930c0593/GenCrRep-12-18-1940.pdf).
2. **Southern African 2015–2016 drought/ENSO.** A peer-reviewed study reports **SADC 2015–2016 cereal production 10.2% below the preceding five-year average**, citing SADC (2016). This is a regional production anomaly, not a quantified ENSO causal effect. [NHESS (2020)](https://nhess.copernicus.org/articles/20/1187/2020/nhess-20-1187-2020.html). FAO separately reports a **27% subregional maize decline in 2015**; different year, commodity and baseline, and should not be pooled with the SADC value. [FAO (2015)](https://www.fao.org/newsroom/detail/El-Ni%C3%B1o-lowers-early-production-outlook-in-Southern-Africa/fr).
3. **Vanuatu, Cyclone Pam, 2015.** The government/development-partner post-disaster assessment estimates **VT 4,175 million of combined crop-sector damage and economic losses** (within approximately VT 6.1 billion across agriculture). Its crops include perennial cash and subsistence crops; **this is not a physical grain loss in tonnes or kcal**. It establishes a documented observed impact and supports a narrative case, not direct placement on a grain production-loss axis. [Post-Disaster Needs Assessment, agriculture section](https://www.dfat.gov.au/sites/default/files/post-disaster-needs-assessment-cyclone-pam.pdf).
4. **1816 Year Without a Summer.** Documentary and historical scholarship reports geographically variable harvest failure and social crises; no homogeneous quantitative loss for the entire region was established in this extraction. [Historical study](https://onlinelibrary.wiley.com/doi/10.1111/geoj.12191/full).

These cases demonstrate that the evidence ladder can begin without pretending to have a complete numerical series for all 23 events.

## Peer-review safeguards

1. **Pre-specify eligibility.** Define a complete inventory of 23 cataloged events and independently record which have quantitative historical grain evidence, qualitative documentation only, or no usable historical loss data. Avoid cherry-picking only severe or convenient cases.
2. **Separate measured, reconstructed, forecast, modeled and assumed.** Preserve evidence class and source vintage; never combine forecasts with realized harvest totals without labeling the distinction.
3. **Define counterfactual explicitly.** A production shortfall `max(0, baseline - observed)` depends on baseline: prior year, five-year mean, detrended fitted time series, or a study's weather counterfactual are **not equivalent**. Retain signed anomalies separately; positive anomalies are informative and should not disappear.
4. **Avoid mixing spatial scales.** Subnational losses should not be compared with national 100%-exposure estimates; only harmonized subregions/crops can be used in quantitative comparisons. Adjust for shifting boundaries where possible.
5. **Avoid mixing event duration.** For multiyear drought and famine, report annual observations and cumulative impacts as separate measures. Do not add overlapping ENSO and regional subevents as statistically independent events.
6. **Disaggregate processes.** Harvested area lost or abandoned, crop yield anomaly, and crop production shortfall are distinct. A famine caused substantially by conflict, distributional failures or policy cannot be reduced to a crop-loss percentage.
7. **Robustness.** Vary baseline window (e.g., three- versus five-year and trend-adjusted where supportable), affected footprint and loss severity; report ranges, not false precision.
8. **Do not overstate validation.** Existing event calculations currently sum SPAM 2020 crop area and production across listed country/state polygons. No historical damage fraction is implemented there. The seed observations below are not yet integrated with those calculations.

## Suggested manuscript structure

**Main Figure:** (a) theoretical H–P lower and upper potential loss bounds over AgRichter magnitude; (b) observed/reconstructed historical event-level agricultural losses **only where compatible in units and spatial scope**, with explicit uncertainty and distinct symbols for observational and modeled estimates; (c) sensitivity of selected matched events to severity and footprint.

**Supplementary Table:** all 23 cataloged events, primary source(s), affected crops/years, exact geography, crop loss or documented absence, mapping quality, baseline choice and admissibility status. Blank entries explicitly mean **not yet established**, not negligible losses.

**Recommended manuscript sentence:** “The AgRichter H–P envelopes characterize end-member *potential* production disruptions under explicitly specified spatial and complete-loss assumptions. We complement these counterfactual bounds with a separately sourced compilation of historical crop-production and harvested-area losses, retaining heterogeneity in geography, commodity, timing and baseline and restricting quantitative comparisons to harmonized event observations.”

## Outstanding work before making quantitative overlay claims

- Extract source workbooks, reconstruct actual country/subdivision footprints, and resolve parser and country/subdivision assignment issues identified in `docs/HISTORICAL_EVENT_AUDIT.md`.
- Obtain primary SADC 2016 and country-level crop series; recover archival Dust Bowl regional yields and harvested areas; identify agricultural yield/harvest observations for older crises where documentary sources permit.
- Build an observed-baseline estimator with uncertainty; calculate matched crop-region-year 2020 exposure as a **counterfactual reference**, not historical production.
- Add numeric cross-comparison only for matched cases with sufficient information. Do not convert Vanuatu monetary damages to kcal or use death counts as production loss.

**Status:** Research design and initial sourced examples; no new simulations, external SPAM rerun or event-level quantitative matching in this step.

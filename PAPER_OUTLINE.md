# The AgRichter Scale — Science Article Outline

**Target journal:** Science (Research Article) — fallbacks: Nature Food, Nature Sustainability, Science Advances
**Working title options:**

1. "An earthquake scale for food: quantifying a millennium of agricultural disruptions"
2. "The AgRichter Scale: a universal magnitude framework for agricultural catastrophes"
3. "How big can a harvest failure get? Bounding global food-production losses across seven orders of magnitude"

**Last updated:** 2026-07-14 (post code-audit revision: corrected country→grid mapping and USDA unit conversion; all figures regenerated)

---

## One-sentence summary

We introduce a logarithmic magnitude scale for agricultural disruptions, bound the production loss achievable at every magnitude with global crop data, and show that historical events already approach the point where losses would exhaust global grain reserves.

## Abstract (structure, ~125 words)

- **Problem:** Agricultural catastrophes (droughts, wars, volcanic events, pests) are described with incommensurable, qualitative vocabularies; there is no common physical scale.
- **Approach:** Define AgRichter magnitude M_D = log10(disrupted harvest area, km²). Using SPAM 2020 5-arcmin data, derive Harvest–Production (H-P) envelopes: rigorous upper/lower bounds on caloric production loss for any disrupted area.
- **Findings:** 24 historical events spanning 1315–present and five orders of magnitude fall within the envelopes; the largest (Great Drought 1876–78, M 6.5) approaches the "Total Stocks" line where losses exceed global ending stocks. Annual exceedance modeling implies a return period of roughly a century for events breaching the 3-month global supply buffer.
- **Implication:** A common scale enables comparing, communicating, and stress-testing food-system risk the way seismology does for earthquakes.

## Significance / press angle

"A Richter scale for famine." The scale is memorable, the events are famous (Dust Bowl, Irish Famine, Chinese famine of 1959), and the punchline — modern reserves buffer roughly a magnitude-6 disruption and history has already produced M 6.5 — is quantitative and sobering.

---

## Main text structure

### 1. Introduction (~600 words)

- Food-system shocks are increasing in salience (climate volatility, conflict, export bans) but lack a common metric; contrast with seismology (Richter/moment magnitude), hurricanes (Saffir–Simpson), tornadoes (EF).
- Existing approaches measure impacts (famine mortality, IPC phases, price spikes) which conflate hazard with vulnerability; we need a *physical* hazard scale.
- Introduce the two-part framework:
  - **AgRichter magnitude** M_D = log10(A_H/km²) — the "size" of the disruption footprint.
  - **H-P envelope** — the physical bound on caloric loss for that footprint, computed from the actual spatial distribution of yields (greedy best-case/worst-case cell selection on SPAM 2020).
- Preview: 24 events over 700 years; thresholds anchored to USDA consumption and stocks; national envelopes; exceedance probability.

### 2. The AgRichter scale and historical events (Fig. 1)

- Define magnitude; justify area (observable, crop-agnostic, comparable across centuries) vs production loss (depends on where/when).
- Event catalog: 24 events, 1315–present, mapped to modern SPAM footprints of affected countries/states ("modern-equivalent footprint" framing — what the event's footprint would disrupt today).
- Events span M 1.2 (Solomon Islands 2015) to M 6.5 (Great Drought 1876–78); driver categories (climate, conflict/policy, pest/disease, geophysical, compound).
- Key observation: all five driver classes reach M ≥ 5.4; no driver class "owns" the tail.

### 3. H-P envelopes bound the possible loss (Fig. 2)

- Construction: sort grid cells by caloric yield density; cumulate best-first (upper bound) and worst-first (lower bound); mathematical properties (origin/endpoint convergence, dominance, conservation) verified at runtime.
- Interpretation: at a given magnitude, the spread between bounds is the "location leverage" — the same footprint can cost 100× more calories if it hits breadbaskets.
- Consumption-anchored thresholds (USDA PSD 2019–21 means, aligned to SPAM 2020): 1-month supply, 3-month supply, total ending stocks — expressed in kcal so they live on the same axis.
- Punchline: upper-bound losses cross the 3-month global buffer near M ≈ 5.5–6; several historical events (China 1959, Great Drought 1876) sit at or above that crossing.
- Envelopes narrow (converge) at the top: at continental scales, *where* the disruption hits stops mattering — size alone determines loss.

### 4. National fingerprints (Fig. 3)

- Country-level envelopes (USA, China, India, Brazil, France, Egypt; gallery of 10 in SI) with nationally scaled thresholds.
- Typology: "wide" envelopes (USA, Brazil — heterogeneous agriculture, high location-leverage) vs "stiff" envelopes (Egypt — concentrated along the Nile, no diversification buffer).
- Policy reading: envelope width is a measurable diversification indicator; stiff-envelope countries cross their stock thresholds at much smaller magnitudes.
- (Optional link to biophysical-redundancy literature, e.g., Fader et al. typologies.)

### 5. The fragility gap: probability of systemic breach (Fig. 4)

- From 64 years of detrended USDA production shocks, calibrate a conservative exponential tail for annual loss exceedance; map through the envelope upper bound to magnitude space; bootstrap CI.
- Report return periods for crossing the 3-month buffer (~10² yr) and total stocks (~5×10² yr) — the "fragility gap" between what history produces and what reserves can absorb.
- Honesty about extrapolation: tail is calibrated, not observed; framed as a conservative (heavier-than-normal) bound; methodology validation in SI (Fig. S3).

### 6. Discussion (~500 words)

- What the scale enables: cross-era comparison, common language for early-warning (IPC ↔ magnitude mapping), reinsurance/stress-testing, scenario design (e.g., "a nuclear-winter scenario is an M 7").
- Limitations (state plainly):
  - Modern-equivalent footprints, not historical reconstructions; SPAM 2020 is a snapshot (single-year, single-cropping-season accounting).
  - Binary disruption assumption (100% loss inside footprint) makes envelopes bounds, not estimates; partial-intensity extension is future work.
  - Global thresholds ignore trade frictions and stock distribution; national thresholds currently scale global stock-to-use ratios by production share (upgrade path: country-level PSD stocks — see improvements).
  - Tail probabilities are model-based beyond M ≈ 6.
- Future: sub-national envelopes, multi-year events, non-grain crops, coupling to trade models.

---

## Figures (main)

| Fig | File | Content | Status |
|---|---|---|---|
| 1 | `results/figure1_agrichter_scale.png/.svg` | AgRichter scale, 4 crop panels, 24 events, 5 driver classes | Regenerated 2026-07-14 (mapping fix). **Consider redesign — see improvements #1** |
| 2 | `results/figure2_hp_envelopes.png/.svg` | H-P envelopes + events + consumption thresholds, 4 crops | Regenerated (mapping + unit fix; threshold lines moved from ~10¹² to ~10¹⁵ kcal — now correct) |
| 3 | `results/figure3_country_EnvelopesMap.png` (map version) or `figure3_country_envelopes.png` (6-panel) | National envelopes / fingerprints | Regenerated; choose map vs panel version |
| 4 | `results/figure4_risk.png` (styled) / `figure4_risk_probability_*.png` | Exceedance probability, return periods, fragility gap | Regenerated (event mapping fix) |

## Supplementary

- **Fig. S1** Global SPAM 2020 production/harvest maps (`figureS1_global_maps.png`).
- **Fig. S2** National fingerprint gallery + comparative normalized envelopes (log-log & semi-log from M 3).
- **Fig. S3** Risk-model validation: trend break, shock distribution, normal vs exponential tail, safety-margin justification.
- **Fig. S4** Tail-model sensitivity (`figureS4_tail_sensitivity.png`): conservative exp vs MLE exp vs GPD (POT) vs Gaussian; bootstrap CIs on return periods at both policy thresholds. GPD shape ξ = −0.38 (bounded tail); KS p-values 0.75/0.79 — all models fit the observed record, they diverge only in extrapolation, which is the argument for the conservative bound.
- **Fig. S5** Fractional-loss sensitivity (`figureS5_fractional_loss_sensitivity.png`): events re-plotted at 100/50/25% loss intensity; scaled points provably stay inside the envelope (concave upper / convex lower bounds through origin); Great Drought and China 1959 stay ≥ 3-month buffer even at 50%.
- **Table S1** National thresholds (`tableS1_supplementary.tex`, corrected 1000-MT conversion, 2019–2021 window).
- **SI text:** envelope algorithm + mathematical properties + convergence validator; event catalog with sources (`ancillary/food_disruptions.csv` has full citations); threshold derivation; unit conventions (SPAM MT → kcal via per-crop caloric content; USDA PSD in 1000 MT).
- **Data/code availability:** GitHub repo; SPAM 2020 V2r0; USDA PSD (Jan 2026 pull).

---

## Improvement opportunities (ranked, pre-submission)

### Must fix (correctness) — DONE 2026-07-14

1. ~~ISO3→FIPS mapping guessed codes from first two letters (Ireland→Iran, Switzerland→China, UK→Gabon, Liechtenstein→Liberia), corrupting the footprints of all multi-country events.~~ Fixed in `spatial_mapper.py`; authoritative static map now takes priority; figures regenerated.
2. ~~USDA PSD unit error: thresholds converted as MT but data is 1000 MT — all threshold lines in Figs 2–3 and Table S1 were 1000× too low.~~ Fixed in `config.py`; regenerated.
3. ~~`Config` default USDA averaging window was 1990–2020 while docs/fig4/tableS1 use 2019–2021.~~ Default changed to (2019, 2021); Figs 2–3 regenerated.
4. ~~Hardcoded `/Users/mjp38/...` root_dir in fig1/fig2 scripts; duplicate `test_events_pipeline` module blocked pytest collection.~~ Fixed (portability + 634 tests now collect).

### High value (referee-proofing) — 4 problems resolved 2026-07-14

3. ~~**Tail model robustness.**~~ DONE: Fig. S4 adds MLE-exponential, GPD (POT), and Gaussian alongside the conservative exponential, with bootstrap CIs on return periods; bootstrap RNG now seeded (42) in fig4 scripts for reproducibility.
4. ~~**Event loss basis consistency.**~~ DONE: `EventCalculator` now converts each crop with its own caloric factor (same basis as envelopes). Allgrain event losses shifted ≤ ±4%; magnitudes unchanged; single-crop values unchanged.
5. ~~**100%-loss / footprint honesty.**~~ DONE (sensitivity): Fig. S5 shows 100/50/25% loss-intensity trajectories; SI caption states the modern-equivalent-footprint convention and proves bounds hold under scaling. Remaining sub-item: deduplicate ENSO 2015–16 parent vs child events if a frequency analysis is added.
6. ~~**Envelope conservation warning.**~~ DONE: root cause was float32 SPAM columns — cumsum over ~1M cells at totals ~7×10⁶ km² rounds away increments < 0.5 km² (24-bit mantissa). `build_envelope` now promotes inputs to float64; all conservation checks pass at rtol 1e-10.
7. **Fig. 1 redesign.** Plotting area vs log(area) is a tautology — every point sits on the y=10^x line by construction, so the figure carries only a 1-D ranking. Replace with a timeline (year vs magnitude, marker = driver class, size = duration) or a magnitude "ladder" graphic. This is the most likely figure a referee will call contentless.
8. **National thresholds from country-level PSD data.** Current scaling (global thresholds × production share) assumes every country holds the global stock-to-use ratio. USDA PSD has per-country consumption and stocks; using them directly is more defensible and creates an interesting result (China's actual stocks are ~half of global stocks; Egypt is import-dependent).
9. **Event catalog transparency.** Publish `historical_events_computed.csv` as a supplementary dataset with per-event footprint definitions and source citations; state the "modern-equivalent footprint" assumption prominently in Methods.

### Nice to have

10. Partial-intensity disruptions (fractional loss rasters) to move from bounds toward estimates.
11. Multicropping/SPAM technology-share sensitivity (`_TA` vs `_TI/_TR` layers); summing `*_A` columns can double-count physical area in multi-crop cells.
12. Country typology quantified: envelope width index (e.g., upper/lower ratio at M 4) tabulated for all producer countries.
13. Repo hygiene (partially done 2026-07-14): removed broken `agrichter/cli` package + `pyproject.toml` entry point; `load_usda_wide()` deduplicated into `agrichter/data/usda_wide.py`; READMEs rewritten to match the real script inventory; test images removed from `agrichter/output/`; fixed China GDAM code (45→48; 45 is Central African Republic) in unit tests. Remaining: ~40 orphaned package modules (envelope.py v1, multi_tier_*, national_*_analyzer, pipeline/, output/, processing/) could be archived in a dedicated commit before the code-availability release.

---

## Suggested Methods subsections

1. Data (SPAM 2020 V2r0; USDA PSD Jan 2026; event sources)
2. Magnitude definition and event footprint mapping (country/state → FIPS → grid cells)
3. H-P envelope construction and mathematical properties
4. Consumption-based thresholds (units: PSD 1000 MT × 10⁹ g × kcal/g)
5. National envelopes and threshold scaling
6. Exceedance-probability model (detrending, shock distribution, bootstrap, tail assumptions)
7. Runtime validation (convergence validator) and reproducibility

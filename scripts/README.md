# AgRichter Scripts

Standalone scripts that call the `agrichter` library to generate the publication figures and tables. Run all scripts from the repository root:

```bash
python3 scripts/<script_name>.py
```

Outputs (PNG + SVG) are written to `results/`; single-crop/single-country variants go to `results/individual_plots/`.

## Main figures

| Script | Output | Content |
|---|---|---|
| `fig1_agrichter_scale.py` | `figure1_agrichter_scale.*` | AgRichter Scale (4 crop panels, historical events); also exports `results/historical_events_computed.csv` |
| `fig2_hp_envelopes.py` | `figure2_hp_envelopes.*` | Global H-P envelopes with events and consumption-based thresholds |
| `fig3_country_envelopes.py` | `figure3_country_envelopes.*` | Country H-P envelopes (6-panel) + per-country individual plots; exports `country_stocks_verification_table.csv` |
| `fig4_risk_probability.py` | `figure4_risk_probability_*.*` | Annual exceedance probability / "Fragility Gap" (full version with events and diagnostics) |
| `fig4_overhaul.py` | `fig4_components/figure4_risk_v4.*` | Styled ("ultra-bold") Figure 4 + standalone legend used in the composed main figure |

## Supplementary figures and tables

| Script | Output | Content |
|---|---|---|
| `figS1_global_maps.py` | `figureS1_global_maps.*` | SPAM 2020 production and harvest-area maps |
| `figS2_comparative.py` | `figureS2_comparative_envelopes_*.*` | Comparative national envelopes (log-log and semi-log) |
| `figS2_overhaul.py` | `supplement_components/*` | National fingerprint gallery components |
| `figS3_envelope_bounds.py` | `figureS3_comparative_envelopes.*` | "Stiff" vs "wide" envelope typology |
| `figS3_methodology_validation.py` | `figureS3_methodology_validation.*` | Detrending, shock distribution, and exponential-tail justification for Figure 4 |
| `figS4_tail_sensitivity.py` | `figureS4_tail_sensitivity.*` | Tail-model sensitivity (conservative exp. vs MLE exp. vs GPD vs Gaussian) with bootstrap CIs on return periods |
| `figS5_fractional_loss.py` | `figureS5_fractional_loss_sensitivity.*` | Event positions under 100% / 50% / 25% loss-intensity assumptions |
| `tableS1_thresholds.py` | `tableS1_supplementary.tex` | National food-security thresholds table |
| `tableS2_verification.py` | `tableS2_verification.tex` / `.csv` | Verification & validation summary (envelope invariants, SPAM vs USDA cross-check, event containment); exits non-zero on failure |

## Utilities

| Script | Purpose |
|---|---|
| `refresh_usda_data.py` | Re-download / refresh USDA PSD input data |
| `verify_fig4_logic.py` | Standalone sanity checks for the Figure 4 risk calculation |

## Notes

- Data prerequisites: SPAM 2020 V2r0 CSVs at repo root, `USDAdata/`, and `ancillary/` (event definitions, country-code table).
- Figure 1's events CSV is an input to `figS5_fractional_loss.py`; run `fig1_agrichter_scale.py` first if `results/historical_events_computed.csv` is missing.

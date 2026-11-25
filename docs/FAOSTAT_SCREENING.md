# FAOSTAT systematic production screening

Implementation of [protocol v1.0](HISTORICAL_EVENT_SELECTION_PROTOCOL.md). This is a separate research pipeline, not a replacement for AgRichter's theoretical potential-loss model.

## Reproduce

From repository root, Python 3.9+ (standard library only). First inspect the [FAOSTAT QCL release](https://www.fao.org/faostat/en/#data/QCL) and determine its latest sufficiently complete harvest year. Record release metadata. Example syntax below uses 2024 as an illustrative cutoff, **not** a verified current release boundary:

```bash
python -m unittest discover -s tests -p 'test_screen_faostat_disruptions.py'
python scripts/screen_faostat_disruptions.py \
  --archive data_raw/faostat/qcl_snapshot.zip --download \
  --end-year 2024 --release-note "Replace with verified QCL edition and cutoff rationale" \
  --out results/historical_screen_v1
```

Use a new output directory for every run; the ZIP is reused without overwriting. Alternatively obtain the normalized ZIP from FAO separately and omit --download. Default CSV encoding is UTF-8 with optional BOM; if the release is verified to use Latin-1 pass --encoding latin-1. Do not commit large mutable raw downloads or unvalidated generated tables. Archive the exact snapshot and manifest together.

Optional --breaks path/to/breaks.csv expects area_code,item_code,break_year,reason. A break year is the first year of the new series regime; no baseline crosses that boundary. The absence of a break file is **not** evidence of geographic or statistical continuity. Boundary and classification audits are required before scientific use.

## Outputs

- source_rows.csv: selected original values, units, flags, notes, item labels and codes, including aggregate reporting areas.
- area_audit.csv: area codes, provisional entity/aggregate classification and observed QCL reporting span. The code-based rule excludes codes >=5000 and combined China (351); independently verify against release-specific FAO country/region codelists, including former entities and territories. Do not sum overlapping areas.
- production_panel.csv: all four crops for each provisional reporting entity and year from 1961 through cutoff; no-area-reporting, missing production, insufficient baseline, and assessable statuses are distinct. No-area-reporting does not imply that a country did not exist. Yield is kept in original units. Rice is paddy and FAO primary cereals is not yet verified equivalent to the model's allgrain definition. Component crops and primary cereals overlap; do not sum them.
- candidates_10pct.csv: subset flagged by the primary trend baseline. Not a catalog of independent events or causal estimates. Main and sensitivity flags are also retained in the full panel.
- coverage.csv: annual crop-specific counts by status and thresholds. Denominator for anomaly prevalence is assessable observations, not all panel rows.
- manifest.json: snapshot SHA-256, cutoff, release note, encoding, break file hash (if supplied), counts and scientific-status flags.

The primary estimator uses the preceding ten years of positive production (minimum seven; at least one in previous two years) and no target or future observations. Five- and ten-year median sensitivities include nonnegative observations with positive median. Missing-flag M is excluded from baseline and target while original source value is retained; other flags are kept for subsequent quality sensitivity. Threshold flags are inclusive. An early-year baseline is missing rather than inferred from later observations.

## Release gate / remaining work

1. Run synthetic tests, then acquire and parse the real archive. Confirm schema, actual item/element codes and labels, flags/units and complete-year cutoff.
2. Check area membership, aggregate overlaps, historical territorial boundaries, reporting activity, and breaks; rerun with reviewed break file. Inspect flags and coverage by area, year, crop and region, including missing and imputed observations.
3. Compare full-panel sensitivity to alternative baselines and official-only/estimated-data exclusions. Investigate sparse crops and small denominators; do not delete inconvenient positive outcomes.
4. Build documentary search log and event-to-observation links, preserving all legacy cases. Reconstruct archival cases under their separate evidence standard.
5. Audit legacy spatial mappings and match historical production and affected crop area before revising figures. No inferred event attribution, loss fraction or return periods from the anomaly flags alone.

**Status:** implementation and synthetic tests committed; live FAOSTAT download/run and independent execution of the tests not verified at commit time. No additions to the 27-event registry and no figure changes.

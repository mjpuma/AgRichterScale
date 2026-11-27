# FAOSTAT systematic production screening

Implementation of [protocol v1.0](HISTORICAL_EVENT_SELECTION_PROTOCOL.md). This is a separate research pipeline, not a replacement for AgRichter's theoretical potential-loss model.

## Reuse the FoodShockCascadeModel snapshot (preferred)

The sibling repository's `main/fetch_faostat_bulk.py` documents the cached archive `inputs/faostat_bulk_cache/production.zip` and extracted `inputs/Production_Crops_Livestock_E_All_Data_(Normalized).csv`. These are likely local-only; the cache itself was not found in the tracked Git tree. No data need be copied or downloaded if either exists. Run from AgRichterScale:

```bash
git pull --ff-only
python3 -m unittest discover -s tests -p 'test_screen_faostat_disruptions.py'
python3 scripts/inspect_faostat_snapshot.py
```

The inspector prefers the sibling ZIP, falls back to the extracted CSV, and records the input checksum, observed year range, per-year commodity coverage, source labels, units and flags in `results/faostat_snapshot_audit.json`. If directories are not siblings, pass the exact input file path as the first argument. If strict UTF-8 decoding fails, confirm the source encoding and retry with `--encoding latin-1` (the FSC downloader uses Latin-1 for reading its CSV). The inspector never changes the snapshot or replaces an existing report; choose a different `--out` for another snapshot.

Review the audit and determine the cutoff before screening. The `--archive` option in the screening script accepts either original ZIP or extracted CSV, records the checksum of the supplied input, and never modifies it. For example, **only if the audit confirms 2024 coverage and that cutoff is appropriate**:

```bash
python3 scripts/screen_faostat_disruptions.py \
  --archive ../FoodShockCascadeModel/inputs/faostat_bulk_cache/production.zip \
  --end-year 2024 --release-note "Existing FSC snapshot; release date and completeness to be documented" \
  --out results/historical_screen_v1
```

If using the extracted CSV instead, supply its path in quotes and the verified encoding. Do not treat maximum year as proof of completeness. The 2026-11-25 verification of the local FoodShockCascadeModel cache, including the 2024 cutoff decision, is recorded in [FAOSTAT_SNAPSHOT_FINDINGS.md](FAOSTAT_SNAPSHOT_FINDINGS.md). A later bulk still needs its own audit before a new cutoff is used.

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

The primary estimator uses the preceding ten years of positive production (minimum seven; at least one in previous two years) and no target or future observations. Five- and ten-year median sensitivities include nonnegative observations with positive median. Missing-flag M is excluded from baseline and target while the original source value is retained. Pass `--official-only` to set aside E, I, and X in the same way; the default screen keeps them. The comparison is in [FAOSTAT_OFFICIAL_ONLY.md](FAOSTAT_OFFICIAL_ONLY.md). Threshold flags are inclusive. An early-year baseline is missing rather than inferred from later observations.

## Release gate / remaining work

1. Run synthetic tests, then acquire and parse the real archive. Confirm schema, actual item/element codes and labels, flags/units and complete-year cutoff.
2. Check area membership, aggregate overlaps, historical territorial boundaries, reporting activity, and breaks; rerun with reviewed break file. Inspect flags and coverage by area, year, crop and region, including missing and imputed observations.
3. Compare full-panel sensitivity to alternative baselines and official-only/estimated-data exclusions. Investigate sparse crops and small denominators; do not delete inconvenient positive outcomes.
4. Build documentary search log and event-to-observation links, preserving all legacy cases. Reconstruct archival cases under their separate evidence standard.
5. Audit legacy spatial mappings and match historical production and affected crop area before revising figures. No inferred event attribution, loss fraction or return periods from the anomaly flags alone.

**Status:** synthetic tests passed and the local FoodShockCascadeModel QCL snapshot was screened through 2024. Schema, units, flags, and the cutoff are in [FAOSTAT_SNAPSHOT_FINDINGS.md](FAOSTAT_SNAPSHOT_FINDINGS.md). Territorial code successions are reviewed in [FAOSTAT_BOUNDARY_AUDIT.md](FAOSTAT_BOUNDARY_AUDIT.md); the break file does not change assessable flags. The official-only rerun is recorded in [FAOSTAT_OFFICIAL_ONLY.md](FAOSTAT_OFFICIAL_ONLY.md). Trend, median, and small-denominator checks are in [FAOSTAT_SCREEN_REVIEW.md](FAOSTAT_SCREEN_REVIEW.md). A [documentary search log](DOCUMENTARY_SEARCH_LOG.md) records a disposition for all 27 registry cases. [Observation links](FAOSTAT_OBSERVATION_LINKS.md) name the reporting area in force, or an explicit no-row, and do not use the 10% flags. The [archival reconstruction](ARCHIVAL_RECONSTRUCTION.md) keeps pre-1961 figures out of the QCL panel. A [FamineWatch crosswalk](FAMINEWATCH_CROSSWALK.md) records which registry cases appear in that famine chronology. The [footprint inventory](EVENT_FOOTPRINT_AUDIT.md) lists every workbook place. The parser now keeps every country and binds each subdivision to its own GDAM code. Figures were not redrawn. A [source check](SOURCE_CHECK.md) re-reads the unverified crop sentences and does not enter them as event losses. No additions to the 27-event registry and no envelope changes.

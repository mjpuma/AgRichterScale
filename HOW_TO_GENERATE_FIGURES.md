# How to Generate Publication Figures

## Simple Instructions

You have **4 standalone scripts** that generate the publication figures using **real SPAM data** and **real historical events**.

### Generate Individual Figures

```bash
# Figure 1: Global Maps (8 panels)
python3 fig1_global_maps.py

# Figure 2: AgRichter Scale (4 panels)
python3 fig2_agrichter_scale.py

# Figure 3: H-P Envelopes (4 panels)
python3 fig3_hp_envelopes.py

# Figure 4: Country H-P Envelopes (4 panels)
python3 fig4_country_envelopes.py
```

### Output

All figures are saved to `results/`:
- `results/figure1_global_maps.png`
- `results/figure2_agrichter_scale.png` (+ individual crop files)
- `results/figure3_hp_envelopes.png` (+ individual crop files)
- `results/figure4_country_envelopes.png` (+ individual country files)

## What These Scripts Do

### ✅ Use Real Data
- Load SPAM 2020 production and harvest data from CSV files
- Load real historical events from Excel files (`ancillary/DisruptionCountry.xls`, etc.)
- Calculate actual event losses using `EventCalculator`
- Calculate real H-P envelopes using `HPEnvelopeCalculatorV2`

### ❌ Do NOT Use Fake Data
- No `create_sample_events_data()` (creates only 3 fake events)
- No `create_sample_envelope_data()` (creates fake envelopes)
- No synthetic or sample data

## Execution Time

- **Figure 1**: ~2-3 minutes (loads SPAM data for 4 crops)
- **Figure 2**: ~5-10 minutes (loads data + calculates 23 real events)
- **Figure 3**: ~15-20 minutes (loads data + calculates events + envelopes)
- **Figure 4**: ~15-20 minutes (same as Figure 3, but for 4 countries)

## If Something Goes Wrong

### Old Scripts Are Archived

All old `generate_*.py` scripts are safely archived in:
```
archive/old_generate_scripts/
```

If you need to recover something, they're still there!

### Check the Logs

Each script has detailed logging. Look for:
- `Loading real historical events...` ✅ Good
- `Using sample data` ❌ Bad (shouldn't happen)
- `Calculated losses for X events` ✅ Good (should be ~23 events)

### Common Issues

1. **"Event Excel files not found"**
   - Check that `ancillary/DisruptionCountry.xls` exists
   - Check that `ancillary/DisruptionStateProvince.xls` exists

2. **"SPAM data not found"**
   - Check that SPAM CSV files exist in correct directories
   - See `FIGURE_DEPENDENCIES.md` for required files

3. **Script runs but only shows 3 events**
   - This means it fell back to synthetic data
   - Check the logs for warnings
   - File an issue - this shouldn't happen!

## Dependencies

See `FIGURE_DEPENDENCIES.md` for complete list of:
- Python packages needed
- AgRichter modules used
- Data files required

## Project Structure

```
AgRichter2025/
├── fig1_global_maps.py          ← Run these 4 scripts
├── fig2_agrichter_scale.py      ←
├── fig3_hp_envelopes.py         ←
├── fig4_country_envelopes.py    ←
│
├── agririchter/                 ← Existing package (don't modify)
│   ├── core/
│   ├── data/
│   ├── analysis/
│   └── visualization/
│
├── results/                     ← Output figures go here
├── ancillary/                   ← Event Excel files
├── spam2020V2r0_*/             ← SPAM data
│
└── archive/                     ← Old scripts (safe to ignore)
    └── old_generate_scripts/
```

## Summary

**Just run the 4 `fig*.py` scripts. They use real data. That's it.**

If you need the old scripts, they're in `archive/old_generate_scripts/`.

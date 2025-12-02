# Self-Contained Publication Figures

This folder contains everything needed to generate the 4 publication figures.

## Structure

```
publication_figures/
├── fig1_global_maps.py          # Figure 1: Global maps
├── fig2_agrichter_scale.py      # Figure 2: AgRichter Scale
├── fig3_hp_envelopes.py         # Figure 3: H-P Envelopes
├── fig4_country_envelopes.py    # Figure 4: Country envelopes
├── run_all.py                   # Run all 4 figures
├── README.md                    # This file
│
└── lib/                         # Self-contained library
    ├── config.py                # Configuration
    ├── grid_manager.py          # Data loading
    ├── event_calculator.py      # Event calculations
    ├── envelope_calculator.py   # Envelope calculations
    ├── agririchter_viz.py       # AgRichter visualization
    ├── envelope_viz.py          # Envelope visualization
    ├── map_generator.py         # Map generation
    └── ... (other modules)
```

## Usage

### Run Individual Figures

```bash
cd publication_figures
python3 fig1_global_maps.py
python3 fig2_agrichter_scale.py
python3 fig3_hp_envelopes.py
python3 fig4_country_envelopes.py
```

### Run All Figures

```bash
cd publication_figures
python3 run_all.py
```

## Output

Figures are saved to `../results/`:
- `../results/figure1_global_maps.png`
- `../results/figure2_agrichter_scale.png`
- `../results/figure3_hp_envelopes.png`
- `../results/figure4_country_envelopes.png`

## Data Requirements

The scripts expect data files in the parent directory:
- `../spam2020V2r0_global_production/...`
- `../spam2020V2r0_global_harvested_area/...`
- `../USDAdata/...`
- `../ancillary/...`

## Benefits

✅ **Self-contained** - No dependency on main agririchter package
✅ **Transparent** - All code visible in lib/ folder
✅ **Debuggable** - Easy to trace through code
✅ **Portable** - Can move this folder anywhere

## Real Data Only

These scripts use **REAL data only**:
- Real SPAM 2020 production and harvest data
- Real historical events from Excel files
- Real envelope calculations

**NO synthetic or sample data is used.**

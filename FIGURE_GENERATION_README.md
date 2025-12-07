# Publication Figure Generation Scripts

## Overview

This directory contains 4 standalone scripts to generate the publication figures using **real SPAM data** and the working AgRichter visualization modules. No copying, no fake data.

## Scripts

### Individual Figure Scripts

1. **`scripts/fig1_hp_envelopes.py`** - Figure 1: H-P Envelopes (4 panels)
   - Generates H-P envelope figures for 4 crops
   - Shows harvest area disruption vs production loss bounds
   - Includes historical events with labels
   - Output: `results/figure1_hp_envelopes.png` + individual crop files

2. **`scripts/fig2_country_envelopes.py`** - Figure 2: Country H-P Envelopes (4 panels)
   - Generates country-level H-P envelopes for USA, China, India, Brazil
   - Uses allgrain data filtered by country
   - Includes country-specific historical events
   - Output: `results/figure2_country_envelopes.png` + individual country files

3. **`scripts/fig3_agrichter_scale.py`** - Figure 3: AgRichter Scale (4 panels)
   - Generates AgRichter Scale figures for 4 crops
   - Shows magnitude (M_D = log10(A_H)) vs production loss
   - Includes historical events with labels
   - Output: `results/figure3_agrichter_scale.png` + individual crop files

4. **`scripts/figS1_global_maps.py`** - Figure S1: Global Production and Harvest Area Maps (8 panels)
   - Generates maps for wheat, maize, rice, and allgrain
   - Shows production and harvest area for each crop
   - Output: `results/figureS1_global_maps.png`

### Master Script

**`generate_all_4_figures.py`** - Runs all 4 figure generation scripts in sequence

## Usage

### Generate Individual Figures

```bash
# Figure 1 (H-P Envelopes)
python3 scripts/fig1_hp_envelopes.py

# Figure 2 (Country Envelopes)
python3 scripts/fig2_country_envelopes.py

# Figure 3 (AgRichter Scale)
python3 scripts/fig3_agrichter_scale.py

# Figure S1 (Global Maps)
python3 scripts/figS1_global_maps.py
```

### Generate All Figures

```bash
python3 generate_all_4_figures.py
```

## What These Scripts Do

### Real Data Processing

- Load SPAM 2020 data from CSV files
- Use GridDataManager for proper coordinate alignment
- Calculate envelopes using HPEnvelopeCalculatorV2
- Load historical events from `ancillary/food_disruptions.csv`

### No Cheating

- ✅ No copying existing figures
- ✅ No fake/synthetic data
- ✅ Uses actual visualization classes from `agririchter/visualization/`
- ✅ Calculates envelopes from scratch using real SPAM data

### Journal-Quality Output

- 300 DPI resolution PNG
- Scalable Vector Graphics (SVG) for manual editing
- Professional font sizes (16pt base, 18pt titles)
- Proper axis labels and legends

## Output Files

All figures are saved directly to the `results/` directory in both PNG and SVG formats.

**Note:** `figureS1_global_maps.svg` is extremely large (~700MB) due to the high number of grid cells plotted. It is recommended to use the PNG version for general viewing.

```
results/
├── figure1_hp_envelopes.png
├── figure1_wheat_individual.png
├── figure1_maize_individual.png
├── figure1_rice_individual.png
├── figure1_allgrain_individual.png
├── figure2_country_envelopes.png
├── figure2_usa_individual.png
├── figure2_china_individual.png
├── figure2_india_individual.png
├── figure2_brazil_individual.png
├── figure3_agrichter_scale.png
├── figure3_wheat_individual.png
├── figure3_maize_individual.png
├── figure3_rice_individual.png
├── figure3_allgrain_individual.png
└── figureS1_global_maps.png
```

## Dependencies

These scripts use the AgRichter package modules:

- `agririchter.core.config` - Configuration management
- `agririchter.data.grid_manager` - SPAM data loading
- `agririchter.analysis.envelope_v2` - Envelope calculation
- `agririchter.visualization.agririchter_scale` - AgRichter Scale plots
- `agririchter.visualization.hp_envelope` - H-P Envelope plots
- `agririchter.visualization.global_map_generator` - Global maps

## Notes

- Figure generation may take several minutes due to large SPAM datasets
- Envelope calculation (Figures 3 & 4) is computationally intensive
- Individual crop/country figures are generated first, then combined into 4-panel figures
- All scripts include detailed logging to track progress

## Troubleshooting

If a script fails:

1. Check that SPAM data files exist in the expected locations
2. Verify that the `results/` directory exists (created automatically)
3. Check the log output for specific error messages
4. Ensure all AgRichter package modules are properly installed

## Clean Slate

To start fresh and regenerate all figures:

```bash
# Remove old figures
rm results/figure*.png

# Regenerate all
python3 generate_all_4_figures.py
```

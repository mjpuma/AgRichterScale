# Publication Figure Generation

This directory contains the scripts to generate publication-quality figures for the AgriRichter analysis.

## Setup

Ensure you have the required dependencies installed. The scripts rely on:
- `agririchter` package (local)
- `pandas`, `numpy`, `matplotlib`, `cartopy`, `geopandas`, `shapely`

## Generating Figures

Run the following scripts from the project root directory:

### Figure 1: Global Production and Harvest Area Maps
Generates an 8-panel figure showing global production and harvest area for wheat, maize, rice, and all grains.
```bash
python3 scripts/fig1_global_maps.py
```
**Output:** `results/figure1_global_maps.png`

### Figure 2: AgriRichter Scale
Generates a 4-panel figure showing the AgriRichter scale (Magnitude vs. Production Loss) for all 4 crop categories, with historical events overlaid.
```bash
python3 scripts/fig2_agrichter_scale.py
```
**Output:** `results/figure2_agrichter_scale.png` (+ individual crop plots)

### Figure 3: H-P Envelopes
Generates a 4-panel figure showing the Harvest-Production (H-P) envelopes. These envelopes define the upper and lower bounds of production loss for a given harvest area disruption.
```bash
python3 scripts/fig3_hp_envelopes.py
```
**Output:** `results/figure3_hp_envelopes.png` (+ individual crop plots)

### Figure 4: Country-Specific H-P Envelopes
Generates a 4-panel figure showing H-P envelopes for specific key countries: USA, China, India, and Brazil.
```bash
python3 scripts/fig4_country_envelopes.py
```
**Output:** `results/figure4_country_envelopes.png` (+ individual country plots)

## Key Updates for Publication

1.  **Real Data Only**: All scripts now strictly use the full SPAM 2020 dataset. No synthetic or sample data is used.
2.  **High Resolution**: Envelopes are calculated with 1000-point resolution and tight tolerances (0.01) for smooth, precise curves.
3.  **Publication Styling**: Figures use increased font sizes (16pt base), bold titles, and journal-standard layouts.
4.  **Verification**: The code includes robust import checks and mathematical validation for the envelopes (monotonicity, dominance).

## Troubleshooting

If you encounter segmentation faults or import errors (often related to `cartopy` or `geopandas` binaries), ensure your environment libraries (`libgeos`, `proj`) match your python package versions. Using `conda` to install these geospatial dependencies is often more reliable than `pip`.


# AgriRichter Project (2025)

## Overview

The AgriRichter project introduces a new scale for quantifying global agricultural production disruptions, analogous to the Richter scale for earthquakes. This repository contains the Python codebase for calculating AgriRichter magnitudes, generating H-P (Harvest-Production) envelopes, and producing the figures for the associated publication.

The core analysis relies on:
-   **SPAM 2020 (V2r0)** global agricultural data (Production and Harvest Area).
-   **Historical Event Data** (e.g., Dust Bowl, Great Famine) mapped to grid cells.
-   **H-P Envelope Framework** to define upper and lower bounds of production loss for a given disrupted area.

## Repository Structure

-   **`agririchter/`**: The main Python package containing the core logic.
    -   `core/`: Configuration and constants.
    -   `data/`: Data loading, grid management, and spatial mapping.
    -   `analysis/`: H-P envelope calculation and event processing.
    -   `visualization/`: Plotting and map generation modules.
-   **`scripts/`**: **Main entry point for figure generation.** Contains standalone scripts to generate publication figures.
-   **`results/`**: Output directory for generated figures and tables.
-   **`ancillary/`**: Helper data files (country codes, event definitions, etc.).
-   **`spam2020V2r0_*/`**: SPAM 2020 dataset directories (must be present locally).
-   **`archive/`**: Old scripts, documentation, and outputs (ignored by Git).

## Figure Generation

The publication figures can be generated using the scripts in the `scripts/` directory. These scripts use the `agririchter` package to process real SPAM data and generate high-quality outputs.

### Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### Running the Scripts

Run the following commands from the root of the repository to generate the figures. The output files (PNG and SVG) will be saved to the `results/` directory.

**Figure 1: AgriRichter Scale**
Shows the relationship between AgriRichter Magnitude ($M_D$) and harvest area for historical events.
```bash
python3 scripts/fig1_agrichter_scale.py
```

**Figure 2: H-P Envelopes (The Innovation)**
Shows the Harvest-Production envelopes and historical events for wheat, maize, rice, and all grains.
```bash
python3 scripts/fig2_hp_envelopes.py
```

**Figure 3: Country H-P Envelopes**
Shows H-P envelopes and scaled thresholds for specific countries (USA, China, India, Brazil, etc.).
```bash
python3 scripts/fig3_country_envelopes.py
```

**Figure 4: Risk Probability Curve**
Calculates the annual exceedance probability of agricultural disruptions based on the historical record.
```bash
python3 scripts/fig4_risk_probability.py
```

**Figure S1: Global Maps (Supplementary)**
Generates global production and harvest area maps for all crops.
```bash
python3 scripts/figS1_global_maps.py
```

**Figure S2: Comparative Vulnerability**
Generates comparative plots of normalized envelopes for multiple countries.
```bash
python3 scripts/figS2_comparative.py
```

**Figure S3: Resilience Typologies**
Compares "stiff" vs "wide" envelopes to highlight national agricultural resilience.
```bash
python3 scripts/figS3_envelope_bounds.py
```

## Documentation

Detailed documentation for the codebase and methodologies can be found in `archive/docs/` if needed for reference, although the current codebase in `agririchter/` is the source of truth.

## License

[Insert License Information Here]



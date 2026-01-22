# AgRichter Project (2025)

## Overview

The AgRichter project introduces a new scale for quantifying global agricultural production disruptions, analogous to the Richter scale for earthquakes. This repository contains the Python codebase for calculating AgRichter magnitudes, generating H-P (Harvest-Production) envelopes, and producing the figures for the associated publication.

The core analysis relies on:
-   **SPAM 2020 (V2r0)** global agricultural data (Production and Harvest Area).
-   **Historical Event Data** (e.g., Dust Bowl, Great Famine) mapped to grid cells.
-   **H-P Envelope Framework** to define upper and lower bounds of production loss for a given disrupted area.

## Repository Architecture

The repository is structured into two main components:

1.  **`agrichter/` (Core Library)**: This is the engine room of the project. It is structured as a modular Python package containing the reusable scientific logic, data loaders, and mathematical algorithms (e.g., H-P envelope calculation, magnitude mapping). Core scripts should **not** be modified for one-off figure adjustments; they provide the "source of truth" for the analysis.
    -   `core/`: Unified configuration, caloric constants, and unit conversions.
    -   `data/`: Robust loaders for SPAM 2020 gridded data and USDA FAS PSD time-series.
    -   `analysis/`: The mathematical core, including the `HPEnvelopeCalculator` and `EventCalculator`.
    -   `visualization/`: Base plotting modules that ensure consistent styling and dimensional analysis across all outputs.

2.  **`scripts/` (Output Generation)**: These are standalone execution scripts that "call" the `agrichter` library to generate specific publication assets. They handle the "layout" logic (e.g., multi-panel subplots, specific country selections) for the journal.

## Repository Structure

-   **`agrichter/`**: The main Python package containing the core logic (see Architecture above).
-   **`scripts/`**: Main entry point for figure generation. Contains scripts to generate publication figures.
-   **`results/`**: Output directory for generated figures and tables.
    -   Main figures (e.g., `figure1_agrichter_scale.png`) are located in the root of this folder.
    -   Individual component plots (e.g., single-crop or single-country versions) are saved in the `results/individual_plots/` subfolder.
-   **`USDAdata/`**: Refreshed USDA PSD data (Jan 2026 update).
-   **`ancillary/`**: Helper data files (country codes, event definitions, etc.).
-   **`archive/`**: Obsolete data, legacy scripts, and old outputs (ignored by Git to keep the repo clean).

## Figure Generation

The publication figures are generated using the scripts in the `scripts/` directory. These scripts import the `agrichter` library to process data and generate consistent, high-quality outputs.

### Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### Running the Scripts

Run the following commands from the root of the repository to generate the figures. The output files (PNG and SVG) will be saved to the `results/` directory.

**Figure 1: AgRichter Scale**
Shows the relationship between AgRichter Magnitude ($M_D$) and harvest area for historical events.
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

Detailed documentation for the codebase and methodologies can be found in `archive/docs/` if needed for reference, although the current codebase in `agrichter/` is the source of truth.

## License

[Insert License Information Here]



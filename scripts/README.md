# AgriRichter Scripts

This directory contains the Python scripts for generating the publication figures and running the analysis pipeline.

## Figure Generation Scripts (Main)

These standalone scripts generate the specific figures for the publication (Nature/Science style).

-   **`fig1_hp_envelopes.py`**: Generates Figure 1 (H-P Envelopes).
-   **`fig2_country_envelopes.py`**: Generates Figure 2 (Country H-P Envelopes).
-   **`fig3_agrichter_scale.py`**: Generates Figure 3 (AgRichter Scale).
-   **`figS1_global_maps.py`**: Generates Figure S1 (Global Maps).
-   **`fig2_comparative.py`**: Generates the comparative national vulnerability plot.

Usage:
```bash
python3 scripts/fig1_hp_envelopes.py
```

## Analysis Pipeline

-   **`run_agririchter_analysis.py`**: A comprehensive command-line interface to execute the complete AgriRichter events analysis pipeline (data loading, calculation, results export).

Usage:
```bash
python3 scripts/run_agririchter_analysis.py --all
```

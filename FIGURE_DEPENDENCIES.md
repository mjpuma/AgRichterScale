# Figure Generation Scripts - Dependencies

## Overview

Each figure generation script has specific dependencies from the AgRichter package and standard Python libraries.

## Quick Reference Table

| Figure | Script                      | Key Modules                                                                   | Complexity |
| ------ | --------------------------- | ----------------------------------------------------------------------------- | ---------- |
| Fig 1  | `fig1_global_maps.py`       | Config, GridDataManager, GlobalMapGenerator                                   | Simple     |
| Fig 2  | `fig2_agrichter_scale.py`   | + SpatialMapper, EventsProcessor, EventCalculator, AgriRichterScaleVisualizer | Medium     |
| Fig 3  | `fig3_hp_envelopes.py`      | + HPEnvelopeCalculatorV2, HPEnvelopeVisualizer                                | Complex    |
| Fig 4  | `fig4_country_envelopes.py` | Same as Fig 3                                                                 | Complex    |

**Legend:**

- Simple: Basic data loading and visualization
- Medium: Adds event calculation from Excel files
- Complex: Adds envelope calculation (computationally intensive)

---

## Figure 1: Global Maps (`fig1_global_maps.py`)

### Standard Library Dependencies

- `logging` - Logging functionality
- `sys` - System-specific parameters and functions
- `pathlib.Path` - File path handling

### Third-Party Dependencies

- `matplotlib.pyplot` - Plotting
- `matplotlib` (mpl) - Matplotlib configuration

### AgRichter Package Dependencies

- `agririchter.core.config.Config` - Configuration management
  - File: `agririchter/core/config.py`
- `agririchter.data.grid_manager.GridDataManager` - SPAM data loading and management
  - File: `agririchter/data/grid_manager.py`
- `agririchter.visualization.global_map_generator.GlobalMapGenerator` - Global map generation
  - File: `agririchter/visualization/global_map_generator.py`

### Data Files Required

- `spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv`
- `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv`
- `USDAdata/usda_psd_1961to2021_{crop}.csv` (for wheat, maize, rice)

---

## Figure 2: AgRichter Scale (`fig2_agrichter_scale.py`)

### Standard Library Dependencies

- `logging` - Logging functionality
- `sys` - System-specific parameters and functions
- `pathlib.Path` - File path handling

### Third-Party Dependencies

- `matplotlib.pyplot` - Plotting
- `matplotlib` (mpl) - Matplotlib configuration
- `pandas` (pd) - Data manipulation

### AgRichter Package Dependencies

- `agririchter.core.config.Config` - Configuration management
  - File: `agririchter/core/config.py`
- `agririchter.data.grid_manager.GridDataManager` - SPAM data loading and management
  - File: `agririchter/data/grid_manager.py`
- `agririchter.data.spatial_mapper.SpatialMapper` - Spatial mapping and country code handling
  - File: `agririchter/data/spatial_mapper.py`
- `agririchter.data.events.EventsProcessor` - Event definition processing
  - File: `agririchter/data/events.py`
- `agririchter.analysis.event_calculator.EventCalculator` - Event loss calculation
  - File: `agririchter/analysis/event_calculator.py`
- `agririchter.visualization.agririchter_scale.AgriRichterScaleVisualizer` - AgRichter Scale visualization
  - File: `agririchter/visualization/agririchter_scale.py`

### Data Files Required

- `spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv`
- `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv`
- `USDAdata/usda_psd_1961to2021_{crop}.csv` (for wheat, maize, rice, allgrain)
- `ancillary/DisruptionCountry.xls` - Country-level event definitions
- `ancillary/DisruptionStateProvince.xls` - State/province-level event definitions
- `ancillary/CountryCode_Convert.xls` - Country code mappings
- `ancillary/food_disruptions.csv` - Event metadata (optional, for event types)

---

## Figure 3: H-P Envelopes (`fig3_hp_envelopes.py`)

### Standard Library Dependencies

- `logging` - Logging functionality
- `sys` - System-specific parameters and functions
- `pathlib.Path` - File path handling

### Third-Party Dependencies

- `matplotlib.pyplot` - Plotting
- `matplotlib` (mpl) - Matplotlib configuration
- `pandas` (pd) - Data manipulation

### AgRichter Package Dependencies

- `agririchter.core.config.Config` - Configuration management
  - File: `agririchter/core/config.py`
- `agririchter.data.grid_manager.GridDataManager` - SPAM data loading and management
  - File: `agririchter/data/grid_manager.py`
- `agririchter.data.spatial_mapper.SpatialMapper` - Spatial mapping and country code handling
  - File: `agririchter/data/spatial_mapper.py`
- `agririchter.data.events.EventsProcessor` - Event definition processing
  - File: `agririchter/data/events.py`
- `agririchter.analysis.event_calculator.EventCalculator` - Event loss calculation
  - File: `agririchter/analysis/event_calculator.py`
- `agririchter.analysis.envelope_v2.HPEnvelopeCalculatorV2` - H-P envelope calculation
  - File: `agririchter/analysis/envelope_v2.py`
- `agririchter.visualization.hp_envelope.HPEnvelopeVisualizer` - H-P envelope visualization
  - File: `agririchter/visualization/hp_envelope.py`

### Data Files Required

- `spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv`
- `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv`
- `USDAdata/usda_psd_1961to2021_{crop}.csv` (for wheat, maize, rice, allgrain)
- `ancillary/DisruptionCountry.xls` - Country-level event definitions
- `ancillary/DisruptionStateProvince.xls` - State/province-level event definitions
- `ancillary/CountryCode_Convert.xls` - Country code mappings
- `ancillary/food_disruptions.csv` - Event metadata (optional)

---

## Figure 4: Country H-P Envelopes (`fig4_country_envelopes.py`)

### Standard Library Dependencies

- `logging` - Logging functionality
- `sys` - System-specific parameters and functions
- `pathlib.Path` - File path handling

### Third-Party Dependencies

- `matplotlib.pyplot` - Plotting
- `matplotlib` (mpl) - Matplotlib configuration
- `pandas` (pd) - Data manipulation

### AgRichter Package Dependencies

- `agririchter.core.config.Config` - Configuration management
  - File: `agririchter/core/config.py`
- `agririchter.data.grid_manager.GridDataManager` - SPAM data loading and management
  - File: `agririchter/data/grid_manager.py`
- `agririchter.data.spatial_mapper.SpatialMapper` - Spatial mapping and country code handling
  - File: `agririchter/data/spatial_mapper.py`
- `agririchter.data.events.EventsProcessor` - Event definition processing
  - File: `agririchter/data/events.py`
- `agririchter.analysis.event_calculator.EventCalculator` - Event loss calculation
  - File: `agririchter/analysis/event_calculator.py`
- `agririchter.analysis.envelope_v2.HPEnvelopeCalculatorV2` - H-P envelope calculation
  - File: `agririchter/analysis/envelope_v2.py`
- `agririchter.visualization.hp_envelope.HPEnvelopeVisualizer` - H-P envelope visualization
  - File: `agririchter/visualization/hp_envelope.py`

### Data Files Required

- `spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv`
- `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv`
- `USDAdata/usda_psd_1961to2021_allgrain.csv` (or individual crop files)
- `ancillary/DisruptionCountry.xls` - Country-level event definitions
- `ancillary/DisruptionStateProvince.xls` - State/province-level event definitions
- `ancillary/CountryCode_Convert.xls` - Country code mappings
- `ancillary/food_disruptions.csv` - Event metadata (optional)

---

## Complete File Path Reference

### Core Modules

- `agririchter/core/config.py` - Configuration management (used by all)
- `agririchter/core/constants.py` - Constants and conversion factors
- `agririchter/core/utils.py` - Utility functions

### Data Modules

- `agririchter/data/grid_manager.py` - SPAM data loading (used by all)
- `agririchter/data/spatial_mapper.py` - Country/region mapping (Figs 2,3,4)
- `agririchter/data/events.py` - Event processing (Figs 2,3,4)
- `agririchter/data/loader.py` - Data loading utilities

### Analysis Modules

- `agririchter/analysis/event_calculator.py` - Event loss calculation (Figs 2,3,4)
- `agririchter/analysis/envelope_v2.py` - H-P envelope calculation (Figs 3,4)
- `agririchter/analysis/envelope_builder.py` - Envelope building utilities (used by envelope_v2)
- `agririchter/analysis/convergence_validator.py` - Envelope validation

### Visualization Modules

- `agririchter/visualization/global_map_generator.py` - Global maps (Fig 1)
- `agririchter/visualization/agririchter_scale.py` - AgRichter Scale (Fig 2)
- `agririchter/visualization/hp_envelope.py` - H-P Envelopes (Figs 3,4)
- `agririchter/visualization/coordinate_mapper.py` - Coordinate mapping utilities
- `agririchter/visualization/plots.py` - General plotting utilities

---

## Common Dependencies Summary

### All Scripts Use:

- Python 3.x
- `matplotlib` (for plotting)
- `pandas` (for data manipulation, except Figure 1)
- `pathlib` (for file path handling)
- `logging` (for progress tracking)

### AgRichter Core Modules Used:

- `Config` - All scripts
- `GridDataManager` - All scripts
- `SpatialMapper` - Figures 2, 3, 4 (for event calculation)
- `EventsProcessor` - Figures 2, 3, 4 (for event loading)
- `EventCalculator` - Figures 2, 3, 4 (for event loss calculation)
- `HPEnvelopeCalculatorV2` - Figures 3, 4 (for envelope calculation)

### Visualization Modules Used:

- `GlobalMapGenerator` - Figure 1
- `AgriRichterScaleVisualizer` - Figure 2
- `HPEnvelopeVisualizer` - Figures 3, 4

---

## Installation Requirements

To run these scripts, ensure you have:

```bash
# Python packages
pip install matplotlib pandas numpy xlrd openpyxl

# AgRichter package (from repository root)
pip install -e .
```

## Data Requirements

All scripts require:

1. SPAM 2020 data files in the correct directory structure
2. USDA PSD data files in `USDAdata/` directory
3. Ancillary files in `ancillary/` directory (for event-based figures)

## Notes

- **Figure 1** is the simplest with fewest dependencies
- **Figures 2, 3, 4** have similar dependencies due to event calculation requirements
- All scripts use **real SPAM data** and **real historical events** (no fake/sample data)
- Event calculation requires Excel file reading (`xlrd` engine for `.xls` files)

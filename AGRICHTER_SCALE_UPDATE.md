# AgRichter Scale Update - October 2025

## Overview

Updated the AgriRichter Scale visualization to better align with the familiar earthquake Richter scale convention, making it more intuitive for readers.

## Key Changes

### 1. Name Change
- **Old**: "AgriRichter Scale"
- **New**: "AgRichter Scale"
- Shorter, cleaner name that maintains the connection to Richter scale

### 2. Axis Orientation (Major Change)
Following the earthquake Richter scale convention:

**Old Layout:**
- X-axis: Magnitude (M_D = log₁₀(A_H))
- Y-axis: Production Loss (kcal) - logarithmic scale

**New Layout (Richter-style):**
- **X-axis**: AgRichter Magnitude (M_D = log₁₀(A_H))
- **Y-axis**: Harvest Area Disrupted (km²) - logarithmic scale

**Rationale:**
- Matches how people are familiar with seeing Richter scale
- Magnitude increases to the right (like earthquake magnitude)
- Physical impact (area affected) increases upward
- More intuitive: larger magnitude → larger area disrupted

### 3. Dual-Scale Concept

The AgRichter Scale works in conjunction with the H-P (Harvest-Production) Envelope:

**AgRichter Scale:**
- Shows magnitude vs. harvest area disrupted
- Analogous to earthquake Richter scale
- Emphasizes the spatial extent of disruption
- Quick assessment of event size

**H-P Envelope:**
- Shows harvest area vs. production loss
- Reveals production efficiency/intensity
- Shows relationship between area and food loss
- Detailed impact assessment

**Together they provide:**
- **Magnitude assessment** (AgRichter Scale)
- **Impact assessment** (H-P Envelope)
- **Complementary views** of agricultural disruption

Think of them as:
- AgRichter Scale = "How big was the event?" (like earthquake magnitude)
- H-P Envelope = "What was the actual impact?" (production loss details)

## Technical Implementation

### Files Modified

1. **`agririchter/visualization/agririchter_scale.py`**
   - Updated axis orientation (magnitude on X, harvest area on Y)
   - Fixed axis label formatting (proper LaTeX subscripts)
   - Added new plotting methods for Richter-style layout
   - Updated documentation

2. **`agririchter/data/loader.py`**
   - Fixed Excel file loading (changed from openpyxl to xlrd for .xls files)
   - Critical bug fix that was preventing event data from loading

3. **`agririchter/pipeline/events_pipeline.py`**
   - Added event processing step (was missing before)
   - Now properly processes event definitions from Excel sheets
   - Fixed event data flow through pipeline

### New Features

- **Richter-style threshold lines**: Horizontal lines at key harvest area levels
- **Improved event labeling**: Uses adjustText for non-overlapping labels
- **Better visual hierarchy**: Events stand out as red circles
- **Clearer axis labels**: Proper mathematical notation

## Generated Outputs

For each crop (wheat, rice, allgrain), the pipeline now generates:

```
outputs/{crop}/
├── data/
│   └── events_{crop}_spam2020.csv
├── figures/
│   ├── agririchter_scale_{crop}.png
│   ├── agririchter_scale_{crop}.svg
│   ├── agririchter_scale_{crop}.eps
│   └── agririchter_scale_{crop}.jpg
└── reports/
    ├── pipeline_summary_{crop}.txt
    └── performance_{crop}.txt
```

## Usage

### Generate All Figures

```bash
# All crops
python generate_all_figures.py

# Specific crops
python generate_all_figures.py --crops wheat rice

# Custom output directory
python generate_all_figures.py --output-dir my_outputs
```

### Example Output

The AgRichter Scale now shows:
- **X-axis**: Magnitude from 2 to 7 (log₁₀ scale)
- **Y-axis**: Harvest area from 100 km² to 10 million km² (log scale)
- **Events**: Plotted as red circles with labels
- **Thresholds**: Horizontal lines showing severity levels

## Event Statistics (Wheat Example)

From the latest run:
- **Total events processed**: 21
- **Events with data**: 12
- **Total harvest area loss**: 236.8 million hectares
- **Magnitude range**: 2.85 to 5.86
- **Largest event**: Drought 1876-1878 (M=5.86, 73 million ha)

## Next Steps

1. ✅ Fix event loading (DONE)
2. ✅ Update AgRichter Scale orientation (DONE)
3. ✅ Fix axis labels (DONE)
4. ✅ Generate figures for all crops (DONE)
5. ⏳ Generate H-P Envelope figures (IN PROGRESS)
6. ⏳ Generate global maps (IN PROGRESS)
7. ⏳ Commit to GitHub

## Notes

- The H-P Envelope visualization is complementary to the AgRichter Scale
- Together they provide a complete picture of agricultural disruption
- The Richter-style orientation makes the scale more accessible to general audiences
- All figures are generated in multiple formats (PNG, SVG, EPS, JPG) for publication

## References

- Original MATLAB implementation: `AgriRichter_Events.m`
- SPAM 2020 data: Global harvest area and production
- Event definitions: `ancillary/DisruptionCountry.xls`, `DisruptionStateProvince.xls`

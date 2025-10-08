# AgriRichter User Guide

## Overview

AgriRichter is a Python-based tool for analyzing historical agricultural disruption events and their impacts on global food production. This guide will help you install, configure, and run the AgriRichter analysis pipeline.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB of available RAM
- 10GB of disk space for data files

### Install Dependencies

1. Clone or download the AgriRichter repository
2. Navigate to the project directory
3. Install required packages:

```bash
pip install -r requirements.txt
```

Required packages include:
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- geopandas >= 0.10.0
- shapely >= 1.8.0
- cartopy >= 0.20.0
- openpyxl >= 3.0.0
- scipy >= 1.7.0

### Verify Installation

Test that the package is properly installed:

```bash
python -c "import agririchter; print('AgriRichter installed successfully')"
```

## Quick Start

### Basic Usage

Run the complete analysis pipeline for wheat:

```bash
python scripts/run_agririchter_analysis.py --crop wheat
```

This will:
1. Load SPAM 2020 gridded data
2. Calculate losses for 21 historical events
3. Generate publication-quality figures
4. Export results to CSV files
5. Create a summary report

### Command-Line Options

```bash
python scripts/run_agririchter_analysis.py [OPTIONS]
```

**Options:**

- `--crop CROP`: Crop type to analyze (choices: wheat, rice, allgrain)
  - `wheat`: Analyze wheat events only
  - `rice`: Analyze rice events only
  - `allgrain`: Analyze all grain crops (wheat, rice, maize, barley, sorghum, millet, oats, rye)

- `--output-dir DIR`: Output directory for results (default: `outputs/`)

- `--spam-dir DIR`: Directory containing SPAM 2020 data files (default: current directory)

- `--ancillary-dir DIR`: Directory containing ancillary data files (default: `ancillary/`)

- `--log-level LEVEL`: Logging level (choices: DEBUG, INFO, WARNING, ERROR; default: INFO)

- `--validate`: Run validation checks comparing with MATLAB outputs

- `--skip-figures`: Skip figure generation (useful for data processing only)

### Example Commands

**Analyze wheat events:**
```bash
python scripts/run_agririchter_analysis.py --crop wheat --output-dir outputs/wheat
```

**Analyze all grains with debug logging:**
```bash
python scripts/run_agririchter_analysis.py --crop allgrain --log-level DEBUG
```

**Process rice data with custom data paths:**
```bash
python scripts/run_agririchter_analysis.py \
    --crop rice \
    --spam-dir /path/to/spam2020 \
    --ancillary-dir /path/to/ancillary \
    --output-dir /path/to/outputs
```

**Run with validation checks:**
```bash
python scripts/run_agririchter_analysis.py --crop wheat --validate
```

## Output Files

### Directory Structure

After running the pipeline, outputs are organized as follows:

```
outputs/
├── wheat/
│   ├── figures/
│   │   ├── hp_envelope_wheat.svg
│   │   ├── hp_envelope_wheat.eps
│   │   ├── hp_envelope_wheat.png
│   │   ├── agririchter_scale_wheat.svg
│   │   ├── agririchter_scale_wheat.eps
│   │   ├── agririchter_scale_wheat.png
│   │   ├── global_production_wheat.svg
│   │   └── global_production_wheat.png
│   ├── data/
│   │   └── events_wheat_spam2020.csv
│   └── reports/
│       ├── summary_wheat.txt
│       └── validation_wheat.txt
├── rice/
│   └── [similar structure]
└── allgrain/
    └── [similar structure]
```

### Output File Descriptions

**Figures:**
- `hp_envelope_*.svg/eps/png`: H-P Envelope visualization showing historical events plotted by magnitude vs production loss
- `agririchter_scale_*.svg/eps/png`: AgriRichter Scale showing events with severity classifications
- `global_production_*.svg/png`: Global production map showing spatial distribution of crop production

**Data:**
- `events_*_spam2020.csv`: Calculated event data with columns:
  - `event_name`: Name of historical event
  - `harvest_area_loss_ha`: Disrupted harvest area in hectares
  - `production_loss_kcal`: Production loss in kilocalories
  - `magnitude`: AgriRichter magnitude (log10 scale)
  - `affected_countries`: List of affected countries
  - `grid_cells_count`: Number of SPAM grid cells affected

**Reports:**
- `summary_*.txt`: Pipeline execution summary with statistics
- `validation_*.txt`: Validation results comparing with MATLAB outputs (if --validate used)

## Understanding the Results

### Event Metrics

**Harvest Area Loss (hectares):**
- Total area of cropland affected by the disruption
- Summed across all affected grid cells
- Used to calculate magnitude

**Production Loss (kcal):**
- Total caloric production lost due to the disruption
- Converted from metric tons using crop-specific caloric content
- Represents food security impact

**Magnitude:**
- Logarithmic scale: M_D = log10(disrupted area in km²)
- Similar to earthquake magnitude scale
- Typical range: 2-7 for historical events
- Higher values indicate larger geographic extent

### Visualizations

**H-P Envelope:**
- X-axis: Event magnitude (log10 scale)
- Y-axis: Production loss (log10 kcal scale)
- Shows relationship between event size and impact
- Envelope curves represent theoretical maximum impacts

**AgriRichter Scale:**
- Similar to H-P Envelope but with severity classifications
- Threshold lines indicate AgriPhase levels (2-5)
- Color-coded by severity
- Red circles mark historical events

**Global Production Map:**
- Spatial distribution of crop production
- Color intensity shows production density
- Helps identify vulnerable regions

## Advanced Usage

### Using as a Python Library

Import and use AgriRichter components in your own scripts:

```python
from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline

# Configure analysis
config = Config(crop_type='wheat', spam_version='2020')

# Run pipeline
pipeline = EventsPipeline(config, output_dir='outputs/wheat')
results = pipeline.run_complete_pipeline()

# Access results
events_df = results['events_dataframe']
figures = results['figures']
```

### Processing Custom Events

To analyze custom events, create Excel files following the format of `DisruptionCountry.xls` and `DisruptionStateProvince.xls`:

```python
from agririchter.data.events import EventsProcessor

# Load custom events
events_processor = EventsProcessor(
    country_file='path/to/custom_countries.xls',
    state_file='path/to/custom_states.xls'
)

# Process with pipeline
# ... (continue with pipeline setup)
```

### Customizing Visualizations

Modify visualization parameters:

```python
from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer

visualizer = HPEnvelopeVisualizer(config)
fig = visualizer.create_hp_envelope(
    events_df=events_df,
    figsize=(12, 8),
    dpi=300,
    show_labels=True,
    label_fontsize=8
)
```

## Performance Considerations

### Memory Usage

- SPAM 2020 data requires ~2-3GB RAM when loaded
- Processing all crops simultaneously may require 4-6GB RAM
- Use `--crop` to process one crop at a time if memory is limited

### Processing Time

Typical processing times on a modern laptop:
- Single crop (wheat/rice): 2-5 minutes
- All grains: 5-10 minutes
- With validation: +2-3 minutes

### Optimization Tips

1. **Cache SPAM data**: Data is cached in memory during pipeline execution
2. **Process crops separately**: Run wheat, rice, and allgrain in separate executions
3. **Skip figures**: Use `--skip-figures` for faster data processing
4. **Use SSD**: Store data files on SSD for faster I/O

## Troubleshooting

For common issues and solutions, see the [Troubleshooting Guide](TROUBLESHOOTING.md).

## Next Steps

- Review the [Data Requirements Guide](DATA_REQUIREMENTS.md) for data setup
- Check the [API Documentation](API_REFERENCE.md) for programmatic usage
- See the [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues

## Support

For questions or issues:
- Check the documentation in the `docs/` directory
- Review example scripts in `demo_*.py` files
- Examine test files in `tests/` for usage examples

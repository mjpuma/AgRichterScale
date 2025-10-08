# AgriRichter Analysis Pipeline Script

## Overview

The `run_agririchter_analysis.py` script provides a command-line interface to execute the complete AgriRichter events analysis pipeline. It orchestrates data loading, event calculation, visualization generation, and results export for historical agricultural disruption events.

## Features

- **Crop-Specific Analysis**: Analyze wheat, rice, maize, or all grains
- **Batch Processing**: Process multiple crops with a single command
- **Flexible Configuration**: Customize data paths, output locations, and analysis parameters
- **Comprehensive Logging**: Configurable logging levels with console and file output
- **Error Handling**: Graceful error handling with detailed reporting
- **Summary Reports**: Automatic generation of analysis summaries and statistics

## Installation

Ensure you have the AgriRichter package installed with all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run analysis for a single crop:

```bash
python scripts/run_agririchter_analysis.py --crop wheat
```

### Process All Crops

Run analysis for wheat, rice, and allgrain:

```bash
python scripts/run_agririchter_analysis.py --all
```

### Custom Output Directory

Specify a custom output directory:

```bash
python scripts/run_agririchter_analysis.py --crop rice --output my_results
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
python scripts/run_agririchter_analysis.py --crop allgrain --log-level DEBUG
```

### Save Logs to File

Save logs to a file for later review:

```bash
python scripts/run_agririchter_analysis.py --crop wheat --log-file analysis.log
```

### Use SPAM 2010 Data

Analyze using SPAM 2010 data instead of SPAM 2020:

```bash
python scripts/run_agririchter_analysis.py --crop wheat --spam-version 2010
```

### Use Static Thresholds

Use static thresholds instead of USDA-based dynamic thresholds:

```bash
python scripts/run_agririchter_analysis.py --crop wheat --use-static-thresholds
```

### Custom Data Directory

Specify a custom root directory for data files:

```bash
python scripts/run_agririchter_analysis.py --crop wheat --root-dir /path/to/data
```

## Command-Line Arguments

### Required Arguments (one of):

- `--crop {wheat,rice,maize,allgrain}`: Crop type to analyze
- `--all`: Run analysis for all crop types (wheat, rice, allgrain)

### Optional Arguments:

- `--root-dir ROOT_DIR`: Root directory containing data files (default: current directory)
- `--output OUTPUT`: Output directory for results (default: `outputs/<crop_type>`)
- `--spam-version {2010,2020}`: SPAM data version to use (default: 2020)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level (default: INFO)
- `--log-file LOG_FILE`: Optional log file path
- `--use-static-thresholds`: Use static thresholds instead of USDA-based dynamic thresholds
- `-h, --help`: Show help message and exit

## Output Structure

The script creates an organized directory structure for outputs:

```
outputs/
├── wheat/
│   ├── data/
│   │   └── events_wheat_spam2020.csv
│   ├── figures/
│   │   ├── production_map_wheat.svg
│   │   ├── production_map_wheat.eps
│   │   ├── production_map_wheat.jpg
│   │   ├── production_map_wheat.png
│   │   ├── hp_envelope_wheat.svg
│   │   ├── hp_envelope_wheat.eps
│   │   ├── hp_envelope_wheat.jpg
│   │   ├── hp_envelope_wheat.png
│   │   ├── agririchter_scale_wheat.svg
│   │   ├── agririchter_scale_wheat.eps
│   │   ├── agririchter_scale_wheat.jpg
│   │   └── agririchter_scale_wheat.png
│   └── reports/
│       └── pipeline_summary_wheat.txt
├── rice/
│   └── [same structure]
└── allgrain/
    └── [same structure]
```

### Output Files

**Data Files:**
- `events_<crop>_spam2020.csv`: Event results with harvest area losses, production losses, and magnitudes

**Figure Files (4 formats each):**
- `production_map_<crop>.*`: Global production map
- `hp_envelope_<crop>.*`: H-P Envelope with historical events
- `agririchter_scale_<crop>.*`: AgriRichter Scale with historical events

**Report Files:**
- `pipeline_summary_<crop>.txt`: Comprehensive analysis summary with statistics

## Exit Codes

- `0`: Success (all crops completed successfully)
- `1`: Failure (one or more crops failed)

## Logging Levels

- **DEBUG**: Detailed diagnostic information for troubleshooting
- **INFO**: General progress information (default)
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for failures

## Examples

### Example 1: Quick Analysis

Run a quick analysis for wheat with default settings:

```bash
python scripts/run_agririchter_analysis.py --crop wheat
```

### Example 2: Complete Analysis

Run complete analysis for all crops with debug logging:

```bash
python scripts/run_agririchter_analysis.py --all --log-level DEBUG --log-file complete_analysis.log
```

### Example 3: Custom Configuration

Run analysis with custom paths and settings:

```bash
python scripts/run_agririchter_analysis.py \
    --crop rice \
    --root-dir /data/agririchter \
    --output /results/rice_analysis \
    --spam-version 2020 \
    --log-level INFO
```

### Example 4: Batch Processing

Process all crops and save logs:

```bash
python scripts/run_agririchter_analysis.py \
    --all \
    --output batch_results \
    --log-file batch_analysis.log
```

## Data Requirements

The script requires the following data files in the root directory:

### SPAM 2020 Data:
- `spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv`
- `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv`

### Ancillary Data:
- `ancillary/DisruptionCountry.xls`
- `ancillary/DisruptionStateProvince.xls`
- `ancillary/CountryCode_Convert.xls`
- `ancillary/Nutrition_SPAMcrops.xls`
- `ancillary/Foodcodes_SPAMtoFAOSTAT.xls`

### Optional Boundary Data:
- `ancillary/gdam_v2_country.txt`
- `ancillary/gdam_v2_state.asc`

## Troubleshooting

### Missing Data Files

If you see errors about missing data files:

1. Check that all required data files are in the correct locations
2. Use `--root-dir` to specify the correct data directory
3. Verify SPAM data files are in the expected subdirectories

### Memory Issues

If you encounter memory issues with large datasets:

1. Process crops individually instead of using `--all`
2. Close other applications to free up memory
3. Consider using a machine with more RAM

### Logging Issues

If logs are not appearing:

1. Check that the log file directory exists and is writable
2. Try using `--log-level DEBUG` for more detailed output
3. Verify console output is not being redirected

### Performance Issues

If the analysis is taking too long:

1. Check that spatial indexing is working correctly
2. Verify SPAM data files are not corrupted
3. Consider using SPAM 2010 data which may be smaller

## Pipeline Stages

The script executes the following stages:

1. **Data Loading**: Load SPAM 2020 grid data, event definitions, and boundary data
2. **Event Calculation**: Calculate harvest area losses, production losses, and magnitudes for all 21 historical events
3. **Visualization Generation**: Create global production maps, H-P envelopes, and AgriRichter scales
4. **Results Export**: Save results to CSV files and figures in multiple formats
5. **Summary Report**: Generate comprehensive summary with statistics and file locations

## Summary Output

After completion, the script prints a summary including:

- Completion status for each crop
- Number of events processed
- Total harvest area loss (hectares)
- Total production loss (kcal)
- Magnitude range (min-max)
- Number of files generated
- Warnings and errors encountered
- Next steps recommendations

## Next Steps

After running the analysis:

1. Review the generated figures in `outputs/<crop>/figures/`
2. Check the event results CSV files in `outputs/<crop>/data/`
3. Read the summary reports in `outputs/<crop>/reports/`
4. Compare results with MATLAB outputs if available
5. Use the data for further analysis or publication

## Support

For issues or questions:

1. Check the log files for detailed error messages
2. Run with `--log-level DEBUG` for more information
3. Review the implementation summary in `TASK_8_IMPLEMENTATION_SUMMARY.md`
4. Check the requirements and design documents in `.kiro/specs/agririchter-events-integration/`

## License

This script is part of the AgriRichter project. See the main project README for license information.

# MATLAB Validation Quick Start Guide

## Overview

This directory contains tools to validate the Python AgriRichter implementation against MATLAB reference outputs.

## Quick Start

### Option 1: With MATLAB Access

1. **Generate MATLAB outputs:**
   ```bash
   # Follow instructions in:
   matlab_reference_generation_guide.md
   ```

2. **Run validation:**
   ```bash
   python validate_matlab_comparison.py
   ```

3. **Review results:**
   ```bash
   # Check comparison_reports/ directory
   cat comparison_reports/matlab_validation_report_*.txt
   ```

### Option 2: Without MATLAB (Test Mode)

```bash
# Run with mock data to test the validation framework
python test_matlab_validation.py
```

## Files

### Main Scripts
- **`validate_matlab_comparison.py`** - Main validation script (Tasks 12.2-12.5)
- **`test_matlab_validation.py`** - Test script with mock data

### Documentation
- **`matlab_reference_generation_guide.md`** - How to generate MATLAB outputs (Task 12.1)
- **`TASK_12_MATLAB_VALIDATION_COMPLETE.md`** - Complete implementation summary
- **`README_MATLAB_VALIDATION.md`** - This file

## Command-Line Options

```bash
# Basic usage (all crops)
python validate_matlab_comparison.py

# Specific crops only
python validate_matlab_comparison.py --crops wheat rice

# Custom directories
python validate_matlab_comparison.py \
    --matlab-dir /path/to/matlab/outputs \
    --python-dir /path/to/python/outputs \
    --comparison-dir /path/to/reports

# Custom threshold (default is 5%)
python validate_matlab_comparison.py --threshold 0.10

# Help
python validate_matlab_comparison.py --help
```

## Expected Directory Structure

```
.
├── matlab_outputs/                    # MATLAB reference outputs
│   ├── matlab_events_wheat_spam2010.csv
│   ├── matlab_events_rice_spam2010.csv
│   ├── matlab_events_allgrain_spam2010.csv
│   └── matlab_execution_info.txt
│
├── python_outputs/                    # Python results (auto-generated)
│   ├── wheat/
│   │   └── python_events_wheat_spam2020.csv
│   ├── rice/
│   │   └── python_events_rice_spam2020.csv
│   └── allgrain/
│       └── python_events_allgrain_spam2020.csv
│
└── comparison_reports/                # Comparison results (auto-generated)
    ├── matlab_validation_report_*.txt
    ├── comparison_visualization_wheat.png
    ├── comparison_visualization_rice.png
    ├── comparison_visualization_allgrain.png
    └── threshold_recommendation.txt (if needed)
```

## What Gets Validated

For each crop (wheat, rice, allgrain):

1. **Harvest Area Loss** (hectares)
   - Event-by-event comparison
   - Percentage differences
   - Threshold checking (default 5%)

2. **Production Loss** (kcal)
   - Event-by-event comparison
   - Percentage differences
   - Threshold checking (default 5%)

3. **Magnitude** (log10 scale)
   - Should match exactly
   - Small differences acceptable due to floating-point precision

## Output Reports

### Text Report
- Executive summary
- Crop-specific statistics
- Event-by-event comparison table
- Investigation findings
- Validation conclusions

### Visualizations
- Harvest area scatter plot (Python vs MATLAB)
- Production loss scatter plot
- Percentage differences bar chart
- Magnitude comparison scatter plot

### Threshold Recommendation (if needed)
- Current vs suggested threshold
- Statistical justification
- Implementation instructions

## Interpreting Results

### ✓ Validation Passed
- Mean differences < 5%
- Python implementation is accurate
- No action needed

### ⚠ Validation Requires Review
- Some differences > 5%
- Review investigation findings
- Common causes:
  - SPAM 2010 vs 2020 data differences
  - Spatial mapping improvements
  - Rounding/precision differences

### Systematic Differences
- Consistent bias across events
- May indicate SPAM version differences
- Threshold adjustment may be recommended

## Troubleshooting

### "MATLAB results not found"
- Generate MATLAB outputs first
- Follow `matlab_reference_generation_guide.md`
- Ensure CSV files are in `matlab_outputs/` directory

### "Python pipeline failed"
- Check data files are present
- Verify SPAM 2020 data location
- Check logs for specific errors

### Large differences (>10%)
- Review spatial mapping for that event
- Check country/state code mappings
- Verify event definition files

## Requirements Addressed

- ✓ 15.1: Run MATLAB code and Python pipeline
- ✓ 15.2: Save results and figures
- ✓ 15.3: Compare within 5% threshold
- ✓ 15.4: Generate comparison reports
- ✓ 15.5: Update thresholds if needed

## Support

For issues or questions:
1. Check `TASK_12_MATLAB_VALIDATION_COMPLETE.md` for details
2. Review `matlab_reference_generation_guide.md` for MATLAB setup
3. Run `test_matlab_validation.py` to verify framework works
4. Check logs in comparison reports

## Example Workflow

```bash
# 1. Test the validation framework
python test_matlab_validation.py

# 2. Generate MATLAB outputs (if you have MATLAB)
# Follow matlab_reference_generation_guide.md

# 3. Run validation for all crops
python validate_matlab_comparison.py

# 4. Review the report
cat comparison_reports/matlab_validation_report_*.txt

# 5. Check visualizations
open comparison_reports/comparison_visualization_*.png

# 6. If threshold adjustment recommended
cat comparison_reports/threshold_recommendation.txt
```

## Status

✓ Task 12.1: MATLAB reference generation guide complete
✓ Task 12.2: Python pipeline and comparison complete
✓ Task 12.3: Difference investigation complete
✓ Task 12.4: Comparison report generation complete
✓ Task 12.5: Threshold evaluation complete

All validation infrastructure is ready to use!

# AgriRichter Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues when using AgriRichter. Issues are organized by category with symptoms, causes, and solutions.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Loading Issues](#data-loading-issues)
- [Spatial Mapping Issues](#spatial-mapping-issues)
- [Calculation Issues](#calculation-issues)
- [Visualization Issues](#visualization-issues)
- [Performance Issues](#performance-issues)
- [Validation Issues](#validation-issues)
- [Debugging Tips](#debugging-tips)
- [FAQ](#faq)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: Package Import Fails

**Symptoms:**
```
ImportError: No module named 'agririchter'
```

**Causes:**
- Package not installed
- Wrong Python environment active
- Installation incomplete

**Solutions:**

1. Verify Python version (3.8+):
   ```bash
   python --version
   ```

2. Install/reinstall package:
   ```bash
   pip install -r requirements.txt
   ```

3. Check if package is in Python path:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

4. Try installing in development mode:
   ```bash
   pip install -e .
   ```

### Issue: GeoPandas Installation Fails

**Symptoms:**
```
ERROR: Failed building wheel for geopandas
```

**Causes:**
- Missing system dependencies (GDAL, GEOS, PROJ)
- Incompatible versions

**Solutions:**

1. **On macOS:**
   ```bash
   brew install gdal geos proj
   pip install geopandas
   ```

2. **On Ubuntu/Debian:**
   ```bash
   sudo apt-get install gdal-bin libgdal-dev libgeos-dev libproj-dev
   pip install geopandas
   ```

3. **On Windows:**
   - Use conda instead of pip:
   ```bash
   conda install -c conda-forge geopandas
   ```

4. **Alternative:** Use pre-built wheels:
   ```bash
   pip install --find-links https://girder.github.io/large_image_wheels geopandas
   ```

### Issue: Cartopy Installation Fails

**Symptoms:**
```
ERROR: Could not build wheels for cartopy
```

**Solutions:**

1. Install via conda (recommended):
   ```bash
   conda install -c conda-forge cartopy
   ```

2. Install system dependencies first:
   ```bash
   # macOS
   brew install proj geos
   
   # Ubuntu/Debian
   sudo apt-get install libproj-dev libgeos-dev
   ```

---

## Data Loading Issues

### Issue: SPAM Files Not Found

**Symptoms:**
```
FileNotFoundError: spam2020V2r0_global_P_TA.csv not found
```

**Causes:**
- Files not downloaded
- Incorrect directory structure
- Wrong path specified

**Solutions:**

1. Verify file exists:
   ```bash
   ls -la spam2020V2r0_global_production/spam2020V2r0_global_production/
   ```

2. Check directory structure matches expected:
   ```
   spam2020V2r0_global_production/
   └── spam2020V2r0_global_production/
       └── spam2020V2r0_global_P_TA.csv
   ```

3. Specify custom path:
   ```bash
   python scripts/run_agririchter_analysis.py \
       --crop wheat \
       --spam-dir /path/to/spam2020
   ```

4. Download files from SPAM website:
   - Visit: https://www.mapspam.info/data/
   - Download SPAM 2020 v2.0 production and harvest area

### Issue: Excel Files Cannot Be Read

**Symptoms:**
```
xlrd.biffh.XLRDError: Excel xlsx file; not supported
```

**Causes:**
- Missing openpyxl package
- Corrupted Excel file
- Wrong file format

**Solutions:**

1. Install openpyxl:
   ```bash
   pip install openpyxl
   ```

2. Verify file is valid Excel format:
   ```bash
   file ancillary/DisruptionCountry.xls
   ```

3. Try re-downloading ancillary files

4. Convert to .xlsx if needed:
   - Open in Excel/LibreOffice
   - Save As → Excel Workbook (.xlsx)

### Issue: Memory Error When Loading Data

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Causes:**
- Insufficient RAM
- 32-bit Python (limited to 2GB)
- Multiple large datasets loaded simultaneously

**Solutions:**

1. Check available memory:
   ```bash
   # macOS/Linux
   free -h
   
   # macOS
   vm_stat
   ```

2. Use 64-bit Python:
   ```bash
   python -c "import sys; print(sys.maxsize > 2**32)"
   # Should print: True
   ```

3. Process one crop at a time:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat
   python scripts/run_agririchter_analysis.py --crop rice
   ```

4. Close other applications to free RAM

5. Use optimized dtypes (already implemented in GridDataManager)

---

## Spatial Mapping Issues

### Issue: Country Code Not Found

**Symptoms:**
```
WARNING: Country code 999 not found in conversion table
```

**Causes:**
- Invalid country code in event definition
- Missing entry in CountryCode_Convert.xls
- Typo in event data

**Solutions:**

1. Check country code in event file:
   ```python
   import pandas as pd
   df = pd.read_excel('ancillary/DisruptionCountry.xls', sheet_name='EventName')
   print(df)
   ```

2. Verify country code exists in conversion table:
   ```python
   codes = pd.read_excel('ancillary/CountryCode_Convert.xls')
   print(codes[codes['GDAM_Code'] == 999])
   ```

3. Add missing country to conversion table if legitimate

4. Fix typo in event definition file

### Issue: No Grid Cells Found for Event

**Symptoms:**
```
WARNING: Event 'EventName' has 0 grid cells matched
```

**Causes:**
- Country has no SPAM data (small islands, etc.)
- ISO3 code mismatch
- Spatial index not created

**Solutions:**

1. Check if country has SPAM data:
   ```python
   from agririchter.data.grid_manager import GridDataManager
   
   grid_manager = GridDataManager(config)
   prod_df, _ = grid_manager.load_spam_data()
   
   # Check available countries
   print(prod_df['iso3'].unique())
   ```

2. Verify ISO3 code mapping:
   ```python
   from agririchter.data.spatial_mapper import SpatialMapper
   
   mapper = SpatialMapper(config, grid_manager)
   iso3 = mapper.get_iso3_from_country_code(country_code)
   print(f"ISO3: {iso3}")
   ```

3. Create spatial index:
   ```python
   grid_manager.create_spatial_index()
   ```

4. Check if event should have data (some historical events may legitimately have no data)

### Issue: State-Level Mapping Fails

**Symptoms:**
```
WARNING: State code 123 not found for country 456
```

**Causes:**
- State codes not in SPAM data
- SPAM data doesn't include state-level detail
- Mismatch between GDAM and SPAM state codes

**Solutions:**

1. Check if SPAM data has state information:
   ```python
   prod_df, _ = grid_manager.load_spam_data()
   print('name_adm1' in prod_df.columns)
   ```

2. Fall back to country-level:
   - Edit event definition to set state flag = 0

3. Use boundary shapefiles for precise matching (optional)

4. Note: Some events may need country-level aggregation if state data unavailable

---

## Calculation Issues

### Issue: Production Loss is Zero

**Symptoms:**
```
WARNING: Event 'EventName' has zero production loss
```

**Causes:**
- No grid cells matched
- Selected crop not grown in region
- Data quality issue

**Solutions:**

1. Check grid cell matching (see Spatial Mapping Issues above)

2. Verify crop is grown in region:
   ```python
   cells = grid_manager.get_grid_cells_by_iso3('USA')
   production = grid_manager.get_crop_production(cells, [1])  # wheat
   print(f"Total production: {production}")
   ```

3. Check if event should affect selected crop:
   - Rice event won't affect wheat-only regions
   - Verify event definition matches crop type

4. Inspect raw SPAM data for region

### Issue: Magnitude is NaN

**Symptoms:**
```
WARNING: Event 'EventName' has NaN magnitude
```

**Causes:**
- Zero harvest area loss
- Negative harvest area (data error)
- Log of zero or negative number

**Solutions:**

1. Check harvest area calculation:
   ```python
   harvest_area = grid_manager.get_crop_harvest_area(cells, [1])
   print(f"Harvest area: {harvest_area} ha")
   ```

2. Verify magnitude calculation:
   ```python
   from agririchter.analysis.event_calculator import EventCalculator
   
   calculator = EventCalculator(config, grid_manager, mapper)
   magnitude = calculator.calculate_magnitude(harvest_area)
   print(f"Magnitude: {magnitude}")
   ```

3. Check for data quality issues in SPAM data

4. Note: NaN magnitude is expected if event has no harvest area loss

### Issue: Results Don't Match MATLAB

**Symptoms:**
```
WARNING: Event 'EventName' differs from MATLAB by 15%
```

**Causes:**
- Different SPAM versions (2010 vs 2020)
- Rounding differences
- Different spatial matching methods
- Bug in Python implementation

**Solutions:**

1. Verify using same SPAM version:
   ```python
   print(config.spam_version)
   ```

2. Check if difference is systematic:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat --validate
   ```

3. Compare intermediate values:
   - Grid cells matched
   - Raw production values
   - Unit conversions

4. Review validation report for patterns

5. Small differences (<5%) are expected due to:
   - Floating-point precision
   - Different spatial libraries
   - Rounding in aggregations

---

## Visualization Issues

### Issue: Figures Not Generated

**Symptoms:**
```
ERROR: Failed to generate hp_envelope figure
```

**Causes:**
- Missing matplotlib backend
- Display not available (headless server)
- Insufficient memory
- Data issues

**Solutions:**

1. Set matplotlib backend:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend
   import matplotlib.pyplot as plt
   ```

2. For headless servers, use Agg backend:
   ```bash
   export MPLBACKEND=Agg
   python scripts/run_agririchter_analysis.py --crop wheat
   ```

3. Check if data is valid:
   ```python
   print(events_df.head())
   print(events_df.isnull().sum())
   ```

4. Try generating figures individually:
   ```python
   from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer
   
   viz = HPEnvelopeVisualizer(config)
   fig = viz.create_hp_envelope(events_df)
   fig.savefig('test.png')
   ```

### Issue: Event Labels Overlap

**Symptoms:**
- Event labels are unreadable
- Text overlaps on figures

**Causes:**
- Too many events in small area
- adjustText not working properly
- Font size too large

**Solutions:**

1. Adjust label parameters:
   ```python
   viz = HPEnvelopeVisualizer(config)
   fig = viz.create_hp_envelope(
       events_df,
       show_labels=True,
       label_fontsize=6  # Smaller font
   )
   ```

2. Manually adjust label positions (edit visualization code)

3. Generate larger figure:
   ```python
   fig = viz.create_hp_envelope(
       events_df,
       figsize=(14, 10)  # Larger figure
   )
   ```

4. Disable labels for cleaner figure:
   ```python
   fig = viz.create_hp_envelope(
       events_df,
       show_labels=False
   )
   ```

### Issue: Map Projection Errors

**Symptoms:**
```
ERROR: Cartopy projection failed
```

**Causes:**
- Missing cartopy data files
- Network issues downloading map data
- Coordinate system mismatch

**Solutions:**

1. Pre-download cartopy data:
   ```python
   import cartopy.crs as ccrs
   import cartopy.feature as cfeature
   
   # This will download required data
   cfeature.COASTLINE
   cfeature.BORDERS
   ```

2. Use offline mode if network unavailable:
   ```python
   import cartopy.config
   cartopy.config['pre_existing_data_dir'] = '/path/to/cartopy/data'
   ```

3. Try simpler projection:
   ```python
   projection = ccrs.PlateCarree()  # Simple projection
   ```

---

## Performance Issues

### Issue: Analysis Takes Too Long

**Symptoms:**
- Pipeline runs for >30 minutes
- System becomes unresponsive

**Causes:**
- Processing all crops simultaneously
- Inefficient spatial operations
- Not using cached data
- Large output files

**Solutions:**

1. Process crops separately:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat
   ```

2. Enable data caching (default):
   ```python
   grid_manager = GridDataManager(config, cache_data=True)
   ```

3. Skip figure generation for faster data processing:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat --skip-figures
   ```

4. Use performance monitoring:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat --log-level DEBUG
   ```

5. Check system resources:
   ```bash
   # Monitor CPU and memory
   top
   # or
   htop
   ```

### Issue: High Memory Usage

**Symptoms:**
- System swapping to disk
- Out of memory errors
- Slow performance

**Solutions:**

1. Process one crop at a time

2. Clear intermediate results:
   ```python
   import gc
   gc.collect()
   ```

3. Use memory profiling:
   ```bash
   pip install memory_profiler
   python -m memory_profiler scripts/run_agririchter_analysis.py --crop wheat
   ```

4. Reduce figure DPI:
   ```python
   fig = viz.create_hp_envelope(events_df, dpi=150)  # Lower DPI
   ```

---

## Validation Issues

### Issue: Validation Report Shows Warnings

**Symptoms:**
```
WARNING: 3 events have losses exceeding expected range
```

**Causes:**
- Data quality issues
- Calculation errors
- Different data versions
- Legitimate extreme values

**Solutions:**

1. Review specific warnings:
   ```bash
   cat outputs/wheat/reports/validation_wheat.txt
   ```

2. Investigate flagged events:
   ```python
   validator = DataValidator(config)
   results = validator.validate_event_results(events_df)
   print(results['warnings'])
   ```

3. Compare with MATLAB if available:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat --validate
   ```

4. Check if warnings are expected:
   - Some historical events may have extreme values
   - SPAM 2020 data may differ from SPAM 2010

### Issue: MATLAB Comparison Fails

**Symptoms:**
```
ERROR: MATLAB reference file not found
```

**Causes:**
- Reference files not generated
- Wrong file path
- File naming mismatch

**Solutions:**

1. Generate MATLAB reference files first:
   - Run original MATLAB code
   - Save outputs to CSV

2. Specify correct path:
   ```python
   validator.compare_with_matlab(
       events_df,
       matlab_file='path/to/matlab_events_wheat.csv'
   )
   ```

3. Skip MATLAB comparison if not needed:
   ```bash
   python scripts/run_agririchter_analysis.py --crop wheat
   # (without --validate flag)
   ```

---

## Debugging Tips

### Enable Debug Logging

Get detailed information about what's happening:

```bash
python scripts/run_agririchter_analysis.py \
    --crop wheat \
    --log-level DEBUG \
    > debug.log 2>&1
```

### Use Interactive Python

Debug issues interactively:

```python
from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager

config = Config(crop_type='wheat', log_level='DEBUG')
grid_manager = GridDataManager(config)

# Load data and inspect
prod_df, harvest_df = grid_manager.load_spam_data()
print(prod_df.head())
print(prod_df.info())
```

### Check Intermediate Results

Save and inspect intermediate data:

```python
# Save grid cells for inspection
cells = grid_manager.get_grid_cells_by_iso3('USA')
cells.to_csv('debug_usa_cells.csv', index=False)

# Save event results
events_df.to_csv('debug_events.csv', index=False)
```

### Use Demo Scripts

Test individual components:

```bash
# Test grid manager
python demo_grid_manager.py

# Test spatial mapper
python demo_spatial_mapper.py

# Test event calculator
python demo_event_calculator.py

# Test full pipeline
python demo_events_pipeline.py
```

### Profile Performance

Identify bottlenecks:

```bash
# Time profiling
python -m cProfile -o profile.stats scripts/run_agririchter_analysis.py --crop wheat

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Memory profiling
python -m memory_profiler scripts/run_agririchter_analysis.py --crop wheat
```

### Validate Data Files

Check data file integrity:

```bash
# Verify SPAM structure
python verify_spam2020_structure.py

# Verify event definitions
python -c "from agririchter.data.events import EventsProcessor; \
           ep = EventsProcessor(); \
           events = ep.load_events(); \
           print(f'Loaded {len(events)} events')"

# Validate spatial mapper
python demo_spatial_mapper.py
```

---

## FAQ

### Q: Can I use SPAM 2010 data instead of SPAM 2020?

**A:** Yes, set `spam_version='2010'` in Config. Note that results will differ from SPAM 2020 due to updated production estimates.

### Q: How do I analyze only specific events?

**A:** Modify the events Excel files to include only desired events, or filter the results DataFrame after calculation.

### Q: Can I add custom events?

**A:** Yes, add new sheets to `DisruptionCountry.xls` and `DisruptionStateProvince.xls` following the existing format.

### Q: Why are some events missing from the output?

**A:** Events with no grid cells matched or zero production loss may be filtered out. Check logs for warnings.

### Q: How accurate are the results compared to MATLAB?

**A:** Results should be within 5% of MATLAB outputs. Differences arise from floating-point precision and spatial library differences.

### Q: Can I run AgriRichter on a cluster/HPC?

**A:** Yes, use non-interactive matplotlib backend (`MPLBACKEND=Agg`) and ensure all dependencies are installed.

### Q: How do I cite AgriRichter in publications?

**A:** Include citations for SPAM 2020 data, GDAM boundaries, and the AgriRichter methodology paper (if published).

### Q: What if my country is not in the conversion table?

**A:** Add the country to `CountryCode_Convert.xls` with appropriate GDAM code, FAOSTAT code, and ISO3 code.

### Q: Can I export results in different formats?

**A:** Yes, modify the export methods in `EventsPipeline` to save in desired formats (JSON, Excel, etc.).

### Q: How do I update to a newer SPAM version?

**A:** Download new SPAM files, update paths in Config, verify column names match, and run validation tests.

---

## Getting Help

### Check Documentation

1. [User Guide](USER_GUIDE.md) - Installation and usage
2. [Data Requirements](DATA_REQUIREMENTS.md) - Required data files
3. [API Reference](API_REFERENCE.md) - Programmatic usage

### Review Examples

- Demo scripts: `demo_*.py`
- Test files: `tests/unit/` and `tests/integration/`
- Verification scripts: `verify_*.py`

### Search Issues

Check if your issue has been reported:
- Review closed issues in repository
- Search documentation for keywords
- Check SPAM and GDAM documentation

### Report a Bug

If you've found a bug, provide:
1. Python version and OS
2. Full error message and traceback
3. Steps to reproduce
4. Data file versions (SPAM 2020, etc.)
5. Log output with `--log-level DEBUG`

### Request Support

For questions or support:
- Open an issue in the repository
- Include relevant log files and error messages
- Describe what you've tried already
- Provide minimal reproducible example

### Contribute

Help improve AgriRichter:
- Fix bugs and submit pull requests
- Improve documentation
- Add test cases
- Report issues with detailed information

---

## Common Error Messages Reference

| Error Message | Likely Cause | Quick Fix |
|--------------|--------------|-----------|
| `FileNotFoundError: spam2020...` | SPAM files not found | Check file paths, use `--spam-dir` |
| `ImportError: No module named 'geopandas'` | Missing dependency | `pip install geopandas` |
| `MemoryError: Unable to allocate` | Insufficient RAM | Process one crop at a time |
| `KeyError: 'iso3'` | Wrong SPAM file format | Verify SPAM 2020 files |
| `ValueError: Country code not found` | Invalid country code | Check conversion table |
| `RuntimeError: Spatial index not created` | Missing spatial index | Call `create_spatial_index()` |
| `WARNING: 0 grid cells matched` | No SPAM data for region | Check ISO3 mapping |
| `ERROR: Failed to generate figure` | Matplotlib backend issue | Set `MPLBACKEND=Agg` |

---

## Additional Resources

- SPAM Documentation: https://www.mapspam.info/
- GDAM Documentation: https://gadm.org/
- GeoPandas Documentation: https://geopandas.org/
- Cartopy Documentation: https://scitools.org.uk/cartopy/
- Pandas Documentation: https://pandas.pydata.org/

---

**Last Updated:** 2025-10-08

For the most current troubleshooting information, check the repository documentation and issues.

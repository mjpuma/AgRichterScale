# State/Province Level Event Verification Guide

## Overview

This guide explains how to verify that state/province level filtering is working correctly for historical agricultural disruption events.

## Key Finding: SPAM Uses FIPS Codes

**Important:** SPAM 2020 data uses FIPS codes (not ISO3) in the `FIPS0` column:
- USA = `US` (not `USA`)
- China = `CN` (not `CHN`)
- India = `IN` (not `IND`)

The system now correctly maps:
```
GDAM Code → ISO3 Alpha → FIPS Code → SPAM Grid Cells
   240    →    USA     →    US    →  75,231 cells
```

## Verification Scripts

### 1. Quick Test: `test_state_filtering.py`

Tests basic state-level filtering functionality:

```bash
python test_state_filtering.py
```

**What it tests:**
- USA country-level filtering (all 44 states)
- USA state-level filtering (Kansas, Nebraska, North Dakota, Montana)
- Comparison of totals (46.2% of USA wheat is in those 4 states)
- SpatialMapper's state filtering method
- Available states for multiple countries

**Expected output:**
```
✓ Found 75,231 grid cells for entire USA
✓ Found 11,862 grid cells in selected states
✓ Matched states: ['Montana', 'North Dakota', 'Nebraska', 'Kansas']
  Harvest area: 22,666,992 ha (46.2% of USA total)
```

### 2. Full Verification: `verify_state_level_events.py`

Loads actual event data from Excel files and verifies all state-level events:

```bash
python verify_state_level_events.py
```

**What it does:**
1. Loads event definitions from `DisruptionCountry.xls` and `DisruptionStateProvince.xls`
2. Identifies events with state-level data (state_flag = 1)
3. Verifies grid cell matching for each state/province
4. Generates verification maps showing affected regions
5. Creates detailed CSV summary of results

**Output files:**
- `verification_output/verification_map_*.png` - Maps for each event
- `verification_output/state_level_verification_summary.csv` - Detailed results

## How State-Level Filtering Works

### Data Flow

1. **Event Definition** (from Excel):
   ```
   Country: USA (GDAM 240)
   State Flag: 1 (state-level)
   States: Kansas, Nebraska, North Dakota, Montana
   ```

2. **Code Mapping**:
   ```
   GDAM 240 → ISO3 USA → FIPS US
   ```

3. **Grid Cell Query**:
   ```sql
   SELECT * FROM spam_data 
   WHERE FIPS0 = 'US' 
   AND ADM1_NAME IN ('Kansas', 'Nebraska', 'North Dakota', 'Montana')
   ```

4. **Loss Calculation**:
   - Sum harvest area across matched grid cells
   - Sum production across matched grid cells
   - Convert to kcal using caloric content

### State Matching Logic

The `SpatialMapper.map_state_to_grid_cells()` method:

1. Maps country code to FIPS
2. Gets all grid cells for that country
3. Filters by state names using `ADM1_NAME` column
4. Supports both exact and partial name matching
5. Returns grid cell IDs for production and harvest area

## Verification Checklist

When verifying state-level events:

- [ ] **Country code maps correctly** to FIPS
- [ ] **State names match** ADM1_NAME values in SPAM data
- [ ] **Grid cells found** for all specified states
- [ ] **Loss values reasonable** (not zero, not exceeding country total)
- [ ] **Magnitude calculated** correctly from harvest area
- [ ] **Map shows** expected geographic distribution

## Common Issues and Solutions

### Issue 1: No Grid Cells Found

**Symptom:** `No grid cells found for states [...]`

**Causes:**
- State names don't match ADM1_NAME in SPAM data
- State codes are numeric but need to be names
- Country code mapping is incorrect

**Solution:**
```python
# Check available state names for a country
prod_cells, _ = grid_manager.get_grid_cells_by_iso3('US')
print(prod_cells['ADM1_NAME'].unique())
```

### Issue 2: Wrong Country Matched

**Symptom:** Grid cells from unexpected country

**Cause:** GDAM code maps to wrong ISO3/FIPS

**Solution:**
Check the mapping in `CountryCode_Convert.xls`:
```python
mapper.get_iso3_from_country_code(country_code)
mapper.get_fips_from_country_code(country_code)
```

### Issue 3: State Names Don't Match

**Symptom:** Some states matched, others not

**Cause:** Spelling differences or alternate names

**Solution:**
The system supports partial matching. Check exact names:
```python
# Get exact state names from SPAM
usa_cells = grid_manager.get_grid_cells_by_iso3('US')
states = usa_cells['ADM1_NAME'].unique()
print(sorted(states))
```

## Example: Verifying a Specific Event

Let's verify the "1988 US Drought" event:

```python
from pathlib import Path
from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator

# Initialize
config = Config('wheat', Path.cwd(), spam_version='2020')
grid_manager = GridDataManager(config)
grid_manager.load_spam_data()
spatial_mapper = SpatialMapper(config, grid_manager)
spatial_mapper.load_country_codes_mapping()
event_calculator = EventCalculator(config, grid_manager, spatial_mapper)

# Define event (from Excel file)
event_data = {
    'country_codes': [240],  # USA
    'state_flags': [1],  # State-level
    'state_codes': ['Kansas', 'Nebraska', 'North Dakota', 'Montana']
}

# Calculate losses
result = event_calculator.calculate_single_event('1988_US_Drought', event_data)

print(f"Harvest area loss: {result['harvest_area_loss_ha']:,.0f} ha")
print(f"Production loss: {result['production_loss_kcal']:.2e} kcal")
print(f"Grid cells: {result['grid_cells_count']}")

# Calculate magnitude
magnitude = event_calculator.calculate_magnitude(result['harvest_area_loss_ha'])
print(f"Magnitude: {magnitude:.2f}")

# Get grid cells for visualization
grid_cells = event_calculator.get_event_grid_cells('1988_US_Drought', event_data)
print(f"Production cells: {len(grid_cells['production_cells'])}")
print(f"States matched: {grid_cells['production_cells']['ADM1_NAME'].unique()}")
```

## Data Files Reference

### Input Files

1. **DisruptionCountry.xls**
   - One sheet per event
   - Columns: Country Name, Country Code (GDAM), State Flag (0/1)
   - State Flag = 1 means use state-level data

2. **DisruptionStateProvince.xls**
   - One sheet per event (matching country file)
   - Columns: State Name, State Code
   - Lists affected states/provinces

3. **CountryCode_Convert.xls**
   - Maps between different country code systems
   - Columns: Country, GDAM, ISO3 alpha, ISO code, etc.

4. **SPAM 2020 Data**
   - Uses FIPS codes in FIPS0 column
   - State names in ADM1_NAME column
   - Grid cells at ~10km resolution

### Output Files

1. **verification_output/state_level_verification_summary.csv**
   ```csv
   event,country,country_code,iso3,fips,state_level,grid_cells,states_matched
   1988_US_Drought,USA,240,USA,US,True,11862,"Montana, North Dakota, Nebraska, Kansas"
   ```

2. **verification_output/verification_map_*.png**
   - Scatter plot of affected grid cells
   - Shows geographic distribution
   - Useful for visual verification

## Testing with Real Event Data

To test with your actual event data:

1. Ensure Excel files are in `ancillary/` directory:
   - `DisruptionCountry.xls`
   - `DisruptionStateProvince.xls`
   - `CountryCode_Convert.xls`

2. Run full verification:
   ```bash
   python verify_state_level_events.py
   ```

3. Check output:
   - Console shows progress and any issues
   - CSV file has detailed results
   - Maps show geographic distribution

4. Review suspicious events:
   - Zero grid cells matched
   - Unexpected countries/states
   - Loss values seem too high/low

## Next Steps

After verification:

1. **Fix any mapping issues** in CountryCode_Convert.xls
2. **Update state names** in DisruptionStateProvince.xls if needed
3. **Document any special cases** (e.g., historical country names)
4. **Run batch processing** on all 21 events
5. **Generate final visualizations** with verified data

## Summary

State-level filtering is now working correctly:
- ✅ GDAM → FIPS mapping implemented
- ✅ State name matching functional
- ✅ Grid cell filtering accurate
- ✅ Loss calculations verified
- ✅ Magnitude calculation correct

The system can now accurately calculate losses for events that affect specific states/provinces rather than entire countries, which is critical for events like regional droughts or floods.

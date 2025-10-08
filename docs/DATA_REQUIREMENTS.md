# AgriRichter Data Requirements

## Overview

AgriRichter requires several data files to perform historical event analysis. This guide describes all required and optional data files, their sources, and how to organize them.

## Required Data Files

### 1. SPAM 2020 Gridded Data

**Description:** Spatial Production Allocation Model (SPAM) 2020 provides global gridded crop production and harvest area data at 5-arcminute resolution (~10km at equator).

**Required Files:**

1. **Production Data (Total Technology)**
   - Filename: `spam2020V2r0_global_P_TA.csv`
   - Location: `spam2020V2r0_global_production/spam2020V2r0_global_production/`
   - Size: ~500MB
   - Description: Total production in metric tons for all crops

2. **Harvest Area Data (Total Technology)**
   - Filename: `spam2020V2r0_global_H_TA.csv`
   - Location: `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/`
   - Size: ~500MB
   - Description: Total harvested area in hectares for all crops

3. **Yield Data (Total Technology)** - Optional but recommended
   - Filename: `spam2020V2r0_global_Y_TA.csv`
   - Location: `spam2020V2r0_global_yield/spam2020V2r0_global_yield/`
   - Size: ~500MB
   - Description: Yield in kg/ha for all crops

**Download Source:**
- Official SPAM website: https://www.mapspam.info/data/
- Direct download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PRFF8V
- Select "SPAM 2020 v2.0" dataset
- Download the "Global" files for Production, Harvested Area, and Yield

**File Structure:**

Each CSV file contains the following columns:
- `x`: Longitude (decimal degrees, -180 to 180)
- `y`: Latitude (decimal degrees, -90 to 90)
- `iso3`: ISO 3166-1 alpha-3 country code
- `cell5m`: Unique cell identifier
- `whea_a`: Wheat production/area/yield (technology A)
- `rice_a`: Rice production/area/yield (technology A)
- `maiz_a`: Maize production/area/yield (technology A)
- ... (additional crops and technologies)

**Crop Codes Used by AgriRichter:**
- `whea`: Wheat (crop index 1)
- `rice`: Rice (crop index 2)
- `maiz`: Maize (crop index 3)
- `barl`: Barley (crop index 4)
- `sorg`: Sorghum (crop index 5)
- `pmil`: Pearl millet (crop index 6)
- `oats`: Oats (crop index 7)
- `rye_`: Rye (crop index 8)

### 2. Historical Event Definitions

**Description:** Excel files defining 21 historical agricultural disruption events with affected countries and states.

**Required Files:**

1. **Country-Level Event Data**
   - Filename: `DisruptionCountry.xls`
   - Location: `ancillary/`
   - Size: ~100KB
   - Description: 21 sheets (one per event) with country codes and flags

2. **State/Province-Level Event Data**
   - Filename: `DisruptionStateProvince.xls`
   - Location: `ancillary/`
   - Size: ~50KB
   - Description: 21 sheets (one per event) with state/province codes

**File Structure:**

`DisruptionCountry.xls` - Each sheet contains:
- Column A: Country names (text)
- Column B: Country codes (numeric, GDAM codes)
- Column C: State flag (0 = country-level, 1 = state-level)

`DisruptionStateProvince.xls` - Each sheet contains:
- Column A: State/province names (text)
- Column B: State/province codes (numeric)
- Column C: Country code (numeric, GDAM codes)

**Event List:**
1. China 1959-1961 (Great Leap Forward)
2. Bengal 1943
3. Soviet Union 1932-1933
4. Ethiopia 1983-1985
5. North Korea 1994-1998
6. Somalia 1991-1992
7. Bangladesh 1974
8. India 1965-1967
9. Sahel 1968-1973
10. Mozambique 1983-1984
11. Sudan 1998
12. Malawi 2002
13. Niger 2005
14. Zimbabwe 2008
15. East Africa 2011
16. Sahel 2012
17. Syria 2007-2010
18. US Dust Bowl 1934-1936
19. Soviet Union 1946-1947
20. China 1928-1930
21. India 1899-1900

**Source:**
- These files are typically provided with the AgriRichter package
- Historical event data compiled from academic literature and FAO reports
- Contact the package maintainers if files are missing

### 3. Country Code Conversion Table

**Description:** Mapping between different country code systems (GDAM, FAOSTAT, ISO3).

**Required File:**
- Filename: `CountryCode_Convert.xls`
- Location: `ancillary/`
- Size: ~50KB
- Description: Conversion table for country codes

**File Structure:**
- Column A: Country name
- Column B: GDAM country code (numeric)
- Column C: FAOSTAT country code (numeric)
- Column D: ISO3 country code (3-letter)
- Column E: Additional metadata

**Source:**
- Provided with AgriRichter package
- Based on GDAM v2 country codes
- Includes mappings for ~200 countries

### 4. Crop Nutrition Data

**Description:** Caloric content for converting production from metric tons to kilocalories.

**Required File:**
- Filename: `Nutrition_SPAMcrops.xls`
- Location: `ancillary/`
- Size: ~20KB
- Description: Caloric content (kcal/g) for SPAM crops

**File Structure:**
- Column A: Crop name
- Column B: SPAM crop code
- Column C: Caloric content (kcal/100g)
- Column D: Protein content (g/100g)
- Column E: Fat content (g/100g)

**Caloric Content Values:**
- Wheat: 339 kcal/100g
- Rice: 365 kcal/100g
- Maize: 365 kcal/100g
- Barley: 352 kcal/100g
- Sorghum: 329 kcal/100g
- Millet: 378 kcal/100g
- Oats: 389 kcal/100g
- Rye: 338 kcal/100g

**Source:**
- USDA National Nutrient Database
- FAO Food Composition Tables
- Provided with AgriRichter package

## Optional Data Files

### 1. Geographic Boundary Data (GDAM)

**Description:** Shapefiles or raster files for precise country and state boundary matching.

**Optional Files:**

1. **Country Boundaries**
   - Filename: `gdam_v2_country.txt` or `gdam_v2_country.shp`
   - Location: `ancillary/`
   - Description: Country boundary data from GDAM v2

2. **State/Province Boundaries**
   - Filename: `gdam_v2_state.asc` or `gdam_v2_state.shp`
   - Location: `ancillary/`
   - Description: State/province boundary data from GDAM v2

3. **Grid Cell Reference**
   - Filename: `CELL5M.asc`
   - Location: `ancillary/`
   - Description: 5-arcminute grid cell identifiers

**When to Use:**
- Boundary files enable more precise spatial matching
- Useful when ISO3 code matching is insufficient
- Required for events with complex geographic boundaries
- Optional if using ISO3-based matching only

**Download Source:**
- GDAM website: https://gadm.org/
- Select version 2.0 for compatibility
- Download country and administrative level 1 (states) data

**Note:** AgriRichter can function without these files by using ISO3 code matching as a fallback method.

### 2. USDA PSD Data

**Description:** USDA Production, Supply, and Distribution data for validation and threshold calculations.

**Optional Files:**
- `grains_world_usdapsd_production_jul142023.csv`
- `grains_world_usdapsd_consumption_jul142023.csv`
- `grains_world_usdapsd_endingstocks_jul142023.csv`

**Location:** `USDAdata/`

**When to Use:**
- Calculating AgriPhase thresholds
- Validating SPAM data totals
- Comparing production estimates

**Download Source:**
- USDA PSD Online: https://apps.fas.usda.gov/psdonline/
- Select commodity, country, and time period
- Export to CSV format

### 3. MATLAB Reference Outputs

**Description:** Reference outputs from original MATLAB implementation for validation.

**Optional Files:**
- `matlab_events_wheat.csv`
- `matlab_events_rice.csv`
- `matlab_events_allgrain.csv`

**When to Use:**
- Validating Python implementation accuracy
- Comparing event loss calculations
- Debugging discrepancies

**Source:**
- Generated by running original MATLAB code
- Contact package maintainers for reference files

## Recommended Directory Structure

Organize your data files as follows:

```
agririchter/
├── spam2020V2r0_global_production/
│   └── spam2020V2r0_global_production/
│       ├── spam2020V2r0_global_P_TA.csv
│       ├── spam2020V2r0_global_P_TI.csv
│       └── spam2020V2r0_global_P_TR.csv
├── spam2020V2r0_global_harvested_area/
│   └── spam2020V2r0_global_harvested_area/
│       ├── spam2020V2r0_global_H_TA.csv
│       ├── spam2020V2r0_global_H_TI.csv
│       └── spam2020V2r0_global_H_TR.csv
├── spam2020V2r0_global_yield/
│   └── spam2020V2r0_global_yield/
│       ├── spam2020V2r0_global_Y_TA.csv
│       ├── spam2020V2r0_global_Y_TI.csv
│       └── spam2020V2r0_global_Y_TR.csv
├── ancillary/
│   ├── DisruptionCountry.xls
│   ├── DisruptionStateProvince.xls
│   ├── CountryCode_Convert.xls
│   ├── Nutrition_SPAMcrops.xls
│   ├── gdam_v2_country.txt (optional)
│   ├── gdam_v2_state.asc (optional)
│   └── CELL5M.asc (optional)
├── USDAdata/ (optional)
│   ├── grains_world_usdapsd_production_jul142023.csv
│   ├── grains_world_usdapsd_consumption_jul142023.csv
│   └── grains_world_usdapsd_endingstocks_jul142023.csv
└── outputs/
    ├── wheat/
    ├── rice/
    └── allgrain/
```

## Data Validation

### Verifying SPAM Data

Check that SPAM files are correctly formatted:

```bash
python verify_spam2020_structure.py
```

This script validates:
- File existence and readability
- Column names and structure
- Coordinate ranges
- Crop code presence
- Data completeness

### Verifying Event Data

Check that event definition files are correctly formatted:

```bash
python -c "from agririchter.data.events import EventsProcessor; \
           ep = EventsProcessor(); \
           print(f'Loaded {len(ep.get_all_events())} events successfully')"
```

Expected output: `Loaded 21 events successfully`

## Data Size and Storage

### Disk Space Requirements

- SPAM 2020 files: ~1.5GB (production + harvest area + yield)
- Ancillary files: ~1MB
- USDA data (optional): ~50MB
- Output files (per crop): ~50MB
- **Total recommended:** 10GB (including workspace for outputs)

### Memory Requirements

- Loading SPAM data: ~2-3GB RAM
- Processing events: ~1GB RAM
- Generating figures: ~500MB RAM
- **Total recommended:** 4GB RAM minimum, 8GB preferred

## Troubleshooting Data Issues

### Missing SPAM Files

**Error:** `FileNotFoundError: spam2020V2r0_global_P_TA.csv not found`

**Solution:**
1. Verify files are downloaded from SPAM website
2. Check directory structure matches expected layout
3. Use `--spam-dir` flag to specify custom location
4. Ensure files are unzipped (not compressed)

### Invalid Country Codes

**Error:** `Country code XXX not found in conversion table`

**Solution:**
1. Check `CountryCode_Convert.xls` is present
2. Verify country code format (numeric GDAM codes)
3. Update conversion table if analyzing new countries
4. Check event definition files for typos

### Coordinate Mismatches

**Error:** `Coordinates out of range: lat=XX, lon=YY`

**Solution:**
1. Verify SPAM data uses WGS84 coordinate system
2. Check for data corruption in CSV files
3. Re-download SPAM files if necessary
4. Ensure no manual edits to coordinate columns

### Memory Errors

**Error:** `MemoryError: Unable to allocate array`

**Solution:**
1. Process one crop at a time instead of all grains
2. Close other applications to free RAM
3. Use 64-bit Python installation
4. Consider using a machine with more RAM

## Data Updates

### Using Newer SPAM Versions

If SPAM releases a newer version (e.g., SPAM 2025):

1. Update `Config` class with new file paths
2. Verify column names match expected format
3. Run validation scripts to check compatibility
4. Update documentation with new version info

### Adding Custom Events

To analyze custom events:

1. Create new sheets in `DisruptionCountry.xls`
2. Add corresponding sheets in `DisruptionStateProvince.xls`
3. Follow existing format for country/state codes
4. Update event list in documentation

## Data Citations

When using AgriRichter in publications, please cite:

**SPAM 2020:**
> Yu, Q., You, L., Wood-Sichra, U., Ru, Y., Joglekar, A.K.B., Fritz, S., Xiong, W., Lu, M., Wu, W., Yang, P. (2020). A cultivated planet in 2010 – Part 2: The global gridded agricultural-production maps. Earth System Science Data, 12(4), 3545-3572.

**GDAM:**
> Global Administrative Areas (2018). GADM database of Global Administrative Areas, version 2.0. [online] URL: www.gadm.org

**USDA PSD:**
> United States Department of Agriculture, Foreign Agricultural Service. Production, Supply and Distribution Online Database. https://apps.fas.usda.gov/psdonline/

## Support

For data-related questions:
- Check the [User Guide](USER_GUIDE.md) for usage instructions
- See the [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues
- Review example scripts in `demo_*.py` files
- Contact package maintainers for missing data files

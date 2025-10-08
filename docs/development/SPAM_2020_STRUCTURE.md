# SPAM 2020 Data Structure Documentation

## Overview

This document describes the structure of SPAM 2020 (Spatial Production Allocation Model) data files and documents any differences from SPAM 2010.

## File Structure

### Production File
- **Filename**: `spam2020V2r0_global_P_TA.csv`
- **Location**: `spam2020V2r0_global_production/spam2020V2r0_global_production/`
- **Total Columns**: 59
- **Unit**: Metric tons (mt)

### Harvest Area File
- **Filename**: `spam2020V2r0_global_H_TA.csv`
- **Location**: `spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/`
- **Total Columns**: 59
- **Unit**: Hectares (ha)

## Column Structure

### Metadata Columns (13 columns)

| Column Name | Description | Type | Notes |
|------------|-------------|------|-------|
| `grid_code` | Unique grid cell identifier | Integer | Primary key |
| `x` | Longitude coordinate | Float | Range: -180 to 180 |
| `y` | Latitude coordinate | Float | Range: -90 to 90 |
| `FIPS0` | Country code (ISO3) | String | 2-letter ISO3 code |
| `FIPS1` | State/province code | String | Administrative level 1 |
| `FIPS2` | District code | String | Administrative level 2 |
| `ADM0_NAME` | Country name | String | Full country name |
| `ADM1_NAME` | State/province name | String | Administrative level 1 name |
| `ADM2_NAME` | District name | String | Administrative level 2 name |
| `rec_type` | Record type | String | 'P' for production, 'H' for harvest |
| `tech_type` | Technology type | String | 'A' for all technologies |
| `unit` | Unit of measurement | String | 'mt' or 'ha' |
| `year_data` | Data year | Integer | 2020 |

### Crop Columns (46 columns)

All crop columns use uppercase 4-letter codes with `_A` suffix (indicating "All technologies").

#### Key Cereal Crops
- `WHEA_A` - Wheat
- `RICE_A` - Rice
- `MAIZ_A` - Maize (Corn)
- `BARL_A` - Barley
- `SORG_A` - Sorghum
- `MILL_A` - Millet
- `OCER_A` - Other cereals

#### Other Major Crops
- `SOYB_A` - Soybeans
- `BEAN_A` - Beans
- `CASS_A` - Cassava
- `POTA_A` - Potatoes
- `SWPO_A` - Sweet potatoes
- `YAMS_A` - Yams
- `SUGC_A` - Sugarcane
- `SUGB_A` - Sugar beets
- `COTT_A` - Cotton
- `RAPE_A` - Rapeseed
- `SUNF_A` - Sunflower
- `GROU_A` - Groundnuts
- `COCO_A` - Coconuts
- `COFF_A` - Coffee
- `TEAS_A` - Tea
- `TOBA_A` - Tobacco
- `RUBB_A` - Rubber

And many more (see full list in data files).

## Key Findings

### Coordinate System
- **Projection**: WGS84 (EPSG:4326)
- **Resolution**: 5 arcminutes (~10km at equator)
- **X (Longitude)**: -180 to 180 degrees
- **Y (Latitude)**: -90 to 90 degrees

### Country Identification
- **Primary**: `FIPS0` column contains ISO3 country codes (e.g., "NO" for Norway, "US" for United States)
- **Alternative**: `ADM0_NAME` contains full country names
- **Note**: FIPS0 is the recommended field for country matching

### State/Province Identification
- **FIPS1**: State/province code
- **ADM1_NAME**: State/province name (use for text matching)

### Data Consistency
- Production and harvest area files have **identical structure**
- Same grid cells, same countries, same crops
- Only difference is the values (production vs harvest area)

## Differences from SPAM 2010

### Similarities
1. Same basic structure (metadata + crop columns)
2. Same coordinate system (WGS84, 5 arcminute resolution)
3. Same crop naming convention (4-letter codes with _A suffix)
4. Same country code system (FIPS0 = ISO3)

### Potential Differences (to be verified if SPAM 2010 data available)
1. **Data year**: 2020 vs 2010
2. **File naming**: `spam2020V2r0` vs `spam2010` prefix
3. **Number of crops**: May have additional crops in 2020
4. **Grid coverage**: May have improved coverage in some regions
5. **Data quality**: Updated production estimates

## Usage Notes

### For AgriRichter Implementation

1. **Country Matching**: Use `FIPS0` column for ISO3 country code matching
2. **State Matching**: Use `ADM1_NAME` for state/province name matching
3. **Coordinate Access**: Use `x` and `y` columns directly (no transformation needed)
4. **Crop Selection**: 
   - Wheat: `WHEA_A`
   - Rice: `RICE_A`
   - Maize: `MAIZ_A`
   - Allgrain: Sum of `WHEA_A`, `RICE_A`, `MAIZ_A`, `BARL_A`, `SORG_A`, `MILL_A`, `OCER_A`, `OILP_A`

5. **Unit Conversions**:
   - Production: metric tons → grams → kcal (using crop-specific caloric content)
   - Harvest area: hectares → km² (multiply by 0.01)

### Data Loading Best Practices

```python
import pandas as pd

# Efficient loading with optimized dtypes
dtypes = {
    'grid_code': 'int32',
    'FIPS0': 'category',
    'FIPS1': 'category',
    'ADM0_NAME': 'category',
    'ADM1_NAME': 'category',
    'rec_type': 'category',
    'tech_type': 'category',
    'unit': 'category',
}

# Crop columns as float32 for memory efficiency
crop_cols = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'SORG_A', 'MILL_A', 'OCER_A']
for col in crop_cols:
    dtypes[col] = 'float32'

df = pd.read_csv('spam2020V2r0_global_P_TA.csv', dtype=dtypes)
```

## Validation Checklist

- [x] Production file exists and is readable
- [x] Harvest area file exists and is readable
- [x] All required metadata columns present
- [x] All key crop columns present (WHEA_A, RICE_A, MAIZ_A, etc.)
- [x] Coordinate columns (x, y) present and valid ranges
- [x] FIPS0 contains valid ISO3 codes
- [x] Both files have identical structure
- [x] Files are consistent with each other

## References

- SPAM 2020 Documentation: https://www.mapspam.info/
- IFPRI SPAM Data Portal: https://dataverse.harvard.edu/dataverse/SPAM

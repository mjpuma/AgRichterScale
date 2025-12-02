# AgriRichter Data Filtering Methodology

## Overview

This document describes the systematic data filtering approach used in AgriRichter to ensure robust and reliable analysis results. The filtering methodology addresses data quality issues in the SPAM 2020 dataset that can lead to unrealistic yield calculations and poor envelope convergence.

## SPAM 2020 Grid Structure

### Grid Specifications
- **Resolution:** 0.083302° × 0.083302° (regular grid)
- **Cell Size:** ~9.25 km × 9.25 km at equator
- **Theoretical Cell Area:** 85.50 km²
- **Total Cells:** 355,748 (wheat), 424,356 (maize), 156,700 (rice)

### Data Quality Issues Identified

1. **Extreme Yield Outliers:** Cells with normal production values but extremely small harvest areas
2. **Unrealistic Yields:** Up to 73 million mt/km² (physically impossible)
3. **Tiny Harvest Areas:** Minimum values of 0.001 km² (0.1 hectares)

## Filtering Methodology

### Primary Filtering Criteria

#### 1. Minimum Harvest Area Threshold
**Threshold:** 0.01 km² (1 hectare)

**Rationale:**
- Removes cells with harvest areas < 1 hectare
- Eliminates data artifacts from SPAM processing
- Addresses rounding errors in very small grid cells
- Maintains 87% of wheat cells (removes 13.0%)

**Impact by Crop:**
- **Wheat:** Removes 46,375 cells (13.0%)
- **Maize:** Removes ~55,000 cells (~13.0%)
- **Rice:** Removes ~20,000 cells (~13.0%)

#### 2. Maximum Yield Threshold
**Threshold:** 2× 99th percentile yield per crop

**Rationale:**
- Caps extreme yield outliers while preserving high-productivity regions
- Crop-specific thresholds account for different yield distributions
- Removes <1% of cells per crop

**Thresholds by Crop:**
- **Wheat:** 4,834,218 mt/km²
- **Maize:** 5,552,240 mt/km²
- **Rice:** 5,437,620 mt/km²

### Alternative Threshold Options

#### Conservative Filtering (5th percentile)
- **Threshold:** 0.002 km²
- **Impact:** Removes 3.7% of cells
- **Use Case:** Maximum data retention

#### Moderate Filtering (1% of cell area)
- **Threshold:** 0.855 km²
- **Impact:** Removes 48.3% of cells
- **Use Case:** Focus on substantial agricultural areas

#### Aggressive Filtering (0.1 km²)
- **Threshold:** 0.1 km²
- **Impact:** Removes 28.8% of cells
- **Use Case:** High-confidence analysis

## Implementation

### Data Validation Class

```python
class SPAMDataFilter:
    """Consistent data filtering for SPAM 2020 data."""
    
    # Default thresholds
    MIN_HARVEST_AREA_KM2 = 0.01  # 1 hectare
    YIELD_PERCENTILE_MULTIPLIER = 2.0  # 2x 99th percentile
    
    @staticmethod
    def filter_crop_data(production, harvest_area, crop_name):
        """Apply consistent filtering to crop data."""
        # Convert harvest area to km²
        harvest_km2 = harvest_area / 100
        
        # Basic validity filter
        valid_mask = (production > 0) & (harvest_km2 > 0)
        
        # Minimum harvest area filter
        area_mask = harvest_km2 >= SPAMDataFilter.MIN_HARVEST_AREA_KM2
        
        # Calculate yields for outlier detection
        yields = production / harvest_km2
        yield_99th = np.percentile(yields[valid_mask], 99)
        max_yield = yield_99th * SPAMDataFilter.YIELD_PERCENTILE_MULTIPLIER
        yield_mask = yields <= max_yield
        
        # Combined filter
        final_mask = valid_mask & area_mask & yield_mask
        
        return final_mask, {
            'total_cells': len(production),
            'valid_cells': np.sum(valid_mask),
            'area_filtered': np.sum(valid_mask) - np.sum(valid_mask & area_mask),
            'yield_filtered': np.sum(valid_mask & area_mask) - np.sum(final_mask),
            'final_cells': np.sum(final_mask),
            'yield_99th': yield_99th,
            'max_yield_threshold': max_yield
        }
```

### Application Across Components

#### 1. Envelope Calculation
- Applied in `envelope_builder.py` and `envelope_v2.py`
- Ensures convergence by removing extreme outliers
- Documented in envelope metadata

#### 2. Event Calculation
- Applied in `event_calculator.py`
- Consistent cell selection for historical events
- Maintains comparability across events

#### 3. Visualization
- Applied in all map and plot generation
- Consistent data representation
- Clear indication of filtered data

#### 4. Statistical Analysis
- Applied in summary statistics
- Documented impact on totals and averages
- Separate reporting of filtered vs. unfiltered results

## Documentation Requirements

### Analysis Reports
All analysis outputs must include:

1. **Filtering Summary:**
   - Total cells before filtering
   - Cells removed by each filter
   - Final cell count and percentage retained

2. **Impact Assessment:**
   - Change in total production/harvest area
   - Effect on yield distributions
   - Envelope convergence improvement

3. **Methodology Reference:**
   - Link to this document
   - Specific thresholds used
   - Justification for any deviations

### Example Documentation

```
Data Filtering Applied:
- Minimum harvest area: 0.01 km² (1 hectare)
- Maximum yield: 2× 99th percentile per crop
- Wheat: 355,748 → 309,373 cells (87.0% retained)
- Impact: Envelope width improved from 78.5% to <1%
- Methodology: docs/DATA_FILTERING_METHODOLOGY.md
```

## Validation and Quality Assurance

### Pre-Filtering Diagnostics
- Yield distribution analysis
- Outlier identification
- Spatial distribution check

### Post-Filtering Validation
- Envelope convergence verification
- Conservation law compliance
- Yield reasonableness assessment

### Sensitivity Analysis
- Test multiple threshold values
- Document impact on key metrics
- Validate against external data sources

## Maintenance and Updates

### Threshold Review
- Annual review of filtering thresholds
- Comparison with new SPAM releases
- Integration of improved data sources

### Methodology Evolution
- Track filtering effectiveness
- Incorporate user feedback
- Update based on scientific literature

## References

1. SPAM 2020 Technical Documentation
2. AgriRichter Envelope Convergence Analysis
3. Agricultural Yield Validation Studies
4. Spatial Data Quality Assessment Guidelines

---

**Document Version:** 1.0  
**Last Updated:** October 16, 2025  
**Next Review:** October 2026
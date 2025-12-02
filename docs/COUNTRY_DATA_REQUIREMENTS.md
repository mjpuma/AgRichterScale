# Country Data Requirements and Validation Procedures

## Overview

This document outlines the data requirements, validation procedures, and quality standards for adding new countries to the multi-tier envelope system. It provides technical specifications and validation criteria to ensure reliable agricultural analysis.

## Data Requirements

### Primary Data Sources

#### SPAM 2020 Data
- **Source**: Spatial Production Allocation Model (SPAM) 2020
- **Resolution**: ~10km grid cells globally
- **Coverage**: Global agricultural production and harvest area data
- **Crops**: Wheat, maize, rice (minimum required)
- **Format**: CSV files with standardized column structure

#### Required SPAM Files
1. **Production Data**: `spam2020V2r0_global_P_TA.csv`
2. **Harvest Area Data**: `spam2020V2r0_global_H_TA.csv`
3. **Yield Data**: `spam2020V2r0_global_Y_TA.csv` (derived from production/area)

#### SPAM Data Structure
```
Required Columns:
- x: Longitude (decimal degrees)
- y: Latitude (decimal degrees)  
- iso3: Country code (FIPS format)
- cell5m: Cell identifier
- [crop]_[system]: Production/area values by crop and system
  - wheat_a: Wheat all systems
  - maize_a: Maize all systems  
  - rice_a: Rice all systems
```

### Country Identification

#### FIPS Codes
Countries are identified using FIPS codes embedded in SPAM data:
- **USA**: 'US'
- **China**: 'CH'
- **Brazil**: 'BR'
- **India**: 'IN'
- **Russia**: 'RS'
- **Argentina**: 'AR'
- **Canada**: 'CA'
- **Australia**: 'AS'
- **Ukraine**: 'UP'
- **France**: 'FR'

#### Verification Process
1. Check SPAM documentation for correct FIPS codes
2. Validate FIPS code exists in loaded SPAM data
3. Verify geographic extent matches expected country boundaries
4. Confirm sufficient data coverage for analysis

## Validation Criteria

### Minimum Data Coverage

#### Cell Count Requirements
- **Large Countries** (>5M km²): 2000+ cells minimum
- **Medium Countries** (1-5M km²): 1000+ cells minimum  
- **Small Countries** (<1M km²): 500+ cells minimum

#### Geographic Coverage
- **Latitude Range**: >2° for meaningful agricultural diversity
- **Longitude Range**: >2° for meaningful agricultural diversity
- **Agricultural Regions**: Coverage of known major agricultural areas

#### Crop Coverage Requirements
- **Priority Crops**: Each priority crop must have >5% cell coverage
- **Data Quality**: >70% of cells with valid (non-zero) production data
- **Spatial Distribution**: Crops distributed across multiple regions

### Data Quality Standards

#### Coordinate Validation
- **Completeness**: >95% of cells have valid coordinates
- **Range Check**: Coordinates within expected country boundaries
- **Precision**: Coordinates to at least 2 decimal places

#### Production Data Validation
- **Non-negative Values**: All production values ≥ 0
- **Realistic Ranges**: Production values within expected agricultural ranges
- **Consistency**: Production and harvest area data are consistent
- **Missing Data**: <30% missing values for priority crops

#### Yield Validation
- **Realistic Yields**: Calculated yields within reasonable ranges for each crop
- **Spatial Consistency**: Yield patterns consistent with known agricultural regions
- **Outlier Detection**: Identification and handling of extreme yield values

## Validation Procedures

### Automated Validation Pipeline

#### Step 1: Data Availability Check
```python
def validate_data_availability(country_code: str, fips_code: str) -> Dict[str, Any]:
    """Check if SPAM data exists for the country."""
    
    checks = {
        'fips_code_found': False,
        'production_data_available': False,
        'harvest_area_data_available': False,
        'coordinate_data_complete': False,
        'cell_count': 0
    }
    
    # Implementation details in country_framework.py
    return checks
```

#### Step 2: Geographic Extent Validation
```python
def validate_geographic_extent(country_data: pd.DataFrame, 
                             expected_extent: Optional[float] = None) -> Dict[str, Any]:
    """Validate geographic extent and coverage."""
    
    extent_checks = {
        'lat_range': 0.0,
        'lon_range': 0.0,
        'total_extent_km2': 0.0,
        'extent_reasonable': False,
        'coverage_adequate': False
    }
    
    # Calculate actual extent
    lat_range = country_data['y'].max() - country_data['y'].min()
    lon_range = country_data['x'].max() - country_data['x'].min()
    
    # Approximate area calculation (rough)
    extent_km2 = lat_range * lon_range * (111 * 111)  # 1° ≈ 111km
    
    extent_checks.update({
        'lat_range': lat_range,
        'lon_range': lon_range,
        'total_extent_km2': extent_km2,
        'extent_reasonable': lat_range > 1.0 and lon_range > 1.0,
        'coverage_adequate': extent_km2 > 10000  # Minimum 10,000 km²
    })
    
    return extent_checks
```

#### Step 3: Crop Coverage Validation
```python
def validate_crop_coverage(country_data: pd.DataFrame, 
                          priority_crops: List[str],
                          min_coverage_percent: float = 5.0) -> Dict[str, Any]:
    """Validate crop coverage and data quality."""
    
    crop_validation = {}
    
    for crop in priority_crops:
        crop_columns = [col for col in country_data.columns 
                       if crop.lower() in col.lower()]
        
        if crop_columns:
            # Aggregate crop data across systems
            crop_data = country_data[crop_columns].sum(axis=1)
            cells_with_crop = (crop_data > 0).sum()
            coverage_percent = (cells_with_crop / len(country_data)) * 100
            
            crop_validation[crop] = {
                'columns_found': crop_columns,
                'cells_with_data': cells_with_crop,
                'total_cells': len(country_data),
                'coverage_percent': coverage_percent,
                'meets_minimum': coverage_percent >= min_coverage_percent,
                'total_production': crop_data.sum(),
                'mean_production': crop_data[crop_data > 0].mean() if cells_with_crop > 0 else 0,
                'data_quality_score': calculate_data_quality_score(crop_data)
            }
        else:
            crop_validation[crop] = {
                'columns_found': [],
                'error': f'No columns found for crop: {crop}'
            }
    
    return crop_validation
```

#### Step 4: Data Quality Assessment
```python
def calculate_data_quality_score(crop_data: pd.Series) -> float:
    """Calculate data quality score for crop data."""
    
    if len(crop_data) == 0:
        return 0.0
    
    # Quality metrics
    non_zero_ratio = (crop_data > 0).sum() / len(crop_data)
    non_null_ratio = crop_data.notna().sum() / len(crop_data)
    
    # Outlier detection (values > 99th percentile)
    if crop_data.sum() > 0:
        p99 = crop_data.quantile(0.99)
        outlier_ratio = (crop_data > p99).sum() / len(crop_data)
        outlier_penalty = min(outlier_ratio * 2, 0.3)  # Max 30% penalty
    else:
        outlier_penalty = 0
    
    # Combine metrics
    quality_score = (non_zero_ratio * 0.4 + non_null_ratio * 0.4 + 
                    (1 - outlier_penalty) * 0.2)
    
    return min(max(quality_score, 0.0), 1.0)
```

### Manual Validation Procedures

#### Visual Inspection
1. **Map Visualization**: Plot country data on map to verify boundaries
2. **Crop Distribution**: Visualize crop distribution patterns
3. **Yield Patterns**: Check yield patterns for agricultural realism
4. **Outlier Review**: Manually inspect extreme values

#### Agricultural Realism Check
1. **Production Totals**: Compare with FAO statistics
2. **Yield Ranges**: Verify yields are within realistic ranges
3. **Regional Patterns**: Confirm patterns match known agricultural regions
4. **Seasonal Consistency**: Check for seasonal crop patterns where applicable

#### Expert Review
1. **Agricultural Specialist**: Review by agricultural domain expert
2. **Regional Expert**: Review by expert familiar with country's agriculture
3. **Data Scientist**: Technical review of data quality and processing
4. **Policy Analyst**: Review of policy relevance and scenarios

## Quality Standards

### Data Completeness Standards

| Metric | Minimum Standard | Preferred Standard |
|--------|------------------|-------------------|
| Cell Count | Country-specific minimum | 2x minimum |
| Coordinate Completeness | 95% | 99% |
| Production Data Completeness | 70% | 85% |
| Priority Crop Coverage | 5% per crop | 10% per crop |
| Geographic Coverage | Major agricultural regions | All agricultural regions |

### Data Quality Standards

| Metric | Minimum Standard | Preferred Standard |
|--------|------------------|-------------------|
| Data Quality Score | 0.6 | 0.8 |
| Outlier Ratio | <10% | <5% |
| Yield Realism | Within 2x FAO range | Within 1.5x FAO range |
| Spatial Consistency | No major gaps | Continuous coverage |

### Performance Standards

| Metric | Minimum Standard | Preferred Standard |
|--------|------------------|-------------------|
| Validation Time | <5 minutes | <2 minutes |
| Memory Usage | <4GB | <2GB |
| Analysis Time | <10 minutes | <5 minutes |
| Error Rate | <5% | <1% |

## Validation Reporting

### Automated Reports

#### Validation Summary Report
```
Country: Brazil (BRA)
FIPS Code: BR
Validation Status: PASSED

Data Coverage:
✓ Cell Count: 2,847 cells (>2,000 required)
✓ Geographic Extent: 8.2M km² (reasonable)
✓ Coordinate Completeness: 99.8%

Crop Coverage:
✓ Maize: 2,156 cells (75.7% coverage)
✓ Wheat: 892 cells (31.3% coverage)  
✓ Rice: 445 cells (15.6% coverage)

Data Quality:
✓ Overall Quality Score: 0.82
✓ Outlier Ratio: 3.2%
✓ Production Data Completeness: 87.4%

Recommendations: None - Ready for production use
```

#### Detailed Validation Report
- Cell-by-cell data quality metrics
- Crop-specific coverage maps
- Yield distribution analysis
- Outlier identification and flagging
- Geographic coverage assessment

### Manual Review Documentation

#### Review Checklist
- [ ] Geographic boundaries verified against reference maps
- [ ] Production totals compared with FAO statistics
- [ ] Yield ranges reviewed for agricultural realism
- [ ] Regional patterns consistent with known agriculture
- [ ] Policy scenarios relevant for country context
- [ ] Template parameters appropriate for country characteristics

#### Expert Sign-off
- Agricultural Specialist: _________________ Date: _________
- Regional Expert: _________________ Date: _________
- Data Scientist: _________________ Date: _________
- Policy Analyst: _________________ Date: _________

## Troubleshooting Guide

### Common Validation Failures

#### FIPS Code Not Found
**Symptoms**: No data returned for country
**Causes**: 
- Incorrect FIPS code in template
- Country not included in SPAM data
- Data loading issues

**Solutions**:
1. Verify FIPS code against SPAM documentation
2. Check alternative country codes
3. Confirm SPAM data is properly loaded
4. Test with known working country first

#### Insufficient Data Coverage
**Symptoms**: Cell count below minimum threshold
**Causes**:
- Country boundaries too restrictive
- SPAM data gaps for the region
- Incorrect geographic filtering

**Solutions**:
1. Review country boundary definitions
2. Lower minimum cell requirements if appropriate
3. Check for data gaps in SPAM coverage
4. Verify coordinate system consistency

#### Low Crop Coverage
**Symptoms**: Priority crops below coverage threshold
**Causes**:
- Crops not significant for the country
- SPAM crop definitions don't match expectations
- Data quality issues for specific crops

**Solutions**:
1. Adjust priority crops for country's agricultural profile
2. Lower coverage thresholds if appropriate
3. Review SPAM crop definitions and naming
4. Check for crop-specific data quality issues

#### Poor Data Quality
**Symptoms**: Low quality scores, high outlier ratios
**Causes**:
- Data processing errors
- Extreme values in source data
- Coordinate system issues

**Solutions**:
1. Review data processing pipeline
2. Implement outlier detection and handling
3. Verify coordinate system transformations
4. Check source data quality

## Continuous Improvement

### Monitoring and Updates

#### Regular Validation
- Monthly validation of all active countries
- Quarterly comprehensive review
- Annual template updates
- Continuous monitoring of data quality metrics

#### Version Control
- Track template versions and changes
- Document validation criteria updates
- Maintain historical validation results
- Version control for validation procedures

#### Feedback Integration
- User feedback on country data quality
- Expert recommendations for improvements
- Performance monitoring and optimization
- Integration of new data sources and methods

### Future Enhancements

#### Planned Improvements
1. **Automated Boundary Detection**: Use satellite data for boundary validation
2. **Real-time Data Updates**: Integration with live agricultural data feeds
3. **Enhanced Quality Metrics**: More sophisticated data quality assessment
4. **Machine Learning Validation**: AI-powered outlier detection and quality assessment
5. **Multi-source Integration**: Combine SPAM with other agricultural datasets

#### Research Directions
1. **Sub-national Analysis**: State/province level validation procedures
2. **Temporal Validation**: Multi-year data consistency checks
3. **Crop-specific Standards**: Tailored validation for different crop types
4. **Climate Integration**: Climate data validation for agricultural realism
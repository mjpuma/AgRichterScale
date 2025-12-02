# Country Addition Guide

## Overview

This guide provides step-by-step procedures for adding new countries to the multi-tier envelope system. The framework is designed to support rapid addition of major agricultural producers with comprehensive validation and configuration management.

## Prerequisites

1. **SPAM 2020 Data**: Ensure SPAM 2020 data is loaded and accessible
2. **System Resources**: Verify sufficient memory for country-scale analysis (8GB+ recommended)
3. **Dependencies**: Confirm all required Python packages are installed (see requirements.txt)

## Quick Start

For countries with existing templates (USA, China, Brazil, India, Russia, Argentina, Canada, Australia, Ukraine, France):

**Note**: The framework includes all 4 key countries for research papers: USA, China, Brazil, and India.

```python
from agririchter.data.country_framework import CountryFramework
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.core.config import Config

# Initialize framework
config = Config()
boundary_manager = CountryBoundaryManager(config, spatial_mapper, grid_manager)
framework = CountryFramework(config, boundary_manager)

# Get and validate template
template = framework.get_country_template('BRA')  # Brazil example
validation_results = framework.validate_country_template(template)

# Create configuration if validation passes
if validation_results['overall_status'] == 'passed':
    config, validation = framework.create_country_configuration(template)
    # Add to boundary manager
    boundary_manager.COUNTRY_CONFIGURATIONS['BRA'] = config
    print("Brazil successfully added!")
else:
    print("Validation failed:", validation_results['errors'])
```

## Detailed Procedures

### Step 1: Check Available Templates

```python
# List all available country templates
available_countries = framework.get_available_templates()
print("Available templates:", available_countries)

# Get specific template details
template = framework.get_country_template('IND')  # India example
print(f"Template for {template.country_name}:")
print(f"  FIPS Code: {template.fips_code}")
print(f"  Priority Crops: {template.priority_crops}")
print(f"  Agricultural Focus: {template.agricultural_focus}")
```

### Step 2: Validate Template

```python
# Comprehensive validation
validation_results = framework.validate_country_template(template)

print(f"Validation Status: {validation_results['overall_status']}")
print(f"Checks Performed: {len(validation_results['checks'])}")

# Review specific checks
for check_name, check_result in validation_results['checks'].items():
    status = "✓" if check_result.get('passed', False) else "✗"
    print(f"  {status} {check_name}: {check_result.get('message', 'No message')}")

# Review recommendations
if validation_results['recommendations']:
    print("\nRecommendations:")
    for rec in validation_results['recommendations']:
        print(f"  - {rec}")
```

### Step 3: Create Configuration

```python
# Create country configuration from validated template
config, validation = framework.create_country_configuration(template, validate_data=True)

print(f"Created configuration for {config.country_name}")
print(f"  Country Code: {config.country_code}")
print(f"  FIPS Code: {config.fips_code}")
print(f"  Agricultural Focus: {config.agricultural_focus}")
```

### Step 4: Add to System

```python
# Add configuration to boundary manager
boundary_manager.COUNTRY_CONFIGURATIONS[template.country_code] = config

# Verify addition by testing data access
try:
    production_df, harvest_area_df = boundary_manager.get_country_data(template.country_code)
    print(f"✓ Successfully loaded {len(production_df)} cells for {config.country_name}")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
```

### Step 5: Test Analysis

```python
# Test with national envelope analyzer
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer

analyzer = NationalEnvelopeAnalyzer(template.country_code)

# Test with one crop first
try:
    results = analyzer.analyze_crop('wheat')
    print(f"✓ Analysis successful for wheat in {config.country_name}")
    print(f"  Envelope width reduction: {results.width_reduction_percent:.1f}%")
except Exception as e:
    print(f"✗ Analysis failed: {e}")
```

## Adding Custom Countries

For countries not in the pre-defined templates:

### Step 1: Create Custom Template

```python
# Create custom template
custom_template = framework.create_custom_template(
    country_code='NGA',  # Nigeria example
    country_name='Nigeria',
    fips_code='NI',  # Check SPAM documentation for correct FIPS code
    iso3_code='NGA',
    agricultural_focus='food_security',
    priority_crops=['wheat', 'maize', 'rice']
)

# Customize additional parameters
custom_template.min_cells_required = 1500
custom_template.min_crop_coverage_percent = 3.0
custom_template.regional_subdivisions = ['north', 'middle_belt', 'south']
custom_template.policy_scenarios = ['population_growth', 'climate_adaptation', 'food_security']
```

### Step 2: Save Template

```python
# Save template for future use
template_path = framework.save_template(custom_template)
print(f"Template saved to: {template_path}")

# Load template later
loaded_template = framework.load_template(template_path)
```

### Step 3: Validate and Add

Follow the same validation and addition steps as above.

## Batch Operations

### Validate Multiple Countries

```python
# Validate all available templates
batch_results = framework.batch_validate_templates()

# Generate summary report
report = framework.generate_batch_report(batch_results)
print(report)

# Validate specific countries
selected_countries = ['BRA', 'IND', 'RUS']
batch_results = framework.batch_validate_templates(selected_countries)
```

### Add Multiple Countries

```python
# Add all validated countries
successful_additions = []
failed_additions = []

for country_code, validation in batch_results.items():
    if validation['overall_status'] == 'passed':
        try:
            template = framework.get_country_template(country_code)
            config, _ = framework.create_country_configuration(template)
            boundary_manager.COUNTRY_CONFIGURATIONS[country_code] = config
            successful_additions.append(country_code)
        except Exception as e:
            failed_additions.append((country_code, str(e)))
    else:
        failed_additions.append((country_code, 'Validation failed'))

print(f"Successfully added: {successful_additions}")
print(f"Failed additions: {failed_additions}")
```

## Troubleshooting

### Common Issues

1. **FIPS Code Not Found**
   - Check SPAM documentation for correct country codes
   - Verify SPAM data includes the target country
   - Try alternative FIPS codes (some countries have multiple codes)

2. **Insufficient Data Coverage**
   - Lower `min_cells_required` in template
   - Check if country boundaries are correctly defined
   - Verify SPAM data quality for the region

3. **Low Crop Coverage**
   - Adjust `min_crop_coverage_percent` for the country's agricultural profile
   - Remove crops that are not significant for the country
   - Check SPAM crop definitions and naming conventions

4. **Geographic Extent Issues**
   - Verify expected geographic extent is reasonable
   - Check coordinate system and projection issues
   - Review boundary data sources

### Validation Debugging

```python
# Get detailed country statistics
stats = boundary_manager.get_country_statistics('BRA')
print("Country Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Generate comprehensive country report
report = boundary_manager.generate_country_report('BRA')
print(report)

# Check data quality
validation = boundary_manager.validate_country_data_coverage('BRA')
print("Data Coverage Validation:")
for key, value in validation.items():
    print(f"  {key}: {value}")
```

## Best Practices

### Template Design

1. **Realistic Requirements**: Set `min_cells_required` and `min_crop_coverage_percent` based on country size and agricultural diversity
2. **Appropriate Focus**: Choose `agricultural_focus` based on country's role in global agriculture:
   - `food_security`: For countries focused on domestic food production
   - `export_capacity`: For major agricultural exporters
   - `efficiency`: For countries with intensive, high-tech agriculture
3. **Relevant Crops**: Include only crops that are agriculturally significant for the country
4. **Regional Subdivisions**: Define meaningful agricultural regions for policy analysis

### Validation Strategy

1. **Start Small**: Test with one country before batch operations
2. **Incremental Validation**: Validate templates before creating configurations
3. **Data Quality Checks**: Always review data coverage and quality metrics
4. **Performance Testing**: Test analysis performance with representative datasets

### System Integration

1. **Backup Configurations**: Save working configurations before making changes
2. **Version Control**: Track template versions and changes
3. **Documentation**: Document any customizations or special requirements
4. **Testing**: Thoroughly test new countries with representative analyses

## Template Reference

### Available Templates

| Country | Code | FIPS | Focus | Priority Crops | Min Cells |
|---------|------|------|-------|----------------|-----------|
| United States | USA | US | export_capacity | wheat, maize, rice | 2500 |
| China | CHN | CH | food_security | wheat, maize, rice | 3500 |
| Brazil | BRA | BR | export_capacity | maize, wheat, rice | 2000 |
| India | IND | IN | food_security | wheat, rice, maize | 3000 |
| Russia | RUS | RS | export_capacity | wheat, maize, rice | 2500 |
| Argentina | ARG | AR | export_capacity | wheat, maize, rice | 1500 |
| Canada | CAN | CA | export_capacity | wheat, maize, rice | 1200 |
| Australia | AUS | AS | export_capacity | wheat, maize, rice | 1000 |
| Ukraine | UKR | UP | export_capacity | wheat, maize, rice | 1200 |
| France | FRA | FR | efficiency | wheat, maize, rice | 800 |

### Configuration Parameters

- **country_code**: 3-letter country identifier (ISO standard preferred)
- **country_name**: Full country name
- **fips_code**: FIPS code used in SPAM data (critical for data access)
- **iso3_code**: ISO3 country code for international standards
- **agricultural_focus**: Primary agricultural focus (food_security, export_capacity, efficiency)
- **priority_crops**: List of agriculturally important crops for the country
- **regional_subdivisions**: Optional list of meaningful agricultural regions
- **policy_scenarios**: Optional list of relevant policy scenarios for analysis
- **min_cells_required**: Minimum number of SPAM cells required for analysis
- **min_crop_coverage_percent**: Minimum percentage coverage required for priority crops

## Support

For additional support:

1. Check the [Multi-Tier Technical Guide](MULTI_TIER_TECHNICAL_GUIDE.md)
2. Review the [API Reference](MULTI_TIER_API_REFERENCE.md)
3. Consult the [Troubleshooting Guide](../TROUBLESHOOTING.md)
4. Examine working examples in the `examples/` directory

## Contributing

To contribute new country templates:

1. Create and validate the template using the framework
2. Test thoroughly with representative analyses
3. Document any special requirements or customizations
4. Submit template files and documentation updates
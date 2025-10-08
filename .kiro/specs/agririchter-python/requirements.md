# Requirements Document

## Introduction

This project involves recreating the AgriRichter Scale analysis from MATLAB to Python using SPAM 2020 data. The AgriRichter Scale is a quantitative framework for measuring the magnitude of agricultural production disruptions, similar to the Richter scale for earthquakes. The system analyzes historical disruption events (famines, droughts, conflicts) and creates visualizations showing the relationship between disrupted harvest area and production losses in kilocalories.

The analysis supports three crop types: 'allgrain' (8 grain crops), 'wheat', and 'rice', and generates three publication-quality figures for scientific publication.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to load and process SPAM 2020 agricultural data, so that I can analyze global crop production and harvest area patterns.

#### Acceptance Criteria

1. WHEN the system loads SPAM 2020 CSV files THEN it SHALL successfully import production data (metric tons) and harvest area data (hectares)
2. WHEN processing crop data THEN the system SHALL support filtering for 'allgrain' (crops 1-8), 'wheat' (crop 1), and 'rice' (crop 2)
3. WHEN converting units THEN the system SHALL convert production from metric tons to grams and then to kilocalories using crop-specific caloric content
4. WHEN converting harvest area THEN the system SHALL convert from hectares to square kilometers
5. IF data files are missing or corrupted THEN the system SHALL provide clear error messages and graceful failure

### Requirement 2

**User Story:** As a researcher, I want to generate a global production map, so that I can visualize the spatial distribution of crop production worldwide.

#### Acceptance Criteria

1. WHEN creating the production map THEN the system SHALL use Robinson projection for accurate global representation
2. WHEN displaying production data THEN the system SHALL apply log10 transformation to handle the wide range of production values
3. WHEN rendering the map THEN the system SHALL use a yellow-to-green colormap for production intensity
4. WHEN saving the map THEN the system SHALL generate publication-quality output at 300 DPI in SVG format
5. WHEN handling missing data THEN the system SHALL display ocean areas in light blue and land areas with no data in white

### Requirement 3

**User Story:** As a researcher, I want to compute the disrupted Harvest-Production (H-P) envelope, so that I can establish upper and lower bounds for production losses given different harvest area disruptions.

#### Acceptance Criteria

1. WHEN computing the envelope THEN the system SHALL sort grid cells by productivity (yield = production/harvest area)
2. WHEN calculating upper bounds THEN the system SHALL use cumulative sum starting with most productive cells
3. WHEN calculating lower bounds THEN the system SHALL use cumulative sum starting with least productive cells
4. WHEN defining disruption ranges THEN the system SHALL support different harvest area ranges for each crop type
5. WHEN handling edge cases THEN the system SHALL remove NaN values and zero production cells before calculations

### Requirement 4

**User Story:** As a researcher, I want to load and process historical disruption events, so that I can analyze the magnitude of past agricultural crises.

#### Acceptance Criteria

1. WHEN loading event data THEN the system SHALL process 21 historical disruption events including famines, droughts, and conflicts
2. WHEN calculating event impacts THEN the system SHALL compute total harvest area loss and production loss for each event
3. WHEN processing geographic data THEN the system SHALL handle both country-level and state/province-level disruption data
4. WHEN converting units THEN the system SHALL ensure harvest area is in hectares and production losses are in kilocalories
5. IF event data is incomplete THEN the system SHALL log warnings but continue processing with available data

### Requirement 5

**User Story:** As a researcher, I want to create the AgriRichter Scale visualization, so that I can display historical events on a magnitude scale similar to earthquake measurements.

#### Acceptance Criteria

1. WHEN plotting the scale THEN the system SHALL use log10 scale for both magnitude and production loss axes
2. WHEN displaying events THEN the system SHALL plot each historical event as a colored marker with event labels
3. WHEN calculating magnitude THEN the system SHALL use M_D = log10(disrupted harvest area in km²)
4. WHEN setting axis ranges THEN the system SHALL adjust ranges appropriately for each crop type (allgrain: 10-15.9, wheat: 10-15.2, rice: 10-15)
5. WHEN saving the figure THEN the system SHALL generate both EPS and JPG formats at publication quality

### Requirement 6

**User Story:** As a researcher, I want to create the H-P envelope visualization with threshold lines, so that I can show the relationship between disrupted harvest area and production losses with severity classifications.

#### Acceptance Criteria

1. WHEN plotting the envelope THEN the system SHALL display the disrupted H-P envelope as a filled gray area with transparency
2. WHEN adding threshold lines THEN the system SHALL plot T1 (green), T2 (yellow), T3 (orange), and T4 (red) horizontal threshold lines
3. WHEN plotting events THEN the system SHALL use different colors and markers for different event categories (green diamonds, yellow squares, orange circles, red triangles)
4. WHEN setting axis limits THEN the system SHALL use appropriate ranges (x: 2.5-7.5, y: 9.5-16.5) for log10 scales
5. WHEN labeling axes THEN the system SHALL use clear scientific notation for harvest area (km²) and production (kcal)

### Requirement 7

**User Story:** As a researcher, I want a command-line interface, so that I can easily run the analysis for different crop types and configurations.

#### Acceptance Criteria

1. WHEN running the CLI THEN the system SHALL accept crop type parameters ('allgrain', 'wheat', 'rice')
2. WHEN specifying output THEN the system SHALL allow custom output directory specification
3. WHEN processing THEN the system SHALL provide progress indicators and status messages
4. WHEN encountering errors THEN the system SHALL display helpful error messages and exit gracefully
5. WHEN completing successfully THEN the system SHALL report generated files and their locations

### Requirement 8

**User Story:** As a developer, I want modular code architecture, so that I can maintain and extend the analysis components independently.

#### Acceptance Criteria

1. WHEN organizing code THEN the system SHALL separate data loading, processing, and visualization into distinct modules
2. WHEN handling configuration THEN the system SHALL use a centralized configuration system for parameters and file paths
3. WHEN implementing functions THEN each function SHALL have a single responsibility and clear input/output contracts
4. WHEN adding validation THEN the system SHALL include comprehensive data validation and error handling
5. WHEN documenting code THEN the system SHALL include docstrings and type hints for all public functions

### Requirement 9

**User Story:** As a researcher, I want data validation and quality checks, so that I can ensure the reliability of my analysis results.

#### Acceptance Criteria

1. WHEN loading data THEN the system SHALL validate file formats and required columns
2. WHEN processing coordinates THEN the system SHALL verify latitude/longitude values are within valid ranges
3. WHEN calculating statistics THEN the system SHALL check for reasonable value ranges and flag outliers
4. WHEN generating outputs THEN the system SHALL verify that all expected files are created successfully
5. IF validation fails THEN the system SHALL provide detailed error reports with suggested corrections

### Requirement 10

**User Story:** As a researcher, I want to save analysis results and intermediate data, so that I can reproduce results and perform additional analysis.

#### Acceptance Criteria

1. WHEN processing events THEN the system SHALL save event loss data to CSV files for each crop type
2. WHEN computing envelopes THEN the system SHALL save envelope boundary data for further analysis
3. WHEN generating figures THEN the system SHALL save both high-resolution images and source data
4. WHEN completing analysis THEN the system SHALL create a summary report with key statistics
5. WHEN organizing outputs THEN the system SHALL use consistent naming conventions and directory structure
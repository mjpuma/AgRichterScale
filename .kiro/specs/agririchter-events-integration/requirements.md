# Requirements Document

## Introduction

This project completes the AgriRichter Python implementation by adding the critical historical events calculation system and ensuring proper integration with SPAM 2020 gridded data. The system must replicate the MATLAB `AgriRichter_Events.m` functionality to calculate actual production losses and disrupted harvest areas for 21 historical agricultural disruption events (famines, droughts, conflicts).

The implementation will enable generation of publication-quality figures with real historical data: (1) Global production/harvest maps, (2) AgriRichter Scale with historical events, and (3) H-P Envelope with historical events plotted.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to update all data paths to use SPAM 2020 data, so that the analysis uses the most current agricultural production dataset.

#### Acceptance Criteria

1. WHEN configuring data paths THEN the system SHALL use SPAM 2020 CSV files instead of SPAM 2010
2. WHEN loading production data THEN the system SHALL read from `spam2020v2r0_global_P_TA.csv`
3. WHEN loading harvest area data THEN the system SHALL read from `spam2020v2r0_global_H_TA.csv`
4. WHEN validating data files THEN the system SHALL verify SPAM 2020 file structure and column names
5. IF SPAM 2020 files are missing THEN the system SHALL provide clear error messages with expected file locations

### Requirement 2

**User Story:** As a researcher, I want proper gridded data handling, so that spatial calculations are accurate and consistent across all analyses.

#### Acceptance Criteria

1. WHEN loading SPAM data THEN the system SHALL preserve x (longitude) and y (latitude) coordinates for each grid cell
2. WHEN processing grid cells THEN the system SHALL maintain the 5-arcminute resolution (0.0833° cell size)
3. WHEN aggregating crop data THEN the system SHALL correctly sum production and harvest area across selected crops
4. WHEN creating spatial grids THEN the system SHALL use consistent coordinate systems (WGS84)
5. WHEN validating grid data THEN the system SHALL check for coordinate consistency and missing spatial references

### Requirement 3

**User Story:** As a researcher, I want to load geographic boundary data, so that I can map historical event regions to SPAM grid cells.

#### Acceptance Criteria

1. WHEN loading country boundaries THEN the system SHALL read GDAM country shapefile or raster data
2. WHEN loading state/province boundaries THEN the system SHALL read GDAM state/province shapefile or raster data
3. WHEN processing boundary data THEN the system SHALL handle coordinate reference system transformations
4. WHEN mapping boundaries to grid cells THEN the system SHALL use spatial intersection algorithms
5. IF boundary files are missing THEN the system SHALL provide fallback to country-level ISO3 code matching

### Requirement 4

**User Story:** As a researcher, I want to load historical event definitions, so that I can identify which geographic regions were affected by each disruption.

#### Acceptance Criteria

1. WHEN loading event data THEN the system SHALL read from `DisruptionCountry.xls` with 21 event sheets
2. WHEN loading state-level data THEN the system SHALL read from `DisruptionStateProvince.xls` with event-specific provinces
3. WHEN parsing event sheets THEN the system SHALL extract country names, country codes, and state flags
4. WHEN processing state flags THEN the system SHALL identify events requiring state-level (flag=1) vs country-level (flag=0) processing
5. WHEN validating event data THEN the system SHALL verify all 21 events have required geographic information

### Requirement 5

**User Story:** As a researcher, I want to map event regions to SPAM grid cells, so that I can calculate actual production losses for each historical event.

#### Acceptance Criteria

1. WHEN processing country-level events THEN the system SHALL identify all SPAM grid cells within affected countries using ISO3 codes
2. WHEN processing state-level events THEN the system SHALL identify SPAM grid cells within specific provinces/states
3. WHEN matching geographic codes THEN the system SHALL use the country code conversion table (`CountryCode_Convert.xls`)
4. WHEN handling spatial matching THEN the system SHALL use either shapefile intersection or raster overlay methods
5. IF geographic matching fails THEN the system SHALL log warnings and continue with available data

### Requirement 6

**User Story:** As a researcher, I want to calculate event-specific production losses, so that I can quantify the magnitude of historical agricultural disruptions.

#### Acceptance Criteria

1. WHEN calculating losses THEN the system SHALL sum production (in kcal) from all affected SPAM grid cells for selected crops
2. WHEN calculating harvest area losses THEN the system SHALL sum harvest area (in hectares) from all affected grid cells
3. WHEN processing crop-specific events THEN the system SHALL filter for wheat (crop 1), rice (crop 2), or allgrain (crops 1-8)
4. WHEN converting units THEN the system SHALL convert metric tons to kcal using crop-specific caloric content
5. WHEN aggregating multi-country events THEN the system SHALL sum losses across all affected countries and states

### Requirement 7

**User Story:** As a researcher, I want to calculate AgriRichter magnitudes for historical events, so that I can compare event severity on a logarithmic scale.

#### Acceptance Criteria

1. WHEN calculating magnitude THEN the system SHALL use formula M_D = log10(disrupted harvest area in km²)
2. WHEN converting harvest area THEN the system SHALL convert hectares to km² (multiply by 0.01)
3. WHEN handling zero values THEN the system SHALL replace zeros with NaN to avoid log(0) errors
4. WHEN storing magnitudes THEN the system SHALL preserve both raw harvest area and calculated magnitude
5. WHEN validating magnitudes THEN the system SHALL check for reasonable ranges (typically 2-7 for historical events)

### Requirement 8

**User Story:** As a researcher, I want to generate a complete events dataset, so that I can use real historical data in all visualizations.

#### Acceptance Criteria

1. WHEN processing all events THEN the system SHALL create a DataFrame with 21 rows (one per event)
2. WHEN storing event data THEN the system SHALL include columns: event_name, harvest_area_loss_ha, production_loss_kcal, magnitude
3. WHEN saving event data THEN the system SHALL export to CSV with crop-specific naming (e.g., `events_wheat_spam2020.csv`)
4. WHEN validating output THEN the system SHALL verify no events have zero or NaN losses unless legitimately no data exists
5. WHEN comparing with MATLAB THEN the system SHALL produce results within 5% of original MATLAB calculations

### Requirement 9

**User Story:** As a researcher, I want to integrate real events data with the H-P envelope, so that I can visualize historical events within the production-disruption framework.

#### Acceptance Criteria

1. WHEN plotting H-P envelope THEN the system SHALL use calculated event magnitudes on x-axis
2. WHEN plotting event markers THEN the system SHALL use calculated production losses on y-axis
3. WHEN labeling events THEN the system SHALL display event names from the historical events list
4. WHEN coloring events THEN the system SHALL use severity classification based on AgriPhase thresholds
5. WHEN validating visualization THEN the system SHALL verify all 21 events appear on the plot with correct positions

### Requirement 10

**User Story:** As a researcher, I want to integrate real events data with the AgriRichter Scale, so that I can display historical events on the magnitude-loss scale.

#### Acceptance Criteria

1. WHEN plotting AgriRichter Scale THEN the system SHALL plot calculated event magnitudes vs production losses
2. WHEN adding event markers THEN the system SHALL use red filled circles for all historical events
3. WHEN adding event labels THEN the system SHALL position text labels to avoid overlap
4. WHEN drawing threshold lines THEN the system SHALL use AgriPhase 2-5 thresholds from USDA PSD data
5. WHEN saving figures THEN the system SHALL generate publication-quality outputs (300 DPI, SVG/EPS/JPG formats)

### Requirement 11

**User Story:** As a researcher, I want comprehensive data validation, so that I can trust the accuracy of calculated event losses.

#### Acceptance Criteria

1. WHEN loading SPAM data THEN the system SHALL validate total global production matches expected ranges
2. WHEN calculating event losses THEN the system SHALL verify losses are less than total global production
3. WHEN processing geographic matches THEN the system SHALL report match success rates (% of events with data)
4. WHEN comparing crops THEN the system SHALL validate crop-specific totals are consistent with SPAM documentation
5. WHEN generating reports THEN the system SHALL create validation summary with data quality metrics

### Requirement 12

**User Story:** As a researcher, I want an end-to-end pipeline script, so that I can generate all figures with one command using real data.

#### Acceptance Criteria

1. WHEN running the pipeline THEN the system SHALL load SPAM 2020 data, calculate events, and generate all figures
2. WHEN processing crops THEN the system SHALL support wheat, rice, and allgrain with single parameter
3. WHEN generating outputs THEN the system SHALL create organized directory structure (figures/, data/, reports/)
4. WHEN handling errors THEN the system SHALL provide informative messages and continue with partial results when possible
5. WHEN completing successfully THEN the system SHALL generate summary report with statistics and file locations

### Requirement 13

**User Story:** As a researcher, I want performance optimization for large datasets, so that the analysis completes in reasonable time.

#### Acceptance Criteria

1. WHEN loading SPAM data THEN the system SHALL use efficient pandas operations (avoid row-by-row iteration)
2. WHEN performing spatial operations THEN the system SHALL use vectorized GeoPandas operations where possible
3. WHEN processing multiple events THEN the system SHALL cache SPAM data to avoid repeated file reads
4. WHEN calculating aggregations THEN the system SHALL use NumPy vectorized operations
5. WHEN monitoring performance THEN the system SHALL complete full analysis (all 21 events, 3 crops) in under 10 minutes

### Requirement 14

**User Story:** As a developer, I want comprehensive logging, so that I can debug issues and track analysis progress.

#### Acceptance Criteria

1. WHEN processing events THEN the system SHALL log progress for each event (e.g., "Processing event 5/21: DustBowl")
2. WHEN encountering errors THEN the system SHALL log detailed error messages with context
3. WHEN completing calculations THEN the system SHALL log summary statistics (total losses, affected areas)
4. WHEN validating data THEN the system SHALL log validation results and warnings
5. WHEN configuring logging THEN the system SHALL support different log levels (DEBUG, INFO, WARNING, ERROR)

### Requirement 15

**User Story:** As a researcher, I want to compare Python results with MATLAB outputs, so that I can validate the migration accuracy.

#### Acceptance Criteria

1. WHEN calculating event losses THEN the system SHALL produce results within 5% of MATLAB values
2. WHEN generating figures THEN the system SHALL match MATLAB figure layouts and styling
3. WHEN computing magnitudes THEN the system SHALL use identical formulas to MATLAB implementation
4. WHEN saving outputs THEN the system SHALL create comparison reports showing Python vs MATLAB differences
5. IF differences exceed 5% THEN the system SHALL flag events for manual review and investigation

# Implementation Plan

- [x] 1. Update configuration for SPAM 2020 data paths

  - [x] 1.1 Update Config class with SPAM 2020 file paths

    - Modify `agririchter/core/config.py` to use spam2020v2r0 file names
    - Add configuration parameter for SPAM version (2010 vs 2020)
    - Update default data directory paths for SPAM 2020 location
    - Add validation to check if SPAM 2020 files exist at configured paths
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Verify SPAM 2020 column names and structure
    - Read SPAM 2020 CSV files and validate column names match expected format
    - Verify crop columns (whea_a, rice_a, maiz_a, etc.) are present
    - Check coordinate columns (x, y) and metadata columns (iso3, cell5m)
    - Document any differences between SPAM 2010 and SPAM 2020 structure
    - _Requirements: 1.4, 2.1, 2.5_

- [x] 2. Implement Grid Data Manager

  - [x] 2.1 Create GridDataManager class with SPAM data loading

    - Create `agririchter/data/grid_manager.py` module
    - Implement `load_spam_data()` to read production and harvest area CSVs
    - Add efficient pandas dtypes for memory optimization (category for iso3, float32 for values)
    - Preserve x, y coordinates for all grid cells
    - Add caching mechanism to avoid repeated file reads
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 13.1, 13.3_

  - [x] 2.2 Implement spatial indexing for grid cells

    - Create GeoDataFrame with Point geometries from x, y coordinates
    - Build spatial index (R-tree) for efficient geographic queries
    - Implement `create_spatial_index()` method
    - Add `get_grid_cells_by_coordinates()` for bounding box queries
    - Test spatial index performance with sample queries
    - _Requirements: 2.2, 2.4, 13.2, 13.3_

  - [x] 2.3 Add grid cell filtering by ISO3 country code

    - Implement `get_grid_cells_by_iso3()` method
    - Create efficient lookup using pandas groupby on iso3 column
    - Cache country-to-grid-cell mappings for reuse
    - Handle cases where ISO3 code not found in data
    - _Requirements: 2.1, 5.1, 13.3_

  - [x] 2.4 Implement crop aggregation methods

    - Create `get_crop_production()` to sum production across selected crops
    - Create `get_crop_harvest_area()` to sum harvest area across selected crops
    - Handle crop index to column name mapping (1-based to column names)
    - Apply crop-specific caloric content for production conversion
    - Validate aggregation results against expected ranges
    - _Requirements: 2.3, 6.1, 6.2, 6.3_

  - [x] 2.5 Add grid data validation
    - Implement `validate_grid_data()` method
    - Check total global production is within expected range
    - Verify coordinate ranges (lat: -90 to 90, lon: -180 to 180)
    - Check for missing or NaN values in critical columns
    - Generate validation report with data quality metrics
    - _Requirements: 2.5, 11.1, 11.4_

- [x] 3. Implement Spatial Mapper

  - [x] 3.1 Create SpatialMapper class with country code mapping

    - Create `agririchter/data/spatial_mapper.py` module
    - Implement `load_country_codes_mapping()` to read CountryCode_Convert.xls
    - Create `get_iso3_from_country_code()` for GDAM to ISO3 conversion
    - Handle multiple country code systems (GDAM, FAOSTAT, ISO3)
    - Add validation for country code mapping completeness
    - _Requirements: 3.1, 3.2, 5.3_

  - [x] 3.2 Implement country-level grid cell mapping

    - Create `map_country_to_grid_cells()` method
    - Use ISO3 code to query GridDataManager for country's grid cells
    - Return list of grid cell IDs for the country
    - Log mapping statistics (number of cells found)
    - Handle countries with no SPAM data gracefully
    - _Requirements: 5.1, 5.3, 5.5_

  - [x] 3.3 Add optional boundary shapefile support

    - Implement `load_boundary_data()` for GDAM shapefiles
    - Add spatial intersection method for precise boundary matching
    - Create fallback logic: try ISO3 first, then shapefile if available
    - Handle coordinate reference system transformations
    - Document shapefile requirements and optional nature
    - _Requirements: 3.1, 3.2, 3.3, 5.4_

  - [x] 3.4 Implement state/province-level mapping

    - Create `map_state_to_grid_cells()` method
    - Handle state code matching using name_adm1 column in SPAM data
    - Support both numeric state codes and string state names
    - Filter grid cells by both country and state
    - Log state-level mapping success rates
    - _Requirements: 5.2, 5.3, 5.5_

  - [x] 3.5 Add spatial mapping validation
    - Implement `validate_spatial_mapping()` method
    - Calculate mapping success rate (% of events with grid cells found)
    - Report coverage statistics per event
    - Identify events with zero grid cells matched
    - Generate spatial mapping quality report
    - _Requirements: 5.5, 11.3, 14.4_

- [x] 4. Implement Event Calculator

  - [x] 4.1 Create EventCalculator class structure

    - Create `agririchter/analysis/event_calculator.py` module
    - Initialize with Config, GridDataManager, and SpatialMapper
    - Set up logging for event processing progress
    - Create data structures for storing event results
    - _Requirements: 6.1, 6.2, 14.1_

  - [x] 4.2 Implement single event calculation

    - Create `calculate_single_event()` method
    - Process event definition (countries, states, flags)
    - Determine if country-level or state-level processing needed
    - Aggregate losses across all affected regions
    - Return dictionary with harvest area loss and production loss
    - _Requirements: 6.1, 6.2, 6.5_

  - [x] 4.3 Add country-level loss calculation

    - Implement `calculate_country_level_loss()` method
    - Map country code to ISO3 using SpatialMapper
    - Query GridDataManager for country's grid cells
    - Sum production and harvest area for selected crops
    - Convert units: metric tons → grams → kcal
    - _Requirements: 5.1, 6.1, 6.2, 6.4_

  - [x] 4.4 Add state-level loss calculation

    - Implement `calculate_state_level_loss()` method
    - Map state codes to grid cells using SpatialMapper
    - Handle multiple states per event
    - Aggregate losses across all affected states
    - Log state-level processing details
    - _Requirements: 5.2, 6.1, 6.2, 6.5_

  - [x] 4.5 Implement magnitude calculation

    - Create `calculate_magnitude()` method
    - Convert harvest area from hectares to km² (multiply by 0.01)
    - Apply log10 transformation: M_D = log10(area_km2)
    - Handle zero values by replacing with NaN
    - Validate magnitude ranges (typically 2-7 for historical events)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 4.6 Implement batch event processing

    - Create `calculate_all_events()` method
    - Load event definitions from EventsProcessor
    - Process all 21 events sequentially with progress logging
    - Store results in DataFrame with columns: event_name, harvest_area_loss_ha, production_loss_kcal, magnitude
    - Handle errors gracefully, continue with remaining events
    - _Requirements: 8.1, 8.2, 14.1, 14.2_

  - [x] 4.7 Add event results validation
    - Implement `validate_event_results()` method
    - Check no event has losses exceeding global production
    - Verify all events have non-zero losses (or document why zero)
    - Validate magnitude ranges are reasonable
    - Flag events with suspicious values for review
    - _Requirements: 8.4, 11.2, 11.3_

- [ ] 5. Integrate events with existing visualizations

  - [x] 5.1 Update H-P Envelope visualization to use real events

    - Modify `agririchter/visualization/hp_envelope.py`
    - Replace sample events data with calculated events DataFrame
    - Plot event magnitudes on x-axis (log10 scale)
    - Plot production losses on y-axis (log10 scale)
    - Add event name labels with adjustText for non-overlapping placement
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 5.2 Update AgriRichter Scale visualization to use real events

    - Modify `agririchter/visualization/agririchter_scale.py`
    - Replace sample events data with calculated events DataFrame
    - Plot event magnitudes vs production losses
    - Use red filled circles for event markers
    - Add event labels with proper positioning
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 5.3 Add event severity classification to visualizations
    - Color-code events by AgriPhase threshold classification
    - Use different marker shapes for different severity levels
    - Add legend showing severity classifications
    - Ensure threshold lines are visible and properly labeled
    - _Requirements: 9.4, 10.4_

- [x] 6. Create end-to-end pipeline orchestrator

  - [x] 6.1 Create EventsPipeline class

    - Create `agririchter/pipeline/events_pipeline.py` module
    - Initialize with Config and output directory
    - Set up logging for pipeline stages
    - Create methods for each pipeline stage
    - _Requirements: 12.1, 14.1_

  - [x] 6.2 Implement data loading stage

    - Create `load_all_data()` method
    - Load SPAM 2020 production and harvest area data
    - Load event definitions from Excel files
    - Load country code mapping and boundary data
    - Return dictionary with all loaded data
    - _Requirements: 12.1, 13.3_

  - [x] 6.3 Implement event calculation stage

    - Create `calculate_events()` method
    - Initialize GridDataManager, SpatialMapper, EventCalculator
    - Run batch event processing for all 21 events
    - Return events DataFrame with calculated losses
    - Log calculation progress and statistics
    - _Requirements: 12.1, 14.1, 14.3_

  - [x] 6.4 Implement visualization generation stage

    - Create `generate_visualizations()` method
    - Generate global production map
    - Generate H-P envelope with real events
    - Generate AgriRichter Scale with real events
    - Return dictionary of figure objects
    - _Requirements: 12.1, 12.3_

  - [x] 6.5 Implement results export stage

    - Create `export_results()` method
    - Save events DataFrame to CSV (crop-specific naming)
    - Save all figures in multiple formats (SVG, EPS, JPG, PNG)
    - Create organized directory structure (figures/, data/, reports/)
    - Return dictionary of exported file paths
    - _Requirements: 8.3, 12.3, 12.5_

  - [x] 6.6 Add pipeline summary report generation

    - Create `generate_summary_report()` method
    - Include statistics: total events, total losses, magnitude ranges
    - List all generated files with paths
    - Include validation results and data quality metrics
    - Save report as text file and return as string
    - _Requirements: 12.5, 14.3_

  - [x] 6.7 Implement complete pipeline execution
    - Create `run_complete_pipeline()` method
    - Execute all stages in sequence: load → calculate → visualize → export
    - Add error handling with informative messages
    - Continue with partial results if non-critical errors occur
    - Return comprehensive results dictionary
    - _Requirements: 12.1, 12.2, 12.4_

- [x] 7. Add comprehensive validation module

  - [x] 7.1 Create DataValidator class

    - Create `agririchter/validation/data_validator.py` module
    - Initialize with Config
    - Set up validation thresholds and expected ranges
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 7.2 Implement SPAM data validation

    - Create `validate_spam_data()` method
    - Check total global production for each crop
    - Verify coordinate completeness and ranges
    - Validate crop-specific totals against SPAM documentation
    - Return validation results dictionary
    - _Requirements: 11.1, 11.4_

  - [x] 7.3 Implement event results validation

    - Create `validate_event_results()` method
    - Check losses don't exceed global production
    - Verify magnitude ranges are reasonable
    - Identify events with zero or suspicious losses
    - Calculate validation statistics
    - _Requirements: 11.2, 11.3_

  - [x] 7.4 Add MATLAB comparison functionality

    - Create `compare_with_matlab()` method
    - Load MATLAB reference results if available
    - Compare event losses (Python vs MATLAB)
    - Calculate percentage differences
    - Flag events with differences > 5%
    - _Requirements: 8.5, 15.1, 15.4, 15.5_

  - [x] 7.5 Implement validation report generation
    - Create `generate_validation_report()` method
    - Compile all validation results into comprehensive report
    - Include data quality metrics and comparison statistics
    - Format as readable text with sections
    - Save to file and return as string
    - _Requirements: 11.5, 15.4_

- [x] 8. Create main execution script

  - [x] 8.1 Create main pipeline script

    - Create `scripts/run_agririchter_analysis.py` script
    - Add command-line argument parsing (crop type, output dir, data paths)
    - Initialize Config with user-specified parameters
    - Create EventsPipeline instance
    - _Requirements: 12.1, 12.2_

  - [x] 8.2 Add crop-specific execution

    - Support --crop argument (wheat, rice, allgrain)
    - Run pipeline for specified crop type
    - Handle multiple crops with loop if --all flag provided
    - Log crop-specific progress
    - _Requirements: 12.2_

  - [x] 8.3 Add output organization

    - Create output directory structure automatically
    - Organize by crop type (outputs/wheat/, outputs/rice/, etc.)
    - Create subdirectories: figures/, data/, reports/
    - Use consistent file naming conventions
    - _Requirements: 12.3_

  - [x] 8.4 Implement error handling and logging

    - Set up logging configuration from command-line arguments
    - Add try-except blocks for graceful error handling
    - Provide informative error messages with context
    - Continue with partial results when possible
    - _Requirements: 12.4, 14.2, 14.5_

  - [x] 8.5 Add completion summary
    - Print summary of generated files
    - Display key statistics (total events, losses, etc.)
    - Report any warnings or errors encountered
    - Provide next steps or recommendations
    - _Requirements: 12.5, 14.3_

- [x] 9. Add performance optimizations

  - [x] 9.1 Optimize SPAM data loading

    - Use efficient pandas dtypes (category, float32)
    - Add chunked reading for very large files if needed
    - Implement data caching to avoid repeated reads
    - Profile memory usage and optimize
    - _Requirements: 13.1, 13.3_

  - [x] 9.2 Optimize spatial operations

    - Use vectorized GeoPandas operations
    - Build spatial index once, reuse for all queries
    - Cache country-to-grid-cell mappings
    - Avoid row-by-row iteration
    - _Requirements: 13.2, 13.3_

  - [x] 9.3 Add performance monitoring
    - Add timing for each pipeline stage
    - Log memory usage at key points
    - Report performance statistics in summary
    - Ensure full analysis completes in under 10 minutes
    - _Requirements: 13.4, 13.5_

- [ ] 10. Create comprehensive tests

  - [x] 10.1 Add unit tests for GridDataManager

    - Test SPAM data loading with sample data
    - Test spatial indexing creation
    - Test grid cell filtering by ISO3
    - Test crop aggregation calculations
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 10.2 Add unit tests for SpatialMapper

    - Test country code mapping
    - Test ISO3 lookup
    - Test grid cell mapping methods
    - Test state-level mapping
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 10.3 Add unit tests for EventCalculator

    - Test single event calculation
    - Test magnitude calculation formula
    - Test unit conversions (MT→kcal, ha→km²)
    - Test loss aggregation
    - _Requirements: 6.1, 6.2, 7.1, 7.2_

  - [x] 10.4 Add integration tests for pipeline

    - Test end-to-end pipeline with sample data
    - Test all 21 events processing
    - Test figure generation with real events
    - Test output file creation
    - _Requirements: 12.1, 12.3_

  - [x] 10.5 Add validation tests
    - Test MATLAB comparison with reference data
    - Test data consistency checks
    - Test spatial coverage validation
    - Test figure quality verification
    - _Requirements: 15.1, 15.2, 15.3_

- [x] 11. Create documentation

  - [x] 11.1 Write user guide

    - Document installation requirements
    - Provide step-by-step usage instructions
    - Include example commands for different crops
    - Document expected outputs and file locations
    - _Requirements: 12.1, 12.2, 12.3_

  - [x] 11.2 Document data requirements

    - List required SPAM 2020 files and download sources
    - Document event definition Excel files structure
    - Describe optional boundary data files
    - Provide data directory structure recommendations
    - _Requirements: 1.1, 1.2, 3.1, 4.1_

  - [x] 11.3 Add API documentation

    - Write docstrings for all public methods
    - Add type hints to function signatures
    - Document class interfaces and responsibilities
    - Create API reference documentation
    - _Requirements: All components_

  - [x] 11.4 Create troubleshooting guide
    - Document common errors and solutions
    - Provide debugging tips
    - Include FAQ section
    - Add contact information for support
    - _Requirements: 12.4, 14.2_

- [x] 12. Validate against MATLAB outputs

  - [x] 12.1 Generate MATLAB reference outputs

    - Run original MATLAB code for wheat, rice, allgrain
    - Save event losses to CSV files
    - Save generated figures
    - Document MATLAB version and settings used
    - _Requirements: 15.1, 15.2_

  - [x] 12.2 Run Python pipeline and compare

    - Execute Python pipeline for all crops
    - Load MATLAB reference results
    - Compare event losses (should be within 5%)
    - Compare magnitudes (should match exactly)
    - _Requirements: 15.1, 15.3_

  - [x] 12.3 Investigate and document differences

    - Identify events with differences > 5%
    - Investigate root causes (rounding, missing data, etc.)
    - Document systematic differences
    - Update code or documentation as needed
    - _Requirements: 15.4, 15.5_

  - [x] 12.4 Create comparison report

    - Generate detailed comparison statistics
    - Include event-by-event comparison tables
    - Add visualizations comparing Python vs MATLAB
    - Document validation conclusions
    - _Requirements: 15.4_

  - [x] 12.5 Update validation thresholds if needed
    - Adjust 5% threshold if systematic differences found
    - Document rationale for any threshold changes
    - Update validation code with new thresholds
    - Re-run validation with updated thresholds
    - _Requirements: 15.5_

# Implementation Plan

- [x] 1. Set up project structure and core interfaces

  - Create directory structure for agririchter package with modules (core, data, processing, analysis, visualization, cli)
  - Define base interfaces and abstract classes for main components
  - Set up package configuration files (pyproject.toml, requirements.txt)
  - _Requirements: 8.1, 8.2_

- [x] 2. Implement configuration management system

  - [x] 2.1 Create Config class with crop-specific parameters

    - Implement crop type validation and parameter loading
    - Define caloric content dictionaries for each crop type
    - Set up threshold values (T1-T4) for different crops
    - _Requirements: 8.2, 7.1_

  - [x] 2.2 Implement file path management
    - Create path resolution for input data files
    - Set up output directory structure management
    - Add validation for file existence and permissions
    - _Requirements: 8.2, 9.1_

- [x] 3. Create data loading and validation module

  - [x] 3.1 Implement SPAM data loader

    - Write CSV parser for SPAM 2020 production data with proper data types
    - Create harvest area data loader with column validation
    - Add coordinate validation for latitude/longitude ranges
    - _Requirements: 1.1, 1.5, 9.2_

  - [x] 3.2 Implement historical events data loader

    - Create parser for disruption event data (21 historical events)
    - Handle both country-level and state/province-level data
    - Add data validation and completeness checks
    - _Requirements: 4.1, 4.4, 4.5_

  - [x] 3.3 Add comprehensive data validation
    - Implement file format and structure validation
    - Create data range and consistency checks
    - Add error reporting with detailed messages
    - _Requirements: 9.1, 9.3, 9.5_

- [x] 4. Implement data processing and transformation

  - [x] 4.1 Create unit conversion functions

    - Implement metric tons to grams to kilocalories conversion
    - Create hectares to square kilometers conversion
    - Add validation for conversion accuracy
    - _Requirements: 1.3, 1.4_

  - [x] 4.2 Implement crop filtering and selection

    - Create crop index filtering for allgrain (1-8), wheat (1), rice (2)
    - Add crop-specific data extraction
    - Implement data subset validation
    - _Requirements: 1.2_

  - [x] 4.3 Create grid data processing
    - Implement production grid creation with coordinate mapping
    - Add log10 transformation for visualization
    - Create yield calculation (production/harvest area)
    - _Requirements: 2.2, 3.1, 3.5_

- [x] 5. Implement core analysis engine

  - [x] 5.1 Create H-P envelope calculation

    - Implement grid cell sorting by productivity (yield)
    - Create cumulative sum calculations for upper and lower bounds
    - Add disruption area range handling for different crop types
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 5.1.1 Implement crop-dependent envelope discretization

    - Create MATLAB-exact disruption area ranges for each crop type (wheat: 1-2.2M km², rice: 1-1.2M km², allgrain: 1-7M km²)
    - Implement convergence detection algorithm to ensure envelope closure
    - Add proper bound truncation at convergence point following MATLAB algorithm
    - _Requirements: 3.1, 3.3, 3.4_

  - [x] 5.1.2 Fix envelope shading and visualization

    - Implement MATLAB-exact fill algorithm using concatenated boundary arrays
    - Add proper NaN and Inf value handling in envelope patch creation
    - Create light blue shading with correct transparency (alpha=0.6, color=[0.8, 0.9, 1.0])
    - _Requirements: 6.1, 6.3_

  - [x] 5.2 Implement AgriRichter magnitude calculation

    - Create magnitude formula: M_D = log10(disrupted harvest area in km²)
    - Add event impact calculation for historical events
    - Implement severity classification using T1-T4 thresholds
    - _Requirements: 5.3, 4.2, 4.3_

  - [x] 5.3 Create historical event processing

    - Implement event loss calculation for harvest area and production
    - Add geographic data processing for countries and states
    - Create event impact aggregation and validation
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 5.3.1 Implement shapefile-based geographic mapping

    - Create shapefile loader for country and administrative boundaries
    - Implement spatial intersection algorithm to map country/province names to SPAM grid cells
    - Add coordinate system transformation and projection handling
    - _Requirements: 4.1, 4.4, 4.5_

  - [ ] 5.3.2 Create event-to-grid mapping system

    - Parse event input files with country and province specifications
    - Use shapefiles to identify affected SPAM grid cells for each event
    - Calculate actual harvest area and production losses from affected grid cells
    - _Requirements: 4.1, 4.4, 4.5_

- [ ] 5.4 Add shapefile and geospatial data handling

  - [ ] 5.4.1 Implement shapefile data loader

    - Create GeoPandas-based shapefile reader for country boundaries
    - Add administrative boundary (state/province) shapefile support
    - Implement coordinate reference system (CRS) validation and transformation
    - _Requirements: 4.4, 4.5_

  - [ ] 5.4.2 Create spatial intersection engine

    - Implement point-in-polygon algorithm for SPAM grid cell assignment
    - Add buffering and tolerance handling for boundary edge cases
    - Create efficient spatial indexing for large-scale grid processing
    - _Requirements: 4.4, 4.5_

- [ ] 6. Implement visualization engine

  - [x] 6.1 Create global production map generator

    - Implement Robinson projection using Cartopy
    - Add yellow-to-green colormap for production intensity
    - Create coastline and ocean rendering with proper colors
    - _Requirements: 2.1, 2.3, 2.5_

  - [x] 6.2 Implement AgriRichter scale visualization

    - Create log10 scale plotting for magnitude vs production loss
    - Add historical event markers with labels and colors
    - Implement axis range adjustment for different crop types
    - _Requirements: 5.1, 5.2, 5.4, 5.5_

  - [x] 6.3 Create H-P envelope visualization

    - Implement filled envelope area with transparency
    - Add threshold lines (T1-T4) with appropriate colors
    - Create event plotting with category-specific markers and colors
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 6.4 Add publication-quality output
    - Implement 300 DPI output for all figure formats
    - Create SVG, EPS, and JPG export functionality
    - Add consistent styling and font management
    - _Requirements: 2.4, 5.5, 6.5_

- [x] 7. Implement MATLAB-exact AgriRichter system

  - [x] 7.1 Create USDA PSD data loader and threshold calculator

    - Load USDA PSD stocks, production, and consumption CSV data
    - Implement crop filtering and year range selection
    - Calculate dynamic AgriPhase thresholds (2-5) using SUR percentiles
    - Replace hardcoded thresholds with USDA-based calculations
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 7.2 Implement AgriRichter Scale visualization

    - Create magnitude-based x-axis (M_D = log10(A_H))
    - Plot theoretical production loss line using uniform production assumption
    - Add historical events as red circles with event name labels
    - Include AgriPhase threshold lines (2-5) with IPC phase colors
    - Use logarithmic y-scale with proper axis limits
    - _Requirements: 6.2, 6.4, 7.3_

  - [x] 7.3 Create MATLAB-exact H-P Envelope visualization

    - Use magnitude (M_D = log10(A_H)) on x-axis instead of raw harvest area
    - Implement gray envelope fill between upper and lower cumulative bounds
    - Add upper bound (black) and lower bound (blue) boundary lines
    - Plot historical events as red circles with labels
    - Include AgriPhase threshold lines as horizontal dashed lines
    - Set proper axis limits (xlim=[2,7], ylim=[1e10,1.62e16])
    - _Requirements: 6.1, 6.3, 6.4, 7.3_

  - [x] 7.4 Implement Balance Time Series visualization
    - Create 2x2 subplot layout for SUR, Stocks, Production, Consumption
    - Plot time series with median and percentile reference lines
    - Add proper datetime x-axis formatting and grid styling
    - Save as PNG with crop-specific naming
    - _Requirements: 7.2, 8.3_

- [x] 8. Create output management system

  - [x] 8.1 Implement file organization

    - Create output directory structure with consistent naming
    - Add CSV export for event loss data by crop type
    - Implement envelope data saving for further analysis
    - _Requirements: 10.1, 10.2, 10.5_

  - [x] 8.2 Generate analysis reports
    - Create summary report with key statistics
    - Add data validation results and quality metrics
    - Implement analysis metadata and provenance tracking
    - _Requirements: 10.4, 9.4_

- [ ] 9. Implement command-line interface

  - [ ] 9.1 Create CLI argument parsing

    - Add crop type selection (allgrain, wheat, rice)
    - Implement output directory specification
    - Create configuration override options
    - _Requirements: 7.1, 7.2_

  - [ ] 9.2 Add user interaction features
    - Implement progress indicators for long-running operations
    - Create informative status messages and logging
    - Add graceful error handling with helpful messages
    - _Requirements: 7.3, 7.4, 7.5_

- [ ] 10. Create comprehensive test suite

  - [ ] 10.1 Implement unit tests

    - Write tests for data loading and validation functions
    - Create unit conversion accuracy tests
    - Add mathematical calculation verification tests
    - _Requirements: 8.4, 9.3_

  - [ ] 10.2 Add integration tests

    - Create end-to-end pipeline tests
    - Test cross-module data flow and communication
    - Add file I/O operation validation tests
    - _Requirements: 8.4, 9.4_

  - [ ] 10.3 Implement validation tests
    - Create result comparison tests with reference data
    - Add figure quality and format validation
    - Implement data integrity checks throughout pipeline
    - _Requirements: 9.3, 9.4_

- [ ] 11. Add documentation and packaging

  - [ ] 11.1 Create comprehensive documentation

    - Write API documentation with docstrings and type hints
    - Create user guide with examples and tutorials
    - Add installation and setup instructions
    - _Requirements: 8.5_

  - [ ] 11.2 Implement packaging and distribution
    - Set up Python package structure with proper metadata
    - Create installation scripts and dependency management
    - Add example data and configuration files
    - _Requirements: 8.1, 8.2_

- [ ] 12. Performance optimization and validation

  - [ ] 12.1 Optimize processing performance

    - Implement memory-efficient data processing for large datasets
    - Add vectorized operations using NumPy for calculations
    - Create chunked processing for memory management
    - _Requirements: 8.4, 9.3_

  - [ ] 12.2 Validate against original MATLAB results
    - Compare output figures with original MATLAB versions
    - Verify numerical accuracy of calculations and conversions
    - Test with sample data to ensure result consistency
    - _Requirements: 9.3, 9.4_

- [ ] 13. Calibrate and validate thresholds

  - [ ] 13.1 Analyze global production patterns and historical events

    - Calculate accurate global production estimates for each crop type
    - Research historical agricultural disruption events and their magnitudes
    - Analyze relationship between disrupted area and production losses
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 13.2 Recalibrate threshold values based on data analysis

    - Adjust T1-T4 thresholds to represent reasonable percentages of global production
    - Validate thresholds against historical event classifications
    - Account for global reserves, trade, and food security margins
    - _Requirements: 4.2, 4.3, 5.3_

  - [ ] 13.3 Calibrate envelope discretization ranges
    - Analyze total harvest area per crop type to determine maximum disruption ranges
    - Implement MATLAB-exact discretization: wheat (1-2.2M km²), rice (1-1.2M km²), allgrain (1-7M km²)
    - Validate envelope closure and convergence behavior for each crop type
    - _Requirements: 3.1, 3.3, 3.4_

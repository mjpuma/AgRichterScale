# AgriRichter API Reference

## Overview

This document provides detailed API documentation for all public classes and methods in the AgriRichter package. Use this reference when integrating AgriRichter into your own Python applications.

## Table of Contents

- [Core Modules](#core-modules)
  - [Config](#config)
  - [Constants](#constants)
  - [Performance](#performance)
- [Data Modules](#data-modules)
  - [GridDataManager](#griddatamanager)
  - [SpatialMapper](#spatialmapper)
  - [EventsProcessor](#eventsprocessor)
  - [USDALoader](#usdaloader)
- [Analysis Modules](#analysis-modules)
  - [EventCalculator](#eventcalculator)
  - [AgriRichter](#agririchter)
  - [Envelope](#envelope)
- [Pipeline Modules](#pipeline-modules)
  - [EventsPipeline](#eventspipeline)
- [Validation Modules](#validation-modules)
  - [DataValidator](#datavalidator)
- [Visualization Modules](#visualization-modules)
  - [HPEnvelopeVisualizer](#hpenvelopevisualizer)
  - [AgriRichterScaleVisualizer](#agririchterscalevisualizer)
  - [MapsVisualizer](#mapsvisualizer)

---

## Core Modules

### Config

**Module:** `agririchter.core.config`

Configuration management for AgriRichter analysis.

#### Class: `Config`

```python
class Config:
    """Configuration class for AgriRichter analysis."""
    
    def __init__(
        self,
        crop_type: str = 'wheat',
        spam_version: str = '2020',
        data_dir: str = '.',
        ancillary_dir: str = 'ancillary',
        output_dir: str = 'outputs',
        log_level: str = 'INFO'
    )
```

**Parameters:**
- `crop_type` (str): Crop to analyze ('wheat', 'rice', 'allgrain')
- `spam_version` (str): SPAM data version ('2010', '2020')
- `data_dir` (str): Root directory containing SPAM data
- `ancillary_dir` (str): Directory containing ancillary files
- `output_dir` (str): Directory for output files
- `log_level` (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

**Attributes:**
- `crop_indices` (List[int]): Crop indices for selected crop type
- `spam_production_file` (str): Path to SPAM production CSV
- `spam_harvest_file` (str): Path to SPAM harvest area CSV
- `spam_yield_file` (str): Path to SPAM yield CSV
- `country_disruption_file` (str): Path to country disruption Excel
- `state_disruption_file` (str): Path to state disruption Excel
- `country_code_file` (str): Path to country code conversion Excel

**Methods:**

```python
def validate_paths(self) -> bool:
    """Validate that all required data files exist."""
```

```python
def get_crop_name(self) -> str:
    """Get human-readable crop name."""
```

```python
def get_output_path(self, filename: str) -> str:
    """Get full path for output file."""
```

**Example:**

```python
from agririchter.core.config import Config

# Create configuration for wheat analysis
config = Config(
    crop_type='wheat',
    spam_version='2020',
    data_dir='/path/to/spam',
    output_dir='outputs/wheat'
)

# Validate paths
if config.validate_paths():
    print("All data files found")
```

---

### Constants

**Module:** `agririchter.core.constants`

Global constants used throughout AgriRichter.

**Constants:**

```python
# Crop indices (1-based)
CROP_WHEAT = 1
CROP_RICE = 2
CROP_MAIZE = 3
CROP_BARLEY = 4
CROP_SORGHUM = 5
CROP_MILLET = 6
CROP_OATS = 7
CROP_RYE = 8

# Crop names
CROP_NAMES = {
    1: 'wheat',
    2: 'rice',
    3: 'maize',
    # ... etc
}

# Caloric content (kcal/g)
CALORIC_CONTENT = {
    'wheat': 3.39,
    'rice': 3.65,
    'maize': 3.65,
    # ... etc
}

# Unit conversions
MT_TO_GRAMS = 1e6
HA_TO_KM2 = 0.01
```

---

### Performance

**Module:** `agririchter.core.performance`

Performance monitoring and optimization utilities.

#### Class: `PerformanceMonitor`

```python
class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, name: str = "AgriRichter")
```

**Methods:**

```python
def start_timer(self, operation: str) -> None:
    """Start timing an operation."""
```

```python
def stop_timer(self, operation: str) -> float:
    """Stop timing and return elapsed time in seconds."""
```

```python
def log_memory_usage(self, label: str = "") -> None:
    """Log current memory usage."""
```

```python
def get_summary(self) -> Dict[str, Any]:
    """Get performance summary statistics."""
```

**Example:**

```python
from agririchter.core.performance import PerformanceMonitor

monitor = PerformanceMonitor("MyAnalysis")

monitor.start_timer("data_loading")
# ... load data ...
elapsed = monitor.stop_timer("data_loading")

monitor.log_memory_usage("After loading")

summary = monitor.get_summary()
print(f"Total time: {summary['total_time']:.2f}s")
```

---

## Data Modules

### GridDataManager

**Module:** `agririchter.data.grid_manager`

Manages SPAM gridded data with spatial indexing.

#### Class: `GridDataManager`

```python
class GridDataManager:
    """Manage SPAM 2020 gridded crop data."""
    
    def __init__(
        self,
        config: Config,
        cache_data: bool = True
    )
```

**Parameters:**
- `config` (Config): Configuration object
- `cache_data` (bool): Whether to cache loaded data in memory

**Methods:**

```python
def load_spam_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load SPAM production and harvest area data.
    
    Returns:
        Tuple of (production_df, harvest_area_df)
    """
```

```python
def create_spatial_index(self) -> None:
    """Create spatial index for efficient geographic queries."""
```

```python
def get_grid_cells_by_iso3(
    self,
    iso3_code: str
) -> pd.DataFrame:
    """
    Get all grid cells for a country.
    
    Parameters:
        iso3_code: ISO 3166-1 alpha-3 country code
        
    Returns:
        DataFrame with grid cells for the country
    """
```

```python
def get_grid_cells_by_coordinates(
    self,
    bounds: Tuple[float, float, float, float]
) -> pd.DataFrame:
    """
    Get grid cells within bounding box.
    
    Parameters:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        DataFrame with grid cells in bounds
    """
```

```python
def get_crop_production(
    self,
    grid_cells: pd.DataFrame,
    crop_indices: List[int]
) -> float:
    """
    Calculate total production for crops in grid cells.
    
    Parameters:
        grid_cells: DataFrame with grid cells
        crop_indices: List of crop indices (1-based)
        
    Returns:
        Total production in kilocalories
    """
```

```python
def get_crop_harvest_area(
    self,
    grid_cells: pd.DataFrame,
    crop_indices: List[int]
) -> float:
    """
    Calculate total harvest area for crops in grid cells.
    
    Parameters:
        grid_cells: DataFrame with grid cells
        crop_indices: List of crop indices (1-based)
        
    Returns:
        Total harvest area in hectares
    """
```

```python
def validate_grid_data(self) -> Dict[str, Any]:
    """
    Validate grid data quality.
    
    Returns:
        Dictionary with validation results
    """
```

**Example:**

```python
from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager

config = Config(crop_type='wheat')
grid_manager = GridDataManager(config)

# Load data
prod_df, harvest_df = grid_manager.load_spam_data()

# Get cells for USA
usa_cells = grid_manager.get_grid_cells_by_iso3('USA')

# Calculate production
production = grid_manager.get_crop_production(
    usa_cells,
    crop_indices=[1]  # wheat
)
print(f"USA wheat production: {production:.2e} kcal")
```

---

### SpatialMapper

**Module:** `agririchter.data.spatial_mapper`

Maps geographic regions to SPAM grid cells.

#### Class: `SpatialMapper`

```python
class SpatialMapper:
    """Map geographic regions to SPAM grid cells."""
    
    def __init__(
        self,
        config: Config,
        grid_manager: GridDataManager
    )
```

**Methods:**

```python
def load_country_codes_mapping(self) -> pd.DataFrame:
    """
    Load country code conversion table.
    
    Returns:
        DataFrame with country code mappings
    """
```

```python
def get_iso3_from_country_code(
    self,
    country_code: float
) -> Optional[str]:
    """
    Convert GDAM country code to ISO3.
    
    Parameters:
        country_code: Numeric GDAM country code
        
    Returns:
        ISO3 code or None if not found
    """
```

```python
def map_country_to_grid_cells(
    self,
    country_code: float
) -> List[str]:
    """
    Map country to SPAM grid cells.
    
    Parameters:
        country_code: Numeric GDAM country code
        
    Returns:
        List of grid cell IDs
    """
```

```python
def map_state_to_grid_cells(
    self,
    country_code: float,
    state_code: float
) -> List[str]:
    """
    Map state/province to SPAM grid cells.
    
    Parameters:
        country_code: Numeric GDAM country code
        state_code: Numeric state code
        
    Returns:
        List of grid cell IDs
    """
```

```python
def validate_spatial_mapping(self) -> Dict[str, Any]:
    """
    Validate spatial mapping quality.
    
    Returns:
        Dictionary with validation results
    """
```

**Example:**

```python
from agririchter.data.spatial_mapper import SpatialMapper

mapper = SpatialMapper(config, grid_manager)

# Map country to grid cells
country_code = 231  # USA
grid_cells = mapper.map_country_to_grid_cells(country_code)
print(f"Found {len(grid_cells)} grid cells for USA")

# Get ISO3 code
iso3 = mapper.get_iso3_from_country_code(country_code)
print(f"ISO3 code: {iso3}")
```

---

### EventsProcessor

**Module:** `agririchter.data.events`

Loads and processes historical event definitions.

#### Class: `EventsProcessor`

```python
class EventsProcessor:
    """Process historical disruption events."""
    
    def __init__(
        self,
        country_file: str = None,
        state_file: str = None
    )
```

**Methods:**

```python
def load_events(self) -> Dict[str, Any]:
    """
    Load all event definitions.
    
    Returns:
        Dictionary mapping event names to event data
    """
```

```python
def get_event(self, event_name: str) -> Dict[str, Any]:
    """
    Get data for a specific event.
    
    Parameters:
        event_name: Name of the event
        
    Returns:
        Dictionary with event data
    """
```

```python
def get_all_events(self) -> List[str]:
    """
    Get list of all event names.
    
    Returns:
        List of event names
    """
```

**Example:**

```python
from agririchter.data.events import EventsProcessor

processor = EventsProcessor()
events = processor.load_events()

# Get specific event
dust_bowl = processor.get_event('DustBowl')
print(f"Countries affected: {dust_bowl['countries']}")
```

---

### USDALoader

**Module:** `agririchter.data.usda`

Loads USDA PSD data for validation and thresholds.

#### Class: `USDALoader`

```python
class USDALoader:
    """Load USDA Production, Supply, and Distribution data."""
    
    def __init__(self, data_dir: str = 'USDAdata')
```

**Methods:**

```python
def load_production_data(self) -> pd.DataFrame:
    """Load global production data."""
```

```python
def load_consumption_data(self) -> pd.DataFrame:
    """Load global consumption data."""
```

```python
def load_stocks_data(self) -> pd.DataFrame:
    """Load global ending stocks data."""
```

```python
def calculate_global_totals(
    self,
    crop: str,
    year: int
) -> Dict[str, float]:
    """
    Calculate global totals for a crop and year.
    
    Parameters:
        crop: Crop name ('wheat', 'rice', 'maize')
        year: Year
        
    Returns:
        Dictionary with production, consumption, stocks
    """
```

---

## Analysis Modules

### EventCalculator

**Module:** `agririchter.analysis.event_calculator`

Calculates losses for historical events.

#### Class: `EventCalculator`

```python
class EventCalculator:
    """Calculate production losses for historical events."""
    
    def __init__(
        self,
        config: Config,
        grid_manager: GridDataManager,
        spatial_mapper: SpatialMapper
    )
```

**Methods:**

```python
def calculate_all_events(
    self,
    events_definitions: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate losses for all events.
    
    Parameters:
        events_definitions: Dictionary of event definitions
        
    Returns:
        DataFrame with event results
    """
```

```python
def calculate_single_event(
    self,
    event_name: str,
    event_data: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate losses for a single event.
    
    Parameters:
        event_name: Name of the event
        event_data: Event definition data
        
    Returns:
        Dictionary with harvest_area_loss_ha, production_loss_kcal, magnitude
    """
```

```python
def calculate_magnitude(
    self,
    harvest_area_ha: float
) -> float:
    """
    Calculate AgriRichter magnitude.
    
    Parameters:
        harvest_area_ha: Harvest area in hectares
        
    Returns:
        Magnitude (log10 scale)
    """
```

**Example:**

```python
from agririchter.analysis.event_calculator import EventCalculator

calculator = EventCalculator(config, grid_manager, spatial_mapper)

# Calculate all events
events_df = calculator.calculate_all_events(events_definitions)

# Calculate single event
result = calculator.calculate_single_event('DustBowl', dust_bowl_data)
print(f"Magnitude: {result['magnitude']:.2f}")
print(f"Production loss: {result['production_loss_kcal']:.2e} kcal")
```

---

### AgriRichter

**Module:** `agririchter.analysis.agririchter`

Main AgriRichter analysis engine.

#### Class: `AgriRichter`

```python
class AgriRichter:
    """Main AgriRichter analysis engine."""
    
    def __init__(self, config: Config)
```

**Methods:**

```python
def run_analysis(self) -> Dict[str, Any]:
    """
    Run complete AgriRichter analysis.
    
    Returns:
        Dictionary with analysis results
    """
```

```python
def calculate_thresholds(self) -> Dict[str, float]:
    """
    Calculate AgriPhase thresholds.
    
    Returns:
        Dictionary mapping phase names to threshold values
    """
```

---

### Envelope

**Module:** `agririchter.analysis.envelope`

H-P Envelope calculations.

#### Class: `EnvelopeCalculator`

```python
class EnvelopeCalculator:
    """Calculate H-P Envelope curves."""
    
    def __init__(self, config: Config)
```

**Methods:**

```python
def calculate_envelope(
    self,
    magnitudes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate envelope curves.
    
    Parameters:
        magnitudes: Array of magnitude values
        
    Returns:
        Tuple of (upper_envelope, lower_envelope)
    """
```

---

## Pipeline Modules

### EventsPipeline

**Module:** `agririchter.pipeline.events_pipeline`

End-to-end pipeline orchestration.

#### Class: `EventsPipeline`

```python
class EventsPipeline:
    """Orchestrate complete events analysis pipeline."""
    
    def __init__(
        self,
        config: Config,
        output_dir: str = 'outputs'
    )
```

**Methods:**

```python
def run_complete_pipeline(self) -> Dict[str, Any]:
    """
    Run complete analysis pipeline.
    
    Returns:
        Dictionary with all results
    """
```

```python
def load_all_data(self) -> Dict[str, Any]:
    """
    Load all required data.
    
    Returns:
        Dictionary with loaded data
    """
```

```python
def calculate_events(self) -> pd.DataFrame:
    """
    Calculate event losses.
    
    Returns:
        DataFrame with event results
    """
```

```python
def generate_visualizations(
    self,
    events_df: pd.DataFrame
) -> Dict[str, plt.Figure]:
    """
    Generate all visualizations.
    
    Parameters:
        events_df: DataFrame with event results
        
    Returns:
        Dictionary mapping figure names to Figure objects
    """
```

```python
def export_results(
    self,
    events_df: pd.DataFrame,
    figures: Dict[str, plt.Figure]
) -> Dict[str, List[str]]:
    """
    Export results to files.
    
    Parameters:
        events_df: DataFrame with event results
        figures: Dictionary of figures
        
    Returns:
        Dictionary with exported file paths
    """
```

```python
def generate_summary_report(
    self,
    results: Dict[str, Any]
) -> str:
    """
    Generate summary report.
    
    Parameters:
        results: Pipeline results
        
    Returns:
        Summary report as string
    """
```

**Example:**

```python
from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline

config = Config(crop_type='wheat')
pipeline = EventsPipeline(config, output_dir='outputs/wheat')

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Access results
events_df = results['events_dataframe']
figures = results['figures']
summary = results['summary_report']

print(summary)
```

---

## Validation Modules

### DataValidator

**Module:** `agririchter.validation.data_validator`

Data quality validation and MATLAB comparison.

#### Class: `DataValidator`

```python
class DataValidator:
    """Validate data quality and compare with MATLAB."""
    
    def __init__(self, config: Config)
```

**Methods:**

```python
def validate_spam_data(
    self,
    production_df: pd.DataFrame,
    harvest_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate SPAM data quality.
    
    Parameters:
        production_df: Production DataFrame
        harvest_df: Harvest area DataFrame
        
    Returns:
        Validation results dictionary
    """
```

```python
def validate_event_results(
    self,
    events_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate event calculation results.
    
    Parameters:
        events_df: DataFrame with event results
        
    Returns:
        Validation results dictionary
    """
```

```python
def compare_with_matlab(
    self,
    python_results: pd.DataFrame,
    matlab_file: str
) -> Dict[str, Any]:
    """
    Compare Python results with MATLAB outputs.
    
    Parameters:
        python_results: Python event results
        matlab_file: Path to MATLAB reference CSV
        
    Returns:
        Comparison results dictionary
    """
```

```python
def generate_validation_report(
    self,
    validation_results: Dict[str, Any]
) -> str:
    """
    Generate validation report.
    
    Parameters:
        validation_results: Validation results
        
    Returns:
        Report as string
    """
```

**Example:**

```python
from agririchter.validation.data_validator import DataValidator

validator = DataValidator(config)

# Validate SPAM data
spam_validation = validator.validate_spam_data(prod_df, harvest_df)

# Validate event results
event_validation = validator.validate_event_results(events_df)

# Compare with MATLAB
comparison = validator.compare_with_matlab(
    events_df,
    'matlab_events_wheat.csv'
)

# Generate report
report = validator.generate_validation_report({
    'spam': spam_validation,
    'events': event_validation,
    'matlab': comparison
})
print(report)
```

---

## Visualization Modules

### HPEnvelopeVisualizer

**Module:** `agririchter.visualization.hp_envelope`

H-P Envelope visualization.

#### Class: `HPEnvelopeVisualizer`

```python
class HPEnvelopeVisualizer:
    """Create H-P Envelope visualizations."""
    
    def __init__(self, config: Config)
```

**Methods:**

```python
def create_hp_envelope(
    self,
    events_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    show_labels: bool = True
) -> plt.Figure:
    """
    Create H-P Envelope figure.
    
    Parameters:
        events_df: DataFrame with event results
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
        show_labels: Whether to show event labels
        
    Returns:
        Matplotlib Figure object
    """
```

---

### AgriRichterScaleVisualizer

**Module:** `agririchter.visualization.agririchter_scale`

AgriRichter Scale visualization.

#### Class: `AgriRichterScaleVisualizer`

```python
class AgriRichterScaleVisualizer:
    """Create AgriRichter Scale visualizations."""
    
    def __init__(self, config: Config)
```

**Methods:**

```python
def create_scale_figure(
    self,
    events_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> plt.Figure:
    """
    Create AgriRichter Scale figure.
    
    Parameters:
        events_df: DataFrame with event results
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
        
    Returns:
        Matplotlib Figure object
    """
```

---

### MapsVisualizer

**Module:** `agririchter.visualization.maps`

Global production map visualization.

#### Class: `MapsVisualizer`

```python
class MapsVisualizer:
    """Create global production maps."""
    
    def __init__(self, config: Config)
```

**Methods:**

```python
def create_production_map(
    self,
    production_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300
) -> plt.Figure:
    """
    Create global production map.
    
    Parameters:
        production_df: DataFrame with production data
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
        
    Returns:
        Matplotlib Figure object
    """
```

---

## Type Hints

AgriRichter uses type hints throughout the codebase. Common types:

```python
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Common type aliases
EventData = Dict[str, Any]
ValidationResults = Dict[str, Any]
FigureDict = Dict[str, plt.Figure]
```

## Error Handling

All public methods may raise the following exceptions:

- `FileNotFoundError`: Required data file not found
- `ValueError`: Invalid parameter or data value
- `KeyError`: Missing required key in data structure
- `RuntimeError`: Unexpected error during processing

Always wrap API calls in try-except blocks:

```python
try:
    results = pipeline.run_complete_pipeline()
except FileNotFoundError as e:
    print(f"Data file missing: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Always validate configuration:**
   ```python
   config = Config(crop_type='wheat')
   if not config.validate_paths():
       raise RuntimeError("Data files not found")
   ```

2. **Use context managers for resources:**
   ```python
   with PerformanceMonitor("Analysis") as monitor:
       # ... perform analysis ...
   ```

3. **Cache data when processing multiple events:**
   ```python
   grid_manager = GridDataManager(config, cache_data=True)
   ```

4. **Check validation results:**
   ```python
   validation = validator.validate_event_results(events_df)
   if not validation['all_valid']:
       print(f"Warnings: {validation['warnings']}")
   ```

## Further Reading

- [User Guide](USER_GUIDE.md) - Installation and usage instructions
- [Data Requirements](DATA_REQUIREMENTS.md) - Required data files
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions

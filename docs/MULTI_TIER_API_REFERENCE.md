# Multi-Tier Envelope System: API Reference

## 📚 **API Overview**

The Multi-Tier Envelope System provides a comprehensive API for agricultural capacity analysis with productivity-based filtering. This reference covers all public classes, methods, and usage patterns.

## 🏗️ **Core Classes**

### **MultiTierEnvelopeEngine**

**Location:** `agririchter.analysis.multi_tier_envelope`

The core engine for multi-tier envelope calculations.

```python
class MultiTierEnvelopeEngine:
    """Core engine for multi-tier envelope calculations with validated methodology."""
    
    def __init__(self, spam_filter: Optional[SPAMDataFilter] = None, 
                 country_boundaries: Optional[CountryBoundaryManager] = None):
        """
        Initialize multi-tier engine.
        
        Args:
            spam_filter: Optional SPAM data filter instance
            country_boundaries: Optional country boundary manager
        """
```

#### **Methods**

##### **calculate_single_tier()**
```python
def calculate_single_tier(self, crop_data: CropDataset, 
                         tier: str = 'comprehensive',
                         custom_tier: Optional[TierConfiguration] = None) -> TierResults:
    """
    Calculate envelope bounds for a single tier.
    
    Args:
        crop_data: Crop dataset with production and harvest area data
        tier: Tier name ('comprehensive', 'commercial') or custom tier
        custom_tier: Custom tier configuration (overrides tier parameter)
        
    Returns:
        TierResults: Envelope bounds and tier statistics
        
    Raises:
        ValueError: If tier is not recognized and no custom_tier provided
        ValidationError: If results fail mathematical validation
        
    Example:
        >>> engine = MultiTierEnvelopeEngine()
        >>> wheat_data = load_crop_data('wheat')
        >>> results = engine.calculate_single_tier(wheat_data, tier='commercial')
        >>> print(f"Width reduction: {results.width_reduction:.1%}")
    """
```

##### **calculate_all_tiers()**
```python
def calculate_all_tiers(self, crop_data: CropDataset) -> MultiTierResults:
    """
    Calculate envelope bounds for all configured tiers.
    
    Args:
        crop_data: Crop dataset with production and harvest area data
        
    Returns:
        MultiTierResults: Results for all tiers with comparison metrics
        
    Example:
        >>> engine = MultiTierEnvelopeEngine()
        >>> wheat_data = load_crop_data('wheat')
        >>> all_results = engine.calculate_all_tiers(wheat_data)
        >>> for tier_name, results in all_results.tier_results.items():
        ...     print(f"{tier_name}: {results.width_reduction:.1%} reduction")
    """
```

##### **calculate_width_reductions()**
```python
def calculate_width_reductions(self, multi_tier_results: MultiTierResults) -> Dict[str, Dict]:
    """
    Calculate width reduction metrics across tiers.
    
    Args:
        multi_tier_results: Results from calculate_all_tiers()
        
    Returns:
        Dict: Width reduction metrics for each tier comparison
        
    Example:
        >>> width_analysis = engine.calculate_width_reductions(all_results)
        >>> commercial_reduction = width_analysis['commercial']['width_reduction']
        >>> print(f"Commercial tier reduces width by {commercial_reduction:.1%}")
    """
```

### **TierConfiguration**

**Location:** `agririchter.analysis.multi_tier_envelope`

Configuration class for defining productivity tiers.

```python
@dataclass
class TierConfiguration:
    """Configuration for a specific productivity tier."""
    
    name: str
    description: str
    yield_percentile_min: float
    yield_percentile_max: float
    policy_applications: List[str]
    target_users: List[str]
    
    def apply_filter(self, yield_data: np.ndarray) -> np.ndarray:
        """
        Apply tier-specific yield filtering.
        
        Args:
            yield_data: Array of yield values
            
        Returns:
            np.ndarray: Boolean mask for cells meeting tier criteria
            
        Example:
            >>> tier = TierConfiguration(
            ...     name='High Productivity',
            ...     yield_percentile_min=50,
            ...     yield_percentile_max=100,
            ...     # ... other parameters
            ... )
            >>> mask = tier.apply_filter(yield_data)
            >>> filtered_yields = yield_data[mask]
        """
```

#### **Predefined Tier Configurations**

```python
# Access predefined tiers
from agririchter.analysis.multi_tier_envelope import TIER_CONFIGURATIONS

# Available tiers
comprehensive_tier = TIER_CONFIGURATIONS['comprehensive']
commercial_tier = TIER_CONFIGURATIONS['commercial']

# Create custom tier
custom_tier = TierConfiguration(
    name='Premium Agriculture',
    description='Top 25% of yields (premium agriculture)',
    yield_percentile_min=75,
    yield_percentile_max=100,
    policy_applications=['premium_markets', 'export_quality'],
    target_users=['premium_producers', 'export_companies']
)
```

### **NationalEnvelopeAnalyzer**

**Location:** `agririchter.analysis.national_envelope_analyzer`

Handles country-specific envelope analysis.

```python
class NationalEnvelopeAnalyzer:
    """Analyzer for country-specific agricultural capacity assessment."""
    
    def __init__(self, country_code: str):
        """
        Initialize analyzer for specific country.
        
        Args:
            country_code: ISO country code (e.g., 'USA', 'CHN')
            
        Example:
            >>> usa_analyzer = NationalEnvelopeAnalyzer('USA')
            >>> china_analyzer = NationalEnvelopeAnalyzer('CHN')
        """
```

#### **Methods**

##### **analyze_national_capacity()**
```python
def analyze_national_capacity(self, crop_data: CropDataset, 
                             tier: str = 'commercial') -> NationalResults:
    """
    Analyze agricultural capacity for the country.
    
    Args:
        crop_data: Crop dataset (will be filtered to country boundaries)
        tier: Tier to use for analysis
        
    Returns:
        NationalResults: Country-specific capacity analysis
        
    Example:
        >>> usa_analyzer = NationalEnvelopeAnalyzer('USA')
        >>> wheat_data = load_crop_data('wheat')
        >>> usa_results = usa_analyzer.analyze_national_capacity(wheat_data, tier='commercial')
        >>> print(f"USA wheat capacity: {usa_results.total_production_capacity:.1f} million tons")
    """
```

##### **create_food_security_assessment()**
```python
def create_food_security_assessment(self, results: NationalResults) -> FoodSecurityReport:
    """
    Generate food security assessment report.
    
    Args:
        results: National analysis results
        
    Returns:
        FoodSecurityReport: Comprehensive food security assessment
        
    Example:
        >>> security_report = usa_analyzer.create_food_security_assessment(usa_results)
        >>> print(f"Food security status: {security_report.security_level}")
        >>> print(f"Import dependency: {security_report.import_dependency:.1%}")
    """
```

### **NationalComparisonAnalyzer**

**Location:** `agririchter.analysis.national_comparison_analyzer`

Compares agricultural capacity across countries.

```python
class NationalComparisonAnalyzer:
    """Analyzer for comparing national agricultural capacities."""
    
    def compare_countries(self, country_results: Dict[str, NationalResults]) -> ComparisonReport:
        """
        Compare agricultural capacity across countries.
        
        Args:
            country_results: Dictionary mapping country codes to their results
            
        Returns:
            ComparisonReport: Comprehensive comparison analysis
            
        Example:
            >>> comparator = NationalComparisonAnalyzer()
            >>> comparison = comparator.compare_countries({
            ...     'USA': usa_results,
            ...     'CHN': china_results,
            ...     'BRA': brazil_results
            ... })
            >>> print(f"Top producer: {comparison.rankings['production_capacity'][0]}")
        """
```

### **CountryBoundaryManager**

**Location:** `agririchter.data.country_boundary_manager`

Manages country boundary data and spatial filtering.

```python
class CountryBoundaryManager:
    """Manager for country boundary data and spatial operations."""
    
    def __init__(self, boundary_source: str = 'SPAM_FIPS'):
        """
        Initialize boundary manager.
        
        Args:
            boundary_source: Source for boundary data ('SPAM_FIPS', 'GADM')
        """
    
    def get_country_mask(self, lat: np.ndarray, lon: np.ndarray, 
                        country_code: str) -> np.ndarray:
        """
        Get boolean mask for cells within country boundaries.
        
        Args:
            lat: Latitude array
            lon: Longitude array  
            country_code: ISO country code
            
        Returns:
            np.ndarray: Boolean mask for country cells
            
        Example:
            >>> boundary_manager = CountryBoundaryManager()
            >>> usa_mask = boundary_manager.get_country_mask(lat, lon, 'USA')
            >>> usa_cells = crop_data[usa_mask]
        """
```

## 📊 **Data Structures**

### **CropDataset**

```python
@dataclass
class CropDataset:
    """Container for crop production and harvest area data."""
    
    production_kcal: pd.DataFrame  # Production in kcal
    harvest_km2: pd.DataFrame      # Harvest area in km²
    lat: np.ndarray               # Latitude coordinates
    lon: np.ndarray               # Longitude coordinates
    crop_name: str                # Crop identifier
    
    def apply_mask(self, mask: np.ndarray) -> 'CropDataset':
        """Apply boolean mask to filter dataset."""
        
    def get_yield_data(self) -> np.ndarray:
        """Calculate yield as production/area."""
        
    def validate_data_quality(self) -> DataQualityReport:
        """Validate data quality and completeness."""
```

### **TierResults**

```python
@dataclass
class TierResults:
    """Results from single tier envelope calculation."""
    
    envelope_bounds: Dict[str, np.ndarray]  # Upper and lower bounds
    tier_config: TierConfiguration          # Tier configuration used
    tier_statistics: Dict[str, float]       # Tier-specific statistics
    validation_results: ValidationReport    # Mathematical validation
    width_reduction: float                  # Width reduction vs comprehensive
    data_retention: float                   # Fraction of data retained
    
    def get_representative_width(self) -> float:
        """Get representative envelope width metric."""
        
    def export_bounds(self, format: str = 'csv') -> str:
        """Export envelope bounds to file."""
```

### **MultiTierResults**

```python
@dataclass
class MultiTierResults:
    """Results from multi-tier envelope calculation."""
    
    tier_results: Dict[str, TierResults]    # Results for each tier
    width_analysis: Dict[str, Dict]         # Width reduction analysis
    base_statistics: Dict[str, float]       # Base dataset statistics
    
    def get_tier_comparison(self) -> pd.DataFrame:
        """Get comparison table of all tiers."""
        
    def plot_tier_comparison(self) -> matplotlib.figure.Figure:
        """Create visualization comparing tiers."""
```

### **NationalResults**

```python
@dataclass
class NationalResults:
    """Results from national agricultural capacity analysis."""
    
    country_code: str
    country_name: str
    tier_results: TierResults
    boundary_statistics: Dict[str, float]
    total_production_capacity: float
    agricultural_efficiency: float
    
    def calculate_per_capita_metrics(self, population: float) -> Dict[str, float]:
        """Calculate per-capita agricultural metrics."""
        
    def assess_trade_potential(self) -> TradeAssessment:
        """Assess export/import potential."""
```

## 🚀 **Usage Examples**

### **Basic Multi-Tier Analysis**

```python
from agririchter.analysis import MultiTierEnvelopeEngine
from agririchter.data import SPAMDataLoader

# Load crop data
loader = SPAMDataLoader()
wheat_data = loader.load_crop_data('wheat')

# Initialize engine
engine = MultiTierEnvelopeEngine()

# Single tier analysis
commercial_results = engine.calculate_single_tier(wheat_data, tier='commercial')
print(f"Commercial tier width reduction: {commercial_results.width_reduction:.1%}")
print(f"Data retention: {commercial_results.data_retention:.1%}")

# Multi-tier analysis
all_results = engine.calculate_all_tiers(wheat_data)
width_analysis = engine.calculate_width_reductions(all_results)

for tier_name, analysis in width_analysis.items():
    print(f"{tier_name}: {analysis['width_reduction']:.1%} width reduction")
```

### **National Analysis Workflow**

```python
from agririchter.analysis import NationalEnvelopeAnalyzer, NationalComparisonAnalyzer

# Analyze individual countries
usa_analyzer = NationalEnvelopeAnalyzer('USA')
china_analyzer = NationalEnvelopeAnalyzer('CHN')

# Load crop data
wheat_data = loader.load_crop_data('wheat')

# Analyze each country
usa_results = usa_analyzer.analyze_national_capacity(wheat_data, tier='commercial')
china_results = china_analyzer.analyze_national_capacity(wheat_data, tier='commercial')

# Generate food security assessments
usa_security = usa_analyzer.create_food_security_assessment(usa_results)
china_security = china_analyzer.create_food_security_assessment(china_results)

# Compare countries
comparator = NationalComparisonAnalyzer()
comparison = comparator.compare_countries({
    'USA': usa_results,
    'CHN': china_results
})

print(f"USA production capacity: {usa_results.total_production_capacity:.1f} million tons")
print(f"China production capacity: {china_results.total_production_capacity:.1f} million tons")
print(f"Efficiency leader: {comparison.rankings['agricultural_efficiency'][0]}")
```

### **Custom Tier Configuration**

```python
from agririchter.analysis import TierConfiguration, MultiTierEnvelopeEngine

# Define custom tier for premium agriculture
premium_tier = TierConfiguration(
    name='Premium Agriculture',
    description='Top 25% of yields (export-quality production)',
    yield_percentile_min=75,
    yield_percentile_max=100,
    policy_applications=['export_promotion', 'premium_markets', 'quality_standards'],
    target_users=['exporters', 'premium_producers', 'quality_certifiers']
)

# Use custom tier
engine = MultiTierEnvelopeEngine()
premium_results = engine.calculate_single_tier(
    wheat_data, 
    custom_tier=premium_tier
)

print(f"Premium tier covers {premium_results.data_retention:.1%} of agricultural area")
print(f"Width reduction: {premium_results.width_reduction:.1%}")
```

### **Policy Scenario Analysis**

```python
from agririchter.analysis import PolicyScenarioAnalyzer

# Define policy scenarios
scenarios = [
    {
        'name': 'drought_resilience',
        'description': 'Assess capacity under drought conditions',
        'yield_reduction': 0.15,  # 15% yield reduction
        'area_reduction': 0.05    # 5% area reduction
    },
    {
        'name': 'trade_disruption',
        'description': 'Assess domestic capacity without imports',
        'import_restriction': True,
        'focus_tier': 'commercial'
    }
]

# Run scenario analysis
scenario_analyzer = PolicyScenarioAnalyzer('USA')
scenario_results = scenario_analyzer.run_scenarios(usa_results, scenarios)

for scenario_name, results in scenario_results.items():
    print(f"{scenario_name}: {results.capacity_impact:.1%} capacity change")
```

### **Performance Optimization**

```python
from agririchter.analysis import MultiTierEnvelopeEngine
from agririchter.core import MultiTierCache, ParallelMultiTierCalculator

# Enable caching for improved performance
cache = MultiTierCache(cache_dir='.cache/multi_tier')
engine = MultiTierEnvelopeEngine()
engine.enable_caching(cache)

# Use parallel processing for multiple crops
parallel_calculator = ParallelMultiTierCalculator(n_workers=4)

crops = ['wheat', 'maize', 'rice']
crop_results = {}

for crop in crops:
    crop_data = loader.load_crop_data(crop)
    crop_results[crop] = parallel_calculator.calculate_all_tiers_parallel(crop_data)

# Results are cached for future use
print("Results cached for future analysis")
```

### **Validation and Quality Assurance**

```python
from agririchter.analysis import EnvelopeValidator

# Validate results
validator = EnvelopeValidator()
validation_report = validator.validate_complete_system(all_results)

if validation_report.overall_status:
    print("✅ All validation tests passed")
else:
    print("❌ Validation issues found:")
    for issue in validation_report.recommendations:
        print(f"  - {issue}")

# Check specific mathematical properties
math_validation = validator.validate_mathematical_properties(commercial_results)
print(f"Monotonicity: {'✅' if math_validation['monotonicity'] else '❌'}")
print(f"Dominance: {'✅' if math_validation['dominance'] else '❌'}")
print(f"Conservation: {'✅' if math_validation['conservation'] else '❌'}")
```

## 🔧 **Configuration and Setup**

### **Environment Configuration**

```python
import os
from agririchter.core import Config

# Set up environment
os.environ['AGRIRICHTER_DATA_PATH'] = '/path/to/spam/data'
os.environ['AGRIRICHTER_CACHE_DIR'] = '/path/to/cache'

# Load configuration
config = Config.load_from_env()

# Initialize components with configuration
engine = MultiTierEnvelopeEngine(config=config)
```

### **Custom Configuration**

```python
from agririchter.core import Config

# Create custom configuration
config = Config(
    data_path='/custom/data/path',
    cache_enabled=True,
    cache_dir='/custom/cache',
    parallel_processing=True,
    max_workers=8,
    validation_strict_mode=True
)

# Use custom configuration
engine = MultiTierEnvelopeEngine(config=config)
```

## 🚨 **Error Handling**

### **Common Exceptions**

```python
from agririchter.exceptions import (
    ValidationError, 
    DataQualityError, 
    ConfigurationError,
    BoundaryError
)

try:
    results = engine.calculate_single_tier(crop_data, tier='commercial')
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Handle validation failure
except DataQualityError as e:
    print(f"Data quality issue: {e}")
    # Handle data quality problems
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
```

### **Graceful Error Recovery**

```python
def robust_multi_tier_analysis(crop_data: CropDataset) -> Optional[MultiTierResults]:
    """Robust analysis with error handling and recovery."""
    
    engine = MultiTierEnvelopeEngine()
    
    try:
        # Attempt full multi-tier analysis
        return engine.calculate_all_tiers(crop_data)
        
    except ValidationError:
        # Fall back to single tier if multi-tier fails
        try:
            commercial_results = engine.calculate_single_tier(crop_data, tier='commercial')
            return MultiTierResults(tier_results={'commercial': commercial_results})
        except Exception:
            return None
            
    except DataQualityError:
        # Apply additional data cleaning and retry
        cleaned_data = apply_additional_cleaning(crop_data)
        return engine.calculate_all_tiers(cleaned_data)
```

## 📈 **Performance Guidelines**

### **Memory Management**

```python
# For large datasets, use chunked processing
from agririchter.core import MemoryOptimizedProcessor

processor = MemoryOptimizedProcessor(chunk_size=5000)
results = processor.process_large_dataset(large_crop_data, commercial_tier)
```

### **Caching Best Practices**

```python
# Enable caching for repeated analyses
cache = MultiTierCache()
engine = MultiTierEnvelopeEngine()
engine.enable_caching(cache)

# Cache will automatically store and retrieve results
# based on data hash and analysis parameters
```

### **Parallel Processing**

```python
# Use parallel processing for multiple analyses
parallel_calculator = ParallelMultiTierCalculator(n_workers=4)

# Process multiple countries in parallel
countries = ['USA', 'CHN', 'BRA', 'IND']
country_results = parallel_calculator.analyze_countries_parallel(
    countries, crop_data, tier='commercial'
)
```

---

**This API reference provides comprehensive documentation for all public interfaces in the multi-tier envelope system. For additional examples and tutorials, see the user guide and policy guide documentation.**
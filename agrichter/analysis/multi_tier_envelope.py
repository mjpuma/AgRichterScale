"""
Multi-Tier Envelope Analysis Module

Implements simplified two-tier envelope system:
- Comprehensive Tier: All agricultural land (0-100th percentile)
- Commercial Tier: Economically viable agriculture (20-100th percentile)

This module provides policy-relevant envelope bounds with 22-35% width reductions
for commercial agriculture analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from ..core.config import Config
from ..core.envelope_cache import EnvelopeCalculationCache
from ..core.parallel_calculator import ParallelMultiTierCalculator
from ..validation.spam_data_filter import SPAMDataFilter
from .envelope import HPEnvelopeCalculator, EnvelopeData
from .convergence_validator import ConvergenceValidator

logger = logging.getLogger(__name__)


@dataclass
class TierConfiguration:
    """Configuration for a specific productivity tier."""
    
    name: str
    description: str
    yield_percentile_min: float
    yield_percentile_max: float
    policy_applications: List[str]
    target_users: List[str]
    expected_width_reduction: str


# Simplified two-tier configuration
TIER_CONFIGURATIONS = {
    'comprehensive': TierConfiguration(
        name='Comprehensive (All Lands)',
        description='All agricultural land including marginal areas',
        yield_percentile_min=0,
        yield_percentile_max=100,
        policy_applications=['theoretical_bounds', 'academic_research', 'baseline_comparison'],
        target_users=['researchers', 'academics'],
        expected_width_reduction='0% (baseline)'
    ),
    'commercial': TierConfiguration(
        name='Commercial Agriculture',
        description='Economically viable agriculture (excludes bottom 20% yields)',
        yield_percentile_min=20,
        yield_percentile_max=100,
        policy_applications=['government_planning', 'investment_decisions', 'food_security_analysis'],
        target_users=['policy_makers', 'investors', 'planners', 'government_agencies'],
        expected_width_reduction='22-35%'
    )
}


@dataclass
class MultiTierResults:
    """Results from multi-tier envelope analysis."""
    
    tier_results: Dict[str, EnvelopeData]
    width_analysis: Dict[str, Any]
    base_statistics: Dict[str, Any]
    crop_type: str
    calculation_metadata: Dict[str, Any]
    
    def get_tier_envelope(self, tier_name: str) -> Optional[EnvelopeData]:
        """Get envelope data for a specific tier."""
        return self.tier_results.get(tier_name)
    
    def get_width_reduction(self, tier_name: str) -> Optional[float]:
        """Get width reduction percentage for a tier compared to comprehensive."""
        if tier_name == 'comprehensive':
            return 0.0
        return self.width_analysis.get(f'{tier_name}_width_reduction_pct')
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all tiers."""
        summary = {
            'crop_type': self.crop_type,
            'tiers_calculated': list(self.tier_results.keys()),
            'base_statistics': self.base_statistics
        }
        
        for tier_name, envelope_data in self.tier_results.items():
            tier_config = TIER_CONFIGURATIONS.get(tier_name, {})
            summary[f'{tier_name}_tier'] = {
                'description': getattr(tier_config, 'description', 'Unknown'),
                'envelope_points': len(envelope_data.disruption_areas),
                'convergence_validated': envelope_data.convergence_validated,
                'width_reduction': self.get_width_reduction(tier_name)
            }
        
        return summary


class MultiTierEnvelopeEngine:
    """
    Core engine for multi-tier envelope calculations.
    
    Implements simplified two-tier system with comprehensive validation
    and integration with existing SPAM data filtering.
    """
    
    def __init__(self, config: Config, spam_filter: Optional[SPAMDataFilter] = None,
                 enable_caching: bool = True, enable_parallel: bool = True):
        """
        Initialize multi-tier envelope engine.
        
        Args:
            config: Configuration instance
            spam_filter: Optional SPAM data filter (creates default if None)
            enable_caching: Enable result caching for performance
            enable_parallel: Enable parallel processing for multiple tiers
        """
        self.config = config
        self.logger = logging.getLogger('agrichter.multi_tier')
        
        # Initialize SPAM data filter
        self.spam_filter = spam_filter or SPAMDataFilter(preset='standard')
        
        # Initialize base envelope calculator
        self.base_calculator = HPEnvelopeCalculator(config)
        
        # Initialize convergence validator
        self.convergence_validator = ConvergenceValidator(tolerance=1e-6)
        
        # Load tier configurations
        self.tier_configs = TIER_CONFIGURATIONS
        
        # Performance optimizations
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        
        # Initialize cache if enabled
        if self.enable_caching:
            self.cache = EnvelopeCalculationCache()
            self.logger.info("Envelope calculation caching enabled")
        else:
            self.cache = None
        
        # Initialize parallel calculator if enabled
        if self.enable_parallel:
            self.parallel_calculator = ParallelMultiTierCalculator()
            self.logger.info(f"Parallel processing enabled: {self.parallel_calculator}")
        else:
            self.parallel_calculator = None
        
        self.logger.info("Multi-tier envelope engine initialized with 2 tiers")
        self.logger.info(f"Tiers available: {list(self.tier_configs.keys())}")
        self.logger.info(f"Performance optimizations: caching={enable_caching}, parallel={enable_parallel}")
    
    def calculate_multi_tier_envelope(self, 
                                    production_kcal: pd.DataFrame, 
                                    harvest_km2: pd.DataFrame,
                                    tiers: Optional[List[str]] = None,
                                    force_recalculate: bool = False) -> MultiTierResults:
        """
        Calculate envelope bounds for multiple productivity tiers.
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
            tiers: List of tier names to calculate (default: all tiers)
            force_recalculate: Force recalculation even if cached results exist
        
        Returns:
            MultiTierResults with envelope data for each tier
        """
        if tiers is None:
            tiers = list(self.tier_configs.keys())
        
        self.logger.info(f"Starting multi-tier envelope calculation for {self.config.crop_type}")
        self.logger.info(f"Calculating tiers: {tiers}")
        
        # Check cache first if enabled and not forcing recalculation
        if self.enable_caching and not force_recalculate:
            cached_result = self._check_cache_for_multi_tier(production_kcal, harvest_km2, tiers)
            if cached_result is not None:
                self.logger.info("Returning cached multi-tier result")
                return cached_result
        
        # Step 1: Apply base SPAM filtering and prepare data
        base_data = self._prepare_base_data(production_kcal, harvest_km2)
        
        # Step 2: Calculate envelope for each tier (with parallel processing if enabled)
        if self.enable_parallel and len(tiers) > 1:
            self.logger.info("Using parallel processing for tier calculations")
            tier_results = self._calculate_tiers_parallel(production_kcal, harvest_km2, tiers)
        else:
            self.logger.info("Using sequential processing for tier calculations")
            tier_results = self._calculate_tiers_sequential(base_data, tiers)
        
        # Step 3: Calculate width analysis
        width_analysis = self._calculate_width_analysis(tier_results)
        
        # Step 4: Generate base statistics
        base_statistics = self._calculate_base_statistics(base_data)
        
        # Step 5: Create results object
        results = MultiTierResults(
            tier_results=tier_results,
            width_analysis=width_analysis,
            base_statistics=base_statistics,
            crop_type=self.config.crop_type,
            calculation_metadata={
                'timestamp': pd.Timestamp.now().isoformat(),
                'tiers_calculated': tiers,
                'spam_filter_applied': True,
                'validation_passed': all(env.convergence_validated for env in tier_results.values()),
                'used_caching': self.enable_caching,
                'used_parallel': self.enable_parallel and len(tiers) > 1
            }
        )
        
        # Cache result if caching is enabled
        if self.enable_caching:
            self._cache_multi_tier_result(production_kcal, harvest_km2, tiers, results)
        
        self.logger.info("Multi-tier envelope calculation completed successfully")
        return results
    
    def _prepare_base_data(self, production_kcal: pd.DataFrame, 
                          harvest_km2: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare base data with SPAM filtering and area capping.
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
        
        Returns:
            Dictionary with filtered production, harvest, and yield data
        """
        self.logger.info("Preparing base data with SPAM filtering")
        
        # Get crop columns based on config (reuse existing logic from HPEnvelopeCalculator)
        crop_columns = self._get_crop_columns(production_kcal)
        
        # Calculate total production and harvest for selected crops
        total_production, total_harvest = self._aggregate_crop_data(
            production_kcal, harvest_km2, crop_columns
        )
        
        # Apply SPAM data filtering
        filter_mask, filter_stats = self.spam_filter.filter_crop_data(
            total_production, total_harvest * 100,  # Convert km² to hectares for filter
            crop_name=self.config.crop_type
        )
        
        # Filter data
        filtered_production = total_production[filter_mask]
        filtered_harvest = total_harvest[filter_mask]
        
        # Calculate yields
        filtered_yields = np.divide(
            filtered_production, 
            filtered_harvest, 
            out=np.zeros_like(filtered_production), 
            where=(filtered_harvest > 0)
        )
        
        self.logger.info(f"Base data prepared: {len(filtered_production)} valid cells")
        self.logger.info(f"SPAM filtering: {filter_stats['retention_rate']:.1f}% retention rate")
        
        return {
            'production': filtered_production,
            'harvest': filtered_harvest,
            'yields': filtered_yields,
            'filter_stats': filter_stats
        }
    
    def _get_crop_columns(self, production_data: pd.DataFrame) -> List[str]:
        """Get crop columns based on configuration (reuse existing logic)."""
        crop_columns = [col for col in production_data.columns if col.endswith('_A')]
        
        if not crop_columns:
            raise ValueError("No crop columns found in production data")
        
        # Filter to selected crop types based on config
        if self.config.crop_type == 'wheat':
            crop_columns = [col for col in crop_columns if 'WHEA' in col.upper()]
        elif self.config.crop_type == 'rice':
            crop_columns = [col for col in crop_columns if 'RICE' in col.upper()]
        elif self.config.crop_type == 'maize':
            crop_columns = [col for col in crop_columns if 'MAIZ' in col.upper()]
        elif self.config.crop_type == 'allgrain':
            # Include grain crops
            grain_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'PMIL_A', 'SMIL_A', 'SORG_A', 'OCER_A']
            crop_columns = [col for col in crop_columns if col in grain_crops]
        
        if not crop_columns:
            raise ValueError(f"No crop columns found for crop type: {self.config.crop_type}")
        
        return crop_columns
    
    def _aggregate_crop_data(self, production_kcal: pd.DataFrame, 
                           harvest_km2: pd.DataFrame, 
                           crop_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate production and harvest data across selected crops."""
        from ..core.constants import CALORIC_CONTENT, GRAMS_PER_METRIC_TON, HECTARES_TO_KM2
        
        # Map crop columns to caloric content
        crop_caloric_map = {
            'BARL_A': CALORIC_CONTENT['Barley'],
            'MAIZ_A': CALORIC_CONTENT['Corn'],
            'OCER_A': CALORIC_CONTENT['MixedGrain'],
            'PMIL_A': CALORIC_CONTENT['Millet'],
            'RICE_A': CALORIC_CONTENT['Rice'],
            'SMIL_A': CALORIC_CONTENT['Millet'],
            'SORG_A': CALORIC_CONTENT['Sorghum'],
            'WHEA_A': CALORIC_CONTENT['Wheat'],
        }
        
        # Convert production from metric tons to kcal for each crop
        total_production = np.zeros(len(production_kcal))
        for crop_col in crop_columns:
            crop_mt = production_kcal[crop_col].values
            caloric_content = crop_caloric_map.get(crop_col, CALORIC_CONTENT['MixedGrain'])
            crop_kcal = crop_mt * GRAMS_PER_METRIC_TON * caloric_content
            total_production += crop_kcal
        
        # Sum harvest area across selected crops and convert from hectares to km²
        total_harvest_ha = harvest_km2[crop_columns].sum(axis=1).values
        total_harvest = total_harvest_ha * HECTARES_TO_KM2
        
        return total_production, total_harvest
    
    def _calculate_tier_envelope(self, base_data: Dict[str, np.ndarray], 
                               tier_name: str) -> EnvelopeData:
        """
        Calculate envelope bounds for a specific tier.
        
        Args:
            base_data: Filtered base data
            tier_name: Name of tier to calculate
        
        Returns:
            EnvelopeData for the specified tier
        """
        tier_config = self.tier_configs[tier_name]
        
        # Apply tier-specific productivity filtering
        tier_data = self._apply_tier_filtering(base_data, tier_config)
        
        # Calculate envelope bounds directly using the core algorithm
        envelope_dict = self._calculate_envelope_bounds_direct(tier_data)
        
        # Create EnvelopeData object
        envelope_data = self.base_calculator.create_envelope_data_object(envelope_dict)
        envelope_data.crop_type = f"{self.config.crop_type}_{tier_name}"
        
        self.logger.info(f"{tier_name} tier: {len(envelope_data.disruption_areas)} envelope points")
        self.logger.info(f"{tier_name} tier: Convergence validated = {envelope_data.convergence_validated}")
        
        return envelope_data
    
    def _apply_tier_filtering(self, base_data: Dict[str, np.ndarray], 
                            tier_config: TierConfiguration) -> Dict[str, np.ndarray]:
        """
        Apply tier-specific productivity filtering.
        
        Args:
            base_data: Base filtered data
            tier_config: Configuration for the tier
        
        Returns:
            Tier-filtered data
        """
        yields = base_data['yields']
        
        # Calculate yield percentiles
        yield_min_threshold = np.percentile(yields, tier_config.yield_percentile_min)
        yield_max_threshold = np.percentile(yields, tier_config.yield_percentile_max)
        
        # Apply tier filtering
        tier_mask = (yields >= yield_min_threshold) & (yields <= yield_max_threshold)
        
        tier_data = {
            'production': base_data['production'][tier_mask],
            'harvest': base_data['harvest'][tier_mask],
            'yields': yields[tier_mask]
        }
        
        cells_retained = len(tier_data['production'])
        retention_rate = (cells_retained / len(base_data['production'])) * 100
        
        self.logger.info(f"{tier_config.name}: {cells_retained} cells retained ({retention_rate:.1f}%)")
        self.logger.info(f"{tier_config.name}: Yield range {yield_min_threshold:.2e} - {yield_max_threshold:.2e}")
        
        return tier_data
    
    def _calculate_envelope_bounds_direct(self, tier_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate envelope bounds directly from tier data.
        
        This method implements the core envelope algorithm without relying on
        the existing HPEnvelopeCalculator to avoid data format issues.
        
        Args:
            tier_data: Tier-filtered data with production, harvest, yields
        
        Returns:
            Dictionary with envelope bounds data
        """
        production = tier_data['production']
        harvest = tier_data['harvest']
        yields = tier_data['yields']
        
        # Create H-P matrix: [harvest_area, production, yield]
        hp_matrix = np.column_stack([harvest, production, yields])
        
        # Sort by yield (productivity) - ascending order
        sort_indices = np.argsort(hp_matrix[:, 2])
        hp_matrix_sorted = hp_matrix[sort_indices]
        
        # Calculate total production and harvest for convergence
        total_production = production.sum()
        total_harvest = harvest.sum()
        
        # Get adaptive disruption range
        disruption_ranges = self._get_adaptive_disruption_range_direct(hp_matrix_sorted, total_harvest)
        
        # Calculate cumulative sums for both directions
        cumsum_small_large = np.cumsum(hp_matrix_sorted, axis=0)  # Lower bound (least productive first)
        cumsum_large_small = np.cumsum(np.flipud(hp_matrix_sorted), axis=0)  # Upper bound (most productive first)
        
        n_points = len(disruption_ranges)
        lower_bound_harvest = np.full(n_points, np.nan)
        lower_bound_production = np.full(n_points, np.nan)
        upper_bound_harvest = np.full(n_points, np.nan)
        upper_bound_production = np.full(n_points, np.nan)
        
        for i, target_area in enumerate(disruption_ranges):
            # Lower bound (least productive first)
            indices_lower = np.where(cumsum_small_large[:, 0] > target_area)[0]
            if len(indices_lower) > 0:
                idx = indices_lower[0]
                lower_bound_harvest[i] = cumsum_small_large[idx, 0]
                lower_bound_production[i] = cumsum_small_large[idx, 1]
            
            # Upper bound (most productive first)
            indices_upper = np.where(cumsum_large_small[:, 0] > target_area)[0]
            if len(indices_upper) > 0:
                idx = indices_upper[0]
                upper_bound_harvest[i] = cumsum_large_small[idx, 0]
                upper_bound_production[i] = cumsum_large_small[idx, 1]
        
        # Remove NaN values
        valid_indices = (~np.isnan(lower_bound_harvest) & ~np.isnan(upper_bound_harvest) & 
                        ~np.isnan(lower_bound_production) & ~np.isnan(upper_bound_production))
        
        envelope_data = {
            'disruption_areas': disruption_ranges[valid_indices],
            'lower_bound_harvest': lower_bound_harvest[valid_indices],
            'lower_bound_production': lower_bound_production[valid_indices],
            'upper_bound_harvest': upper_bound_harvest[valid_indices],
            'upper_bound_production': upper_bound_production[valid_indices],
            'convergence_point': (total_harvest, total_production),
            'convergence_validated': True,  # Will be validated later
            'mathematical_properties': {},  # Will be filled by validator
            'convergence_statistics': {}    # Will be filled by validator
        }
        
        # Add explicit convergence point if needed
        envelope_data = self._add_convergence_point_direct(envelope_data, total_production, total_harvest)
        
        # Validate and enforce mathematical correctness
        validation_result = self.convergence_validator.validate_mathematical_properties(
            envelope_data, total_production, total_harvest
        )
        
        if not validation_result.is_valid:
            self.logger.warning("Envelope failed mathematical validation, attempting correction")
            envelope_data = self.convergence_validator.enforce_convergence(
                envelope_data, total_production, total_harvest
            )
            
            # Re-validate after correction
            validation_result = self.convergence_validator.validate_mathematical_properties(
                envelope_data, total_production, total_harvest
            )
        
        # Add validation metadata
        envelope_data['convergence_validated'] = validation_result.is_valid
        envelope_data['mathematical_properties'] = validation_result.properties
        envelope_data['convergence_statistics'] = validation_result.statistics
        
        return envelope_data
    
    def _get_adaptive_disruption_range_direct(self, hp_matrix_sorted: np.ndarray, 
                                            total_harvest: float) -> np.ndarray:
        """
        Get adaptive disruption range for direct calculation.
        
        Args:
            hp_matrix_sorted: Sorted H-P matrix
            total_harvest: Total harvest area
        
        Returns:
            Adaptive disruption range array
        """
        # Use a simple but effective range for testing
        # Start with small areas and progress to total harvest
        base_points = [1, 10, 50, 100, 200, 400, 600, 800, 1000]
        
        # Add logarithmic progression to fill the range
        if total_harvest > 1000:
            n_additional = 20
            log_min = np.log10(1000)
            log_max = np.log10(total_harvest * 0.95)
            if log_max > log_min:
                log_points = np.linspace(log_min, log_max, n_additional)
                additional_points = 10 ** log_points
                adaptive_range = base_points + additional_points.tolist()
            else:
                adaptive_range = base_points
        else:
            # For smaller datasets, use finer resolution
            adaptive_range = base_points + list(np.linspace(100, total_harvest * 0.95, 15))
        
        # Remove duplicates, sort, and ensure we don't exceed total harvest
        adaptive_range = sorted(list(set(adaptive_range)))
        adaptive_range = [r for r in adaptive_range if r <= total_harvest]
        
        # Always add the total harvest area as the final point
        if total_harvest not in adaptive_range:
            adaptive_range.append(total_harvest)
        
        return np.array(adaptive_range)
    
    def _add_convergence_point_direct(self, envelope_data: Dict[str, np.ndarray], 
                                    total_production: float, 
                                    total_harvest: float) -> Dict[str, np.ndarray]:
        """
        Add explicit convergence point to envelope data.
        
        Args:
            envelope_data: Current envelope data
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
        
        Returns:
            Envelope data with explicit convergence point
        """
        lower_harvest = envelope_data['lower_bound_harvest']
        max_harvest = np.max(lower_harvest) if len(lower_harvest) > 0 else 0
        
        # Check if we already have the convergence point (within 1% tolerance)
        if max_harvest >= total_harvest * 0.99:
            # Find the closest point to total harvest
            closest_idx = np.argmin(np.abs(lower_harvest - total_harvest))
            
            # Update the closest point to be exactly the convergence point
            envelope_data['lower_bound_harvest'][closest_idx] = total_harvest
            envelope_data['lower_bound_production'][closest_idx] = total_production
            envelope_data['upper_bound_harvest'][closest_idx] = total_harvest
            envelope_data['upper_bound_production'][closest_idx] = total_production
            envelope_data['disruption_areas'][closest_idx] = total_harvest
        else:
            # Add explicit convergence point
            envelope_data['lower_bound_harvest'] = np.append(
                envelope_data['lower_bound_harvest'], total_harvest
            )
            envelope_data['lower_bound_production'] = np.append(
                envelope_data['lower_bound_production'], total_production
            )
            envelope_data['upper_bound_harvest'] = np.append(
                envelope_data['upper_bound_harvest'], total_harvest
            )
            envelope_data['upper_bound_production'] = np.append(
                envelope_data['upper_bound_production'], total_production
            )
            envelope_data['disruption_areas'] = np.append(
                envelope_data['disruption_areas'], total_harvest
            )
        
        return envelope_data
    
    def _calculate_width_analysis(self, tier_results: Dict[str, EnvelopeData]) -> Dict[str, Any]:
        """
        Calculate width reduction analysis between tiers.
        
        Args:
            tier_results: Envelope results for each tier
        
        Returns:
            Width analysis dictionary
        """
        width_analysis = {}
        
        if 'comprehensive' not in tier_results:
            self.logger.warning("Comprehensive tier not found, cannot calculate width reductions")
            return width_analysis
        
        comprehensive_envelope = tier_results['comprehensive']
        
        # Calculate representative width for comprehensive tier (baseline)
        comprehensive_width = self._calculate_representative_width(comprehensive_envelope)
        width_analysis['comprehensive_width'] = comprehensive_width
        
        # Calculate width reductions for other tiers
        for tier_name, tier_envelope in tier_results.items():
            if tier_name == 'comprehensive':
                continue
            
            tier_width = self._calculate_representative_width(tier_envelope)
            width_reduction = ((comprehensive_width - tier_width) / comprehensive_width) * 100
            
            width_analysis[f'{tier_name}_width'] = tier_width
            width_analysis[f'{tier_name}_width_reduction_pct'] = width_reduction
            
            self.logger.info(f"{tier_name} tier: {width_reduction:.1f}% width reduction")
        
        return width_analysis
    
    def _calculate_representative_width(self, envelope_data: EnvelopeData) -> float:
        """
        Calculate representative envelope width for comparison.
        
        Uses median production width as representative measure.
        """
        production_widths = envelope_data.upper_bound_production - envelope_data.lower_bound_production
        return float(np.median(production_widths))
    
    def _calculate_base_statistics(self, base_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate base statistics for the filtered data."""
        return {
            'total_cells': len(base_data['production']),
            'total_production': float(base_data['production'].sum()),
            'total_harvest': float(base_data['harvest'].sum()),
            'mean_yield': float(base_data['yields'].mean()),
            'median_yield': float(np.median(base_data['yields'])),
            'yield_std': float(base_data['yields'].std()),
            'spam_filter_stats': base_data.get('filter_stats', {})
        }
    
    def get_tier_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available tiers."""
        tier_info = {}
        for tier_name, tier_config in self.tier_configs.items():
            tier_info[tier_name] = {
                'name': tier_config.name,
                'description': tier_config.description,
                'yield_percentile_range': f"{tier_config.yield_percentile_min}-{tier_config.yield_percentile_max}%",
                'policy_applications': tier_config.policy_applications,
                'target_users': tier_config.target_users,
                'expected_width_reduction': tier_config.expected_width_reduction
            }
        return tier_info
    
    def validate_multi_tier_results(self, results: MultiTierResults) -> Dict[str, Any]:
        """
        Validate multi-tier results for mathematical correctness.
        
        Args:
            results: Multi-tier results to validate
        
        Returns:
            Validation report
        """
        validation_report = {
            'overall_valid': True,
            'tier_validations': {},
            'width_reduction_validation': {},
            'issues': []
        }
        
        # Validate each tier individually
        for tier_name, envelope_data in results.tier_results.items():
            tier_valid = envelope_data.is_mathematically_valid()
            validation_report['tier_validations'][tier_name] = tier_valid
            
            if not tier_valid:
                validation_report['overall_valid'] = False
                validation_report['issues'].append(f"{tier_name} tier failed mathematical validation")
        
        # Validate width reductions
        if 'commercial' in results.tier_results and 'comprehensive' in results.tier_results:
            commercial_reduction = results.get_width_reduction('commercial')
            if commercial_reduction is not None:
                # Check if width reduction is in expected range (22-35%)
                in_range = 22 <= commercial_reduction <= 35
                validation_report['width_reduction_validation']['commercial'] = {
                    'reduction_pct': commercial_reduction,
                    'in_expected_range': in_range,
                    'expected_range': '22-35%'
                }
                
                if not in_range:
                    validation_report['issues'].append(
                        f"Commercial tier width reduction ({commercial_reduction:.1f}%) outside expected range (22-35%)"
                    )
        
        return validation_report
    
    # Performance optimization methods
    
    def _check_cache_for_multi_tier(self, production_df: pd.DataFrame, 
                                   harvest_df: pd.DataFrame, 
                                   tiers: List[str]) -> Optional[MultiTierResults]:
        """Check cache for existing multi-tier calculation."""
        if not self.cache:
            return None
        
        try:
            additional_params = {
                'tiers': '_'.join(sorted(tiers)),
                'crop_type': self.config.crop_type
            }
            
            cached_result = self.cache.get_cached_result(
                production_df, harvest_df, 'multi_tier', 
                additional_params=additional_params
            )
            
            if cached_result:
                self.logger.info("Found cached multi-tier result")
                return cached_result
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
        
        return None
    
    def _cache_multi_tier_result(self, production_df: pd.DataFrame, 
                                harvest_df: pd.DataFrame, 
                                tiers: List[str], 
                                results: MultiTierResults):
        """Cache multi-tier calculation result."""
        if not self.cache:
            return
        
        try:
            additional_params = {
                'tiers': '_'.join(sorted(tiers)),
                'crop_type': self.config.crop_type
            }
            
            self.cache.cache_result(
                results, production_df, harvest_df, 'multi_tier',
                additional_params=additional_params
            )
            
            self.logger.debug("Cached multi-tier result")
            
        except Exception as e:
            self.logger.warning(f"Caching failed: {e}")
    
    def _calculate_tiers_parallel(self, production_df: pd.DataFrame, 
                                 harvest_df: pd.DataFrame, 
                                 tiers: List[str]) -> Dict[str, EnvelopeData]:
        """Calculate tiers using parallel processing."""
        if not self.parallel_calculator:
            # Fallback to sequential if parallel not available
            base_data = self._prepare_base_data(production_df, harvest_df)
            return self._calculate_tiers_sequential(base_data, tiers)
        
        try:
            # Use parallel calculator
            parallel_results = self.parallel_calculator.calculate_all_tiers_parallel(
                production_df, harvest_df, HPEnvelopeCalculator, self.config, tiers
            )
            
            # Convert results to EnvelopeData objects
            tier_results = {}
            for tier_name, envelope_dict in parallel_results.tier_results.items():
                envelope_data = self.base_calculator.create_envelope_data_object(envelope_dict)
                envelope_data.crop_type = f"{self.config.crop_type}_{tier_name}"
                tier_results[tier_name] = envelope_data
            
            return tier_results
            
        except Exception as e:
            self.logger.warning(f"Parallel calculation failed: {e}, falling back to sequential")
            # Fallback to sequential processing
            base_data = self._prepare_base_data(production_df, harvest_df)
            return self._calculate_tiers_sequential(base_data, tiers)
    
    def _calculate_tiers_sequential(self, base_data: Dict[str, np.ndarray], 
                                   tiers: List[str]) -> Dict[str, EnvelopeData]:
        """Calculate tiers using sequential processing."""
        tier_results = {}
        
        for tier_name in tiers:
            if tier_name not in self.tier_configs:
                self.logger.warning(f"Unknown tier: {tier_name}, skipping")
                continue
            
            # Check individual tier cache first
            if self.enable_caching:
                cached_tier = self._check_cache_for_single_tier(base_data, tier_name)
                if cached_tier:
                    tier_results[tier_name] = cached_tier
                    continue
            
            self.logger.info(f"Calculating {tier_name} tier envelope")
            tier_envelope = self._calculate_tier_envelope(base_data, tier_name)
            tier_results[tier_name] = tier_envelope
            
            # Cache individual tier result
            if self.enable_caching:
                self._cache_single_tier_result(base_data, tier_name, tier_envelope)
        
        return tier_results
    
    def _check_cache_for_single_tier(self, base_data: Dict[str, np.ndarray], 
                                    tier_name: str) -> Optional[EnvelopeData]:
        """Check cache for single tier calculation."""
        # For single tier caching, we'd need to create a DataFrame from base_data
        # This is a simplified implementation
        return None
    
    def _cache_single_tier_result(self, base_data: Dict[str, np.ndarray], 
                                 tier_name: str, envelope_data: EnvelopeData):
        """Cache single tier calculation result."""
        # For single tier caching, we'd need to create a DataFrame from base_data
        # This is a simplified implementation
        pass
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the engine."""
        stats = {
            'caching_enabled': self.enable_caching,
            'parallel_enabled': self.enable_parallel,
            'available_tiers': list(self.tier_configs.keys())
        }
        
        if self.cache:
            stats['cache_statistics'] = self.cache.get_cache_statistics()
        
        if self.parallel_calculator:
            stats['parallel_workers'] = self.parallel_calculator.n_workers
            stats['parallel_mode'] = 'processes' if self.parallel_calculator.use_processes else 'threads'
        
        return stats
    
    def clear_cache(self):
        """Clear all cached results."""
        if self.cache:
            self.cache.clear_cache()
            self.logger.info("Cache cleared")
    
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        if self.parallel_calculator:
            self.parallel_calculator.shutdown()
            self.logger.info("Parallel calculator shutdown")


# Convenience functions for common use cases
def calculate_crop_multi_tier(crop_type: str, 
                            production_data: pd.DataFrame, 
                            harvest_data: pd.DataFrame,
                            tiers: Optional[List[str]] = None) -> MultiTierResults:
    """
    Convenience function for calculating multi-tier envelopes for a specific crop.
    
    Args:
        crop_type: Type of crop ('wheat', 'rice', 'maize', 'allgrain')
        production_data: Production data DataFrame
        harvest_data: Harvest area data DataFrame
        tiers: List of tiers to calculate (default: all)
    
    Returns:
        MultiTierResults
    """
    # Create temporary config
    config = Config(crop_type=crop_type)
    
    # Initialize engine
    engine = MultiTierEnvelopeEngine(config)
    
    # Calculate multi-tier envelope
    return engine.calculate_multi_tier_envelope(production_data, harvest_data, tiers)


def get_available_tiers() -> Dict[str, str]:
    """Get available tier names and descriptions."""
    return {name: config.description for name, config in TIER_CONFIGURATIONS.items()}
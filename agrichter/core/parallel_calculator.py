"""
Parallel Multi-Tier Calculator

Provides parallel processing capabilities for multi-tier envelope calculations
to improve performance on multi-core systems.
"""

import logging
import time
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from multiprocessing import cpu_count

from .config import Config

logger = logging.getLogger(__name__)


def _calculate_single_tier_worker(production_df: pd.DataFrame, harvest_df: pd.DataFrame,
                                 tier_name: str, tier_config: Any, 
                                 envelope_calculator_class: type,
                                 config_dict: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Worker function for parallel tier calculation.
    
    This function runs in a separate process, so it needs to be at module level
    and receive serializable arguments.
    
    Args:
        production_df: Production data
        harvest_df: Harvest area data
        tier_name: Name of the tier
        tier_config: Tier configuration
        envelope_calculator_class: Class for envelope calculation
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (tier_name, envelope_result)
    """
    try:
        # Recreate config object in worker process
        config = Config(**config_dict)
        
        # Create envelope calculator
        calculator = envelope_calculator_class(config)
        
        # Apply tier filtering
        from ..validation.spam_data_filter import SPAMDataFilter
        spam_filter = SPAMDataFilter(config)
        
        # Filter data based on tier configuration
        if tier_config.yield_percentile_min > 0 or tier_config.yield_percentile_max < 100:
            # Calculate yields for filtering
            yields = production_df.values / np.maximum(harvest_df.values, 1e-10)
            
            # Calculate percentile thresholds
            valid_yields = yields[np.isfinite(yields) & (yields > 0)]
            if len(valid_yields) > 0:
                min_threshold = np.percentile(valid_yields, tier_config.yield_percentile_min)
                max_threshold = np.percentile(valid_yields, tier_config.yield_percentile_max)
                
                # Create mask for tier filtering
                tier_mask = (yields >= min_threshold) & (yields <= max_threshold)
                
                # Apply mask to data
                production_filtered = production_df.copy()
                harvest_filtered = harvest_df.copy()
                
                production_filtered.values[~tier_mask] = 0
                harvest_filtered.values[~tier_mask] = 0
            else:
                production_filtered = production_df
                harvest_filtered = harvest_df
        else:
            production_filtered = production_df
            harvest_filtered = harvest_df
        
        # Calculate envelope for this tier
        envelope_result = calculator.calculate_hp_envelope(
            production_filtered, harvest_filtered
        )
        
        logger.debug(f"Completed tier calculation: {tier_name}")
        return tier_name, envelope_result
        
    except Exception as e:
        logger.error(f"Error in worker for tier {tier_name}: {e}")
        raise


class ParallelMultiTierCalculator:
    """
    Parallel processing for multi-tier calculations.
    
    Provides parallel execution of tier calculations to improve performance
    on multi-core systems.
    """
    
    def __init__(self, n_workers: Optional[int] = None, 
                 use_processes: bool = True):
        """
        Initialize parallel calculator.
        
        Args:
            n_workers: Number of worker processes/threads. If None, uses CPU count
            use_processes: If True, use processes; if False, use threads
        """
        if n_workers is None:
            n_workers = min(4, cpu_count())  # Limit to 4 to avoid memory issues
        
        self.n_workers = n_workers
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=n_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=n_workers)
        
        logger.info(f"ParallelMultiTierCalculator initialized with {n_workers} "
                   f"{'processes' if use_processes else 'threads'}")
    
    def calculate_all_tiers_parallel(self, production_df: pd.DataFrame, 
                                   harvest_df: pd.DataFrame,
                                   envelope_calculator_class: type,
                                   config: Config,
                                   tiers: Optional[List[str]] = None) -> Any:
        """
        Calculate all tiers in parallel for improved performance.
        
        Args:
            production_df: Production data
            harvest_df: Harvest area data
            envelope_calculator_class: Class for envelope calculation
            config: Configuration object
            tiers: List of tier names to calculate. If None, calculates all tiers
            
        Returns:
            MultiTierResults with all tier calculations
        """
        if tiers is None:
            tiers = list(TIER_CONFIGURATIONS.keys())
        
        start_time = time.time()
        
        # Import here to avoid circular imports
        from ..analysis.multi_tier_envelope import TIER_CONFIGURATIONS
        
        # Prepare configuration for serialization (for process-based execution)
        config_dict = {
            'crop_type': config.crop_type,
            'root_dir': str(config.root_dir),
            'spam_version': config.spam_version
        }
        
        # Submit tier calculations to worker processes/threads
        future_to_tier = {}
        
        for tier_name in tiers:
            if tier_name not in TIER_CONFIGURATIONS:
                logger.warning(f"Unknown tier: {tier_name}, skipping")
                continue
            
            tier_config = TIER_CONFIGURATIONS[tier_name]
            
            if self.use_processes:
                # For processes, use the worker function
                future = self.executor.submit(
                    _calculate_single_tier_worker,
                    production_df, harvest_df, tier_name, tier_config,
                    envelope_calculator_class, config_dict
                )
            else:
                # For threads, use lambda (can access local variables)
                future = self.executor.submit(
                    self._calculate_single_tier_thread,
                    production_df, harvest_df, tier_name, tier_config,
                    envelope_calculator_class, config
                )
            
            future_to_tier[future] = tier_name
        
        # Collect results as they complete
        tier_results = {}
        completed_count = 0
        
        for future in as_completed(future_to_tier):
            tier_name = future_to_tier[future]
            completed_count += 1
            
            try:
                result_tier_name, envelope_result = future.result()
                tier_results[result_tier_name] = envelope_result
                
                logger.info(f"Completed tier {result_tier_name} "
                           f"({completed_count}/{len(future_to_tier)})")
                
            except Exception as e:
                logger.error(f"Tier {tier_name} failed: {e}")
                # Continue with other tiers
        
        calculation_time = time.time() - start_time
        
        # Calculate width analysis
        width_analysis = self._calculate_width_analysis(tier_results)
        
        # Calculate base statistics
        base_statistics = self._calculate_base_statistics(production_df, harvest_df)
        
        # Import here to avoid circular imports
        from ..analysis.multi_tier_envelope import MultiTierResults
        
        # Create results object
        results = MultiTierResults(
            tier_results=tier_results,
            width_analysis=width_analysis,
            base_statistics=base_statistics,
            crop_type=config.crop_type,
            calculation_metadata={
                'parallel_execution': True,
                'n_workers': self.n_workers,
                'execution_mode': 'processes' if self.use_processes else 'threads',
                'calculation_time_seconds': calculation_time,
                'tiers_requested': tiers,
                'tiers_completed': list(tier_results.keys())
            }
        )
        
        logger.info(f"Parallel calculation completed in {calculation_time:.2f}s "
                   f"({len(tier_results)}/{len(tiers)} tiers successful)")
        
        return results
    
    def _calculate_single_tier_thread(self, production_df: pd.DataFrame, 
                                    harvest_df: pd.DataFrame,
                                    tier_name: str, tier_config: Any,
                                    envelope_calculator_class: type,
                                    config: Config) -> Tuple[str, Any]:
        """
        Calculate single tier in thread (can access shared memory).
        
        Args:
            production_df: Production data
            harvest_df: Harvest area data
            tier_name: Name of the tier
            tier_config: Tier configuration
            envelope_calculator_class: Class for envelope calculation
            config: Configuration object
            
        Returns:
            Tuple of (tier_name, envelope_result)
        """
        try:
            # Create envelope calculator
            calculator = envelope_calculator_class(config)
            
            # Apply tier filtering (same logic as worker function)
            if tier_config.yield_percentile_min > 0 or tier_config.yield_percentile_max < 100:
                yields = production_df.values / np.maximum(harvest_df.values, 1e-10)
                
                valid_yields = yields[np.isfinite(yields) & (yields > 0)]
                if len(valid_yields) > 0:
                    min_threshold = np.percentile(valid_yields, tier_config.yield_percentile_min)
                    max_threshold = np.percentile(valid_yields, tier_config.yield_percentile_max)
                    
                    tier_mask = (yields >= min_threshold) & (yields <= max_threshold)
                    
                    production_filtered = production_df.copy()
                    harvest_filtered = harvest_df.copy()
                    
                    production_filtered.values[~tier_mask] = 0
                    harvest_filtered.values[~tier_mask] = 0
                else:
                    production_filtered = production_df
                    harvest_filtered = harvest_df
            else:
                production_filtered = production_df
                harvest_filtered = harvest_df
            
            # Calculate envelope for this tier
            envelope_result = calculator.calculate_hp_envelope(
                production_filtered, harvest_filtered
            )
            
            return tier_name, envelope_result
            
        except Exception as e:
            logger.error(f"Error in thread for tier {tier_name}: {e}")
            raise
    
    def _calculate_width_analysis(self, tier_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate width reduction analysis between tiers.
        
        Args:
            tier_results: Dictionary of tier results
            
        Returns:
            Width analysis dictionary
        """
        width_analysis = {}
        
        if 'comprehensive' in tier_results:
            comprehensive_result = tier_results['comprehensive']
            
            # Calculate representative width for comprehensive tier
            if hasattr(comprehensive_result, 'disruption_areas') and hasattr(comprehensive_result, 'upper_envelope'):
                comp_areas = comprehensive_result.disruption_areas
                comp_upper = comprehensive_result.upper_envelope
                comp_lower = comprehensive_result.lower_envelope
                
                # Calculate width at middle disruption level
                if len(comp_areas) > 0:
                    mid_idx = len(comp_areas) // 2
                    comp_width = comp_upper[mid_idx] - comp_lower[mid_idx]
                    
                    width_analysis['comprehensive_width'] = comp_width
                    
                    # Calculate width reductions for other tiers
                    for tier_name, tier_result in tier_results.items():
                        if tier_name != 'comprehensive':
                            if hasattr(tier_result, 'upper_envelope') and len(tier_result.upper_envelope) > mid_idx:
                                tier_width = tier_result.upper_envelope[mid_idx] - tier_result.lower_envelope[mid_idx]
                                
                                if comp_width > 0:
                                    reduction_pct = ((comp_width - tier_width) / comp_width) * 100
                                    width_analysis[f'{tier_name}_width_reduction_pct'] = reduction_pct
                                    width_analysis[f'{tier_name}_width'] = tier_width
        
        return width_analysis
    
    def _calculate_base_statistics(self, production_df: pd.DataFrame, 
                                 harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate base statistics for the dataset.
        
        Args:
            production_df: Production data
            harvest_df: Harvest area data
            
        Returns:
            Base statistics dictionary
        """
        return {
            'total_grid_cells': len(production_df),
            'total_production': production_df.sum().sum(),
            'total_harvest_area': harvest_df.sum().sum(),
            'data_shape': production_df.shape,
            'non_zero_cells': (production_df > 0).sum().sum()
        }
    
    def calculate_countries_parallel(self, country_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                                   envelope_calculator_class: type,
                                   config: Config,
                                   tier: str = 'commercial') -> Dict[str, Any]:
        """
        Calculate envelope for multiple countries in parallel.
        
        Args:
            country_data: Dictionary mapping country codes to (production_df, harvest_df)
            envelope_calculator_class: Class for envelope calculation
            config: Configuration object
            tier: Tier to calculate
            
        Returns:
            Dictionary mapping country codes to envelope results
        """
        start_time = time.time()
        
        # Submit country calculations to workers
        future_to_country = {}
        
        for country_code, (production_df, harvest_df) in country_data.items():
            if self.use_processes:
                config_dict = {
                    'crop_type': config.crop_type,
                    'root_dir': str(config.root_dir),
                    'spam_version': config.spam_version
                }
                
                # Import here to avoid circular imports
                from ..analysis.multi_tier_envelope import TIER_CONFIGURATIONS
                tier_config = TIER_CONFIGURATIONS[tier]
                
                future = self.executor.submit(
                    _calculate_single_tier_worker,
                    production_df, harvest_df, tier, tier_config,
                    envelope_calculator_class, config_dict
                )
            else:
                future = self.executor.submit(
                    self._calculate_single_tier_thread,
                    production_df, harvest_df, tier, tier_config,
                    envelope_calculator_class, config
                )
            
            future_to_country[future] = country_code
        
        # Collect results
        country_results = {}
        completed_count = 0
        
        for future in as_completed(future_to_country):
            country_code = future_to_country[future]
            completed_count += 1
            
            try:
                _, envelope_result = future.result()
                country_results[country_code] = envelope_result
                
                logger.info(f"Completed country {country_code} "
                           f"({completed_count}/{len(future_to_country)})")
                
            except Exception as e:
                logger.error(f"Country {country_code} failed: {e}")
        
        calculation_time = time.time() - start_time
        
        logger.info(f"Parallel country calculation completed in {calculation_time:.2f}s "
                   f"({len(country_results)}/{len(country_data)} countries successful)")
        
        return country_results
    
    def shutdown(self):
        """Shutdown the executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.info("ParallelMultiTierCalculator shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __repr__(self) -> str:
        return (
            f"ParallelMultiTierCalculator("
            f"workers={self.n_workers}, "
            f"mode={'processes' if self.use_processes else 'threads'})"
        )
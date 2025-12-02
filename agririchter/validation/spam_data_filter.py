"""
SPAM Data Filtering Module

Provides consistent data filtering for SPAM 2020 data across all AgriRichter analyses.
Addresses data quality issues including extreme yield outliers and tiny harvest areas.

See docs/DATA_FILTERING_METHODOLOGY.md for complete methodology documentation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class SPAMDataFilter:
    """
    Consistent data filtering for SPAM 2020 data.
    
    This class implements the standardized filtering methodology to address
    data quality issues in SPAM 2020 that can lead to unrealistic yield
    calculations and poor envelope convergence.
    """
    
    # Default filtering thresholds
    MIN_HARVEST_AREA_KM2 = 0.01  # 1 hectare minimum
    YIELD_PERCENTILE_MULTIPLIER = 2.0  # 2x 99th percentile
    
    # Alternative threshold presets
    THRESHOLD_PRESETS = {
        'conservative': {'min_harvest_km2': 0.002, 'yield_multiplier': 3.0},
        'standard': {'min_harvest_km2': 0.01, 'yield_multiplier': 2.0},
        'aggressive': {'min_harvest_km2': 0.1, 'yield_multiplier': 1.5}
    }
    
    def __init__(self, 
                 min_harvest_area_km2: float = None,
                 yield_percentile_multiplier: float = None,
                 preset: str = None):
        """
        Initialize SPAM data filter.
        
        Args:
            min_harvest_area_km2: Minimum harvest area threshold in km²
            yield_percentile_multiplier: Multiplier for 99th percentile yield threshold
            preset: Use predefined threshold preset ('conservative', 'standard', 'aggressive')
        """
        if preset and preset in self.THRESHOLD_PRESETS:
            config = self.THRESHOLD_PRESETS[preset]
            self.min_harvest_area_km2 = config['min_harvest_km2']
            self.yield_percentile_multiplier = config['yield_multiplier']
            logger.info(f"Using {preset} filtering preset")
        else:
            self.min_harvest_area_km2 = min_harvest_area_km2 or self.MIN_HARVEST_AREA_KM2
            self.yield_percentile_multiplier = yield_percentile_multiplier or self.YIELD_PERCENTILE_MULTIPLIER
        
        logger.info(f"SPAM filter initialized: min_harvest={self.min_harvest_area_km2} km², "
                   f"yield_multiplier={self.yield_percentile_multiplier}")
    
    def filter_crop_data(self, 
                        production: np.ndarray, 
                        harvest_area: np.ndarray, 
                        crop_name: str = "unknown") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply consistent filtering to crop production and harvest area data.
        
        Args:
            production: Production values in metric tons
            harvest_area: Harvest area values in hectares
            crop_name: Name of crop for logging and reporting
            
        Returns:
            Tuple of (filter_mask, filtering_stats)
            - filter_mask: Boolean array indicating which cells pass filtering
            - filtering_stats: Dictionary with filtering statistics and metadata
        """
        logger.info(f"Applying SPAM data filtering to {crop_name} data")
        
        # Convert harvest area to km²
        harvest_km2 = harvest_area / 100.0
        
        # Initialize statistics
        total_cells = len(production)
        stats = {
            'crop_name': crop_name,
            'total_cells': total_cells,
            'thresholds': {
                'min_harvest_area_km2': self.min_harvest_area_km2,
                'yield_percentile_multiplier': self.yield_percentile_multiplier
            }
        }
        
        # Step 1: Basic validity filter (positive values)
        valid_mask = (production > 0) & (harvest_km2 > 0) & np.isfinite(production) & np.isfinite(harvest_km2)
        valid_cells = np.sum(valid_mask)
        
        stats['valid_cells'] = valid_cells
        stats['invalid_cells'] = total_cells - valid_cells
        
        if valid_cells == 0:
            logger.warning(f"No valid cells found for {crop_name}")
            return np.zeros(total_cells, dtype=bool), stats
        
        logger.info(f"{crop_name}: {valid_cells:,} valid cells out of {total_cells:,}")
        
        # Step 2: Minimum harvest area filter
        area_mask = harvest_km2 >= self.min_harvest_area_km2
        area_filtered_mask = valid_mask & area_mask
        area_filtered_cells = np.sum(area_filtered_mask)
        area_removed = valid_cells - area_filtered_cells
        
        stats['area_filtered_cells'] = area_filtered_cells
        stats['area_removed_cells'] = area_removed
        stats['area_removed_pct'] = (area_removed / valid_cells * 100) if valid_cells > 0 else 0
        
        logger.info(f"{crop_name}: Removed {area_removed:,} cells with harvest area < {self.min_harvest_area_km2} km²")
        
        if area_filtered_cells == 0:
            logger.warning(f"No cells remain after area filtering for {crop_name}")
            return np.zeros(total_cells, dtype=bool), stats
        
        # Step 3: Yield outlier filter
        yields = np.divide(production, harvest_km2, out=np.zeros_like(production), where=harvest_km2>0)
        valid_yields = yields[area_filtered_mask]
        
        if len(valid_yields) > 0:
            yield_99th = np.percentile(valid_yields, 99)
            max_yield_threshold = yield_99th * self.yield_percentile_multiplier
            
            yield_mask = yields <= max_yield_threshold
            final_mask = area_filtered_mask & yield_mask
            final_cells = np.sum(final_mask)
            yield_removed = area_filtered_cells - final_cells
            
            stats['yield_99th_percentile'] = yield_99th
            stats['max_yield_threshold'] = max_yield_threshold
            stats['yield_removed_cells'] = yield_removed
            stats['yield_removed_pct'] = (yield_removed / area_filtered_cells * 100) if area_filtered_cells > 0 else 0
            
            logger.info(f"{crop_name}: 99th percentile yield = {yield_99th:.2f} mt/km²")
            logger.info(f"{crop_name}: Max yield threshold = {max_yield_threshold:.2f} mt/km²")
            logger.info(f"{crop_name}: Removed {yield_removed:,} cells with extreme yields")
        else:
            final_mask = area_filtered_mask
            final_cells = area_filtered_cells
            stats['yield_removed_cells'] = 0
            stats['yield_removed_pct'] = 0
        
        # Final statistics
        stats['final_cells'] = final_cells
        stats['total_removed_cells'] = total_cells - final_cells
        stats['retention_rate'] = (final_cells / total_cells * 100) if total_cells > 0 else 0
        
        logger.info(f"{crop_name}: Final result: {final_cells:,} cells retained ({stats['retention_rate']:.1f}%)")
        
        return final_mask, stats
    
    def filter_dataframe(self, 
                        df: pd.DataFrame, 
                        production_col: str, 
                        harvest_col: str, 
                        crop_name: str = "unknown") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply filtering to a pandas DataFrame.
        
        Args:
            df: DataFrame containing production and harvest data
            production_col: Name of production column
            harvest_col: Name of harvest area column
            crop_name: Name of crop for logging
            
        Returns:
            Tuple of (filtered_dataframe, filtering_stats)
        """
        if production_col not in df.columns or harvest_col not in df.columns:
            raise ValueError(f"Required columns not found: {production_col}, {harvest_col}")
        
        production = df[production_col].values
        harvest_area = df[harvest_col].values
        
        filter_mask, stats = self.filter_crop_data(production, harvest_area, crop_name)
        
        filtered_df = df[filter_mask].copy()
        
        # Add filtering metadata to DataFrame
        filtered_df.attrs['spam_filter_applied'] = True
        filtered_df.attrs['spam_filter_stats'] = stats
        
        return filtered_df, stats
    
    def generate_filtering_report(self, stats: Dict[str, Any]) -> str:
        """
        Generate a human-readable filtering report.
        
        Args:
            stats: Filtering statistics from filter_crop_data
            
        Returns:
            Formatted report string
        """
        crop_name = stats.get('crop_name', 'Unknown')
        
        report = f"""
SPAM Data Filtering Report - {crop_name.upper()}
{'=' * 50}

Input Data:
  Total cells: {stats['total_cells']:,}
  Valid cells: {stats['valid_cells']:,}
  Invalid cells: {stats['invalid_cells']:,}

Filtering Thresholds:
  Minimum harvest area: {stats['thresholds']['min_harvest_area_km2']} km²
  Yield multiplier: {stats['thresholds']['yield_percentile_multiplier']}x 99th percentile

Filtering Results:
  Area filter removed: {stats.get('area_removed_cells', 0):,} cells ({stats.get('area_removed_pct', 0):.1f}%)
  Yield filter removed: {stats.get('yield_removed_cells', 0):,} cells ({stats.get('yield_removed_pct', 0):.1f}%)
  
Final Results:
  Cells retained: {stats['final_cells']:,}
  Retention rate: {stats['retention_rate']:.1f}%
  Total removed: {stats['total_removed_cells']:,}

Yield Statistics:
  99th percentile: {stats.get('yield_99th_percentile', 0):.2f} mt/km²
  Max threshold: {stats.get('max_yield_threshold', 0):.2f} mt/km²

Methodology: docs/DATA_FILTERING_METHODOLOGY.md
"""
        return report.strip()
    
    @staticmethod
    def get_filtering_summary(stats: Dict[str, Any]) -> str:
        """
        Get a concise filtering summary for inclusion in analysis outputs.
        
        Args:
            stats: Filtering statistics from filter_crop_data
            
        Returns:
            Concise summary string
        """
        crop_name = stats.get('crop_name', 'Unknown')
        retention_rate = stats.get('retention_rate', 0)
        final_cells = stats.get('final_cells', 0)
        total_cells = stats.get('total_cells', 0)
        
        return (f"Data Filtering Applied - {crop_name.upper()}: "
                f"{total_cells:,} → {final_cells:,} cells ({retention_rate:.1f}% retained)")


# Convenience function for quick filtering
def filter_spam_data(production: np.ndarray, 
                     harvest_area: np.ndarray, 
                     crop_name: str = "unknown",
                     preset: str = "standard") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function for applying standard SPAM data filtering.
    
    Args:
        production: Production values in metric tons
        harvest_area: Harvest area values in hectares
        crop_name: Name of crop for logging
        preset: Filtering preset ('conservative', 'standard', 'aggressive')
        
    Returns:
        Tuple of (filter_mask, filtering_stats)
    """
    filter_obj = SPAMDataFilter(preset=preset)
    return filter_obj.filter_crop_data(production, harvest_area, crop_name)
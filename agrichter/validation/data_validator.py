"""Data validation module for AgRichter analysis."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from agrichter.core.config import Config


class DataValidator:
    """Comprehensive data validation for AgRichter analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize DataValidator.
        
        Args:
            config: Configuration object with crop type and paths
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up validation thresholds and expected ranges
        self._setup_validation_thresholds()
    
    def _setup_validation_thresholds(self) -> None:
        """Set up validation thresholds and expected ranges."""
        # Expected global production ranges (metric tons) for SPAM 2020
        # Based on SPAM 2020 documentation and FAO statistics
        self.expected_production_ranges = {
            'wheat': (600e6, 800e6),      # 600-800 million MT
            'rice': (700e6, 900e6),       # 700-900 million MT  
            'maize': (1000e6, 1300e6),    # 1000-1300 million MT
            'allgrain': (2500e6, 3500e6)  # 2500-3500 million MT (sum of major grains)
        }
        
        # Expected harvest area ranges (hectares) for SPAM 2020
        self.expected_harvest_ranges = {
            'wheat': (200e6, 250e6),      # 200-250 million ha
            'rice': (150e6, 200e6),       # 150-200 million ha
            'maize': (180e6, 220e6),      # 180-220 million ha
            'allgrain': (600e6, 800e6)    # 600-800 million ha
        }
        
        # Coordinate ranges (WGS84)
        self.coordinate_ranges = {
            'latitude': (-90.0, 90.0),
            'longitude': (-180.0, 180.0)
        }
        
        # Magnitude ranges for historical events
        self.magnitude_ranges = {
            'min': 2.0,  # Minimum reasonable magnitude
            'max': 8.0,  # Maximum reasonable magnitude
            'typical_min': 3.0,  # Typical minimum for recorded events
            'typical_max': 7.0   # Typical maximum for recorded events
        }
        
        # Production loss thresholds (kcal)
        self.production_loss_ranges = {
            'min': 1e12,   # 1 trillion kcal minimum
            'max': 1e18,   # 1 quintillion kcal maximum
            'typical_max': 1e16  # Typical maximum for single event
        }
        
        # MATLAB comparison tolerance
        self.matlab_comparison_tolerance = 0.05  # 5% difference threshold

    
    def validate_spam_data(self, production_df: pd.DataFrame, 
                          harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate SPAM data completeness and ranges.
        
        Args:
            production_df: SPAM production DataFrame
            harvest_df: SPAM harvest area DataFrame
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating SPAM data...")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check DataFrames are not empty
        if production_df.empty:
            results['valid'] = False
            results['errors'].append("Production DataFrame is empty")
            return results
        
        if harvest_df.empty:
            results['valid'] = False
            results['errors'].append("Harvest area DataFrame is empty")
            return results
        
        # Validate coordinate columns exist
        coord_validation = self._validate_coordinates(production_df, harvest_df)
        results['statistics']['coordinates'] = coord_validation
        if not coord_validation['valid']:
            results['valid'] = False
            results['errors'].extend(coord_validation['errors'])
        
        # Validate crop columns exist
        crop_validation = self._validate_crop_columns(production_df, harvest_df)
        results['statistics']['crops'] = crop_validation
        if not crop_validation['valid']:
            results['valid'] = False
            results['errors'].extend(crop_validation['errors'])
        
        # Check total global production for each crop
        production_totals = self._validate_production_totals(production_df)
        results['statistics']['production_totals'] = production_totals
        if production_totals['warnings']:
            results['warnings'].extend(production_totals['warnings'])
        
        # Check harvest area totals
        harvest_totals = self._validate_harvest_totals(harvest_df)
        results['statistics']['harvest_totals'] = harvest_totals
        if harvest_totals['warnings']:
            results['warnings'].extend(harvest_totals['warnings'])
        
        # Check for missing or NaN values
        missing_data = self._check_missing_data(production_df, harvest_df)
        results['statistics']['missing_data'] = missing_data
        if missing_data['critical_missing']:
            results['valid'] = False
            results['errors'].extend(missing_data['errors'])
        if missing_data['warnings']:
            results['warnings'].extend(missing_data['warnings'])
        
        # Validate coordinate ranges
        coord_ranges = self._validate_coordinate_ranges(production_df, harvest_df)
        results['statistics']['coordinate_ranges'] = coord_ranges
        if not coord_ranges['valid']:
            results['warnings'].extend(coord_ranges['warnings'])
        
        self.logger.info(f"SPAM validation complete. Valid: {results['valid']}, "
                        f"Errors: {len(results['errors'])}, Warnings: {len(results['warnings'])}")
        
        return results
    
    def _validate_coordinates(self, production_df: pd.DataFrame, 
                             harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate coordinate columns exist and are consistent."""
        result = {'valid': True, 'errors': []}
        
        required_cols = ['x', 'y']
        
        for col in required_cols:
            if col not in production_df.columns:
                result['valid'] = False
                result['errors'].append(f"Production data missing '{col}' column")
            
            if col not in harvest_df.columns:
                result['valid'] = False
                result['errors'].append(f"Harvest area data missing '{col}' column")
        
        # Check if coordinates match between datasets
        if result['valid']:
            prod_coords = set(zip(production_df['x'], production_df['y']))
            harv_coords = set(zip(harvest_df['x'], harvest_df['y']))
            
            if prod_coords != harv_coords:
                result['errors'].append(
                    f"Coordinate mismatch: Production has {len(prod_coords)} cells, "
                    f"Harvest has {len(harv_coords)} cells"
                )
        
        return result
    
    def _validate_crop_columns(self, production_df: pd.DataFrame,
                               harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate crop columns exist for selected crop type."""
        result = {'valid': True, 'errors': [], 'found_crops': []}
        
        # Get crop indices for current crop type
        crop_indices = self.config.get_crop_indices()
        
        # SPAM crop column naming: lowercase crop name + '_a' suffix
        # Mapping from crop index to column name
        crop_column_map = {
            14: 'barl_a',  # Barley
            26: 'maiz_a',  # Maize
            28: 'ocer_a',  # Other cereals
            37: 'pmil_a',  # Pearl millet
            42: 'rice_a',  # Rice
            45: 'sorg_a',  # Sorghum
            57: 'whea_a'   # Wheat
        }
        
        for idx in crop_indices:
            if idx in crop_column_map:
                col_name = crop_column_map[idx]
                
                if col_name not in production_df.columns:
                    result['valid'] = False
                    result['errors'].append(
                        f"Production data missing crop column '{col_name}' (index {idx})"
                    )
                else:
                    result['found_crops'].append(col_name)
                
                if col_name not in harvest_df.columns:
                    result['valid'] = False
                    result['errors'].append(
                        f"Harvest area data missing crop column '{col_name}' (index {idx})"
                    )
        
        return result
    
    def _validate_production_totals(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Check total global production against expected ranges."""
        result = {'totals': {}, 'warnings': []}
        
        # Get expected range for current crop type
        crop_type = self.config.crop_type
        expected_range = self.expected_production_ranges.get(crop_type)
        
        if expected_range is None:
            result['warnings'].append(f"No expected range defined for crop type '{crop_type}'")
            return result
        
        # Calculate total production for selected crops
        crop_indices = self.config.get_crop_indices()
        crop_column_map = {
            14: 'barl_a', 26: 'maiz_a', 28: 'ocer_a', 37: 'pmil_a',
            42: 'rice_a', 45: 'sorg_a', 57: 'whea_a'
        }
        
        total_production = 0.0
        for idx in crop_indices:
            if idx in crop_column_map:
                col_name = crop_column_map[idx]
                if col_name in production_df.columns:
                    crop_total = production_df[col_name].sum()
                    result['totals'][col_name] = crop_total
                    total_production += crop_total
        
        result['total_production_mt'] = total_production
        result['expected_range_mt'] = expected_range
        
        # Check if within expected range
        min_expected, max_expected = expected_range
        if total_production < min_expected:
            result['warnings'].append(
                f"Total production ({total_production:.2e} MT) is below expected minimum "
                f"({min_expected:.2e} MT) for {crop_type}"
            )
        elif total_production > max_expected:
            result['warnings'].append(
                f"Total production ({total_production:.2e} MT) exceeds expected maximum "
                f"({max_expected:.2e} MT) for {crop_type}"
            )
        
        return result
    
    def _validate_harvest_totals(self, harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """Check total harvest area against expected ranges."""
        result = {'totals': {}, 'warnings': []}
        
        # Get expected range for current crop type
        crop_type = self.config.crop_type
        expected_range = self.expected_harvest_ranges.get(crop_type)
        
        if expected_range is None:
            result['warnings'].append(f"No expected range defined for crop type '{crop_type}'")
            return result
        
        # Calculate total harvest area for selected crops
        crop_indices = self.config.get_crop_indices()
        crop_column_map = {
            14: 'barl_a', 26: 'maiz_a', 28: 'ocer_a', 37: 'pmil_a',
            42: 'rice_a', 45: 'sorg_a', 57: 'whea_a'
        }
        
        total_harvest = 0.0
        for idx in crop_indices:
            if idx in crop_column_map:
                col_name = crop_column_map[idx]
                if col_name in harvest_df.columns:
                    crop_total = harvest_df[col_name].sum()
                    result['totals'][col_name] = crop_total
                    total_harvest += crop_total
        
        result['total_harvest_ha'] = total_harvest
        result['expected_range_ha'] = expected_range
        
        # Check if within expected range
        min_expected, max_expected = expected_range
        if total_harvest < min_expected:
            result['warnings'].append(
                f"Total harvest area ({total_harvest:.2e} ha) is below expected minimum "
                f"({min_expected:.2e} ha) for {crop_type}"
            )
        elif total_harvest > max_expected:
            result['warnings'].append(
                f"Total harvest area ({total_harvest:.2e} ha) exceeds expected maximum "
                f"({max_expected:.2e} ha) for {crop_type}"
            )
        
        return result
    
    def _check_missing_data(self, production_df: pd.DataFrame,
                           harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing or NaN values in critical columns."""
        result = {
            'critical_missing': False,
            'errors': [],
            'warnings': [],
            'production_missing': {},
            'harvest_missing': {}
        }
        
        # Check coordinate columns (critical)
        for col in ['x', 'y', 'iso3']:
            if col in production_df.columns:
                missing_count = production_df[col].isna().sum()
                if missing_count > 0:
                    result['critical_missing'] = True
                    result['errors'].append(
                        f"Production data has {missing_count} missing values in '{col}' column"
                    )
            
            if col in harvest_df.columns:
                missing_count = harvest_df[col].isna().sum()
                if missing_count > 0:
                    result['critical_missing'] = True
                    result['errors'].append(
                        f"Harvest data has {missing_count} missing values in '{col}' column"
                    )
        
        # Check crop columns (non-critical, but report)
        crop_indices = self.config.get_crop_indices()
        crop_column_map = {
            14: 'barl_a', 26: 'maiz_a', 28: 'ocer_a', 37: 'pmil_a',
            42: 'rice_a', 45: 'sorg_a', 57: 'whea_a'
        }
        
        for idx in crop_indices:
            if idx in crop_column_map:
                col_name = crop_column_map[idx]
                
                if col_name in production_df.columns:
                    missing_count = production_df[col_name].isna().sum()
                    result['production_missing'][col_name] = missing_count
                    if missing_count > 0:
                        result['warnings'].append(
                            f"Production data has {missing_count} NaN values in '{col_name}'"
                        )
                
                if col_name in harvest_df.columns:
                    missing_count = harvest_df[col_name].isna().sum()
                    result['harvest_missing'][col_name] = missing_count
                    if missing_count > 0:
                        result['warnings'].append(
                            f"Harvest data has {missing_count} NaN values in '{col_name}'"
                        )
        
        return result
    
    def _validate_coordinate_ranges(self, production_df: pd.DataFrame,
                                   harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate coordinate values are within expected ranges."""
        result = {'valid': True, 'warnings': [], 'ranges': {}}
        
        # Check production coordinates
        if 'x' in production_df.columns and 'y' in production_df.columns:
            prod_x_range = (production_df['x'].min(), production_df['x'].max())
            prod_y_range = (production_df['y'].min(), production_df['y'].max())
            
            result['ranges']['production_x'] = prod_x_range
            result['ranges']['production_y'] = prod_y_range
            
            # Check longitude (x)
            lon_min, lon_max = self.coordinate_ranges['longitude']
            if prod_x_range[0] < lon_min or prod_x_range[1] > lon_max:
                result['warnings'].append(
                    f"Production longitude range {prod_x_range} exceeds expected {(lon_min, lon_max)}"
                )
            
            # Check latitude (y)
            lat_min, lat_max = self.coordinate_ranges['latitude']
            if prod_y_range[0] < lat_min or prod_y_range[1] > lat_max:
                result['warnings'].append(
                    f"Production latitude range {prod_y_range} exceeds expected {(lat_min, lat_max)}"
                )
        
        # Check harvest coordinates
        if 'x' in harvest_df.columns and 'y' in harvest_df.columns:
            harv_x_range = (harvest_df['x'].min(), harvest_df['x'].max())
            harv_y_range = (harvest_df['y'].min(), harvest_df['y'].max())
            
            result['ranges']['harvest_x'] = harv_x_range
            result['ranges']['harvest_y'] = harv_y_range
            
            # Check longitude (x)
            lon_min, lon_max = self.coordinate_ranges['longitude']
            if harv_x_range[0] < lon_min or harv_x_range[1] > lon_max:
                result['warnings'].append(
                    f"Harvest longitude range {harv_x_range} exceeds expected {(lon_min, lon_max)}"
                )
            
            # Check latitude (y)
            lat_min, lat_max = self.coordinate_ranges['latitude']
            if harv_y_range[0] < lat_min or harv_y_range[1] > lat_max:
                result['warnings'].append(
                    f"Harvest latitude range {harv_y_range} exceeds expected {(lat_min, lat_max)}"
                )
        
        return result

    
    def validate_event_results(self, events_df: pd.DataFrame,
                               global_production_kcal: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate event results for reasonable values.
        
        Args:
            events_df: DataFrame with event results (must have columns: 
                      event_name, harvest_area_loss_ha, production_loss_kcal, magnitude)
            global_production_kcal: Optional total global production in kcal for comparison
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating event results...")
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'suspicious_events': []
        }
        
        # Check required columns exist
        required_cols = ['event_name', 'harvest_area_loss_ha', 'production_loss_kcal', 'magnitude']
        missing_cols = [col for col in required_cols if col not in events_df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Check for empty DataFrame
        if events_df.empty:
            results['valid'] = False
            results['errors'].append("Events DataFrame is empty")
            return results
        
        # Validate production losses don't exceed global production
        if global_production_kcal is not None:
            loss_validation = self._validate_losses_vs_global(events_df, global_production_kcal)
            results['statistics']['loss_validation'] = loss_validation
            if not loss_validation['valid']:
                results['valid'] = False
                results['errors'].extend(loss_validation['errors'])
            if loss_validation['warnings']:
                results['warnings'].extend(loss_validation['warnings'])
        
        # Validate magnitude ranges
        magnitude_validation = self._validate_magnitude_ranges(events_df)
        results['statistics']['magnitude_validation'] = magnitude_validation
        if magnitude_validation['warnings']:
            results['warnings'].extend(magnitude_validation['warnings'])
        if magnitude_validation['suspicious_events']:
            results['suspicious_events'].extend(magnitude_validation['suspicious_events'])
        
        # Identify events with zero or suspicious losses
        zero_loss_events = self._identify_zero_loss_events(events_df)
        results['statistics']['zero_loss_events'] = zero_loss_events
        if zero_loss_events['events']:
            results['warnings'].append(
                f"Found {len(zero_loss_events['events'])} events with zero losses: "
                f"{zero_loss_events['events']}"
            )
        
        # Calculate validation statistics
        stats = self._calculate_event_statistics(events_df)
        results['statistics']['event_stats'] = stats
        
        # Check for NaN values
        nan_check = self._check_nan_values(events_df)
        results['statistics']['nan_check'] = nan_check
        if nan_check['has_nans']:
            results['warnings'].extend(nan_check['warnings'])
        
        self.logger.info(f"Event validation complete. Valid: {results['valid']}, "
                        f"Errors: {len(results['errors'])}, Warnings: {len(results['warnings'])}, "
                        f"Suspicious events: {len(results['suspicious_events'])}")
        
        return results
    
    def _validate_losses_vs_global(self, events_df: pd.DataFrame,
                                   global_production_kcal: float) -> Dict[str, Any]:
        """Check that event losses don't exceed global production."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        for idx, row in events_df.iterrows():
            event_name = row['event_name']
            production_loss = row['production_loss_kcal']
            
            if production_loss > global_production_kcal:
                result['valid'] = False
                result['errors'].append(
                    f"Event '{event_name}' has production loss ({production_loss:.2e} kcal) "
                    f"exceeding global production ({global_production_kcal:.2e} kcal)"
                )
            elif production_loss > 0.5 * global_production_kcal:
                result['warnings'].append(
                    f"Event '{event_name}' has production loss ({production_loss:.2e} kcal) "
                    f"exceeding 50% of global production ({global_production_kcal:.2e} kcal)"
                )
        
        return result
    
    def _validate_magnitude_ranges(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Verify magnitude ranges are reasonable."""
        result = {
            'warnings': [],
            'suspicious_events': [],
            'magnitude_stats': {}
        }
        
        magnitudes = events_df['magnitude'].dropna()
        
        if len(magnitudes) == 0:
            result['warnings'].append("No valid magnitudes found in events data")
            return result
        
        result['magnitude_stats'] = {
            'min': magnitudes.min(),
            'max': magnitudes.max(),
            'mean': magnitudes.mean(),
            'median': magnitudes.median()
        }
        
        # Check for magnitudes outside reasonable range
        for idx, row in events_df.iterrows():
            event_name = row['event_name']
            magnitude = row['magnitude']
            
            if pd.isna(magnitude):
                continue
            
            if magnitude < self.magnitude_ranges['min']:
                result['warnings'].append(
                    f"Event '{event_name}' has magnitude {magnitude:.2f} below minimum "
                    f"reasonable value {self.magnitude_ranges['min']}"
                )
                result['suspicious_events'].append(event_name)
            
            elif magnitude > self.magnitude_ranges['max']:
                result['warnings'].append(
                    f"Event '{event_name}' has magnitude {magnitude:.2f} above maximum "
                    f"reasonable value {self.magnitude_ranges['max']}"
                )
                result['suspicious_events'].append(event_name)
            
            elif magnitude < self.magnitude_ranges['typical_min']:
                result['warnings'].append(
                    f"Event '{event_name}' has magnitude {magnitude:.2f} below typical minimum "
                    f"{self.magnitude_ranges['typical_min']} (may be valid but unusual)"
                )
            
            elif magnitude > self.magnitude_ranges['typical_max']:
                result['warnings'].append(
                    f"Event '{event_name}' has magnitude {magnitude:.2f} above typical maximum "
                    f"{self.magnitude_ranges['typical_max']} (may be valid but unusual)"
                )
        
        return result
    
    def _identify_zero_loss_events(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify events with zero or near-zero losses."""
        result = {'events': [], 'details': {}}
        
        for idx, row in events_df.iterrows():
            event_name = row['event_name']
            harvest_loss = row['harvest_area_loss_ha']
            production_loss = row['production_loss_kcal']
            
            if harvest_loss == 0 or production_loss == 0:
                result['events'].append(event_name)
                result['details'][event_name] = {
                    'harvest_area_loss_ha': harvest_loss,
                    'production_loss_kcal': production_loss,
                    'reason': 'Zero loss'
                }
            elif harvest_loss < 1.0 or production_loss < 1e10:
                result['events'].append(event_name)
                result['details'][event_name] = {
                    'harvest_area_loss_ha': harvest_loss,
                    'production_loss_kcal': production_loss,
                    'reason': 'Near-zero loss (suspiciously small)'
                }
        
        return result
    
    def _calculate_event_statistics(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for event results."""
        stats = {}
        
        # Harvest area loss statistics
        harvest_losses = events_df['harvest_area_loss_ha'].dropna()
        if len(harvest_losses) > 0:
            stats['harvest_area_loss'] = {
                'min': harvest_losses.min(),
                'max': harvest_losses.max(),
                'mean': harvest_losses.mean(),
                'median': harvest_losses.median(),
                'total': harvest_losses.sum()
            }
        
        # Production loss statistics
        production_losses = events_df['production_loss_kcal'].dropna()
        if len(production_losses) > 0:
            stats['production_loss'] = {
                'min': production_losses.min(),
                'max': production_losses.max(),
                'mean': production_losses.mean(),
                'median': production_losses.median(),
                'total': production_losses.sum()
            }
        
        # Magnitude statistics
        magnitudes = events_df['magnitude'].dropna()
        if len(magnitudes) > 0:
            stats['magnitude'] = {
                'min': magnitudes.min(),
                'max': magnitudes.max(),
                'mean': magnitudes.mean(),
                'median': magnitudes.median()
            }
        
        stats['total_events'] = len(events_df)
        stats['events_with_data'] = len(events_df[events_df['production_loss_kcal'] > 0])
        
        return stats
    
    def _check_nan_values(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Check for NaN values in event results."""
        result = {'has_nans': False, 'warnings': [], 'nan_counts': {}}
        
        for col in ['harvest_area_loss_ha', 'production_loss_kcal', 'magnitude']:
            nan_count = events_df[col].isna().sum()
            result['nan_counts'][col] = nan_count
            
            if nan_count > 0:
                result['has_nans'] = True
                result['warnings'].append(
                    f"Column '{col}' has {nan_count} NaN values out of {len(events_df)} events"
                )
        
        return result

    
    def compare_with_matlab(self, python_results_df: pd.DataFrame,
                           matlab_results_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Compare Python results with MATLAB reference results.
        
        Note: MATLAB results used SPAM2010, so differences are expected when using SPAM2020.
        This comparison is primarily useful for validating calculation methodology rather than
        exact numerical agreement.
        
        Args:
            python_results_df: DataFrame with Python event results
            matlab_results_path: Optional path to MATLAB results CSV file
        
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing Python results with MATLAB reference...")
        
        results = {
            'comparison_available': False,
            'warnings': [],
            'differences': {},
            'statistics': {}
        }
        
        # Add warning about SPAM version difference
        if self.config.spam_version == '2020':
            results['warnings'].append(
                "MATLAB results used SPAM2010 data. Current analysis uses SPAM2020. "
                "Significant differences are expected due to updated production data."
            )
        
        # If no MATLAB path provided, try default location
        if matlab_results_path is None:
            matlab_results_path = self.config.root_dir / 'outputs' / f'matlab_reference_{self.config.crop_type}.csv'
        
        # Check if MATLAB results file exists
        if not matlab_results_path.exists():
            results['warnings'].append(
                f"MATLAB reference results not found at {matlab_results_path}. "
                "Comparison skipped."
            )
            return results
        
        # Load MATLAB results
        try:
            matlab_df = pd.read_csv(matlab_results_path)
            results['comparison_available'] = True
            self.logger.info(f"Loaded MATLAB reference results from {matlab_results_path}")
        except Exception as e:
            results['warnings'].append(f"Failed to load MATLAB results: {e}")
            return results
        
        # Perform comparison
        comparison = self._compare_event_losses(python_results_df, matlab_df)
        results['differences'] = comparison['differences']
        results['statistics'] = comparison['statistics']
        results['warnings'].extend(comparison['warnings'])
        
        # Flag events with differences > 5%
        flagged_events = comparison['flagged_events']
        if flagged_events:
            results['warnings'].append(
                f"Found {len(flagged_events)} events with differences > {self.matlab_comparison_tolerance*100}%: "
                f"{flagged_events}"
            )
        
        self.logger.info(f"MATLAB comparison complete. Flagged events: {len(flagged_events)}")
        
        return results
    
    def _compare_event_losses(self, python_df: pd.DataFrame,
                             matlab_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare event losses between Python and MATLAB implementations."""
        result = {
            'differences': {},
            'statistics': {},
            'warnings': [],
            'flagged_events': []
        }
        
        # Check for event_name column in both DataFrames
        if 'event_name' not in python_df.columns:
            result['warnings'].append("Python results missing 'event_name' column")
            return result
        
        if 'event_name' not in matlab_df.columns:
            result['warnings'].append("MATLAB results missing 'event_name' column")
            return result
        
        # Find common events
        python_events = set(python_df['event_name'].unique())
        matlab_events = set(matlab_df['event_name'].unique())
        common_events = python_events.intersection(matlab_events)
        
        result['statistics']['python_only_events'] = list(python_events - matlab_events)
        result['statistics']['matlab_only_events'] = list(matlab_events - python_events)
        result['statistics']['common_events'] = list(common_events)
        
        if not common_events:
            result['warnings'].append("No common events found between Python and MATLAB results")
            return result
        
        # Compare production losses for common events
        percentage_diffs = []
        absolute_diffs = []
        
        for event_name in common_events:
            python_row = python_df[python_df['event_name'] == event_name]
            matlab_row = matlab_df[matlab_df['event_name'] == event_name]
            
            if python_row.empty or matlab_row.empty:
                continue
            
            # Get production loss values
            python_loss = python_row['production_loss_kcal'].iloc[0]
            
            # Try different possible column names for MATLAB results
            matlab_loss = None
            for col in ['production_loss_kcal', 'ProductionLoss_kcal', 'LostProd_kcal']:
                if col in matlab_row.columns:
                    matlab_loss = matlab_row[col].iloc[0]
                    break
            
            if matlab_loss is None:
                result['warnings'].append(
                    f"Could not find production loss column in MATLAB results for event '{event_name}'"
                )
                continue
            
            # Calculate differences
            absolute_diff = python_loss - matlab_loss
            
            if matlab_loss != 0:
                percentage_diff = (absolute_diff / matlab_loss) * 100
            else:
                percentage_diff = float('inf') if python_loss != 0 else 0.0
            
            result['differences'][event_name] = {
                'python_loss_kcal': python_loss,
                'matlab_loss_kcal': matlab_loss,
                'absolute_diff_kcal': absolute_diff,
                'percentage_diff': percentage_diff
            }
            
            percentage_diffs.append(abs(percentage_diff))
            absolute_diffs.append(abs(absolute_diff))
            
            # Flag if difference exceeds tolerance
            if abs(percentage_diff) > self.matlab_comparison_tolerance * 100:
                result['flagged_events'].append(event_name)
        
        # Calculate summary statistics
        if percentage_diffs:
            result['statistics']['percentage_differences'] = {
                'min': min(percentage_diffs),
                'max': max(percentage_diffs),
                'mean': np.mean(percentage_diffs),
                'median': np.median(percentage_diffs)
            }
        
        if absolute_diffs:
            result['statistics']['absolute_differences'] = {
                'min': min(absolute_diffs),
                'max': max(absolute_diffs),
                'mean': np.mean(absolute_diffs),
                'median': np.median(absolute_diffs)
            }
        
        result['statistics']['events_compared'] = len(common_events)
        result['statistics']['events_flagged'] = len(result['flagged_events'])
        
        return result

    
    def generate_validation_report(self, 
                                   spam_validation: Optional[Dict[str, Any]] = None,
                                   event_validation: Optional[Dict[str, Any]] = None,
                                   matlab_comparison: Optional[Dict[str, Any]] = None,
                                   output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            spam_validation: Results from validate_spam_data()
            event_validation: Results from validate_event_results()
            matlab_comparison: Results from compare_with_matlab()
            output_path: Optional path to save report file
        
        Returns:
            Formatted validation report as string
        """
        self.logger.info("Generating validation report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AgRichter Data Validation Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Crop Type: {self.config.crop_type}")
        report_lines.append(f"SPAM Version: {self.config.spam_version}")
        report_lines.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # SPAM Data Validation Section
        if spam_validation:
            report_lines.append("-" * 80)
            report_lines.append("SPAM DATA VALIDATION")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            report_lines.append(f"Overall Status: {'PASS' if spam_validation['valid'] else 'FAIL'}")
            report_lines.append(f"Errors: {len(spam_validation['errors'])}")
            report_lines.append(f"Warnings: {len(spam_validation['warnings'])}")
            report_lines.append("")
            
            if spam_validation['errors']:
                report_lines.append("Errors:")
                for error in spam_validation['errors']:
                    report_lines.append(f"  - {error}")
                report_lines.append("")
            
            if spam_validation['warnings']:
                report_lines.append("Warnings:")
                for warning in spam_validation['warnings']:
                    report_lines.append(f"  - {warning}")
                report_lines.append("")
            
            # Production totals
            if 'production_totals' in spam_validation['statistics']:
                prod_stats = spam_validation['statistics']['production_totals']
                report_lines.append("Production Statistics:")
                if 'total_production_mt' in prod_stats:
                    report_lines.append(f"  Total Production: {prod_stats['total_production_mt']:.2e} MT")
                if 'expected_range_mt' in prod_stats:
                    min_exp, max_exp = prod_stats['expected_range_mt']
                    report_lines.append(f"  Expected Range: {min_exp:.2e} - {max_exp:.2e} MT")
                if 'totals' in prod_stats:
                    report_lines.append("  By Crop:")
                    for crop, total in prod_stats['totals'].items():
                        report_lines.append(f"    {crop}: {total:.2e} MT")
                report_lines.append("")
            
            # Harvest area totals
            if 'harvest_totals' in spam_validation['statistics']:
                harv_stats = spam_validation['statistics']['harvest_totals']
                report_lines.append("Harvest Area Statistics:")
                if 'total_harvest_ha' in harv_stats:
                    report_lines.append(f"  Total Harvest Area: {harv_stats['total_harvest_ha']:.2e} ha")
                if 'expected_range_ha' in harv_stats:
                    min_exp, max_exp = harv_stats['expected_range_ha']
                    report_lines.append(f"  Expected Range: {min_exp:.2e} - {max_exp:.2e} ha")
                report_lines.append("")
            
            # Coordinate ranges
            if 'coordinate_ranges' in spam_validation['statistics']:
                coord_stats = spam_validation['statistics']['coordinate_ranges']
                if 'ranges' in coord_stats:
                    report_lines.append("Coordinate Ranges:")
                    for key, value in coord_stats['ranges'].items():
                        report_lines.append(f"  {key}: {value}")
                    report_lines.append("")
        
        # Event Results Validation Section
        if event_validation:
            report_lines.append("-" * 80)
            report_lines.append("EVENT RESULTS VALIDATION")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            report_lines.append(f"Overall Status: {'PASS' if event_validation['valid'] else 'FAIL'}")
            report_lines.append(f"Errors: {len(event_validation['errors'])}")
            report_lines.append(f"Warnings: {len(event_validation['warnings'])}")
            report_lines.append(f"Suspicious Events: {len(event_validation['suspicious_events'])}")
            report_lines.append("")
            
            if event_validation['errors']:
                report_lines.append("Errors:")
                for error in event_validation['errors']:
                    report_lines.append(f"  - {error}")
                report_lines.append("")
            
            if event_validation['warnings']:
                report_lines.append("Warnings:")
                for warning in event_validation['warnings'][:10]:  # Limit to first 10
                    report_lines.append(f"  - {warning}")
                if len(event_validation['warnings']) > 10:
                    report_lines.append(f"  ... and {len(event_validation['warnings']) - 10} more warnings")
                report_lines.append("")
            
            # Event statistics
            if 'event_stats' in event_validation['statistics']:
                stats = event_validation['statistics']['event_stats']
                report_lines.append("Event Statistics:")
                report_lines.append(f"  Total Events: {stats.get('total_events', 0)}")
                report_lines.append(f"  Events with Data: {stats.get('events_with_data', 0)}")
                
                if 'production_loss' in stats:
                    pl = stats['production_loss']
                    report_lines.append("  Production Loss:")
                    report_lines.append(f"    Min: {pl['min']:.2e} kcal")
                    report_lines.append(f"    Max: {pl['max']:.2e} kcal")
                    report_lines.append(f"    Mean: {pl['mean']:.2e} kcal")
                    report_lines.append(f"    Median: {pl['median']:.2e} kcal")
                    report_lines.append(f"    Total: {pl['total']:.2e} kcal")
                
                if 'magnitude' in stats:
                    mag = stats['magnitude']
                    report_lines.append("  Magnitude:")
                    report_lines.append(f"    Min: {mag['min']:.2f}")
                    report_lines.append(f"    Max: {mag['max']:.2f}")
                    report_lines.append(f"    Mean: {mag['mean']:.2f}")
                    report_lines.append(f"    Median: {mag['median']:.2f}")
                
                report_lines.append("")
            
            # Zero loss events
            if 'zero_loss_events' in event_validation['statistics']:
                zero_loss = event_validation['statistics']['zero_loss_events']
                if zero_loss['events']:
                    report_lines.append(f"Events with Zero/Near-Zero Losses ({len(zero_loss['events'])}):")
                    for event in zero_loss['events'][:5]:  # Show first 5
                        if event in zero_loss['details']:
                            details = zero_loss['details'][event]
                            report_lines.append(f"  - {event}: {details['reason']}")
                    if len(zero_loss['events']) > 5:
                        report_lines.append(f"  ... and {len(zero_loss['events']) - 5} more")
                    report_lines.append("")
        
        # MATLAB Comparison Section
        if matlab_comparison:
            report_lines.append("-" * 80)
            report_lines.append("MATLAB COMPARISON")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            if matlab_comparison['comparison_available']:
                report_lines.append("Status: Comparison Available")
                
                if 'statistics' in matlab_comparison:
                    stats = matlab_comparison['statistics']
                    
                    if 'events_compared' in stats:
                        report_lines.append(f"Events Compared: {stats['events_compared']}")
                    if 'events_flagged' in stats:
                        report_lines.append(f"Events Flagged (>{self.matlab_comparison_tolerance*100}% diff): {stats['events_flagged']}")
                    
                    if 'percentage_differences' in stats:
                        pd_stats = stats['percentage_differences']
                        report_lines.append("Percentage Differences:")
                        report_lines.append(f"  Min: {pd_stats['min']:.2f}%")
                        report_lines.append(f"  Max: {pd_stats['max']:.2f}%")
                        report_lines.append(f"  Mean: {pd_stats['mean']:.2f}%")
                        report_lines.append(f"  Median: {pd_stats['median']:.2f}%")
                    
                    if 'python_only_events' in stats and stats['python_only_events']:
                        report_lines.append(f"Python-only Events: {len(stats['python_only_events'])}")
                    
                    if 'matlab_only_events' in stats and stats['matlab_only_events']:
                        report_lines.append(f"MATLAB-only Events: {len(stats['matlab_only_events'])}")
                
                report_lines.append("")
            else:
                report_lines.append("Status: Comparison Not Available")
                report_lines.append("")
            
            if matlab_comparison['warnings']:
                report_lines.append("Notes:")
                for warning in matlab_comparison['warnings']:
                    report_lines.append(f"  - {warning}")
                report_lines.append("")
        
        # Summary Section
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        overall_valid = True
        if spam_validation and not spam_validation['valid']:
            overall_valid = False
        if event_validation and not event_validation['valid']:
            overall_valid = False
        
        report_lines.append(f"Overall Validation Status: {'PASS' if overall_valid else 'FAIL'}")
        
        total_errors = 0
        total_warnings = 0
        if spam_validation:
            total_errors += len(spam_validation['errors'])
            total_warnings += len(spam_validation['warnings'])
        if event_validation:
            total_errors += len(event_validation['errors'])
            total_warnings += len(event_validation['warnings'])
        if matlab_comparison:
            total_warnings += len(matlab_comparison['warnings'])
        
        report_lines.append(f"Total Errors: {total_errors}")
        report_lines.append(f"Total Warnings: {total_warnings}")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Join all lines into report string
        report = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report)
                self.logger.info(f"Validation report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save validation report: {e}")
        
        return report

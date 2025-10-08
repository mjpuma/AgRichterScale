"""H-P Envelope calculation for AgriRichter framework."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.utils import ProgressReporter


class HPEnvelopeError(Exception):
    """Exception raised for H-P envelope calculation errors."""
    pass


class HPEnvelopeCalculator:
    """Calculator for Harvest-Production (H-P) envelope analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize H-P envelope calculator.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agririchter.envelope')
        
        # Get disruption ranges for current crop type
        self.disruption_ranges = config.disruption_range
        
    def calculate_hp_envelope(self, production_kcal: pd.DataFrame, 
                             harvest_km2: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate the Harvest-Production envelope (upper and lower bounds).
        
        This implements the exact MATLAB algorithm:
        HPmatrix = [TotalHarvest(:),TotalProduction(:),TotalProduction(:)./TotalHarvest(:)];
        HPmatrix_sorted = sortrows(HPmatrix,3); % sort by grid-cell *yield*
        HPmatrix_cumsum_SmallLarge = cumsum(HPmatrix_sorted);
        HPmatrix_cumsum_LargeSmall = cumsum(flipud(HPmatrix_sorted));
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
        
        Returns:
            Dictionary with envelope data arrays
        """
        try:
            self.logger.info("Starting H-P envelope calculation")
            
            # Step 1: Prepare data matrix
            hp_matrix = self._prepare_hp_matrix(production_kcal, harvest_km2)
            
            # Step 2: Sort by yield (productivity)
            hp_matrix_sorted = self._sort_by_yield(hp_matrix)
            
            # Step 3: Determine appropriate disruption range based on actual data
            disruption_ranges = self._get_adaptive_disruption_range(hp_matrix_sorted)
            
            # Step 4: Calculate envelope bounds using MATLAB-exact algorithm
            envelope_data = self._calculate_envelope_bounds(
                hp_matrix_sorted, disruption_ranges
            )
            
            self.logger.info("H-P envelope calculation completed successfully")
            
            return envelope_data
            
        except Exception as e:
            raise HPEnvelopeError(f"Failed to calculate H-P envelope: {str(e)}")
    
    def _prepare_hp_matrix(self, production_kcal: pd.DataFrame, 
                          harvest_km2: pd.DataFrame) -> np.ndarray:
        """
        Prepare the Harvest-Production matrix following MATLAB algorithm.
        
        MATLAB: HPmatrix = [TotalHarvest(:),TotalProduction(:),TotalProduction(:)./TotalHarvest(:)];
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
        
        Returns:
            Matrix with [harvest_area, production, yield] for each grid cell
        """
        # Get crop columns based on config
        crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
        
        if not crop_columns:
            raise HPEnvelopeError("No crop columns found in production data")
        
        # Filter to selected crop types based on config
        if self.config.crop_type == 'wheat':
            crop_columns = [col for col in crop_columns if 'WHEA' in col.upper()]
        elif self.config.crop_type == 'rice':
            crop_columns = [col for col in crop_columns if 'RICE' in col.upper()]
        elif self.config.crop_type == 'allgrain':
            # Include grain crops (indices 1-8 in MATLAB, but we use column names)
            grain_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'PMIL_A', 'SMIL_A', 'SORG_A', 'OCER_A']
            crop_columns = [col for col in crop_columns if col in grain_crops]
        
        if not crop_columns:
            raise HPEnvelopeError(f"No crop columns found for crop type: {self.config.crop_type}")
        
        self.logger.info(f"Using crop columns: {crop_columns}")
        
        # Sum across selected crops for each grid cell (MATLAB: sum across columns)
        total_production = production_kcal[crop_columns].sum(axis=1).values
        total_harvest = harvest_km2[crop_columns].sum(axis=1).values
        
        # Calculate yields (production/harvest area) - MATLAB: TotalProduction(:)./TotalHarvest(:)
        yields = np.divide(
            total_production, 
            total_harvest, 
            out=np.zeros_like(total_production), 
            where=(total_harvest != 0)
        )
        
        # Replace inf and -inf with NaN
        yields = np.where(np.isinf(yields), np.nan, yields)
        
        # Create H-P matrix: [harvest_area, production, yield] - MATLAB format
        hp_matrix = np.column_stack([total_harvest, total_production, yields])
        
        # Remove NaNs and zeros following MATLAB code
        # MATLAB: Remove NaNs, then remove zeros
        valid_mask = (~np.isnan(hp_matrix).any(axis=1)) & (hp_matrix[:, 1] > 0) & (hp_matrix[:, 0] > 0)
        hp_matrix = hp_matrix[valid_mask]
        
        self.logger.info(f"Prepared H-P matrix: {len(hp_matrix)} valid grid cells")
        self.logger.info(f"Yield range: {hp_matrix[:, 2].min():.2e} - {hp_matrix[:, 2].max():.2e} kcal/km²")
        
        return hp_matrix
    
    def _sort_by_yield(self, hp_matrix: np.ndarray) -> np.ndarray:
        """
        Sort grid cells by yield (productivity).
        
        Args:
            hp_matrix: Matrix with [harvest_area, production, yield]
        
        Returns:
            Matrix sorted by yield (column 2)
        """
        # Sort by yield (column 2, ascending order)
        sort_indices = np.argsort(hp_matrix[:, 2])
        hp_matrix_sorted = hp_matrix[sort_indices]
        
        self.logger.debug(f"Sorted matrix by yield: range {hp_matrix_sorted[0, 2]:.2e} to {hp_matrix_sorted[-1, 2]:.2e}")
        
        return hp_matrix_sorted
    
    def _get_adaptive_disruption_range(self, hp_matrix_sorted: np.ndarray) -> np.ndarray:
        """
        Get adaptive disruption range based on actual SPAM data extent.
        
        This method determines the appropriate disruption range by:
        1. Using MATLAB-exact ranges as the baseline
        2. Calculating total available harvest area from data
        3. Truncating range to match data availability
        4. Ensuring sufficient points for envelope visualization
        
        Args:
            hp_matrix_sorted: Sorted H-P matrix [harvest_area, production, yield]
        
        Returns:
            Adaptive disruption range array
        """
        # Calculate total harvest area available in the data
        total_harvest_area = hp_matrix_sorted[:, 0].sum()
        max_single_cell = hp_matrix_sorted[:, 0].max()
        
        self.logger.info(f"Data extent - Total harvest: {total_harvest_area:.1f} km², Max cell: {max_single_cell:.1f} km²")
        
        # Get MATLAB-exact disruption ranges
        matlab_ranges = np.array(self.disruption_ranges)
        
        # Method 1: Use MATLAB ranges up to data limit
        valid_matlab_ranges = matlab_ranges[matlab_ranges <= total_harvest_area]
        
        if len(valid_matlab_ranges) >= 20:  # Sufficient points for good envelope
            self.logger.info(f"Using MATLAB ranges up to data limit: {len(valid_matlab_ranges)} points")
            return valid_matlab_ranges
        
        # Method 2: Create adaptive range if MATLAB ranges are insufficient
        self.logger.info(f"MATLAB ranges insufficient ({len(valid_matlab_ranges)} points), creating adaptive range")
        
        # Start with MATLAB base points that fit
        base_points = [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000]
        valid_base = [p for p in base_points if p <= total_harvest_area]
        
        # Add logarithmic progression to fill the range
        if total_harvest_area > 10000:
            # For large datasets, use the MATLAB 2000 km² increment pattern
            additional_points = list(range(10000, int(total_harvest_area), 2000))
            adaptive_range = valid_base + additional_points
        else:
            # For smaller datasets, use finer resolution
            n_additional = max(20 - len(valid_base), 10)  # Ensure at least 20-30 total points
            if total_harvest_area > max(valid_base):
                log_min = np.log10(max(valid_base))
                log_max = np.log10(total_harvest_area)
                log_points = np.linspace(log_min, log_max, n_additional)
                additional_points = 10 ** log_points
                adaptive_range = valid_base + additional_points.tolist()
            else:
                adaptive_range = valid_base
        
        # Remove duplicates and sort
        adaptive_range = sorted(list(set(adaptive_range)))
        
        # Ensure we don't exceed total harvest area
        adaptive_range = [r for r in adaptive_range if r <= total_harvest_area]
        
        self.logger.info(f"Created adaptive disruption range: {len(adaptive_range)} points, "
                        f"range {adaptive_range[0]:.1f} - {adaptive_range[-1]:.1f} km²")
        
        return np.array(adaptive_range)
    
    def get_disruption_range_info(self) -> Dict[str, Any]:
        """
        Get information about disruption range methodology.
        
        Returns:
            Dictionary with disruption range methodology details
        """
        return {
            'methodology': 'Adaptive MATLAB-exact disruption ranges',
            'description': (
                'Uses MATLAB-exact disruption ranges as baseline, then adapts to actual SPAM data extent. '
                'Ensures sufficient envelope points for proper visualization while maintaining MATLAB compatibility.'
            ),
            'matlab_ranges': {
                'allgrain': '1-7M km² (3508 points)',
                'wheat': '1-2.2M km² (1110 points)', 
                'rice': '1-1.4M km² (698 points)'
            },
            'adaptive_logic': [
                '1. Calculate total harvest area from SPAM data',
                '2. Use MATLAB ranges up to data limit if sufficient (≥20 points)',
                '3. Create adaptive range if MATLAB ranges insufficient',
                '4. Maintain MATLAB base points [1,10,50,100,200,400,600,800,1000,4000,6000,8000]',
                '5. Add progression points using MATLAB 2000 km² increment or logarithmic spacing'
            ],
            'parameters_identified': {
                'total_harvest_area': 'Sum of all crop harvest areas in dataset',
                'max_cell_harvest': 'Maximum harvest area in single grid cell',
                'min_envelope_points': '20 points minimum for proper visualization',
                'matlab_increment': '2000 km² for large datasets (>10k km²)',
                'log_spacing': 'For smaller datasets (<10k km²)'
            }
        }
    
    def _calculate_cumulative_sums(self, hp_matrix_sorted: np.ndarray, 
                                  ascending: bool = True) -> np.ndarray:
        """
        Calculate cumulative sums for envelope calculation.
        
        Args:
            hp_matrix_sorted: Sorted H-P matrix
            ascending: If True, start with lowest yields; if False, start with highest yields
        
        Returns:
            Cumulative sum matrix
        """
        if ascending:
            # Small to large (least productive first)
            cumsum_matrix = np.cumsum(hp_matrix_sorted, axis=0)
        else:
            # Large to small (most productive first)
            cumsum_matrix = np.cumsum(hp_matrix_sorted[::-1], axis=0)
        
        return cumsum_matrix
    
    def _calculate_envelope_bounds(self, hp_matrix: np.ndarray, 
                                  disruption_areas: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate upper and lower envelope bounds using exact MATLAB algorithm.
        
        MATLAB algorithm:
        for i_disturb = 1:length(Disturb_HA)
            index_temp = min(find(HPmatrix_cumsum_SmallLarge(:,1)>Disturb_HA(i_disturb)));
            if ~isempty(index_temp)
                LowerBound_HA(i_disturb) = HPmatrix_cumsum_SmallLarge(index_temp,1);
                LowerBound_Prod(i_disturb) = HPmatrix_cumsum_SmallLarge(index_temp,2);
        
        Args:
            hp_matrix: Sorted H-P matrix [harvest_area, production, yield]
            disruption_areas: Array of disruption areas to calculate bounds for
        
        Returns:
            Dictionary with envelope bounds
        """
        # Calculate cumulative sums for both directions (MATLAB algorithm)
        cumsum_small_large = np.cumsum(hp_matrix, axis=0)  # Lower bound (least productive first)
        cumsum_large_small = np.cumsum(np.flipud(hp_matrix), axis=0)  # Upper bound (most productive first)
        
        n_points = len(disruption_areas)
        lower_bound_harvest = np.full(n_points, np.nan)
        lower_bound_production = np.full(n_points, np.nan)
        upper_bound_harvest = np.full(n_points, np.nan)
        upper_bound_production = np.full(n_points, np.nan)
        
        for i, target_area in enumerate(disruption_areas):
            if i % 100 == 0:
                progress = (i / len(disruption_areas)) * 100
                self.logger.info(f"Calculating envelope bounds: {progress:.1f}% ({i}/{len(disruption_areas)}) - "
                               f"Processed disruption area: {target_area:.0f} km²")
            
            # Lower bound (least productive first) - MATLAB algorithm
            # index_temp = min(find(HPmatrix_cumsum_SmallLarge(:,1)>Disturb_HA(i_disturb)));
            indices_lower = np.where(cumsum_small_large[:, 0] > target_area)[0]
            if len(indices_lower) > 0:
                idx = indices_lower[0]
                lower_bound_harvest[i] = cumsum_small_large[idx, 0]
                lower_bound_production[i] = cumsum_small_large[idx, 1]
            
            # Upper bound (most productive first) - MATLAB algorithm
            # index_temp = min(find(HPmatrix_cumsum_LargeSmall(:,1)>Disturb_HA(i_disturb)));
            indices_upper = np.where(cumsum_large_small[:, 0] > target_area)[0]
            if len(indices_upper) > 0:
                idx = indices_upper[0]
                upper_bound_harvest[i] = cumsum_large_small[idx, 0]
                upper_bound_production[i] = cumsum_large_small[idx, 1]
        
        self.logger.info("Calculating envelope bounds: Envelope bounds calculation complete")
        
        # Remove NaN values (MATLAB algorithm)
        # valid_indices = ~isnan(LowerBound_HA) & ~isnan(UpperBound_HA) & ...
        valid_indices = (~np.isnan(lower_bound_harvest) & ~np.isnan(upper_bound_harvest) & 
                        ~np.isnan(lower_bound_production) & ~np.isnan(upper_bound_production))
        
        lower_bound_harvest = lower_bound_harvest[valid_indices]
        lower_bound_production = lower_bound_production[valid_indices]
        upper_bound_harvest = upper_bound_harvest[valid_indices]
        upper_bound_production = upper_bound_production[valid_indices]
        disruption_areas_valid = disruption_areas[valid_indices]
        
        self.logger.info(f"Valid envelope points after NaN removal: {len(lower_bound_harvest)}")
        
        # Ensure bounds meet at convergence point (MATLAB algorithm)
        # convergence_idx = find(abs(LowerBound_Prod - UpperBound_Prod) < 0.01 * max(UpperBound_Prod), 1, 'first');
        if len(upper_bound_production) > 0:
            # Use a more relaxed convergence threshold to avoid premature truncation
            convergence_threshold = 0.05 * np.max(upper_bound_production)  # 5% instead of 1%
            convergence_indices = np.where(np.abs(lower_bound_production - upper_bound_production) < convergence_threshold)[0]
            
            # Only truncate if we have a substantial envelope (at least 10 points) and convergence occurs late
            if len(convergence_indices) > 0 and len(lower_bound_harvest) > 10:
                convergence_idx = convergence_indices[0]
                
                # Only truncate if convergence happens in the latter half of the envelope
                if convergence_idx > len(lower_bound_harvest) // 2:
                    self.logger.info(f"Found convergence at index {convergence_idx} (area: {lower_bound_harvest[convergence_idx]:.0f} km²)")
                    
                    # Truncate bounds at convergence (MATLAB algorithm)
                    lower_bound_harvest = lower_bound_harvest[:convergence_idx+1]
                    lower_bound_production = lower_bound_production[:convergence_idx+1]
                    upper_bound_harvest = upper_bound_harvest[:convergence_idx+1]
                    upper_bound_production = upper_bound_production[:convergence_idx+1]
                    disruption_areas_valid = disruption_areas_valid[:convergence_idx+1]
                else:
                    self.logger.info(f"Convergence found early at index {convergence_idx}, keeping full envelope for better visualization")
            else:
                if len(convergence_indices) > 0:
                    self.logger.info(f"Small envelope ({len(lower_bound_harvest)} points), keeping all points for visualization")
                else:
                    self.logger.info("No convergence found, keeping full envelope")
        
        return {
            'disruption_areas': disruption_areas_valid,
            'lower_bound_harvest': lower_bound_harvest,
            'lower_bound_production': lower_bound_production,
            'upper_bound_harvest': upper_bound_harvest,
            'upper_bound_production': upper_bound_production
        }
    
    def _find_disruption_index(self, cumulative_harvest: np.ndarray, 
                              target_area: float) -> Optional[int]:
        """
        Find the index where cumulative harvest area exceeds target disruption area.
        
        Args:
            cumulative_harvest: Array of cumulative harvest areas
            target_area: Target disruption area
        
        Returns:
            Index where cumulative area first exceeds target, or None if not found
        """
        indices = np.where(cumulative_harvest > target_area)[0]
        return indices[0] if len(indices) > 0 else None
    
    def create_envelope_for_plotting(self, envelope_data: Dict[str, np.ndarray],
                                   loss_factor: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Create envelope data formatted for plotting.
        
        Args:
            envelope_data: Raw envelope data
            loss_factor: Production loss factor (0-1)
        
        Returns:
            Dictionary with plotting-ready envelope data
        """
        # Apply loss factor to production values
        lower_production = envelope_data['lower_bound_production'] * loss_factor
        upper_production = envelope_data['upper_bound_production'] * loss_factor
        
        # Create closed polygon for filling
        # Order: lower_bound (left to right) + upper_bound (right to left)
        harvest_polygon = np.concatenate([
            envelope_data['lower_bound_harvest'],
            envelope_data['upper_bound_harvest'][::-1]
        ])
        
        production_polygon = np.concatenate([
            lower_production,
            upper_production[::-1]
        ])
        
        return {
            'harvest_polygon': harvest_polygon,
            'production_polygon': production_polygon,
            'lower_harvest': envelope_data['lower_bound_harvest'],
            'lower_production': lower_production,
            'upper_harvest': envelope_data['upper_bound_harvest'],
            'upper_production': upper_production,
            'disruption_areas': envelope_data['disruption_areas']
        }
    
    def validate_envelope_data(self, envelope_data: Dict[str, np.ndarray]) -> bool:
        """
        Validate calculated envelope data.
        
        Args:
            envelope_data: Envelope data to validate
        
        Returns:
            True if data is valid
        
        Raises:
            HPEnvelopeError: If validation fails
        """
        required_keys = [
            'disruption_areas', 'lower_bound_harvest', 'lower_bound_production',
            'upper_bound_harvest', 'upper_bound_production'
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in envelope_data:
                raise HPEnvelopeError(f"Missing required key in envelope data: {key}")
        
        # Check array lengths
        n_points = len(envelope_data['disruption_areas'])
        for key in required_keys:
            if len(envelope_data[key]) != n_points:
                raise HPEnvelopeError(f"Array length mismatch for {key}")
        
        # Check that upper bound >= lower bound
        upper_prod = envelope_data['upper_bound_production']
        lower_prod = envelope_data['lower_bound_production']
        
        if np.any(upper_prod < lower_prod):
            self.logger.warning("Found cases where upper bound < lower bound")
        
        # Check for reasonable values
        if np.any(envelope_data['lower_bound_harvest'] < 0):
            raise HPEnvelopeError("Negative harvest area values found")
        
        if np.any(envelope_data['lower_bound_production'] < 0):
            raise HPEnvelopeError("Negative production values found")
        
        self.logger.info("Envelope data validation passed")
        return True
    
    def get_envelope_statistics(self, envelope_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate statistics for envelope data.
        
        Args:
            envelope_data: Envelope data
        
        Returns:
            Dictionary with envelope statistics
        """
        stats = {
            'crop_type': envelope_data.get('crop_type', self.config.crop_type),
            'n_disruption_points': len(envelope_data['disruption_areas']),
            'min_disruption_area': float(envelope_data['disruption_areas'].min()),
            'max_disruption_area': float(envelope_data['disruption_areas'].max()),
            'lower_bound_stats': {
                'min_harvest': float(envelope_data['lower_bound_harvest'].min()),
                'max_harvest': float(envelope_data['lower_bound_harvest'].max()),
                'min_production': float(envelope_data['lower_bound_production'].min()),
                'max_production': float(envelope_data['lower_bound_production'].max())
            },
            'upper_bound_stats': {
                'min_harvest': float(envelope_data['upper_bound_harvest'].min()),
                'max_harvest': float(envelope_data['upper_bound_harvest'].max()),
                'min_production': float(envelope_data['upper_bound_production'].min()),
                'max_production': float(envelope_data['upper_bound_production'].max())
            }
        }
        
        # Calculate envelope width (difference between upper and lower bounds)
        harvest_width = envelope_data['upper_bound_harvest'] - envelope_data['lower_bound_harvest']
        production_width = envelope_data['upper_bound_production'] - envelope_data['lower_bound_production']
        
        stats['envelope_width'] = {
            'avg_harvest_width': float(harvest_width.mean()),
            'max_harvest_width': float(harvest_width.max()),
            'avg_production_width': float(production_width.mean()),
            'max_production_width': float(production_width.max())
        }
        
        return stats
    
    def save_envelope_data(self, envelope_data: Dict[str, np.ndarray], 
                          output_path: str) -> None:
        """
        Save envelope data to CSV file.
        
        Args:
            envelope_data: Envelope data to save
            output_path: Output file path
        """
        # Create DataFrame
        df = pd.DataFrame({
            'disruption_area_km2': envelope_data['disruption_areas'],
            'lower_bound_harvest_km2': envelope_data['lower_bound_harvest'],
            'lower_bound_production_kcal': envelope_data['lower_bound_production'],
            'upper_bound_harvest_km2': envelope_data['upper_bound_harvest'],
            'upper_bound_production_kcal': envelope_data['upper_bound_production']
        })
        
        # Add metadata
        df.attrs['crop_type'] = envelope_data.get('crop_type', self.config.crop_type)
        df.attrs['calculation_date'] = pd.Timestamp.now().isoformat()
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"Envelope data saved to {output_path}")
    
    def create_envelope_report(self, envelope_data: Dict[str, np.ndarray]) -> str:
        """
        Create detailed envelope analysis report.
        
        Args:
            envelope_data: Envelope data
        
        Returns:
            Formatted report string
        """
        stats = self.get_envelope_statistics(envelope_data)
        
        report_lines = [
            "=" * 60,
            "AgriRichter H-P Envelope Analysis Report",
            "=" * 60,
            f"Crop Type: {stats['crop_type']}",
            f"Analysis Points: {stats['n_disruption_points']}",
            "",
            "DISRUPTION AREA RANGE",
            "-" * 22,
            f"Minimum: {stats['min_disruption_area']:,.0f} km²",
            f"Maximum: {stats['max_disruption_area']:,.0f} km²",
            "",
            "LOWER BOUND (Least Productive Cells)",
            "-" * 37,
            f"Harvest Area Range: {stats['lower_bound_stats']['min_harvest']:,.0f} - {stats['lower_bound_stats']['max_harvest']:,.0f} km²",
            f"Production Range: {stats['lower_bound_stats']['min_production']:.2e} - {stats['lower_bound_stats']['max_production']:.2e} kcal",
            "",
            "UPPER BOUND (Most Productive Cells)",
            "-" * 36,
            f"Harvest Area Range: {stats['upper_bound_stats']['min_harvest']:,.0f} - {stats['upper_bound_stats']['max_harvest']:,.0f} km²",
            f"Production Range: {stats['upper_bound_stats']['min_production']:.2e} - {stats['upper_bound_stats']['max_production']:.2e} kcal",
            "",
            "ENVELOPE CHARACTERISTICS",
            "-" * 24,
            f"Average Harvest Width: {stats['envelope_width']['avg_harvest_width']:,.0f} km²",
            f"Maximum Harvest Width: {stats['envelope_width']['max_harvest_width']:,.0f} km²",
            f"Average Production Width: {stats['envelope_width']['avg_production_width']:.2e} kcal",
            f"Maximum Production Width: {stats['envelope_width']['max_production_width']:.2e} kcal",
            ""
        ]
        
        return "\n".join(report_lines)
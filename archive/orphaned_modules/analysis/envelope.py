"""H-P Envelope calculation for AgRichter framework."""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.utils import ProgressReporter
from .convergence_validator import ConvergenceValidator, MathematicalProperties


class HPEnvelopeError(Exception):
    """Exception raised for H-P envelope calculation errors."""
    pass


@dataclass
class EnvelopeData:
    """Enhanced envelope data structure with convergence validation."""
    
    # Core envelope data
    disruption_areas: np.ndarray
    lower_bound_harvest: np.ndarray
    lower_bound_production: np.ndarray
    upper_bound_harvest: np.ndarray
    upper_bound_production: np.ndarray
    
    # Convergence fields
    convergence_point: Tuple[float, float]  # (total_harvest, total_production)
    convergence_validated: bool
    mathematical_properties: Dict[str, bool]
    convergence_statistics: Dict[str, float]
    
    # Metadata
    crop_type: str = "unknown"
    calculation_date: str = ""
    
    def validate_convergence(self) -> bool:
        """Validate that bounds converge properly."""
        return self.convergence_validated and all(self.mathematical_properties.values())
    
    def get_convergence_statistics(self) -> Dict[str, Any]:
        """Get convergence analysis statistics."""
        return {
            'convergence_point': self.convergence_point,
            'convergence_validated': self.convergence_validated,
            'mathematical_properties': self.mathematical_properties,
            'statistics': self.convergence_statistics,
            'envelope_points': len(self.disruption_areas)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            'disruption_areas': self.disruption_areas,
            'lower_bound_harvest': self.lower_bound_harvest,
            'lower_bound_production': self.lower_bound_production,
            'upper_bound_harvest': self.upper_bound_harvest,
            'upper_bound_production': self.upper_bound_production,
            'convergence_point': self.convergence_point,
            'convergence_validated': self.convergence_validated,
            'mathematical_properties': self.mathematical_properties,
            'convergence_statistics': self.convergence_statistics,
            'crop_type': self.crop_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvelopeData':
        """Create EnvelopeData from dictionary."""
        return cls(
            disruption_areas=data['disruption_areas'],
            lower_bound_harvest=data['lower_bound_harvest'],
            lower_bound_production=data['lower_bound_production'],
            upper_bound_harvest=data['upper_bound_harvest'],
            upper_bound_production=data['upper_bound_production'],
            convergence_point=data.get('convergence_point', (0.0, 0.0)),
            convergence_validated=data.get('convergence_validated', False),
            mathematical_properties=data.get('mathematical_properties', {}),
            convergence_statistics=data.get('convergence_statistics', {}),
            crop_type=data.get('crop_type', 'unknown'),
            calculation_date=data.get('calculation_date', '')
        )
    
    def get_envelope_width_at_point(self, harvest_area: float) -> Tuple[float, float]:
        """
        Get envelope width (upper - lower) at a specific harvest area.
        
        Args:
            harvest_area: Harvest area to query
        
        Returns:
            Tuple of (harvest_width, production_width)
        """
        # Find closest point
        idx = np.argmin(np.abs(self.lower_bound_harvest - harvest_area))
        
        harvest_width = self.upper_bound_harvest[idx] - self.lower_bound_harvest[idx]
        production_width = self.upper_bound_production[idx] - self.lower_bound_production[idx]
        
        return harvest_width, production_width
    
    def get_convergence_approach_rate(self) -> float:
        """
        Calculate how quickly the envelope bounds approach each other.
        
        Returns:
            Rate of convergence (width reduction per unit harvest area)
        """
        if len(self.disruption_areas) < 2:
            return 0.0
        
        # Calculate production width at each point
        production_widths = self.upper_bound_production - self.lower_bound_production
        
        # Calculate rate of width reduction
        width_changes = np.diff(production_widths)
        harvest_changes = np.diff(self.lower_bound_harvest)
        
        # Avoid division by zero
        valid_changes = harvest_changes > 0
        if not np.any(valid_changes):
            return 0.0
        
        # Average rate of width reduction
        rates = -width_changes[valid_changes] / harvest_changes[valid_changes]
        return float(np.mean(rates))
    
    def is_mathematically_valid(self) -> bool:
        """Check if envelope satisfies all mathematical requirements."""
        if not self.mathematical_properties:
            return False
        
        required_properties = [
            'starts_at_origin',
            'converges_at_endpoint', 
            'upper_dominates_lower',
            'conservation_satisfied',
            'monotonic_harvest'
        ]
        
        return all(self.mathematical_properties.get(prop, False) for prop in required_properties)


class HPEnvelopeCalculator:
    """Calculator for Harvest-Production (H-P) envelope analysis with multi-tier support."""
    
    def __init__(self, config: Config):
        """
        Initialize H-P envelope calculator.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agrichter.envelope')
        
        # Get disruption ranges for current crop type
        self.disruption_ranges = config.disruption_range
        
        # Initialize convergence validator
        self.convergence_validator = ConvergenceValidator(tolerance=1e-6)
        
        # Initialize multi-tier engine (lazy loading)
        self._multi_tier_engine = None
        
    def calculate_hp_envelope(self, production_kcal: pd.DataFrame, 
                             harvest_km2: pd.DataFrame,
                             tier: str = 'comprehensive') -> Dict[str, np.ndarray]:
        """
        Calculate the Harvest-Production envelope (upper and lower bounds) with optional multi-tier support.
        
        This implements the exact MATLAB algorithm with optional productivity-based filtering:
        HPmatrix = [TotalHarvest(:),TotalProduction(:),TotalProduction(:)./TotalHarvest(:)];
        HPmatrix_sorted = sortrows(HPmatrix,3); % sort by grid-cell *yield*
        HPmatrix_cumsum_SmallLarge = cumsum(HPmatrix_sorted);
        HPmatrix_cumsum_LargeSmall = cumsum(flipud(HPmatrix_sorted));
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data (NOTE: SPAM data is in hectares, converted internally to km²)
            tier: Productivity tier to calculate ('comprehensive', 'commercial', or 'all')
        
        Returns:
            Dictionary with envelope data arrays (or multi-tier results if tier='all')
        """
        try:
            # Handle multi-tier calculation
            if tier == 'all':
                return self.calculate_multi_tier_envelope(production_kcal, harvest_km2)
            elif tier != 'comprehensive':
                return self.calculate_single_tier_envelope(production_kcal, harvest_km2, tier)
            
            # Original comprehensive calculation (backward compatibility)
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
            production_kcal: Production data (NOTE: SPAM data is in metric tons, converted internally to kcal)
            harvest_km2: Harvest area data (NOTE: SPAM data is in hectares, converted internally to km²)
        
        Returns:
            Matrix with [harvest_area_km2, production_kcal, yield_kcal_per_km2] for each grid cell
        """
        from ..core.constants import CALORIC_CONTENT, GRAMS_PER_METRIC_TON, HECTARES_TO_KM2
        
        # Get crop columns based on config
        crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
        
        if not crop_columns:
            raise HPEnvelopeError("No crop columns found in production data")
        
        # Filter to selected crop types based on config
        if self.config.crop_type == 'wheat':
            crop_columns = [col for col in crop_columns if 'WHEA' in col.upper()]
        elif self.config.crop_type == 'maize':  # FIX: This was missing!
            crop_columns = [col for col in crop_columns if 'MAIZ' in col.upper()]
        elif self.config.crop_type == 'rice':
            crop_columns = [col for col in crop_columns if 'RICE' in col.upper()]
        elif self.config.crop_type == 'allgrain':
            # Include grain crops (indices 1-8 in MATLAB, but we use column names)
            grain_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'PMIL_A', 'SMIL_A', 'SORG_A', 'OCER_A']
            crop_columns = [col for col in crop_columns if col in grain_crops]
        
        if not crop_columns:
            raise HPEnvelopeError(f"No crop columns found for crop type: {self.config.crop_type}")
        
        self.logger.info(f"Using crop columns: {crop_columns}")
        
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
        
        # Convert production from metric tons to kcal for each crop separately
        # SPAM production data is in metric tons, we need kcal
        total_production = np.zeros(len(production_kcal))
        for crop_col in crop_columns:
            crop_mt = production_kcal[crop_col].values
            caloric_content = crop_caloric_map.get(crop_col, CALORIC_CONTENT['MixedGrain'])
            crop_kcal = crop_mt * GRAMS_PER_METRIC_TON * caloric_content
            total_production += crop_kcal
        
        # Sum harvest area across selected crops and convert from hectares to km²
        # SPAM harvest data is in hectares, we need km²
        total_harvest_ha = harvest_km2[crop_columns].sum(axis=1).values
        total_harvest = total_harvest_ha * HECTARES_TO_KM2
        
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
        Calculate upper and lower envelope bounds using exact MATLAB algorithm with convergence fixes.
        
        This enhanced version ensures:
        1. Full range calculation (no premature truncation)
        2. Explicit convergence point addition
        3. Mathematical correctness enforcement
        
        Args:
            hp_matrix: Sorted H-P matrix [harvest_area, production, yield]
            disruption_areas: Array of disruption areas to calculate bounds for
        
        Returns:
            Dictionary with envelope bounds including convergence validation
        """
        # Calculate total production and harvest for convergence validation
        total_production = hp_matrix[:, 1].sum()
        total_harvest = hp_matrix[:, 0].sum()
        
        self.logger.info(f"Calculating envelope with convergence validation")
        self.logger.info(f"Total production: {total_production:.2e}, Total harvest: {total_harvest:.2f}")
        
        # Ensure full range calculation by extending disruption areas if needed
        disruption_areas = self._ensure_full_range_calculation(disruption_areas, total_harvest)
        
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
            indices_lower = np.where(cumsum_small_large[:, 0] > target_area)[0]
            if len(indices_lower) > 0:
                idx = indices_lower[0]
                lower_bound_harvest[i] = cumsum_small_large[idx, 0]
                lower_bound_production[i] = cumsum_small_large[idx, 1]
            
            # Upper bound (most productive first) - MATLAB algorithm
            indices_upper = np.where(cumsum_large_small[:, 0] > target_area)[0]
            if len(indices_upper) > 0:
                idx = indices_upper[0]
                upper_bound_harvest[i] = cumsum_large_small[idx, 0]
                upper_bound_production[i] = cumsum_large_small[idx, 1]
        
        self.logger.info("Calculating envelope bounds: Envelope bounds calculation complete")
        
        # Remove NaN values (MATLAB algorithm)
        valid_indices = (~np.isnan(lower_bound_harvest) & ~np.isnan(upper_bound_harvest) & 
                        ~np.isnan(lower_bound_production) & ~np.isnan(upper_bound_production))
        
        lower_bound_harvest = lower_bound_harvest[valid_indices]
        lower_bound_production = lower_bound_production[valid_indices]
        upper_bound_harvest = upper_bound_harvest[valid_indices]
        upper_bound_production = upper_bound_production[valid_indices]
        disruption_areas_valid = disruption_areas[valid_indices]
        
        self.logger.info(f"Valid envelope points after NaN removal: {len(lower_bound_harvest)}")
        
        # Create initial envelope data
        envelope_data = {
            'disruption_areas': disruption_areas_valid,
            'lower_bound_harvest': lower_bound_harvest,
            'lower_bound_production': lower_bound_production,
            'upper_bound_harvest': upper_bound_harvest,
            'upper_bound_production': upper_bound_production
        }
        
        # Add explicit convergence point if needed
        envelope_data = self._add_convergence_point(envelope_data, total_production, total_harvest)
        
        # Validate and enforce mathematical correctness
        validation_result = self.convergence_validator.validate_mathematical_properties(
            envelope_data, total_production, total_harvest
        )
        
        if not validation_result.is_valid:
            self.logger.warning("Envelope failed mathematical validation, attempting correction")
            envelope_data = self.convergence_validator.enforce_convergence(
                envelope_data, total_production, total_harvest
            )
        
        # Add convergence metadata
        envelope_data['convergence_validated'] = validation_result.is_valid
        envelope_data['mathematical_properties'] = validation_result.properties
        envelope_data['convergence_statistics'] = validation_result.statistics
        
        return envelope_data
    
    def create_envelope_data_object(self, envelope_dict: Dict[str, Any]) -> EnvelopeData:
        """
        Create an EnvelopeData object from calculation results.
        
        Args:
            envelope_dict: Dictionary with envelope calculation results
        
        Returns:
            EnvelopeData object with validation methods
        """
        return EnvelopeData(
            disruption_areas=envelope_dict['disruption_areas'],
            lower_bound_harvest=envelope_dict['lower_bound_harvest'],
            lower_bound_production=envelope_dict['lower_bound_production'],
            upper_bound_harvest=envelope_dict['upper_bound_harvest'],
            upper_bound_production=envelope_dict['upper_bound_production'],
            convergence_point=envelope_dict.get('convergence_point', (0.0, 0.0)),
            convergence_validated=envelope_dict.get('convergence_validated', False),
            mathematical_properties=envelope_dict.get('mathematical_properties', {}),
            convergence_statistics=envelope_dict.get('convergence_statistics', {}),
            crop_type=envelope_dict.get('crop_type', self.config.crop_type),
            calculation_date=pd.Timestamp.now().isoformat()
        )
    
    def _ensure_full_range_calculation(self, disruption_areas: np.ndarray, 
                                      total_harvest: float) -> np.ndarray:
        """
        Ensure calculation covers full harvest area range to prevent premature truncation.
        
        This method extends the disruption areas to include the total harvest area
        if it's not already covered, ensuring the envelope reaches the convergence point.
        
        Args:
            disruption_areas: Original disruption areas
            total_harvest: Total harvest area from all cells
        
        Returns:
            Extended disruption areas ensuring full range coverage
        """
        max_disruption = np.max(disruption_areas)
        
        if max_disruption < total_harvest * 0.99:  # If we don't reach at least 99% of total
            self.logger.info(f"Extending disruption range from {max_disruption:.1f} to {total_harvest:.1f} km²")
            
            # Add points leading up to total harvest area
            additional_points = []
            
            # Add intermediate points if there's a large gap
            if total_harvest > max_disruption * 1.5:
                # Add logarithmic progression to fill the gap
                n_fill = min(10, int(np.log10(total_harvest / max_disruption) * 5))
                fill_points = np.logspace(
                    np.log10(max_disruption * 1.1), 
                    np.log10(total_harvest * 0.95), 
                    n_fill
                )
                additional_points.extend(fill_points)
            
            # Always add the total harvest area as the final point
            additional_points.append(total_harvest)
            
            # Combine and sort
            extended_areas = np.concatenate([disruption_areas, additional_points])
            extended_areas = np.unique(extended_areas)  # Remove duplicates
            extended_areas = np.sort(extended_areas)
            
            self.logger.info(f"Extended disruption areas: {len(extended_areas)} points "
                           f"(added {len(extended_areas) - len(disruption_areas)} points)")
            
            return extended_areas
        
        return disruption_areas
    
    def _add_convergence_point(self, envelope_data: Dict[str, np.ndarray], 
                              total_production: float, 
                              total_harvest: float) -> Dict[str, np.ndarray]:
        """
        Explicitly add convergence point at (total_harvest, total_production).
        
        This method ensures that the envelope bounds converge at the mathematical
        endpoint where both bounds must equal total production.
        
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
            
            self.logger.info(f"Updated existing point to convergence: ({total_harvest:.1f}, {total_production:.2e})")
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
            
            self.logger.info(f"Added explicit convergence point: ({total_harvest:.1f}, {total_production:.2e})")
        
        # Store convergence point for reference
        envelope_data['convergence_point'] = (total_harvest, total_production)
        
        return envelope_data
    
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
        Validate calculated envelope data including convergence properties.
        
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
        
        violations = np.sum(upper_prod < lower_prod)
        if violations > 0:
            self.logger.warning(f"Found {violations} cases where upper bound < lower bound")
        
        # Check for reasonable values
        if np.any(envelope_data['lower_bound_harvest'] < 0):
            raise HPEnvelopeError("Negative harvest area values found")
        
        if np.any(envelope_data['lower_bound_production'] < 0):
            raise HPEnvelopeError("Negative production values found")
        
        # Validate convergence properties if available
        if 'convergence_validated' in envelope_data:
            convergence_status = envelope_data['convergence_validated']
            self.logger.info(f"Convergence validation status: {'PASSED' if convergence_status else 'FAILED'}")
            
            if 'mathematical_properties' in envelope_data:
                props = envelope_data['mathematical_properties']
                failed_props = [k for k, v in props.items() if not v]
                if failed_props:
                    self.logger.warning(f"Failed mathematical properties: {failed_props}")
        
        # Check for convergence point
        if 'convergence_point' in envelope_data:
            conv_point = envelope_data['convergence_point']
            self.logger.info(f"Convergence point: ({conv_point[0]:.1f} km², {conv_point[1]:.2e} kcal)")
        
        self.logger.info("Envelope data validation passed")
        return True
    
    def get_envelope_statistics(self, envelope_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate statistics for envelope data.
        
        Args:
            envelope_data: Envelope data (dict or EnvelopeData)
        
        Returns:
            Dictionary with envelope statistics
        """
        # Handle both dict and EnvelopeData formats
        if isinstance(envelope_data, EnvelopeData):
            data_dict = envelope_data.to_dict()
        else:
            data_dict = envelope_data
        
        stats = {
            'crop_type': data_dict.get('crop_type', self.config.crop_type),
            'n_disruption_points': len(data_dict['disruption_areas']),
            'min_disruption_area': float(data_dict['disruption_areas'].min()),
            'max_disruption_area': float(data_dict['disruption_areas'].max()),
            'lower_bound_stats': {
                'min_harvest': float(data_dict['lower_bound_harvest'].min()),
                'max_harvest': float(data_dict['lower_bound_harvest'].max()),
                'min_production': float(data_dict['lower_bound_production'].min()),
                'max_production': float(data_dict['lower_bound_production'].max())
            },
            'upper_bound_stats': {
                'min_harvest': float(data_dict['upper_bound_harvest'].min()),
                'max_harvest': float(data_dict['upper_bound_harvest'].max()),
                'min_production': float(data_dict['upper_bound_production'].min()),
                'max_production': float(data_dict['upper_bound_production'].max())
            }
        }
        
        # Calculate envelope width (difference between upper and lower bounds)
        harvest_width = data_dict['upper_bound_harvest'] - data_dict['lower_bound_harvest']
        production_width = data_dict['upper_bound_production'] - data_dict['lower_bound_production']
        
        stats['envelope_width'] = {
            'avg_harvest_width': float(harvest_width.mean()),
            'max_harvest_width': float(harvest_width.max()),
            'avg_production_width': float(production_width.mean()),
            'max_production_width': float(production_width.max())
        }
        
        # Add convergence information if available
        if 'convergence_validated' in data_dict:
            stats['convergence_info'] = {
                'convergence_validated': data_dict['convergence_validated'],
                'convergence_point': data_dict.get('convergence_point', (0.0, 0.0)),
                'mathematical_properties': data_dict.get('mathematical_properties', {}),
                'convergence_statistics': data_dict.get('convergence_statistics', {})
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
            "AgRichter H-P Envelope Analysis Report",
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
    
    # ========================================================================
    # MULTI-TIER ENVELOPE SUPPORT (Task 1.3)
    # ========================================================================
    
    def calculate_multi_tier_envelope(self, production_kcal: pd.DataFrame, 
                                    harvest_km2: pd.DataFrame,
                                    tiers: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate envelope bounds for multiple productivity tiers.
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
            tiers: List of tier names to calculate (default: all available tiers)
        
        Returns:
            Dictionary mapping tier names to envelope data dictionaries
        """
        # Initialize multi-tier engine if needed
        if self._multi_tier_engine is None:
            self._initialize_multi_tier_engine()
        
        # Calculate multi-tier results
        multi_tier_results = self._multi_tier_engine.calculate_multi_tier_envelope(
            production_kcal, harvest_km2, tiers
        )
        
        # Convert to dictionary format for backward compatibility
        tier_envelopes = {}
        for tier_name, envelope_data in multi_tier_results.tier_results.items():
            tier_envelopes[tier_name] = envelope_data.to_dict()
        
        # Add width analysis metadata
        tier_envelopes['_width_analysis'] = multi_tier_results.width_analysis
        tier_envelopes['_base_statistics'] = multi_tier_results.base_statistics
        
        return tier_envelopes
    
    def calculate_single_tier_envelope(self, production_kcal: pd.DataFrame, 
                                     harvest_km2: pd.DataFrame,
                                     tier: str) -> Dict[str, np.ndarray]:
        """
        Calculate envelope bounds for a single productivity tier.
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
            tier: Tier name ('comprehensive', 'commercial')
        
        Returns:
            Dictionary with envelope data arrays
        """
        # Initialize multi-tier engine if needed
        if self._multi_tier_engine is None:
            self._initialize_multi_tier_engine()
        
        # Calculate single tier
        multi_tier_results = self._multi_tier_engine.calculate_multi_tier_envelope(
            production_kcal, harvest_km2, [tier]
        )
        
        # Return the single tier result
        if tier in multi_tier_results.tier_results:
            return multi_tier_results.tier_results[tier].to_dict()
        else:
            raise HPEnvelopeError(f"Tier '{tier}' not found in results")
    
    def get_available_tiers(self) -> Dict[str, str]:
        """
        Get available productivity tiers and their descriptions.
        
        Returns:
            Dictionary mapping tier names to descriptions
        """
        # Initialize multi-tier engine if needed
        if self._multi_tier_engine is None:
            self._initialize_multi_tier_engine()
        
        tier_info = self._multi_tier_engine.get_tier_info()
        return {name: info['description'] for name, info in tier_info.items()}
    
    def get_tier_selection_guide(self) -> Dict[str, Dict[str, Any]]:
        """
        Get guidance for selecting appropriate tiers for different use cases.
        
        Returns:
            Dictionary with tier selection guidance
        """
        # Initialize multi-tier engine if needed
        if self._multi_tier_engine is None:
            self._initialize_multi_tier_engine()
        
        return self._multi_tier_engine.get_tier_info()
    
    def compare_tier_widths(self, production_kcal: pd.DataFrame, 
                           harvest_km2: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare envelope widths across different tiers.
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
        
        Returns:
            Dictionary with width comparison analysis
        """
        # Calculate all tiers
        multi_tier_results = self.calculate_multi_tier_envelope(production_kcal, harvest_km2)
        
        # Extract width analysis
        width_analysis = multi_tier_results.get('_width_analysis', {})
        
        # Add tier descriptions
        tier_info = self.get_tier_selection_guide()
        
        comparison = {
            'width_analysis': width_analysis,
            'tier_descriptions': {name: info['description'] for name, info in tier_info.items()},
            'policy_applications': {name: info['policy_applications'] for name, info in tier_info.items()},
            'target_users': {name: info['target_users'] for name, info in tier_info.items()}
        }
        
        return comparison
    
    def _initialize_multi_tier_engine(self):
        """Initialize the multi-tier engine with current configuration."""
        try:
            from .multi_tier_envelope import MultiTierEnvelopeEngine
            from ..validation.spam_data_filter import SPAMDataFilter
            
            # Create SPAM filter with standard preset
            spam_filter = SPAMDataFilter(preset='standard')
            
            # Initialize multi-tier engine
            self._multi_tier_engine = MultiTierEnvelopeEngine(self.config, spam_filter)
            
            self.logger.info("Multi-tier envelope engine initialized")
            
        except ImportError as e:
            raise HPEnvelopeError(
                f"Multi-tier envelope functionality requires additional components: {e}"
            )
        except Exception as e:
            raise HPEnvelopeError(
                f"Failed to initialize multi-tier engine: {e}"
            )
"""H-P Envelope calculation V2 using robust envelope builder."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.utils import ProgressReporter
from .envelope_builder import build_envelope
from .convergence_validator import ConvergenceValidator
from .envelope import EnvelopeData


class HPEnvelopeError(Exception):
    """Exception raised for H-P envelope calculation errors."""
    pass


class HPEnvelopeCalculatorV2:
    """
    Calculator for Harvest-Production (H-P) envelope analysis using robust envelope builder.
    
    This is a parallel implementation that uses the new mathematically robust
    envelope_builder module while maintaining API compatibility with the original
    HPEnvelopeCalculator.
    """
    
    def __init__(self, config: Config):
        """
        Initialize H-P envelope calculator V2.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agrichter.envelope_v2')
        
        # Get disruption ranges for current crop type
        self.disruption_ranges = config.disruption_range
        
        # Initialize convergence validator
        self.convergence_validator = ConvergenceValidator(tolerance=1e-6)
        
    def calculate_hp_envelope(self, production_kcal: pd.DataFrame, 
                             harvest_km2: pd.DataFrame,
                             yield_df: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """
        Calculate the Harvest-Production envelope (upper and lower bounds).
        
        This implementation uses the robust envelope_builder module which provides:
        - Mathematical rigor (monotonicity, dominance, conservation)
        - Comprehensive validation and QA reporting
        - Explicit unit tracking
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data (NOTE: SPAM data is in hectares, converted internally to km²)
            yield_df: Optional yield data in tons/ha (if provided, will be validated against P/H)
        
        Returns:
            Dictionary with envelope data arrays (compatible with original format)
        """
        try:
            self.logger.info("Starting H-P envelope calculation (V2 - robust builder)")
            
            # Step 1: Prepare data (convert to per-cell arrays)
            P_kcal, H_ha = self._prepare_cell_data(production_kcal, harvest_km2)
            
            # Step 2: Prepare yield data if provided (must align with production/harvest)
            Y_data = None
            if yield_df is not None:
                self.logger.info("Aligning SPAM yield data with production/harvest cells")
                # Merge yield data with production data to ensure alignment
                # Use cell5m as the key to match cells
                prod_cells = list(production_kcal['cell5m']) if 'cell5m' in production_kcal.columns else list(production_kcal.index)
                yield_aligned = yield_df[yield_df['cell5m'].isin(prod_cells)] if 'cell5m' in yield_df.columns else yield_df.loc[yield_df.index.isin(prod_cells)]
                
                if len(yield_aligned) == len(production_kcal):
                    Y_data, _ = self._prepare_cell_data(yield_aligned, harvest_km2)
                    self.logger.info(f"Using {len(yield_aligned)} aligned yield values")
                else:
                    self.logger.warning(f"Yield data alignment failed ({len(yield_aligned)} vs {len(production_kcal)} cells), computing from P/H")
                    Y_data = None
            
            if Y_data is None:
                self.logger.info("Computing yield from P/H")
            
            # Step 3: Build envelope using robust builder
            # Note: envelope_builder expects P in metric tons, but we'll adapt it
            # to work with kcal by treating kcal as the production unit.
            #
            # METHOD EXPLANATION (SORTING LOGIC):
            # The core of the envelope calculation relies on a greedy selection algorithm that sorts grid cells 
            # by yield density ($Y_j = P_j / A_j$).
            #
            # 1. Upper Bound (Worst-case Scenario): 
            #    - Disruption is assumed to preferentially affect the most productive agricultural land.
            #    - Cells are sorted by yield in descending order: $Y_1 \ge Y_2 \ge \dots \ge Y_n$.
            #    - We accumulate cells until the target harvest area magnitude is reached.
            #    - Mathematically: $\overline{L}(A_{target}) \approx \sum_{i=1}^{k} P_i$ where $\sum_{i=1}^{k} A_i \approx A_{target}$.
            #
            # 2. Lower Bound (Best-case Scenario):
            #    - Disruption is assumed to affect the least productive land first (e.g., marginal lands).
            #    - Cells are sorted by yield in ascending order: $Y_1 \le Y_2 \le \dots \le Y_n$.
            #    - We accumulate cells similarly.
            #
            # This approach rigorously bounds all possible spatial configurations of a disruption 
            # of a given magnitude $M_D$ (where $A_{target} = 10^{M_D}$).
            #
            # Use 1000 points for publication-quality smooth curves
            envelope_result = build_envelope(
                P_mt=P_kcal,  # Using kcal as production unit
                H_ha=H_ha,
                Y_mt_per_ha=Y_data,  # Use actual yield if provided
                tol=0.01,  # Tighter tolerance for higher precision
                interpolate=True,
                n_points=1000  # Increased from default for smoother publication figures
            )
            
            # Step 4: Convert to original format for compatibility
            envelope_data = self._convert_to_original_format(envelope_result)
            
            # Step 5: Apply convergence validation and enforcement
            total_production = P_kcal.sum()
            # CRITICAL FIX: Convert total harvest to km² for convergence check
            # Envelope data is in km², so validation target must match
            total_harvest_km2 = H_ha.sum() * 0.01
            
            envelope_data = self._apply_convergence_validation(
                envelope_data, total_production, total_harvest_km2
            )
            
            self.logger.info("H-P envelope calculation (V2) completed successfully")
            self.logger.info(f"QA Summary: {envelope_result['summary'].get('qa_status', 'UNKNOWN')}")
            
            return envelope_data
            
        except Exception as e:
            raise HPEnvelopeError(f"Failed to calculate H-P envelope (V2): {str(e)}")
    
    def _prepare_cell_data(self, production_kcal: pd.DataFrame, 
                          harvest_km2: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare per-cell production and harvest area arrays.
        
        Args:
            production_kcal: Production data (NOTE: SPAM data is in metric tons, converted internally to kcal)
            harvest_km2: Harvest area data (NOTE: SPAM data is in hectares, converted internally to km²)
        
        Returns:
            Tuple of (production_kcal_array, harvest_ha_array)
        """
        from ..core.constants import CALORIC_CONTENT, GRAMS_PER_METRIC_TON, HECTARES_TO_KM2
        
        # Get crop columns based on config
        crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
        
        if not crop_columns:
            raise HPEnvelopeError("No crop columns found in production data")
        
        # Filter to selected crop types based on config
        if self.config.crop_type == 'wheat':
            crop_columns = [col for col in crop_columns if 'WHEA' in col.upper()]
        elif self.config.crop_type == 'rice':
            crop_columns = [col for col in crop_columns if 'RICE' in col.upper()]
        elif self.config.crop_type == 'maize':
            crop_columns = [col for col in crop_columns if 'MAIZ' in col.upper()]
        elif self.config.crop_type == 'allgrain':
            # Include grain crops (updated for SPAM 2020)
            grain_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'MILL_A', 'SORG_A', 'OCER_A']
            crop_columns = [col for col in crop_columns if col in grain_crops]
        
        if not crop_columns:
            raise HPEnvelopeError(f"No crop columns found for crop type: {self.config.crop_type}")
        
        self.logger.info(f"Using crop columns: {crop_columns}")
        
        # Map crop columns to caloric content
        crop_caloric_map = {
            'BARL_A': CALORIC_CONTENT['Barley'],
            'MAIZ_A': CALORIC_CONTENT['Corn'],
            'MILL_A': CALORIC_CONTENT['Millet'],
            'OCER_A': CALORIC_CONTENT['MixedGrain'],
            'PMIL_A': CALORIC_CONTENT['Millet'],
            'RICE_A': CALORIC_CONTENT['Rice'],
            'SMIL_A': CALORIC_CONTENT['Millet'],
            'SORG_A': CALORIC_CONTENT['Sorghum'],
            'WHEA_A': CALORIC_CONTENT['Wheat'],
        }
        
        # Convert production from metric tons to kcal for each crop separately
        total_production = np.zeros(len(production_kcal))
        for crop_col in crop_columns:
            crop_mt = production_kcal[crop_col].values
            caloric_content = crop_caloric_map.get(crop_col, CALORIC_CONTENT['MixedGrain'])
            crop_kcal = crop_mt * GRAMS_PER_METRIC_TON * caloric_content
            total_production += crop_kcal
        
        # Sum harvest area across selected crops (keep in hectares for envelope_builder)
        total_harvest_ha = harvest_km2[crop_columns].sum(axis=1).values
        
        # Pre-filter: Remove cells with zero or negative values
        # (envelope_builder will do this too, but doing it here provides better logging)
        valid_mask = (total_production > 0) & (total_harvest_ha > 0)
        n_dropped = np.sum(~valid_mask)
        
        if n_dropped > 0:
            dropped_production = total_production[~valid_mask].sum()
            dropped_harvest = total_harvest_ha[~valid_mask].sum()
            self.logger.info(
                f"Pre-filtering: Dropping {n_dropped} cells with zero/negative values "
                f"(P={dropped_production:.2e} kcal, H={dropped_harvest:.2f} ha)"
            )
            total_production = total_production[valid_mask]
            total_harvest_ha = total_harvest_ha[valid_mask]
        
        self.logger.info(f"Prepared cell data: {len(total_production)} cells")
        self.logger.info(f"Total production: {total_production.sum():.2e} kcal")
        self.logger.info(f"Total harvest: {total_harvest_ha.sum():.2f} ha")
        
        return total_production, total_harvest_ha
    
    def _convert_to_original_format(self, envelope_result: Dict) -> Dict[str, np.ndarray]:
        """
        Convert envelope_builder result to original HPEnvelopeCalculator format.
        
        Args:
            envelope_result: Result from build_envelope()
        
        Returns:
            Dictionary in original format with keys:
            - disruption_areas
            - lower_bound_harvest
            - lower_bound_production
            - upper_bound_harvest
            - upper_bound_production
        """
        # Use interpolated data if available, otherwise use discrete data
        if 'interpolated' in envelope_result:
            interp = envelope_result['interpolated']
            Hq_km2 = interp['Hq_km2']
            P_low = interp['P_low_interp']
            P_up = interp['P_up_interp']
        else:
            # Use discrete data
            lower = envelope_result['lower']
            upper = envelope_result['upper']
            Hq_km2 = lower['cum_H_km2']
            P_low = lower['cum_P_mt']  # Actually kcal in our case
            P_up = upper['cum_P_mt']  # Actually kcal in our case
        
        # Create result in original format
        result = {
            'disruption_areas': Hq_km2.copy(),
            'lower_bound_harvest': Hq_km2.copy(),
            'lower_bound_production': P_low.copy(),
            'upper_bound_harvest': Hq_km2.copy(),
            'upper_bound_production': P_up.copy(),
            'crop_type': self.config.crop_type,
            'v2_summary': envelope_result['summary']  # Include V2 QA data
        }
        
        return result
    
    def _apply_convergence_validation(self, envelope_data: Dict[str, np.ndarray],
                                    total_production: float, 
                                    total_harvest: float) -> Dict[str, np.ndarray]:
        """
        Apply convergence validation and enforcement to V2 envelope results.
        
        Args:
            envelope_data: Envelope data from envelope_builder
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
        
        Returns:
            Envelope data with convergence validation applied
        """
        self.logger.info("Applying convergence validation to V2 envelope")
        
        # Validate mathematical properties
        validation_result = self.convergence_validator.validate_mathematical_properties(
            envelope_data, total_production, total_harvest
        )
        
        if not validation_result.is_valid:
            self.logger.warning("V2 envelope failed mathematical validation, attempting correction")
            
            # Apply convergence enforcement
            envelope_data = self.convergence_validator.enforce_convergence(
                envelope_data, total_production, total_harvest
            )
            
            # Re-validate after correction
            validation_result = self.convergence_validator.validate_mathematical_properties(
                envelope_data, total_production, total_harvest
            )
        
        # Add convergence metadata
        envelope_data['convergence_validated'] = validation_result.is_valid
        envelope_data['mathematical_properties'] = validation_result.properties
        envelope_data['convergence_statistics'] = validation_result.statistics
        envelope_data['convergence_point'] = (total_harvest, total_production)
        
        # Log convergence status
        if validation_result.is_valid:
            self.logger.info("V2 envelope convergence validation: PASSED")
        else:
            self.logger.warning("V2 envelope convergence validation: FAILED")
            for error in validation_result.errors:
                self.logger.warning(f"  - {error}")
        
        return envelope_data
    
    def create_envelope_data_object(self, envelope_dict: Dict[str, Any]) -> EnvelopeData:
        """
        Create an EnvelopeData object from V2 calculation results.
        
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
        
        violations = np.sum(upper_prod < lower_prod)
        if violations > 0:
            self.logger.warning(f"Found {violations} cases where upper bound < lower bound")
            # This should not happen with V2 builder due to clipping
            if 'v2_summary' in envelope_data:
                self.logger.info("V2 builder should have prevented this via clipping")
        
        # Validate convergence properties if available
        if 'convergence_validated' in envelope_data:
            convergence_status = envelope_data['convergence_validated']
            self.logger.info(f"V2 convergence validation status: {'PASSED' if convergence_status else 'FAILED'}")
            
            if 'mathematical_properties' in envelope_data:
                props = envelope_data['mathematical_properties']
                failed_props = [k for k, v in props.items() if not v]
                if failed_props:
                    self.logger.warning(f"Failed mathematical properties: {failed_props}")
        
        # Check for convergence point
        if 'convergence_point' in envelope_data:
            conv_point = envelope_data['convergence_point']
            self.logger.info(f"V2 convergence point: ({conv_point[0]:.1f} km², {conv_point[1]:.2e} kcal)")
        
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
            'calculator_version': 'V2 (robust builder)',
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
        
        # Calculate envelope width
        harvest_width = envelope_data['upper_bound_harvest'] - envelope_data['lower_bound_harvest']
        production_width = envelope_data['upper_bound_production'] - envelope_data['lower_bound_production']
        
        stats['envelope_width'] = {
            'avg_harvest_width': float(harvest_width.mean()),
            'max_harvest_width': float(harvest_width.max()),
            'avg_production_width': float(production_width.mean()),
            'max_production_width': float(production_width.max())
        }
        
        # Include V2 QA summary if available
        if 'v2_summary' in envelope_data:
            stats['v2_qa_summary'] = envelope_data['v2_summary']
        
        # Add convergence information if available
        if 'convergence_validated' in envelope_data:
            stats['convergence_info'] = {
                'convergence_validated': envelope_data['convergence_validated'],
                'convergence_point': envelope_data.get('convergence_point', (0.0, 0.0)),
                'mathematical_properties': envelope_data.get('mathematical_properties', {}),
                'convergence_statistics': envelope_data.get('convergence_statistics', {})
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
        df.attrs['calculator_version'] = 'V2 (robust builder)'
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
            "AgRichter H-P Envelope Analysis Report (V2)",
            "=" * 60,
            f"Calculator Version: {stats['calculator_version']}",
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
        
        # Add V2 QA summary if available
        if 'v2_qa_summary' in stats:
            report_lines.extend([
                "V2 QUALITY ASSURANCE",
                "-" * 20,
                f"QA Status: {stats['v2_qa_summary'].get('qa_status', 'UNKNOWN')}",
                f"Valid Cells: {stats['v2_qa_summary'].get('n_valid_cells', 'N/A')}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def get_disruption_range_info(self) -> Dict[str, Any]:
        """
        Get information about disruption range methodology.
        
        Returns:
            Dictionary with disruption range methodology details
        """
        return {
            'methodology': 'V2 Robust Builder with Adaptive Interpolation',
            'description': (
                'Uses robust envelope_builder module with mathematical guarantees. '
                'Interpolates discrete envelope onto unified query grid for plotting.'
            ),
            'features': [
                'Monotonicity guaranteed',
                'Dominance constraint enforced (upper >= lower)',
                'Conservation of totals verified',
                'Comprehensive QA validation',
                'Explicit unit tracking'
            ]
        }

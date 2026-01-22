"""Unit conversion utilities for AgRichter framework."""

import logging
from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.constants import GRAMS_PER_METRIC_TON, HECTARES_TO_KM2


class UnitConversionError(Exception):
    """Exception raised for unit conversion errors."""
    pass


class UnitConverter:
    """Handles all unit conversions for AgRichter analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize unit converter.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agrichter.converter')
        
        # Get conversion constants
        self.conversions = self.config.get_unit_conversions()
        self.caloric_content = self.config.get_caloric_content()
        
        # Validation ranges
        self.validation_ranges = {
            'production_mt_min': 0.0,
            'production_mt_max': 1e8,  # 100 million metric tons per cell (extreme upper bound)
            'harvest_ha_min': 0.0,
            'harvest_ha_max': 1e6,    # 1 million hectares per cell (extreme upper bound)
            'yield_min': 0.001,       # 1 kg per hectare (very low but possible)
            'yield_max': 100.0        # 100 metric tons per hectare (very high but possible)
        }
    
    def convert_production_to_kcal(self, production_mt: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Convert production from metric tons to kilocalories.
        
        Args:
            production_mt: Production in metric tons
        
        Returns:
            Production in kilocalories
        
        Raises:
            UnitConversionError: If conversion fails or values are invalid
        """
        try:
            # Validate input
            self._validate_production_input(production_mt)
            
            # Convert: metric tons → grams → kcal
            # 1 metric ton = 1,000,000 grams
            # kcal = grams × caloric_content (kcal/g)
            
            production_grams = production_mt * self.conversions['grams_per_metric_ton']
            production_kcal = production_grams * self.caloric_content
            
            self.logger.debug(f"Converted production: {self._get_conversion_summary(production_mt, production_kcal, 'MT', 'kcal')}")
            
            return production_kcal
            
        except Exception as e:
            raise UnitConversionError(f"Failed to convert production to kcal: {str(e)}")
    
    def convert_harvest_area_to_km2(self, harvest_ha: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Convert harvest area from hectares to square kilometers.
        
        Args:
            harvest_ha: Harvest area in hectares
        
        Returns:
            Harvest area in square kilometers
        
        Raises:
            UnitConversionError: If conversion fails or values are invalid
        """
        try:
            # Validate input
            self._validate_harvest_area_input(harvest_ha)
            
            # Convert: hectares → km²
            # 1 hectare = 0.01 km²
            harvest_km2 = harvest_ha * self.conversions['hectares_to_km2']
            
            self.logger.debug(f"Converted harvest area: {self._get_conversion_summary(harvest_ha, harvest_km2, 'ha', 'km²')}")
            
            return harvest_km2
            
        except Exception as e:
            raise UnitConversionError(f"Failed to convert harvest area to km²: {str(e)}")
    
    def calculate_yield(self, production_mt: Union[pd.Series, np.ndarray, float], 
                       harvest_ha: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Calculate yield (production per unit area).
        
        Args:
            production_mt: Production in metric tons
            harvest_ha: Harvest area in hectares
        
        Returns:
            Yield in metric tons per hectare
        
        Raises:
            UnitConversionError: If calculation fails or values are invalid
        """
        try:
            # Validate inputs
            self._validate_production_input(production_mt)
            self._validate_harvest_area_input(harvest_ha)
            
            # Handle division by zero
            if isinstance(harvest_ha, (pd.Series, np.ndarray)):
                # For arrays/series, use numpy's divide with where clause
                yield_mt_ha = np.divide(
                    production_mt, 
                    harvest_ha, 
                    out=np.zeros_like(production_mt, dtype=float), 
                    where=(harvest_ha != 0)
                )
            else:
                # For scalars
                if harvest_ha == 0:
                    yield_mt_ha = 0.0
                else:
                    yield_mt_ha = production_mt / harvest_ha
            
            # Validate yield ranges
            self._validate_yield_output(yield_mt_ha)
            
            self.logger.debug(f"Calculated yield: {self._get_yield_summary(production_mt, harvest_ha, yield_mt_ha)}")
            
            return yield_mt_ha
            
        except Exception as e:
            raise UnitConversionError(f"Failed to calculate yield: {str(e)}")
    
    def convert_magnitude_to_harvest_area(self, magnitude: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Convert AgRichter magnitude back to harvest area in km².
        
        Args:
            magnitude: AgRichter magnitude (log10 scale)
        
        Returns:
            Harvest area in km²
        """
        try:
            # M_D = log10(harvest_area_km2)
            # Therefore: harvest_area_km2 = 10^M_D
            harvest_area_km2 = np.power(10, magnitude)
            
            self.logger.debug(f"Converted magnitude to harvest area: M={magnitude} → {harvest_area_km2} km²")
            
            return harvest_area_km2
            
        except Exception as e:
            raise UnitConversionError(f"Failed to convert magnitude to harvest area: {str(e)}")
    
    def convert_harvest_area_to_magnitude(self, harvest_area_km2: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
        """
        Convert harvest area in km² to AgRichter magnitude.
        
        Args:
            harvest_area_km2: Harvest area in km²
        
        Returns:
            AgRichter magnitude (log10 scale)
        """
        try:
            # Validate input (must be positive for log)
            if isinstance(harvest_area_km2, (pd.Series, np.ndarray)):
                if (harvest_area_km2 <= 0).any():
                    self.logger.warning("Found zero or negative harvest areas, setting to NaN for magnitude calculation")
                    harvest_area_km2 = harvest_area_km2.replace(0, np.nan)
            else:
                if harvest_area_km2 <= 0:
                    return np.nan
            
            # M_D = log10(harvest_area_km2)
            magnitude = np.log10(harvest_area_km2)
            
            self.logger.debug(f"Converted harvest area to magnitude: {harvest_area_km2} km² → M={magnitude}")
            
            return magnitude
            
        except Exception as e:
            raise UnitConversionError(f"Failed to convert harvest area to magnitude: {str(e)}")
    
    def batch_convert_crop_data(self, crop_data: pd.DataFrame, 
                               crop_columns: List[str],
                               conversion_type: str) -> pd.DataFrame:
        """
        Batch convert multiple crop columns.
        
        Args:
            crop_data: DataFrame with crop data
            crop_columns: List of crop column names to convert
            conversion_type: Type of conversion ('production_to_kcal', 'harvest_to_km2')
        
        Returns:
            DataFrame with converted values
        
        Raises:
            UnitConversionError: If conversion type is invalid or conversion fails
        """
        try:
            converted_data = crop_data.copy()
            
            if conversion_type == 'production_to_kcal':
                conversion_func = self.convert_production_to_kcal
                unit_from, unit_to = 'MT', 'kcal'
            elif conversion_type == 'harvest_to_km2':
                conversion_func = self.convert_harvest_area_to_km2
                unit_from, unit_to = 'ha', 'km²'
            else:
                raise UnitConversionError(f"Invalid conversion type: {conversion_type}")
            
            converted_count = 0
            for col in crop_columns:
                if col in converted_data.columns:
                    converted_data[col] = conversion_func(converted_data[col])
                    converted_count += 1
            
            self.logger.info(f"Batch converted {converted_count} columns from {unit_from} to {unit_to}")
            
            return converted_data
            
        except Exception as e:
            raise UnitConversionError(f"Batch conversion failed: {str(e)}")
    
    def _validate_production_input(self, production: Union[pd.Series, np.ndarray, float]) -> None:
        """Validate production input values."""
        if isinstance(production, (pd.Series, np.ndarray)):
            # Check for negative values
            if (production < 0).any():
                raise UnitConversionError("Production values cannot be negative")
            
            # Check for extremely large values
            if (production > self.validation_ranges['production_mt_max']).any():
                max_val = production.max()
                self.logger.warning(f"Found extremely large production value: {max_val} MT")
        else:
            # Scalar validation
            if production < 0:
                raise UnitConversionError("Production value cannot be negative")
            if production > self.validation_ranges['production_mt_max']:
                self.logger.warning(f"Extremely large production value: {production} MT")
    
    def _validate_harvest_area_input(self, harvest_area: Union[pd.Series, np.ndarray, float]) -> None:
        """Validate harvest area input values."""
        if isinstance(harvest_area, (pd.Series, np.ndarray)):
            # Check for negative values
            if (harvest_area < 0).any():
                raise UnitConversionError("Harvest area values cannot be negative")
            
            # Check for extremely large values
            if (harvest_area > self.validation_ranges['harvest_ha_max']).any():
                max_val = harvest_area.max()
                self.logger.warning(f"Found extremely large harvest area value: {max_val} ha")
        else:
            # Scalar validation
            if harvest_area < 0:
                raise UnitConversionError("Harvest area value cannot be negative")
            if harvest_area > self.validation_ranges['harvest_ha_max']:
                self.logger.warning(f"Extremely large harvest area value: {harvest_area} ha")
    
    def _validate_yield_output(self, yield_values: Union[pd.Series, np.ndarray, float]) -> None:
        """Validate calculated yield values."""
        if isinstance(yield_values, (pd.Series, np.ndarray)):
            # Check for unreasonable yields
            low_yields = (yield_values > 0) & (yield_values < self.validation_ranges['yield_min'])
            high_yields = yield_values > self.validation_ranges['yield_max']
            
            if low_yields.any():
                count = low_yields.sum()
                self.logger.warning(f"Found {count} cells with very low yields (<{self.validation_ranges['yield_min']} t/ha)")
            
            if high_yields.any():
                count = high_yields.sum()
                max_yield = yield_values.max()
                self.logger.warning(f"Found {count} cells with very high yields (max: {max_yield:.2f} t/ha)")
        else:
            # Scalar validation
            if 0 < yield_values < self.validation_ranges['yield_min']:
                self.logger.warning(f"Very low yield: {yield_values:.4f} t/ha")
            elif yield_values > self.validation_ranges['yield_max']:
                self.logger.warning(f"Very high yield: {yield_values:.2f} t/ha")
    
    def _get_conversion_summary(self, input_val: Union[pd.Series, np.ndarray, float], 
                               output_val: Union[pd.Series, np.ndarray, float],
                               unit_from: str, unit_to: str) -> str:
        """Generate conversion summary for logging."""
        if isinstance(input_val, (pd.Series, np.ndarray)):
            return (f"Array conversion {unit_from}→{unit_to}: "
                   f"sum {input_val.sum():.2e} {unit_from} → {output_val.sum():.2e} {unit_to}")
        else:
            return f"Scalar conversion {unit_from}→{unit_to}: {input_val:.2e} → {output_val:.2e}"
    
    def _get_yield_summary(self, production: Union[pd.Series, np.ndarray, float],
                          harvest: Union[pd.Series, np.ndarray, float],
                          yield_val: Union[pd.Series, np.ndarray, float]) -> str:
        """Generate yield calculation summary for logging."""
        if isinstance(yield_val, (pd.Series, np.ndarray)):
            valid_yields = yield_val[yield_val > 0]
            if len(valid_yields) > 0:
                return (f"Yield calculation: {len(valid_yields)} valid cells, "
                       f"mean yield: {valid_yields.mean():.2f} t/ha, "
                       f"range: {valid_yields.min():.2f}-{valid_yields.max():.2f} t/ha")
            else:
                return "Yield calculation: no valid yields calculated"
        else:
            return f"Yield calculation: {yield_val:.2f} t/ha"
    
    def get_conversion_factors(self) -> Dict[str, float]:
        """
        Get all conversion factors used by the converter.
        
        Returns:
            Dictionary of conversion factors
        """
        return {
            'grams_per_metric_ton': self.conversions['grams_per_metric_ton'],
            'hectares_to_km2': self.conversions['hectares_to_km2'],
            'caloric_content_kcal_per_g': self.caloric_content,
            'crop_type': self.config.crop_type
        }
    
    def create_conversion_report(self, original_data: pd.DataFrame, 
                               converted_data: pd.DataFrame,
                               crop_columns: List[str]) -> str:
        """
        Create a detailed conversion report.
        
        Args:
            original_data: Original DataFrame before conversion
            converted_data: DataFrame after conversion
            crop_columns: List of crop columns that were converted
        
        Returns:
            Formatted conversion report
        """
        report_lines = [
            "=" * 50,
            "AgRichter Unit Conversion Report",
            "=" * 50,
            f"Crop Type: {self.config.crop_type}",
            f"Caloric Content: {self.caloric_content:.2f} kcal/g",
            "",
            "CONVERSION FACTORS",
            "-" * 20,
            f"Metric Tons to Grams: {self.conversions['grams_per_metric_ton']:,}",
            f"Hectares to km²: {self.conversions['hectares_to_km2']}",
            "",
            "CONVERTED COLUMNS",
            "-" * 18
        ]
        
        for col in crop_columns:
            if col in original_data.columns and col in converted_data.columns:
                orig_sum = original_data[col].sum()
                conv_sum = converted_data[col].sum()
                
                report_lines.append(f"{col}:")
                report_lines.append(f"  Original sum: {orig_sum:,.0f}")
                report_lines.append(f"  Converted sum: {conv_sum:.2e}")
                report_lines.append(f"  Conversion factor: {conv_sum/orig_sum:.2e}" if orig_sum > 0 else "  No data")
                report_lines.append("")
        
        return "\n".join(report_lines)
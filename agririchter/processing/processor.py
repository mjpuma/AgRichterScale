"""Main data processing module for AgriRichter framework."""

import logging
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np

from ..core.base import BaseProcessor
from ..core.config import Config
from .converters import UnitConverter


class DataProcessingError(Exception):
    """Exception raised for data processing errors."""
    pass


class DataProcessor(BaseProcessor):
    """Main data processor for AgriRichter analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize data processor.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agririchter.processor')
        self.converter = UnitConverter(config)
        
        # Get crop-specific parameters
        self.crop_indices = config.get_crop_indices()
        self.grid_params = config.get_grid_params()
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing entry point.
        
        Args:
            data: Input DataFrame to process
        
        Returns:
            Processed DataFrame
        """
        # This is a generic interface - specific processing methods below
        return data
    
    def filter_crop_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data for specific crop types based on configuration.
        
        Args:
            data: Full SPAM DataFrame
        
        Returns:
            DataFrame with selected crop columns plus metadata
        """
        try:
            self.logger.info(f"Filtering data for crop type: {self.config.crop_type}")
            
            # Get SPAM 2020 crop column names
            spam_crop_columns = [
                'BANA_A', 'BARL_A', 'BEAN_A', 'CASS_A', 'CHIC_A', 'CITR_A', 
                'CNUT_A', 'COCO_A', 'COFF_A', 'COTT_A', 'COWP_A', 'GROU_A', 
                'LENT_A', 'MAIZ_A', 'MILL_A', 'OCER_A', 'OFIB_A', 'OILP_A', 
                'ONIO_A', 'OOIL_A', 'OPUL_A', 'ORTS_A', 'PIGE_A', 'PLNT_A', 
                'PMIL_A', 'POTA_A', 'RAPE_A', 'RCOF_A', 'REST_A', 'RICE_A', 
                'RUBB_A', 'SESA_A', 'SORG_A', 'SOYB_A', 'SUGB_A', 'SUGC_A', 
                'SUNF_A', 'SWPO_A', 'TEAS_A', 'TEMF_A', 'TOBA_A', 'TOMA_A', 
                'TROF_A', 'VEGE_A', 'WHEA_A', 'YAMS_A'
            ]
            
            # Map crop indices to actual column names
            # For SPAM 2020, we need to map the column positions to column names
            column_position_to_name = {
                13: 'BANA_A', 14: 'BARL_A', 15: 'BEAN_A', 16: 'CASS_A', 17: 'CHIC_A', 18: 'CITR_A',
                19: 'CNUT_A', 20: 'COCO_A', 21: 'COFF_A', 22: 'COTT_A', 23: 'COWP_A', 24: 'GROU_A',
                25: 'LENT_A', 26: 'MAIZ_A', 27: 'MILL_A', 28: 'OCER_A', 29: 'OFIB_A', 30: 'OILP_A',
                31: 'ONIO_A', 32: 'OOIL_A', 33: 'OPUL_A', 34: 'ORTS_A', 35: 'PIGE_A', 36: 'PLNT_A',
                37: 'PMIL_A', 38: 'POTA_A', 39: 'RAPE_A', 40: 'RCOF_A', 41: 'REST_A', 42: 'RICE_A',
                43: 'RUBB_A', 44: 'SESA_A', 45: 'SORG_A', 46: 'SOYB_A', 47: 'SUGB_A', 48: 'SUGC_A',
                49: 'SUNF_A', 50: 'SWPO_A', 51: 'TEAS_A', 52: 'TEMF_A', 53: 'TOBA_A', 54: 'TOMA_A',
                55: 'TROF_A', 56: 'VEGE_A', 57: 'WHEA_A', 58: 'YAMS_A'
            }
            
            selected_crop_columns = [column_position_to_name[i] for i in self.crop_indices 
                                   if i in column_position_to_name]
            
            # Include essential metadata columns for SPAM 2020
            metadata_columns = ['grid_code', 'x', 'y', 'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME']
            
            # Combine metadata and crop columns
            all_columns = metadata_columns + selected_crop_columns
            
            # Filter columns that exist in the data
            available_columns = [col for col in all_columns if col in data.columns]
            
            if not available_columns:
                raise DataProcessingError("No required columns found in data")
            
            filtered_data = data[available_columns].copy()
            
            self.logger.info(f"Filtered data: {len(selected_crop_columns)} crop columns, "
                           f"{len(filtered_data)} rows")
            
            return filtered_data
            
        except Exception as e:
            raise DataProcessingError(f"Failed to filter crop data: {str(e)}")
    
    def convert_production_to_kcal(self, production_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert production data from metric tons to kilocalories.
        
        Args:
            production_df: Production DataFrame
        
        Returns:
            DataFrame with production in kcal
        """
        try:
            self.logger.info("Converting production data from metric tons to kcal")
            
            # Get crop columns (SPAM 2020 uses '_A' suffix)
            crop_columns = [col for col in production_df.columns if col.endswith('_A')]
            
            if not crop_columns:
                raise DataProcessingError("No crop columns found for conversion")
            
            # Convert using batch conversion
            converted_df = self.converter.batch_convert_crop_data(
                production_df, crop_columns, 'production_to_kcal'
            )
            
            # Log conversion summary
            original_sum = production_df[crop_columns].sum().sum()
            converted_sum = converted_df[crop_columns].sum().sum()
            
            self.logger.info(f"Production conversion complete: "
                           f"{original_sum:.2e} MT → {converted_sum:.2e} kcal")
            
            return converted_df
            
        except Exception as e:
            raise DataProcessingError(f"Failed to convert production to kcal: {str(e)}")
    
    def convert_harvest_area_to_km2(self, harvest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert harvest area data from hectares to square kilometers.
        
        Args:
            harvest_df: Harvest area DataFrame
        
        Returns:
            DataFrame with harvest area in km²
        """
        try:
            self.logger.info("Converting harvest area from hectares to km²")
            
            # Get crop columns (SPAM 2020 uses '_A' suffix)
            crop_columns = [col for col in harvest_df.columns if col.endswith('_A')]
            
            if not crop_columns:
                raise DataProcessingError("No crop columns found for conversion")
            
            # Convert using batch conversion
            converted_df = self.converter.batch_convert_crop_data(
                harvest_df, crop_columns, 'harvest_to_km2'
            )
            
            # Log conversion summary
            original_sum = harvest_df[crop_columns].sum().sum()
            converted_sum = converted_df[crop_columns].sum().sum()
            
            self.logger.info(f"Harvest area conversion complete: "
                           f"{original_sum:.2e} ha → {converted_sum:.2e} km²")
            
            return converted_df
            
        except Exception as e:
            raise DataProcessingError(f"Failed to convert harvest area to km²: {str(e)}")
    
    def calculate_yields(self, production_df: pd.DataFrame, 
                        harvest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate yields (production per unit harvest area).
        
        Args:
            production_df: Production DataFrame (in kcal)
            harvest_df: Harvest area DataFrame (in km²)
        
        Returns:
            DataFrame with yield calculations
        """
        try:
            self.logger.info("Calculating yields (production per unit area)")
            
            # Ensure coordinate consistency
            if len(production_df) != len(harvest_df):
                raise DataProcessingError("Production and harvest data have different lengths")
            
            # Get crop columns (SPAM 2020 uses '_A' suffix)
            crop_columns = [col for col in production_df.columns if col.endswith('_A')]
            
            # Initialize yield DataFrame with coordinates
            yield_df = production_df[['x', 'y', 'grid_code']].copy()
            
            # Calculate yields for each crop
            for col in crop_columns:
                if col in harvest_df.columns:
                    yield_df[col] = self.converter.calculate_yield(
                        production_df[col], harvest_df[col]
                    )
            
            # Add total yield calculation (sum across selected crops)
            total_production = production_df[crop_columns].sum(axis=1)
            total_harvest = harvest_df[crop_columns].sum(axis=1)
            yield_df['total_yield'] = self.converter.calculate_yield(total_production, total_harvest)
            
            # Log yield statistics
            valid_yields = yield_df['total_yield'][yield_df['total_yield'] > 0]
            if len(valid_yields) > 0:
                self.logger.info(f"Yield calculation complete: {len(valid_yields)} valid cells, "
                               f"mean yield: {valid_yields.mean():.2f} kcal/km²")
            
            return yield_df
            
        except Exception as e:
            raise DataProcessingError(f"Failed to calculate yields: {str(e)}")
    
    def create_production_grid(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create gridded production data for mapping.
        
        Args:
            data: Production DataFrame with x, y coordinates
        
        Returns:
            2D numpy array with gridded production data
        """
        try:
            self.logger.info("Creating production grid for mapping")
            
            # Get grid parameters
            ncols = int(self.grid_params['ncols'])
            nrows = int(self.grid_params['nrows'])
            xllcorner = self.grid_params['xllcorner']
            yllcorner = self.grid_params['yllcorner']
            cellsize = self.grid_params['cellsize']
            
            # Create coordinate vectors
            x_coords = np.arange(xllcorner, xllcorner + ncols * cellsize, cellsize)
            y_coords = np.arange(yllcorner + (nrows-1) * cellsize, yllcorner - cellsize, -cellsize)
            
            # Initialize grid with NaN
            grid = np.full((nrows, ncols), np.nan)
            
            # Get crop columns and sum production (SPAM 2020 uses '_A' suffix)
            crop_columns = [col for col in data.columns if col.endswith('_A')]
            if crop_columns:
                total_production = data[crop_columns].sum(axis=1)
            else:
                raise DataProcessingError("No crop columns found for grid creation")
            
            # Apply log10 transformation (add 1 to handle zeros)
            log_production = np.log10(total_production + 1)
            
            # Map data to grid
            for idx, row in data.iterrows():
                # Find closest grid indices
                x_idx = self._find_closest_index(x_coords, row['x'])
                y_idx = self._find_closest_index(y_coords, row['y'])
                
                if 0 <= x_idx < ncols and 0 <= y_idx < nrows:
                    grid[y_idx, x_idx] = log_production.iloc[idx]
            
            # Log grid statistics
            valid_cells = ~np.isnan(grid)
            self.logger.info(f"Production grid created: {ncols}×{nrows}, "
                           f"{valid_cells.sum()} cells with data "
                           f"({valid_cells.sum()/(ncols*nrows)*100:.1f}% coverage)")
            
            return grid
            
        except Exception as e:
            raise DataProcessingError(f"Failed to create production grid: {str(e)}")
    
    def aggregate_crop_data(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate crop data for summary statistics.
        
        Args:
            data: Crop DataFrame
        
        Returns:
            Dictionary with aggregated statistics
        """
        try:
            crop_columns = [col for col in data.columns if col.endswith('_A')]
            
            if not crop_columns:
                return {}
            
            # Calculate totals
            totals = {}
            for col in crop_columns:
                totals[col] = float(data[col].sum())
            
            # Calculate overall total
            totals['total_all_crops'] = sum(totals.values())
            
            # Calculate coverage statistics
            coverage = {}
            for col in crop_columns:
                non_zero_cells = (data[col] > 0).sum()
                coverage[f"{col}_coverage_pct"] = (non_zero_cells / len(data)) * 100
            
            # Combine statistics
            stats = {**totals, **coverage}
            stats['total_cells'] = len(data)
            stats['crop_type'] = self.config.crop_type
            
            return stats
            
        except Exception as e:
            raise DataProcessingError(f"Failed to aggregate crop data: {str(e)}")
    
    def process_complete_dataset(self, production_df: pd.DataFrame, 
                                harvest_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Complete processing pipeline for production and harvest data.
        
        Args:
            production_df: Raw production DataFrame
            harvest_df: Raw harvest area DataFrame
        
        Returns:
            Dictionary with processed DataFrames
        """
        try:
            self.logger.info("Starting complete data processing pipeline")
            
            # Step 1: Filter crop data
            filtered_production = self.filter_crop_data(production_df)
            filtered_harvest = self.filter_crop_data(harvest_df)
            
            # Step 2: Convert units
            production_kcal = self.convert_production_to_kcal(filtered_production)
            harvest_km2 = self.convert_harvest_area_to_km2(filtered_harvest)
            
            # Step 3: Calculate yields
            yields = self.calculate_yields(production_kcal, harvest_km2)
            
            # Step 4: Create production grid
            production_grid = self.create_production_grid(production_kcal)
            
            # Step 5: Generate summary statistics
            production_stats = self.aggregate_crop_data(production_kcal)
            harvest_stats = self.aggregate_crop_data(harvest_km2)
            
            self.logger.info("Complete data processing pipeline finished successfully")
            
            return {
                'production_kcal': production_kcal,
                'harvest_km2': harvest_km2,
                'yields': yields,
                'production_grid': pd.DataFrame(production_grid),  # Convert to DataFrame for consistency
                'production_stats': pd.DataFrame([production_stats]),
                'harvest_stats': pd.DataFrame([harvest_stats])
            }
            
        except Exception as e:
            raise DataProcessingError(f"Complete processing pipeline failed: {str(e)}")
    
    def _find_closest_index(self, array: np.ndarray, value: float) -> int:
        """
        Find index of closest value in array.
        
        Args:
            array: Array to search
            value: Target value
        
        Returns:
            Index of closest value
        """
        return int(np.argmin(np.abs(array - value)))
    
    def create_processing_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """
        Create detailed processing report.
        
        Args:
            results: Processing results dictionary
        
        Returns:
            Formatted processing report
        """
        report_lines = [
            "=" * 60,
            "AgriRichter Data Processing Report",
            "=" * 60,
            f"Crop Type: {self.config.crop_type}",
            f"Selected Crop Indices: {self.crop_indices}",
            "",
            "PROCESSING RESULTS",
            "-" * 20
        ]
        
        # Add statistics for each processed dataset
        for name, data in results.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                report_lines.extend([
                    f"{name.upper()}:",
                    f"  Rows: {len(data):,}",
                    f"  Columns: {len(data.columns)}",
                ])
                
                # Add crop-specific statistics
                crop_cols = [col for col in data.columns if col.endswith('_a')]
                if crop_cols:
                    total_sum = data[crop_cols].sum().sum()
                    report_lines.append(f"  Total sum: {total_sum:.2e}")
                
                report_lines.append("")
        
        # Add conversion factors used
        conversion_factors = self.converter.get_conversion_factors()
        report_lines.extend([
            "CONVERSION FACTORS",
            "-" * 19,
            f"Caloric Content: {conversion_factors['caloric_content_kcal_per_g']:.2f} kcal/g",
            f"MT to Grams: {conversion_factors['grams_per_metric_ton']:,}",
            f"Hectares to km²: {conversion_factors['hectares_to_km2']}",
            ""
        ])
        
        return "\n".join(report_lines)
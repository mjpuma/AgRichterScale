"""Data loading module for AgriRichter framework."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from ..core.base import BaseDataLoader
from ..core.config import Config
from ..core.utils import validate_coordinate_ranges, ProgressReporter


class DataLoadError(Exception):
    """Exception raised for data loading errors."""
    pass


class DataLoader(BaseDataLoader):
    """Data loader for SPAM 2020 and ancillary data."""
    
    def __init__(self, config: Config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agririchter.data')
        
        # SPAM 2020 column names (based on actual data structure)
        self.spam_columns = [
            'grid_code', 'x', 'y', 'FIPS0', 'FIPS1', 'FIPS2', 
            'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME', 'rec_type', 'tech_type', 'unit',
            'BANA_A', 'BARL_A', 'BEAN_A', 'CASS_A', 'CHIC_A', 'CITR_A', 
            'CNUT_A', 'COCO_A', 'COFF_A', 'COTT_A', 'COWP_A', 'GROU_A', 
            'LENT_A', 'MAIZ_A', 'MILL_A', 'OCER_A', 'OFIB_A', 'OILP_A', 
            'ONIO_A', 'OOIL_A', 'OPUL_A', 'ORTS_A', 'PIGE_A', 'PLNT_A', 
            'PMIL_A', 'POTA_A', 'RAPE_A', 'RCOF_A', 'REST_A', 'RICE_A', 
            'RUBB_A', 'SESA_A', 'SORG_A', 'SOYB_A', 'SUGB_A', 'SUGC_A', 
            'SUNF_A', 'SWPO_A', 'TEAS_A', 'TEMF_A', 'TOBA_A', 'TOMA_A', 
            'TROF_A', 'VEGE_A', 'WHEA_A', 'YAMS_A', 'year_data'
        ]
        
        # Crop column indices (0-based, starting from BANA_A)
        self.crop_column_indices = list(range(12, 57))  # columns 12-56 contain crop data
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file with proper error handling.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            Loaded DataFrame
        
        Raises:
            DataLoadError: If file cannot be loaded
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataLoadError(f"File not found: {path}")
            
            self.logger.info(f"Loading data from {path}")
            
            # Load CSV with appropriate data types for SPAM 2020
            df = pd.read_csv(
                path,
                dtype={
                    'grid_code': 'int64',
                    'x': 'float64',
                    'y': 'float64',
                    'FIPS0': 'category',
                    'FIPS1': 'category', 
                    'FIPS2': 'category',
                    'ADM0_NAME': 'category',
                    'ADM1_NAME': 'category',
                    'ADM2_NAME': 'category',
                    'rec_type': 'category',
                    'tech_type': 'category',
                    'unit': 'category',
                    'year_data': 'category'
                },
                low_memory=False
            )
            
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load {file_path}: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate loaded SPAM data.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            True if data is valid
        
        Raises:
            DataLoadError: If validation fails
        """
        # Check required columns
        required_cols = ['x', 'y'] + [self.spam_columns[i] for i in self.crop_column_indices]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise DataLoadError(f"Missing required columns: {missing_cols}")
        
        # Check coordinate ranges
        invalid_coords = 0
        for idx, row in data.iterrows():
            if not validate_coordinate_ranges(row['y'], row['x']):
                invalid_coords += 1
        
        if invalid_coords > 0:
            self.logger.warning(f"Found {invalid_coords} rows with invalid coordinates")
        
        # Check for completely empty crop data
        crop_cols = [self.spam_columns[i] for i in self.crop_column_indices]
        crop_data = data[crop_cols]
        
        if crop_data.isna().all().all():
            raise DataLoadError("All crop data is missing")
        
        # Log data quality metrics
        total_cells = len(data)
        non_zero_cells = (crop_data > 0).any(axis=1).sum()
        
        self.logger.info(f"Data validation complete:")
        self.logger.info(f"  Total grid cells: {total_cells}")
        self.logger.info(f"  Cells with crop data: {non_zero_cells}")
        self.logger.info(f"  Coverage: {non_zero_cells/total_cells*100:.1f}%")
        
        return True
    
    def load_spam_production(self) -> pd.DataFrame:
        """
        Load SPAM 2020 production data.
        
        Returns:
            Production DataFrame
        """
        file_path = self.config.get_file_paths()['production']
        data = self.load_data(file_path)
        
        # Validate and clean data
        self.validate_data(data)
        
        # SPAM 2020 doesn't need the China ISO3 fix from SPAM 2010
        
        return data
    
    def load_spam_harvest_area(self) -> pd.DataFrame:
        """
        Load SPAM 2020 harvest area data.
        
        Returns:
            Harvest area DataFrame
        """
        file_path = self.config.get_file_paths()['harvest_area']
        data = self.load_data(file_path)
        
        # Validate data
        self.validate_data(data)
        
        return data
    
    def load_nutrition_data(self) -> pd.DataFrame:
        """
        Load crop nutrition data from Excel file.
        
        Returns:
            Nutrition DataFrame with caloric content per crop
        """
        file_path = self.config.get_file_paths()['nutrition']
        
        try:
            self.logger.info(f"Loading nutrition data from {file_path}")
            
            # Load Excel file (skip header row, get columns B onwards)
            df = pd.read_excel(
                file_path,
                sheet_name='Sheet1',
                skiprows=1,  # Skip header row
                usecols="B:Z",  # Columns B onwards
                engine='openpyxl'
            )
            
            # Replace -9999 values with NaN (from original MATLAB code)
            df = df.replace(-9999, np.nan)
            
            self.logger.info(f"Loaded nutrition data for {len(df)} crops")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load nutrition data: {str(e)}")
    
    def load_country_codes(self) -> pd.DataFrame:
        """
        Load country code conversion table.
        
        Returns:
            Country codes DataFrame
        """
        file_path = self.config.get_file_paths()['country_codes']
        
        try:
            self.logger.info(f"Loading country codes from {file_path}")
            
            df = pd.read_excel(
                file_path,
                sheet_name='Sheet1',
                skiprows=1,  # Skip header row
                usecols="A:K",  # Columns A to K
                engine='openpyxl'
            )
            
            # Set proper column names
            df.columns = [
                'Country', 'PumaIndex', 'FAOSTAT', 'Siebert', 'ISOCode', 
                'FAO_perhaps', 'GAUL', 'GDAM', 'WorldBank', 'ISO3Alpha', 
                'USDAPSD'
            ]
            
            self.logger.info(f"Loaded {len(df)} country code mappings")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load country codes: {str(e)}")
    
    def load_historical_events(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical disruption events data.
        
        Returns:
            Dictionary with 'country' and 'state' DataFrames
        """
        file_paths = self.config.get_file_paths()
        
        try:
            # Load country-level disruptions
            self.logger.info("Loading historical disruption events")
            
            country_df = pd.read_excel(
                file_paths['disruption_country'],
                sheet_name=None,  # Load all sheets
                engine='xlrd'  # Use xlrd for .xls files
            )
            
            state_df = pd.read_excel(
                file_paths['disruption_state'],
                sheet_name=None,  # Load all sheets
                engine='xlrd'  # Use xlrd for .xls files
            )
            
            self.logger.info(f"Loaded disruption data for {len(country_df)} country sheets")
            self.logger.info(f"Loaded disruption data for {len(state_df)} state sheets")
            
            return {
                'country': country_df,
                'state': state_df
            }
            
        except Exception as e:
            raise DataLoadError(f"Failed to load historical events: {str(e)}")
    
    def get_crop_data_subset(self, data: pd.DataFrame, crop_indices: List[int]) -> pd.DataFrame:
        """
        Extract subset of crop data for specified crop indices.
        
        Args:
            data: Full SPAM DataFrame
            crop_indices: List of crop indices to extract (1-based)
        
        Returns:
            DataFrame with selected crop columns plus metadata
        """
        # Convert 1-based indices to 0-based column indices
        crop_col_indices = [i + 8 for i in crop_indices]  # +8 to account for metadata columns
        
        # Get crop column names
        crop_cols = [self.spam_columns[i] for i in crop_col_indices]
        
        # Include metadata columns
        metadata_cols = ['iso3', 'x', 'y', 'cell5m', 'name_cntr', 'name_adm1', 'name_adm2']
        
        # Select columns
        selected_cols = metadata_cols + crop_cols
        subset = data[selected_cols].copy()
        
        self.logger.info(f"Extracted {len(crop_cols)} crop columns for {len(subset)} grid cells")
        
        return subset
    
    def validate_coordinate_consistency(self, production_df: pd.DataFrame, 
                                      harvest_df: pd.DataFrame) -> bool:
        """
        Validate that production and harvest area data have consistent coordinates.
        
        Args:
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
        
        Returns:
            True if coordinates are consistent
        
        Raises:
            DataLoadError: If coordinates are inconsistent
        """
        # Check if both DataFrames have same number of rows
        if len(production_df) != len(harvest_df):
            raise DataLoadError(
                f"Production and harvest data have different lengths: "
                f"{len(production_df)} vs {len(harvest_df)}"
            )
        
        # Check coordinate consistency
        coord_diff_x = (production_df['x'] - harvest_df['x']).abs().max()
        coord_diff_y = (production_df['y'] - harvest_df['y']).abs().max()
        
        tolerance = 1e-6  # Small tolerance for floating point comparison
        
        if coord_diff_x > tolerance or coord_diff_y > tolerance:
            raise DataLoadError(
                f"Coordinate mismatch between production and harvest data: "
                f"max diff x={coord_diff_x}, y={coord_diff_y}"
            )
        
        self.logger.info("Coordinate consistency validation passed")
        return True
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required data files.
        
        Returns:
            Dictionary containing all loaded DataFrames
        """
        self.logger.info("Starting comprehensive data loading")
        
        # Check file existence first
        missing_files = self.config.validate_files_exist()
        if missing_files:
            raise DataLoadError(f"Missing required files: {missing_files}")
        
        progress = ProgressReporter(5, "Loading data files")
        
        try:
            # Load main SPAM data
            progress.update(1, "Loading production data")
            production_df = self.load_spam_production()
            
            progress.update(1, "Loading harvest area data")
            harvest_df = self.load_spam_harvest_area()
            
            # Validate coordinate consistency
            self.validate_coordinate_consistency(production_df, harvest_df)
            
            # Load ancillary data
            progress.update(1, "Loading nutrition data")
            nutrition_df = self.load_nutrition_data()
            
            progress.update(1, "Loading country codes")
            country_codes_df = self.load_country_codes()
            
            progress.update(1, "Loading historical events")
            historical_events = self.load_historical_events()
            
            progress.complete("All data loaded successfully")
            
            return {
                'production': production_df,
                'harvest_area': harvest_df,
                'nutrition': nutrition_df,
                'country_codes': country_codes_df,
                'historical_events': historical_events
            }
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
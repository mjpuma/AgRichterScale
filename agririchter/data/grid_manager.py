"""Grid Data Manager for SPAM 2020 gridded agricultural data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from ..core.config import Config


logger = logging.getLogger(__name__)


class GridDataManager:
    """
    Manages SPAM 2020 gridded production and harvest area data.
    
    Provides efficient loading, spatial indexing, and querying of gridded
    agricultural data for event loss calculations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize GridDataManager.
        
        Args:
            config: Configuration object with data paths and parameters
        """
        self.config = config
        self.production_df: Optional[pd.DataFrame] = None
        self.harvest_area_df: Optional[pd.DataFrame] = None
        self.production_gdf: Optional[gpd.GeoDataFrame] = None
        self.harvest_area_gdf: Optional[gpd.GeoDataFrame] = None
        self._data_loaded = False
        self._spatial_index_created = False
        self._cache: Dict = {}
        
        logger.info(f"Initialized GridDataManager with SPAM version {config.get_spam_version()}")
    
    def load_spam_data(self, use_chunked_reading: bool = False, chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load SPAM 2020 production and harvest area data with optimizations.
        
        Args:
            use_chunked_reading: If True, use chunked reading for very large files
            chunk_size: Number of rows per chunk when using chunked reading
        
        Returns:
            Tuple of (production_df, harvest_area_df)
        
        Raises:
            FileNotFoundError: If SPAM data files don't exist
            ValueError: If data validation fails
        """
        if self._data_loaded:
            logger.info("SPAM data already loaded, returning cached data")
            return self.production_df, self.harvest_area_df
        
        import time
        start_time = time.time()
        
        file_paths = self.config.get_file_paths()
        production_file = file_paths['production']
        harvest_area_file = file_paths['harvest_area']
        
        # Validate files exist
        if not production_file.exists():
            raise FileNotFoundError(
                f"Production file not found: {production_file}\n"
                f"Expected SPAM {self.config.get_spam_version()} data at this location."
            )
        
        if not harvest_area_file.exists():
            raise FileNotFoundError(
                f"Harvest area file not found: {harvest_area_file}\n"
                f"Expected SPAM {self.config.get_spam_version()} data at this location."
            )
        
        logger.info(f"Loading SPAM {self.config.get_spam_version()} production data from {production_file}")
        logger.info(f"Loading SPAM {self.config.get_spam_version()} harvest area data from {harvest_area_file}")
        
        # Define optimized dtypes for memory efficiency
        dtypes = self._get_optimized_dtypes()
        
        # Load production data with optimizations
        try:
            if use_chunked_reading:
                logger.info(f"Using chunked reading with chunk_size={chunk_size}")
                chunks = []
                for chunk in pd.read_csv(production_file, dtype=dtypes, chunksize=chunk_size):
                    chunks.append(chunk)
                self.production_df = pd.concat(chunks, ignore_index=True)
            else:
                # Use optimized pandas settings for faster loading
                self.production_df = pd.read_csv(
                    production_file, 
                    dtype=dtypes,
                    engine='c',  # Use C engine for faster parsing
                    low_memory=False  # Read entire file at once for better performance
                )
            
            # Optimize memory usage by converting object columns to category
            self._optimize_memory_usage(self.production_df)
            
            logger.info(f"Loaded production data: {len(self.production_df)} grid cells")
            logger.info(f"Production data memory usage: {self.production_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        except Exception as e:
            raise ValueError(f"Failed to load production data: {e}")
        
        # Load harvest area data with optimizations
        try:
            if use_chunked_reading:
                chunks = []
                for chunk in pd.read_csv(harvest_area_file, dtype=dtypes, chunksize=chunk_size):
                    chunks.append(chunk)
                self.harvest_area_df = pd.concat(chunks, ignore_index=True)
            else:
                self.harvest_area_df = pd.read_csv(
                    harvest_area_file, 
                    dtype=dtypes,
                    engine='c',
                    low_memory=False
                )
            
            # Optimize memory usage
            self._optimize_memory_usage(self.harvest_area_df)
            
            logger.info(f"Loaded harvest area data: {len(self.harvest_area_df)} grid cells")
            logger.info(f"Harvest area data memory usage: {self.harvest_area_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        except Exception as e:
            raise ValueError(f"Failed to load harvest area data: {e}")
        
        # Validate data structure
        self._validate_data_structure()
        
        # Preserve coordinates
        self._ensure_coordinates()
        
        self._data_loaded = True
        
        load_time = time.time() - start_time
        logger.info(f"SPAM data loaded successfully in {load_time:.2f} seconds")
        
        return self.production_df, self.harvest_area_df
    
    def _get_optimized_dtypes(self) -> Dict[str, str]:
        """
        Get optimized dtypes for memory-efficient data loading.
        
        Returns:
            Dictionary mapping column names to dtypes
        """
        # Base dtypes for metadata columns
        dtypes = {
            'grid_code': 'int32',
            'FIPS0': 'category',
            'FIPS1': 'category',
            'FIPS2': 'category',
            'ADM0_NAME': 'category',
            'ADM1_NAME': 'category',
            'ADM2_NAME': 'category',
            'rec_type': 'category',
            'tech_type': 'category',
            'unit': 'category',
            'year_data': 'str',  # Can be 'avg(2019-2021)' or year number
            'x': 'float32',
            'y': 'float32'
        }
        
        # All crop columns as float32 for memory efficiency
        # SPAM 2020 has 46 crop columns with _A suffix
        crop_suffixes = ['_A']
        crop_codes = [
            'WHEA', 'RICE', 'MAIZ', 'BARL', 'PMIL', 'SMIL', 'SORG', 'OCER',
            'POTA', 'SWPO', 'YAMS', 'CASS', 'ORTS', 'BEAN', 'CHIC', 'COWP',
            'PIGE', 'LENT', 'OPUL', 'SOYB', 'GROU', 'CNUT', 'OILP', 'SUNF',
            'RAPE', 'SESA', 'OOIL', 'SUGC', 'SUGB', 'COTT', 'OFIB', 'ACOF',
            'RCOF', 'COCO', 'TEAS', 'TOBA', 'BANA', 'PLNT', 'TROF', 'TEMF',
            'VEGE', 'REST', 'MILL', 'RUBB', 'PALM', 'OILC'
        ]
        
        for code in crop_codes:
            for suffix in crop_suffixes:
                col_name = f"{code}{suffix}"
                dtypes[col_name] = 'float32'
        
        return dtypes
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> None:
        """
        Optimize memory usage of DataFrame by downcasting numeric types.
        
        Args:
            df: DataFrame to optimize (modified in place)
        """
        # Convert any remaining object columns to category if they have low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df)
            
            # If less than 50% unique values, convert to category
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
                logger.debug(f"Converted column {col} to category (cardinality: {num_unique})")
        
        # Downcast numeric columns where possible
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    def _validate_data_structure(self) -> None:
        """
        Validate that loaded data has expected structure.
        
        Raises:
            ValueError: If data structure is invalid
        """
        required_cols = ['x', 'y', 'FIPS0', 'grid_code']
        
        # Check production data
        missing_prod = [col for col in required_cols if col not in self.production_df.columns]
        if missing_prod:
            raise ValueError(f"Production data missing required columns: {missing_prod}")
        
        # Check harvest area data
        missing_harvest = [col for col in required_cols if col not in self.harvest_area_df.columns]
        if missing_harvest:
            raise ValueError(f"Harvest area data missing required columns: {missing_harvest}")
        
        # Check if dataframes have same number of rows (warning if different)
        if len(self.production_df) != len(self.harvest_area_df):
            logger.warning(
                f"Production and harvest area data have different row counts: "
                f"{len(self.production_df)} vs {len(self.harvest_area_df)}. "
                f"This is acceptable if coverage differs."
            )
        
        logger.info("Data structure validation passed")
    
    def _ensure_coordinates(self) -> None:
        """
        Ensure x, y coordinates are preserved and valid.
        
        Raises:
            ValueError: If coordinates are invalid
        """
        # Check for missing coordinates
        prod_missing = self.production_df[['x', 'y']].isna().sum()
        if prod_missing.any():
            logger.warning(f"Production data has missing coordinates: {prod_missing.to_dict()}")
        
        harvest_missing = self.harvest_area_df[['x', 'y']].isna().sum()
        if harvest_missing.any():
            logger.warning(f"Harvest area data has missing coordinates: {harvest_missing.to_dict()}")
        
        # Validate coordinate ranges
        x_range = (self.production_df['x'].min(), self.production_df['x'].max())
        y_range = (self.production_df['y'].min(), self.production_df['y'].max())
        
        if not (-180 <= x_range[0] and x_range[1] <= 180):
            raise ValueError(f"Invalid longitude range: {x_range}")
        
        if not (-90 <= y_range[0] and y_range[1] <= 90):
            raise ValueError(f"Invalid latitude range: {y_range}")
        
        logger.info(f"Coordinate ranges - X: {x_range}, Y: {y_range}")
    
    def is_loaded(self) -> bool:
        """
        Check if SPAM data has been loaded.
        
        Returns:
            True if data is loaded
        """
        return self._data_loaded
    
    def get_production_data(self) -> pd.DataFrame:
        """
        Get production DataFrame.
        
        Returns:
            Production DataFrame
        
        Raises:
            RuntimeError: If data not loaded
        """
        if not self._data_loaded:
            raise RuntimeError("SPAM data not loaded. Call load_spam_data() first.")
        return self.production_df
    
    def get_harvest_area_data(self) -> pd.DataFrame:
        """
        Get harvest area DataFrame.
        
        Returns:
            Harvest area DataFrame
        
        Raises:
            RuntimeError: If data not loaded
        """
        if not self._data_loaded:
            raise RuntimeError("SPAM data not loaded. Call load_spam_data() first.")
        return self.harvest_area_df
    
    def clear_cache(self) -> None:
        """Clear internal cache to free memory."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def create_spatial_index(self) -> None:
        """
        Create GeoDataFrame with Point geometries and spatial index.
        
        Builds R-tree spatial index for efficient geographic queries.
        Uses vectorized operations for optimal performance.
        
        Raises:
            RuntimeError: If data not loaded
        """
        if not self._data_loaded:
            raise RuntimeError("SPAM data not loaded. Call load_spam_data() first.")
        
        if self._spatial_index_created:
            logger.info("Spatial index already created, reusing existing index")
            return
        
        import time
        start_time = time.time()
        
        logger.info("Creating spatial index for grid cells...")
        
        # Create Point geometries using vectorized operations (much faster than list comprehension)
        logger.info("Creating Point geometries for production data...")
        production_geometry = gpd.points_from_xy(
            self.production_df['x'], 
            self.production_df['y']
        )
        
        logger.info("Creating Point geometries for harvest area data...")
        harvest_geometry = gpd.points_from_xy(
            self.harvest_area_df['x'], 
            self.harvest_area_df['y']
        )
        
        # Create GeoDataFrames with WGS84 CRS
        logger.info("Creating GeoDataFrames...")
        self.production_gdf = gpd.GeoDataFrame(
            self.production_df,
            geometry=production_geometry,
            crs='EPSG:4326'
        )
        
        self.harvest_area_gdf = gpd.GeoDataFrame(
            self.harvest_area_df,
            geometry=harvest_geometry,
            crs='EPSG:4326'
        )
        
        # Build spatial index explicitly (R-tree)
        logger.info("Building R-tree spatial index...")
        _ = self.production_gdf.sindex
        _ = self.harvest_area_gdf.sindex
        
        self._spatial_index_created = True
        
        index_time = time.time() - start_time
        logger.info(f"Spatial index created for {len(self.production_gdf)} grid cells in {index_time:.2f} seconds")
    
    def get_grid_cells_by_coordinates(
        self, 
        bounds: Tuple[float, float, float, float]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get grid cells within bounding box.
        
        Args:
            bounds: Tuple of (min_x, min_y, max_x, max_y) in degrees
        
        Returns:
            Tuple of (production_cells, harvest_area_cells) DataFrames
        
        Raises:
            RuntimeError: If spatial index not created
        """
        if not self._spatial_index_created:
            raise RuntimeError("Spatial index not created. Call create_spatial_index() first.")
        
        min_x, min_y, max_x, max_y = bounds
        
        # Validate bounds
        if not (-180 <= min_x <= 180 and -180 <= max_x <= 180):
            raise ValueError(f"Invalid longitude bounds: ({min_x}, {max_x})")
        if not (-90 <= min_y <= 90 and -90 <= max_y <= 90):
            raise ValueError(f"Invalid latitude bounds: ({min_y}, {max_y})")
        
        # Create cache key
        cache_key = f"bbox_{min_x}_{min_y}_{max_x}_{max_y}"
        
        if cache_key in self._cache:
            logger.debug(f"Returning cached results for bounds {bounds}")
            return self._cache[cache_key]
        
        logger.info(f"Querying grid cells within bounds: {bounds}")
        
        # Query using spatial index
        # Filter by bounding box
        prod_mask = (
            (self.production_gdf['x'] >= min_x) &
            (self.production_gdf['x'] <= max_x) &
            (self.production_gdf['y'] >= min_y) &
            (self.production_gdf['y'] <= max_y)
        )
        
        harvest_mask = (
            (self.harvest_area_gdf['x'] >= min_x) &
            (self.harvest_area_gdf['x'] <= max_x) &
            (self.harvest_area_gdf['y'] >= min_y) &
            (self.harvest_area_gdf['y'] <= max_y)
        )
        
        production_cells = self.production_gdf[prod_mask].copy()
        harvest_area_cells = self.harvest_area_gdf[harvest_mask].copy()
        
        logger.info(f"Found {len(production_cells)} grid cells in bounding box")
        
        # Cache results
        self._cache[cache_key] = (production_cells, harvest_area_cells)
        
        return production_cells, harvest_area_cells
    
    def get_grid_cells_by_iso3(
        self, 
        iso3_code: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get grid cells for a specific country by ISO3 code.
        
        Uses cached results for better performance when querying
        the same country multiple times.
        
        Args:
            iso3_code: ISO3 country code (e.g., 'USA', 'CHN', 'IND')
        
        Returns:
            Tuple of (production_cells, harvest_area_cells) DataFrames
        
        Raises:
            RuntimeError: If data not loaded
            ValueError: If ISO3 code not found
        """
        if not self._data_loaded:
            raise RuntimeError("SPAM data not loaded. Call load_spam_data() first.")
        
        # Check cache first
        cache_key = f"iso3_{iso3_code}"
        if cache_key in self._cache:
            logger.debug(f"Returning cached results for ISO3: {iso3_code}")
            return self._cache[cache_key]
        
        logger.debug(f"Filtering grid cells for ISO3 code: {iso3_code}")
        
        # Use vectorized boolean indexing (faster than query or iterrows)
        # Create boolean masks once
        prod_mask = self.production_df['FIPS0'] == iso3_code
        harv_mask = self.harvest_area_df['FIPS0'] == iso3_code
        
        # Apply masks to get filtered DataFrames
        production_cells = self.production_df[prod_mask].copy()
        harvest_area_cells = self.harvest_area_df[harv_mask].copy()
        
        if len(production_cells) == 0:
            logger.warning(f"No grid cells found for ISO3 code: {iso3_code}")
            # Return empty DataFrames rather than raising error
            return production_cells, harvest_area_cells
        
        logger.debug(f"Found {len(production_cells)} grid cells for {iso3_code}")
        
        # Cache results
        self._cache[cache_key] = (production_cells, harvest_area_cells)
        
        return production_cells, harvest_area_cells
    
    def get_available_iso3_codes(self) -> List[str]:
        """
        Get list of all available ISO3 country codes in the data.
        
        Returns:
            Sorted list of ISO3 codes
        
        Raises:
            RuntimeError: If data not loaded
        """
        if not self._data_loaded:
            raise RuntimeError("SPAM data not loaded. Call load_spam_data() first.")
        
        # Use production data (both should have same countries)
        iso3_codes = sorted(self.production_df['FIPS0'].unique().tolist())
        return iso3_codes
    
    def _get_crop_column_names(self, crop_indices: List[int]) -> List[str]:
        """
        Map crop indices to SPAM column names.
        
        Args:
            crop_indices: List of 1-based crop indices
        
        Returns:
            List of SPAM column names (e.g., ['WHEA_A', 'RICE_A'])
        """
        # SPAM 2020 crop column mapping (1-based index to column name)
        # Based on CROP_INDICES in constants.py
        crop_map = {
            14: 'BARL_A',   # Barley
            26: 'MAIZ_A',   # Maize
            28: 'OCER_A',   # Other cereals
            37: 'PMIL_A',   # Pearl millet
            42: 'RICE_A',   # Rice
            45: 'SORG_A',   # Sorghum
            57: 'WHEA_A',   # Wheat
        }
        
        column_names = []
        for idx in crop_indices:
            if idx in crop_map:
                column_names.append(crop_map[idx])
            else:
                logger.warning(f"Crop index {idx} not found in mapping, skipping")
        
        return column_names
    
    def get_crop_production(
        self, 
        grid_cells: pd.DataFrame, 
        crop_indices: List[int],
        convert_to_kcal: bool = True
    ) -> float:
        """
        Sum production across selected crops for given grid cells.
        
        Uses vectorized NumPy operations for optimal performance.
        
        Args:
            grid_cells: DataFrame of grid cells (from production data)
            crop_indices: List of 1-based crop indices to aggregate
            convert_to_kcal: If True, convert from metric tons to kcal
        
        Returns:
            Total production (in kcal if convert_to_kcal=True, else metric tons)
        """
        if len(grid_cells) == 0:
            return 0.0
        
        # Get crop column names
        crop_columns = self._get_crop_column_names(crop_indices)
        
        if not crop_columns:
            logger.warning("No valid crop columns found")
            return 0.0
        
        # Check which columns exist in the data
        available_columns = [col for col in crop_columns if col in grid_cells.columns]
        
        if not available_columns:
            logger.warning(f"None of the requested crop columns found in data: {crop_columns}")
            return 0.0
        
        # Use vectorized NumPy sum for better performance
        # This is faster than pandas sum().sum() for large datasets
        total_production_mt = np.nansum(grid_cells[available_columns].values)
        
        if convert_to_kcal:
            # Convert metric tons to grams to kcal using vectorized operation
            caloric_content = self.config.get_caloric_content()  # kcal/g
            grams_per_mt = self.config.get_unit_conversions()['grams_per_metric_ton']
            total_production_kcal = total_production_mt * grams_per_mt * caloric_content
            
            logger.debug(
                f"Production: {total_production_mt:.2f} MT = "
                f"{total_production_kcal:.2e} kcal "
                f"(caloric content: {caloric_content} kcal/g)"
            )
            
            return float(total_production_kcal)
        else:
            return float(total_production_mt)
    
    def get_crop_harvest_area(
        self, 
        grid_cells: pd.DataFrame, 
        crop_indices: List[int]
    ) -> float:
        """
        Sum harvest area across selected crops for given grid cells.
        
        Uses vectorized NumPy operations for optimal performance.
        
        Args:
            grid_cells: DataFrame of grid cells (from harvest area data)
            crop_indices: List of 1-based crop indices to aggregate
        
        Returns:
            Total harvest area in hectares
        """
        if len(grid_cells) == 0:
            return 0.0
        
        # Get crop column names
        crop_columns = self._get_crop_column_names(crop_indices)
        
        if not crop_columns:
            logger.warning("No valid crop columns found")
            return 0.0
        
        # Check which columns exist in the data
        available_columns = [col for col in crop_columns if col in grid_cells.columns]
        
        if not available_columns:
            logger.warning(f"None of the requested crop columns found in data: {crop_columns}")
            return 0.0
        
        # Use vectorized NumPy sum for better performance
        total_harvest_area_ha = np.nansum(grid_cells[available_columns].values)
        
        logger.debug(f"Harvest area: {total_harvest_area_ha:.2f} hectares")
        
        return float(total_harvest_area_ha)
    
    def validate_grid_data(self) -> Dict[str, any]:
        """
        Validate grid data quality and completeness.
        
        Returns:
            Dictionary with validation results and metrics
        
        Raises:
            RuntimeError: If data not loaded
        """
        if not self._data_loaded:
            raise RuntimeError("SPAM data not loaded. Call load_spam_data() first.")
        
        logger.info("Validating grid data...")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # 1. Check coordinate ranges
        x_min = self.production_df['x'].min()
        x_max = self.production_df['x'].max()
        y_min = self.production_df['y'].min()
        y_max = self.production_df['y'].max()
        
        validation_results['metrics']['x_range'] = (float(x_min), float(x_max))
        validation_results['metrics']['y_range'] = (float(y_min), float(y_max))
        
        if not (-180 <= x_min and x_max <= 180):
            validation_results['errors'].append(
                f"Invalid longitude range: ({x_min}, {x_max}), expected [-180, 180]"
            )
            validation_results['valid'] = False
        
        if not (-90 <= y_min and y_max <= 90):
            validation_results['errors'].append(
                f"Invalid latitude range: ({y_min}, {y_max}), expected [-90, 90]"
            )
            validation_results['valid'] = False
        
        # 2. Check for missing coordinates
        prod_missing_x = self.production_df['x'].isna().sum()
        prod_missing_y = self.production_df['y'].isna().sum()
        harvest_missing_x = self.harvest_area_df['x'].isna().sum()
        harvest_missing_y = self.harvest_area_df['y'].isna().sum()
        
        validation_results['metrics']['missing_coordinates'] = {
            'production_x': int(prod_missing_x),
            'production_y': int(prod_missing_y),
            'harvest_x': int(harvest_missing_x),
            'harvest_y': int(harvest_missing_y)
        }
        
        if prod_missing_x > 0 or prod_missing_y > 0:
            validation_results['warnings'].append(
                f"Production data has {prod_missing_x} missing x and {prod_missing_y} missing y coordinates"
            )
        
        if harvest_missing_x > 0 or harvest_missing_y > 0:
            validation_results['warnings'].append(
                f"Harvest area data has {harvest_missing_x} missing x and {harvest_missing_y} missing y coordinates"
            )
        
        # 3. Check for missing FIPS0 (ISO3) codes
        prod_missing_iso3 = self.production_df['FIPS0'].isna().sum()
        harvest_missing_iso3 = self.harvest_area_df['FIPS0'].isna().sum()
        
        validation_results['metrics']['missing_iso3'] = {
            'production': int(prod_missing_iso3),
            'harvest': int(harvest_missing_iso3)
        }
        
        if prod_missing_iso3 > 0:
            validation_results['warnings'].append(
                f"Production data has {prod_missing_iso3} missing ISO3 codes"
            )
        
        if harvest_missing_iso3 > 0:
            validation_results['warnings'].append(
                f"Harvest area data has {harvest_missing_iso3} missing ISO3 codes"
            )
        
        # 4. Validate global production totals
        crop_indices = self.config.get_crop_indices()
        crop_columns = self._get_crop_column_names(crop_indices)
        
        if crop_columns:
            available_columns = [col for col in crop_columns if col in self.production_df.columns]
            
            if available_columns:
                total_production_mt = self.production_df[available_columns].sum().sum()
                total_harvest_area_ha = self.harvest_area_df[available_columns].sum().sum()
                
                validation_results['metrics']['total_production_mt'] = float(total_production_mt)
                validation_results['metrics']['total_harvest_area_ha'] = float(total_harvest_area_ha)
                
                # Expected ranges (rough estimates for validation)
                # These are approximate global totals for major grains
                expected_ranges = {
                    'allgrain': (2e9, 4e9),      # 2-4 billion MT
                    'wheat': (6e8, 9e8),         # 600-900 million MT
                    'rice': (7e8, 9e8),          # 700-900 million MT
                    'maize': (1e9, 1.5e9),       # 1-1.5 billion MT
                }
                
                crop_type = self.config.crop_type
                if crop_type in expected_ranges:
                    min_expected, max_expected = expected_ranges[crop_type]
                    
                    if total_production_mt < min_expected:
                        validation_results['warnings'].append(
                            f"Total production ({total_production_mt:.2e} MT) is below expected range "
                            f"for {crop_type}: [{min_expected:.2e}, {max_expected:.2e}]"
                        )
                    elif total_production_mt > max_expected:
                        validation_results['warnings'].append(
                            f"Total production ({total_production_mt:.2e} MT) is above expected range "
                            f"for {crop_type}: [{min_expected:.2e}, {max_expected:.2e}]"
                        )
                
                # Check for zero or negative values
                for col in available_columns:
                    negative_count = (self.production_df[col] < 0).sum()
                    if negative_count > 0:
                        validation_results['errors'].append(
                            f"Production column {col} has {negative_count} negative values"
                        )
                        validation_results['valid'] = False
                    
                    negative_harvest = (self.harvest_area_df[col] < 0).sum()
                    if negative_harvest > 0:
                        validation_results['errors'].append(
                            f"Harvest area column {col} has {negative_harvest} negative values"
                        )
                        validation_results['valid'] = False
        
        # 5. Check data consistency between production and harvest area
        if len(self.production_df) != len(self.harvest_area_df):
            validation_results['warnings'].append(
                f"Row count mismatch: production has {len(self.production_df)} rows, "
                f"harvest area has {len(self.harvest_area_df)} rows. "
                f"This is acceptable if coverage differs between datasets."
            )
        
        validation_results['metrics']['grid_cell_count'] = len(self.production_df)
        validation_results['metrics']['country_count'] = len(self.production_df['FIPS0'].unique())
        
        # Log summary
        if validation_results['valid']:
            logger.info("Grid data validation PASSED")
        else:
            logger.error(f"Grid data validation FAILED with {len(validation_results['errors'])} errors")
        
        if validation_results['warnings']:
            logger.warning(f"Grid data validation has {len(validation_results['warnings'])} warnings")
        
        return validation_results
    
    def generate_validation_report(self) -> str:
        """
        Generate human-readable validation report.
        
        Returns:
            Formatted validation report string
        
        Raises:
            RuntimeError: If data not loaded
        """
        validation_results = self.validate_grid_data()
        
        report_lines = [
            "=" * 60,
            "GRID DATA VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {'PASSED' if validation_results['valid'] else 'FAILED'}",
            f"SPAM Version: {self.config.get_spam_version()}",
            f"Crop Type: {self.config.crop_type}",
            "",
            "METRICS:",
            f"  Grid Cells: {validation_results['metrics'].get('grid_cell_count', 'N/A'):,}",
            f"  Countries: {validation_results['metrics'].get('country_count', 'N/A')}",
            f"  X Range: {validation_results['metrics'].get('x_range', 'N/A')}",
            f"  Y Range: {validation_results['metrics'].get('y_range', 'N/A')}",
        ]
        
        if 'total_production_mt' in validation_results['metrics']:
            report_lines.extend([
                f"  Total Production: {validation_results['metrics']['total_production_mt']:.2e} MT",
                f"  Total Harvest Area: {validation_results['metrics']['total_harvest_area_ha']:.2e} ha",
            ])
        
        if validation_results['errors']:
            report_lines.extend([
                "",
                f"ERRORS ({len(validation_results['errors'])}):",
            ])
            for error in validation_results['errors']:
                report_lines.append(f"  - {error}")
        
        if validation_results['warnings']:
            report_lines.extend([
                "",
                f"WARNINGS ({len(validation_results['warnings'])}):",
            ])
            for warning in validation_results['warnings']:
                report_lines.append(f"  - {warning}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def has_spatial_index(self) -> bool:
        """
        Check if spatial index has been created.
        
        Returns:
            True if spatial index exists
        """
        return self._spatial_index_created
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self._data_loaded else "not loaded"
        spatial = "indexed" if self._spatial_index_created else "not indexed"
        rows = len(self.production_df) if self._data_loaded else 0
        return f"GridDataManager(status={status}, spatial={spatial}, grid_cells={rows})"

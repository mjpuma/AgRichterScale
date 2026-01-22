"""Spatial Mapper for mapping geographic regions to SPAM grid cells."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
# geopandas and shapely imported lazily to avoid segfaults on environments without binary compatibility
# import geopandas as gpd
# from shapely.geometry import Point, Polygon

from ..core.config import Config
from .grid_manager import GridDataManager


logger = logging.getLogger(__name__)


class SpatialMapper:
    """
    Maps geographic regions (countries, states) to SPAM grid cells.
    
    Handles country code conversions and spatial matching for event
    loss calculations.
    """
    
    def __init__(self, config: Config, grid_manager: GridDataManager):
        """
        Initialize SpatialMapper.
        
        Args:
            config: Configuration object with data paths
            grid_manager: GridDataManager instance for querying grid cells
        """
        self.config = config
        self.grid_manager = grid_manager
        self.country_codes_mapping: Optional[pd.DataFrame] = None
        self.boundary_data_loaded = False
        # Type hint as Any for lazy import compatibility
        self.country_boundaries: Optional[Any] = None
        self.state_boundaries: Optional[Any] = None
        self._cache: Dict = {}
        self._country_grid_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._state_grid_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        
        logger.info("Initialized SpatialMapper")
    
    def prebuild_country_mappings(self) -> None:
        """
        Pre-build and cache all country-to-grid-cell mappings.
        
        This method builds mappings for all countries in the SPAM data
        at once, which is more efficient than building them on-demand
        when processing multiple events.
        """
        if not self.grid_manager.is_loaded():
            logger.warning("Grid data not loaded, cannot prebuild mappings")
            return
        
        import time
        start_time = time.time()
        
        logger.info("Pre-building country-to-grid-cell mappings...")
        
        # Get all unique FIPS codes from SPAM data
        production_df = self.grid_manager.get_production_data()
        harvest_df = self.grid_manager.get_harvest_area_data()
        
        unique_fips = production_df['FIPS0'].unique()
        
        # Use vectorized groupby for efficient mapping
        logger.info(f"Building mappings for {len(unique_fips)} countries...")
        
        # Group by FIPS0 once for all countries (vectorized operation)
        prod_grouped = production_df.groupby('FIPS0', observed=True)
        harv_grouped = harvest_df.groupby('FIPS0', observed=True)
        
        # Cache all groups
        for fips_code in unique_fips:
            if pd.notna(fips_code):
                cache_key = f"fips_{fips_code}"
                try:
                    prod_cells = prod_grouped.get_group(fips_code)
                    harv_cells = harv_grouped.get_group(fips_code)
                    self._country_grid_cache[cache_key] = (prod_cells, harv_cells)
                except KeyError:
                    # Country exists in one dataset but not the other
                    pass
        
        build_time = time.time() - start_time
        logger.info(
            f"Pre-built mappings for {len(self._country_grid_cache)} countries "
            f"in {build_time:.2f} seconds"
        )
    
    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self._cache.clear()
        self._country_grid_cache.clear()
        self._state_grid_cache.clear()
        logger.info("All caches cleared")
    
    def load_country_codes_mapping(self) -> pd.DataFrame:
        """
        Load country code conversion table.
        
        Reads CountryCode_Convert.xls which maps between different
        country code systems (GDAM, FAOSTAT, ISO3, etc.).
        
        Returns:
            DataFrame with country code mappings
        
        Raises:
            FileNotFoundError: If country code file doesn't exist
            ValueError: If file format is invalid
        """
        if self.country_codes_mapping is not None:
            logger.info("Country codes mapping already loaded, returning cached data")
            return self.country_codes_mapping
        
        file_paths = self.config.get_file_paths()
        country_codes_file = file_paths['country_codes']
        
        if not country_codes_file.exists():
            raise FileNotFoundError(
                f"Country codes file not found: {country_codes_file}\n"
                f"Expected file: CountryCode_Convert.xls"
            )
        
        logger.info(f"Loading country codes mapping from {country_codes_file}")
        
        try:
            self.country_codes_mapping = pd.read_excel(country_codes_file)
            logger.info(f"Loaded {len(self.country_codes_mapping)} country code mappings")
        except Exception as e:
            raise ValueError(f"Failed to load country codes mapping: {e}")
        
        # Validate required columns
        required_columns = ['Country', 'GDAM ', 'ISO3 alpha']
        missing_columns = [col for col in required_columns if col not in self.country_codes_mapping.columns]
        
        if missing_columns:
            raise ValueError(
                f"Country codes mapping missing required columns: {missing_columns}\n"
                f"Available columns: {self.country_codes_mapping.columns.tolist()}"
            )
        
        # Log available code systems
        logger.info(f"Available code systems: {self.country_codes_mapping.columns.tolist()}")
        
        # Validate mapping completeness
        self._validate_country_code_mapping()
        
        return self.country_codes_mapping
    
    def _validate_country_code_mapping(self) -> None:
        """
        Validate country code mapping completeness.
        
        Checks for missing values and logs warnings for incomplete mappings.
        """
        if self.country_codes_mapping is None:
            return
        
        # Check for missing ISO3 codes
        missing_iso3 = self.country_codes_mapping['ISO3 alpha'].isna().sum()
        if missing_iso3 > 0:
            logger.warning(
                f"{missing_iso3} countries have missing ISO3 alpha codes"
            )
        
        # Check for missing GDAM codes
        missing_gdam = self.country_codes_mapping['GDAM '].isna().sum()
        if missing_gdam > 0:
            logger.warning(
                f"{missing_gdam} countries have missing GDAM codes"
            )
        
        # Check for duplicate ISO3 codes
        iso3_counts = self.country_codes_mapping['ISO3 alpha'].value_counts()
        duplicates = iso3_counts[iso3_counts > 1]
        if len(duplicates) > 0:
            logger.warning(
                f"Found {len(duplicates)} duplicate ISO3 codes: {duplicates.index.tolist()}"
            )
        
        logger.info("Country code mapping validation complete")
    
    def _build_iso3_to_fips_mapping(self) -> Dict[str, str]:
        """
        Build ISO3 to FIPS mapping from SPAM data and country codes mapping.
        
        This dynamically creates the mapping by matching country names
        between the CountryCode_Convert table and SPAM data.
        
        Returns:
            Dictionary mapping ISO3 codes to FIPS codes
        """
        if not self.grid_manager.is_loaded():
            logger.warning("Grid data not loaded, cannot build FIPS mapping")
            return {}
        
        # Get unique countries from SPAM data
        production_df = self.grid_manager.get_production_data()
        spam_countries = production_df[['FIPS0', 'ADM0_NAME']].drop_duplicates()
        
        # Build mapping
        iso3_to_fips = {}
        
        for _, spam_row in spam_countries.iterrows():
            fips_code = spam_row['FIPS0']
            spam_name = spam_row['ADM0_NAME'].lower().strip()
            
            # Try to find matching country in mapping table
            best_match = None
            best_match_score = 0
            
            for _, mapping_row in self.country_codes_mapping.iterrows():
                mapping_name = str(mapping_row['Country']).lower().strip()
                iso3 = mapping_row['ISO3 alpha']
                
                if pd.isna(iso3):
                    continue
                
                # Exact match is best
                if spam_name == mapping_name:
                    iso3_to_fips[iso3] = fips_code
                    best_match = None  # Found exact match, stop looking
                    break
                
                # Calculate match score for partial matches
                # Prefer longer matches and matches at start of string
                if spam_name in mapping_name:
                    score = len(spam_name) / len(mapping_name)
                    if mapping_name.startswith(spam_name):
                        score += 0.5
                    if score > best_match_score:
                        best_match_score = score
                        best_match = (iso3, fips_code)
                elif mapping_name in spam_name:
                    score = len(mapping_name) / len(spam_name)
                    if spam_name.startswith(mapping_name):
                        score += 0.5
                    if score > best_match_score:
                        best_match_score = score
                        best_match = (iso3, fips_code)
            
            # Use best match if we found one and didn't find exact match
            if best_match and best_match_score > 0.7:  # Require 70% match
                iso3_to_fips[best_match[0]] = best_match[1]
        
        logger.info(f"Built ISO3→FIPS mapping for {len(iso3_to_fips)} countries")
        return iso3_to_fips
    
    def get_fips_from_country_code(
        self,
        country_code: float,
        code_system: str = 'GDAM '
    ) -> Optional[str]:
        """
        Convert country code to FIPS code (used by SPAM data).
        
        SPAM 2020 uses FIPS codes in the FIPS0 column (e.g., 'US', 'CH', 'IN').
        This method maps from GDAM or other code systems to FIPS by:
        1. Converting to ISO3
        2. Looking up FIPS from dynamically built mapping
        
        Args:
            country_code: Numeric country code
            code_system: Source code system ('GDAM ', 'FAOSTAT', etc.)
        
        Returns:
            FIPS code (e.g., 'US', 'CH') or None if not found
        """
        # First get ISO3 alpha code
        iso3_code = self.get_iso3_from_country_code(country_code, code_system)
        
        if not iso3_code:
            return None
        
        # Build mapping if not already cached
        cache_key = 'iso3_to_fips_mapping'
        if cache_key not in self._cache:
            self._cache[cache_key] = self._build_iso3_to_fips_mapping()
        
        iso3_to_fips = self._cache[cache_key]
        
        # Look up FIPS code
        fips_code = iso3_to_fips.get(iso3_code)
        
        if fips_code:
            logger.debug(f"Mapped ISO3 {iso3_code} to FIPS {fips_code}")
            return fips_code
        
        # Fallback: try common patterns
        # Some FIPS codes are just first 2 letters of ISO3
        fallback_fips = iso3_code[:2]
        logger.warning(
            f"No FIPS mapping found for ISO3 {iso3_code}, "
            f"trying fallback: {fallback_fips}"
        )
        
        # Verify fallback exists in SPAM data
        if self.grid_manager.is_loaded():
            try:
                prod_cells, _ = self.grid_manager.get_grid_cells_by_iso3(fallback_fips)
                if len(prod_cells) > 0:
                    logger.info(f"Fallback FIPS {fallback_fips} found in SPAM data")
                    # Cache this for future use
                    iso3_to_fips[iso3_code] = fallback_fips
                    return fallback_fips
            except:
                pass
        
        # If dynamic mapping didn't find it, use complete ISO3→FIPS mapping
        # This mapping was generated from SPAM 2020 data
        complete_mapping = {
            'AFG': 'AF', 'AGO': 'AO', 'ALB': 'AL', 'ARE': 'TC', 'ARG': 'AR',
            'ARM': 'AM', 'ATG': 'AC', 'AUS': 'AS', 'AUT': 'AU', 'AZE': 'AJ',
            'BDI': 'BY', 'BEL': 'BE', 'BEN': 'BN', 'BFA': 'UV', 'BGD': 'BG',
            'BGR': 'BU', 'BHR': 'BA', 'BHS': 'BF', 'BIH': 'BK', 'BLR': 'BO',
            'BLZ': 'BH', 'BOL': 'BL', 'BRA': 'BR', 'BRB': 'BB', 'BRN': 'BX',
            'BTN': 'BT', 'BWA': 'BC', 'CAF': 'CT', 'CAN': 'CA', 'CHE': 'SZ',
            'CHL': 'CI', 'CHN': 'CH', 'CMR': 'CM', 'COD': 'CF', 'COG': 'CG',
            'COL': 'CO', 'CRI': 'CS', 'CSK': 'LO', 'CUB': 'CU', 'CYP': 'CY',
            'CZE': 'EZ', 'DEU': 'GM', 'DJI': 'DJ', 'DMA': 'DR', 'DNK': 'DA',
            'DOM': 'DO', 'DZA': 'AG', 'ECU': 'EC', 'EGY': 'EG', 'ERI': 'ER',
            'ESP': 'SP', 'EST': 'EN', 'ETH': 'ET', 'FIN': 'FI', 'FJI': 'FJ',
            'FRA': 'FR', 'GAB': 'GB', 'GBR': 'UK', 'GEO': 'GG', 'GHA': 'GH',
            'GIN': 'PU', 'GMB': 'GA', 'GNQ': 'GV', 'GRC': 'GR', 'GRD': 'GJ',
            'GTM': 'GT', 'GUY': 'GY', 'HND': 'HO', 'HRV': 'HR', 'HTI': 'HA',
            'HUN': 'HU', 'IDN': 'ID', 'IND': 'IN', 'IRL': 'EI', 'IRN': 'IR',
            'IRQ': 'IZ', 'ISL': 'IC', 'ISR': 'IS', 'ITA': 'IT', 'JAM': 'JM',
            'JOR': 'JO', 'JPN': 'JA', 'KAZ': 'KZ', 'KEN': 'KE', 'KGZ': 'KG',
            'KHM': 'CB', 'KNA': 'SC', 'KOR': 'KS', 'KWT': 'KU', 'LAO': 'LA',
            'LBN': 'LE', 'LBR': 'LI', 'LBY': 'LY', 'LCA': 'ST', 'LKA': 'CE',
            'LSO': 'LT', 'LTU': 'LH', 'LUX': 'LU', 'LVA': 'LG', 'MAR': 'MO',
            'MDA': 'MD', 'MDG': 'MA', 'MEX': 'MX', 'MKD': 'MK', 'MLI': 'ML',
            'MLT': 'MT', 'MMR': 'BM', 'MNE': 'MW', 'MNG': 'MG', 'MOZ': 'MZ',
            'MRT': 'MR', 'MUS': 'MP', 'MWI': 'MI', 'MYS': 'MY', 'NAM': 'WA',
            'NCL': 'NC', 'NER': 'NG', 'NGA': 'NI', 'NIC': 'NU', 'NLD': 'NL',
            'NOR': 'NO', 'NPL': 'NP', 'NZL': 'NZ', 'OMN': 'MU', 'PAK': 'PK',
            'PAN': 'PM', 'PER': 'PE', 'PHL': 'RP', 'PNG': 'PP', 'POL': 'PL',
            'PRI': 'PQ', 'PRK': 'KN', 'PRT': 'PO', 'PRY': 'PA', 'PSE': 'WE',
            'QAT': 'QA', 'ROU': 'RO', 'RUS': 'RS', 'RWA': 'RW', 'SAU': 'SA',
            'SDN': 'SU', 'SEN': 'SG', 'SLB': 'BP', 'SLE': 'SL', 'SLV': 'ES',
            'SOM': 'SO', 'SRB': 'RI', 'SSD': 'OD', 'STP': 'TP', 'SUR': 'NS',
            'SVK': 'LO', 'SVN': 'SI', 'SWE': 'SW', 'SWZ': 'WZ', 'SYR': 'SY',
            'TCD': 'CD', 'TGO': 'TG', 'THA': 'TH', 'TJK': 'TI', 'TKM': 'TX',
            'TLS': 'TT', 'TTO': 'TD', 'TUN': 'TS', 'TUR': 'TU', 'TZA': 'TZ',
            'UGA': 'UG', 'UKR': 'UP', 'URY': 'UY', 'USA': 'US', 'UZB': 'UZ',
            'VCT': 'VC', 'VEN': 'VE', 'VNM': 'VM', 'VUT': 'NH', 'YEM': 'YM',
            'ZAF': 'SF', 'ZMB': 'ZA', 'ZWE': 'ZI',
        }
        
        # Try complete mapping as final fallback
        fips_from_complete = complete_mapping.get(iso3_code)
        if fips_from_complete:
            logger.info(f"Found FIPS {fips_from_complete} for ISO3 {iso3_code} in complete mapping")
            # Cache this for future use
            iso3_to_fips[iso3_code] = fips_from_complete
            return fips_from_complete
        
        # No mapping found
        logger.error(f"No FIPS mapping found for ISO3 {iso3_code}")
        return None
    
    def get_iso3_from_country_code(
        self, 
        country_code: float, 
        code_system: str = 'GDAM '
    ) -> Optional[str]:
        """
        Convert country code to ISO3 alpha code.
        
        Args:
            country_code: Numeric country code
            code_system: Source code system ('GDAM ', 'FAOSTAT', etc.')
        
        Returns:
            ISO3 alpha code (e.g., 'USA', 'CHN') or None if not found
        """
        if self.country_codes_mapping is None:
            self.load_country_codes_mapping()
        
        # Check cache first
        cache_key = f"code_{code_system}_{country_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Handle NaN country codes
        if pd.isna(country_code):
            logger.warning("Received NaN country code")
            return None
        
        # Validate code system exists
        if code_system not in self.country_codes_mapping.columns:
            logger.error(
                f"Code system '{code_system}' not found in mapping. "
                f"Available: {self.country_codes_mapping.columns.tolist()}"
            )
            return None
        
        # Find matching row
        matches = self.country_codes_mapping[
            self.country_codes_mapping[code_system] == country_code
        ]
        
        if len(matches) == 0:
            logger.warning(
                f"No ISO3 code found for {code_system} code: {country_code}"
            )
            return None
        
        if len(matches) > 1:
            logger.warning(
                f"Multiple matches found for {code_system} code {country_code}, "
                f"using first match"
            )
        
        iso3_code = matches.iloc[0]['ISO3 alpha']
        
        # Handle NaN ISO3 codes
        if pd.isna(iso3_code):
            country_name = matches.iloc[0]['Country']
            logger.warning(
                f"ISO3 code is NaN for {code_system} code {country_code} "
                f"(Country: {country_name})"
            )
            return None
        
        # Cache result
        self._cache[cache_key] = iso3_code
        
        logger.debug(f"Mapped {code_system} code {country_code} to ISO3: {iso3_code}")
        
        return iso3_code
    
    def get_country_name_from_code(
        self, 
        country_code: float, 
        code_system: str = 'GDAM '
    ) -> Optional[str]:
        """
        Get country name from country code.
        
        Args:
            country_code: Numeric country code
            code_system: Source code system
        
        Returns:
            Country name or None if not found
        """
        if self.country_codes_mapping is None:
            self.load_country_codes_mapping()
        
        if pd.isna(country_code):
            return None
        
        matches = self.country_codes_mapping[
            self.country_codes_mapping[code_system] == country_code
        ]
        
        if len(matches) == 0:
            return None
        
        return matches.iloc[0]['Country']
    
    def get_all_code_systems(self) -> List[str]:
        """
        Get list of all available country code systems.
        
        Returns:
            List of code system column names
        """
        if self.country_codes_mapping is None:
            self.load_country_codes_mapping()
        
        # Exclude non-code columns
        exclude_cols = ['Country']
        code_systems = [
            col for col in self.country_codes_mapping.columns 
            if col not in exclude_cols
        ]
        
        return code_systems
    
    def map_country_to_grid_cells(
        self, 
        country_code: float, 
        code_system: str = 'GDAM '
    ) -> Tuple[List[str], List[str]]:
        """
        Map country to SPAM grid cells.
        
        Uses FIPS code to query GridDataManager for all grid cells
        within the country boundaries.
        
        Args:
            country_code: Numeric country code
            code_system: Source code system (default: 'GDAM ')
        
        Returns:
            Tuple of (production_grid_cell_ids, harvest_area_grid_cell_ids)
            Returns empty lists if country not found or has no SPAM data
        """
        # Check cache first
        cache_key = f"country_cells_{code_system}_{country_code}"
        if cache_key in self._cache:
            logger.debug(f"Returning cached grid cells for country code {country_code}")
            return self._cache[cache_key]
        
        # Convert country code to FIPS (SPAM uses FIPS codes)
        fips_code = self.get_fips_from_country_code(country_code, code_system)
        
        if fips_code is None:
            country_name = self.get_country_name_from_code(country_code, code_system)
            logger.warning(
                f"Could not map country code {country_code} to FIPS. "
                f"Country: {country_name or 'Unknown'}"
            )
            return [], []
        
        # Ensure grid data is loaded
        if not self.grid_manager.is_loaded():
            logger.info("Grid data not loaded, loading now...")
            self.grid_manager.load_spam_data()
        
        # Query grid cells by FIPS code (SPAM uses FIPS0 column)
        try:
            production_cells, harvest_area_cells = self.grid_manager.get_grid_cells_by_iso3(fips_code)
        except Exception as e:
            logger.error(f"Error querying grid cells for FIPS {fips_code}: {e}")
            return [], []
        
        # Extract grid cell IDs
        production_ids = production_cells['grid_code'].astype(str).tolist() if len(production_cells) > 0 else []
        harvest_area_ids = harvest_area_cells['grid_code'].astype(str).tolist() if len(harvest_area_cells) > 0 else []
        
        # Log mapping statistics
        country_name = self.get_country_name_from_code(country_code, code_system)
        if len(production_ids) == 0:
            logger.warning(
                f"No SPAM grid cells found for {country_name or 'Unknown'} "
                f"(FIPS: {fips_code}, code: {country_code})"
            )
        else:
            logger.info(
                f"Mapped {country_name or 'Unknown'} (FIPS: {fips_code}) to "
                f"{len(production_ids)} production cells, {len(harvest_area_ids)} harvest area cells"
            )
        
        # Cache results
        result = (production_ids, harvest_area_ids)
        self._cache[cache_key] = result
        
        return result
    
    def get_country_grid_cells_dataframe(
        self, 
        country_code: float, 
        code_system: str = 'GDAM '
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get full DataFrames of grid cells for a country.
        
        Args:
            country_code: Numeric country code
            code_system: Source code system
        
        Returns:
            Tuple of (production_df, harvest_area_df)
        """
        # Convert country code to FIPS (SPAM uses FIPS codes)
        fips_code = self.get_fips_from_country_code(country_code, code_system)
        
        if fips_code is None:
            country_name = self.get_country_name_from_code(country_code, code_system)
            logger.warning(f"Could not map country code {country_code} to FIPS (Country: {country_name})")
            return pd.DataFrame(), pd.DataFrame()
        
        # Ensure grid data is loaded
        if not self.grid_manager.is_loaded():
            self.grid_manager.load_spam_data()
        
        # Query grid cells by FIPS code
        try:
            production_cells, harvest_area_cells = self.grid_manager.get_grid_cells_by_iso3(fips_code)
            return production_cells, harvest_area_cells
        except Exception as e:
            logger.error(f"Error querying grid cells for FIPS {fips_code}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def load_boundary_data(
        self, 
        country_shapefile: Optional[Path] = None,
        state_shapefile: Optional[Path] = None
    ) -> None:
        """
        Load GDAM boundary shapefiles (optional).
        
        This method provides optional support for precise boundary-based
        spatial matching. If shapefiles are not available, the system
        falls back to ISO3 code matching which is faster and usually sufficient.
        
        Args:
            country_shapefile: Path to country boundary shapefile
            state_shapefile: Path to state/province boundary shapefile
        
        Note:
            Boundary data is OPTIONAL. The system works without it using
            ISO3 code matching from SPAM data.
        """
        logger.info("Loading boundary data (optional)...")
        
        try:
            import geopandas as gpd
        except ImportError:
            logger.warning("geopandas not available, cannot load boundary shapefiles")
            return
            
        # Try to load country boundaries
        if country_shapefile and country_shapefile.exists():
            try:
                logger.info(f"Loading country boundaries from {country_shapefile}")
                self.country_boundaries = gpd.read_file(country_shapefile)
                
                # Ensure WGS84 CRS
                if self.country_boundaries.crs != 'EPSG:4326':
                    logger.info(
                        f"Transforming country boundaries from {self.country_boundaries.crs} "
                        f"to EPSG:4326"
                    )
                    self.country_boundaries = self.country_boundaries.to_crs('EPSG:4326')
                
                logger.info(f"Loaded {len(self.country_boundaries)} country boundaries")
            except Exception as e:
                logger.warning(f"Failed to load country boundaries: {e}")
                self.country_boundaries = None
        else:
            logger.info("Country shapefile not provided or doesn't exist, using ISO3 matching only")
        
        # Try to load state boundaries
        if state_shapefile and state_shapefile.exists():
            try:
                logger.info(f"Loading state boundaries from {state_shapefile}")
                self.state_boundaries = gpd.read_file(state_shapefile)
                
                # Ensure WGS84 CRS
                if self.state_boundaries.crs != 'EPSG:4326':
                    logger.info(
                        f"Transforming state boundaries from {self.state_boundaries.crs} "
                        f"to EPSG:4326"
                    )
                    self.state_boundaries = self.state_boundaries.to_crs('EPSG:4326')
                
                logger.info(f"Loaded {len(self.state_boundaries)} state boundaries")
            except Exception as e:
                logger.warning(f"Failed to load state boundaries: {e}")
                self.state_boundaries = None
        else:
            logger.info("State shapefile not provided or doesn't exist, using name matching only")
        
        self.boundary_data_loaded = (
            self.country_boundaries is not None or 
            self.state_boundaries is not None
        )
        
        if self.boundary_data_loaded:
            logger.info("Boundary data loaded successfully")
        else:
            logger.info(
                "No boundary data loaded. System will use ISO3 code matching, "
                "which is faster and usually sufficient."
            )
    
    def map_country_to_grid_cells_spatial(
        self, 
        country_code: float,
        code_system: str = 'GDAM '
    ) -> Tuple[List[str], List[str]]:
        """
        Map country to grid cells using spatial intersection (if boundaries loaded).
        
        This is a fallback method that uses precise spatial intersection
        with boundary shapefiles. Only used if boundary data is loaded.
        
        Args:
            country_code: Numeric country code
            code_system: Source code system
        
        Returns:
            Tuple of (production_grid_cell_ids, harvest_area_grid_cell_ids)
        """
        if self.country_boundaries is None:
            logger.warning(
                "Country boundaries not loaded, cannot perform spatial intersection. "
                "Use map_country_to_grid_cells() for ISO3 matching."
            )
            return [], []
        
        # Get ISO3 code for matching with boundary data
        iso3_code = self.get_iso3_from_country_code(country_code, code_system)
        
        if iso3_code is None:
            logger.warning(f"Could not map country code {country_code} to ISO3")
            return [], []
        
        # Find country boundary
        # Note: Boundary data column names may vary, adjust as needed
        country_boundary = None
        for col in ['ISO3', 'ISO_A3', 'iso3', 'ISO3_CODE']:
            if col in self.country_boundaries.columns:
                matches = self.country_boundaries[self.country_boundaries[col] == iso3_code]
                if len(matches) > 0:
                    country_boundary = matches.iloc[0].geometry
                    break
        
        if country_boundary is None:
            logger.warning(f"No boundary found for ISO3: {iso3_code}")
            return [], []
        
        # Ensure grid spatial index is created
        if not self.grid_manager.has_spatial_index():
            logger.info("Creating spatial index for grid cells...")
            self.grid_manager.create_spatial_index()
        
        # Get grid GeoDataFrames
        production_gdf = self.grid_manager.production_gdf
        harvest_area_gdf = self.grid_manager.harvest_area_gdf
        
        # Perform spatial intersection
        logger.info(f"Performing spatial intersection for {iso3_code}...")
        
        production_within = production_gdf[production_gdf.within(country_boundary)]
        harvest_area_within = harvest_area_gdf[harvest_area_gdf.within(country_boundary)]
        
        production_ids = production_within['grid_code'].astype(str).tolist()
        harvest_area_ids = harvest_area_within['grid_code'].astype(str).tolist()
        
        logger.info(
            f"Spatial intersection found {len(production_ids)} production cells, "
            f"{len(harvest_area_ids)} harvest area cells for {iso3_code}"
        )
        
        return production_ids, harvest_area_ids
    
    def map_country_with_fallback(
        self,
        country_code: float,
        code_system: str = 'GDAM ',
        prefer_spatial: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Map country to grid cells with fallback logic.
        
        Tries ISO3 matching first (fast), then spatial intersection if
        boundaries are available and ISO3 matching fails or prefer_spatial=True.
        
        Args:
            country_code: Numeric country code
            code_system: Source code system
            prefer_spatial: If True, try spatial intersection first
        
        Returns:
            Tuple of (production_grid_cell_ids, harvest_area_grid_cell_ids)
        """
        if prefer_spatial and self.country_boundaries is not None:
            # Try spatial first
            logger.debug("Trying spatial intersection first (prefer_spatial=True)")
            prod_ids, harv_ids = self.map_country_to_grid_cells_spatial(country_code, code_system)
            
            if len(prod_ids) > 0:
                return prod_ids, harv_ids
            
            # Fall back to ISO3 matching
            logger.debug("Spatial intersection found no cells, falling back to ISO3 matching")
        
        # Try ISO3 matching (default, faster)
        prod_ids, harv_ids = self.map_country_to_grid_cells(country_code, code_system)
        
        if len(prod_ids) > 0:
            return prod_ids, harv_ids
        
        # If ISO3 failed and boundaries available, try spatial
        if self.country_boundaries is not None:
            logger.debug("ISO3 matching found no cells, trying spatial intersection")
            return self.map_country_to_grid_cells_spatial(country_code, code_system)
        
        return [], []
    
    def map_state_to_grid_cells(
        self,
        country_code: float,
        state_codes: List[float],
        code_system: str = 'GDAM '
    ) -> Tuple[List[str], List[str]]:
        """
        Map state/province to grid cells.
        
        Filters grid cells by both country (ISO3) and state name (ADM1_NAME).
        Supports both numeric state codes and string state names.
        
        Args:
            country_code: Numeric country code
            state_codes: List of numeric state codes or state names
            code_system: Source code system
        
        Returns:
            Tuple of (production_grid_cell_ids, harvest_area_grid_cell_ids)
        """
        # Check cache first
        cache_key = f"state_cells_{code_system}_{country_code}_{hash(tuple(state_codes))}"
        if cache_key in self._cache:
            logger.debug(f"Returning cached grid cells for state codes {state_codes}")
            return self._cache[cache_key]
        
        # Get FIPS code (SPAM uses FIPS codes)
        fips_code = self.get_fips_from_country_code(country_code, code_system)
        
        if fips_code is None:
            country_name = self.get_country_name_from_code(country_code, code_system)
            logger.warning(f"Could not map country code {country_code} to FIPS (Country: {country_name})")
            return [], []
        
        # Ensure grid data is loaded
        if not self.grid_manager.is_loaded():
            self.grid_manager.load_spam_data()
        
        # Get all grid cells for the country using FIPS code
        production_cells, harvest_area_cells = self.grid_manager.get_grid_cells_by_iso3(fips_code)
        
        if len(production_cells) == 0:
            logger.warning(f"No grid cells found for country FIPS: {fips_code}")
            return [], []
        
        # Filter by state codes/names
        # State codes could be numeric (FIPS1) or state names (ADM1_NAME)
        production_state_cells = self._filter_by_state(production_cells, state_codes)
        harvest_area_state_cells = self._filter_by_state(harvest_area_cells, state_codes)
        
        # Extract grid cell IDs
        production_ids = production_state_cells['grid_code'].astype(str).tolist() if len(production_state_cells) > 0 else []
        harvest_area_ids = harvest_area_state_cells['grid_code'].astype(str).tolist() if len(harvest_area_state_cells) > 0 else []
        
        # Log mapping statistics
        country_name = self.get_country_name_from_code(country_code, code_system)
        success_rate = len(production_ids) / len(production_cells) * 100 if len(production_cells) > 0 else 0
        
        if len(production_ids) == 0:
            logger.warning(
                f"No grid cells found for states {state_codes} in {country_name or 'Unknown'} "
                f"(FIPS: {fips_code})"
            )
        else:
            logger.info(
                f"Mapped {len(state_codes)} states in {country_name or 'Unknown'} to "
                f"{len(production_ids)} production cells, {len(harvest_area_ids)} harvest area cells "
                f"(success rate: {success_rate:.1f}%)"
            )
        
        # Cache results
        result = (production_ids, harvest_area_ids)
        self._cache[cache_key] = result
        
        return result
    
    def _filter_by_state(
        self,
        grid_cells: pd.DataFrame,
        state_codes: List[float]
    ) -> pd.DataFrame:
        """
        Filter grid cells by state codes or names.
        
        Args:
            grid_cells: DataFrame of grid cells
            state_codes: List of state codes (numeric or string)
        
        Returns:
            Filtered DataFrame
        """
        if len(grid_cells) == 0 or len(state_codes) == 0:
            return pd.DataFrame()
        
        # Try matching by FIPS1 (numeric state code)
        if 'FIPS1' in grid_cells.columns:
            # Convert state_codes to numeric if possible
            numeric_codes = []
            for code in state_codes:
                try:
                    numeric_codes.append(float(code))
                except (ValueError, TypeError):
                    pass
            
            if numeric_codes:
                fips1_matches = grid_cells[grid_cells['FIPS1'].isin(numeric_codes)]
                if len(fips1_matches) > 0:
                    logger.debug(f"Matched {len(fips1_matches)} cells by FIPS1 code")
                    return fips1_matches
        
        # Try matching by ADM1_NAME (state name)
        if 'ADM1_NAME' in grid_cells.columns:
            # Convert state_codes to strings for name matching
            state_names = [str(code) for code in state_codes]
            
            # Try exact match first
            name_matches = grid_cells[grid_cells['ADM1_NAME'].isin(state_names)]
            if len(name_matches) > 0:
                logger.debug(f"Matched {len(name_matches)} cells by ADM1_NAME (exact)")
                return name_matches
            
            # Try case-insensitive partial match
            name_matches_list = []
            for state_name in state_names:
                state_name_lower = state_name.lower()
                matches = grid_cells[
                    grid_cells['ADM1_NAME'].str.lower().str.contains(state_name_lower, na=False)
                ]
                if len(matches) > 0:
                    name_matches_list.append(matches)
            
            if name_matches_list:
                name_matches = pd.concat(name_matches_list).drop_duplicates()
                logger.debug(f"Matched {len(name_matches)} cells by ADM1_NAME (partial)")
                return name_matches
        
        logger.warning(
            f"Could not match state codes {state_codes} using FIPS1 or ADM1_NAME. "
            f"Available columns: {grid_cells.columns.tolist()}"
        )
        
        return pd.DataFrame()
    
    def get_state_grid_cells_dataframe(
        self,
        country_code: float,
        state_codes: List[float],
        code_system: str = 'GDAM '
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get full DataFrames of grid cells for states.
        
        Args:
            country_code: Numeric country code
            state_codes: List of state codes
            code_system: Source code system
        
        Returns:
            Tuple of (production_df, harvest_area_df)
        """
        # Get FIPS code
        fips_code = self.get_fips_from_country_code(country_code, code_system)
        
        if fips_code is None:
            country_name = self.get_country_name_from_code(country_code, code_system)
            logger.warning(f"Could not map country code {country_code} to FIPS (Country: {country_name})")
            return pd.DataFrame(), pd.DataFrame()
        
        # Ensure grid data is loaded
        if not self.grid_manager.is_loaded():
            self.grid_manager.load_spam_data()
        
        # Get all grid cells for the country
        production_cells, harvest_area_cells = self.grid_manager.get_grid_cells_by_iso3(fips_code)
        
        # Filter by state codes
        production_state_cells = self._filter_by_state(production_cells, state_codes)
        harvest_area_state_cells = self._filter_by_state(harvest_area_cells, state_codes)
        
        return production_state_cells, harvest_area_state_cells
    
    def validate_spatial_mapping(
        self,
        event_mappings: Dict[str, Tuple[List[str], List[str]]]
    ) -> Dict[str, any]:
        """
        Validate spatial mapping results.
        
        Calculates mapping success rates, coverage statistics, and identifies
        events with zero grid cells matched.
        
        Args:
            event_mappings: Dictionary mapping event names to 
                           (production_ids, harvest_area_ids) tuples
        
        Returns:
            Dictionary with validation results and statistics
        """
        logger.info("Validating spatial mapping results...")
        
        validation_results = {
            'valid': True,
            'total_events': len(event_mappings),
            'events_with_cells': 0,
            'events_without_cells': 0,
            'total_production_cells': 0,
            'total_harvest_cells': 0,
            'events_with_zero_cells': [],
            'coverage_by_event': {},
            'warnings': [],
            'errors': []
        }
        
        # Analyze each event mapping
        for event_name, (prod_ids, harv_ids) in event_mappings.items():
            has_cells = len(prod_ids) > 0 or len(harv_ids) > 0
            
            if has_cells:
                validation_results['events_with_cells'] += 1
                validation_results['total_production_cells'] += len(prod_ids)
                validation_results['total_harvest_cells'] += len(harv_ids)
            else:
                validation_results['events_without_cells'] += 1
                validation_results['events_with_zero_cells'].append(event_name)
            
            # Store per-event coverage
            validation_results['coverage_by_event'][event_name] = {
                'production_cells': len(prod_ids),
                'harvest_area_cells': len(harv_ids),
                'has_data': has_cells
            }
        
        # Calculate success rate
        if validation_results['total_events'] > 0:
            success_rate = (
                validation_results['events_with_cells'] / 
                validation_results['total_events'] * 100
            )
            validation_results['success_rate_percent'] = success_rate
        else:
            validation_results['success_rate_percent'] = 0.0
        
        # Generate warnings for events with zero cells
        if validation_results['events_without_cells'] > 0:
            validation_results['warnings'].append(
                f"{validation_results['events_without_cells']} events have zero grid cells matched: "
                f"{validation_results['events_with_zero_cells']}"
            )
        
        # Check if success rate is acceptable (>80%)
        if validation_results['success_rate_percent'] < 80:
            validation_results['warnings'].append(
                f"Low mapping success rate: {validation_results['success_rate_percent']:.1f}% "
                f"(expected >80%)"
            )
        
        # Calculate average cells per event
        if validation_results['events_with_cells'] > 0:
            avg_prod_cells = (
                validation_results['total_production_cells'] / 
                validation_results['events_with_cells']
            )
            avg_harv_cells = (
                validation_results['total_harvest_cells'] / 
                validation_results['events_with_cells']
            )
            validation_results['avg_production_cells_per_event'] = avg_prod_cells
            validation_results['avg_harvest_cells_per_event'] = avg_harv_cells
        
        # Log summary
        logger.info(
            f"Spatial mapping validation: {validation_results['events_with_cells']}/{validation_results['total_events']} "
            f"events have grid cells ({validation_results['success_rate_percent']:.1f}% success rate)"
        )
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logger.warning(warning)
        
        return validation_results
    
    def generate_spatial_mapping_report(
        self,
        event_mappings: Dict[str, Tuple[List[str], List[str]]]
    ) -> str:
        """
        Generate human-readable spatial mapping quality report.
        
        Args:
            event_mappings: Dictionary mapping event names to grid cell IDs
        
        Returns:
            Formatted report string
        """
        validation_results = self.validate_spatial_mapping(event_mappings)
        
        report_lines = [
            "=" * 60,
            "SPATIAL MAPPING QUALITY REPORT",
            "=" * 60,
            "",
            f"Total Events: {validation_results['total_events']}",
            f"Events with Grid Cells: {validation_results['events_with_cells']}",
            f"Events without Grid Cells: {validation_results['events_without_cells']}",
            f"Success Rate: {validation_results['success_rate_percent']:.1f}%",
            "",
            "COVERAGE STATISTICS:",
            f"  Total Production Cells: {validation_results['total_production_cells']:,}",
            f"  Total Harvest Area Cells: {validation_results['total_harvest_cells']:,}",
        ]
        
        if 'avg_production_cells_per_event' in validation_results:
            report_lines.extend([
                f"  Avg Production Cells/Event: {validation_results['avg_production_cells_per_event']:.1f}",
                f"  Avg Harvest Cells/Event: {validation_results['avg_harvest_cells_per_event']:.1f}",
            ])
        
        # List events with zero cells
        if validation_results['events_with_zero_cells']:
            report_lines.extend([
                "",
                f"EVENTS WITH ZERO GRID CELLS ({len(validation_results['events_with_zero_cells'])}):",
            ])
            for event_name in validation_results['events_with_zero_cells']:
                report_lines.append(f"  - {event_name}")
        
        # List events with cells (top 10 by cell count)
        events_with_data = [
            (name, data['production_cells']) 
            for name, data in validation_results['coverage_by_event'].items()
            if data['has_data']
        ]
        events_with_data.sort(key=lambda x: x[1], reverse=True)
        
        if events_with_data:
            report_lines.extend([
                "",
                f"TOP EVENTS BY GRID CELL COUNT (showing up to 10):",
            ])
            for event_name, cell_count in events_with_data[:10]:
                harv_count = validation_results['coverage_by_event'][event_name]['harvest_area_cells']
                report_lines.append(
                    f"  - {event_name}: {cell_count} production cells, {harv_count} harvest cells"
                )
        
        # Add warnings
        if validation_results['warnings']:
            report_lines.extend([
                "",
                f"WARNINGS ({len(validation_results['warnings'])}):",
            ])
            for warning in validation_results['warnings']:
                report_lines.append(f"  - {warning}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def get_mapping_statistics(self) -> Dict[str, any]:
        """
        Get statistics about loaded mappings and cache.
        
        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            'country_codes_loaded': self.country_codes_mapping is not None,
            'boundary_data_loaded': self.boundary_data_loaded,
            'cache_size': len(self._cache),
            'has_country_boundaries': self.country_boundaries is not None,
            'has_state_boundaries': self.state_boundaries is not None
        }
        
        if self.country_codes_mapping is not None:
            stats['total_countries'] = len(self.country_codes_mapping)
            stats['countries_with_iso3'] = self.country_codes_mapping['ISO3 alpha'].notna().sum()
            stats['countries_with_gdam'] = self.country_codes_mapping['GDAM '].notna().sum()
        
        if self.country_boundaries is not None:
            stats['country_boundary_count'] = len(self.country_boundaries)
        
        if self.state_boundaries is not None:
            stats['state_boundary_count'] = len(self.state_boundaries)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear internal cache to free memory."""
        self._cache.clear()
        logger.info("Spatial mapper cache cleared")
    
    def __repr__(self) -> str:
        """String representation."""
        mapping_status = "loaded" if self.country_codes_mapping is not None else "not loaded"
        boundary_status = "loaded" if self.boundary_data_loaded else "not loaded"
        return (
            f"SpatialMapper(mapping={mapping_status}, "
            f"boundaries={boundary_status}, "
            f"cache_size={len(self._cache)})"
        )

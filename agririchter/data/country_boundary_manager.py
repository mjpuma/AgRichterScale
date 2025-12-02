"""Country Boundary Manager for national-level agricultural analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..core.config import Config
from .spatial_mapper import SpatialMapper
from .grid_manager import GridDataManager


logger = logging.getLogger(__name__)


@dataclass
class CountryConfiguration:
    """Configuration for country-specific analysis."""
    
    country_code: str
    country_name: str
    fips_code: str  # SPAM uses FIPS codes
    iso3_code: str
    agricultural_focus: str  # 'food_security', 'export_capacity', 'efficiency'
    priority_crops: List[str]  # Most relevant crops for this country
    regional_subdivisions: Optional[List[str]] = None
    policy_scenarios: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_focuses = ['food_security', 'export_capacity', 'efficiency']
        if self.agricultural_focus not in valid_focuses:
            raise ValueError(f"agricultural_focus must be one of {valid_focuses}")


class CountryBoundaryManager:
    """
    Manages country boundary data and filtering for national agricultural analysis.
    
    Leverages existing SpatialMapper infrastructure and SPAM FIPS codes for
    efficient country-level data filtering without requiring external boundary files.
    """
    
    # Pre-defined country configurations for major agricultural producers
    # Note: Additional countries can be added using the CountryFramework
    COUNTRY_CONFIGURATIONS = {
        'USA': CountryConfiguration(
            country_code='USA',
            country_name='United States',
            fips_code='US',
            iso3_code='USA',
            agricultural_focus='export_capacity',
            priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['corn_belt', 'great_plains', 'california'],
            policy_scenarios=['drought_resilience', 'trade_disruption', 'climate_adaptation']
        ),
        'CHN': CountryConfiguration(
            country_code='CHN',
            country_name='China',
            fips_code='CH',
            iso3_code='CHN',
            agricultural_focus='food_security',
            priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['northeast', 'north_china_plain', 'yangtze_river'],
            policy_scenarios=['self_sufficiency', 'urbanization_pressure', 'water_scarcity']
        )
    }
    
    def __init__(self, config: Config, spatial_mapper: SpatialMapper, grid_manager: GridDataManager):
        """
        Initialize CountryBoundaryManager.
        
        Args:
            config: Configuration object
            spatial_mapper: SpatialMapper instance for country code handling
            grid_manager: GridDataManager instance for SPAM data access
        """
        self.config = config
        self.spatial_mapper = spatial_mapper
        self.grid_manager = grid_manager
        self._country_data_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._country_statistics_cache: Dict[str, Dict] = {}
        
        logger.info("Initialized CountryBoundaryManager")
    
    def get_country_configuration(self, country_code: str) -> Optional[CountryConfiguration]:
        """
        Get country configuration by country code.
        
        Args:
            country_code: Country code (e.g., 'USA', 'CHN')
        
        Returns:
            CountryConfiguration or None if not found
        """
        return self.COUNTRY_CONFIGURATIONS.get(country_code.upper())
    
    def get_available_countries(self) -> List[str]:
        """
        Get list of available countries with configurations.
        
        Returns:
            List of country codes
        """
        return list(self.COUNTRY_CONFIGURATIONS.keys())
    
    def get_country_data(
        self, 
        country_code: str,
        force_reload: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get SPAM data filtered for a specific country using FIPS codes.
        
        Args:
            country_code: Country code (e.g., 'USA', 'CHN')
            force_reload: Force reload from source data
        
        Returns:
            Tuple of (production_df, harvest_area_df) filtered for the country
        
        Raises:
            ValueError: If country not supported or no data found
        """
        country_code = country_code.upper()
        
        # Check cache first
        if not force_reload and country_code in self._country_data_cache:
            logger.debug(f"Returning cached data for {country_code}")
            return self._country_data_cache[country_code]
        
        # Get country configuration
        country_config = self.get_country_configuration(country_code)
        if country_config is None:
            raise ValueError(
                f"Country {country_code} not supported. "
                f"Available countries: {self.get_available_countries()}"
            )
        
        # Ensure grid data is loaded
        if not self.grid_manager.is_loaded():
            logger.info("Loading SPAM data for country filtering...")
            self.grid_manager.load_spam_data()
        
        # Get country data using FIPS code
        fips_code = country_config.fips_code
        logger.info(f"Filtering SPAM data for {country_config.country_name} (FIPS: {fips_code})")
        
        try:
            production_df, harvest_area_df = self.grid_manager.get_grid_cells_by_iso3(fips_code)
        except Exception as e:
            raise ValueError(f"Failed to get data for {country_code} (FIPS: {fips_code}): {e}")
        
        if len(production_df) == 0:
            raise ValueError(
                f"No SPAM data found for {country_config.country_name} (FIPS: {fips_code}). "
                f"Check if country code is correct."
            )
        
        # Align production and harvest area data to common grid cells
        # This is critical for multi-tier analysis to avoid broadcasting errors
        if len(production_df) != len(harvest_area_df):
            logger.info(
                f"Aligning data for {country_config.country_name}: "
                f"{len(production_df)} production cells, {len(harvest_area_df)} harvest area cells"
            )
            
            # Find common grid cells based on x,y coordinates
            prod_coords = set(zip(production_df['x'], production_df['y']))
            harv_coords = set(zip(harvest_area_df['x'], harvest_area_df['y']))
            common_coords = prod_coords.intersection(harv_coords)
            
            if len(common_coords) == 0:
                raise ValueError(
                    f"No common grid cells found between production and harvest area data for {country_config.country_name}"
                )
            
            # Filter both DataFrames to common coordinates
            common_coords_list = list(common_coords)
            prod_mask = production_df.set_index(['x', 'y']).index.isin(common_coords_list)
            harv_mask = harvest_area_df.set_index(['x', 'y']).index.isin(common_coords_list)
            
            production_df = production_df[prod_mask].reset_index(drop=True)
            harvest_area_df = harvest_area_df[harv_mask].reset_index(drop=True)
            
            # Sort both DataFrames by coordinates to ensure alignment
            production_df = production_df.sort_values(['x', 'y']).reset_index(drop=True)
            harvest_area_df = harvest_area_df.sort_values(['x', 'y']).reset_index(drop=True)
            
            logger.info(
                f"Aligned to {len(production_df)} common grid cells for {country_config.country_name}"
            )
        
        # Log final data coverage
        logger.info(
            f"Found {len(production_df)} production cells and "
            f"{len(harvest_area_df)} harvest area cells for {country_config.country_name}"
        )
        
        # Cache results
        self._country_data_cache[country_code] = (production_df, harvest_area_df)
        
        return production_df, harvest_area_df
    
    def validate_country_data_coverage(
        self, 
        country_code: str,
        min_cells: int = 1000
    ) -> Dict[str, any]:
        """
        Validate that country has sufficient data coverage for analysis.
        
        Args:
            country_code: Country code to validate
            min_cells: Minimum number of cells required
        
        Returns:
            Dictionary with validation results
        """
        country_code = country_code.upper()
        
        try:
            production_df, harvest_area_df = self.get_country_data(country_code)
        except ValueError as e:
            return {
                'valid': False,
                'error': str(e),
                'country_code': country_code
            }
        
        country_config = self.get_country_configuration(country_code)
        
        validation_results = {
            'valid': True,
            'country_code': country_code,
            'country_name': country_config.country_name if country_config else 'Unknown',
            'fips_code': country_config.fips_code if country_config else 'Unknown',
            'production_cells': len(production_df),
            'harvest_area_cells': len(harvest_area_df),
            'min_cells_required': min_cells,
            'meets_minimum': len(production_df) >= min_cells,
            'warnings': [],
            'crop_coverage': {}
        }
        
        # Check minimum cell requirement
        if not validation_results['meets_minimum']:
            validation_results['valid'] = False
            validation_results['warnings'].append(
                f"Insufficient data coverage: {len(production_df)} cells < {min_cells} required"
            )
        
        # Check crop coverage for priority crops
        if country_config and country_config.priority_crops:
            for crop in country_config.priority_crops:
                crop_cols = [col for col in production_df.columns if crop.lower() in col.lower()]
                if crop_cols:
                    # Check if crop has non-zero values
                    crop_data = production_df[crop_cols].sum(axis=1)
                    cells_with_crop = (crop_data > 0).sum()
                    validation_results['crop_coverage'][crop] = {
                        'cells_with_data': cells_with_crop,
                        'coverage_percent': (cells_with_crop / len(production_df)) * 100 if len(production_df) > 0 else 0
                    }
                    
                    if cells_with_crop == 0:
                        validation_results['warnings'].append(
                            f"No data found for priority crop: {crop}"
                        )
                else:
                    validation_results['warnings'].append(
                        f"Priority crop {crop} not found in SPAM data columns"
                    )
        
        # Geographic distribution check
        if len(production_df) > 0:
            lat_range = production_df['y'].max() - production_df['y'].min()
            lon_range = production_df['x'].max() - production_df['x'].min()
            
            validation_results['geographic_extent'] = {
                'latitude_range': lat_range,
                'longitude_range': lon_range,
                'lat_min': production_df['y'].min(),
                'lat_max': production_df['y'].max(),
                'lon_min': production_df['x'].min(),
                'lon_max': production_df['x'].max()
            }
            
            # Check if geographic extent is reasonable
            if lat_range < 1.0 or lon_range < 1.0:
                validation_results['warnings'].append(
                    f"Small geographic extent: {lat_range:.2f}° lat × {lon_range:.2f}° lon"
                )
        
        return validation_results
    
    def get_country_statistics(self, country_code: str) -> Dict[str, any]:
        """
        Get comprehensive statistics for a country's agricultural data.
        
        Args:
            country_code: Country code
        
        Returns:
            Dictionary with country statistics
        """
        country_code = country_code.upper()
        
        # Check cache first
        if country_code in self._country_statistics_cache:
            return self._country_statistics_cache[country_code]
        
        try:
            production_df, harvest_area_df = self.get_country_data(country_code)
        except ValueError as e:
            return {'error': str(e), 'country_code': country_code}
        
        country_config = self.get_country_configuration(country_code)
        
        statistics = {
            'country_code': country_code,
            'country_name': country_config.country_name if country_config else 'Unknown',
            'fips_code': country_config.fips_code if country_config else 'Unknown',
            'total_cells': len(production_df),
            'geographic_extent': {},
            'crop_statistics': {},
            'data_quality': {}
        }
        
        if len(production_df) > 0:
            # Geographic extent
            statistics['geographic_extent'] = {
                'lat_min': float(production_df['y'].min()),
                'lat_max': float(production_df['y'].max()),
                'lon_min': float(production_df['x'].min()),
                'lon_max': float(production_df['x'].max()),
                'lat_range': float(production_df['y'].max() - production_df['y'].min()),
                'lon_range': float(production_df['x'].max() - production_df['x'].min())
            }
            
            # Crop statistics
            crop_columns = [col for col in production_df.columns 
                          if any(crop in col.lower() for crop in ['wheat', 'maize', 'rice'])]
            
            for col in crop_columns:
                crop_data = production_df[col]
                non_zero_data = crop_data[crop_data > 0]
                
                if len(non_zero_data) > 0:
                    statistics['crop_statistics'][col] = {
                        'cells_with_data': len(non_zero_data),
                        'coverage_percent': (len(non_zero_data) / len(production_df)) * 100,
                        'total_production': float(non_zero_data.sum()),
                        'mean_production': float(non_zero_data.mean()),
                        'median_production': float(non_zero_data.median()),
                        'max_production': float(non_zero_data.max()),
                        'min_production': float(non_zero_data.min())
                    }
            
            # Data quality metrics
            statistics['data_quality'] = {
                'cells_with_coordinates': len(production_df.dropna(subset=['x', 'y'])),
                'coordinate_completeness': (len(production_df.dropna(subset=['x', 'y'])) / len(production_df)) * 100,
                'cells_with_production_data': len(production_df[crop_columns].dropna(how='all')),
                'production_data_completeness': (len(production_df[crop_columns].dropna(how='all')) / len(production_df)) * 100 if crop_columns else 0
            }
        
        # Cache results
        self._country_statistics_cache[country_code] = statistics
        
        return statistics
    
    def compare_countries(
        self, 
        country_codes: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Compare agricultural statistics between countries.
        
        Args:
            country_codes: List of country codes to compare
            metrics: Specific metrics to compare (optional)
        
        Returns:
            Dictionary with comparison results
        """
        if metrics is None:
            metrics = ['total_cells', 'geographic_extent', 'crop_coverage']
        
        comparison_results = {
            'countries': country_codes,
            'metrics': metrics,
            'country_data': {},
            'rankings': {},
            'summary': {}
        }
        
        # Get statistics for each country
        for country_code in country_codes:
            try:
                stats = self.get_country_statistics(country_code)
                comparison_results['country_data'][country_code] = stats
            except Exception as e:
                logger.warning(f"Failed to get statistics for {country_code}: {e}")
                comparison_results['country_data'][country_code] = {'error': str(e)}
        
        # Calculate rankings
        valid_countries = [
            code for code, data in comparison_results['country_data'].items()
            if 'error' not in data
        ]
        
        if len(valid_countries) > 1:
            # Rank by total cells
            cell_counts = {
                code: comparison_results['country_data'][code]['total_cells']
                for code in valid_countries
            }
            comparison_results['rankings']['by_total_cells'] = sorted(
                cell_counts.items(), key=lambda x: x[1], reverse=True
            )
            
            # Rank by geographic extent
            geographic_extents = {
                code: (
                    comparison_results['country_data'][code]['geographic_extent']['lat_range'] *
                    comparison_results['country_data'][code]['geographic_extent']['lon_range']
                )
                for code in valid_countries
                if comparison_results['country_data'][code]['geographic_extent']
            }
            comparison_results['rankings']['by_geographic_extent'] = sorted(
                geographic_extents.items(), key=lambda x: x[1], reverse=True
            )
        
        # Summary statistics
        if valid_countries:
            total_cells_all = sum(
                comparison_results['country_data'][code]['total_cells']
                for code in valid_countries
            )
            comparison_results['summary'] = {
                'total_countries': len(valid_countries),
                'total_cells_all_countries': total_cells_all,
                'average_cells_per_country': total_cells_all / len(valid_countries) if valid_countries else 0
            }
        
        return comparison_results
    
    def generate_country_report(self, country_code: str) -> str:
        """
        Generate a comprehensive report for a country's agricultural data.
        
        Args:
            country_code: Country code
        
        Returns:
            Formatted report string
        """
        try:
            validation = self.validate_country_data_coverage(country_code)
            statistics = self.get_country_statistics(country_code)
        except Exception as e:
            return f"Error generating report for {country_code}: {e}"
        
        country_config = self.get_country_configuration(country_code)
        
        report_lines = [
            "=" * 80,
            f"COUNTRY AGRICULTURAL DATA REPORT: {country_code}",
            "=" * 80,
            "",
            "BASIC INFORMATION:",
            f"  Country Name: {statistics.get('country_name', 'Unknown')}",
            f"  FIPS Code: {statistics.get('fips_code', 'Unknown')}",
            f"  Agricultural Focus: {country_config.agricultural_focus if country_config else 'Unknown'}",
            "",
            "DATA COVERAGE:",
            f"  Total Grid Cells: {statistics.get('total_cells', 0):,}",
            f"  Meets Minimum Requirement (1000 cells): {'✓' if validation.get('meets_minimum', False) else '✗'}",
            ""
        ]
        
        # Geographic extent
        if 'geographic_extent' in statistics and statistics['geographic_extent']:
            geo = statistics['geographic_extent']
            report_lines.extend([
                "GEOGRAPHIC EXTENT:",
                f"  Latitude Range: {geo['lat_min']:.2f}° to {geo['lat_max']:.2f}° ({geo['lat_range']:.2f}°)",
                f"  Longitude Range: {geo['lon_min']:.2f}° to {geo['lon_max']:.2f}° ({geo['lon_range']:.2f}°)",
                ""
            ])
        
        # Crop coverage
        if 'crop_coverage' in validation and validation['crop_coverage']:
            report_lines.extend([
                "CROP COVERAGE:",
            ])
            for crop, coverage in validation['crop_coverage'].items():
                report_lines.append(
                    f"  {crop.title()}: {coverage['cells_with_data']:,} cells "
                    f"({coverage['coverage_percent']:.1f}% coverage)"
                )
            report_lines.append("")
        
        # Data quality
        if 'data_quality' in statistics and statistics['data_quality']:
            quality = statistics['data_quality']
            report_lines.extend([
                "DATA QUALITY:",
                f"  Coordinate Completeness: {quality.get('coordinate_completeness', 0):.1f}%",
                f"  Production Data Completeness: {quality.get('production_data_completeness', 0):.1f}%",
                ""
            ])
        
        # Warnings
        if validation.get('warnings'):
            report_lines.extend([
                f"WARNINGS ({len(validation['warnings'])}):",
            ])
            for warning in validation['warnings']:
                report_lines.append(f"  - {warning}")
            report_lines.append("")
        
        # Configuration details
        if country_config:
            report_lines.extend([
                "CONFIGURATION:",
                f"  Priority Crops: {', '.join(country_config.priority_crops)}",
            ])
            if country_config.regional_subdivisions:
                report_lines.append(f"  Regional Subdivisions: {', '.join(country_config.regional_subdivisions)}")
            if country_config.policy_scenarios:
                report_lines.append(f"  Policy Scenarios: {', '.join(country_config.policy_scenarios)}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self._country_data_cache.clear()
        self._country_statistics_cache.clear()
        logger.info("CountryBoundaryManager cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'country_data_cache_size': len(self._country_data_cache),
            'country_statistics_cache_size': len(self._country_statistics_cache)
        }
    
    def add_country_from_framework(self, country_configuration: CountryConfiguration) -> bool:
        """
        Add a country configuration from the CountryFramework.
        
        Args:
            country_configuration: CountryConfiguration object from framework
        
        Returns:
            True if successfully added, False if country already exists
        """
        country_code = country_configuration.country_code.upper()
        
        if country_code in self.COUNTRY_CONFIGURATIONS:
            logger.warning(f"Country {country_code} already exists in configurations")
            return False
        
        # Add to configurations
        self.COUNTRY_CONFIGURATIONS[country_code] = country_configuration
        
        logger.info(f"Added country {country_configuration.country_name} ({country_code}) to system")
        return True
    
    def get_framework_ready_countries(self) -> List[str]:
        """
        Get list of countries that can be added via CountryFramework.
        
        Returns:
            List of country codes available in framework but not yet added
        """
        try:
            # Import here to avoid circular imports
            from .country_framework import CountryFramework
            
            # Create temporary framework instance to check available templates
            framework = CountryFramework(self.config, self)
            available_templates = framework.get_available_templates()
            current_countries = set(self.COUNTRY_CONFIGURATIONS.keys())
            
            return [code for code in available_templates if code not in current_countries]
        
        except ImportError:
            logger.warning("CountryFramework not available")
            return []
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CountryBoundaryManager("
            f"countries={len(self.COUNTRY_CONFIGURATIONS)}, "
            f"cached_data={len(self._country_data_cache)}, "
            f"cached_stats={len(self._country_statistics_cache)})"
        )
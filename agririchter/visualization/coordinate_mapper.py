"""SPAM Coordinate Mapping and Validation System."""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.config import Config
from ..core.constants import GRID_PARAMS


@dataclass
class CoordinateValidationReport:
    """Report containing coordinate validation results."""
    
    validation_results: Dict[str, bool]
    coordinate_statistics: Dict[str, float]
    recommendations: List[str]
    
    def is_valid(self) -> bool:
        """Check if all validation criteria pass."""
        return all(self.validation_results.values())
    
    @property
    def issues(self) -> List[str]:
        """Get list of validation issues."""
        return [key for key, value in self.validation_results.items() if not value]


@dataclass
class CoordinateDebugReport:
    """Report containing coordinate debugging information."""
    
    debug_info: Dict[str, Any]
    
    def get_coordinate_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get coordinate ranges from debug info."""
        return self.debug_info.get('coordinate_ranges', {})
    
    def get_potential_issues(self) -> List[str]:
        """Get list of potential coordinate issues."""
        return self.debug_info.get('potential_issues', [])


class SPAMCoordinateMapper:
    """Handles accurate coordinate mapping from SPAM grid to geographic coordinates."""
    
    def __init__(self, cell5m_path: Optional[str] = None):
        """
        Initialize SPAM coordinate mapper.
        
        Args:
            cell5m_path: Optional path to CELL5M.asc reference file
        """
        self.cell5m_path = cell5m_path
        self.coordinate_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # SPAM 2020 grid parameters (5-arcminute resolution)
        self.grid_params = {
            'ncols': 4320,
            'nrows': 2160,
            'xllcorner': -180.0,
            'yllcorner': -90.0,
            'cellsize': 0.0833333333333333,  # 5 arcminutes
            'nodata_value': -9999
        }
        
        self._initialize_coordinate_system()
    
    def _initialize_coordinate_system(self):
        """Initialize coordinate reference system."""
        self.logger.info("Initializing SPAM coordinate reference system")
        
        # Calculate grid bounds
        self.x_min = self.grid_params['xllcorner']
        self.x_max = self.x_min + (self.grid_params['ncols'] * self.grid_params['cellsize'])
        self.y_min = self.grid_params['yllcorner']
        self.y_max = self.y_min + (self.grid_params['nrows'] * self.grid_params['cellsize'])
        
        self.logger.debug(f"Grid bounds: X[{self.x_min}, {self.x_max}], Y[{self.y_min}, {self.y_max}]")
    
    def validate_coordinates(self, lons: np.ndarray, lats: np.ndarray) -> CoordinateValidationReport:
        """
        Validate coordinate ranges and geographic accuracy.
        
        Args:
            lons: Array of longitude values
            lats: Array of latitude values
        
        Returns:
            Coordinate validation report
        """
        validation_results = {
            'longitude_range_valid': self._validate_longitude_range(lons),
            'latitude_range_valid': self._validate_latitude_range(lats),
            'coordinate_density_appropriate': self._check_coordinate_density(lons, lats),
            'geographic_distribution_realistic': self._check_geographic_distribution(lons, lats),
            'grid_alignment_correct': self._check_grid_alignment(lons, lats)
        }
        
        coordinate_statistics = self._calculate_coordinate_statistics(lons, lats)
        recommendations = self._generate_coordinate_recommendations(validation_results)
        
        return CoordinateValidationReport(
            validation_results=validation_results,
            coordinate_statistics=coordinate_statistics,
            recommendations=recommendations
        )
    
    def _validate_longitude_range(self, lons: np.ndarray) -> bool:
        """Validate longitude values are within valid range."""
        return (-180 <= lons.min()) and (lons.max() <= 180)
    
    def _validate_latitude_range(self, lats: np.ndarray) -> bool:
        """Validate latitude values are within valid range."""
        return (-90 <= lats.min()) and (lats.max() <= 90)
    
    def _check_coordinate_density(self, lons: np.ndarray, lats: np.ndarray) -> bool:
        """Check if coordinate density is appropriate for SPAM grid."""
        # Calculate approximate grid spacing
        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        
        if len(unique_lons) < 2 or len(unique_lats) < 2:
            return False
        
        # Check if spacing is close to SPAM 5-arcminute resolution
        lon_spacing = np.median(np.diff(np.sort(unique_lons)))
        lat_spacing = np.median(np.diff(np.sort(unique_lats)))
        
        expected_spacing = self.grid_params['cellsize']
        tolerance = expected_spacing * 0.1  # 10% tolerance
        
        lon_spacing_ok = abs(lon_spacing - expected_spacing) <= tolerance
        lat_spacing_ok = abs(lat_spacing - expected_spacing) <= tolerance
        
        return lon_spacing_ok and lat_spacing_ok
    
    def _check_geographic_distribution(self, lons: np.ndarray, lats: np.ndarray) -> bool:
        """Check if geographic distribution is realistic for agricultural data."""
        # Check for reasonable global coverage
        lon_range = lons.max() - lons.min()
        lat_range = lats.max() - lats.min()
        
        # Agricultural data should span significant geographic area
        min_lon_range = 10.0  # degrees
        min_lat_range = 5.0   # degrees
        
        return lon_range >= min_lon_range and lat_range >= min_lat_range
    
    def _check_grid_alignment(self, lons: np.ndarray, lats: np.ndarray) -> bool:
        """Check if coordinates align with SPAM grid structure."""
        # Check if coordinates align with expected grid cell centers
        cellsize = self.grid_params['cellsize']
        
        # Calculate expected grid positions
        expected_lon_offset = self.x_min + cellsize / 2
        expected_lat_offset = self.y_min + cellsize / 2
        
        # Check alignment for a sample of coordinates
        sample_size = min(1000, len(lons))
        sample_indices = np.random.choice(len(lons), sample_size, replace=False)
        
        sample_lons = lons[sample_indices]
        sample_lats = lats[sample_indices]
        
        # Check if coordinates are close to grid cell centers
        lon_alignment = np.abs((sample_lons - expected_lon_offset) % cellsize - cellsize/2) < cellsize/10
        lat_alignment = np.abs((sample_lats - expected_lat_offset) % cellsize - cellsize/2) < cellsize/10
        
        alignment_ratio = np.mean(lon_alignment & lat_alignment)
        
        return alignment_ratio > 0.8  # 80% of coordinates should be well-aligned
    
    def _calculate_coordinate_statistics(self, lons: np.ndarray, lats: np.ndarray) -> Dict[str, float]:
        """Calculate coordinate statistics."""
        return {
            'longitude_min': float(lons.min()),
            'longitude_max': float(lons.max()),
            'longitude_mean': float(lons.mean()),
            'longitude_std': float(lons.std()),
            'latitude_min': float(lats.min()),
            'latitude_max': float(lats.max()),
            'latitude_mean': float(lats.mean()),
            'latitude_std': float(lats.std()),
            'total_points': len(lons),
            'unique_longitudes': len(np.unique(lons)),
            'unique_latitudes': len(np.unique(lats))
        }
    
    def _generate_coordinate_recommendations(self, validation_results: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not validation_results['longitude_range_valid']:
            recommendations.append("Check longitude values - some are outside valid range [-180, 180]")
        
        if not validation_results['latitude_range_valid']:
            recommendations.append("Check latitude values - some are outside valid range [-90, 90]")
        
        if not validation_results['coordinate_density_appropriate']:
            recommendations.append("Coordinate density doesn't match expected SPAM 5-arcminute grid")
        
        if not validation_results['geographic_distribution_realistic']:
            recommendations.append("Geographic distribution seems too limited for agricultural data")
        
        if not validation_results['grid_alignment_correct']:
            recommendations.append("Coordinates don't align well with SPAM grid structure")
        
        if not recommendations:
            recommendations.append("All coordinate validation checks passed")
        
        return recommendations
    
    def debug_coordinate_mapping(self, production_df: pd.DataFrame) -> CoordinateDebugReport:
        """
        Generate detailed coordinate mapping debug information.
        
        Args:
            production_df: DataFrame with coordinate and production data
        
        Returns:
            Coordinate debug report
        """
        debug_info = {
            'coordinate_ranges': {
                'longitude': (float(production_df['x'].min()), float(production_df['x'].max())),
                'latitude': (float(production_df['y'].min()), float(production_df['y'].max()))
            },
            'top_production_locations': self._get_top_production_coordinates(production_df),
            'coordinate_distribution': self._analyze_coordinate_distribution(production_df),
            'potential_issues': self._identify_coordinate_issues(production_df)
        }
        
        return CoordinateDebugReport(debug_info)
    
    def _get_top_production_coordinates(self, production_df: pd.DataFrame) -> List[Dict[str, float]]:
        """Get coordinates of top production locations."""
        if 'production' not in production_df.columns:
            # Try to find a production column
            prod_cols = [col for col in production_df.columns if 'production' in col.lower() or col.endswith('_A')]
            if prod_cols:
                production_col = prod_cols[0]
            else:
                return []
        else:
            production_col = 'production'
        
        # Get top 10 production locations
        top_locations = production_df.nlargest(10, production_col)
        
        return [
            {
                'longitude': float(row['x']),
                'latitude': float(row['y']),
                'production': float(row[production_col])
            }
            for _, row in top_locations.iterrows()
        ]
    
    def _analyze_coordinate_distribution(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze coordinate distribution patterns."""
        lons = production_df['x'].values
        lats = production_df['y'].values
        
        return {
            'longitude_distribution': {
                'unique_values': len(np.unique(lons)),
                'spacing_median': float(np.median(np.diff(np.sort(np.unique(lons))))),
                'spacing_std': float(np.std(np.diff(np.sort(np.unique(lons)))))
            },
            'latitude_distribution': {
                'unique_values': len(np.unique(lats)),
                'spacing_median': float(np.median(np.diff(np.sort(np.unique(lats))))),
                'spacing_std': float(np.std(np.diff(np.sort(np.unique(lats)))))
            },
            'grid_regularity': self._assess_grid_regularity(lons, lats)
        }
    
    def _assess_grid_regularity(self, lons: np.ndarray, lats: np.ndarray) -> Dict[str, Any]:
        """Assess regularity of coordinate grid."""
        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        
        # Check spacing regularity
        lon_spacings = np.diff(np.sort(unique_lons))
        lat_spacings = np.diff(np.sort(unique_lats))
        
        lon_regularity = np.std(lon_spacings) / np.mean(lon_spacings) if len(lon_spacings) > 0 else 1.0
        lat_regularity = np.std(lat_spacings) / np.mean(lat_spacings) if len(lat_spacings) > 0 else 1.0
        
        return {
            'longitude_regularity_cv': float(lon_regularity),
            'latitude_regularity_cv': float(lat_regularity),
            'is_regular_grid': lon_regularity < 0.1 and lat_regularity < 0.1,
            'expected_cellsize': self.grid_params['cellsize'],
            'actual_median_spacing': {
                'longitude': float(np.median(lon_spacings)) if len(lon_spacings) > 0 else 0.0,
                'latitude': float(np.median(lat_spacings)) if len(lat_spacings) > 0 else 0.0
            }
        }
    
    def _identify_coordinate_issues(self, production_df: pd.DataFrame) -> List[str]:
        """Identify potential coordinate issues."""
        issues = []
        
        lons = production_df['x'].values
        lats = production_df['y'].values
        
        # Check for obvious issues
        if len(np.unique(lons)) < 10:
            issues.append("Very few unique longitude values - possible data truncation")
        
        if len(np.unique(lats)) < 10:
            issues.append("Very few unique latitude values - possible data truncation")
        
        # Check for coordinate precision issues
        lon_decimals = self._estimate_decimal_places(lons)
        lat_decimals = self._estimate_decimal_places(lats)
        
        if lon_decimals < 3:
            issues.append("Low longitude precision - may cause coordinate mapping errors")
        
        if lat_decimals < 3:
            issues.append("Low latitude precision - may cause coordinate mapping errors")
        
        # Check for suspicious coordinate patterns
        if self._has_suspicious_patterns(lons, lats):
            issues.append("Suspicious coordinate patterns detected - check data processing")
        
        return issues
    
    def _estimate_decimal_places(self, values: np.ndarray) -> int:
        """Estimate number of decimal places in coordinate values."""
        # Convert to string and count decimal places
        sample_size = min(100, len(values))
        sample_values = np.random.choice(values, sample_size, replace=False)
        
        decimal_places = []
        for val in sample_values:
            str_val = f"{val:.10f}".rstrip('0')
            if '.' in str_val:
                decimal_places.append(len(str_val.split('.')[1]))
            else:
                decimal_places.append(0)
        
        return int(np.median(decimal_places))
    
    def _has_suspicious_patterns(self, lons: np.ndarray, lats: np.ndarray) -> bool:
        """Check for suspicious coordinate patterns."""
        # Check for too many repeated values
        lon_repetition = len(lons) / len(np.unique(lons))
        lat_repetition = len(lats) / len(np.unique(lats))
        
        # If coordinates repeat too much, it might indicate an issue
        if lon_repetition > 100 or lat_repetition > 100:
            return True
        
        # Check for unrealistic clustering
        lon_range = lons.max() - lons.min()
        lat_range = lats.max() - lats.min()
        
        # If all coordinates are in a very small area, it might be suspicious
        if lon_range < 1.0 and lat_range < 1.0 and len(lons) > 1000:
            return True
        
        return False
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get SPAM grid information."""
        return {
            'grid_parameters': self.grid_params.copy(),
            'coordinate_bounds': {
                'longitude': (self.x_min, self.x_max),
                'latitude': (self.y_min, self.y_max)
            },
            'resolution': {
                'degrees': self.grid_params['cellsize'],
                'arcminutes': self.grid_params['cellsize'] * 60,
                'approximate_km_at_equator': self.grid_params['cellsize'] * 111.32
            }
        }
"""Coordinate System Debugging Tools for SPAM Data Analysis."""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .coordinate_mapper import SPAMCoordinateMapper
from ..data.grid_manager import GridDataManager


@dataclass
class LatitudeBandingDiagnosis:
    """Diagnosis results for latitude banding issues."""
    
    latitude_distribution: Dict[str, Any]
    longitude_distribution: Dict[str, Any]
    grid_analysis: Dict[str, Any]
    potential_causes: List[str]
    recommendations: List[str]
    
    def has_banding_issues(self) -> bool:
        """Check if latitude banding issues are detected."""
        return len(self.potential_causes) > 0
    
    def get_primary_cause(self) -> str:
        """Get the most likely cause of banding."""
        if not self.potential_causes:
            return "No banding issues detected"
        return self.potential_causes[0]


@dataclass
class DiagnosticPlotsResult:
    """Result containing diagnostic plot information."""
    
    plot_paths: List[str]
    plot_descriptions: Dict[str, str]
    analysis_summary: Dict[str, Any]
    
    def get_plot_count(self) -> int:
        """Get number of diagnostic plots created."""
        return len(self.plot_paths)


class CoordinateSystemDebugger:
    """Tools for debugging coordinate mapping issues."""
    
    def __init__(self, spam_data_manager: Optional[GridDataManager] = None):
        """
        Initialize coordinate system debugger.
        
        Args:
            spam_data_manager: Optional SPAM data manager instance
        """
        self.spam_data_manager = spam_data_manager
        self.coordinate_mapper = SPAMCoordinateMapper()
        self.logger = logging.getLogger(__name__)
    
    def diagnose_latitude_banding(self, production_df: pd.DataFrame) -> LatitudeBandingDiagnosis:
        """
        Diagnose causes of latitude banding in global maps.
        
        Args:
            production_df: DataFrame with coordinate and production data
        
        Returns:
            Latitude banding diagnosis
        """
        self.logger.info("Diagnosing latitude banding issues")
        
        # Analyze coordinate distribution patterns
        lat_distribution = self._analyze_latitude_distribution(production_df)
        lon_distribution = self._analyze_longitude_distribution(production_df)
        
        # Check for regular grid patterns
        grid_analysis = self._analyze_grid_regularity(production_df)
        
        # Identify potential causes
        potential_causes = []
        recommendations = []
        
        if grid_analysis['highly_regular']:
            potential_causes.append("Regular SPAM grid structure (5-arcminute resolution)")
            recommendations.append("Use point size variation or alpha blending to reduce banding appearance")
        
        if lat_distribution['has_discrete_bands']:
            potential_causes.append("Discrete latitude values from grid structure")
            recommendations.append("Consider using interpolation or smoothing for visualization")
        
        if not self._check_projection_handling(production_df):
            potential_causes.append("Incorrect cartopy projection or coordinate transformation")
            recommendations.append("Verify cartopy projection settings and coordinate reference system")
        
        if self._has_coordinate_precision_issues(production_df):
            potential_causes.append("Low coordinate precision causing artificial clustering")
            recommendations.append("Check coordinate data precision and rounding")
        
        if not potential_causes:
            potential_causes.append("No obvious banding issues detected")
            recommendations.append("Coordinate system appears to be functioning correctly")
        
        return LatitudeBandingDiagnosis(
            latitude_distribution=lat_distribution,
            longitude_distribution=lon_distribution,
            grid_analysis=grid_analysis,
            potential_causes=potential_causes,
            recommendations=recommendations
        )
    
    def _analyze_latitude_distribution(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze latitude distribution patterns."""
        lats = production_df['y'].values
        unique_lats = np.unique(lats)
        
        # Calculate spacing between latitude values
        lat_spacings = np.diff(np.sort(unique_lats))
        
        # Check for discrete banding
        spacing_std = np.std(lat_spacings) if len(lat_spacings) > 0 else 0
        spacing_mean = np.mean(lat_spacings) if len(lat_spacings) > 0 else 0
        
        # Detect if spacing is very regular (indicating grid structure)
        is_regular = spacing_std / spacing_mean < 0.1 if spacing_mean > 0 else False
        
        # Check for discrete bands (many repeated latitude values)
        lat_counts = pd.Series(lats).value_counts()
        has_discrete_bands = (lat_counts > 10).sum() > len(unique_lats) * 0.1
        
        return {
            'unique_latitudes': len(unique_lats),
            'latitude_range': (float(lats.min()), float(lats.max())),
            'spacing_statistics': {
                'mean': float(spacing_mean),
                'std': float(spacing_std),
                'median': float(np.median(lat_spacings)) if len(lat_spacings) > 0 else 0.0
            },
            'is_regular_spacing': is_regular,
            'has_discrete_bands': has_discrete_bands,
            'band_density': float(len(lats) / len(unique_lats)),
            'expected_spam_spacing': 0.0833333  # 5 arcminutes
        }
    
    def _analyze_longitude_distribution(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze longitude distribution patterns."""
        lons = production_df['x'].values
        unique_lons = np.unique(lons)
        
        # Calculate spacing between longitude values
        lon_spacings = np.diff(np.sort(unique_lons))
        
        spacing_std = np.std(lon_spacings) if len(lon_spacings) > 0 else 0
        spacing_mean = np.mean(lon_spacings) if len(lon_spacings) > 0 else 0
        
        is_regular = spacing_std / spacing_mean < 0.1 if spacing_mean > 0 else False
        
        return {
            'unique_longitudes': len(unique_lons),
            'longitude_range': (float(lons.min()), float(lons.max())),
            'spacing_statistics': {
                'mean': float(spacing_mean),
                'std': float(spacing_std),
                'median': float(np.median(lon_spacings)) if len(lon_spacings) > 0 else 0.0
            },
            'is_regular_spacing': is_regular,
            'expected_spam_spacing': 0.0833333  # 5 arcminutes
        }
    
    def _analyze_grid_regularity(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze grid regularity patterns."""
        lons = production_df['x'].values
        lats = production_df['y'].values
        
        # Check coordinate spacing regularity
        unique_lons = np.unique(lons)
        unique_lats = np.unique(lats)
        
        lon_spacings = np.diff(np.sort(unique_lons))
        lat_spacings = np.diff(np.sort(unique_lats))
        
        # Calculate regularity metrics
        lon_regularity = np.std(lon_spacings) / np.mean(lon_spacings) if len(lon_spacings) > 0 and np.mean(lon_spacings) > 0 else 1.0
        lat_regularity = np.std(lat_spacings) / np.mean(lat_spacings) if len(lat_spacings) > 0 and np.mean(lat_spacings) > 0 else 1.0
        
        # Grid is considered highly regular if coefficient of variation is low
        highly_regular = lon_regularity < 0.1 and lat_regularity < 0.1
        
        # Check if spacing matches SPAM grid
        expected_spacing = 0.0833333  # 5 arcminutes
        lon_spacing_match = abs(np.median(lon_spacings) - expected_spacing) < 0.01 if len(lon_spacings) > 0 else False
        lat_spacing_match = abs(np.median(lat_spacings) - expected_spacing) < 0.01 if len(lat_spacings) > 0 else False
        
        return {
            'highly_regular': highly_regular,
            'longitude_regularity_cv': float(lon_regularity),
            'latitude_regularity_cv': float(lat_regularity),
            'matches_spam_grid': lon_spacing_match and lat_spacing_match,
            'grid_characteristics': {
                'longitude_spacing': float(np.median(lon_spacings)) if len(lon_spacings) > 0 else 0.0,
                'latitude_spacing': float(np.median(lat_spacings)) if len(lat_spacings) > 0 else 0.0,
                'expected_spacing': expected_spacing
            }
        }
    
    def _check_projection_handling(self, production_df: pd.DataFrame) -> bool:
        """Check if projection handling is correct."""
        # Basic checks for coordinate system issues
        lons = production_df['x'].values
        lats = production_df['y'].values
        
        # Check if coordinates are in reasonable ranges
        lon_range_ok = (-180 <= lons.min()) and (lons.max() <= 180)
        lat_range_ok = (-90 <= lats.min()) and (lats.max() <= 90)
        
        # Check if coordinate distribution makes sense for agricultural data
        lon_span = lons.max() - lons.min()
        lat_span = lats.max() - lats.min()
        
        # Agricultural data should span reasonable geographic area
        reasonable_span = lon_span > 5 and lat_span > 2  # At least 5° longitude, 2° latitude
        
        return lon_range_ok and lat_range_ok and reasonable_span
    
    def _has_coordinate_precision_issues(self, production_df: pd.DataFrame) -> bool:
        """Check for coordinate precision issues."""
        lons = production_df['x'].values
        lats = production_df['y'].values
        
        # Estimate decimal precision
        lon_precision = self._estimate_decimal_precision(lons)
        lat_precision = self._estimate_decimal_precision(lats)
        
        # SPAM coordinates should have at least 4-5 decimal places for 5-arcminute resolution
        return lon_precision < 4 or lat_precision < 4
    
    def _estimate_decimal_precision(self, values: np.ndarray) -> int:
        """Estimate decimal precision of coordinate values."""
        # Sample some values and check decimal places
        sample_size = min(100, len(values))
        sample_values = np.random.choice(values, sample_size, replace=False)
        
        decimal_places = []
        for val in sample_values:
            # Convert to string and count decimal places
            str_val = f"{val:.10f}".rstrip('0')
            if '.' in str_val:
                decimal_places.append(len(str_val.split('.')[1]))
            else:
                decimal_places.append(0)
        
        return int(np.median(decimal_places))
    
    def create_coordinate_diagnostic_plots(self, production_df: pd.DataFrame, 
                                         output_dir: Optional[Path] = None) -> DiagnosticPlotsResult:
        """
        Create diagnostic plots for coordinate system analysis.
        
        Args:
            production_df: DataFrame with coordinate and production data
            output_dir: Optional directory to save plots
        
        Returns:
            Diagnostic plots result
        """
        self.logger.info("Creating coordinate diagnostic plots")
        
        if output_dir is None:
            output_dir = Path('results/coordinate_diagnostics')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with multiple diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Coordinate System Diagnostic Analysis', fontsize=16)
        
        plot_paths = []
        plot_descriptions = {}
        
        # Plot 1: Simple scatter (no projection)
        axes[0, 0].scatter(production_df['x'], production_df['y'], s=0.1, alpha=0.5, c='blue')
        axes[0, 0].set_xlabel('Longitude (degrees)')
        axes[0, 0].set_ylabel('Latitude (degrees)')
        axes[0, 0].set_title('Raw Coordinate Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Production-weighted scatter
        prod_cols = [col for col in production_df.columns if col.endswith('_A') or 'production' in col.lower()]
        if prod_cols:
            production_col = prod_cols[0]
            prod_values = production_df[production_col].values
            
            # Use log scale for better visualization
            log_prod = np.log10(prod_values + 1)
            
            scatter = axes[0, 1].scatter(
                production_df['x'], production_df['y'],
                c=log_prod, s=0.5, alpha=0.7, cmap='YlOrRd'
            )
            plt.colorbar(scatter, ax=axes[0, 1], label=f'Log10({production_col} + 1)')
            axes[0, 1].set_xlabel('Longitude (degrees)')
            axes[0, 1].set_ylabel('Latitude (degrees)')
            axes[0, 1].set_title('Production-Weighted Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No production data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Production Data Not Available')
        
        # Plot 3: Latitude distribution histogram
        axes[1, 0].hist(production_df['y'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Latitude (degrees)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Latitude Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Longitude distribution histogram
        axes[1, 1].hist(production_df['x'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Longitude (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Longitude Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save main diagnostic plot
        main_plot_path = output_dir / 'coordinate_diagnostics_main.png'
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(str(main_plot_path))
        plot_descriptions[str(main_plot_path)] = "Main coordinate diagnostic plots"
        
        # Create additional spacing analysis plot
        spacing_plot_path = self._create_spacing_analysis_plot(production_df, output_dir)
        if spacing_plot_path:
            plot_paths.append(spacing_plot_path)
            plot_descriptions[spacing_plot_path] = "Coordinate spacing analysis"
        
        # Generate analysis summary
        analysis_summary = self._generate_diagnostic_summary(production_df)
        
        return DiagnosticPlotsResult(
            plot_paths=plot_paths,
            plot_descriptions=plot_descriptions,
            analysis_summary=analysis_summary
        )
    
    def _create_spacing_analysis_plot(self, production_df: pd.DataFrame, 
                                    output_dir: Path) -> Optional[str]:
        """Create coordinate spacing analysis plot."""
        try:
            lons = production_df['x'].values
            lats = production_df['y'].values
            
            unique_lons = np.unique(lons)
            unique_lats = np.unique(lats)
            
            if len(unique_lons) < 2 or len(unique_lats) < 2:
                return None
            
            lon_spacings = np.diff(np.sort(unique_lons))
            lat_spacings = np.diff(np.sort(unique_lats))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Coordinate Spacing Analysis', fontsize=14)
            
            # Longitude spacing histogram
            axes[0].hist(lon_spacings, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0].axvline(0.0833333, color='red', linestyle='--', 
                           label='Expected SPAM spacing (5 arcmin)')
            axes[0].set_xlabel('Longitude Spacing (degrees)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Longitude Spacing Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Latitude spacing histogram
            axes[1].hist(lat_spacings, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1].axvline(0.0833333, color='red', linestyle='--', 
                           label='Expected SPAM spacing (5 arcmin)')
            axes[1].set_xlabel('Latitude Spacing (degrees)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Latitude Spacing Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            spacing_plot_path = output_dir / 'coordinate_spacing_analysis.png'
            plt.savefig(spacing_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(spacing_plot_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to create spacing analysis plot: {e}")
            return None
    
    def _generate_diagnostic_summary(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate diagnostic analysis summary."""
        lons = production_df['x'].values
        lats = production_df['y'].values
        
        # Basic statistics
        coordinate_stats = {
            'total_points': len(production_df),
            'unique_longitudes': len(np.unique(lons)),
            'unique_latitudes': len(np.unique(lats)),
            'longitude_range': (float(lons.min()), float(lons.max())),
            'latitude_range': (float(lats.min()), float(lats.max())),
            'coordinate_density': len(production_df) / (len(np.unique(lons)) * len(np.unique(lats)))
        }
        
        # Validation results
        validation_report = self.coordinate_mapper.validate_coordinates(lons, lats)
        
        # Banding diagnosis
        banding_diagnosis = self.diagnose_latitude_banding(production_df)
        
        return {
            'coordinate_statistics': coordinate_stats,
            'validation_results': validation_report.validation_results,
            'validation_recommendations': validation_report.recommendations,
            'banding_analysis': {
                'has_banding_issues': banding_diagnosis.has_banding_issues(),
                'primary_cause': banding_diagnosis.get_primary_cause(),
                'recommendations': banding_diagnosis.recommendations
            },
            'grid_analysis': banding_diagnosis.grid_analysis
        }
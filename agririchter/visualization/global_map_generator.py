"""Enhanced Global Map Generator with proper coordinate handling."""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, BoundaryNorm

# Lazy import cartopy to prevent segfaults on incompatible systems
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    ccrs = None
    cfeature = None
    CARTOPY_AVAILABLE = False

from dataclasses import dataclass

from ..core.config import Config
from .coordinate_mapper import SPAMCoordinateMapper


@dataclass
class PublicationMapsResult:
    """Result from publication map generation."""
    figure: plt.Figure
    panel_results: Dict[str, Dict]
    coordinate_validation: Dict[str, any]


class GlobalMapGenerator:
    """Enhanced global map generation with proper coordinate handling."""
    
    def __init__(self, config: Config):
        """Initialize global map generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if CARTOPY_AVAILABLE:
            self.projection = ccrs.PlateCarree()
        else:
            self.logger.warning("Cartopy not available. Map generation will fail.")
            self.projection = None
        
        # Agricultural extent (including southern hemisphere agriculture)
        self.global_extent = [-180, 180, -56, 75]
        
        # Initialize coordinate mapper
        self.coordinate_mapper = SPAMCoordinateMapper()
        
        # Set up publication-quality matplotlib settings for journal (Nature style)
        plt.rcParams.update({
            'font.size': 18,           # Larger base font
            'axes.titlesize': 22,      # Larger panel titles
            'axes.labelsize': 20,      # Larger axis labels
            'xtick.labelsize': 16,     # Larger tick labels
            'ytick.labelsize': 16,     # Larger tick labels
            'legend.fontsize': 16,     # Larger legend
            'figure.titlesize': 24,    # Larger figure title
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.weight': 'normal',   # Clear font weight
            'axes.linewidth': 1.5      # Thicker axes
        })
    
    def create_global_map_panel(self, ax, data_df: pd.DataFrame, title: str, 
                               metric_type: str, vmin=None, vmax=None):
        """Create a single map panel with corrected coordinate mapping.
        
        Args:
            ax: Matplotlib axis with cartopy projection
            data_df: DataFrame with data and coordinates
            title: Panel title
            metric_type: 'production' or 'harvest_area'
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            
        Returns:
            Scatter plot object or None
        """
        # Check for data availability
        if data_df is None or len(data_df) == 0:
            self.logger.warning(f"No data provided for {title}")
            self._setup_empty_panel(ax, title)
            return None
        
        # SPAM coordinates are already correct - no validation needed
        self.logger.debug(f"Plotting {len(data_df):,} data points for {title}")
        # Set up cartopy projection with explicit extent
        ax.set_extent(self.global_extent, crs=self.projection)
        
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Get crop columns and calculate values
        crop_cols = [col for col in data_df.columns if col.endswith('_A')]
        
        if not crop_cols:
            self.logger.warning(f"No crop columns found for {title}")
            self._setup_empty_panel(ax, title)
            return None
        
        # Calculate total values per grid cell
        total_values = data_df[crop_cols].sum(axis=1)
        
        # Filter to only cells with values > 0
        has_values = total_values > 0
        if not np.any(has_values):
            self.logger.warning(f"No non-zero values found for {title}")
            self._setup_empty_panel(ax, title)
            return None
        
        # Get data subset with values
        subset_df = data_df[has_values].copy()
        values = total_values[has_values].values
        lons = subset_df['x'].values
        lats = subset_df['y'].values
        
        # Apply coordinate filtering to agricultural extent
        coord_mask = (
            (lons >= self.global_extent[0]) & (lons <= self.global_extent[1]) &
            (lats >= self.global_extent[2]) & (lats <= self.global_extent[3])
        )
        
        if not np.any(coord_mask):
            self.logger.warning(f"No coordinates within agricultural extent for {title}")
            self._setup_empty_panel(ax, title)
            return None
        
        # Apply coordinate mask
        lons = lons[coord_mask]
        lats = lats[coord_mask]
        values = values[coord_mask]
        
        # Data quality filtering
        if metric_type == 'production':
            # Remove extreme outliers for production data
            q99 = np.percentile(values, 99.5)
            quality_mask = values <= q99 * 2
        else:
            # More conservative filtering for harvest area
            q99 = np.percentile(values, 99.8)
            quality_mask = values <= q99 * 1.5
        
        lons = lons[quality_mask]
        lats = lats[quality_mask]
        values = values[quality_mask]
        
        if len(values) == 0:
            self.logger.warning(f"No values remaining after quality filtering for {title}")
            self._setup_empty_panel(ax, title)
            return None
        
        # Set up color mapping
        if metric_type == 'production':
            # Green color scheme for production
            colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', 
                     '#41ab5d', '#238b45', '#006d2c', '#00441b']
            cmap = mcolors.ListedColormap(colors)
            label = 'Production (tonnes)'
        else:
            # Orange/Red color scheme for harvest area
            colors = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', 
                     '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
            cmap = mcolors.ListedColormap(colors)
            label = 'Harvest Area (ha)'
        
        # Set value range
        if vmin is None:
            vmin = np.percentile(values, 5)
        if vmax is None:
            vmax = np.percentile(values, 95)
        
        # Ensure positive values for log scale
        vmin = max(vmin, 1.0)
        vmax = max(vmax, vmin * 10)
        
        # Create discrete bins for better visualization
        bins = np.logspace(np.log10(vmin), np.log10(vmax), len(colors))
        norm = BoundaryNorm(bins, len(colors))
        
        # Calculate point size based on data density
        point_size = self._calculate_point_size(len(values))
        
        # Plot data with proper coordinate transformation
        scatter = ax.scatter(
            lons, lats, 
            c=values,
            s=point_size,
            alpha=0.7,
            cmap=cmap,
            norm=norm,
            transform=self.projection,
            edgecolors='none'
        )
        
        # Add colorbar with discrete levels - larger for journal
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02, 
                          boundaries=bins, ticks=bins[::2])
        cbar.set_label(label, fontsize=12, weight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Format colorbar labels
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
        
        self.logger.info(f"Plotted {len(values)} points for {title}")
        
        # Set title and grid - larger for journal
        ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
        ax.gridlines(draw_labels=False, alpha=0.3, linewidth=0.8)
        
        return scatter
    
    def _setup_empty_panel(self, ax, title: str):
        """Set up an empty panel with basic features."""
        ax.set_extent(self.global_extent, crs=self.projection)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
        ax.gridlines(draw_labels=False, alpha=0.3, linewidth=0.8)
        
        # Add "No Data" text - larger for journal
        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes,
               ha='center', va='center', fontsize=14, weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _calculate_point_size(self, num_points: int) -> float:
        """Calculate appropriate point size based on data density."""
        if num_points < 1000:
            return 2.0
        elif num_points < 10000:
            return 1.0
        elif num_points < 100000:
            return 0.5
        else:
            return 0.3
    
    def generate_publication_maps(self, crop_data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                 output_path: Optional[Path] = None) -> PublicationMapsResult:
        """Generate publication-ready 8-panel global maps.
        
        Args:
            crop_data_dict: Dictionary with crop data {crop: {'production': df, 'harvest': df}}
            output_path: Optional path to save the figure
            
        Returns:
            PublicationMapsResult with figure and validation info
        """
        self.logger.info("Generating publication-ready global maps")
        
        # Create figure with 4x2 subplot grid - optimized for journal
        fig = plt.figure(figsize=(18, 16))  # Wider, less tall
        fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # Updated order: All Grains first
        crops = ['allgrain', 'wheat', 'maize', 'rice']
        crop_names = {
            'wheat': 'Wheat', 
            'maize': 'Maize', 
            'rice': 'Rice', 
            'allgrain': 'All Grains'
        }
        
        panel_results = {}
        coordinate_validation = {}
        
        # Create 8 panels (4 crops × 2 metrics)
        panel_idx = 1
        
        for i, crop in enumerate(crops):
            if crop not in crop_data_dict:
                self.logger.warning(f"No data available for {crop}")
                continue
            
            prod_df = crop_data_dict[crop].get('production')
            harv_df = crop_data_dict[crop].get('harvest')
            
            # Production panel
            ax_prod = fig.add_subplot(4, 2, panel_idx, projection=self.projection)
            scatter_prod = self.create_global_map_panel(
                ax_prod, prod_df, 
                f"{crop_names[crop]} Production", 'production'
            )
            
            panel_results[f"{crop}_production"] = {
                'scatter': scatter_prod,
                'data_points': len(prod_df) if prod_df is not None else 0
            }
            
            # Validate coordinates for this crop
            if prod_df is not None and len(prod_df) > 0:
                coord_validation = self.coordinate_mapper.validate_coordinates(
                    prod_df['x'].values, prod_df['y'].values
                )
                coordinate_validation[f"{crop}_production"] = {
                    'valid': coord_validation.is_valid(),
                    'issues': coord_validation.issues,
                    'statistics': coord_validation.coordinate_statistics
                }
            
            panel_idx += 1
            
            # Harvest area panel
            ax_harv = fig.add_subplot(4, 2, panel_idx, projection=self.projection)
            scatter_harv = self.create_global_map_panel(
                ax_harv, harv_df,
                f"{crop_names[crop]} Harvest Area", 'harvest_area'
            )
            
            panel_results[f"{crop}_harvest"] = {
                'scatter': scatter_harv,
                'data_points': len(harv_df) if harv_df is not None else 0
            }
            
            # Validate coordinates for harvest data
            if harv_df is not None and len(harv_df) > 0:
                coord_validation = self.coordinate_mapper.validate_coordinates(
                    harv_df['x'].values, harv_df['y'].values
                )
                coordinate_validation[f"{crop}_harvest"] = {
                    'valid': coord_validation.is_valid(),
                    'issues': coord_validation.issues,
                    'statistics': coord_validation.coordinate_statistics
                }
            
            panel_idx += 1
        
        # Add overall title - larger for journal
        plt.suptitle('Figure 1: Global Agricultural Production and Harvest Area Maps', 
                    fontsize=20, fontweight='bold', y=0.97)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Publication maps saved to {output_path}")
        
        return PublicationMapsResult(
            figure=fig,
            panel_results=panel_results,
            coordinate_validation=coordinate_validation
        )
    
    def debug_coordinate_mapping(self, crop_data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                output_dir: Optional[Path] = None) -> Dict[str, any]:
        """Debug coordinate mapping for all crops.
        
        Args:
            crop_data_dict: Dictionary with crop data
            output_dir: Optional directory to save debug plots
            
        Returns:
            Dictionary with debug results for each crop
        """
        self.logger.info("Running coordinate mapping debug analysis")
        
        debug_results = {}
        
        for crop, data in crop_data_dict.items():
            self.logger.info(f"Debugging coordinates for {crop}")
            
            prod_df = data.get('production')
            if prod_df is not None and len(prod_df) > 0:
                # Run coordinate debugging
                debug_output_path = None
                if output_dir:
                    debug_output_path = output_dir / f"{crop}_coordinate_debug.png"
                
                diagnostic_result = self.debugger.create_coordinate_diagnostic_plots(
                    prod_df, debug_output_path
                )
                
                # Run banding diagnosis
                banding_diagnosis = self.debugger.diagnose_latitude_banding(prod_df)
                
                debug_results[crop] = {
                    'coordinate_statistics': diagnostic_result.coordinate_statistics,
                    'diagnostic_summary': diagnostic_result.diagnostic_summary,
                    'banding_diagnosis': {
                        'potential_causes': banding_diagnosis.potential_causes,
                        'recommendations': banding_diagnosis.recommendations,
                        'grid_analysis': banding_diagnosis.grid_analysis
                    }
                }
                
                self.logger.info(f"Debug analysis complete for {crop}")
            else:
                debug_results[crop] = {'error': 'No production data available'}
        
        return debug_results
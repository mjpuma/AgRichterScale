"""
Multi-panel map visualizations for publication.

Creates publication-quality multi-panel figures with:
- Consistent color scales across crops
- Discretized ColorBrewer palettes
- Professional layout for Nature Food / Nature journals
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..core.config import Config
from ..core.constants import GRID_PARAMS

logger = logging.getLogger(__name__)


class MultiPanelMapper:
    """Create multi-panel maps for publication."""
    
    def __init__(self, config: Config):
        """Initialize mapper with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_eight_panel_combined_map(
        self,
        production_data: Dict[str, pd.DataFrame],
        harvest_data: Dict[str, pd.DataFrame],
        output_path: Optional[Path] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """Create 8-panel combined map (4 rows × 2 columns).
        
        Layout (4 rows × 2 columns):
        Row 1: Wheat Production      | Wheat Harvest Area
        Row 2: Rice Production       | Rice Harvest Area
        Row 3: Maize Production      | Maize Harvest Area
        Row 4: AllGrain Production   | AllGrain Harvest Area
        
        Args:
            production_data: Dictionary with crop production DataFrames
            harvest_data: Dictionary with crop harvest area DataFrames
            output_path: Optional path to save figure
            dpi: Resolution for output
            
        Returns:
            matplotlib Figure object
        """
        self.logger.info("Creating 8-panel combined production and harvest area map (4×2 layout)")
        
        # Define crop order and labels
        crops = ['wheat', 'rice', 'maize', 'allgrain']
        labels = ['Wheat', 'Rice', 'Maize', 'All Grain']
        
        # Create figure with 4 rows, 2 columns - taller and less white space
        fig = plt.figure(figsize=(12, 18), dpi=dpi)
        
        # Create discrete colormaps - greens for production, oranges for harvest area (land)
        prod_cmap = self._create_discrete_cmap('YlGn', n_colors=9)
        harvest_cmap = self._create_discrete_cmap('YlOrBr', n_colors=9)
        
        # Calculate value ranges from actual data
        prod_vmin, prod_vmax = self._calculate_global_scale(production_data, crops[:3])
        harvest_vmin, harvest_vmax = self._calculate_global_scale(harvest_data, crops[:3])
        
        self.logger.info(f"Production scale: {prod_vmin:.2f} to {prod_vmax:.2f} (log10)")
        self.logger.info(f"Harvest scale: {harvest_vmin:.2f} to {harvest_vmax:.2f} (log10)")
        
        # Plot each crop as a row (production left, harvest right)
        for i, (crop, label) in enumerate(zip(crops, labels)):
            # Production (left column)
            ax_prod = fig.add_subplot(4, 2, 2*i + 1, projection=ccrs.PlateCarree())
            crop_prod_vmin, crop_prod_vmax = (prod_vmin, prod_vmax) if crop != 'allgrain' else self._calculate_crop_scale(production_data[crop], crop)
            self._plot_crop_panel(ax_prod, production_data[crop], f'{label} Production',
                                prod_cmap, crop_prod_vmin, crop_prod_vmax, crop, fontsize=16)
            
            # Harvest area (right column)
            ax_harvest = fig.add_subplot(4, 2, 2*i + 2, projection=ccrs.PlateCarree())
            crop_harvest_vmin, crop_harvest_vmax = (harvest_vmin, harvest_vmax) if crop != 'allgrain' else self._calculate_crop_scale(harvest_data[crop], crop)
            self._plot_crop_panel(ax_harvest, harvest_data[crop], f'{label} Harvest Area',
                                harvest_cmap, crop_harvest_vmin, crop_harvest_vmax, crop, fontsize=16)
        
        # Add separate colorbars for production and harvest
        self._add_dual_colorbars(fig, prod_cmap, prod_vmin, prod_vmax, 
                                harvest_cmap, harvest_vmin, harvest_vmax)
        
        # Tighter layout with less white space
        plt.tight_layout(rect=[0, 0.03, 1, 0.99], h_pad=1.0, w_pad=0.5)
        
        if output_path:
            self.logger.info(f"Saving 8-panel combined map to {output_path}")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            
            # Also save SVG version
            svg_path = output_path.with_suffix('.svg')
            self.logger.info(f"Saving SVG version to {svg_path}")
            fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
        
        self.logger.info("8-panel combined map created successfully")
        return fig
    
    def create_four_panel_production_map(
        self,
        crop_data: Dict[str, pd.DataFrame],
        output_path: Optional[Path] = None,
        dpi: int = 300,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        n_colors: int = 9
    ) -> plt.Figure:
        """Create 4-panel production map (wheat, rice, maize, allgrain).
        
        Args:
            crop_data: Dictionary with crop names as keys and production DataFrames as values
            output_path: Optional path to save figure
            dpi: Resolution for output
            vmin: Minimum value for color scale (log10 kcal)
            vmax: Maximum value for color scale (log10 kcal)
            n_colors: Number of discrete colors in palette
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Creating 4-panel production map...")
        
        # Define crop order and labels
        crops = ['wheat', 'rice', 'maize', 'allgrain']
        labels = ['Wheat', 'Rice', 'Maize', 'All Grain']
        
        # Create figure with 2x2 grid
        fig = plt.figure(figsize=(16, 12), dpi=dpi)
        
        # Calculate global color scale if not provided
        if vmin is None or vmax is None:
            vmin, vmax = self._calculate_global_scale(crop_data, crops[:3])  # wheat, rice, maize
            self.logger.info(f"Global production scale: {vmin:.2f} to {vmax:.2f} (log10 kcal)")
        
        # Create discretized colormap (YlOrRd from ColorBrewer)
        cmap = self._create_discrete_cmap('YlOrRd', n_colors)
        
        # Create each panel
        for idx, (crop, label) in enumerate(zip(crops, labels)):
            if crop not in crop_data:
                self.logger.warning(f"No data for {crop}, skipping panel")
                continue
            
            # Use separate scale for allgrain
            if crop == 'allgrain':
                crop_vmin, crop_vmax = self._calculate_crop_scale(crop_data[crop], crop)
                self.logger.info(f"All grain scale: {crop_vmin:.2f} to {crop_vmax:.2f} (log10 kcal)")
            else:
                crop_vmin, crop_vmax = vmin, vmax
            
            # Create subplot
            ax = fig.add_subplot(2, 2, idx + 1, projection=ccrs.Robinson())
            
            # Plot data
            self._plot_crop_panel(
                ax, crop_data[crop], label, cmap, crop_vmin, crop_vmax, crop
            )
        
        # Add shared colorbar for wheat/rice/maize
        self._add_shared_colorbar(fig, cmap, vmin, vmax, 'Production (log₁₀ kcal)')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            # Save PNG
            self.logger.info(f"Saving 4-panel production map to {output_path}")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            # Save SVG for Inkscape editing
            svg_path = output_path.with_suffix('.svg')
            self.logger.info(f"Saving SVG version to {svg_path}")
            fig.savefig(svg_path, format='svg', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        self.logger.info("4-panel production map created successfully")
        return fig
    
    def create_four_panel_harvest_map(
        self,
        crop_data: Dict[str, pd.DataFrame],
        output_path: Optional[Path] = None,
        dpi: int = 300,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        n_colors: int = 9
    ) -> plt.Figure:
        """Create 4-panel harvest area map (wheat, rice, maize, allgrain).
        
        Args:
            crop_data: Dictionary with crop names as keys and harvest DataFrames as values
            output_path: Optional path to save figure
            dpi: Resolution for output
            vmin: Minimum value for color scale (log10 ha)
            vmax: Maximum value for color scale (log10 ha)
            n_colors: Number of discrete colors in palette
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Creating 4-panel harvest area map...")
        
        # Define crop order and labels
        crops = ['wheat', 'rice', 'maize', 'allgrain']
        labels = ['Wheat', 'Rice', 'Maize', 'All Grain']
        
        # Create figure with 2x2 grid
        fig = plt.figure(figsize=(16, 12), dpi=dpi)
        
        # Calculate global color scale if not provided
        if vmin is None or vmax is None:
            vmin, vmax = self._calculate_global_scale(crop_data, crops[:3])  # wheat, rice, maize
            self.logger.info(f"Global harvest scale: {vmin:.2f} to {vmax:.2f} (log10 ha)")
        
        # Create discretized colormap (YlGn from ColorBrewer)
        cmap = self._create_discrete_cmap('YlGn', n_colors)
        
        # Create each panel
        for idx, (crop, label) in enumerate(zip(crops, labels)):
            if crop not in crop_data:
                self.logger.warning(f"No data for {crop}, skipping panel")
                continue
            
            # Use separate scale for allgrain
            if crop == 'allgrain':
                crop_vmin, crop_vmax = self._calculate_crop_scale(crop_data[crop], crop)
                self.logger.info(f"All grain scale: {crop_vmin:.2f} to {crop_vmax:.2f} (log10 ha)")
            else:
                crop_vmin, crop_vmax = vmin, vmax
            
            # Create subplot
            ax = fig.add_subplot(2, 2, idx + 1, projection=ccrs.Robinson())
            
            # Plot data
            self._plot_crop_panel(
                ax, crop_data[crop], label, cmap, crop_vmin, crop_vmax, crop
            )
        
        # Add shared colorbar for wheat/rice/maize
        self._add_shared_colorbar(fig, cmap, vmin, vmax, 'Harvest Area (log₁₀ ha)')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            # Save PNG
            self.logger.info(f"Saving 4-panel harvest map to {output_path}")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            # Save SVG for Inkscape editing
            svg_path = output_path.with_suffix('.svg')
            self.logger.info(f"Saving SVG version to {svg_path}")
            fig.savefig(svg_path, format='svg', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        self.logger.info("4-panel harvest area map created successfully")
        return fig
    
    def _plot_crop_panel(
        self,
        ax: plt.Axes,
        data_df: pd.DataFrame,
        title: str,
        cmap: mcolors.ListedColormap,
        vmin: float,
        vmax: float,
        crop_name: str,
        fontsize: int = 14
    ) -> None:
        """Plot a single crop panel."""
        # Prepare grid data
        lon_grid, lat_grid, data_grid = self._prepare_grid_data(data_df, crop_name)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='gray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Mask zero values
        data_masked = np.ma.masked_where(data_grid <= 0, data_grid)
        
        # Plot data using scatter
        if data_masked.count() > 0:
            # Get coordinates of data points
            data_indices = np.where(~data_masked.mask)
            data_lats = lat_grid[data_indices]
            data_lons = lon_grid[data_indices]
            data_values = data_masked[data_indices]
            
            # Adjust point size based on number of points
            if data_masked.count() < 10000:
                point_size = 15
            elif data_masked.count() < 100000:
                point_size = 4
            else:
                point_size = 1.5
            
            # Create discrete normalization matching colorbar
            n_colors = cmap.N
            boundaries = np.linspace(vmin, vmax, n_colors + 1)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N)
            
            # Plot as scatter points
            scatter = ax.scatter(
                data_lons, data_lats,
                c=data_values,
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                edgecolors='none'
            )
        
        # Set title with larger font
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=10)
        
        # Set global extent
        ax.set_global()
    
    def _prepare_grid_data(self, df: pd.DataFrame, crop_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare grid data for plotting.
        
        Args:
            df: DataFrame with SPAM data (contains all crops)
            crop_name: Name of crop to filter ('wheat', 'rice', 'maize', 'allgrain')
        """
        # Get grid parameters
        ncols = int(GRID_PARAMS['ncols'])
        nrows = int(GRID_PARAMS['nrows'])
        xllcorner = GRID_PARAMS['xllcorner']
        yllcorner = GRID_PARAMS['yllcorner']
        cellsize = GRID_PARAMS['cellsize']
        
        # Create coordinate grids
        lon_1d = np.linspace(xllcorner + cellsize/2,
                            xllcorner + (ncols-1)*cellsize + cellsize/2,
                            ncols)
        lat_1d = np.linspace(yllcorner + cellsize/2,
                            yllcorner + (nrows-1)*cellsize + cellsize/2,
                            nrows)
        
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        
        # Filter crop columns based on crop name
        if crop_name == 'wheat':
            crop_columns = [col for col in df.columns if 'WHEA' in col.upper() and (col.endswith('_A') or col.endswith('_Y'))]
        elif crop_name == 'rice':
            crop_columns = [col for col in df.columns if 'RICE' in col.upper() and (col.endswith('_A') or col.endswith('_Y'))]
        elif crop_name == 'maize':
            crop_columns = [col for col in df.columns if 'MAIZ' in col.upper() and (col.endswith('_A') or col.endswith('_Y'))]
        elif crop_name == 'allgrain':
            # Include all grain crops
            grain_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'PMIL_A', 'SMIL_A', 'SORG_A', 'OCER_A',
                          'WHEA_Y', 'RICE_Y', 'MAIZ_Y', 'BARL_Y', 'PMIL_Y', 'SMIL_Y', 'SORG_Y', 'OCER_Y']
            crop_columns = [col for col in df.columns if col in grain_crops]
        else:
            # Fallback: use all columns
            crop_columns = [col for col in df.columns if col.endswith('_A') or col.endswith('_Y')]
        
        if not crop_columns:
            raise ValueError(f"No crop columns found for {crop_name}")
        
        # Calculate total per grid cell
        total_data = df[crop_columns].sum(axis=1)
        
        # Initialize grid
        data_grid = np.zeros((nrows, ncols))
        
        # Map data to grid using coordinates
        for idx, row in df.iterrows():
            if pd.notna(row['x']) and pd.notna(row['y']):
                # Find closest grid cell
                lon_idx = np.argmin(np.abs(lon_1d - row['x']))
                lat_idx = np.argmin(np.abs(lat_1d - row['y']))
                
                # Add value (convert to log10 for visualization)
                value = total_data.iloc[idx] if isinstance(idx, int) else total_data.loc[idx]
                if value > 0:
                    data_grid[lat_idx, lon_idx] = np.log10(value)
        
        return lon_grid, lat_grid, data_grid
    
    def _calculate_global_scale(
        self,
        crop_data: Dict[str, pd.DataFrame],
        crops: List[str]
    ) -> Tuple[float, float]:
        """Calculate global min/max for consistent color scale."""
        all_values = []
        
        for crop in crops:
            if crop in crop_data:
                lon_grid, lat_grid, data_grid = self._prepare_grid_data(crop_data[crop], crop)
                values = data_grid[data_grid > 0]
                if len(values) > 0:
                    all_values.extend(values)
        
        if len(all_values) == 0:
            return 0, 10
        
        all_values = np.array(all_values)
        vmin = np.percentile(all_values, 1)  # 1st percentile
        vmax = np.percentile(all_values, 99)  # 99th percentile
        
        return vmin, vmax
    
    def _calculate_crop_scale(self, df: pd.DataFrame, crop_name: str) -> Tuple[float, float]:
        """Calculate min/max for a single crop."""
        lon_grid, lat_grid, data_grid = self._prepare_grid_data(df, crop_name)
        values = data_grid[data_grid > 0]
        
        if len(values) == 0:
            return 0, 10
        
        vmin = np.percentile(values, 1)
        vmax = np.percentile(values, 99)
        
        return vmin, vmax
    
    def _create_discrete_cmap(self, name: str, n_colors: int) -> mcolors.ListedColormap:
        """Create discretized colormap from ColorBrewer palette."""
        # Get base colormap
        base_cmap = plt.cm.get_cmap(name)
        
        # Sample colors evenly
        colors = base_cmap(np.linspace(0, 1, n_colors))
        
        # Create discrete colormap
        return mcolors.ListedColormap(colors)
    
    def _add_shared_colorbar(
        self,
        fig: plt.Figure,
        cmap: mcolors.ListedColormap,
        vmin: float,
        vmax: float,
        label: str
    ) -> None:
        """Add shared colorbar at bottom of figure with ticks at color boundaries."""
        # Create colorbar axis
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        
        # Create discrete normalization with boundaries at color transitions
        n_colors = cmap.N
        boundaries = np.linspace(vmin, vmax, n_colors + 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        
        # Create scalar mappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Add colorbar
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', 
                           spacing='proportional', boundaries=boundaries)
        
        # Set ticks at boundaries
        cbar.set_ticks(boundaries)
        
        # Format tick labels to show actual values (10^x)
        # Convert log10 values back to actual values for display
        tick_labels = []
        for val in boundaries:
            actual_val = 10**val
            if actual_val < 1000:
                tick_labels.append(f'{actual_val:.0f}')
            elif actual_val < 1e6:
                tick_labels.append(f'{actual_val/1e3:.0f}K')
            elif actual_val < 1e9:
                tick_labels.append(f'{actual_val/1e6:.1f}M')
            else:
                tick_labels.append(f'{actual_val/1e9:.1f}B')
        
        cbar.set_ticklabels(tick_labels)
        cbar.set_label(label.replace('log₁₀ ', ''), fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
    
    def _add_dual_colorbars(
        self,
        fig: plt.Figure,
        prod_cmap: mcolors.ListedColormap,
        prod_vmin: float,
        prod_vmax: float,
        harvest_cmap: mcolors.ListedColormap,
        harvest_vmin: float,
        harvest_vmax: float
    ) -> None:
        """Add two separate colorbars for production and harvest area."""
        # Production colorbar (left side) - moved higher
        prod_cbar_ax = fig.add_axes([0.12, 0.04, 0.35, 0.02])
        
        n_colors = prod_cmap.N
        prod_boundaries = np.linspace(prod_vmin, prod_vmax, n_colors + 1)
        prod_norm = mcolors.BoundaryNorm(prod_boundaries, prod_cmap.N)
        
        prod_sm = plt.cm.ScalarMappable(cmap=prod_cmap, norm=prod_norm)
        prod_sm.set_array([])
        
        prod_cbar = fig.colorbar(prod_sm, cax=prod_cbar_ax, orientation='horizontal',
                                spacing='proportional', boundaries=prod_boundaries)
        prod_cbar.set_ticks(prod_boundaries)
        
        # Format production tick labels
        prod_tick_labels = []
        for val in prod_boundaries:
            actual_val = 10**val
            if actual_val < 1e6:
                prod_tick_labels.append(f'{actual_val/1e3:.0f}K')
            elif actual_val < 1e9:
                prod_tick_labels.append(f'{actual_val/1e6:.0f}M')
            elif actual_val < 1e12:
                prod_tick_labels.append(f'{actual_val/1e9:.0f}B')
            else:
                prod_tick_labels.append(f'{actual_val/1e12:.0f}T')
        
        prod_cbar.set_ticklabels(prod_tick_labels)
        prod_cbar.set_label('Production (kcal)', fontsize=13, fontweight='bold')
        prod_cbar.ax.tick_params(labelsize=10)
        
        # Harvest area colorbar (right side) - moved higher
        harvest_cbar_ax = fig.add_axes([0.53, 0.04, 0.35, 0.02])
        
        harvest_boundaries = np.linspace(harvest_vmin, harvest_vmax, n_colors + 1)
        harvest_norm = mcolors.BoundaryNorm(harvest_boundaries, harvest_cmap.N)
        
        harvest_sm = plt.cm.ScalarMappable(cmap=harvest_cmap, norm=harvest_norm)
        harvest_sm.set_array([])
        
        harvest_cbar = fig.colorbar(harvest_sm, cax=harvest_cbar_ax, orientation='horizontal',
                                    spacing='proportional', boundaries=harvest_boundaries)
        harvest_cbar.set_ticks(harvest_boundaries)
        
        # Format harvest tick labels
        harvest_tick_labels = []
        for val in harvest_boundaries:
            actual_val = 10**val
            if actual_val < 1000:
                harvest_tick_labels.append(f'{actual_val:.0f}')
            elif actual_val < 1e6:
                harvest_tick_labels.append(f'{actual_val/1e3:.0f}K')
            else:
                harvest_tick_labels.append(f'{actual_val/1e6:.0f}M')
        
        harvest_cbar.set_ticklabels(harvest_tick_labels)
        harvest_cbar.set_label('Harvest Area (ha)', fontsize=13, fontweight='bold')
        harvest_cbar.ax.tick_params(labelsize=10)

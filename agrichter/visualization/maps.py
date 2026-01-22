"""Global production map visualization using Cartopy."""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from ..core.config import Config
from ..core.constants import GRID_PARAMS
from ..data.grid_loader import GridLoader
from .publication import PublicationFormatter


class GlobalProductionMapper:
    """Creates global production maps with Robinson projection."""
    
    def __init__(self, config: Config):
        """Initialize the global production mapper.
        
        Args:
            config: Configuration object with crop and output settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up custom colormap (yellow to green)
        self.production_colormap = self._create_production_colormap()
        
        # Initialize grid loader
        self.grid_loader = GridLoader()
        
    def _create_production_colormap(self) -> LinearSegmentedColormap:
        """Create yellow-to-green colormap for production intensity.
        
        Returns:
            Custom colormap for production visualization
        """
        colors = ['#FFFF99', '#CCFF66', '#99FF33', '#66FF00', '#33CC00', '#009900']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('production', colors, N=n_bins)
        return cmap
    
    def _prepare_grid_data(self, production_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare production data for mapping using proper SPAM grid structure.
        
        Args:
            production_df: DataFrame with production data in kcal
            
        Returns:
            Tuple of (longitude_grid, latitude_grid, production_grid)
        """
        self.logger.info("Preparing grid data for mapping using SPAM grid structure...")
        
        try:
            # Use grid loader to map production data properly
            lon_grid, lat_grid, production_grid = self.grid_loader.map_production_to_grid(production_df)
            return lon_grid, lat_grid, production_grid
            
        except Exception as e:
            self.logger.warning(f"Failed to use SPAM grid structure: {str(e)}")
            self.logger.info("Falling back to coordinate-based mapping...")
            
            # Fallback to coordinate-based approach
            return self._prepare_grid_data_fallback(production_df)
    
    def _prepare_grid_data_fallback(self, production_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback method using coordinate-based mapping.
        
        Args:
            production_df: DataFrame with production data in kcal
            
        Returns:
            Tuple of (longitude_grid, latitude_grid, production_grid)
        """
        self.logger.info("Using coordinate-based grid mapping (fallback)...")
        
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
        
        # Initialize production grid
        production_grid = np.zeros((nrows, ncols))
        
        # Get crop columns
        crop_columns = [col for col in production_df.columns if col.endswith('_A')]
        
        if not crop_columns:
            self.logger.warning("No crop columns found in production data")
            return lon_grid, lat_grid, production_grid
        
        # Calculate total production per grid cell
        total_production = production_df[crop_columns].sum(axis=1)
        
        # Filter to only cells with production > 0 for efficiency
        has_production = total_production > 0
        production_subset = production_df[has_production].copy()
        production_values = total_production[has_production]
        
        self.logger.info(f"Processing {len(production_subset)} cells with production out of {len(production_df)} total")
        
        # Map production values to grid using vectorized operations
        if len(production_subset) > 0:
            # Find grid indices for all points at once
            lon_indices = np.searchsorted(lon_1d, production_subset['x'].values, side='left')
            lat_indices = np.searchsorted(lat_1d, production_subset['y'].values, side='left')
            
            # Clip indices to valid range
            lon_indices = np.clip(lon_indices, 0, ncols - 1)
            lat_indices = np.clip(lat_indices, 0, nrows - 1)
            
            # Convert production to log10 for visualization
            log_production = np.log10(production_values.values)
            
            # Assign values to grid (handle multiple points mapping to same cell by taking maximum)
            for i, (lat_idx, lon_idx, log_prod) in enumerate(zip(lat_indices, lon_indices, log_production)):
                production_grid[lat_idx, lon_idx] = max(production_grid[lat_idx, lon_idx], log_prod)
        
        # Count non-zero cells in grid
        non_zero_cells = (production_grid > 0).sum()
        
        self.logger.info(f"Grid data prepared: {ncols}x{nrows} cells")
        self.logger.info(f"Cells with production data: {non_zero_cells}")
        
        if non_zero_cells > 0:
            self.logger.info(f"Production range: {production_grid[production_grid > 0].min():.2f} - {production_grid.max():.2f} (log10 kcal)")
        else:
            self.logger.warning("No production data mapped to grid!")
        
        return lon_grid, lat_grid, production_grid
    
    def create_global_map(self, 
                         production_df: pd.DataFrame,
                         title: Optional[str] = None,
                         output_path: Optional[Path] = None,
                         dpi: int = 300,
                         figsize: Tuple[float, float] = (12, 8),
                         use_publication_style: bool = True,
                         save_formats: Optional[list] = None) -> plt.Figure:
        """Create global production map with Robinson projection.
        
        Args:
            production_df: DataFrame with production data in kcal
            title: Optional title for the map
            output_path: Optional path to save the figure
            dpi: Resolution for output image
            figsize: Figure size in inches
            use_publication_style: Whether to apply publication formatting
            save_formats: List of formats to save (if output_path provided)
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating global production map...")
        
        # Prepare grid data
        lon_grid, lat_grid, production_grid = self._prepare_grid_data(production_df)
        
        # Use publication formatter if requested
        formatter = PublicationFormatter(self.config) if use_publication_style else None
        
        if formatter:
            formatter.apply_publication_style()
            fig = formatter.create_figure(figsize=figsize, dpi=dpi)
        else:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Mask zero values for plotting
        production_masked = np.ma.masked_where(production_grid <= 0, production_grid)
        
        # Plot production data
        if production_masked.count() > 0:  # Only plot if we have data
            # Use scatter for all data to avoid streaky appearance from pcolormesh
            # This ensures all cells are visible
            self.logger.info(f"Plotting {production_masked.count()} cells with production data")
            
            # Get coordinates of data points
            data_indices = np.where(~production_masked.mask)
            data_lats = lat_grid[data_indices]
            data_lons = lon_grid[data_indices]
            data_values = production_masked[data_indices]
            
            # Adjust point size based on number of points
            if production_masked.count() < 10000:
                point_size = 20
                edge_width = 0.1
            elif production_masked.count() < 100000:
                point_size = 5
                edge_width = 0
            else:
                point_size = 2
                edge_width = 0
            
            # Plot as scatter points
            scatter = ax.scatter(data_lons, data_lats, 
                               c=data_values, 
                               cmap=self.production_colormap,
                               s=point_size,
                               alpha=0.8,
                               transform=ccrs.PlateCarree(),
                               edgecolors='black' if edge_width > 0 else 'none',
                               linewidths=edge_width)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                              pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label('Production Intensity (log₁₀ kcal)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        else:
            self.logger.warning("No production data to plot")
        
        # Set title
        if title is None:
            crop_type = self.config.crop_type.title()
            title = f'Global {crop_type} Production Distribution'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid lines
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
        
        # Set global extent
        ax.set_global()
        
        # Adjust layout (ignore layout engine warnings with colorbar)
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore colorbar layout warnings
        
        # Save figure if path provided
        if output_path:
            if formatter and save_formats:
                # Use publication formatter for multiple formats
                formatter.save_figure(fig, output_path, save_formats, dpi=dpi)
            else:
                # Standard single format save
                self.logger.info(f"Saving map to {output_path}")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
        
        # Restore original style if formatter was used
        if formatter:
            formatter.restore_original_style()
        
        self.logger.info("Global production map created successfully")
        return fig
    
    def create_production_density_map(self,
                                    production_df: pd.DataFrame,
                                    harvest_df: pd.DataFrame,
                                    title: Optional[str] = None,
                                    output_path: Optional[Path] = None,
                                    dpi: int = 300,
                                    figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create map showing production density (yield per area).
        
        Args:
            production_df: DataFrame with production data in kcal
            harvest_df: DataFrame with harvest area data in km²
            title: Optional title for the map
            output_path: Optional path to save the figure
            dpi: Resolution for output image
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating production density map...")
        
        # Get crop columns
        crop_columns = [col for col in production_df.columns if col.endswith('_A')]
        
        if not crop_columns:
            raise ValueError("No crop columns found in production data")
        
        # Calculate total production and harvest area
        total_production = production_df[crop_columns].sum(axis=1)
        total_harvest = harvest_df[crop_columns].sum(axis=1)
        
        # Calculate yield (production per unit area)
        yield_data = total_production / total_harvest
        yield_data = yield_data.replace([np.inf, -np.inf], np.nan)
        
        # Create modified production dataframe with yield values for grid mapping
        # Keep the structure but use only one crop column with yield data
        yield_df = production_df.copy()
        
        # Zero out all crop columns except the first one
        for col in crop_columns[1:]:
            yield_df[col] = 0
        
        # Put yield data in the first crop column
        yield_df[crop_columns[0]] = yield_data
        
        # Prepare grid data using the same method as production/harvest
        lon_grid, lat_grid, yield_grid = self._prepare_grid_data(yield_df)
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Mask invalid values
        yield_masked = np.ma.masked_where(yield_grid <= 0, yield_grid)
        
        # Plot yield data using scatter (same approach as production and harvest)
        if yield_masked.count() > 0:
            self.logger.info(f"Plotting {yield_masked.count()} cells with yield data")
            
            # Get coordinates of data points
            data_indices = np.where(~yield_masked.mask)
            data_lats = lat_grid[data_indices]
            data_lons = lon_grid[data_indices]
            data_values = yield_masked[data_indices]
            
            # Adjust point size based on number of points
            if yield_masked.count() < 10000:
                point_size = 20
                edge_width = 0.1
            elif yield_masked.count() < 100000:
                point_size = 5
                edge_width = 0
            else:
                point_size = 2
                edge_width = 0
            
            # Plot as scatter points
            scatter = ax.scatter(data_lons, data_lats, 
                               c=data_values, 
                               cmap=self.production_colormap,
                               s=point_size,
                               alpha=0.8,
                               transform=ccrs.PlateCarree(),
                               edgecolors='black' if edge_width > 0 else 'none',
                               linewidths=edge_width)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                              pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label('Production Density (log₁₀ kcal/km²)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        
        # Set title
        if title is None:
            crop_type = self.config.crop_type.title()
            title = f'Global {crop_type} Production Density'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid lines
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
        
        # Set global extent
        ax.set_global()
        
        # Adjust layout (ignore layout engine warnings with colorbar)
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore colorbar layout warnings
        
        # Save figure if path provided
        if output_path:
            self.logger.info(f"Saving density map to {output_path}")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        self.logger.info("Production density map created successfully")
        return fig
    
    def _prepare_yield_grid(self, yield_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare yield data for mapping.
        
        Args:
            yield_df: DataFrame with yield data
            
        Returns:
            Tuple of (longitude_grid, latitude_grid, yield_grid)
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
        
        # Initialize yield grid
        yield_grid = np.zeros((nrows, ncols))
        
        # Map yield values to grid
        for idx, row in yield_df.iterrows():
            if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['yield']):
                # Find closest grid cell
                lon_idx = np.argmin(np.abs(lon_1d - row['x']))
                lat_idx = np.argmin(np.abs(lat_1d - row['y']))
                
                # Add yield value (convert to log10 for visualization)
                yield_value = row['yield']
                if yield_value > 0:
                    yield_grid[lat_idx, lon_idx] = np.log10(yield_value)
        
        return lon_grid, lat_grid, yield_grid
    
    def create_yield_map(self,
                        yield_df: pd.DataFrame,
                        title: Optional[str] = None,
                        output_path: Optional[Path] = None,
                        dpi: int = 150,
                        figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create global yield map using actual SPAM yield data.
        
        Args:
            yield_df: DataFrame with yield data in tons/ha
            title: Optional title for the map
            output_path: Optional path to save the figure
            dpi: Resolution for output image
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating global yield map...")
        
        # Prepare grid data (yield columns end with _Y)
        lon_grid, lat_grid, yield_grid = self._prepare_grid_data(yield_df)
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Mask zero values
        yield_masked = np.ma.masked_where(yield_grid <= 0, yield_grid)
        
        # Plot yield data using scatter
        if yield_masked.count() > 0:
            self.logger.info(f"Plotting {yield_masked.count()} cells with yield data")
            
            # Get coordinates of data points
            data_indices = np.where(~yield_masked.mask)
            data_lats = lat_grid[data_indices]
            data_lons = lon_grid[data_indices]
            data_values = yield_masked[data_indices]
            
            # Adjust point size based on number of points
            if yield_masked.count() < 10000:
                point_size = 20
                edge_width = 0.1
            elif yield_masked.count() < 100000:
                point_size = 5
                edge_width = 0
            else:
                point_size = 2
                edge_width = 0
            
            # Use green-yellow colormap for yield (productivity)
            scatter = ax.scatter(data_lons, data_lats, 
                               c=data_values, 
                               cmap='YlGn',  # Yellow to Green (productivity/growth)
                               s=point_size,
                               alpha=0.8,
                               transform=ccrs.PlateCarree(),
                               edgecolors='black' if edge_width > 0 else 'none',
                               linewidths=edge_width)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                              pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label('Yield (log₁₀ tons/ha)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        
        # Set title
        if title is None:
            crop_type = self.config.crop_type.title()
            title = f'Global {crop_type} Yield'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set global extent
        ax.set_global()
        
        # Adjust layout
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore colorbar layout warnings
        
        # Save if path provided
        if output_path:
            self.logger.info(f"Saving yield map to {output_path}")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        self.logger.info("Global yield map created successfully")
        return fig
    
    def create_harvest_area_map(self,
                                harvest_df: pd.DataFrame,
                                title: Optional[str] = None,
                                output_path: Optional[Path] = None,
                                dpi: int = 150,
                                figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
        """Create global harvest area map.
        
        Args:
            harvest_df: DataFrame with harvest area data in ha
            title: Optional title for the map
            output_path: Optional path to save the figure
            dpi: Resolution for output image
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating global harvest area map...")
        
        # Prepare grid data
        lon_grid, lat_grid, harvest_grid = self._prepare_grid_data(harvest_df)
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Mask zero values
        harvest_masked = np.ma.masked_where(harvest_grid <= 0, harvest_grid)
        
        # Plot harvest data
        if harvest_masked.count() > 0:
            self.logger.info(f"Plotting {harvest_masked.count()} cells with harvest data")
            
            # Get coordinates of data points
            data_indices = np.where(~harvest_masked.mask)
            data_lats = lat_grid[data_indices]
            data_lons = lon_grid[data_indices]
            data_values = harvest_masked[data_indices]
            
            # Adjust point size based on number of points
            if harvest_masked.count() < 10000:
                point_size = 20
                edge_width = 0.1
            elif harvest_masked.count() < 100000:
                point_size = 5
                edge_width = 0
            else:
                point_size = 2
                edge_width = 0
            
            # Use green colormap for harvest area
            scatter = ax.scatter(data_lons, data_lats, 
                               c=data_values, 
                               cmap='YlGn',  # Yellow to Green
                               s=point_size,
                               alpha=0.8,
                               transform=ccrs.PlateCarree(),
                               edgecolors='black' if edge_width > 0 else 'none',
                               linewidths=edge_width)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                              pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label('Harvest Area (log₁₀ ha)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        
        # Set title
        if title is None:
            crop_type = self.config.crop_type.title()
            title = f'Global {crop_type} Harvest Area'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set global extent
        ax.set_global()
        
        # Adjust layout
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore colorbar layout warnings
        
        # Save if path provided
        if output_path:
            self.logger.info(f"Saving harvest area map to {output_path}")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        self.logger.info("Global harvest area map created successfully")
        return fig

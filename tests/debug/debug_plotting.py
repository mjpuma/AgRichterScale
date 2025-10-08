#!/usr/bin/env python3
"""
Debug the actual plotting process to see why maps are blank.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.visualization.maps import GlobalProductionMapper

def debug_plotting_step_by_step():
    """Debug each step of the plotting process."""
    
    logger = logging.getLogger(__name__)
    
    # Initialize components
    config = Config(crop_type='wheat', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    mapper = GlobalProductionMapper(config)
    
    # Get a small sample with guaranteed production
    logger.info("Getting sample data with production...")
    full_df = loader.load_spam_production()
    crop_production = processor.filter_crop_data(full_df)
    production_kcal = processor.convert_production_to_kcal(crop_production)
    
    # Get cells with production
    crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
    total_production = production_kcal[crop_columns].sum(axis=1)
    has_production = total_production > 0
    production_sample = production_kcal[has_production].sample(n=500, random_state=42)
    
    logger.info(f"Sample size: {len(production_sample)} cells")
    
    # Step 1: Test grid preparation
    logger.info("\n=== STEP 1: Grid Preparation ===")
    lon_grid, lat_grid, production_grid = mapper._prepare_grid_data(production_sample)
    
    non_zero_cells = (production_grid > 0).sum()
    logger.info(f"Grid cells with data: {non_zero_cells}")
    logger.info(f"Production grid shape: {production_grid.shape}")
    logger.info(f"Production range: {production_grid[production_grid > 0].min():.2f} - {production_grid.max():.2f}")
    
    # Step 2: Test masking
    logger.info("\n=== STEP 2: Data Masking ===")
    production_masked = np.ma.masked_where(production_grid <= 0, production_grid)
    logger.info(f"Masked array count (non-masked): {production_masked.count()}")
    logger.info(f"Masked array range: {production_masked.min():.2f} - {production_masked.max():.2f}")
    
    # Step 3: Test basic plotting without Cartopy
    logger.info("\n=== STEP 3: Basic Plot Test ===")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if production_masked.count() > 0:
        im = ax.pcolormesh(lon_grid, lat_grid, production_masked, 
                          cmap=mapper.production_colormap, shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title('Basic Production Plot (No Projection)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Save basic plot
        basic_output = Path('test_outputs/debug_basic_plot.png')
        fig.savefig(basic_output, dpi=150, bbox_inches='tight')
        logger.info(f"Basic plot saved: {basic_output}")
        
        if basic_output.exists():
            logger.info(f"✅ Basic plot file created ({basic_output.stat().st_size / 1024:.1f} KB)")
        else:
            logger.error("❌ Basic plot file not created")
    else:
        logger.error("❌ No data to plot in basic test")
    
    plt.close(fig)
    
    # Step 4: Test with Robinson projection
    logger.info("\n=== STEP 4: Robinson Projection Test ===")
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Add basic map features first
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.7)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    logger.info("Map features added")
    
    # Plot data with transform
    if production_masked.count() > 0:
        logger.info("Adding production data to Robinson projection...")
        im = ax.pcolormesh(lon_grid, lat_grid, production_masked,
                          transform=ccrs.PlateCarree(),
                          cmap=mapper.production_colormap,
                          shading='auto',
                          alpha=0.8)
        
        logger.info("Production data plotted")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                          pad=0.05, shrink=0.8, aspect=40)
        cbar.set_label('Production Intensity (log₁₀ kcal)', fontsize=10)
        
        logger.info("Colorbar added")
    
    ax.set_title('Robinson Projection Test', fontsize=14, fontweight='bold', pad=20)
    ax.set_global()
    
    # Save Robinson plot
    robinson_output = Path('test_outputs/debug_robinson_plot.png')
    fig.savefig(robinson_output, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    logger.info(f"Robinson plot saved: {robinson_output}")
    
    if robinson_output.exists():
        logger.info(f"✅ Robinson plot file created ({robinson_output.stat().st_size / 1024:.1f} KB)")
    else:
        logger.error("❌ Robinson plot file not created")
    
    plt.close(fig)
    
    # Step 5: Test coordinate ranges
    logger.info("\n=== STEP 5: Coordinate Range Analysis ===")
    
    # Check if data coordinates are within expected ranges
    sample_coords = production_sample[['x', 'y']].copy()
    logger.info(f"Sample coordinate ranges:")
    logger.info(f"  Longitude: {sample_coords['x'].min():.2f} to {sample_coords['x'].max():.2f}")
    logger.info(f"  Latitude: {sample_coords['y'].min():.2f} to {sample_coords['y'].max():.2f}")
    
    # Check grid coordinate ranges
    logger.info(f"Grid coordinate ranges:")
    logger.info(f"  Longitude: {lon_grid.min():.2f} to {lon_grid.max():.2f}")
    logger.info(f"  Latitude: {lat_grid.min():.2f} to {lat_grid.max():.2f}")
    
    # Check where data actually gets mapped
    data_locations = np.where(production_grid > 0)
    if len(data_locations[0]) > 0:
        mapped_lats = lat_grid[data_locations]
        mapped_lons = lon_grid[data_locations]
        logger.info(f"Mapped data coordinate ranges:")
        logger.info(f"  Longitude: {mapped_lons.min():.2f} to {mapped_lons.max():.2f}")
        logger.info(f"  Latitude: {mapped_lats.min():.2f} to {mapped_lats.max():.2f}")
        
        # Show some specific mapped points
        logger.info(f"Sample mapped points:")
        for i in range(min(5, len(mapped_lats))):
            lat_idx, lon_idx = data_locations[0][i], data_locations[1][i]
            logger.info(f"  Grid[{lat_idx:4d},{lon_idx:4d}]: lat={mapped_lats[i]:7.2f}, lon={mapped_lons[i]:8.2f}, val={production_grid[lat_idx, lon_idx]:.2f}")

def main():
    """Debug plotting step by step."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory
    Path('test_outputs').mkdir(exist_ok=True)
    
    logger.info("🔍 Debugging Plotting Process")
    logger.info("=" * 50)
    
    debug_plotting_step_by_step()

if __name__ == "__main__":
    main()
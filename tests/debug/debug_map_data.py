#!/usr/bin/env python3
"""
Debug script to investigate why maps show no production values.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.core.constants import GRID_PARAMS

def debug_coordinate_mapping():
    """Debug coordinate mapping and data distribution."""
    
    logger = logging.getLogger(__name__)
    
    # Initialize components
    config = Config(crop_type='wheat', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    
    # Load small sample
    logger.info("Loading production data sample...")
    production_df = loader.load_spam_production().head(1000)
    
    # Filter and convert
    crop_production = processor.filter_crop_data(production_df)
    production_kcal = processor.convert_production_to_kcal(crop_production)
    
    # Check coordinate ranges
    logger.info(f"\nCoordinate Analysis:")
    logger.info(f"X (longitude) range: {production_kcal['x'].min():.3f} to {production_kcal['x'].max():.3f}")
    logger.info(f"Y (latitude) range: {production_kcal['y'].min():.3f} to {production_kcal['y'].max():.3f}")
    
    # Check production values
    crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
    if crop_columns:
        total_production = production_kcal[crop_columns].sum(axis=1)
        non_zero_production = total_production[total_production > 0]
        
        logger.info(f"\nProduction Analysis:")
        logger.info(f"Total cells: {len(production_kcal)}")
        logger.info(f"Cells with production: {len(non_zero_production)}")
        logger.info(f"Production range: {non_zero_production.min():.2e} to {non_zero_production.max():.2e} kcal")
        
        # Show some sample data
        logger.info(f"\nSample data with production:")
        sample_with_prod = production_kcal[total_production > 0].head(10)
        for idx, row in sample_with_prod.iterrows():
            prod_val = total_production.iloc[idx] if idx < len(total_production) else 0
            logger.info(f"  x={row['x']:8.3f}, y={row['y']:7.3f}, prod={prod_val:.2e} kcal")
    
    # Check grid parameters
    logger.info(f"\nGrid Parameters:")
    for key, value in GRID_PARAMS.items():
        logger.info(f"  {key}: {value}")
    
    # Test coordinate mapping logic
    logger.info(f"\nTesting coordinate mapping logic:")
    
    # Grid setup (same as in maps.py)
    ncols = int(GRID_PARAMS['ncols'])
    nrows = int(GRID_PARAMS['nrows'])
    xllcorner = GRID_PARAMS['xllcorner']
    yllcorner = GRID_PARAMS['yllcorner']
    cellsize = GRID_PARAMS['cellsize']
    
    # Create coordinate arrays
    lon_1d = np.linspace(xllcorner + cellsize/2, 
                        xllcorner + (ncols-1)*cellsize + cellsize/2, 
                        ncols)
    lat_1d = np.linspace(yllcorner + cellsize/2,
                        yllcorner + (nrows-1)*cellsize + cellsize/2,
                        nrows)
    
    logger.info(f"Grid longitude range: {lon_1d.min():.3f} to {lon_1d.max():.3f}")
    logger.info(f"Grid latitude range: {lat_1d.min():.3f} to {lat_1d.max():.3f}")
    
    # Test mapping for a few sample points
    if len(non_zero_production) > 0:
        logger.info(f"\nTesting coordinate mapping for sample points:")
        sample_indices = non_zero_production.head(5).index
        
        for idx in sample_indices:
            row = production_kcal.loc[idx]
            prod_val = total_production.loc[idx]
            
            # Find closest grid indices
            lon_idx = np.argmin(np.abs(lon_1d - row['x']))
            lat_idx = np.argmin(np.abs(lat_1d - row['y']))
            
            closest_lon = lon_1d[lon_idx]
            closest_lat = lat_1d[lat_idx]
            
            logger.info(f"  Data point: x={row['x']:8.3f}, y={row['y']:7.3f}, prod={prod_val:.2e}")
            logger.info(f"  Maps to grid: lon_idx={lon_idx:4d}, lat_idx={lat_idx:4d}")
            logger.info(f"  Grid coords: x={closest_lon:8.3f}, y={closest_lat:7.3f}")
            logger.info(f"  Distance: dx={abs(row['x']-closest_lon):.6f}, dy={abs(row['y']-closest_lat):.6f}")
            logger.info("")
    
    # Check if coordinates are within expected ranges
    logger.info(f"\nCoordinate validation:")
    valid_lon = (production_kcal['x'] >= -180) & (production_kcal['x'] <= 180)
    valid_lat = (production_kcal['y'] >= -90) & (production_kcal['y'] <= 90)
    
    logger.info(f"Valid longitudes: {valid_lon.sum()}/{len(production_kcal)} ({100*valid_lon.mean():.1f}%)")
    logger.info(f"Valid latitudes: {valid_lat.sum()}/{len(production_kcal)} ({100*valid_lat.mean():.1f}%)")
    
    # Check for NaN coordinates
    nan_x = production_kcal['x'].isna().sum()
    nan_y = production_kcal['y'].isna().sum()
    logger.info(f"NaN coordinates: x={nan_x}, y={nan_y}")

def main():
    """Debug coordinate mapping issues."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Debugging Map Data Issues")
    logger.info("=" * 50)
    
    debug_coordinate_mapping()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick verification that maps now contain production data.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.visualization.maps import GlobalProductionMapper

def verify_map_has_data():
    """Verify that the mapping process produces visible data."""
    
    logger = logging.getLogger(__name__)
    
    # Test with wheat
    config = Config(crop_type='wheat', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    mapper = GlobalProductionMapper(config)
    
    # Load full dataset and get sample with production
    logger.info("Loading and sampling wheat production data...")
    full_df = loader.load_spam_production()
    crop_production = processor.filter_crop_data(full_df)
    production_kcal = processor.convert_production_to_kcal(crop_production)
    
    # Get cells with production
    crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
    total_production = production_kcal[crop_columns].sum(axis=1)
    has_production = total_production > 0
    production_sample = production_kcal[has_production].sample(n=1000, random_state=42)
    
    logger.info(f"Sample: {len(production_sample)} cells with production")
    
    # Test grid preparation
    lon_grid, lat_grid, production_grid = mapper._prepare_grid_data(production_sample)
    
    # Check results
    non_zero_cells = (production_grid > 0).sum()
    total_cells = production_grid.size
    
    logger.info(f"Grid results:")
    logger.info(f"  Total grid cells: {total_cells:,}")
    logger.info(f"  Cells with data: {non_zero_cells:,}")
    logger.info(f"  Data coverage: {100 * non_zero_cells / total_cells:.4f}%")
    
    if non_zero_cells > 0:
        logger.info(f"  Production range: {production_grid[production_grid > 0].min():.2f} - {production_grid.max():.2f} (log10)")
        logger.info("✅ SUCCESS: Grid contains production data!")
        
        # Show geographic distribution
        data_lats, data_lons = np.where(production_grid > 0)
        lat_range = lat_grid[data_lats, data_lons]
        lon_range = lon_grid[data_lats, data_lons]
        
        logger.info(f"  Geographic spread:")
        logger.info(f"    Latitude: {lat_range.min():.1f}° to {lat_range.max():.1f}°")
        logger.info(f"    Longitude: {lon_range.min():.1f}° to {lon_range.max():.1f}°")
        
        return True
    else:
        logger.error("❌ FAILED: No production data in grid!")
        return False

def main():
    """Verify map data generation."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Verifying Map Data Generation")
    logger.info("=" * 40)
    
    success = verify_map_has_data()
    
    if success:
        logger.info("\n🎉 Maps should now show production data!")
        logger.info("Check the generated PNG files in test_outputs/")
    else:
        logger.error("\n❌ Maps still have issues - need further debugging")

if __name__ == "__main__":
    main()
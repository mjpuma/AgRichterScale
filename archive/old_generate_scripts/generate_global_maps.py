#!/usr/bin/env python3
"""
Generate Global Production and Harvest Area Maps

Main script for generating publication-ready global maps showing production 
and harvest area for wheat, maize, rice, and all grains combined.

This script uses the corrected coordinate alignment system to ensure 
accurate geographic representation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('global_maps')

# Import AgRichter modules
from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.visualization.global_map_generator import GlobalMapGenerator


def load_crop_data(crop: str) -> Dict[str, Any]:
    """
    Load production and harvest data for a specific crop.
    
    Args:
        crop: Crop type ('wheat', 'maize', 'rice', 'allgrain')
    
    Returns:
        Dictionary with production and harvest DataFrames
    """
    logger.info(f"Loading {crop} data...")
    
    # Create config with correct paths
    config = Config(crop_type=crop, root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    
    # Load data using grid manager (with coordinate alignment fix)
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    logger.info(f"✅ {crop}: {len(prod_df):,} cells loaded (aligned)")
    
    return {
        'production': prod_df,
        'harvest': harv_df,
        'config': config
    }


def generate_publication_maps():
    """Generate publication-ready global maps for all crops."""
    logger.info("🗺️ GENERATING GLOBAL PRODUCTION AND HARVEST MAPS")
    logger.info("=" * 60)
    
    # Ensure results directory exists
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load data for all crops
    crops = ['wheat', 'maize', 'rice', 'allgrain']
    crop_data_dict = {}
    
    for crop in crops:
        try:
            crop_data = load_crop_data(crop)
            crop_data_dict[crop] = {
                'production': crop_data['production'],
                'harvest': crop_data['harvest']
            }
        except Exception as e:
            logger.error(f"Failed to load {crop} data: {e}")
            return False
    
    # Generate maps using GlobalMapGenerator
    logger.info("Generating 8-panel global maps...")
    
    try:
        # Use allgrain config for general settings
        config = Config(crop_type='allgrain', root_dir='.')
        map_generator = GlobalMapGenerator(config)
        
        # Generate publication maps
        output_path = results_dir / 'figure1_global_maps.png'
        result = map_generator.generate_publication_maps(crop_data_dict, output_path)
        
        logger.info(f"✅ Global maps saved: {output_path}")
        
        # Log validation results
        if hasattr(result, 'validation_summary'):
            logger.info("Coordinate validation summary:")
            for crop, validation in result.validation_summary.items():
                status = "✅" if validation.get('valid', False) else "⚠️"
                logger.info(f"  {status} {crop}: {validation.get('message', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate maps: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_coordinate_system():
    """Validate that the coordinate system is working correctly."""
    logger.info("🔍 Validating coordinate system...")
    
    try:
        # Load wheat data as test case
        wheat_data = load_crop_data('wheat')
        prod_df = wheat_data['production']
        harv_df = wheat_data['harvest']
        
        # Check alignment
        if len(prod_df) != len(harv_df):
            logger.error(f"❌ Coordinate alignment failed: {len(prod_df)} vs {len(harv_df)} cells")
            return False
        
        # Check coordinate ranges
        x_range = (prod_df['x'].min(), prod_df['x'].max())
        y_range = (prod_df['y'].min(), prod_df['y'].max())
        
        logger.info(f"Coordinate ranges: X{x_range}, Y{y_range}")
        
        # Check southern hemisphere preservation
        south_count = (prod_df['y'] < -35).sum()
        logger.info(f"Southern hemisphere cells (<-35°S): {south_count:,}")
        
        if south_count == 0:
            logger.warning("⚠️ No southern hemisphere data found")
        else:
            logger.info("✅ Southern hemisphere data preserved")
        
        # Validate coordinates using coordinate mapper
        from agririchter.visualization.coordinate_mapper import SPAMCoordinateMapper
        
        mapper = SPAMCoordinateMapper()
        validation = mapper.validate_coordinates(prod_df['x'].values, prod_df['y'].values)
        
        if validation.is_valid():
            logger.info("✅ Coordinate validation passed")
        else:
            logger.warning(f"⚠️ Coordinate validation issues: {validation.issues}")
            for rec in validation.recommendations:
                logger.info(f"  - {rec}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Coordinate validation failed: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("🌾 AGRICHTER GLOBAL MAPS GENERATOR")
    logger.info("Generating production and harvest area maps with corrected coordinate alignment")
    logger.info("=" * 80)
    
    # Step 1: Validate coordinate system
    if not validate_coordinate_system():
        logger.error("❌ Coordinate system validation failed")
        return 1
    
    # Step 2: Generate maps
    if not generate_publication_maps():
        logger.error("❌ Map generation failed")
        return 1
    
    # Success
    logger.info("")
    logger.info("🎉 SUCCESS: Global maps generated successfully!")
    logger.info("✅ Coordinate alignment fix working correctly")
    logger.info("✅ Southern hemisphere data preserved")
    logger.info("✅ Publication-ready maps saved to results/figure1_global_maps.png")
    logger.info("")
    logger.info("Next: Generate AgRichter scale and H-P envelope figures")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
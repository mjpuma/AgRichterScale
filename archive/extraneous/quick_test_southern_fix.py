#!/usr/bin/env python3
"""
Quick test of southern hemisphere fix
"""

import logging
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('quick_test')

def quick_test():
    """Quick test of southern hemisphere preservation."""
    logger.info("🧪 Quick Test: Southern Hemisphere Preservation")
    
    try:
        # Test with allgrain (most comprehensive)
        config = Config('allgrain', root_dir='.')
        config.spam_data_dir = Path('spam2020V2r0_global_production/spam2020V2r0_global_production')
        config.harvest_data_dir = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area')
        
        grid_manager = GridDataManager(config)
        production_df, harvest_df = grid_manager.load_spam_data()
        
        # Check southern hemisphere
        south_prod = (production_df['y'] < -35).sum()
        south_harv = (harvest_df['y'] < -35).sum()
        
        logger.info(f"Results:")
        logger.info(f"  Total cells: {len(production_df):,}")
        logger.info(f"  Southern production cells: {south_prod:,}")
        logger.info(f"  Southern harvest cells: {south_harv:,}")
        logger.info(f"  Latitude range: [{production_df['y'].min():.3f}°, {production_df['y'].max():.3f}°]")
        
        if south_prod > 0:
            logger.info("🎉 SUCCESS: Southern hemisphere data preserved!")
            return True
        else:
            logger.warning("❌ FAILED: No southern hemisphere data")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
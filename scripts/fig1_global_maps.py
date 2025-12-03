#!/usr/bin/env python3
"""
Figure 1: Global Production and Harvest Area Maps (8 panels)

Generates publication-ready global maps showing production and harvest area
for wheat, maize, rice, and all grains combined.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add parent directory to path to allow importing from agririchter
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figure1')

# Set journal-quality fonts (Nature style - Larger)
mpl.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.weight': 'normal',
    'axes.linewidth': 1.5,
})

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.visualization.global_map_generator import GlobalMapGenerator


def main():
    """Generate Figure 1: Global Maps."""
    logger.info("=" * 60)
    logger.info("FIGURE 1: Global Production and Harvest Area Maps")
    logger.info("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Load data for all crops
        crops = ['wheat', 'maize', 'rice', 'allgrain']
        crop_data = {}
        
        for crop in crops:
            logger.info(f"Loading {crop} data...")
            config = Config(crop_type=crop, root_dir='.')
            config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
            config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
            
            grid_manager = GridDataManager(config)
            prod_df, harv_df = grid_manager.load_spam_data()
            
            crop_data[crop] = {
                'production': prod_df,
                'harvest': harv_df
            }
            logger.info(f"  ✅ {crop}: {len(prod_df):,} cells loaded")
        
        # Generate maps
        logger.info("Generating 8-panel global maps...")
        config = Config(crop_type='allgrain', root_dir='.')
        map_generator = GlobalMapGenerator(config)
        
        output_path = results_dir / 'figure1_global_maps.png'
        map_generator.generate_publication_maps(crop_data, output_path)
        
        logger.info("")
        logger.info("🎉 SUCCESS!")
        logger.info(f"✅ Figure 1 saved: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

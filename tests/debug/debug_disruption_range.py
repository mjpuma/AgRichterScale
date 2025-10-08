#!/usr/bin/env python3
"""
Debug the disruption range vs actual data range mismatch.
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

def debug_disruption_range():
    """Debug disruption range vs actual data."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Debugging Disruption Range vs Data Range")
    
    # Initialize components
    config = Config(crop_type='allgrain', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    
    # Load and process data (larger sample)
    logger.info("Loading data...")
    production_df = loader.load_spam_production().head(10000)  # Larger sample
    harvest_df = loader.load_spam_harvest_area().head(10000)
    
    # Process data
    crop_production = processor.filter_crop_data(production_df)
    crop_harvest = processor.filter_crop_data(harvest_df)
    
    production_kcal = processor.convert_production_to_kcal(crop_production)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    # Find matching coordinates
    common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
        harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
    )
    
    logger.info(f"Found {len(common_coords)} matching coordinates")
    
    # Get matching data
    prod_matched = production_kcal[production_kcal['grid_code'].isin(common_coords['grid_code'])]
    harv_matched = harvest_km2[harvest_km2['grid_code'].isin(common_coords['grid_code'])]
    
    # Calculate total harvest area available
    crop_columns = [col for col in harv_matched.columns if col.endswith('_A')]
    grain_crops = ['BARL_A', 'MAIZ_A', 'OCER_A', 'PMIL_A', 'RICE_A', 'SORG_A', 'WHEA_A']
    crop_columns = [col for col in crop_columns if col in grain_crops]
    
    total_harvest = harv_matched[crop_columns].sum(axis=1).sum()
    max_harvest_per_cell = harv_matched[crop_columns].sum(axis=1).max()
    
    logger.info(f"\n=== DATA RANGE ANALYSIS ===")
    logger.info(f"Total harvest area in sample: {total_harvest:.1f} km²")
    logger.info(f"Max harvest per cell: {max_harvest_per_cell:.1f} km²")
    logger.info(f"Number of cells with harvest > 0: {(harv_matched[crop_columns].sum(axis=1) > 0).sum()}")
    
    # Check disruption range
    disruption_range = config.disruption_range
    logger.info(f"\n=== DISRUPTION RANGE ANALYSIS ===")
    logger.info(f"Disruption range: {len(disruption_range)} points")
    logger.info(f"Min disruption: {min(disruption_range)} km²")
    logger.info(f"Max disruption: {max(disruption_range)} km²")
    
    # Find how many disruption points are within our data range
    valid_disruptions = [d for d in disruption_range if d <= total_harvest]
    logger.info(f"Valid disruption points (≤ total harvest): {len(valid_disruptions)}")
    logger.info(f"Valid range: {min(valid_disruptions) if valid_disruptions else 'None'} - {max(valid_disruptions) if valid_disruptions else 'None'} km²")
    
    # Create a more appropriate disruption range for our sample
    if total_harvest > 0:
        # Create logarithmic range from 1 km² to total harvest area
        n_points = 50
        log_min = np.log10(1)
        log_max = np.log10(total_harvest)
        log_range = np.linspace(log_min, log_max, n_points)
        appropriate_range = 10 ** log_range
        
        logger.info(f"\n=== SUGGESTED DISRUPTION RANGE ===")
        logger.info(f"Appropriate range: {len(appropriate_range)} points")
        logger.info(f"Range: {appropriate_range[0]:.1f} - {appropriate_range[-1]:.1f} km²")
        
        return appropriate_range, total_harvest
    
    return None, total_harvest

def test_with_appropriate_range():
    """Test envelope calculation with appropriate disruption range."""
    
    logger = logging.getLogger(__name__)
    
    # Get appropriate range
    appropriate_range, total_harvest = debug_disruption_range()
    
    if appropriate_range is not None:
        logger.info(f"\n=== TESTING WITH APPROPRIATE RANGE ===")
        
        from agririchter.analysis.envelope import HPEnvelopeCalculator
        
        # Temporarily modify the config
        config = Config(crop_type='allgrain', root_dir='.')
        config.disruption_range = appropriate_range.tolist()
        
        # Load data
        loader = DataLoader(config)
        processor = DataProcessor(config)
        
        production_df = loader.load_spam_production().head(10000)
        harvest_df = loader.load_spam_harvest_area().head(10000)
        
        crop_production = processor.filter_crop_data(production_df)
        crop_harvest = processor.filter_crop_data(harvest_df)
        
        production_kcal = processor.convert_production_to_kcal(crop_production)
        harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
        
        # Find matching coordinates
        common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
            harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
        )
        
        prod_matched = production_kcal[production_kcal['grid_code'].isin(common_coords['grid_code'])]
        harv_matched = harvest_km2[harvest_km2['grid_code'].isin(common_coords['grid_code'])]
        
        # Calculate envelope with appropriate range
        envelope_calc = HPEnvelopeCalculator(config)
        envelope_data = envelope_calc.calculate_hp_envelope(prod_matched, harv_matched)
        
        logger.info(f"Envelope points with appropriate range: {len(envelope_data['disruption_areas'])}")
        
        # Test visualization
        from agririchter.visualization.plots import EnvelopePlotter
        
        plotter = EnvelopePlotter(config)
        fig = plotter.create_envelope_plot(
            envelope_data=envelope_data,
            title="Envelope with Appropriate Disruption Range",
            use_publication_style=False
        )
        
        # Save test
        output_path = Path('test_outputs/appropriate_range_envelope.png')
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Appropriate range envelope saved: {output_path}")
        
        # Check patches
        axes = fig.get_axes()[0]
        patches = [p for p in axes.patches if hasattr(p, 'get_facecolor')]
        logger.info(f"Number of patches: {len(patches)}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)

if __name__ == "__main__":
    test_with_appropriate_range()
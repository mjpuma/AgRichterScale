#!/usr/bin/env python3
"""
Debug why the envelope fill is not showing up.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.analysis.envelope import HPEnvelopeCalculator
from agririchter.visualization.plots import EnvelopePlotter

def debug_envelope_fill():
    """Debug why envelope fill is not showing."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Debugging Envelope Fill Issue")
    
    # Initialize components
    config = Config(crop_type='allgrain', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    analyzer = HPEnvelopeCalculator(config)
    
    # Load and process data (small sample)
    logger.info("Loading data...")
    production_df = loader.load_spam_production().head(3000)
    harvest_df = loader.load_spam_harvest_area().head(3000)
    
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
    sample_coords = common_coords.sample(n=min(3000, len(common_coords)), random_state=42)
    prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_coords['grid_code'])]
    harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_coords['grid_code'])]
    
    # Calculate envelope
    logger.info("Calculating envelope...")
    envelope_data = analyzer.calculate_hp_envelope(prod_matched, harv_matched)
    
    # Debug envelope data
    logger.info(f"\n=== ENVELOPE DATA DEBUG ===")
    logger.info(f"Disruption areas: {len(envelope_data['disruption_areas'])} points")
    
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    logger.info(f"Lower bound harvest range: {lower_harvest.min():.1f} - {lower_harvest.max():.1f} km²")
    logger.info(f"Lower bound production range: {lower_production.min():.2e} - {lower_production.max():.2e} kcal")
    logger.info(f"Upper bound harvest range: {upper_harvest.min():.1f} - {upper_harvest.max():.1f} km²")
    logger.info(f"Upper bound production range: {upper_production.min():.2e} - {upper_production.max():.2e} kcal")
    
    # Check for valid data
    valid_mask = (
        np.isfinite(lower_harvest) & np.isfinite(lower_production) &
        np.isfinite(upper_harvest) & np.isfinite(upper_production) &
        (lower_harvest > 0) & (lower_production > 0) &
        (upper_harvest > 0) & (upper_production > 0)
    )
    
    logger.info(f"Valid data points: {np.sum(valid_mask)}/{len(valid_mask)}")
    
    if np.sum(valid_mask) < 3:
        logger.error("❌ Not enough valid points for polygon fill!")
        logger.info("Need at least 3 points to create a filled polygon")
        return
    
    # Test manual polygon creation
    logger.info("\n=== MANUAL POLYGON TEST ===")
    
    # Get valid data
    lower_harvest_valid = lower_harvest[valid_mask]
    lower_production_valid = lower_production[valid_mask]
    upper_harvest_valid = upper_harvest[valid_mask]
    upper_production_valid = upper_production[valid_mask]
    
    logger.info(f"Valid points after filtering: {len(lower_harvest_valid)}")
    
    if len(lower_harvest_valid) >= 3:
        # Create polygon manually
        X_polygon = np.concatenate([
            [lower_harvest_valid[0]],
            upper_harvest_valid,
            np.flipud(lower_harvest_valid)
        ])
        
        Y_polygon = np.concatenate([
            [lower_production_valid[0]],
            upper_production_valid,
            np.flipud(lower_production_valid)
        ])
        
        logger.info(f"Polygon X range: {X_polygon.min():.1f} - {X_polygon.max():.1f}")
        logger.info(f"Polygon Y range: {Y_polygon.min():.2e} - {Y_polygon.max():.2e}")
        
        # Convert to log10
        X_polygon_log = np.log10(X_polygon)
        Y_polygon_log = np.log10(Y_polygon)
        
        logger.info(f"Log10 X range: {X_polygon_log.min():.2f} - {X_polygon_log.max():.2f}")
        logger.info(f"Log10 Y range: {Y_polygon_log.min():.2f} - {Y_polygon_log.max():.2f}")
        
        # Check for valid log values
        valid_log_mask = np.isfinite(X_polygon_log) & np.isfinite(Y_polygon_log)
        logger.info(f"Valid log points: {np.sum(valid_log_mask)}/{len(valid_log_mask)}")
        
        if np.sum(valid_log_mask) >= 3:
            # Create test plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot the fill
            light_blue_color = [0.8, 0.9, 1.0]
            ax.fill(X_polygon_log[valid_log_mask], Y_polygon_log[valid_log_mask], 
                   color=light_blue_color, alpha=0.6, edgecolor='none',
                   label='H-P Envelope')
            
            # Plot boundary lines
            log_lower_harvest = np.log10(lower_harvest_valid)
            log_lower_production = np.log10(lower_production_valid)
            log_upper_harvest = np.log10(upper_harvest_valid)
            log_upper_production = np.log10(upper_production_valid)
            
            ax.plot(log_lower_harvest, log_lower_production, 'b-', linewidth=2, 
                   alpha=0.8, label='Lower Bound')
            ax.plot(log_upper_harvest, log_upper_production, 'r-', linewidth=2,
                   alpha=0.8, label='Upper Bound')
            
            ax.set_xlabel('Log₁₀ Harvest Area (km²)')
            ax.set_ylabel('Log₁₀ Production (kcal)')
            ax.set_title('Manual Envelope Fill Test')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save test plot
            output_path = Path('test_outputs/manual_envelope_fill_test.png')
            output_path.parent.mkdir(exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Manual fill test saved: {output_path}")
            
            plt.close(fig)
            
            # Now test with the actual plotter
            logger.info("\n=== TESTING ACTUAL PLOTTER ===")
            plotter = EnvelopePlotter(config)
            
            fig = plotter.create_envelope_plot(
                envelope_data=envelope_data,
                title="Actual Plotter Test",
                use_publication_style=False
            )
            
            # Check if fill patch exists
            axes = fig.get_axes()[0]
            patches = [p for p in axes.patches if hasattr(p, 'get_facecolor')]
            
            logger.info(f"Number of patches found: {len(patches)}")
            
            for i, patch in enumerate(patches):
                face_color = patch.get_facecolor()
                alpha = patch.get_alpha()
                logger.info(f"Patch {i}: color={face_color}, alpha={alpha}")
            
            # Save actual plotter test
            output_path = Path('test_outputs/actual_plotter_test.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Actual plotter test saved: {output_path}")
            
            plt.close(fig)
        else:
            logger.error("❌ Not enough valid log points for polygon")
    else:
        logger.error("❌ Not enough valid points after filtering")

if __name__ == "__main__":
    debug_envelope_fill()
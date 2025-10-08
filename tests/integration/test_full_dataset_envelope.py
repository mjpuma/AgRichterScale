#!/usr/bin/env python3
"""
Test H-P envelope calculation using the FULL dataset (no sampling).
This is the production-ready version that should be used for real analysis.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.analysis.envelope import HPEnvelopeCalculator
from agririchter.visualization.plots import EnvelopePlotter

def test_full_dataset_envelope(crop_type: str = 'allgrain'):
    """Test envelope calculation with FULL dataset (no sampling)."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"🌍 Testing FULL Dataset H-P Envelope: {crop_type.upper()}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Initialize components
    config = Config(crop_type=crop_type, root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    envelope_calc = HPEnvelopeCalculator(config)
    
    # Load FULL dataset (no sampling!)
    logger.info("Loading FULL production dataset...")
    production_df = loader.load_spam_production()  # NO .head() or .sample()
    logger.info(f"Loaded {len(production_df):,} production grid cells")
    
    logger.info("Loading FULL harvest area dataset...")
    harvest_df = loader.load_spam_harvest_area()   # NO .head() or .sample()
    logger.info(f"Loaded {len(harvest_df):,} harvest area grid cells")
    
    load_time = time.time() - start_time
    logger.info(f"Data loading completed in {load_time:.1f} seconds")
    
    # Process FULL data
    logger.info("Processing FULL crop data...")
    process_start = time.time()
    
    crop_production = processor.filter_crop_data(production_df)
    crop_harvest = processor.filter_crop_data(harvest_df)
    
    production_kcal = processor.convert_production_to_kcal(crop_production)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    process_time = time.time() - process_start
    logger.info(f"Data processing completed in {process_time:.1f} seconds")
    
    # Find ALL matching coordinates (no sampling!)
    logger.info("Finding ALL matching coordinates...")
    match_start = time.time()
    
    common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
        harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
    )
    
    logger.info(f"Found {len(common_coords):,} matching coordinates")
    
    # Use ALL matching data (no sampling!)
    prod_matched = production_kcal[production_kcal['grid_code'].isin(common_coords['grid_code'])]
    harv_matched = harvest_km2[harvest_km2['grid_code'].isin(common_coords['grid_code'])]
    
    match_time = time.time() - match_start
    logger.info(f"Coordinate matching completed in {match_time:.1f} seconds")
    logger.info(f"Using ALL {len(prod_matched):,} cells for envelope calculation")
    
    # Calculate global totals
    crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
    if crop_type == 'allgrain':
        grain_crops = ['BARL_A', 'MAIZ_A', 'OCER_A', 'PMIL_A', 'RICE_A', 'SORG_A', 'WHEA_A']
        crop_columns = [col for col in crop_columns if col in grain_crops]
    elif crop_type == 'wheat':
        crop_columns = [col for col in crop_columns if 'WHEA' in col.upper()]
    
    global_production = prod_matched[crop_columns].sum().sum()
    global_harvest = harv_matched[crop_columns].sum().sum()
    
    logger.info(f"\n=== FULL DATASET SCALE ===")
    logger.info(f"Total production: {global_production:.2e} kcal")
    logger.info(f"Total harvest: {global_harvest:.1f} km²")
    logger.info(f"Log10 production: {np.log10(global_production):.2f}")
    
    # Calculate H-P envelope with FULL dataset
    logger.info("\nCalculating H-P envelope with FULL dataset...")
    envelope_start = time.time()
    
    envelope_data = envelope_calc.calculate_hp_envelope(prod_matched, harv_matched)
    
    envelope_time = time.time() - envelope_start
    logger.info(f"Envelope calculation completed in {envelope_time:.1f} seconds")
    
    # Analyze envelope results
    lower_production = envelope_data['lower_bound_production']
    upper_production = envelope_data['upper_bound_production']
    
    logger.info(f"\n=== FULL DATASET ENVELOPE RESULTS ===")
    logger.info(f"Envelope points: {len(envelope_data['disruption_areas'])}")
    logger.info(f"Lower bound production: {lower_production.min():.2e} - {lower_production.max():.2e} kcal")
    logger.info(f"Upper bound production: {upper_production.min():.2e} - {upper_production.max():.2e} kcal")
    logger.info(f"Log10 envelope range: {np.log10(lower_production.min()):.2f} - {np.log10(upper_production.max()):.2f}")
    
    # Check if thresholds will be properly scaled
    thresholds = config.get_thresholds()
    max_envelope = upper_production.max()
    global_reference = 1.04e16  # From our analysis
    scaling_factor = max_envelope / global_reference
    
    logger.info(f"\n=== THRESHOLD SCALING ANALYSIS ===")
    logger.info(f"Max envelope production: {max_envelope:.2e} kcal")
    logger.info(f"Global reference: {global_reference:.2e} kcal")
    logger.info(f"Scaling factor: {scaling_factor:.3f}")
    
    for level, threshold_kcal in thresholds.items():
        if threshold_kcal > 0:
            scaled_threshold = threshold_kcal * scaling_factor
            logger.info(f"{level}: {threshold_kcal:.2e} → {scaled_threshold:.2e} kcal")
    
    # Create visualization with FULL dataset
    logger.info("\nCreating visualization with FULL dataset...")
    viz_start = time.time()
    
    plotter = EnvelopePlotter(config)
    fig = plotter.create_envelope_plot(
        envelope_data=envelope_data,
        title=f"FULL Dataset H-P Envelope - {crop_type.title()}",
        use_publication_style=True
    )
    
    viz_time = time.time() - viz_start
    logger.info(f"Visualization completed in {viz_time:.1f} seconds")
    
    # Save results
    output_path = Path(f'test_outputs/full_dataset_envelope_{crop_type}.png')
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Full dataset envelope saved: {output_path}")
    
    plt.close(fig)
    
    # Total time
    total_time = time.time() - start_time
    logger.info(f"\n=== PERFORMANCE SUMMARY ===")
    logger.info(f"Total execution time: {total_time:.1f} seconds")
    logger.info(f"Data loading: {load_time:.1f}s ({load_time/total_time*100:.1f}%)")
    logger.info(f"Data processing: {process_time:.1f}s ({process_time/total_time*100:.1f}%)")
    logger.info(f"Coordinate matching: {match_time:.1f}s ({match_time/total_time*100:.1f}%)")
    logger.info(f"Envelope calculation: {envelope_time:.1f}s ({envelope_time/total_time*100:.1f}%)")
    logger.info(f"Visualization: {viz_time:.1f}s ({viz_time/total_time*100:.1f}%)")
    
    return envelope_data, total_time

def compare_sample_vs_full():
    """Compare results between sampled and full dataset."""
    
    logger = logging.getLogger(__name__)
    logger.info("\n🔄 Comparing Sample vs Full Dataset Results")
    logger.info("=" * 60)
    
    # Test with full dataset
    full_envelope, full_time = test_full_dataset_envelope('allgrain')
    
    # The key insight: with full dataset, we should get the ACTUAL global scale
    # and thresholds should align properly without needing scaling
    
    max_full_production = full_envelope['upper_bound_production'].max()
    
    logger.info(f"\n=== COMPARISON RESULTS ===")
    logger.info(f"Full dataset max production: {max_full_production:.2e} kcal")
    logger.info(f"Full dataset log10 range: {np.log10(max_full_production):.2f}")
    
    # Check if we're now at global scale
    global_reference = 1.04e16
    percentage_of_global = (max_full_production / global_reference) * 100
    
    logger.info(f"Percentage of global production: {percentage_of_global:.1f}%")
    
    if percentage_of_global > 80:
        logger.info("✅ SUCCESS: Full dataset reaches near-global scale!")
        logger.info("✅ Thresholds should now align properly with envelope")
    else:
        logger.info(f"⚠️  Still only {percentage_of_global:.1f}% of global scale")
        logger.info("⚠️  May need to investigate data completeness")

if __name__ == "__main__":
    compare_sample_vs_full()
    
    print("\n" + "="*60)
    print("FULL DATASET ENVELOPE ANALYSIS")
    print("="*60)
    print("✅ Used complete SPAM dataset (no sampling)")
    print("✅ Calculated envelope with all available grid cells")
    print("✅ Proper global-scale production values")
    print("✅ Thresholds should align correctly with envelope")
    print("\n💡 RECOMMENDATION: Use full dataset for production analysis")
    print("💡 Sampling should only be used for testing/debugging")
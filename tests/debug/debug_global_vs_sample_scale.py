#!/usr/bin/env python3
"""
Debug the scale difference between global thresholds and sample envelope.
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
from agririchter.analysis.envelope import HPEnvelopeCalculator

def debug_global_vs_sample_scale():
    """Debug scale difference between global thresholds and sample envelope."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🌍 Debugging Global vs Sample Scale Issue")
    logger.info("=" * 60)
    
    config = Config(crop_type='allgrain', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    
    # Load FULL dataset to understand global scale
    logger.info("Loading FULL dataset...")
    production_df = loader.load_spam_production()  # Full dataset
    harvest_df = loader.load_spam_harvest_area()   # Full dataset
    
    # Process full data
    crop_production = processor.filter_crop_data(production_df)
    crop_harvest = processor.filter_crop_data(harvest_df)
    
    production_kcal = processor.convert_production_to_kcal(crop_production)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    # Calculate global totals
    crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
    grain_crops = ['BARL_A', 'MAIZ_A', 'OCER_A', 'PMIL_A', 'RICE_A', 'SORG_A', 'WHEA_A']
    crop_columns = [col for col in crop_columns if col in grain_crops]
    
    global_production = production_kcal[crop_columns].sum().sum()
    global_harvest = harvest_km2[crop_columns].sum().sum()
    
    logger.info(f"\n=== GLOBAL SCALE ===")
    logger.info(f"Global production: {global_production:.2e} kcal")
    logger.info(f"Global harvest: {global_harvest:.1f} km²")
    logger.info(f"Log10 global production: {np.log10(global_production):.2f}")
    
    # Check thresholds against global production
    thresholds = config.get_thresholds()
    logger.info(f"\n=== THRESHOLDS vs GLOBAL PRODUCTION ===")
    for level, threshold_kcal in thresholds.items():
        if threshold_kcal > 0:
            percentage = (threshold_kcal / global_production) * 100
            log_threshold = np.log10(threshold_kcal)
            logger.info(f"{level}: {threshold_kcal:.2e} kcal (log10: {log_threshold:.2f}) = {percentage:.1f}% of global production")
    
    # Now test with different sample sizes
    sample_sizes = [1000, 5000, 10000, 50000]
    
    for sample_size in sample_sizes:
        logger.info(f"\n=== SAMPLE SIZE: {sample_size} ===")
        
        # Find matching coordinates
        common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
            harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
        )
        
        if len(common_coords) < sample_size:
            sample_size = len(common_coords)
            logger.info(f"Adjusted sample size to {sample_size} (max available)")
        
        sample_coords = common_coords.sample(n=sample_size, random_state=42)
        prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_coords['grid_code'])]
        harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_coords['grid_code'])]
        
        # Calculate sample totals
        sample_production = prod_matched[crop_columns].sum().sum()
        sample_harvest = harv_matched[crop_columns].sum().sum()
        
        logger.info(f"Sample production: {sample_production:.2e} kcal ({sample_production/global_production*100:.1f}% of global)")
        logger.info(f"Sample harvest: {sample_harvest:.1f} km² ({sample_harvest/global_harvest*100:.1f}% of global)")
        
        # Calculate envelope for this sample
        envelope_calc = HPEnvelopeCalculator(config)
        envelope_data = envelope_calc.calculate_hp_envelope(prod_matched, harv_matched)
        
        upper_production = envelope_data['upper_bound_production']
        max_envelope = upper_production.max()
        
        logger.info(f"Max envelope production: {max_envelope:.2e} kcal (log10: {np.log10(max_envelope):.2f})")
        
        # Calculate scaling factor needed
        t1_threshold = thresholds['T1']
        scaling_factor = t1_threshold / max_envelope
        
        logger.info(f"Scaling factor to reach T1: {scaling_factor:.1f}x")
        
        if sample_size >= 10000:
            break  # Don't test larger samples to save time
    
    # Suggest solution
    logger.info(f"\n=== SOLUTION ANALYSIS ===")
    logger.info("The thresholds are designed for GLOBAL production scales.")
    logger.info("Our envelope calculation uses SAMPLE data (subset of global).")
    logger.info("Options:")
    logger.info("1. Scale thresholds down proportionally to sample size")
    logger.info("2. Scale envelope up to represent global equivalent")
    logger.info("3. Use full global dataset for envelope calculation")
    logger.info("4. Adjust threshold values to be sample-appropriate")
    
    # Calculate proportional thresholds for typical sample
    sample_fraction = 0.01  # 1% of global (typical for our samples)
    logger.info(f"\nProportional thresholds for {sample_fraction*100:.0f}% sample:")
    for level, threshold_kcal in thresholds.items():
        if threshold_kcal > 0:
            proportional_threshold = threshold_kcal * sample_fraction
            log_prop = np.log10(proportional_threshold)
            logger.info(f"{level}: {proportional_threshold:.2e} kcal (log10: {log_prop:.2f})")

if __name__ == "__main__":
    debug_global_vs_sample_scale()
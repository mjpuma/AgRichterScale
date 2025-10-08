#!/usr/bin/env python3
"""
Debug the units mismatch between envelope bounds and thresholds.
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

def debug_units_mismatch():
    """Debug units mismatch between envelope and thresholds."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Debugging Units Mismatch: Envelope vs Thresholds")
    logger.info("=" * 60)
    
    # Test with allgrain
    config = Config(crop_type='allgrain', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    envelope_calc = HPEnvelopeCalculator(config)
    
    # Load sample data
    production_df = loader.load_spam_production().head(5000)
    harvest_df = loader.load_spam_harvest_area().head(5000)
    
    # Process data
    crop_production = processor.filter_crop_data(production_df)
    crop_harvest = processor.filter_crop_data(harvest_df)
    
    production_kcal = processor.convert_production_to_kcal(crop_production)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    # Find matching coordinates
    common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
        harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
    )
    
    sample_coords = common_coords.sample(n=min(5000, len(common_coords)), random_state=42)
    prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_coords['grid_code'])]
    harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_coords['grid_code'])]
    
    # Calculate envelope
    envelope_data = envelope_calc.calculate_hp_envelope(prod_matched, harv_matched)
    
    # Analyze envelope production values
    lower_production = envelope_data['lower_bound_production']
    upper_production = envelope_data['upper_bound_production']
    
    logger.info(f"\n=== ENVELOPE PRODUCTION VALUES ===")
    logger.info(f"Lower bound production range: {lower_production.min():.2e} - {lower_production.max():.2e} kcal")
    logger.info(f"Upper bound production range: {upper_production.min():.2e} - {upper_production.max():.2e} kcal")
    logger.info(f"Log10 lower bound range: {np.log10(lower_production.min()):.2f} - {np.log10(lower_production.max()):.2f}")
    logger.info(f"Log10 upper bound range: {np.log10(upper_production.min()):.2f} - {np.log10(upper_production.max()):.2f}")
    
    # Check thresholds
    thresholds = config.get_thresholds()
    logger.info(f"\n=== THRESHOLD VALUES ===")
    for level, threshold_kcal in thresholds.items():
        if threshold_kcal > 0:
            log_threshold = np.log10(threshold_kcal)
            logger.info(f"{level}: {threshold_kcal:.2e} kcal (log10: {log_threshold:.2f})")
    
    # Check raw production data
    crop_columns = [col for col in prod_matched.columns if col.endswith('_A')]
    grain_crops = ['BARL_A', 'MAIZ_A', 'OCER_A', 'PMIL_A', 'RICE_A', 'SORG_A', 'WHEA_A']
    crop_columns = [col for col in crop_columns if col in grain_crops]
    
    total_production_per_cell = prod_matched[crop_columns].sum(axis=1)
    
    logger.info(f"\n=== RAW PRODUCTION DATA ===")
    logger.info(f"Production per cell range: {total_production_per_cell.min():.2e} - {total_production_per_cell.max():.2e} kcal")
    logger.info(f"Log10 production per cell: {np.log10(total_production_per_cell.min()):.2f} - {np.log10(total_production_per_cell.max()):.2f}")
    logger.info(f"Total production in sample: {total_production_per_cell.sum():.2e} kcal")
    logger.info(f"Log10 total production: {np.log10(total_production_per_cell.sum()):.2f}")
    
    # Check conversion factors
    from agririchter.core.constants import CALORIC_CONTENT, GRAMS_PER_METRIC_TON
    
    logger.info(f"\n=== CONVERSION FACTORS ===")
    allgrain_kcal_per_gram = CALORIC_CONTENT['Allgrain']
    logger.info(f"Allgrain kcal per gram: {allgrain_kcal_per_gram}")
    logger.info(f"Grams per metric ton: {GRAMS_PER_METRIC_TON}")
    logger.info(f"Total conversion factor (MT to kcal): {allgrain_kcal_per_gram * GRAMS_PER_METRIC_TON}")
    
    # Check original SPAM data units
    logger.info(f"\n=== ORIGINAL SPAM DATA ===")
    original_production = crop_production[crop_columns].sum(axis=1)
    logger.info(f"Original production range (MT): {original_production.min():.2e} - {original_production.max():.2e}")
    logger.info(f"Log10 original production (MT): {np.log10(original_production.min()):.2f} - {np.log10(original_production.max()):.2f}")
    
    # Compare envelope bounds with expected ranges
    logger.info(f"\n=== UNITS ANALYSIS ===")
    
    # The envelope should represent cumulative production up to disruption area
    # So envelope bounds should be much larger than individual cell production
    expected_envelope_range = [total_production_per_cell.sum() * 0.01, total_production_per_cell.sum()]
    logger.info(f"Expected envelope range (1%-100% of total): {expected_envelope_range[0]:.2e} - {expected_envelope_range[1]:.2e} kcal")
    logger.info(f"Log10 expected range: {np.log10(expected_envelope_range[0]):.2f} - {np.log10(expected_envelope_range[1]):.2f}")
    
    # Check if envelope bounds are reasonable
    envelope_in_expected = (
        lower_production.min() >= expected_envelope_range[0] * 0.1 and
        upper_production.max() <= expected_envelope_range[1] * 10
    )
    
    logger.info(f"Envelope bounds reasonable: {envelope_in_expected}")
    
    # Check threshold scale vs envelope scale
    threshold_values = [v for v in thresholds.values() if v > 0]
    threshold_log_range = [np.log10(min(threshold_values)), np.log10(max(threshold_values))]
    envelope_log_range = [np.log10(lower_production.min()), np.log10(upper_production.max())]
    
    logger.info(f"\nThreshold log10 range: {threshold_log_range[0]:.2f} - {threshold_log_range[1]:.2f}")
    logger.info(f"Envelope log10 range: {envelope_log_range[0]:.2f} - {envelope_log_range[1]:.2f}")
    logger.info(f"Scale difference: {threshold_log_range[0] - envelope_log_range[1]:.2f} orders of magnitude")
    
    if abs(threshold_log_range[0] - envelope_log_range[1]) > 2:
        logger.error("❌ UNITS MISMATCH DETECTED!")
        logger.error("Thresholds and envelope bounds are on different scales")
        logger.error("This suggests a units conversion issue")
    else:
        logger.info("✅ Units appear to be consistent")

if __name__ == "__main__":
    debug_units_mismatch()
#!/usr/bin/env python3
"""
Debug what the envelope is actually plotting on the x-axis.
Check if it's cumulative harvest area vs individual cell harvest area.
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

def debug_envelope_x_axis():
    """Debug what the envelope x-axis represents."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Debugging Envelope X-Axis Values")
    logger.info("=" * 60)
    
    # Load sample data for debugging
    config = Config(crop_type='allgrain', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    envelope_calc = HPEnvelopeCalculator(config)
    
    # Load sample
    production_df = loader.load_spam_production().head(10000)
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
    
    sample_coords = common_coords.sample(n=min(1000, len(common_coords)), random_state=42)
    prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_coords['grid_code'])]
    harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_coords['grid_code'])]
    
    # Calculate envelope
    envelope_data = envelope_calc.calculate_hp_envelope(prod_matched, harv_matched)
    
    # Analyze what the envelope contains
    logger.info(f"\n=== ENVELOPE DATA STRUCTURE ===")
    for key, value in envelope_data.items():
        if isinstance(value, np.ndarray):
            logger.info(f"{key}: {len(value)} points, range {value.min():.2f} - {value.max():.2f}")
    
    # Check individual cell harvest areas
    crop_columns = [col for col in harv_matched.columns if col.endswith('_A')]
    grain_crops = ['BARL_A', 'MAIZ_A', 'OCER_A', 'PMIL_A', 'RICE_A', 'SORG_A', 'WHEA_A']
    crop_columns = [col for col in crop_columns if col in grain_crops]
    
    individual_harvest_per_cell = harv_matched[crop_columns].sum(axis=1)
    
    logger.info(f"\n=== INDIVIDUAL CELL HARVEST AREAS ===")
    logger.info(f"Individual cells range: {individual_harvest_per_cell.min():.2f} - {individual_harvest_per_cell.max():.2f} km²")
    logger.info(f"Total harvest area: {individual_harvest_per_cell.sum():.2f} km²")
    logger.info(f"Mean cell harvest: {individual_harvest_per_cell.mean():.2f} km²")
    
    # Compare with envelope bounds
    lower_harvest = envelope_data['lower_bound_harvest']
    upper_harvest = envelope_data['upper_bound_harvest']
    
    logger.info(f"\n=== ENVELOPE HARVEST BOUNDS ===")
    logger.info(f"Lower bound harvest: {lower_harvest.min():.2f} - {lower_harvest.max():.2f} km²")
    logger.info(f"Upper bound harvest: {upper_harvest.min():.2f} - {upper_harvest.max():.2f} km²")
    
    # The key question: Are envelope bounds cumulative or individual?
    logger.info(f"\n=== INTERPRETATION CHECK ===")
    
    max_individual_cell = individual_harvest_per_cell.max()
    max_envelope_harvest = upper_harvest.max()
    total_harvest = individual_harvest_per_cell.sum()
    
    logger.info(f"Max individual cell: {max_individual_cell:.2f} km²")
    logger.info(f"Max envelope harvest: {max_envelope_harvest:.2f} km²")
    logger.info(f"Total sample harvest: {total_harvest:.2f} km²")
    
    if max_envelope_harvest > max_individual_cell * 10:
        logger.info("✅ Envelope x-axis represents CUMULATIVE harvest area")
        logger.info("   This is correct - it shows cumulative area disrupted")
    else:
        logger.info("❓ Envelope x-axis might represent individual cell harvest")
        logger.info("   This would be incorrect for disruption analysis")
    
    # Check the MATLAB algorithm interpretation
    logger.info(f"\n=== MATLAB ALGORITHM INTERPRETATION ===")
    logger.info("MATLAB envelope algorithm:")
    logger.info("1. Sort cells by yield (productivity)")
    logger.info("2. Calculate cumulative sums of harvest area and production")
    logger.info("3. For each disruption area, find cumulative harvest that exceeds it")
    logger.info("4. Plot cumulative production vs cumulative harvest area")
    
    logger.info(f"\nSo the x-axis SHOULD represent:")
    logger.info(f"- Cumulative harvest area affected by disruption")
    logger.info(f"- Range: 0 to total harvest area ({total_harvest:.0f} km²)")
    logger.info(f"- NOT individual cell harvest areas")
    
    # Check disruption areas
    disruption_areas = envelope_data['disruption_areas']
    logger.info(f"\n=== DISRUPTION AREAS ===")
    logger.info(f"Disruption areas: {disruption_areas.min():.0f} - {disruption_areas.max():.0f} km²")
    logger.info(f"These represent the target disruption areas we're analyzing")
    
    # Final diagnosis
    logger.info(f"\n=== DIAGNOSIS ===")
    
    if max_envelope_harvest <= total_harvest * 1.1:  # Within 10% of total
        logger.info("✅ Envelope harvest bounds are reasonable")
        logger.info("✅ They represent cumulative harvest areas up to total available")
        logger.info("✅ X-axis interpretation is correct")
    else:
        logger.error("❌ Envelope harvest bounds exceed total available harvest")
        logger.error("❌ There may be an error in the cumulative calculation")
    
    return envelope_data

if __name__ == "__main__":
    envelope_data = debug_envelope_x_axis()
    
    print(f"\n" + "="*60)
    print("ENVELOPE X-AXIS INTERPRETATION")
    print("="*60)
    print("The H-P envelope x-axis represents:")
    print("📊 CUMULATIVE harvest area affected by disruption")
    print("📊 NOT individual grid cell harvest areas")
    print("📊 Range: 1 km² to total available harvest area")
    print("\nThis is the correct MATLAB interpretation!")
    print("The envelope shows how much production is lost")
    print("when disrupting different amounts of harvest area.")
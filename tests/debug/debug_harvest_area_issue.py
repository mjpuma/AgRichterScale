#!/usr/bin/env python3
"""
Debug the harvest area values - they seem way too small.
Check if it's a conversion issue or sum vs average problem.
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
from agririchter.core.constants import HECTARES_TO_KM2

def debug_harvest_area_values():
    """Debug harvest area values to find the issue."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Debugging Harvest Area Values")
    logger.info("=" * 60)
    
    # Load data
    config = Config(crop_type='allgrain', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    
    # Load sample for debugging
    logger.info("Loading harvest area data sample...")
    harvest_df = loader.load_spam_harvest_area().head(10000)
    
    # Check original SPAM data
    logger.info(f"\n=== ORIGINAL SPAM HARVEST DATA ===")
    logger.info(f"Dataset shape: {harvest_df.shape}")
    logger.info(f"Columns: {list(harvest_df.columns)}")
    
    # Find crop columns
    crop_columns = [col for col in harvest_df.columns if col.endswith('_A')]
    grain_crops = ['BARL_A', 'MAIZ_A', 'OCER_A', 'PMIL_A', 'RICE_A', 'SORG_A', 'WHEA_A']
    available_grain_crops = [col for col in crop_columns if col in grain_crops]
    
    logger.info(f"Available grain crop columns: {available_grain_crops}")
    
    # Check original values (should be in hectares)
    for crop in available_grain_crops[:3]:  # Check first 3 crops
        values = harvest_df[crop]
        non_zero = values[values > 0]
        
        logger.info(f"\n{crop} (original hectares):")
        logger.info(f"  Non-zero values: {len(non_zero):,}/{len(values):,}")
        logger.info(f"  Range: {values.min():.2f} - {values.max():.2f} ha")
        logger.info(f"  Mean (non-zero): {non_zero.mean():.2f} ha")
        logger.info(f"  Total: {values.sum():.0f} ha")
    
    # Check conversion factor
    logger.info(f"\n=== CONVERSION FACTOR ===")
    logger.info(f"HECTARES_TO_KM2 constant: {HECTARES_TO_KM2}")
    logger.info(f"Expected: 0.01 (since 1 km² = 100 ha)")
    
    # Test conversion
    crop_harvest = processor.filter_crop_data(harvest_df)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    logger.info(f"\n=== AFTER CONVERSION TO KM² ===")
    for crop in available_grain_crops[:3]:
        original_ha = harvest_df[crop]
        converted_km2 = harvest_km2[crop]
        
        logger.info(f"\n{crop}:")
        logger.info(f"  Original total (ha): {original_ha.sum():.0f}")
        logger.info(f"  Converted total (km²): {converted_km2.sum():.2f}")
        logger.info(f"  Conversion check: {original_ha.sum() * HECTARES_TO_KM2:.2f} km²")
        logger.info(f"  Max cell (km²): {converted_km2.max():.2f}")
    
    # Check total across all grains
    total_original_ha = harvest_df[available_grain_crops].sum().sum()
    total_converted_km2 = harvest_km2[available_grain_crops].sum().sum()
    
    logger.info(f"\n=== TOTAL GRAIN HARVEST ===")
    logger.info(f"Total original (ha): {total_original_ha:,.0f}")
    logger.info(f"Total converted (km²): {total_converted_km2:,.2f}")
    logger.info(f"Manual conversion check: {total_original_ha * HECTARES_TO_KM2:,.2f} km²")
    
    # Compare with expected global values
    logger.info(f"\n=== GLOBAL SCALE COMPARISON ===")
    logger.info("Expected global grain harvest area (approximate):")
    logger.info("  Wheat: ~220 million ha = 2.2 million km²")
    logger.info("  Rice: ~160 million ha = 1.6 million km²") 
    logger.info("  Maize: ~190 million ha = 1.9 million km²")
    logger.info("  Total grains: ~700+ million ha = 7+ million km²")
    
    logger.info(f"\nOur sample ({len(harvest_df):,} cells):")
    logger.info(f"  Total: {total_converted_km2:,.0f} km²")
    logger.info(f"  Percentage of expected global: {total_converted_km2/7000000*100:.1f}%")
    
    # Check if we're using the right data
    logger.info(f"\n=== DATA COMPLETENESS CHECK ===")
    
    # Check how many cells have harvest data
    cells_with_harvest = (harvest_df[available_grain_crops].sum(axis=1) > 0).sum()
    logger.info(f"Cells with harvest data: {cells_with_harvest:,}/{len(harvest_df):,} ({cells_with_harvest/len(harvest_df)*100:.1f}%)")
    
    # Check per-crop coverage
    for crop in available_grain_crops:
        cells_with_crop = (harvest_df[crop] > 0).sum()
        logger.info(f"{crop}: {cells_with_crop:,} cells ({cells_with_crop/len(harvest_df)*100:.1f}%)")
    
    # Load FULL dataset to check total
    logger.info(f"\n=== FULL DATASET CHECK ===")
    logger.info("Loading FULL harvest dataset...")
    
    full_harvest_df = loader.load_spam_harvest_area()
    full_crop_harvest = processor.filter_crop_data(full_harvest_df)
    full_harvest_km2 = processor.convert_harvest_area_to_km2(full_crop_harvest)
    
    full_total_km2 = full_harvest_km2[available_grain_crops].sum().sum()
    
    logger.info(f"Full dataset total harvest: {full_total_km2:,.0f} km²")
    logger.info(f"Percentage of expected global: {full_total_km2/7000000*100:.1f}%")
    
    # Check individual crop totals in full dataset
    logger.info(f"\n=== FULL DATASET BY CROP ===")
    for crop in available_grain_crops:
        crop_total = full_harvest_km2[crop].sum()
        logger.info(f"{crop}: {crop_total:,.0f} km²")
    
    # Diagnose the issue
    logger.info(f"\n=== DIAGNOSIS ===")
    
    if full_total_km2 < 1000000:  # Less than 1M km²
        logger.error("❌ MAJOR ISSUE: Total harvest area is way too small!")
        logger.error("Possible causes:")
        logger.error("1. Wrong conversion factor (should be 0.01)")
        logger.error("2. Missing data in SPAM dataset")
        logger.error("3. Wrong crop column selection")
        logger.error("4. Data loading issue")
    elif full_total_km2 < 5000000:  # Less than 5M km²
        logger.warning("⚠️  Harvest area seems low but might be reasonable")
        logger.warning("Could be due to SPAM data coverage or crop selection")
    else:
        logger.info("✅ Harvest area values seem reasonable")
    
    return full_total_km2

if __name__ == "__main__":
    total_harvest = debug_harvest_area_values()
    
    print(f"\n" + "="*60)
    print("HARVEST AREA DIAGNOSIS")
    print("="*60)
    print(f"Total grain harvest area: {total_harvest:,.0f} km²")
    
    if total_harvest < 1000000:
        print("❌ CRITICAL ISSUE: Harvest area values are way too small")
        print("🔧 Need to investigate conversion or data loading")
    elif total_harvest < 5000000:
        print("⚠️  Harvest area seems low - check data completeness")
    else:
        print("✅ Harvest area values appear reasonable")
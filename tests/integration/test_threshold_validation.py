#!/usr/bin/env python3
"""
Threshold validation test for AgRichter framework.
Validates that thresholds are reasonable based on global production data.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add the agrichter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agrichter.core.config import Config
from agrichter.data.loader import DataLoader
from agrichter.processing.processor import DataProcessor
from agrichter.analysis.agrichter import AgRichterAnalyzer

def analyze_global_production(crop_type: str, sample_size: int = 10000):
    """Analyze global production to validate thresholds."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"THRESHOLD VALIDATION: {crop_type.upper()}")
        logger.info(f"{'='*60}")
        
        # Initialize components
        config = Config(crop_type=crop_type, root_dir='.')
        loader = DataLoader(config)
        processor = DataProcessor(config)
        analyzer = AgRichterAnalyzer(config)
        
        # Get configuration
        thresholds = config.get_thresholds()
        caloric_content = config.get_caloric_content()
        crop_indices = config.get_crop_indices()
        
        logger.info(f"Crop indices: {crop_indices}")
        logger.info(f"Caloric content: {caloric_content:.2f} kcal/g")
        logger.info(f"Current thresholds:")
        for level, value in thresholds.items():
            logger.info(f"  {level}: {value:.2e} kcal")
        
        # Load production data
        logger.info(f"Loading production data (sample size: {sample_size})...")
        production_df = loader.load_spam_production().head(sample_size)
        harvest_df = loader.load_spam_harvest_area().head(sample_size)
        
        # Process data
        crop_production = processor.filter_crop_data(production_df)
        crop_harvest = processor.filter_crop_data(harvest_df)
        
        # Convert to kcal and km²
        production_kcal = processor.convert_production_to_kcal(crop_production)
        harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
        
        # Get crop columns
        crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
        
        if not crop_columns:
            logger.warning(f"No crop data found for {crop_type}")
            return False
        
        # Calculate statistics
        total_production = production_kcal[crop_columns].sum(axis=1)
        total_harvest = harvest_km2[crop_columns].sum(axis=1)
        
        # Remove zero values for meaningful statistics
        non_zero_production = total_production[total_production > 0]
        non_zero_harvest = total_harvest[total_harvest > 0]
        
        logger.info(f"\nProduction Statistics (sample of {len(non_zero_production)} cells with production):")
        logger.info(f"  Min production: {non_zero_production.min():.2e} kcal")
        logger.info(f"  Max production: {non_zero_production.max():.2e} kcal")
        logger.info(f"  Mean production: {non_zero_production.mean():.2e} kcal")
        logger.info(f"  Median production: {non_zero_production.median():.2e} kcal")
        logger.info(f"  95th percentile: {non_zero_production.quantile(0.95):.2e} kcal")
        logger.info(f"  99th percentile: {non_zero_production.quantile(0.99):.2e} kcal")
        
        logger.info(f"\nHarvest Area Statistics (sample of {len(non_zero_harvest)} cells with harvest):")
        logger.info(f"  Min harvest: {non_zero_harvest.min():.2f} km²")
        logger.info(f"  Max harvest: {non_zero_harvest.max():.2f} km²")
        logger.info(f"  Mean harvest: {non_zero_harvest.mean():.2f} km²")
        logger.info(f"  Median harvest: {non_zero_harvest.median():.2f} km²")
        
        # Validate thresholds against production data
        logger.info(f"\nThreshold Validation:")
        
        # Check how many cells would be affected at each threshold
        for level, threshold in thresholds.items():
            affected_cells = (non_zero_production >= threshold).sum()
            percentage = (affected_cells / len(non_zero_production)) * 100
            logger.info(f"  {level} ({threshold:.2e} kcal): {affected_cells} cells ({percentage:.2f}%) would be affected")
        
        # Calculate reasonable threshold suggestions based on percentiles
        logger.info(f"\nSuggested thresholds based on production percentiles:")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = non_zero_production.quantile(p/100)
            logger.info(f"  {p}th percentile: {value:.2e} kcal")
        
        # Test magnitude calculations for different areas
        logger.info(f"\nMagnitude calculations for different disruption areas:")
        test_areas = [1, 10, 100, 1000, 10000, 100000, 1000000]
        for area in test_areas:
            magnitude = analyzer.calculate_magnitude(area)
            logger.info(f"  {area:>7,} km² → Magnitude {magnitude:.2f}")
        
        # Test severity classifications
        logger.info(f"\nSeverity classifications for threshold values:")
        for level, threshold in thresholds.items():
            classification = analyzer.classify_event_severity(threshold)
            logger.info(f"  {threshold:.2e} kcal → {classification}")
        
        # Estimate global production (very rough)
        if len(production_df) >= 1000:  # Only if we have reasonable sample
            estimated_global = (non_zero_production.sum() / sample_size) * 981508  # Total grid cells
            logger.info(f"\nEstimated global production (very rough): {estimated_global:.2e} kcal")
            
            # Compare thresholds to estimated global production
            logger.info(f"Threshold as % of estimated global production:")
            for level, threshold in thresholds.items():
                percentage = (threshold / estimated_global) * 100
                logger.info(f"  {level}: {percentage:.4f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Threshold validation failed for {crop_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Validate thresholds for all crop types."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    crop_types = ['wheat', 'rice', 'maize', 'allgrain']
    
    logger.info("🎯 AgRichter Threshold Validation")
    logger.info("=" * 60)
    logger.info("This test validates that the hardcoded thresholds are reasonable")
    logger.info("based on actual global production data patterns.")
    
    results = {}
    for crop_type in crop_types:
        results[crop_type] = analyze_global_production(crop_type, sample_size=5000)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    all_passed = True
    for crop_type, success in results.items():
        status = "✅ COMPLETED" if success else "❌ FAILED"
        logger.info(f"{crop_type.upper():>10}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info(f"\n📊 RECOMMENDATIONS:")
        logger.info("1. Review the percentile-based threshold suggestions above")
        logger.info("2. Consider if current thresholds align with historical events")
        logger.info("3. Validate against known agricultural disruption scales")
        logger.info("4. Consider regional variations in production density")
        logger.info("5. Test with full dataset for more accurate global estimates")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
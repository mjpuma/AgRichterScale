#!/usr/bin/env python3
"""
Global threshold validation test for AgriRichter framework.
Validates thresholds against total global production and reserves.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.analysis.agririchter import AgriRichterAnalyzer

def estimate_global_production(crop_type: str, full_dataset: bool = False):
    """Estimate total global production for threshold validation."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"GLOBAL PRODUCTION ANALYSIS: {crop_type.upper()}")
        logger.info(f"{'='*60}")
        
        # Initialize components
        config = Config(crop_type=crop_type, root_dir='.')
        loader = DataLoader(config)
        processor = DataProcessor(config)
        
        # Get configuration
        thresholds = config.get_thresholds()
        caloric_content = config.get_caloric_content()
        crop_indices = config.get_crop_indices()
        
        logger.info(f"Crop indices: {crop_indices}")
        logger.info(f"Caloric content: {caloric_content:.2f} kcal/g")
        
        # Load production data
        if full_dataset:
            logger.info("Loading FULL global production dataset...")
            production_df = loader.load_spam_production()
        else:
            logger.info("Loading production sample (20,000 cells)...")
            production_df = loader.load_spam_production().head(20000)
        
        # Process data
        crop_production = processor.filter_crop_data(production_df)
        production_kcal = processor.convert_production_to_kcal(crop_production)
        
        # Get crop columns
        crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
        
        if not crop_columns:
            logger.warning(f"No crop data found for {crop_type}")
            return False
        
        # Calculate total global production
        total_global_production = production_kcal[crop_columns].sum().sum()
        
        # Estimate full global if using sample
        if not full_dataset:
            total_cells = 981508  # Known total from SPAM data
            sample_cells = len(production_df)
            scaling_factor = total_cells / sample_cells
            estimated_global = total_global_production * scaling_factor
            logger.info(f"Sample production ({sample_cells:,} cells): {total_global_production:.2e} kcal")
            logger.info(f"Estimated global production: {estimated_global:.2e} kcal")
            total_global_production = estimated_global
        else:
            logger.info(f"Total global production: {total_global_production:.2e} kcal")
        
        # Analyze thresholds as percentage of global production
        logger.info(f"\nThreshold Analysis (as % of global production):")
        logger.info(f"Current thresholds:")
        
        for level, threshold in thresholds.items():
            percentage = (threshold / total_global_production) * 100
            logger.info(f"  {level}: {threshold:.2e} kcal ({percentage:.2f}% of global)")
        
        # Provide context for what these percentages mean
        logger.info(f"\nInterpretation:")
        t1_pct = (thresholds['T1'] / total_global_production) * 100
        t4_pct = (thresholds['T4'] / total_global_production) * 100
        
        if t1_pct < 1:
            logger.info(f"✓ T1 represents {t1_pct:.2f}% - reasonable for minor disruptions")
        elif t1_pct < 10:
            logger.info(f"⚠ T1 represents {t1_pct:.2f}% - significant disruption threshold")
        else:
            logger.info(f"❌ T1 represents {t1_pct:.2f}% - may be too high for minor disruptions")
        
        if t4_pct < 50:
            logger.info(f"✓ T4 represents {t4_pct:.2f}% - reasonable for catastrophic events")
        elif t4_pct < 100:
            logger.info(f"⚠ T4 represents {t4_pct:.2f}% - very severe disruption")
        else:
            logger.info(f"❌ T4 represents {t4_pct:.2f}% - exceeds total global production!")
        
        # Calculate some reference disruption scenarios
        logger.info(f"\nReference disruption scenarios:")
        
        # Small country disruption (e.g., 0.1% of global)
        small_disruption = total_global_production * 0.001
        logger.info(f"  0.1% global loss: {small_disruption:.2e} kcal")
        
        # Regional disruption (e.g., 1% of global)
        regional_disruption = total_global_production * 0.01
        logger.info(f"  1.0% global loss: {regional_disruption:.2e} kcal")
        
        # Major disruption (e.g., 5% of global)
        major_disruption = total_global_production * 0.05
        logger.info(f"  5.0% global loss: {major_disruption:.2e} kcal")
        
        # Catastrophic disruption (e.g., 20% of global)
        catastrophic_disruption = total_global_production * 0.20
        logger.info(f" 20.0% global loss: {catastrophic_disruption:.2e} kcal")
        
        # Compare with current thresholds
        logger.info(f"\nComparison with current thresholds:")
        scenarios = {
            "0.1% loss": small_disruption,
            "1.0% loss": regional_disruption, 
            "5.0% loss": major_disruption,
            "20.0% loss": catastrophic_disruption
        }
        
        analyzer = AgriRichterAnalyzer(config)
        
        for scenario, loss_value in scenarios.items():
            classification = analyzer.classify_event_severity(loss_value)
            logger.info(f"  {scenario:>10}: {loss_value:.2e} kcal → {classification}")
        
        # Suggest if thresholds seem reasonable
        logger.info(f"\nThreshold Assessment:")
        
        # Check if T1 is reasonable (should be < 1% of global)
        if thresholds['T1'] < total_global_production * 0.01:
            logger.info("✓ T1 threshold appears reasonable for minor disruptions")
        else:
            logger.info("⚠ T1 threshold may be too high for minor disruptions")
        
        # Check if T4 is reasonable (should be < 50% of global)
        if thresholds['T4'] < total_global_production * 0.5:
            logger.info("✓ T4 threshold appears reasonable for catastrophic events")
        else:
            logger.info("⚠ T4 threshold may be too high - exceeds reasonable catastrophic levels")
        
        return True
        
    except Exception as e:
        logger.error(f"Global production analysis failed for {crop_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Validate thresholds against global production for all crop types."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    crop_types = ['wheat', 'rice', 'maize', 'allgrain']
    
    logger.info("🌍 Global Threshold Validation")
    logger.info("=" * 60)
    logger.info("Validating AgriRichter thresholds against estimated global production")
    logger.info("Thresholds should represent reasonable percentages of global production")
    
    results = {}
    for crop_type in crop_types:
        results[crop_type] = estimate_global_production(crop_type, full_dataset=False)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("GLOBAL THRESHOLD VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    all_passed = True
    for crop_type, success in results.items():
        status = "✅ COMPLETED" if success else "❌ FAILED"
        logger.info(f"{crop_type.upper():>10}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info(f"\n📊 KEY INSIGHTS:")
        logger.info("1. Thresholds represent aggregate global production losses")
        logger.info("2. T1-T4 should span from minor (0.1%) to catastrophic (20%+) disruptions")
        logger.info("3. Current thresholds may need adjustment based on analysis above")
        logger.info("4. Consider historical event magnitudes for calibration")
        logger.info("5. Account for global reserves and trade in threshold setting")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
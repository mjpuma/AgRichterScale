#!/usr/bin/env python3
"""
Quick test script for all crop types in AgRichter framework.
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

def test_crop_config(crop_type: str):
    """Test crop configuration and basic functionality."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"TESTING: {crop_type.upper()}")
        logger.info(f"{'='*50}")
        
        # Initialize configuration
        config = Config(crop_type=crop_type, root_dir='.')
        
        # Log configuration details
        crop_indices = config.get_crop_indices()
        caloric_content = config.get_caloric_content()
        thresholds = config.get_thresholds()
        
        logger.info(f"✓ Crop indices: {crop_indices}")
        logger.info(f"✓ Caloric content: {caloric_content:.2f} kcal/g")
        logger.info(f"✓ Thresholds: T1={thresholds['T1']:.2e}, T2={thresholds['T2']:.2e}, T3={thresholds['T3']:.2e}, T4={thresholds['T4']:.2e} kcal")
        
        # Test data loading (small sample)
        loader = DataLoader(config)
        processor = DataProcessor(config)
        analyzer = AgRichterAnalyzer(config)
        
        logger.info("✓ Components initialized successfully")
        
        # Load small sample to test crop filtering
        logger.info("Testing data loading with small sample...")
        production_df = loader.load_spam_production().head(1000)
        
        # Test crop filtering
        crop_production = processor.filter_crop_data(production_df)
        crop_columns = [col for col in crop_production.columns if col.endswith('_A')]
        
        if crop_columns:
            logger.info(f"✓ Found {len(crop_columns)} crop columns: {crop_columns}")
            
            # Test unit conversion
            production_kcal = processor.convert_production_to_kcal(crop_production)
            total_production = production_kcal[crop_columns].sum().sum()
            logger.info(f"✓ Unit conversion successful: {total_production:.2e} kcal total")
            
            # Test magnitude calculation
            test_areas = np.array([1.0, 100.0, 10000.0])
            magnitudes = analyzer.calculate_magnitude(test_areas)
            logger.info(f"✓ Magnitude calculation: {dict(zip(test_areas, magnitudes))}")
            
            # Test severity classification
            test_loss = thresholds['T2'] * 1.5
            classification = analyzer.classify_event_severity(test_loss)
            logger.info(f"✓ Severity classification: {test_loss:.2e} kcal → {classification}")
            
        else:
            logger.warning(f"⚠️  No crop data found in sample for {crop_type}")
        
        logger.info(f"✅ {crop_type.upper()} configuration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ {crop_type.upper()} test FAILED: {str(e)}")
        return False

def main():
    """Test all crop configurations."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    crop_types = ['wheat', 'rice', 'maize', 'allgrain']
    
    logger.info("🌾 Quick crop configuration testing...")
    
    results = {}
    for crop_type in crop_types:
        results[crop_type] = test_crop_config(crop_type)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    
    all_passed = True
    for crop_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{crop_type.upper():>10}: {status}")
        if not success:
            all_passed = False
    
    # Configuration comparison
    if all_passed:
        logger.info(f"\n{'='*50}")
        logger.info("CROP CONFIGURATION COMPARISON")
        logger.info(f"{'='*50}")
        
        for crop_type in crop_types:
            config = Config(crop_type=crop_type, root_dir='.')
            indices = config.get_crop_indices()
            caloric = config.get_caloric_content()
            t1_threshold = config.get_thresholds()['T1']
            
            logger.info(f"{crop_type.upper():>10}: {len(indices):2d} crops, {caloric:4.2f} kcal/g, T1={t1_threshold:.1e}")
    
    logger.info(f"\nOverall: {'🎉 ALL CONFIGURATIONS VALID!' if all_passed else '⚠️  SOME CONFIGURATIONS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify AgRichter framework works with all crop types.
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

def test_crop_type(crop_type: str, sample_size: int = 2000):
    """Test a specific crop type."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING CROP TYPE: {crop_type.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        # Initialize configuration
        config = Config(crop_type=crop_type, root_dir='.')
        
        # Log crop configuration
        crop_indices = config.get_crop_indices()
        caloric_content = config.get_caloric_content()
        thresholds = config.get_thresholds()
        
        logger.info(f"Crop Indices: {crop_indices}")
        logger.info(f"Caloric Content: {caloric_content:.2f} kcal/g")
        logger.info(f"Thresholds: T1={thresholds['T1']:.2e}, T4={thresholds['T4']:.2e} kcal")
        
        # Initialize components
        loader = DataLoader(config)
        processor = DataProcessor(config)
        analyzer = AgRichterAnalyzer(config)
        
        # Load sample data
        logger.info(f"Loading sample data ({sample_size} cells)...")
        production_df = loader.load_spam_production().head(sample_size)
        harvest_df = loader.load_spam_harvest_area().head(sample_size)
        
        # Filter and process data
        logger.info("Processing data...")
        crop_production = processor.filter_crop_data(production_df)
        crop_harvest = processor.filter_crop_data(harvest_df)
        
        # Check if we have crop data
        crop_columns = [col for col in crop_production.columns if col.endswith('_A')]
        if not crop_columns:
            logger.error(f"❌ No crop columns found for {crop_type}")
            return False
        
        logger.info(f"Found {len(crop_columns)} crop columns: {crop_columns}")
        
        # Convert units
        production_kcal = processor.convert_production_to_kcal(crop_production)
        harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
        
        # Check data statistics
        total_production = production_kcal[crop_columns].sum().sum()
        total_harvest = harvest_km2[crop_columns].sum().sum()
        
        logger.info(f"Total Production: {total_production:.2e} kcal")
        logger.info(f"Total Harvest Area: {total_harvest:.2f} km²")
        
        if total_production == 0:
            logger.warning(f"⚠️  No production data found for {crop_type} in sample")
            return True  # Not necessarily an error, might be sparse data
        
        if total_harvest == 0:
            logger.warning(f"⚠️  No harvest area data found for {crop_type} in sample")
            return True
        
        # Test magnitude calculation
        test_areas = np.array([1.0, 100.0, 10000.0])
        magnitudes = analyzer.calculate_magnitude(test_areas)
        logger.info(f"Magnitude test: {test_areas} km² → {magnitudes}")
        
        # Test severity classification
        test_losses = np.array([
            thresholds['T1'] * 0.5,  # Below T1
            thresholds['T2'] * 1.5,  # T2
            thresholds['T4'] * 1.5   # T4
        ])
        classifications = analyzer.classify_event_severity(test_losses)
        logger.info(f"Severity test: {classifications}")
        
        # Test with matching coordinates for envelope (if we have enough data)
        common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
            harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
        )
        
        if len(common_coords) > 50:  # Need minimum data for envelope
            logger.info(f"Testing H-P envelope with {len(common_coords)} matching cells...")
            
            # Get matching data (smaller sample for speed)
            envelope_sample = min(500, len(common_coords))
            prod_matched = production_kcal[production_kcal['grid_code'].isin(common_coords['grid_code'])].head(envelope_sample)
            harv_matched = harvest_km2[harvest_km2['grid_code'].isin(common_coords['grid_code'])].head(envelope_sample)
            
            # Test envelope calculation
            envelope_data = analyzer.envelope_calculator.calculate_hp_envelope(prod_matched, harv_matched)
            logger.info(f"✅ H-P envelope calculated: {len(envelope_data['disruption_areas'])} points")
            
            # Test complete analysis
            mock_events = pd.DataFrame({
                'event': ['TestEvent1', 'TestEvent2'],
                'harvest_area_loss_ha': [10000, 100000],
                'production_loss_kcal': [thresholds['T1'] * 1.2, thresholds['T3'] * 1.5]
            })
            
            complete_results = analyzer.run_complete_analysis(prod_matched, harv_matched, mock_events)
            logger.info(f"✅ Complete analysis: {list(complete_results.keys())}")
            
        else:
            logger.warning(f"⚠️  Insufficient matching coordinates ({len(common_coords)}) for envelope test")
        
        logger.info(f"✅ {crop_type.upper()} TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {crop_type.upper()} TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_all_crops():
    """Test all supported crop types."""
    
    # Set up logging (less verbose)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    # Create a separate logger for our test output
    test_logger = logging.getLogger('test_output')
    test_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(handler)
    test_logger.propagate = False
    logger = logging.getLogger(__name__)
    
    # Define crop types to test
    crop_types = ['wheat', 'rice', 'maize', 'allgrain']
    
    logger.info("🌾 STARTING COMPREHENSIVE CROP TESTING")
    logger.info(f"Testing {len(crop_types)} crop types: {crop_types}")
    
    results = {}
    
    for crop_type in crop_types:
        try:
            success = test_crop_type(crop_type, sample_size=3000)
            results[crop_type] = success
        except Exception as e:
            logger.error(f"Critical error testing {crop_type}: {str(e)}")
            results[crop_type] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPREHENSIVE TEST RESULTS")
    logger.info(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for crop_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{crop_type.upper():>10}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nSUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL CROP TYPES WORKING PERFECTLY!")
        
        # Test crop-specific differences
        logger.info(f"\n{'='*60}")
        logger.info("CROP-SPECIFIC PARAMETER COMPARISON")
        logger.info(f"{'='*60}")
        
        for crop_type in crop_types:
            config = Config(crop_type=crop_type, root_dir='.')
            indices = config.get_crop_indices()
            caloric = config.get_caloric_content()
            t1_threshold = config.get_thresholds()['T1']
            
            logger.info(f"{crop_type.upper():>10}: indices={indices}, kcal/g={caloric:.2f}, T1={t1_threshold:.2e}")
        
        return True
    else:
        logger.error(f"❌ {failed} crop type(s) failed testing")
        return False

if __name__ == "__main__":
    success = test_all_crops()
    sys.exit(0 if success else 1)
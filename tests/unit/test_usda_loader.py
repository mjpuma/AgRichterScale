#!/usr/bin/env python3
"""Test script for USDA data loader and threshold calculator."""

import sys
import logging
from pathlib import Path

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.data.usda import create_usda_threshold_system
from agririchter.core.config import Config

def test_usda_loader():
    """Test USDA data loading functionality."""
    print("Testing USDA Data Loader...")
    
    try:
        # Initialize USDA system
        loader, calculator = create_usda_threshold_system()
        print("✓ USDA system initialized successfully")
        
        # Test loading wheat data
        wheat_data = loader.load_crop_data('wheat', year_range=(1990, 2020))
        print(f"✓ Loaded wheat data: {len(wheat_data)} years")
        print(f"  Columns: {list(wheat_data.columns)}")
        print(f"  Year range: {wheat_data['Year'].min()}-{wheat_data['Year'].max()}")
        print(f"  Sample data:\n{wheat_data.head(3)}")
        
        # Test SUR calculation
        sur_values = loader.calculate_sur(wheat_data)
        print(f"✓ Calculated SUR values: mean={sur_values.mean():.3f}, std={sur_values.std():.3f}")
        
        # Test threshold calculation
        sur_thresholds = calculator.calculate_sur_thresholds('wheat', year_range=(1990, 2020))
        print(f"✓ SUR thresholds: {sur_thresholds}")
        
        # Test dynamic threshold calculation
        dynamic_thresholds = calculator.calculate_dynamic_thresholds('wheat', year_range=(1990, 2020))
        print(f"✓ Dynamic production thresholds: {dynamic_thresholds}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing USDA loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test integration with Config class."""
    print("\nTesting Config Integration...")
    
    try:
        # Test with dynamic thresholds enabled
        config = Config('wheat', use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
        print("✓ Config with dynamic thresholds initialized")
        
        thresholds = config.get_thresholds()
        print(f"✓ Retrieved thresholds: {thresholds}")
        
        # Test USDA data access
        usda_data = config.get_usda_data((1990, 2020))
        if usda_data:
            print(f"✓ Retrieved USDA data: {list(usda_data.keys())}")
        
        # Test SUR thresholds
        sur_thresholds = config.get_sur_thresholds()
        if sur_thresholds:
            print(f"✓ Retrieved SUR thresholds: {sur_thresholds}")
        
        # Test IPC colors
        ipc_colors = config.get_ipc_colors()
        print(f"✓ Retrieved IPC colors: {ipc_colors}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing config integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_crops():
    """Test loading data for all crop types."""
    print("\nTesting All Crop Types...")
    
    try:
        loader, calculator = create_usda_threshold_system()
        
        for crop in ['wheat', 'rice', 'maize']:
            print(f"\nTesting {crop}:")
            
            # Load data
            data = loader.load_crop_data(crop, year_range=(1990, 2020))
            print(f"  ✓ Loaded {len(data)} years of data")
            
            # Calculate thresholds
            thresholds = calculator.calculate_dynamic_thresholds(crop, year_range=(1990, 2020))
            print(f"  ✓ Calculated thresholds: {list(thresholds.keys())}")
        
        # Test allgrain (combined)
        print(f"\nTesting allgrain (combined):")
        allgrain_thresholds = calculator.calculate_dynamic_thresholds('allgrain', year_range=(1990, 2020))
        print(f"  ✓ Calculated combined thresholds: {list(allgrain_thresholds.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing all crops: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("USDA Data Loader Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_usda_loader()
    success &= test_config_integration() 
    success &= test_all_crops()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
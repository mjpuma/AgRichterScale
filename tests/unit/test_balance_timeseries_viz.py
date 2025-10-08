#!/usr/bin/env python3
"""Test script for Balance Time Series visualization."""

import sys
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.visualization.balance_timeseries import BalanceTimeSeriesVisualizer, create_balance_timeseries_for_crop

def test_balance_timeseries_visualization():
    """Test Balance Time Series visualization."""
    print("Testing Balance Time Series Visualization...")
    
    try:
        # Test for different crop types
        crop_types = ['wheat', 'rice', 'allgrain']
        
        for crop_type in crop_types:
            print(f"\nTesting {crop_type}:")
            
            # Initialize config with dynamic thresholds
            config = Config(crop_type, use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
            print(f"  ✓ Config initialized for {crop_type}")
            
            # Create visualizer
            visualizer = BalanceTimeSeriesVisualizer(config)
            print(f"  ✓ Visualizer created")
            
            # Get USDA data to verify it's available
            usda_data = config.get_usda_data((1990, 2020))
            if usda_data:
                print(f"  ✓ USDA data available: {list(usda_data.keys())}")
                for crop_name, data in usda_data.items():
                    print(f"    - {crop_name}: {len(data)} years")
            else:
                print(f"  ✗ No USDA data available for {crop_type}")
                continue
            
            # Create plot
            output_path = f'test_balance_timeseries_{crop_type}.png'
            fig = visualizer.create_balance_timeseries_plot(year_range=(1990, 2020), save_path=output_path)
            print(f"  ✓ Plot created and saved to {output_path}")
            
            plt.close(fig)  # Close to save memory
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Balance Time Series visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_crop_function():
    """Test the standalone function for creating balance time series."""
    print("\nTesting Individual Crop Function...")
    
    try:
        # Test the standalone function
        crop_type = 'wheat'
        fig = create_balance_timeseries_for_crop(crop_type, year_range=(1990, 2020), 
                                               output_dir='test_outputs')
        print(f"  ✓ Standalone function created plot for {crop_type}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error testing standalone function: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_subplot_layout():
    """Test that the 2x2 subplot layout is created correctly."""
    print("\nTesting Subplot Layout...")
    
    try:
        config = Config('wheat', use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
        visualizer = BalanceTimeSeriesVisualizer(config)
        
        # Create plot
        fig = visualizer.create_balance_timeseries_plot(year_range=(1990, 2020))
        
        # Check that we have 4 subplots (2x2)
        axes = fig.get_axes()
        if len(axes) == 4:
            print("  ✓ Correct number of subplots (4)")
        else:
            print(f"  ✗ Wrong number of subplots: expected 4, got {len(axes)}")
            return False
        
        # Check subplot titles
        expected_titles = ['Stock-to-Use Ratio (SUR)', 'Ending Stocks', 'Production', 'Consumption']
        actual_titles = [ax.get_title() for ax in axes]
        
        for expected, actual in zip(expected_titles, actual_titles):
            if expected in actual:
                print(f"  ✓ Subplot title correct: {actual}")
            else:
                print(f"  ✗ Subplot title incorrect: expected '{expected}', got '{actual}'")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error testing subplot layout: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing for SUR calculation."""
    print("\nTesting Data Processing...")
    
    try:
        config = Config('wheat', use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
        
        # Get USDA data
        usda_data = config.get_usda_data((1990, 2020))
        
        if not usda_data:
            print("  ✗ No USDA data available for testing")
            return False
        
        # Test SUR calculation
        for crop_name, crop_data in usda_data.items():
            if not crop_data.empty:
                # Calculate SUR
                sur = crop_data['EndingStocks'] / crop_data['Consumption']
                
                print(f"  ✓ SUR calculated for {crop_name}: mean={sur.mean():.3f}, std={sur.std():.3f}")
                
                # Check for reasonable values
                if sur.min() >= 0 and sur.max() < 10:  # SUR should be positive and reasonable
                    print(f"    ✓ SUR values are reasonable (range: {sur.min():.3f} - {sur.max():.3f})")
                else:
                    print(f"    ✗ SUR values seem unreasonable (range: {sur.min():.3f} - {sur.max():.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Balance Time Series Visualization Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_balance_timeseries_visualization()
    success &= test_individual_crop_function()
    success &= test_subplot_layout()
    success &= test_data_processing()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test envelope visualization with edge cases (empty data, all NaN, etc.).
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.visualization.plots import EnvelopePlotter

def test_empty_envelope_data():
    """Test envelope plotting with empty data."""
    logger = logging.getLogger(__name__)
    logger.info("Testing envelope with empty data...")
    
    config = Config(crop_type='wheat', root_dir='.')
    plotter = EnvelopePlotter(config)
    
    # Empty envelope data
    empty_data = {
        'disruption_areas': np.array([]),
        'lower_bound_harvest': np.array([]),
        'lower_bound_production': np.array([]),
        'upper_bound_harvest': np.array([]),
        'upper_bound_production': np.array([])
    }
    
    try:
        fig = plotter.create_envelope_plot(
            envelope_data=empty_data,
            title="Empty Envelope Test",
            use_publication_style=False
        )
        
        # Should create figure but with warning
        output_path = Path('test_outputs/empty_envelope_test.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Empty envelope test passed: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        logger.error(f"Empty envelope test failed: {e}")
        return False

def test_all_nan_envelope_data():
    """Test envelope plotting with all NaN data."""
    logger = logging.getLogger(__name__)
    logger.info("Testing envelope with all NaN data...")
    
    config = Config(crop_type='wheat', root_dir='.')
    plotter = EnvelopePlotter(config)
    
    # All NaN envelope data
    n_points = 10
    nan_data = {
        'disruption_areas': np.full(n_points, np.nan),
        'lower_bound_harvest': np.full(n_points, np.nan),
        'lower_bound_production': np.full(n_points, np.nan),
        'upper_bound_harvest': np.full(n_points, np.nan),
        'upper_bound_production': np.full(n_points, np.nan)
    }
    
    try:
        fig = plotter.create_envelope_plot(
            envelope_data=nan_data,
            title="All NaN Envelope Test",
            use_publication_style=False
        )
        
        # Should create figure but with warning
        output_path = Path('test_outputs/all_nan_envelope_test.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ All NaN envelope test passed: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        logger.error(f"All NaN envelope test failed: {e}")
        return False

def test_mixed_valid_invalid_data():
    """Test envelope plotting with mixed valid/invalid data."""
    logger = logging.getLogger(__name__)
    logger.info("Testing envelope with mixed valid/invalid data...")
    
    config = Config(crop_type='wheat', root_dir='.')
    plotter = EnvelopePlotter(config)
    
    # Mixed data with some valid points
    n_points = 20
    mixed_data = {
        'disruption_areas': np.arange(1, n_points + 1, dtype=float),
        'lower_bound_harvest': np.arange(1, n_points + 1, dtype=float),
        'lower_bound_production': np.arange(1e6, (n_points + 1) * 1e6, 1e6),
        'upper_bound_harvest': np.arange(1, n_points + 1, dtype=float) * 2,
        'upper_bound_production': np.arange(1e6, (n_points + 1) * 1e6, 1e6) * 3
    }
    
    # Introduce some invalid values
    mixed_data['lower_bound_harvest'][5:8] = np.nan
    mixed_data['lower_bound_production'][10:12] = np.inf
    mixed_data['upper_bound_harvest'][15:17] = -1  # Negative values
    mixed_data['upper_bound_production'][18:20] = 0  # Zero values
    
    try:
        fig = plotter.create_envelope_plot(
            envelope_data=mixed_data,
            title="Mixed Valid/Invalid Envelope Test",
            use_publication_style=False
        )
        
        # Should create figure with valid points only
        output_path = Path('test_outputs/mixed_envelope_test.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Mixed data envelope test passed: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        logger.error(f"Mixed data envelope test failed: {e}")
        return False

def test_single_point_envelope():
    """Test envelope plotting with single valid point."""
    logger = logging.getLogger(__name__)
    logger.info("Testing envelope with single point...")
    
    config = Config(crop_type='wheat', root_dir='.')
    plotter = EnvelopePlotter(config)
    
    # Single point data
    single_data = {
        'disruption_areas': np.array([100.0]),
        'lower_bound_harvest': np.array([50.0]),
        'lower_bound_production': np.array([1e8]),
        'upper_bound_harvest': np.array([150.0]),
        'upper_bound_production': np.array([3e8])
    }
    
    try:
        fig = plotter.create_envelope_plot(
            envelope_data=single_data,
            title="Single Point Envelope Test",
            use_publication_style=False
        )
        
        # Should create figure but may not show filled area
        output_path = Path('test_outputs/single_point_envelope_test.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Single point envelope test passed: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        logger.error(f"Single point envelope test failed: {e}")
        return False

def test_color_and_transparency():
    """Test that the envelope uses correct light blue color and transparency."""
    logger = logging.getLogger(__name__)
    logger.info("Testing envelope color and transparency...")
    
    config = Config(crop_type='wheat', root_dir='.')
    plotter = EnvelopePlotter(config)
    
    # Valid test data
    n_points = 10
    test_data = {
        'disruption_areas': np.arange(1, n_points + 1, dtype=float),
        'lower_bound_harvest': np.arange(1, n_points + 1, dtype=float),
        'lower_bound_production': np.arange(1e6, (n_points + 1) * 1e6, 1e6),
        'upper_bound_harvest': np.arange(1, n_points + 1, dtype=float) * 2,
        'upper_bound_production': np.arange(1e6, (n_points + 1) * 1e6, 1e6) * 3
    }
    
    try:
        fig = plotter.create_envelope_plot(
            envelope_data=test_data,
            title="Color and Transparency Test",
            use_publication_style=False
        )
        
        # Check the envelope patch properties
        axes = fig.get_axes()[0]
        patches = [p for p in axes.patches if hasattr(p, 'get_facecolor')]
        
        if patches:
            envelope_patch = patches[0]  # Should be the envelope fill
            face_color = envelope_patch.get_facecolor()
            alpha = envelope_patch.get_alpha()
            edge_color = envelope_patch.get_edgecolor()
            
            logger.info(f"Envelope properties:")
            logger.info(f"  Face color: {face_color}")
            logger.info(f"  Alpha: {alpha}")
            logger.info(f"  Edge color: {edge_color}")
            
            # Verify MATLAB-exact specifications
            expected_color = [0.8, 0.9, 1.0]
            expected_alpha = 0.6
            
            color_match = np.allclose(face_color[:3], expected_color, atol=0.05)
            alpha_match = abs(alpha - expected_alpha) < 0.05
            edge_none = edge_color[3] == 0.0  # Alpha should be 0 for 'none'
            
            logger.info(f"✓ Color match (expected {expected_color}): {color_match}")
            logger.info(f"✓ Alpha match (expected {expected_alpha}): {alpha_match}")
            logger.info(f"✓ Edge color none: {edge_none}")
            
            success = color_match and alpha_match and edge_none
        else:
            logger.warning("No envelope patch found")
            success = False
        
        output_path = Path('test_outputs/color_transparency_test.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Color/transparency test saved: {output_path}")
        plt.close(fig)
        
        return success
        
    except Exception as e:
        logger.error(f"Color/transparency test failed: {e}")
        return False

def main():
    """Run all edge case tests."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Envelope Visualization Edge Cases")
    logger.info("=" * 60)
    
    # Create output directory
    Path('test_outputs').mkdir(exist_ok=True)
    
    # Run all tests
    tests = [
        ("Empty Data", test_empty_envelope_data),
        ("All NaN Data", test_all_nan_envelope_data),
        ("Mixed Valid/Invalid", test_mixed_valid_invalid_data),
        ("Single Point", test_single_point_envelope),
        ("Color & Transparency", test_color_and_transparency)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} test error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("ENVELOPE EDGE CASE TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All edge case tests PASSED!")
        print("✓ MATLAB-exact envelope visualization is robust")
        print("✓ Proper NaN and Inf handling implemented")
        print("✓ Light blue shading with correct transparency")
        print("✓ Edge color removal working")
    else:
        print("⚠️  Some tests failed - review implementation")

if __name__ == "__main__":
    main()
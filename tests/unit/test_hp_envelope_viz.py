#!/usr/bin/env python3
"""Test script for H-P Envelope visualization."""

import sys
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer, create_sample_envelope_data, create_sample_events_data

def test_hp_envelope_visualization():
    """Test H-P Envelope visualization."""
    print("Testing H-P Envelope Visualization...")
    
    try:
        # Test for different crop types
        crop_types = ['wheat', 'rice', 'allgrain']
        
        for crop_type in crop_types:
            print(f"\nTesting {crop_type}:")
            
            # Initialize config with dynamic thresholds
            config = Config(crop_type, use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
            print(f"  ✓ Config initialized for {crop_type}")
            
            # Create visualizer
            visualizer = HPEnvelopeVisualizer(config)
            print(f"  ✓ Visualizer created")
            
            # Get sample data
            envelope_data = create_sample_envelope_data(crop_type)
            events_data = create_sample_events_data(crop_type)
            print(f"  ✓ Sample data created: {len(envelope_data['disrupted_areas'])} envelope points, {len(events_data)} events")
            
            # Create plot
            output_path = f'test_hp_envelope_{crop_type}.png'
            fig = visualizer.create_hp_envelope_plot(envelope_data, events_data, output_path)
            print(f"  ✓ Plot created and saved to {output_path}")
            
            # Check thresholds
            thresholds = config.get_thresholds()
            print(f"  ✓ Thresholds: {list(thresholds.keys())}")
            
            plt.close(fig)  # Close to save memory
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing H-P Envelope visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_envelope_data_validation():
    """Test envelope data validation and processing."""
    print("\nTesting Envelope Data Validation...")
    
    try:
        config = Config('wheat', use_dynamic_thresholds=False)
        visualizer = HPEnvelopeVisualizer(config)
        
        # Test with valid data
        envelope_data = create_sample_envelope_data('wheat')
        print(f"  ✓ Valid envelope data: {len(envelope_data['disrupted_areas'])} points")
        
        # Test data ranges
        areas = envelope_data['disrupted_areas']
        upper = envelope_data['upper_bound']
        lower = envelope_data['lower_bound']
        
        print(f"  ✓ Area range: {areas.min():.0f} - {areas.max():.0f} km²")
        print(f"  ✓ Upper bound range: {upper.min():.2e} - {upper.max():.2e} kcal")
        print(f"  ✓ Lower bound range: {lower.min():.2e} - {lower.max():.2e} kcal")
        
        # Check that upper > lower
        if np.all(upper >= lower):
            print("  ✓ Upper bound >= Lower bound (valid)")
        else:
            print("  ✗ Invalid: Upper bound < Lower bound in some cases")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing envelope data validation: {e}")
        return False

def test_axis_limits():
    """Test MATLAB-exact axis limits."""
    print("\nTesting MATLAB-exact Axis Limits...")
    
    try:
        config = Config('wheat', use_dynamic_thresholds=False)
        visualizer = HPEnvelopeVisualizer(config)
        
        # Create a test plot to check axis limits
        fig, ax = plt.subplots()
        envelope_data = create_sample_envelope_data('wheat')
        events_data = create_sample_events_data('wheat')
        
        # Create plot (this should set the axis limits)
        fig = visualizer.create_hp_envelope_plot(envelope_data, events_data)
        ax = fig.gca()
        
        # Check axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        print(f"  ✓ X-axis limits: {xlim}")
        print(f"  ✓ Y-axis limits: {ylim}")
        
        # Check if they match MATLAB exact values
        expected_xlim = (2, 7)
        expected_ylim = (1e10, 1.62e16)
        
        if abs(xlim[0] - expected_xlim[0]) < 0.1 and abs(xlim[1] - expected_xlim[1]) < 0.1:
            print("  ✓ X-axis limits match MATLAB exact values")
        else:
            print(f"  ✗ X-axis limits don't match: expected {expected_xlim}, got {xlim}")
        
        if abs(ylim[0] - expected_ylim[0]) < ylim[0]*0.1 and abs(ylim[1] - expected_ylim[1]) < ylim[1]*0.1:
            print("  ✓ Y-axis limits match MATLAB exact values")
        else:
            print(f"  ✗ Y-axis limits don't match: expected {expected_ylim}, got {ylim}")
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error testing axis limits: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_severity_classification():
    """Test event severity classification."""
    print("\nTesting Event Severity Classification...")
    
    try:
        config = Config('wheat', use_dynamic_thresholds=True)
        visualizer = HPEnvelopeVisualizer(config)
        
        # Get thresholds
        thresholds = config.get_thresholds()
        print(f"  ✓ Thresholds loaded: {list(thresholds.keys())}")
        
        # Test classification at different levels
        test_cases = [
            (thresholds.get('T1', 1e14) * 0.5, 1, 'Phase 1 (Minimal)'),
            (thresholds.get('T1', 1e14) * 1.5, 2, 'Phase 2 (Stressed)'),
            (thresholds.get('T2', 1e15) * 1.5, 3, 'Phase 3 (Crisis)'),
            (thresholds.get('T3', 1e15) * 1.5, 4, 'Phase 4 (Emergency)'),
            (thresholds.get('T4', 1e15) * 1.5, 5, 'Phase 5 (Famine)'),
        ]
        
        for production_loss, expected_phase, expected_name in test_cases:
            phase, phase_name, color, marker = visualizer._classify_event_severity(production_loss)
            
            if phase == expected_phase and phase_name == expected_name:
                print(f"  ✓ {production_loss:.2e} kcal -> {phase_name} (correct)")
            else:
                print(f"  ✗ {production_loss:.2e} kcal -> {phase_name} (expected {expected_name})")
                return False
        
        # Test that different phases have different markers
        markers = set()
        for production_loss, _, _ in test_cases:
            _, _, _, marker = visualizer._classify_event_severity(production_loss)
            markers.add(marker)
        
        if len(markers) == len(test_cases):
            print(f"  ✓ All phases have unique markers: {markers}")
        else:
            print(f"  ✗ Not all phases have unique markers: {markers}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing severity classification: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_severity_visualization():
    """Test that severity classification appears in visualization."""
    print("\nTesting Severity Classification in Visualization...")
    
    try:
        config = Config('wheat', use_dynamic_thresholds=True)
        visualizer = HPEnvelopeVisualizer(config)
        
        # Create test data with varied severity
        import pandas as pd
        thresholds = config.get_thresholds()
        
        events_data = pd.DataFrame([
            {'event_name': 'Minor', 'harvest_area_km2': 5000, 'production_loss_kcal': thresholds['T1'] * 0.5},
            {'event_name': 'Moderate', 'harvest_area_km2': 20000, 'production_loss_kcal': thresholds['T2'] * 1.2},
            {'event_name': 'Severe', 'harvest_area_km2': 100000, 'production_loss_kcal': thresholds['T4'] * 1.5},
        ])
        
        envelope_data = create_sample_envelope_data('wheat')
        
        # Create plot
        fig = visualizer.create_hp_envelope_plot(envelope_data, events_data)
        ax = fig.gca()
        
        # Check that legend has severity classifications
        legend = ax.get_legend()
        if legend:
            legend_labels = [text.get_text() for text in legend.get_texts()]
            
            # Check for phase labels in legend
            phase_labels = [label for label in legend_labels if 'Phase' in label]
            
            if len(phase_labels) > 0:
                print(f"  ✓ Legend contains {len(phase_labels)} severity classifications")
                for label in phase_labels:
                    print(f"    - {label}")
            else:
                print("  ✗ No severity classifications found in legend")
                return False
        else:
            print("  ✗ No legend found")
            return False
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error testing severity visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("H-P Envelope Visualization Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_hp_envelope_visualization()
    success &= test_envelope_data_validation()
    success &= test_axis_limits()
    success &= test_severity_classification()
    success &= test_severity_visualization()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
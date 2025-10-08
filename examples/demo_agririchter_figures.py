#!/usr/bin/env python3
"""Demonstration script showing both AgriRichter Scale and H-P Envelope figures."""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.visualization.agririchter_scale import AgriRichterScaleVisualizer, create_sample_events_data
from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer, create_sample_envelope_data
from agririchter.output.manager import OutputManager

def demo_agririchter_figures():
    """Demonstrate both key AgriRichter figures."""
    print("AgriRichter Figures Demonstration")
    print("=" * 50)
    
    # Initialize configuration with dynamic thresholds
    config = Config('wheat', use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
    print(f"✓ Config initialized for wheat with dynamic thresholds")
    
    # Get thresholds
    thresholds = config.get_thresholds()
    print(f"✓ Dynamic thresholds: {list(thresholds.keys())}")
    
    # Create sample events data
    events_data = create_sample_events_data('wheat')
    print(f"✓ Sample events data: {len(events_data)} events")
    
    # Create sample envelope data
    envelope_data = create_sample_envelope_data('wheat')
    print(f"✓ Sample envelope data: {len(envelope_data['disrupted_areas'])} points")
    
    print("\n1. Creating AgriRichter Scale Figure...")
    # Create AgriRichter Scale visualization
    agririchter_viz = AgriRichterScaleVisualizer(config)
    agririchter_fig = agririchter_viz.create_agririchter_scale_plot(
        events_data, 
        save_path='demo_agririchter_scale.png'
    )
    print("   ✓ AgriRichter Scale figure created and saved")
    
    print("\n2. Creating H-P Envelope Figure...")
    # Create H-P Envelope visualization
    envelope_viz = HPEnvelopeVisualizer(config)
    envelope_fig = envelope_viz.create_hp_envelope_plot(
        envelope_data, 
        events_data,
        save_path='demo_hp_envelope.png'
    )
    print("   ✓ H-P Envelope figure created and saved")
    
    print("\n3. Using Output Manager for Complete Export...")
    # Demonstrate complete output management
    output_manager = OutputManager("demo_outputs")
    
    # Prepare complete analysis data
    analysis_data = {
        'events_data': events_data,
        'envelope_data': envelope_data,
        'thresholds': thresholds,
        'sur_thresholds': config.get_sur_thresholds(),
        'usda_data': config.get_usda_data((1990, 2020)),
        'figures': {
            'agririchter_scale': agririchter_fig,
            'hp_envelope': envelope_fig
        }
    }
    
    # Export everything
    exported_paths = output_manager.export_complete_analysis(analysis_data, 'wheat')
    print(f"   ✓ Complete analysis exported:")
    print(f"     - Data files: {len(exported_paths['data'])}")
    print(f"     - Figure files: {len(exported_paths['figures'])}")
    print(f"     - Report files: {len(exported_paths['reports'])}")
    
    print("\n4. Summary of Available Figures:")
    print("   ✓ AgriRichter Scale: Magnitude vs Production Loss with thresholds")
    print("   ✓ H-P Envelope: Harvest-Production envelope with events")
    print("   ✓ Balance Time Series: USDA PSD data analysis (4 subplots)")
    print("   ✓ Production Maps: Global production visualization")
    
    print("\nFigure Files Created:")
    print(f"   - demo_agririchter_scale.png")
    print(f"   - demo_hp_envelope.png")
    print(f"   - Complete set in demo_outputs/figures/wheat/")
    
    # Show figures
    plt.show()
    
    return True

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        success = demo_agririchter_figures()
        if success:
            print("\n✓ Demo completed successfully!")
        else:
            print("\n✗ Demo failed!")
    except Exception as e:
        print(f"\n✗ Demo error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Demo script for H-P Envelope visualization with real event data.
Tests the updated visualization with calculated event losses.
"""

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator
from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer, create_sample_envelope_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_events() -> dict:
    """Create sample event definitions for testing."""
    # Sample events with different geographic scopes
    sample_events = {
        'USA Drought 2012': {
            'country_codes': [240],  # USA
            'state_flags': [0],  # Country-level
            'state_codes': []
        },
        'China Flood 2020': {
            'country_codes': [48],  # China
            'state_flags': [0],  # Country-level
            'state_codes': []
        },
        'India Drought 2015': {
            'country_codes': [105],  # India
            'state_flags': [0],  # Country-level
            'state_codes': []
        },
        'Australia Drought 2019': {
            'country_codes': [14],  # Australia
            'state_flags': [0],  # Country-level
            'state_codes': []
        },
        'South Sudan Drought 2017': {
            'country_codes': [212],  # South Sudan
            'state_flags': [0],  # Country-level
            'state_codes': []
        }
    }
    return sample_events


def main():
    """Main demo function."""
    print("="*80)
    print("H-P ENVELOPE VISUALIZATION WITH REAL EVENTS")
    print("="*80)
    print()
    
    # Initialize components
    print("1. Initializing components...")
    config = Config('wheat', Path.cwd(), spam_version='2020')
    grid_manager = GridDataManager(config)
    grid_manager.load_spam_data()
    mapper = SpatialMapper(config, grid_manager)
    mapper.load_country_codes_mapping()
    calculator = EventCalculator(config, grid_manager, mapper)
    
    # Calculate events
    print("\n2. Calculating sample events...")
    sample_events = create_sample_events()
    events_df = calculator.calculate_all_events(sample_events)
    
    print(f"\nCalculated {len(events_df)} events:")
    print(events_df[['event_name', 'harvest_area_loss_ha', 'production_loss_kcal', 'magnitude']])
    
    # Create envelope data
    print("\n3. Creating H-P envelope data...")
    envelope_data = create_sample_envelope_data('wheat')
    
    # Create visualization
    print("\n4. Creating H-P Envelope visualization...")
    visualizer = HPEnvelopeVisualizer(config)
    
    # Create plot with real events
    fig = visualizer.create_hp_envelope_plot(
        envelope_data=envelope_data,
        events_data=events_df,
        save_path='outputs/hp_envelope_real_events_wheat.png'
    )
    
    print("\n5. Visualization complete!")
    print(f"   Saved to: outputs/hp_envelope_real_events_wheat.png")
    print(f"   Also saved as: .svg, .eps formats")
    
    # Show statistics
    print("\n" + "="*80)
    print("EVENT STATISTICS")
    print("="*80)
    print(f"Total events plotted: {len(events_df)}")
    print(f"Magnitude range: {events_df['magnitude'].min():.2f} to {events_df['magnitude'].max():.2f}")
    print(f"Production loss range: {events_df['production_loss_kcal'].min():.2e} to {events_df['production_loss_kcal'].max():.2e} kcal")
    print(f"Harvest area range: {events_df['harvest_area_loss_ha'].min():.0f} to {events_df['harvest_area_loss_ha'].max():.0f} ha")
    
    # Display the plot
    plt.show()
    
    print("\n✓ Demo complete!")


if __name__ == '__main__':
    main()

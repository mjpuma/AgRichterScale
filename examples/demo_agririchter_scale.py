#!/usr/bin/env python3
"""
Demo script to test AgriRichter Scale visualization with real events data.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.visualization.agririchter_scale import AgriRichterScaleVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_events_data():
    """Create test events data with realistic values."""
    events = [
        {'event_name': 'Dust Bowl (1934-1936)', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 5e14},
        {'event_name': 'Soviet Famine (1921-1922)', 'harvest_area_loss_ha': 20000000, 'production_loss_kcal': 8e14},
        {'event_name': 'Great Famine (1315-1317)', 'harvest_area_loss_ha': 5000000, 'production_loss_kcal': 3e14},
        {'event_name': 'Bangladesh Famine (1974)', 'harvest_area_loss_ha': 3000000, 'production_loss_kcal': 2e14},
        {'event_name': 'Chinese Famine (1959-1961)', 'harvest_area_loss_ha': 8000000, 'production_loss_kcal': 4e14},
    ]
    return pd.DataFrame(events)


def test_agririchter_scale_with_real_events():
    """Test AgriRichter Scale visualization with real events data."""
    logger.info("Testing AgriRichter Scale visualization with real events data")
    
    # Test for wheat
    logger.info("\n=== Testing Wheat ===")
    config = Config('wheat', use_dynamic_thresholds=True)
    visualizer = AgriRichterScaleVisualizer(config)
    
    # Create test events data
    events_data = create_test_events_data()
    logger.info(f"Created test events data with {len(events_data)} events")
    logger.info(f"Events columns: {events_data.columns.tolist()}")
    
    # Create plot
    output_path = Path('agririchter/output/test_agririchter_scale_wheat.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = visualizer.create_agririchter_scale_plot(events_data, output_path)
    logger.info(f"Created AgriRichter Scale plot: {output_path}")
    
    # Test for rice
    logger.info("\n=== Testing Rice ===")
    config = Config('rice', use_dynamic_thresholds=True)
    visualizer = AgriRichterScaleVisualizer(config)
    
    output_path = Path('agririchter/output/test_agririchter_scale_rice.png')
    fig = visualizer.create_agririchter_scale_plot(events_data, output_path)
    logger.info(f"Created AgriRichter Scale plot: {output_path}")
    
    # Test for allgrain
    logger.info("\n=== Testing Allgrain ===")
    config = Config('allgrain', use_dynamic_thresholds=True)
    visualizer = AgriRichterScaleVisualizer(config)
    
    output_path = Path('agririchter/output/test_agririchter_scale_allgrain.png')
    fig = visualizer.create_agririchter_scale_plot(events_data, output_path)
    logger.info(f"Created AgriRichter Scale plot: {output_path}")
    
    logger.info("\n=== All tests completed successfully! ===")


def test_with_zero_values():
    """Test handling of zero and invalid values."""
    logger.info("\n=== Testing with zero and invalid values ===")
    
    events = [
        {'event_name': 'Valid Event', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 5e14},
        {'event_name': 'Zero Harvest', 'harvest_area_loss_ha': 0, 'production_loss_kcal': 5e14},
        {'event_name': 'Zero Production', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 0},
        {'event_name': 'NaN Harvest', 'harvest_area_loss_ha': np.nan, 'production_loss_kcal': 5e14},
        {'event_name': 'Another Valid', 'harvest_area_loss_ha': 5000000, 'production_loss_kcal': 3e14},
    ]
    events_data = pd.DataFrame(events)
    
    config = Config('wheat', use_dynamic_thresholds=True)
    visualizer = AgriRichterScaleVisualizer(config)
    
    output_path = Path('agririchter/output/test_agririchter_scale_filtered.png')
    fig = visualizer.create_agririchter_scale_plot(events_data, output_path)
    logger.info("Successfully handled zero and invalid values")


def test_with_harvest_area_km2():
    """Test with events data that already has harvest_area_km2."""
    logger.info("\n=== Testing with harvest_area_km2 column ===")
    
    events = [
        {'event_name': 'Event 1', 'harvest_area_km2': 100000, 'production_loss_kcal': 5e14},
        {'event_name': 'Event 2', 'harvest_area_km2': 200000, 'production_loss_kcal': 8e14},
    ]
    events_data = pd.DataFrame(events)
    
    config = Config('wheat', use_dynamic_thresholds=True)
    visualizer = AgriRichterScaleVisualizer(config)
    
    output_path = Path('agririchter/output/test_agririchter_scale_km2.png')
    fig = visualizer.create_agririchter_scale_plot(events_data, output_path)
    logger.info("Successfully handled harvest_area_km2 column")


if __name__ == "__main__":
    try:
        test_agririchter_scale_with_real_events()
        test_with_zero_values()
        test_with_harvest_area_km2()
        print("\n✓ All AgriRichter Scale visualization tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)

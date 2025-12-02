#!/usr/bin/env python3
"""Debug script to test pipeline step by step."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing event loading...")
    
    # Create config
    config = Config(crop_type='wheat', root_dir='.')
    
    # Create loader
    loader = DataLoader(config)
    
    # Load events
    logger.info("Loading event definitions...")
    events_data = loader.load_historical_events()
    
    logger.info(f"\nCountry sheets loaded: {list(events_data['country'].keys())}")
    logger.info(f"State sheets loaded: {list(events_data['state'].keys())}")
    
    # Check first event
    if events_data['country']:
        first_event = list(events_data['country'].keys())[0]
        logger.info(f"\nFirst event: {first_event}")
        logger.info(f"Country data shape: {events_data['country'][first_event].shape}")
        logger.info(f"Country data:\n{events_data['country'][first_event].head()}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

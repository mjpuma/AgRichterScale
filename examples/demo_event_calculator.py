"""Demo script for EventCalculator functionality."""

import logging
from pathlib import Path

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate EventCalculator functionality."""
    
    logger.info("=" * 70)
    logger.info("EventCalculator Demo")
    logger.info("=" * 70)
    
    # 1. Initialize configuration
    logger.info("\n1. Initializing configuration...")
    root_dir = Path.cwd()
    config = Config(crop_type='wheat', root_dir=root_dir, spam_version='2020')
    logger.info(f"   Config: {config}")
    
    # 2. Initialize GridDataManager
    logger.info("\n2. Initializing GridDataManager...")
    grid_manager = GridDataManager(config)
    
    # Check if SPAM data exists
    file_paths = config.get_file_paths()
    if not file_paths['production'].exists():
        logger.error(f"SPAM production file not found: {file_paths['production']}")
        logger.error("Please ensure SPAM 2020 data is available in the correct location.")
        return
    
    grid_manager.load_spam_data()
    logger.info(f"   Loaded {len(grid_manager.production_df)} grid cells")
    
    # 3. Initialize SpatialMapper
    logger.info("\n3. Initializing SpatialMapper...")
    spatial_mapper = SpatialMapper(config, grid_manager)
    
    if not file_paths['country_codes'].exists():
        logger.error(f"Country codes file not found: {file_paths['country_codes']}")
        logger.error("Please ensure CountryCode_Convert.xls is available.")
        return
    
    spatial_mapper.load_country_codes_mapping()
    logger.info(f"   Loaded {len(spatial_mapper.country_codes_mapping)} country code mappings")
    
    # 4. Initialize EventCalculator
    logger.info("\n4. Initializing EventCalculator...")
    event_calculator = EventCalculator(config, grid_manager, spatial_mapper)
    logger.info(f"   {event_calculator}")
    
    # 5. Test magnitude calculation
    logger.info("\n5. Testing magnitude calculation...")
    test_harvest_areas = [10_000, 100_000, 1_000_000, 10_000_000]
    for ha in test_harvest_areas:
        magnitude = event_calculator.calculate_magnitude(ha)
        logger.info(f"   {ha:,} ha → M = {magnitude:.2f}")
    
    # 6. Test country-level loss calculation
    logger.info("\n6. Testing country-level loss calculation...")
    
    # Test USA (GDAM code 231)
    logger.info("   Testing USA (GDAM code 231)...")
    harvest_loss, production_loss, grid_cells = event_calculator.calculate_country_level_loss(231)
    logger.info(f"   USA: {harvest_loss:,.2f} ha, {production_loss:.2e} kcal, {grid_cells} cells")
    
    # Test China (GDAM code 45)
    logger.info("   Testing China (GDAM code 45)...")
    harvest_loss, production_loss, grid_cells = event_calculator.calculate_country_level_loss(45)
    logger.info(f"   China: {harvest_loss:,.2f} ha, {production_loss:.2e} kcal, {grid_cells} cells")
    
    # 7. Test single event calculation
    logger.info("\n7. Testing single event calculation...")
    
    test_event = {
        'country_codes': [231, 45],  # USA and China
        'state_flags': [0, 0],  # Both country-level
        'state_codes': []
    }
    
    result = event_calculator.calculate_single_event('TestEvent_USA_China', test_event)
    logger.info(f"   Event result:")
    logger.info(f"     Harvest area loss: {result['harvest_area_loss_ha']:,.2f} ha")
    logger.info(f"     Production loss: {result['production_loss_kcal']:.2e} kcal")
    logger.info(f"     Affected countries: {result['affected_countries']}")
    logger.info(f"     Grid cells: {result['grid_cells_count']}")
    
    magnitude = event_calculator.calculate_magnitude(result['harvest_area_loss_ha'])
    logger.info(f"     Magnitude: {magnitude:.2f}")
    
    # 8. Test batch event processing
    logger.info("\n8. Testing batch event processing...")
    
    events_definitions = {
        'USA_Drought': {
            'country_codes': [231],
            'state_flags': [0],
            'state_codes': []
        },
        'China_Flood': {
            'country_codes': [45],
            'state_flags': [0],
            'state_codes': []
        },
        'India_Drought': {
            'country_codes': [100],  # India GDAM code
            'state_flags': [0],
            'state_codes': []
        }
    }
    
    results_df = event_calculator.calculate_all_events(events_definitions)
    logger.info(f"\n   Batch processing results:")
    logger.info(f"\n{results_df.to_string()}")
    
    # 9. Test validation
    logger.info("\n9. Testing event results validation...")
    
    validation_results = event_calculator.validate_event_results(results_df)
    logger.info(f"   Validation status: {'PASSED' if validation_results['valid'] else 'FAILED'}")
    logger.info(f"   Errors: {len(validation_results['errors'])}")
    logger.info(f"   Warnings: {len(validation_results['warnings'])}")
    logger.info(f"   Suspicious events: {len(validation_results['suspicious_events'])}")
    
    # 10. Generate validation report
    logger.info("\n10. Generating validation report...")
    
    report = event_calculator.generate_validation_report(results_df)
    logger.info("\n" + report)
    
    logger.info("\n" + "=" * 70)
    logger.info("Demo complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

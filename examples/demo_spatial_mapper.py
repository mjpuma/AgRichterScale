"""Demo script for SpatialMapper functionality."""

import logging
from pathlib import Path

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run SpatialMapper demo."""
    logger.info("=" * 60)
    logger.info("SPATIAL MAPPER DEMO")
    logger.info("=" * 60)
    
    # Initialize configuration
    logger.info("\n1. Initializing configuration...")
    config = Config(crop_type='wheat', root_dir='.', spam_version='2020')
    logger.info(f"Config: {config}")
    
    # Initialize GridDataManager
    logger.info("\n2. Initializing GridDataManager...")
    grid_manager = GridDataManager(config)
    
    try:
        grid_manager.load_spam_data()
        logger.info(f"GridDataManager: {grid_manager}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load SPAM data: {e}")
        logger.error("Please ensure SPAM 2020 data files are in the correct location")
        return
    
    # Initialize SpatialMapper
    logger.info("\n3. Initializing SpatialMapper...")
    spatial_mapper = SpatialMapper(config, grid_manager)
    logger.info(f"SpatialMapper: {spatial_mapper}")
    
    # Load country codes mapping
    logger.info("\n4. Loading country codes mapping...")
    try:
        country_codes_df = spatial_mapper.load_country_codes_mapping()
        logger.info(f"Loaded {len(country_codes_df)} country code mappings")
        logger.info(f"Available code systems: {spatial_mapper.get_all_code_systems()}")
        
        # Show sample mappings
        logger.info("\nSample country code mappings:")
        sample_countries = country_codes_df.head(5)[['Country', 'GDAM ', 'ISO3 alpha']]
        for _, row in sample_countries.iterrows():
            logger.info(f"  {row['Country']}: GDAM={row['GDAM ']}, ISO3={row['ISO3 alpha']}")
    except Exception as e:
        logger.error(f"Failed to load country codes: {e}")
        return
    
    # Test ISO3 code lookup
    logger.info("\n5. Testing ISO3 code lookup...")
    
    # Get a few valid country codes to test
    test_countries = country_codes_df[
        country_codes_df['GDAM '].notna() & 
        country_codes_df['ISO3 alpha'].notna()
    ].head(3)
    
    for _, row in test_countries.iterrows():
        gdam_code = row['GDAM ']
        country_name = row['Country']
        expected_iso3 = row['ISO3 alpha']
        
        iso3 = spatial_mapper.get_iso3_from_country_code(gdam_code, 'GDAM ')
        logger.info(f"  {country_name} (GDAM {gdam_code}) -> ISO3: {iso3}")
        
        if iso3 != expected_iso3:
            logger.warning(f"    Expected {expected_iso3}, got {iso3}")
    
    # Test country-level grid cell mapping
    logger.info("\n6. Testing country-level grid cell mapping...")
    
    # Try to map USA if available
    usa_row = country_codes_df[country_codes_df['ISO3 alpha'] == 'USA']
    if len(usa_row) > 0:
        gdam_code = usa_row.iloc[0]['GDAM ']
        logger.info(f"Mapping USA (GDAM code: {gdam_code}) to grid cells...")
        
        prod_ids, harv_ids = spatial_mapper.map_country_to_grid_cells(gdam_code, 'GDAM ')
        logger.info(f"  Found {len(prod_ids)} production cells, {len(harv_ids)} harvest area cells")
        
        if len(prod_ids) > 0:
            logger.info(f"  Sample production cell IDs: {prod_ids[:5]}")
    else:
        logger.info("USA not found in country codes, trying first available country...")
        first_country = test_countries.iloc[0]
        gdam_code = first_country['GDAM ']
        country_name = first_country['Country']
        
        logger.info(f"Mapping {country_name} (GDAM code: {gdam_code}) to grid cells...")
        prod_ids, harv_ids = spatial_mapper.map_country_to_grid_cells(gdam_code, 'GDAM ')
        logger.info(f"  Found {len(prod_ids)} production cells, {len(harv_ids)} harvest area cells")
    
    # Test state-level mapping
    logger.info("\n7. Testing state-level grid cell mapping...")
    
    # Use the same country as above
    if len(usa_row) > 0:
        gdam_code = usa_row.iloc[0]['GDAM ']
        country_name = 'USA'
    else:
        gdam_code = first_country['GDAM ']
        country_name = first_country['Country']
    
    # Test with dummy state codes
    state_codes = [1.0, 2.0]
    logger.info(f"Mapping states {state_codes} in {country_name} to grid cells...")
    
    prod_ids, harv_ids = spatial_mapper.map_state_to_grid_cells(
        gdam_code, state_codes, 'GDAM '
    )
    logger.info(f"  Found {len(prod_ids)} production cells, {len(harv_ids)} harvest area cells")
    
    # Test validation with sample event mappings
    logger.info("\n8. Testing spatial mapping validation...")
    
    # Create sample event mappings
    event_mappings = {
        'Event1': (['cell1', 'cell2', 'cell3'], ['cell1', 'cell2', 'cell3']),
        'Event2': (['cell4', 'cell5'], ['cell4', 'cell5']),
        'Event3': ([], []),  # Event with no cells
        'Event4': (['cell6'], ['cell6']),
    }
    
    validation_results = spatial_mapper.validate_spatial_mapping(event_mappings)
    logger.info(f"Validation results:")
    logger.info(f"  Total events: {validation_results['total_events']}")
    logger.info(f"  Events with cells: {validation_results['events_with_cells']}")
    logger.info(f"  Events without cells: {validation_results['events_without_cells']}")
    logger.info(f"  Success rate: {validation_results['success_rate_percent']:.1f}%")
    logger.info(f"  Total production cells: {validation_results['total_production_cells']}")
    
    if validation_results['events_with_zero_cells']:
        logger.info(f"  Events with zero cells: {validation_results['events_with_zero_cells']}")
    
    # Generate validation report
    logger.info("\n9. Generating spatial mapping quality report...")
    report = spatial_mapper.generate_spatial_mapping_report(event_mappings)
    print("\n" + report)
    
    # Get mapping statistics
    logger.info("\n10. Getting mapping statistics...")
    stats = spatial_mapper.get_mapping_statistics()
    logger.info("Mapping statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Test boundary data loading (optional)
    logger.info("\n11. Testing boundary data loading (optional)...")
    logger.info("Note: Boundary data is optional and not required for basic functionality")
    
    # Try to load boundary data if files exist
    ancillary_dir = Path('ancillary')
    country_shapefile = ancillary_dir / 'gdam_v2_country.shp'
    state_shapefile = ancillary_dir / 'gdam_v2_state.shp'
    
    spatial_mapper.load_boundary_data(
        country_shapefile=country_shapefile if country_shapefile.exists() else None,
        state_shapefile=state_shapefile if state_shapefile.exists() else None
    )
    
    logger.info(f"Boundary data loaded: {spatial_mapper.boundary_data_loaded}")
    
    # Test cache clearing
    logger.info("\n12. Testing cache management...")
    logger.info(f"Cache size before clearing: {len(spatial_mapper._cache)}")
    spatial_mapper.clear_cache()
    logger.info(f"Cache size after clearing: {len(spatial_mapper._cache)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("SPATIAL MAPPER DEMO COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

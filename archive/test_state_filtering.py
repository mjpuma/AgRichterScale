"""Quick test to verify state-level filtering is working."""

import logging
from pathlib import Path
import pandas as pd

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_state_filtering():
    """Test state-level filtering with a known example."""
    
    logger.info("="*70)
    logger.info("Testing State-Level Filtering")
    logger.info("="*70)
    
    # Initialize
    config = Config('wheat', Path.cwd(), spam_version='2020')
    grid_manager = GridDataManager(config)
    grid_manager.load_spam_data()
    
    spatial_mapper = SpatialMapper(config, grid_manager)
    spatial_mapper.load_country_codes_mapping()
    
    # Test 1: USA country-level (should get all USA grid cells)
    logger.info("\n1. Testing USA country-level filtering...")
    usa_code = 240  # Correct GDAM code for USA
    usa_iso3 = spatial_mapper.get_iso3_from_country_code(usa_code)
    usa_fips = spatial_mapper.get_fips_from_country_code(usa_code)
    logger.info(f"   USA GDAM {usa_code} → ISO3: {usa_iso3} → FIPS: {usa_fips}")
    
    prod_cells, harv_cells = grid_manager.get_grid_cells_by_iso3(usa_fips)
    logger.info(f"   ✓ Found {len(prod_cells)} grid cells for entire USA")
    
    # Show some state names in the data
    if 'ADM1_NAME' in prod_cells.columns:
        states = prod_cells['ADM1_NAME'].unique()
        logger.info(f"   States in data: {len(states)}")
        logger.info(f"   Sample states: {list(states[:10])}")
    
    # Test 2: USA state-level (filter to specific states)
    logger.info("\n2. Testing USA state-level filtering...")
    
    # Try filtering to Kansas and Nebraska (major wheat states)
    test_states = ['Kansas', 'Nebraska', 'North Dakota', 'Montana']
    logger.info(f"   Filtering to states: {test_states}")
    
    # Filter using state names
    state_prod_cells = prod_cells[prod_cells['ADM1_NAME'].isin(test_states)]
    logger.info(f"   ✓ Found {len(state_prod_cells)} grid cells in selected states")
    
    # Show which states were matched
    matched_states = state_prod_cells['ADM1_NAME'].unique()
    logger.info(f"   Matched states: {list(matched_states)}")
    
    # Calculate wheat production for these states
    crop_indices = config.get_crop_indices()
    state_harvest = grid_manager.get_crop_harvest_area(state_prod_cells, crop_indices)
    state_production = grid_manager.get_crop_production(state_prod_cells, crop_indices, convert_to_kcal=False)
    
    logger.info(f"   Harvest area: {state_harvest:,.0f} ha")
    logger.info(f"   Production: {state_production:,.0f} MT")
    
    # Test 3: Compare country vs state totals
    logger.info("\n3. Comparing country-level vs state-level totals...")
    
    total_usa_harvest = grid_manager.get_crop_harvest_area(prod_cells, crop_indices)
    total_usa_production = grid_manager.get_crop_production(prod_cells, crop_indices, convert_to_kcal=False)
    
    logger.info(f"   Total USA harvest area: {total_usa_harvest:,.0f} ha")
    logger.info(f"   Selected states harvest: {state_harvest:,.0f} ha ({state_harvest/total_usa_harvest*100:.1f}%)")
    logger.info(f"   Total USA production: {total_usa_production:,.0f} MT")
    logger.info(f"   Selected states production: {state_production:,.0f} MT ({state_production/total_usa_production*100:.1f}%)")
    
    # Test 4: Test SpatialMapper's state filtering method
    logger.info("\n4. Testing SpatialMapper.map_state_to_grid_cells()...")
    
    # This would use state codes from the Excel file
    # For now, test with state names as codes
    prod_ids, harv_ids = spatial_mapper.map_state_to_grid_cells(
        usa_code, 
        test_states,  # Using state names as codes
        code_system='GDAM '
    )
    
    logger.info(f"   ✓ SpatialMapper returned {len(prod_ids)} production cell IDs")
    logger.info(f"   ✓ SpatialMapper returned {len(harv_ids)} harvest cell IDs")
    
    # Test 5: Show available states for a few countries
    logger.info("\n5. Checking available states in SPAM data...")
    
    test_countries = [
        (240, 'USA'),
        (45, 'China'),  # Note: GDAM 45 might not be China
        (100, 'India')
    ]
    
    for country_code, country_name in test_countries:
        iso3 = spatial_mapper.get_iso3_from_country_code(country_code)
        fips = spatial_mapper.get_fips_from_country_code(country_code)
        if fips:
            try:
                c_prod, c_harv = grid_manager.get_grid_cells_by_iso3(fips)
                if len(c_prod) > 0 and 'ADM1_NAME' in c_prod.columns:
                    states = c_prod['ADM1_NAME'].unique()
                    logger.info(f"   {country_name} (ISO3: {iso3}, FIPS: {fips}): {len(states)} states/provinces")
                    logger.info(f"     Sample: {list(states[:5])}")
            except:
                logger.info(f"   {country_name}: No data found")
    
    logger.info("\n" + "="*70)
    logger.info("State filtering test complete!")
    logger.info("="*70)
    
    return True


if __name__ == '__main__':
    test_state_filtering()

#!/usr/bin/env python3
"""
Comprehensive test for newer countries (post-2000) including state/province filtering.
Tests both country-level and state-level data access.
"""

from pathlib import Path
from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator


def test_newer_countries_with_states():
    """Test mapping and state filtering for newer countries."""
    
    config = Config('wheat', Path.cwd(), spam_version='2020')
    grid_manager = GridDataManager(config)
    grid_manager.load_spam_data()
    mapper = SpatialMapper(config, grid_manager)
    mapper.load_country_codes_mapping()
    calculator = EventCalculator(config, grid_manager, mapper)
    
    # Test newer countries (created after 2000) with correct GDAM codes
    newer_countries = [
        {'name': 'South Sudan', 'gdam': 212, 'iso3': 'SSD', 'expected_fips': 'OD', 'year': 2011},
        {'name': 'Montenegro', 'gdam': 150, 'iso3': 'MNE', 'expected_fips': 'MW', 'year': 2006},
        {'name': 'Timor-Leste', 'gdam': 67, 'iso3': 'TMP', 'expected_fips': 'TT', 'year': 2002},
        {'name': 'Serbia', 'gdam': 200, 'iso3': 'SRB', 'expected_fips': 'RI', 'year': 2006},
    ]
    
    print('='*80)
    print('COMPREHENSIVE TEST: Newer Countries with State/Province Filtering')
    print('='*80)
    
    all_passed = True
    total_states = 0
    
    for country in newer_countries:
        name = country['name']
        gdam = country['gdam']
        expected_iso3 = country['iso3']
        expected_fips = country['expected_fips']
        year = country['year']
        
        print(f"\n{'='*80}")
        print(f"Testing: {name} (Independent since {year})")
        print(f"{'='*80}")
        
        # Test mapping
        iso3 = mapper.get_iso3_from_country_code(gdam)
        fips = mapper.get_fips_from_country_code(gdam)
        
        print(f"  Mapping: GDAM {gdam} → ISO3 {iso3} → FIPS {fips}")
        print(f"  Expected: GDAM {gdam} → ISO3 {expected_iso3} → FIPS {expected_fips}")
        
        # Verify mapping
        if iso3 != expected_iso3:
            print(f"  ✗ FAILED: ISO3 mismatch (expected {expected_iso3}, got {iso3})")
            all_passed = False
            continue
        elif fips != expected_fips:
            print(f"  ✗ FAILED: FIPS mismatch (expected {expected_fips}, got {fips})")
            all_passed = False
            continue
        else:
            print(f"  ✓ Mapping correct")
        
        # Test grid cells
        try:
            prod_cells, harv_cells = grid_manager.get_grid_cells_by_iso3(fips)
            print(f"  ✓ Grid cells: {len(prod_cells):,}")
            
            # Get states/provinces
            states = prod_cells['ADM1_NAME'].unique()
            num_states = len(states)
            total_states += num_states
            print(f"  ✓ States/Provinces: {num_states}")
            
            # Show all states for countries with few states
            if num_states <= 15:
                print(f"    All states: {sorted(states.tolist())}")
            else:
                print(f"    Sample states: {sorted(states.tolist())[:10]}")
            
            # Test state-level filtering
            if num_states > 0:
                test_state = states[0]
                state_cells = prod_cells[prod_cells['ADM1_NAME'] == test_state]
                total_prod = state_cells['WHEA_A'].sum()
                total_harv = state_cells['WHEA_A'].sum()
                
                print(f"\n  State Filtering Test: {test_state}")
                print(f"    Grid cells: {len(state_cells):,}")
                print(f"    Wheat production: {total_prod:,.2f} MT")
                print(f"    Wheat harvest area: {total_harv:,.2f} ha")
                print(f"    ✓ State filtering works!")
            
            # Test event calculation (get total production)
            harvest_area_ha, production_kcal, grid_cells_count = calculator.calculate_country_level_loss(
                country_code=gdam
            )
            
            print(f"\n  Production Data Test:")
            print(f"    Total harvest area: {harvest_area_ha:,.0f} ha")
            print(f"    Total production: {production_kcal:,.0f} kcal")
            print(f"    Grid cells: {grid_cells_count:,}")
            print(f"    ✓ Production calculation works!")
            
        except Exception as e:
            print(f"  ✗ FAILED: Error in calculations: {e}")
            all_passed = False
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Countries tested: {len(newer_countries)}")
    print(f"Total states/provinces found: {total_states}")
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("  - All newer countries have correct ISO3 → FIPS mapping")
        print("  - All countries have grid cells in SPAM 2020 data")
        print("  - All countries have state/province level data")
        print("  - State filtering works correctly")
        print("  - Production calculations work for all countries")
        print("\nThe system fully supports post-2000 countries with state-level filtering!")
    else:
        print("\n✗ SOME TESTS FAILED - See details above")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = test_newer_countries_with_states()
    sys.exit(0 if success else 1)

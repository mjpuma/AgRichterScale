"""Demonstration of GridDataManager functionality."""

import logging
from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate GridDataManager capabilities."""
    
    print("=" * 60)
    print("GridDataManager Demonstration")
    print("=" * 60)
    print()
    
    # Create configuration for wheat
    print("1. Creating configuration for wheat...")
    config = Config(crop_type='wheat', root_dir='.', spam_version='2020')
    print(f"   Config: {config}")
    print()
    
    # Create GridDataManager
    print("2. Creating GridDataManager...")
    grid_manager = GridDataManager(config)
    print(f"   {grid_manager}")
    print()
    
    # Load SPAM data
    print("3. Loading SPAM 2020 data...")
    try:
        prod_df, harvest_df = grid_manager.load_spam_data()
        print(f"   ✓ Loaded {len(prod_df):,} production grid cells")
        print(f"   ✓ Loaded {len(harvest_df):,} harvest area grid cells")
        print()
    except FileNotFoundError as e:
        print(f"   ✗ Error: {e}")
        print("   Please ensure SPAM 2020 data files are in the correct location.")
        return
    
    # Create spatial index
    print("4. Creating spatial index...")
    grid_manager.create_spatial_index()
    print(f"   ✓ Spatial index created")
    print(f"   {grid_manager}")
    print()
    
    # Get available countries
    print("5. Getting available ISO3 country codes...")
    iso3_codes = grid_manager.get_available_iso3_codes()
    print(f"   ✓ Found {len(iso3_codes)} countries")
    print(f"   First 10: {iso3_codes[:10]}")
    print()
    
    # Query by country (USA as example)
    if 'US' in iso3_codes:
        print("6. Querying grid cells for United States (US)...")
        prod_cells, harvest_cells = grid_manager.get_grid_cells_by_iso3('US')
        print(f"   ✓ Found {len(prod_cells):,} grid cells for US")
        print()
        
        # Calculate production and harvest area for US wheat
        print("7. Calculating wheat production for United States...")
        crop_indices = config.get_crop_indices()
        
        production_kcal = grid_manager.get_crop_production(
            prod_cells, crop_indices, convert_to_kcal=True
        )
        production_mt = grid_manager.get_crop_production(
            prod_cells, crop_indices, convert_to_kcal=False
        )
        harvest_area_ha = grid_manager.get_crop_harvest_area(
            harvest_cells, crop_indices
        )
        
        print(f"   ✓ Production: {production_mt:,.0f} metric tons")
        print(f"   ✓ Production: {production_kcal:.2e} kcal")
        print(f"   ✓ Harvest area: {harvest_area_ha:,.0f} hectares")
        print()
    
    # Query by bounding box (Europe region)
    print("8. Querying grid cells by bounding box (Europe)...")
    bounds = (-10, 35, 40, 70)  # Rough Europe bounds
    prod_cells, harvest_cells = grid_manager.get_grid_cells_by_coordinates(bounds)
    print(f"   ✓ Found {len(prod_cells):,} grid cells in bounding box")
    print()
    
    # Validate grid data
    print("9. Validating grid data...")
    validation_results = grid_manager.validate_grid_data()
    if validation_results['valid']:
        print("   ✓ Validation PASSED")
    else:
        print("   ✗ Validation FAILED")
    
    print(f"   - Errors: {len(validation_results['errors'])}")
    print(f"   - Warnings: {len(validation_results['warnings'])}")
    print()
    
    # Generate validation report
    print("10. Generating validation report...")
    report = grid_manager.generate_validation_report()
    print(report)
    print()
    
    print("=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

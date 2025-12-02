#!/usr/bin/env python3
"""Script to verify SPAM 2020 data structure and document differences from SPAM 2010."""

import sys
from pathlib import Path
from agririchter.core.config import Config

def main():
    """Verify SPAM 2020 structure."""
    print("=" * 70)
    print("SPAM 2020 Data Structure Verification")
    print("=" * 70)
    print()
    
    # Initialize config with SPAM 2020
    try:
        config = Config(crop_type='wheat', spam_version='2020')
        print(f"✓ Config initialized successfully")
        print(f"  SPAM Version: {config.get_spam_version()}")
        print(f"  Root Directory: {config.root_dir}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize config: {e}")
        return 1
    
    # Validate SPAM files
    print("Validating SPAM 2020 files...")
    print("-" * 70)
    
    validation_results = config.validate_spam_files()
    
    # Production file validation
    print("\nProduction File:")
    print(f"  Path: {config.data_files['production']}")
    prod_results = validation_results.get('production', {})
    print(f"  Exists: {validation_results.get('production_exists', False)}")
    print(f"  Readable: {prod_results.get('file_readable', False)}")
    print(f"  Columns Valid: {prod_results.get('columns_valid', False)}")
    print(f"  Total Columns: {prod_results.get('total_columns', 0)}")
    
    if prod_results.get('metadata_cols_present'):
        print(f"  Metadata Columns: {len(prod_results['metadata_cols_present'])} present")
    if prod_results.get('crop_cols_present'):
        print(f"  Crop Columns: {len(prod_results['crop_cols_present'])} present")
    
    if prod_results.get('coordinate_cols_valid'):
        x_range = prod_results.get('x_range', (0, 0))
        y_range = prod_results.get('y_range', (0, 0))
        print(f"  X (Longitude) Range: {x_range[0]:.2f} to {x_range[1]:.2f}")
        print(f"  Y (Latitude) Range: {y_range[0]:.2f} to {y_range[1]:.2f}")
    
    if prod_results.get('errors'):
        print(f"  Errors:")
        for error in prod_results['errors']:
            print(f"    - {error}")
    
    # Harvest area file validation
    print("\nHarvest Area File:")
    print(f"  Path: {config.data_files['harvest_area']}")
    harvest_results = validation_results.get('harvest_area', {})
    print(f"  Exists: {validation_results.get('harvest_area_exists', False)}")
    print(f"  Readable: {harvest_results.get('file_readable', False)}")
    print(f"  Columns Valid: {harvest_results.get('columns_valid', False)}")
    print(f"  Total Columns: {harvest_results.get('total_columns', 0)}")
    
    if harvest_results.get('metadata_cols_present'):
        print(f"  Metadata Columns: {len(harvest_results['metadata_cols_present'])} present")
    if harvest_results.get('crop_cols_present'):
        print(f"  Crop Columns: {len(harvest_results['crop_cols_present'])} present")
    
    if harvest_results.get('errors'):
        print(f"  Errors:")
        for error in harvest_results['errors']:
            print(f"    - {error}")
    
    # Files consistency
    print(f"\nFiles Consistent: {validation_results.get('files_consistent', False)}")
    
    # Key crop columns check
    print("\n" + "=" * 70)
    print("Key Crop Columns Verification")
    print("=" * 70)
    
    key_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'SORG_A', 'MILL_A', 'OCER_A']
    crop_cols_present = prod_results.get('crop_cols_present', [])
    
    for crop in key_crops:
        status = "✓" if crop in crop_cols_present else "✗"
        print(f"  {status} {crop}")
    
    # Metadata columns check
    print("\n" + "=" * 70)
    print("Metadata Columns Verification")
    print("=" * 70)
    
    required_metadata = ['grid_code', 'x', 'y', 'FIPS0', 'ADM0_NAME', 'ADM1_NAME']
    metadata_cols_present = prod_results.get('metadata_cols_present', [])
    
    for col in required_metadata:
        status = "✓" if col in metadata_cols_present else "✗"
        print(f"  {status} {col}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_valid = (
        validation_results.get('production_exists', False) and
        validation_results.get('harvest_area_exists', False) and
        prod_results.get('columns_valid', False) and
        harvest_results.get('columns_valid', False) and
        validation_results.get('files_consistent', False)
    )
    
    if all_valid:
        print("✓ All SPAM 2020 files are valid and ready to use")
        print()
        print("Key findings:")
        print("  - FIPS0 column contains ISO3 country codes")
        print("  - Crop columns use uppercase with _A suffix (e.g., WHEA_A)")
        print("  - Coordinate columns are 'x' (longitude) and 'y' (latitude)")
        print("  - Both production and harvest area files have identical structure")
        return 0
    else:
        print("✗ SPAM 2020 validation failed")
        print("\nPlease check the errors above and ensure:")
        print("  1. SPAM 2020 files are in the correct location")
        print("  2. Files are readable and not corrupted")
        print("  3. File structure matches expected format")
        return 1

if __name__ == '__main__':
    sys.exit(main())

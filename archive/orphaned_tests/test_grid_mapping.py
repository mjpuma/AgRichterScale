#!/usr/bin/env python3
"""
Test script for proper SPAM grid mapping using CELL5M.asc.
"""

import sys
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the agrichter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agrichter.core.config import Config
from agrichter.data.loader import DataLoader
from agrichter.processing.processor import DataProcessor
from agrichter.visualization.maps import GlobalProductionMapper

def test_grid_mapping(crop_type: str = 'wheat', sample_size: int = 5000):
    """Test grid mapping with proper SPAM grid structure."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING GRID MAPPING: {crop_type.upper()}")
        logger.info(f"{'='*60}")
        
        # Initialize components
        config = Config(crop_type=crop_type, root_dir='.')
        loader = DataLoader(config)
        processor = DataProcessor(config)
        mapper = GlobalProductionMapper(config)
        
        # Load production data with cells that have production
        logger.info("Loading production data...")
        full_production_df = loader.load_spam_production()
        
        # Process and filter for cells with production
        crop_production_full = processor.filter_crop_data(full_production_df)
        production_kcal_full = processor.convert_production_to_kcal(crop_production_full)
        
        # Get crop columns
        crop_columns = [col for col in production_kcal_full.columns if col.endswith('_A')]
        
        if not crop_columns:
            logger.warning(f"No crop data found for {crop_type}")
            return False
        
        # Filter to cells with actual production
        total_production = production_kcal_full[crop_columns].sum(axis=1)
        has_production = total_production > 0
        production_with_data = production_kcal_full[has_production]
        
        logger.info(f"Found {len(production_with_data)} cells with production")
        
        # Take a sample for testing
        if len(production_with_data) > sample_size:
            production_sample = production_with_data.sample(n=sample_size, random_state=42)
            logger.info(f"Using sample of {sample_size} cells")
        else:
            production_sample = production_with_data
            logger.info(f"Using all {len(production_sample)} cells with production")
        
        # Check if grid_code column exists
        if 'grid_code' not in production_sample.columns:
            logger.error("No grid_code column found in production data!")
            return False
        
        logger.info(f"Grid code range in sample: {production_sample['grid_code'].min()} to {production_sample['grid_code'].max()}")
        
        # Process the sample
        crop_production = processor.filter_crop_data(production_sample)
        production_kcal = processor.convert_production_to_kcal(crop_production)
        
        # Calculate statistics
        total_production_sample = production_kcal[crop_columns].sum().sum()
        logger.info(f"Total production in sample: {total_production_sample:.2e} kcal")
        
        # Create output directory
        output_dir = Path('test_outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Generate map with proper grid mapping
        logger.info("Generating map with proper SPAM grid structure...")
        output_path = output_dir / f'grid_mapped_{crop_type}.png'
        
        fig = mapper.create_global_map(
            production_kcal,
            title=f'Global {crop_type.title()} Production (Grid Mapped - {len(production_sample):,} cells)',
            output_path=output_path,
            dpi=200,
            figsize=(14, 8)
        )
        
        # Close figure to free memory
        plt.close(fig)
        
        # Verify output file was created
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024
            logger.info(f"✅ Grid-mapped map saved: {output_path}")
            logger.info(f"   File size: {file_size:.1f} KB")
            
            return True
        else:
            logger.error(f"❌ Map file not created: {output_path}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Grid mapping test failed for {crop_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test grid mapping for crop types."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    crop_types = ['wheat', 'allgrain']
    
    logger.info("🗺️  Testing Proper SPAM Grid Mapping")
    logger.info("=" * 60)
    logger.info("Using CELL5M.asc for proper grid structure")
    
    results = {}
    for crop_type in crop_types:
        results[crop_type] = test_grid_mapping(crop_type, sample_size=10000)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("GRID MAPPING TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    all_passed = True
    for crop_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{crop_type.upper():>10}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info(f"\n🎉 GRID MAPPING SUCCESSFUL!")
        logger.info("Maps should now show proper agricultural production patterns")
        
        # List generated files
        output_dir = Path('test_outputs')
        if output_dir.exists():
            grid_files = list(output_dir.glob('grid_mapped_*.png'))
            if grid_files:
                logger.info(f"\n📁 Generated grid-mapped files:")
                for file_path in grid_files:
                    logger.info(f"   {file_path}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Generate All Key Figures for AgriRichter Analysis

This script generates:
1. Global harvest area maps
2. Global production maps
3. AgriRichter scale figure with events
4. H-P envelope figure with events

For crops: wheat, rice, allgrain
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_figures_for_crop(crop_type, output_dir='outputs'):
    """
    Generate all key figures for a specific crop.
    
    Args:
        crop_type: 'wheat', 'rice', or 'allgrain'
        output_dir: Directory to save outputs
    """
    logger.info("=" * 80)
    logger.info(f"GENERATING FIGURES FOR {crop_type.upper()}")
    logger.info("=" * 80)
    
    try:
        # Create configuration (use current directory as root)
        config = Config(crop_type=crop_type, root_dir='.')
        
        # Create output directory
        crop_output_dir = Path(output_dir) / crop_type
        crop_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {crop_output_dir}")
        
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = EventsPipeline(config, str(crop_output_dir))
        
        # Run complete pipeline
        logger.info("Running complete pipeline...")
        results = pipeline.run_complete_pipeline()
        
        # Check results
        if results:
            logger.info(f"\n✓ Successfully generated figures for {crop_type}")
            logger.info(f"  Output directory: {crop_output_dir}")
            
            # List generated files
            output_files = list(crop_output_dir.glob('*'))
            logger.info(f"  Generated {len(output_files)} files:")
            for f in sorted(output_files):
                logger.info(f"    - {f.name}")
            
            return True
        else:
            logger.error(f"✗ Pipeline returned no results for {crop_type}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error generating figures for {crop_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate all key AgriRichter figures'
    )
    parser.add_argument(
        '--crops',
        nargs='+',
        default=['wheat', 'rice', 'allgrain'],
        choices=['wheat', 'rice', 'allgrain', 'corn'],
        help='Crops to process (default: wheat rice allgrain)'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("AGRIRICHTER FIGURE GENERATION")
    logger.info("=" * 80)
    logger.info(f"Crops: {', '.join(args.crops)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Process each crop
    results = {}
    for crop in args.crops:
        success = generate_figures_for_crop(crop, args.output_dir)
        results[crop] = success
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    successful = [crop for crop, success in results.items() if success]
    failed = [crop for crop, success in results.items() if not success]
    
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    for crop in successful:
        logger.info(f"  ✓ {crop}")
    
    if failed:
        logger.info(f"\nFailed: {len(failed)}/{len(results)}")
        for crop in failed:
            logger.info(f"  ✗ {crop}")
        return 1
    else:
        logger.info("\n✓ All figures generated successfully!")
        logger.info(f"\nOutputs saved to: {args.output_dir}/")
        return 0


if __name__ == '__main__':
    sys.exit(main())

"""Demo script for AgriRichter Events Pipeline.

This script demonstrates how to use the EventsPipeline to:
1. Load SPAM 2020 data and event definitions
2. Calculate losses for all 21 historical events
3. Generate publication-quality visualizations
4. Export results to organized directory structure
5. Generate comprehensive summary report

Usage:
    python demo_events_pipeline.py
"""

import logging
from pathlib import Path
from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline


def setup_logging():
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Run the complete events analysis pipeline."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("AgriRichter Events Pipeline Demo")
    logger.info("=" * 80)
    
    # Configuration
    crop_type = 'wheat'  # Options: 'wheat', 'rice', 'allgrain'
    output_dir = Path('outputs') / f'events_{crop_type}'
    
    logger.info(f"Crop Type: {crop_type}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("")
    
    try:
        # Initialize configuration
        config = Config(crop_type=crop_type)
        
        # Initialize pipeline
        pipeline = EventsPipeline(config, str(output_dir))
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Display results summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE RESULTS SUMMARY")
        logger.info("=" * 80)
        
        if results['status'] == 'completed':
            logger.info("✓ Pipeline completed successfully!")
        elif results['status'] == 'completed_with_warnings':
            logger.info(f"⚠ Pipeline completed with {len(results['errors'])} warnings")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        else:
            logger.error("✗ Pipeline failed")
            return
        
        # Display event statistics
        events_df = results['events_df']
        if events_df is not None:
            logger.info("")
            logger.info(f"Total Events Processed: {len(events_df)}")
            logger.info(f"Total Harvest Area Loss: {events_df['harvest_area_loss_ha'].sum():,.0f} ha")
            logger.info(f"Total Production Loss: {events_df['production_loss_kcal'].sum():,.0f} kcal")
            logger.info(f"Magnitude Range: {events_df['magnitude'].min():.2f} - {events_df['magnitude'].max():.2f}")
        
        # Display generated files
        exported_files = results['exported_files']
        if exported_files:
            logger.info("")
            logger.info("Generated Files:")
            logger.info(f"  - CSV files: {len(exported_files.get('csv_files', []))}")
            logger.info(f"  - Figure files: {len(exported_files.get('figure_files', []))}")
            logger.info(f"  - Report files: {len(exported_files.get('report_files', []))}")
        
        logger.info("")
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == '__main__':
    main()

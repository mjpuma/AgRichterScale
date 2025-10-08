#!/usr/bin/env python3
"""
Main execution script for AgriRichter events analysis pipeline.

This script provides a command-line interface to run the complete AgriRichter
analysis pipeline, including data loading, event calculation, visualization
generation, and results export.

Usage:
    python scripts/run_agririchter_analysis.py --crop wheat
    python scripts/run_agririchter_analysis.py --crop allgrain --output outputs/allgrain
    python scripts/run_agririchter_analysis.py --all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import agririchter package
sys.path.insert(0, str(Path(__file__).parent.parent))

from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    Configure logging for the pipeline execution.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Run AgriRichter events analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis for wheat
  python scripts/run_agririchter_analysis.py --crop wheat
  
  # Run analysis for all grains with custom output directory
  python scripts/run_agririchter_analysis.py --crop allgrain --output outputs/allgrain
  
  # Run analysis for all crop types
  python scripts/run_agririchter_analysis.py --all
  
  # Run with debug logging
  python scripts/run_agririchter_analysis.py --crop rice --log-level DEBUG
        """
    )
    
    # Crop selection arguments
    crop_group = parser.add_mutually_exclusive_group(required=True)
    crop_group.add_argument(
        '--crop',
        type=str,
        choices=['wheat', 'rice', 'maize', 'allgrain'],
        help='Crop type to analyze'
    )
    crop_group.add_argument(
        '--all',
        action='store_true',
        help='Run analysis for all crop types (wheat, rice, allgrain)'
    )
    
    # Path arguments
    parser.add_argument(
        '--root-dir',
        type=str,
        default='.',
        help='Root directory containing data files (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: outputs/<crop_type>)'
    )
    
    # SPAM version
    parser.add_argument(
        '--spam-version',
        type=str,
        choices=['2010', '2020'],
        default='2020',
        help='SPAM data version to use (default: 2020)'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Optional log file path'
    )
    
    # Threshold configuration
    parser.add_argument(
        '--use-static-thresholds',
        action='store_true',
        help='Use static thresholds instead of USDA-based dynamic thresholds'
    )
    
    return parser.parse_args()


def run_single_crop_analysis(
    crop_type: str,
    root_dir: str,
    output_dir: str,
    spam_version: str,
    use_dynamic_thresholds: bool
) -> Dict[str, Any]:
    """
    Run analysis pipeline for a single crop type.
    
    Args:
        crop_type: Crop type to analyze
        root_dir: Root directory containing data files
        output_dir: Output directory for results
        spam_version: SPAM data version
        use_dynamic_thresholds: Whether to use dynamic thresholds
        
    Returns:
        Dictionary with pipeline results
        
    Raises:
        Exception: If pipeline execution fails
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"Starting analysis for crop type: {crop_type.upper()}")
    logger.info("=" * 80)
    
    try:
        # Initialize configuration
        logger.info(f"Initializing configuration for {crop_type}...")
        config = Config(
            crop_type=crop_type,
            root_dir=root_dir,
            use_dynamic_thresholds=use_dynamic_thresholds,
            spam_version=spam_version
        )
        
        # Validate required files exist
        logger.info("Validating data files...")
        missing_files = config.validate_files_exist()
        if missing_files:
            logger.warning(f"Some data files are missing: {missing_files}")
            logger.warning("Pipeline will continue but may fail if these files are required")
        
        # Create EventsPipeline instance
        logger.info(f"Creating pipeline with output directory: {output_dir}")
        pipeline = EventsPipeline(config, output_dir)
        
        # Run complete pipeline
        logger.info("Executing complete pipeline...")
        results = pipeline.run_complete_pipeline()
        
        # Check results status
        if results['status'] == 'completed':
            logger.info(f"✓ Analysis for {crop_type} completed successfully!")
        elif results['status'] == 'completed_with_warnings':
            logger.warning(f"⚠ Analysis for {crop_type} completed with warnings")
            logger.warning(f"Warnings: {results['errors']}")
        else:
            logger.error(f"✗ Analysis for {crop_type} failed")
            
        return results
        
    except Exception as e:
        logger.error(f"✗ Analysis for {crop_type} failed with error: {e}", exc_info=True)
        raise


def main() -> int:
    """
    Main entry point for the script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("AgriRichter Events Analysis Pipeline")
    logger.info("=" * 80)
    
    # Determine which crops to analyze
    if args.all:
        crops_to_analyze = ['wheat', 'rice', 'allgrain']
        logger.info("Running analysis for all crop types: wheat, rice, allgrain")
    else:
        crops_to_analyze = [args.crop]
        logger.info(f"Running analysis for crop type: {args.crop}")
    
    # Track results for all crops
    all_results = {}
    failed_crops = []
    
    # Process each crop
    for crop_type in crops_to_analyze:
        try:
            # Determine output directory
            if args.output:
                # Use user-specified output directory
                if len(crops_to_analyze) > 1:
                    # Multiple crops: create subdirectory for each
                    output_dir = Path(args.output) / crop_type
                else:
                    # Single crop: use output directory directly
                    output_dir = Path(args.output)
            else:
                # Default: outputs/<crop_type>
                output_dir = Path('outputs') / crop_type
            
            # Run analysis for this crop
            results = run_single_crop_analysis(
                crop_type=crop_type,
                root_dir=args.root_dir,
                output_dir=str(output_dir),
                spam_version=args.spam_version,
                use_dynamic_thresholds=not args.use_static_thresholds
            )
            
            all_results[crop_type] = results
            
        except Exception as e:
            logger.error(f"Failed to complete analysis for {crop_type}: {e}")
            failed_crops.append(crop_type)
            all_results[crop_type] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Print final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    successful_crops = [c for c in crops_to_analyze if c not in failed_crops]
    
    if successful_crops:
        logger.info(f"✓ Successfully completed: {', '.join(successful_crops)}")
    
    if failed_crops:
        logger.error(f"✗ Failed: {', '.join(failed_crops)}")
    
    # Print statistics for each crop
    for crop_type, results in all_results.items():
        logger.info("")
        logger.info(f"--- {crop_type.upper()} ---")
        
        if results['status'] in ['completed', 'completed_with_warnings']:
            events_df = results.get('events_df')
            if events_df is not None and len(events_df) > 0:
                logger.info(f"  Events processed: {len(events_df)}")
                logger.info(f"  Total harvest area loss: {events_df['harvest_area_loss_ha'].sum():,.0f} ha")
                logger.info(f"  Total production loss: {events_df['production_loss_kcal'].sum():,.0f} kcal")
                logger.info(f"  Magnitude range: {events_df['magnitude'].min():.2f} - {events_df['magnitude'].max():.2f}")
            
            exported_files = results.get('exported_files', {})
            total_files = sum(len(files) for files in exported_files.values())
            logger.info(f"  Files generated: {total_files}")
            
            if results['status'] == 'completed_with_warnings':
                logger.warning(f"  Warnings: {len(results.get('errors', []))}")
        else:
            logger.error(f"  Status: {results['status']}")
            if 'error' in results:
                logger.error(f"  Error: {results['error']}")
    
    # Print next steps
    logger.info("")
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("1. Review the generated figures in the outputs/<crop>/figures/ directories")
    logger.info("2. Check the event results CSV files in outputs/<crop>/data/")
    logger.info("3. Read the summary reports in outputs/<crop>/reports/")
    logger.info("4. Compare results with MATLAB outputs if available")
    logger.info("=" * 80)
    
    # Return exit code
    if failed_crops:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())

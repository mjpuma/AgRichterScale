"""Demo script for DataValidator functionality."""

import logging
from pathlib import Path
import pandas as pd

from agririchter.core.config import Config
from agririchter.validation.data_validator import DataValidator
from agririchter.data.grid_manager import GridManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_spam_validation():
    """Demonstrate SPAM data validation."""
    logger.info("=" * 80)
    logger.info("SPAM DATA VALIDATION DEMO")
    logger.info("=" * 80)
    
    # Initialize configuration
    config = Config(crop_type='wheat', root_dir='./data', spam_version='2020')
    validator = DataValidator(config)
    
    # Load SPAM data using GridManager
    logger.info("Loading SPAM data...")
    grid_manager = GridManager(config)
    
    try:
        production_df = pd.read_csv(config.data_files['production'])
        harvest_df = pd.read_csv(config.data_files['harvest_area'])
        
        logger.info(f"Production data shape: {production_df.shape}")
        logger.info(f"Harvest area data shape: {harvest_df.shape}")
        
        # Validate SPAM data
        logger.info("\nValidating SPAM data...")
        spam_validation = validator.validate_spam_data(production_df, harvest_df)
        
        # Print results
        logger.info(f"\nValidation Status: {'PASS' if spam_validation['valid'] else 'FAIL'}")
        logger.info(f"Errors: {len(spam_validation['errors'])}")
        logger.info(f"Warnings: {len(spam_validation['warnings'])}")
        
        if spam_validation['errors']:
            logger.error("\nErrors found:")
            for error in spam_validation['errors']:
                logger.error(f"  - {error}")
        
        if spam_validation['warnings']:
            logger.warning("\nWarnings:")
            for warning in spam_validation['warnings'][:5]:  # Show first 5
                logger.warning(f"  - {warning}")
            if len(spam_validation['warnings']) > 5:
                logger.warning(f"  ... and {len(spam_validation['warnings']) - 5} more warnings")
        
        # Show production statistics
        if 'production_totals' in spam_validation['statistics']:
            prod_stats = spam_validation['statistics']['production_totals']
            logger.info("\nProduction Statistics:")
            logger.info(f"  Total: {prod_stats.get('total_production_mt', 0):.2e} MT")
            if 'expected_range_mt' in prod_stats:
                min_exp, max_exp = prod_stats['expected_range_mt']
                logger.info(f"  Expected: {min_exp:.2e} - {max_exp:.2e} MT")
        
        return spam_validation
        
    except FileNotFoundError as e:
        logger.error(f"SPAM data files not found: {e}")
        logger.error("Please ensure SPAM 2020 data is available in ./data directory")
        return None
    except Exception as e:
        logger.error(f"Error during SPAM validation: {e}")
        return None


def demo_event_validation():
    """Demonstrate event results validation."""
    logger.info("\n" + "=" * 80)
    logger.info("EVENT RESULTS VALIDATION DEMO")
    logger.info("=" * 80)
    
    # Initialize configuration
    config = Config(crop_type='wheat', root_dir='./data', spam_version='2020')
    validator = DataValidator(config)
    
    # Check if event results exist
    output_path = config.get_output_paths()['event_losses_csv']
    
    if not output_path.exists():
        logger.warning(f"Event results not found at {output_path}")
        logger.warning("Run the events pipeline first to generate event results")
        
        # Create sample data for demonstration
        logger.info("\nCreating sample event data for demonstration...")
        sample_events = pd.DataFrame({
            'event_name': ['GreatFamine', 'DustBowl', 'NoSummer'],
            'harvest_area_loss_ha': [5000000.0, 3000000.0, 2000000.0],
            'production_loss_kcal': [5e14, 3e14, 2e14],
            'magnitude': [4.5, 4.0, 3.8]
        })
        events_df = sample_events
    else:
        logger.info(f"Loading event results from {output_path}")
        events_df = pd.read_csv(output_path)
    
    logger.info(f"Events data shape: {events_df.shape}")
    
    # Validate event results
    logger.info("\nValidating event results...")
    global_production = 2.5e15  # Approximate global wheat production in kcal
    event_validation = validator.validate_event_results(events_df, global_production)
    
    # Print results
    logger.info(f"\nValidation Status: {'PASS' if event_validation['valid'] else 'FAIL'}")
    logger.info(f"Errors: {len(event_validation['errors'])}")
    logger.info(f"Warnings: {len(event_validation['warnings'])}")
    logger.info(f"Suspicious Events: {len(event_validation['suspicious_events'])}")
    
    if event_validation['errors']:
        logger.error("\nErrors found:")
        for error in event_validation['errors']:
            logger.error(f"  - {error}")
    
    if event_validation['warnings']:
        logger.warning("\nWarnings (first 5):")
        for warning in event_validation['warnings'][:5]:
            logger.warning(f"  - {warning}")
    
    # Show event statistics
    if 'event_stats' in event_validation['statistics']:
        stats = event_validation['statistics']['event_stats']
        logger.info("\nEvent Statistics:")
        logger.info(f"  Total Events: {stats.get('total_events', 0)}")
        logger.info(f"  Events with Data: {stats.get('events_with_data', 0)}")
        
        if 'production_loss' in stats:
            pl = stats['production_loss']
            logger.info(f"  Production Loss Range: {pl['min']:.2e} - {pl['max']:.2e} kcal")
            logger.info(f"  Mean Production Loss: {pl['mean']:.2e} kcal")
        
        if 'magnitude' in stats:
            mag = stats['magnitude']
            logger.info(f"  Magnitude Range: {mag['min']:.2f} - {mag['max']:.2f}")
            logger.info(f"  Mean Magnitude: {mag['mean']:.2f}")
    
    return event_validation


def demo_matlab_comparison():
    """Demonstrate MATLAB comparison."""
    logger.info("\n" + "=" * 80)
    logger.info("MATLAB COMPARISON DEMO")
    logger.info("=" * 80)
    
    # Initialize configuration
    config = Config(crop_type='wheat', root_dir='./data', spam_version='2020')
    validator = DataValidator(config)
    
    # Check if event results exist
    output_path = config.get_output_paths()['event_losses_csv']
    
    if not output_path.exists():
        logger.warning("Event results not found. Using sample data...")
        events_df = pd.DataFrame({
            'event_name': ['GreatFamine', 'DustBowl'],
            'production_loss_kcal': [5e14, 3e14]
        })
    else:
        events_df = pd.read_csv(output_path)
    
    # Perform MATLAB comparison
    logger.info("\nComparing with MATLAB reference results...")
    matlab_comparison = validator.compare_with_matlab(events_df)
    
    # Print results
    logger.info(f"\nComparison Available: {matlab_comparison['comparison_available']}")
    
    if matlab_comparison['warnings']:
        logger.info("\nNotes:")
        for warning in matlab_comparison['warnings']:
            logger.info(f"  - {warning}")
    
    if matlab_comparison['comparison_available']:
        stats = matlab_comparison.get('statistics', {})
        logger.info(f"\nEvents Compared: {stats.get('events_compared', 0)}")
        logger.info(f"Events Flagged: {stats.get('events_flagged', 0)}")
        
        if 'percentage_differences' in stats:
            pd_stats = stats['percentage_differences']
            logger.info("\nPercentage Differences:")
            logger.info(f"  Mean: {pd_stats['mean']:.2f}%")
            logger.info(f"  Median: {pd_stats['median']:.2f}%")
            logger.info(f"  Range: {pd_stats['min']:.2f}% - {pd_stats['max']:.2f}%")
    
    return matlab_comparison


def demo_validation_report():
    """Demonstrate comprehensive validation report generation."""
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION REPORT GENERATION DEMO")
    logger.info("=" * 80)
    
    # Initialize configuration
    config = Config(crop_type='wheat', root_dir='./data', spam_version='2020')
    validator = DataValidator(config)
    
    # Run all validations (using sample data if needed)
    spam_validation = None
    event_validation = None
    matlab_comparison = None
    
    try:
        # Try SPAM validation
        production_df = pd.read_csv(config.data_files['production'])
        harvest_df = pd.read_csv(config.data_files['harvest_area'])
        spam_validation = validator.validate_spam_data(production_df, harvest_df)
    except Exception as e:
        logger.warning(f"Could not perform SPAM validation: {e}")
    
    # Create sample event data
    sample_events = pd.DataFrame({
        'event_name': ['GreatFamine', 'DustBowl', 'NoSummer'],
        'harvest_area_loss_ha': [5000000.0, 3000000.0, 2000000.0],
        'production_loss_kcal': [5e14, 3e14, 2e14],
        'magnitude': [4.5, 4.0, 3.8]
    })
    event_validation = validator.validate_event_results(sample_events, 2.5e15)
    matlab_comparison = validator.compare_with_matlab(sample_events)
    
    # Generate comprehensive report
    logger.info("\nGenerating validation report...")
    output_path = Path('./data/outputs/validation_report.txt')
    report = validator.generate_validation_report(
        spam_validation=spam_validation,
        event_validation=event_validation,
        matlab_comparison=matlab_comparison,
        output_path=output_path
    )
    
    # Print report
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 80)
    print(report)
    
    if output_path.exists():
        logger.info(f"\nReport saved to: {output_path}")


def main():
    """Run all validation demos."""
    logger.info("Starting DataValidator Demo")
    logger.info("=" * 80)
    
    # Run demos
    spam_validation = demo_spam_validation()
    event_validation = demo_event_validation()
    matlab_comparison = demo_matlab_comparison()
    demo_validation_report()
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

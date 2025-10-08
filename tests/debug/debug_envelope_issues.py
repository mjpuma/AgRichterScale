#!/usr/bin/env python3
"""
Debug script to investigate envelope visualization issues.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.analysis.agririchter import AgriRichterAnalyzer

def debug_envelope_data():
    """Debug envelope data to understand visualization issues."""
    
    logger = logging.getLogger(__name__)
    
    # Initialize components
    config = Config(crop_type='wheat', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    analyzer = AgriRichterAnalyzer(config)
    
    # Load and process data
    logger.info("Loading production and harvest data...")
    production_df = loader.load_spam_production().head(5000)  # Smaller sample for debugging
    harvest_df = loader.load_spam_harvest_area().head(5000)
    
    # Process data
    crop_production = processor.filter_crop_data(production_df)
    crop_harvest = processor.filter_crop_data(harvest_df)
    
    production_kcal = processor.convert_production_to_kcal(crop_production)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    # Find matching coordinates
    common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
        harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
    )
    
    logger.info(f"Found {len(common_coords)} matching coordinates")
    
    if len(common_coords) < 100:
        logger.error("Insufficient matching coordinates")
        return
    
    # Get matching data
    sample_coords = common_coords.sample(n=min(1000, len(common_coords)), random_state=42)
    prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_coords['grid_code'])]
    harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_coords['grid_code'])]
    
    logger.info(f"Using {len(prod_matched)} cells for envelope calculation")
    
    # Calculate envelope
    logger.info("Calculating envelope...")
    envelope_data = analyzer.envelope_calculator.calculate_hp_envelope(prod_matched, harv_matched)
    
    # Debug envelope data
    logger.info(f"\n=== ENVELOPE DATA ANALYSIS ===")
    logger.info(f"Disruption areas: {len(envelope_data['disruption_areas'])} points")
    logger.info(f"Disruption area range: {envelope_data['disruption_areas'].min():.1f} - {envelope_data['disruption_areas'].max():.1f} km²")
    
    # Check bounds
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    logger.info(f"\nLower bound:")
    logger.info(f"  Harvest: {lower_harvest.min():.1f} - {lower_harvest.max():.1f} km²")
    logger.info(f"  Production: {lower_production.min():.2e} - {lower_production.max():.2e} kcal")
    
    logger.info(f"\nUpper bound:")
    logger.info(f"  Harvest: {upper_harvest.min():.1f} - {upper_harvest.max():.1f} km²")
    logger.info(f"  Production: {upper_production.min():.2e} - {upper_production.max():.2e} kcal")
    
    # Check for zero values
    zero_lower_prod = (lower_production == 0).sum()
    zero_upper_prod = (upper_production == 0).sum()
    logger.info(f"\nZero production values:")
    logger.info(f"  Lower bound: {zero_lower_prod}/{len(lower_production)} ({100*zero_lower_prod/len(lower_production):.1f}%)")
    logger.info(f"  Upper bound: {zero_upper_prod}/{len(upper_production)} ({100*zero_upper_prod/len(upper_production):.1f}%)")
    
    # Check log10 conversion
    valid_lower = lower_production > 0
    valid_upper = upper_production > 0
    
    if valid_lower.any():
        log_lower = np.log10(lower_production[valid_lower])
        logger.info(f"\nLog10 lower production range: {log_lower.min():.2f} - {log_lower.max():.2f}")
    
    if valid_upper.any():
        log_upper = np.log10(upper_production[valid_upper])
        logger.info(f"Log10 upper production range: {log_upper.min():.2f} - {log_upper.max():.2f}")
    
    # Check thresholds
    thresholds = config.get_thresholds()
    logger.info(f"\nThreshold values (log10):")
    for level, value in thresholds.items():
        if value > 0:
            log_threshold = np.log10(value)
            logger.info(f"  {level}: {value:.2e} kcal (log10: {log_threshold:.2f})")
    
    # Create a simple debug plot
    logger.info("\nCreating debug plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Linear scale
    if valid_lower.any() and valid_upper.any():
        ax1.fill_between(lower_harvest[valid_lower], lower_production[valid_lower], 
                        upper_production[valid_upper][:len(lower_harvest[valid_lower])],
                        alpha=0.3, color='lightblue', label='Envelope')
        ax1.plot(lower_harvest[valid_lower], lower_production[valid_lower], 'b-', label='Lower')
        ax1.plot(upper_harvest[valid_upper], upper_production[valid_upper], 'r-', label='Upper')
    
    ax1.set_xlabel('Harvest Area (km²)')
    ax1.set_ylabel('Production (kcal)')
    ax1.set_title('Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    if valid_lower.any() and valid_upper.any():
        ax2.fill_between(lower_harvest[valid_lower], log_lower, 
                        log_upper[:len(log_lower)],
                        alpha=0.3, color='lightblue', label='Envelope')
        ax2.plot(lower_harvest[valid_lower], log_lower, 'b-', label='Lower')
        ax2.plot(upper_harvest[valid_upper], log_upper, 'r-', label='Upper')
        
        # Add threshold lines
        for level, value in thresholds.items():
            if value > 0:
                log_threshold = np.log10(value)
                ax2.axhline(y=log_threshold, linestyle='--', alpha=0.7, label=f'{level}')
    
    ax2.set_xlabel('Harvest Area (km²)')
    ax2.set_ylabel('Production (log₁₀ kcal)')
    ax2.set_title('Log Scale')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Save debug plot
    output_path = Path('test_outputs/debug_envelope_analysis.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Debug plot saved: {output_path}")
    
    return envelope_data

def check_event_data_structure():
    """Check the structure of event data and country/province mapping."""
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n=== EVENT DATA STRUCTURE ANALYSIS ===")
    
    # Check if we have event data files
    event_files = [
        'events/historical_events.csv',
        'events/country_mapping.csv', 
        'events/province_mapping.csv'
    ]
    
    for file_path in event_files:
        if Path(file_path).exists():
            logger.info(f"✅ Found: {file_path}")
        else:
            logger.info(f"❌ Missing: {file_path}")
    
    # Check SPAM data for country/province codes
    config = Config(crop_type='wheat', root_dir='.')
    loader = DataLoader(config)
    
    production_df = loader.load_spam_production().head(1000)
    
    logger.info(f"\nSPAM data columns with country/admin info:")
    admin_columns = [col for col in production_df.columns if any(x in col.upper() for x in ['FIPS', 'ADM', 'NAME', 'COUNTRY'])]
    
    for col in admin_columns:
        unique_vals = production_df[col].nunique()
        sample_vals = production_df[col].dropna().unique()[:5]
        logger.info(f"  {col}: {unique_vals} unique values, sample: {list(sample_vals)}")

def main():
    """Debug envelope and event issues."""
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory
    Path('test_outputs').mkdir(exist_ok=True)
    
    logger.info("🔍 Debugging Envelope and Event Issues")
    logger.info("=" * 60)
    
    # Debug envelope
    envelope_data = debug_envelope_data()
    
    # Check event structure
    check_event_data_structure()
    
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. Check if envelope bounds have valid (non-zero) production values")
    logger.info("2. Verify axis ranges match the actual data ranges")
    logger.info("3. Implement proper country/province mapping for historical events")
    logger.info("4. Create real event data based on geographic boundaries")

if __name__ == "__main__":
    main()
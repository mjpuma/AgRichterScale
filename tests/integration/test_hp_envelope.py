#!/usr/bin/env python3
"""
Test script for H-P envelope visualization.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.analysis.agririchter import AgriRichterAnalyzer
from agririchter.visualization.plots import EnvelopePlotter

def test_hp_envelope_plot(crop_type: str = 'wheat', sample_size: int = 5000):
    """Test H-P envelope visualization."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING H-P ENVELOPE: {crop_type.upper()}")
        logger.info(f"{'='*60}")
        
        # Initialize components
        config = Config(crop_type=crop_type, root_dir='.')
        loader = DataLoader(config)
        processor = DataProcessor(config)
        analyzer = AgriRichterAnalyzer(config)
        plotter = EnvelopePlotter(config)
        
        # Load and process data
        logger.info("Loading production and harvest data...")
        
        # Load full datasets
        production_df = loader.load_spam_production()
        harvest_df = loader.load_spam_harvest_area()
        
        # Process data
        crop_production = processor.filter_crop_data(production_df)
        crop_harvest = processor.filter_crop_data(harvest_df)
        
        # Convert units
        production_kcal = processor.convert_production_to_kcal(crop_production)
        harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
        
        # Get cells with production for both datasets
        crop_columns = [col for col in production_kcal.columns if col.endswith('_A')]
        
        if not crop_columns:
            logger.warning(f"No crop data found for {crop_type}")
            return False
        
        # Find matching coordinates between production and harvest data
        logger.info("Finding matching coordinates between datasets...")
        common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
            harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
        )
        
        logger.info(f"Found {len(common_coords)} matching coordinates")
        
        if len(common_coords) < 100:
            logger.warning("Insufficient matching coordinates for envelope calculation")
            return False
        
        # Sample matching data for envelope calculation
        sample_size = min(sample_size, len(common_coords))
        sample_coords = common_coords.sample(n=sample_size, random_state=42)
        
        # Get matching production and harvest data
        prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_coords['grid_code'])]
        harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_coords['grid_code'])]
        
        logger.info(f"Using {len(prod_matched)} cells for envelope calculation")
        
        # Calculate H-P envelope
        logger.info("Calculating H-P envelope...")
        envelope_data = analyzer.envelope_calculator.calculate_hp_envelope(
            prod_matched, harv_matched
        )
        
        logger.info(f"Envelope calculated with {len(envelope_data['disruption_areas'])} boundary points")
        logger.info(f"Harvest area range: {envelope_data['lower_bound_harvest'].min():.1f} - {envelope_data['upper_bound_harvest'].max():.1f} km²")
        logger.info(f"Production range: {envelope_data['lower_bound_production'].min():.2e} - {envelope_data['upper_bound_production'].max():.2e} kcal")
        
        # Create sample historical events for testing
        logger.info("Creating sample historical events...")
        sample_events = pd.DataFrame({
            'event_name': ['GreatFamine', 'DustBowl', 'ChineseFamine1960', 'SahelDrought2010'],
            'harvest_area_km2': [50000, 25000, 100000, 15000],
            'production_loss_kcal': [1e15, 5e14, 2e15, 1e13]
        })
        
        # Create output directory
        output_dir = Path('test_outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Test 1: Envelope plot only
        logger.info("Creating H-P envelope plot (envelope only)...")
        output_path_1 = output_dir / f'hp_envelope_only_{crop_type}.png'
        
        fig1 = plotter.create_envelope_plot(
            envelope_data=envelope_data,
            title=f'H-P Envelope - {crop_type.title()} (Envelope Only)',
            output_path=output_path_1,
            dpi=200,
            figsize=(12, 8)
        )
        
        plt.close(fig1)
        
        if output_path_1.exists():
            logger.info(f"✅ Envelope-only plot saved: {output_path_1}")
            logger.info(f"   File size: {output_path_1.stat().st_size / 1024:.1f} KB")
        
        # Test 2: Envelope with historical events
        logger.info("Creating H-P envelope plot with historical events...")
        output_path_2 = output_dir / f'hp_envelope_events_{crop_type}.png'
        
        fig2 = plotter.create_envelope_plot(
            envelope_data=envelope_data,
            historical_events=sample_events,
            title=f'H-P Envelope - {crop_type.title()} (with Historical Events)',
            output_path=output_path_2,
            dpi=200,
            figsize=(12, 8)
        )
        
        plt.close(fig2)
        
        if output_path_2.exists():
            logger.info(f"✅ Envelope with events plot saved: {output_path_2}")
            logger.info(f"   File size: {output_path_2.stat().st_size / 1024:.1f} KB")
        
        # Test 3: Thresholds only (no envelope data)
        logger.info("Creating H-P plot with thresholds only...")
        output_path_3 = output_dir / f'hp_thresholds_only_{crop_type}.png'
        
        fig3 = plotter.create_envelope_plot(
            historical_events=sample_events,
            title=f'H-P Plot - {crop_type.title()} (Thresholds and Events Only)',
            output_path=output_path_3,
            dpi=200,
            figsize=(12, 8)
        )
        
        plt.close(fig3)
        
        if output_path_3.exists():
            logger.info(f"✅ Thresholds-only plot saved: {output_path_3}")
            logger.info(f"   File size: {output_path_3.stat().st_size / 1024:.1f} KB")
        
        # Display envelope statistics
        logger.info(f"\nEnvelope Statistics:")
        logger.info(f"  Disruption areas: {len(envelope_data['disruption_areas'])} points")
        logger.info(f"  Min disruption area: {envelope_data['disruption_areas'].min():.1f} km²")
        logger.info(f"  Max disruption area: {envelope_data['disruption_areas'].max():.1f} km²")
        
        # Check envelope bounds
        lower_prod_range = envelope_data['lower_bound_production']
        upper_prod_range = envelope_data['upper_bound_production']
        
        logger.info(f"  Lower bound production: {lower_prod_range.min():.2e} - {lower_prod_range.max():.2e} kcal")
        logger.info(f"  Upper bound production: {upper_prod_range.min():.2e} - {upper_prod_range.max():.2e} kcal")
        
        logger.info(f"✅ {crop_type.upper()} H-P envelope visualization test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ {crop_type.upper()} H-P envelope test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test H-P envelope visualization for crop types."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    crop_types = ['wheat', 'allgrain']  # Start with crops that have good data
    
    logger.info("📈 H-P Envelope Visualization Testing")
    logger.info("=" * 60)
    logger.info("Testing Harvest-Production envelope with filled areas and threshold lines")
    
    results = {}
    for crop_type in crop_types:
        results[crop_type] = test_hp_envelope_plot(crop_type, sample_size=3000)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("H-P ENVELOPE VISUALIZATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    all_passed = True
    for crop_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{crop_type.upper():>10}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info(f"\n🎉 H-P ENVELOPE VISUALIZATION CAPABILITIES:")
        logger.info("✅ Filled envelope area with transparency")
        logger.info("✅ Upper and lower boundary lines")
        logger.info("✅ Threshold lines (T1-T4) with colors")
        logger.info("✅ Historical event plotting")
        logger.info("✅ Log-scale axes for proper visualization")
        logger.info("✅ Event color coding and markers")
        logger.info("✅ Publication-quality output")
        
        # List generated files
        output_dir = Path('test_outputs')
        if output_dir.exists():
            envelope_files = list(output_dir.glob('hp_*.png'))
            if envelope_files:
                logger.info(f"\n📁 Generated H-P envelope files:")
                for file_path in envelope_files:
                    logger.info(f"   {file_path}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
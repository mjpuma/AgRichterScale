#!/usr/bin/env python3
"""
Test script to verify AgRichter analysis engine with SPAM 2020 data.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add the agrichter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agrichter.core.config import Config
from agrichter.data.loader import DataLoader
from agrichter.processing.processor import DataProcessor
from agrichter.analysis.agrichter import AgRichterAnalyzer
from agrichter.analysis.envelope import HPEnvelopeCalculator

def test_analysis_engine():
    """Test the AgRichter analysis engine."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration for wheat
        logger.info("Initializing configuration for wheat analysis...")
        config = Config(crop_type='wheat', root_dir='.')
        
        # Initialize components
        loader = DataLoader(config)
        processor = DataProcessor(config)
        analyzer = AgRichterAnalyzer(config)
        
        # Load and process a sample of data (to keep test fast)
        logger.info("Loading sample data...")
        production_df = loader.load_spam_production()
        harvest_df = loader.load_spam_harvest_area()
        
        # Use a smaller sample for testing
        sample_size = 5000
        production_sample = production_df.head(sample_size)
        harvest_sample = harvest_df.head(sample_size)
        
        logger.info(f"Processing sample of {sample_size} grid cells...")
        
        # Filter and convert data
        wheat_production = processor.filter_crop_data(production_sample)
        wheat_harvest = processor.filter_crop_data(harvest_sample)
        
        # Convert units
        production_kcal = processor.convert_production_to_kcal(wheat_production)
        harvest_km2 = processor.convert_harvest_area_to_km2(wheat_harvest)
        
        logger.info("✅ Data processing complete")
        
        # Test H-P envelope calculation
        logger.info("Testing H-P envelope calculation...")
        envelope_calculator = HPEnvelopeCalculator(config)
        
        # For envelope calculation, we need matching coordinates
        # Let's create a simple test with the same coordinates
        common_coords = production_kcal[['grid_code', 'x', 'y']].merge(
            harvest_km2[['grid_code', 'x', 'y']], on=['grid_code', 'x', 'y']
        )
        
        if len(common_coords) > 100:  # Need sufficient data points
            # Get matching data
            prod_matched = production_kcal[production_kcal['grid_code'].isin(common_coords['grid_code'])]
            harv_matched = harvest_km2[harvest_km2['grid_code'].isin(common_coords['grid_code'])]
            
            # Take first 1000 matching points for speed
            prod_matched = prod_matched.head(1000)
            harv_matched = harv_matched.head(1000)
            
            envelope_data = envelope_calculator.calculate_hp_envelope(prod_matched, harv_matched)
            
            logger.info("✅ H-P envelope calculation successful!")
            logger.info(f"   Envelope points: {len(envelope_data['disruption_areas'])}")
            logger.info(f"   Production range: {envelope_data['lower_bound_production'].min():.2e} - {envelope_data['upper_bound_production'].max():.2e} kcal")
            
            # Validate envelope data
            envelope_calculator.validate_envelope_data(envelope_data)
            logger.info("✅ Envelope data validation passed")
            
        else:
            logger.warning("⚠️  Insufficient matching coordinates for envelope calculation")
        
        # Test AgRichter magnitude calculation
        logger.info("Testing AgRichter magnitude calculation...")
        
        # Test with sample harvest areas
        test_areas = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])  # km²
        magnitudes = analyzer.calculate_magnitude(test_areas)
        
        logger.info("✅ Magnitude calculation successful!")
        for area, mag in zip(test_areas, magnitudes):
            logger.info(f"   {area:,.0f} km² → Magnitude {mag:.2f}")
        
        # Test severity classification
        logger.info("Testing severity classification...")
        
        # Get thresholds for wheat
        thresholds = config.get_thresholds()
        test_losses = np.array([
            thresholds['T1'] * 0.5,  # Below T1
            thresholds['T1'] * 1.5,  # T1
            thresholds['T2'] * 1.5,  # T2
            thresholds['T3'] * 1.5,  # T3
            thresholds['T4'] * 1.5   # T4
        ])
        
        classifications = analyzer.classify_event_severity(test_losses)
        
        logger.info("✅ Severity classification successful!")
        for loss, classification in zip(test_losses, classifications):
            logger.info(f"   {loss:.2e} kcal → {classification}")
        
        # Test Richter scale data creation
        logger.info("Testing Richter scale data creation...")
        
        scale_data = analyzer.create_richter_scale_data(
            min_magnitude=0.0, max_magnitude=6.0, n_points=100
        )
        
        logger.info("✅ Richter scale data creation successful!")
        logger.info(f"   Scale points: {len(scale_data['magnitudes'])}")
        logger.info(f"   Magnitude range: {scale_data['magnitudes'].min():.1f} - {scale_data['magnitudes'].max():.1f}")
        logger.info(f"   Production range: {scale_data['production_kcal'].min():.2e} - {scale_data['production_kcal'].max():.2e} kcal")
        
        # Test with mock historical events
        logger.info("Testing historical events processing...")
        
        # Create mock historical events data
        mock_events = pd.DataFrame({
            'event': ['Event1', 'Event2', 'Event3', 'Event4'],
            'harvest_area_loss_ha': [1000, 50000, 500000, 2000000],  # hectares
            'production_loss_kcal': [
                thresholds['T1'] * 0.8,  # Below T1
                thresholds['T2'] * 1.2,  # T2
                thresholds['T3'] * 1.5,  # T3
                thresholds['T4'] * 2.0   # T4
            ]
        })
        
        processed_events = analyzer.process_historical_events(mock_events)
        
        logger.info("✅ Historical events processing successful!")
        logger.info(f"   Processed {len(processed_events)} events")
        
        for _, event in processed_events.iterrows():
            logger.info(f"   {event['event']}: Magnitude {event['magnitude']:.2f}, Severity {event['severity_class']}")
        
        # Test event distribution analysis
        event_analysis = analyzer.analyze_event_distribution(processed_events)
        
        logger.info("✅ Event distribution analysis successful!")
        logger.info(f"   Severity distribution: {event_analysis['severity_distribution']}")
        
        # Test complete analysis (with smaller dataset)
        logger.info("Testing complete analysis pipeline...")
        
        if len(common_coords) > 100:
            complete_results = analyzer.run_complete_analysis(
                prod_matched, harv_matched, mock_events
            )
            
            logger.info("✅ Complete analysis pipeline successful!")
            logger.info(f"   Analysis components: {list(complete_results.keys())}")
            
            # Generate analysis report
            report = analyzer.create_analysis_report(complete_results)
            logger.info("✅ Analysis report generated")
            
            # Save report to file
            with open('analysis_report.txt', 'w') as f:
                f.write(report)
            logger.info("📄 Analysis report saved to 'analysis_report.txt'")
        
        logger.info("🎉 AgRichter analysis engine test SUCCESSFUL!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analysis_engine()
    sys.exit(0 if success else 1)
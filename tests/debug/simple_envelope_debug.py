#!/usr/bin/env python3
"""
Simple debug for envelope issues.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.processing.processor import DataProcessor
from agririchter.analysis.agririchter import AgriRichterAnalyzer

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Quick envelope check
    config = Config(crop_type='wheat', root_dir='.')
    loader = DataLoader(config)
    processor = DataProcessor(config)
    analyzer = AgriRichterAnalyzer(config)
    
    # Load small sample
    production_df = loader.load_spam_production().head(2000)
    harvest_df = loader.load_spam_harvest_area().head(2000)
    
    # Process
    crop_production = processor.filter_crop_data(production_df)
    crop_harvest = processor.filter_crop_data(harvest_df)
    production_kcal = processor.convert_production_to_kcal(crop_production)
    harvest_km2 = processor.convert_harvest_area_to_km2(crop_harvest)
    
    # Find matching
    common_coords = production_kcal[['grid_code']].merge(harvest_km2[['grid_code']], on='grid_code')
    logger.info(f"Matching coordinates: {len(common_coords)}")
    
    if len(common_coords) > 100:
        # Get sample
        sample_codes = common_coords['grid_code'].sample(n=500, random_state=42)
        prod_matched = production_kcal[production_kcal['grid_code'].isin(sample_codes)]
        harv_matched = harvest_km2[harvest_km2['grid_code'].isin(sample_codes)]
        
        # Calculate envelope
        envelope_data = analyzer.envelope_calculator.calculate_hp_envelope(prod_matched, harv_matched)
        
        # Check data
        logger.info(f"Envelope points: {len(envelope_data['disruption_areas'])}")
        
        lower_prod = envelope_data['lower_bound_production']
        upper_prod = envelope_data['upper_bound_production']
        
        # Check for zeros
        zero_lower = (lower_prod == 0).sum()
        zero_upper = (upper_prod == 0).sum()
        
        logger.info(f"Zero lower production: {zero_lower}/{len(lower_prod)} ({100*zero_lower/len(lower_prod):.1f}%)")
        logger.info(f"Zero upper production: {zero_upper}/{len(upper_prod)} ({100*zero_upper/len(upper_prod):.1f}%)")
        
        # Check ranges
        if (lower_prod > 0).any():
            valid_lower = lower_prod[lower_prod > 0]
            logger.info(f"Valid lower production range: {valid_lower.min():.2e} - {valid_lower.max():.2e} kcal")
            logger.info(f"Valid lower log10 range: {np.log10(valid_lower.min()):.2f} - {np.log10(valid_lower.max()):.2f}")
        
        if (upper_prod > 0).any():
            valid_upper = upper_prod[upper_prod > 0]
            logger.info(f"Valid upper production range: {valid_upper.min():.2e} - {valid_upper.max():.2e} kcal")
            logger.info(f"Valid upper log10 range: {np.log10(valid_upper.min()):.2f} - {np.log10(valid_upper.max()):.2f}")
        
        # Check thresholds
        thresholds = config.get_thresholds()
        logger.info(f"Threshold T1 log10: {np.log10(thresholds['T1']):.2f}")
        logger.info(f"Threshold T4 log10: {np.log10(thresholds['T4']):.2f}")
        
        # Check axis ranges from constants
        from agririchter.core.constants import PRODUCTION_RANGES
        if 'wheat' in PRODUCTION_RANGES:
            y_min, y_max = PRODUCTION_RANGES['wheat']
            logger.info(f"Expected Y-axis range: {y_min} - {y_max}")
        
        # Check if data falls within expected range
        if (lower_prod > 0).any() and (upper_prod > 0).any():
            data_min = min(np.log10(valid_lower.min()), np.log10(valid_upper.min()))
            data_max = max(np.log10(valid_lower.max()), np.log10(valid_upper.max()))
            logger.info(f"Actual data log10 range: {data_min:.2f} - {data_max:.2f}")
            
            if 'wheat' in PRODUCTION_RANGES:
                y_min, y_max = PRODUCTION_RANGES['wheat']
                if data_max < y_min or data_min > y_max:
                    logger.warning(f"⚠️  DATA OUTSIDE EXPECTED RANGE! Data: {data_min:.2f}-{data_max:.2f}, Expected: {y_min}-{y_max}")
                else:
                    logger.info(f"✅ Data within expected range")

if __name__ == "__main__":
    main()
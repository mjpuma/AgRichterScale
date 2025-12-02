#!/usr/bin/env python3
"""
Validation script for USA and China national multi-tier analysis.

This script validates that the USA and China analysis is complete and working
with the actual SPAM data files in the correct locations.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('usa_china_validation')


def validate_usa_china_analysis():
    """Validate USA and China national analysis with correct data paths."""
    
    logger.info("🚀 Validating USA and China National Multi-Tier Analysis")
    logger.info("=" * 80)
    
    # Test crops
    test_crops = ['wheat', 'maize', 'rice']
    test_countries = ['USA', 'CHN']
    
    validation_results = {
        'countries_validated': [],
        'crops_validated': [],
        'analysis_results': {},
        'validation_status': {},
        'width_reductions': {},
        'policy_insights': {}
    }
    
    for crop_type in test_crops:
        logger.info(f"\n📊 VALIDATING CROP: {crop_type.upper()}")
        logger.info("-" * 60)
        
        try:
            # Initialize configuration with correct root directory (current directory)
            config = Config(crop_type=crop_type, root_dir='.')
            
            # Verify SPAM data files exist
            file_paths = config.get_file_paths()
            production_file = file_paths['production']
            harvest_file = file_paths['harvest_area']
            
            logger.info(f"Production file: {production_file}")
            logger.info(f"Harvest file: {harvest_file}")
            
            if not production_file.exists():
                logger.error(f"✗ Production file not found: {production_file}")
                continue
                
            if not harvest_file.exists():
                logger.error(f"✗ Harvest file not found: {harvest_file}")
                continue
                
            logger.info("✓ SPAM data files found")
            
            # Initialize components
            grid_manager = GridDataManager(config)
            spatial_mapper = SpatialMapper(config, grid_manager)
            country_manager = CountryBoundaryManager(config, spatial_mapper, grid_manager)
            multi_tier_engine = MultiTierEnvelopeEngine(config)
            national_analyzer = NationalEnvelopeAnalyzer(config, country_manager, multi_tier_engine)
            
            logger.info(f"✓ Initialized components for {crop_type}")
            
            # Test individual country analyses
            country_results = {}
            
            for country_code in test_countries:
                logger.info(f"\n🌍 Validating {country_code} for {crop_type}")
                
                try:
                    # Validate country data coverage first
                    validation = country_manager.validate_country_data_coverage(country_code)
                    logger.info(f"   Data validation: {'✓' if validation['valid'] else '✗'}")
                    
                    if validation['valid']:
                        logger.info(f"   Production cells: {validation['production_cells']:,}")
                        logger.info(f"   Meets minimum: {'✓' if validation['meets_minimum'] else '✗'}")
                        
                        # Perform national analysis
                        analysis_results = national_analyzer.analyze_national_capacity(country_code)
                        country_results[country_code] = analysis_results
                        
                        # Log key results
                        logger.info(f"   ✓ Analysis completed successfully")
                        logger.info(f"   Total production: {analysis_results.national_statistics['total_production_mt']:,.0f} MT")
                        logger.info(f"   Average yield: {analysis_results.national_statistics['average_yield_mt_per_ha']:.2f} MT/ha")
                        
                        # Log tier effectiveness
                        width_reductions = {}
                        for tier_name in analysis_results.multi_tier_results.tier_results.keys():
                            width_reduction = analysis_results.get_width_reduction(tier_name)
                            if width_reduction is not None:
                                width_reductions[tier_name] = width_reduction
                                logger.info(f"   {tier_name} tier width reduction: {width_reduction:.1f}%")
                        
                        # Store results
                        validation_results['analysis_results'][f"{country_code}_{crop_type}"] = {
                            'total_production_mt': analysis_results.national_statistics['total_production_mt'],
                            'average_yield_mt_per_ha': analysis_results.national_statistics['average_yield_mt_per_ha'],
                            'width_reductions': width_reductions,
                            'validation_passed': analysis_results.validation_results.get('overall_valid', False)
                        }
                        
                        # Validation status
                        validation_status = analysis_results.validation_results.get('overall_valid', False)
                        logger.info(f"   Validation status: {'✓' if validation_status else '✗'}")
                        
                        if country_code not in validation_results['countries_validated']:
                            validation_results['countries_validated'].append(country_code)
                            
                    else:
                        logger.warning(f"   ✗ Insufficient data coverage for {country_code}")
                        
                except Exception as e:
                    logger.error(f"   ✗ Failed to analyze {country_code}: {str(e)}")
                    
            # Mark crop as validated if we got results
            if country_results:
                validation_results['crops_validated'].append(crop_type)
                logger.info(f"✓ Completed {crop_type} validation")
            else:
                logger.warning(f"✗ No successful analyses for {crop_type}")
                
        except Exception as e:
            logger.error(f"✗ Failed to initialize {crop_type}: {str(e)}")
    
    # Generate final validation report
    logger.info("\n" + "=" * 80)
    logger.info("📋 FINAL VALIDATION REPORT")
    logger.info("=" * 80)
    
    # Check acceptance criteria
    usa_validated = 'USA' in validation_results['countries_validated']
    china_validated = 'CHN' in validation_results['countries_validated']
    wheat_validated = 'wheat' in validation_results['crops_validated']
    maize_validated = 'maize' in validation_results['crops_validated']
    rice_validated = 'rice' in validation_results['crops_validated']
    
    # Check width reductions achieved
    width_reductions_achieved = False
    for analysis_key, analysis_data in validation_results['analysis_results'].items():
        if analysis_data['width_reductions']:
            for tier, reduction in analysis_data['width_reductions'].items():
                if tier == 'commercial' and reduction > 0:
                    width_reductions_achieved = True
                    break
    
    logger.info("Acceptance Criteria:")
    logger.info(f"  USA analysis completed: {'✓' if usa_validated else '✗'}")
    logger.info(f"  China analysis completed: {'✓' if china_validated else '✗'}")
    logger.info(f"  Wheat analysis completed: {'✓' if wheat_validated else '✗'}")
    logger.info(f"  Maize analysis completed: {'✓' if maize_validated else '✗'}")
    logger.info(f"  Rice analysis completed: {'✓' if rice_validated else '✗'}")
    logger.info(f"  Width reductions achieved: {'✓' if width_reductions_achieved else '✗'}")
    
    # Overall validation status
    all_criteria_met = all([
        usa_validated, china_validated, wheat_validated, 
        maize_validated, rice_validated, width_reductions_achieved
    ])
    
    logger.info(f"\nOverall validation status: {'✓ PASSED' if all_criteria_met else '✗ FAILED'}")
    
    # Export results
    output_file = Path('usa_china_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    logger.info(f"Results exported to: {output_file}")
    
    if all_criteria_met:
        logger.info("\n🎉 USA and China analysis validation SUCCESSFUL!")
        logger.info("Task: 'USA and China analysis complete and validated' - COMPLETED")
    else:
        logger.warning("\n⚠️ USA and China analysis validation INCOMPLETE")
        logger.warning("Some acceptance criteria not met - review implementation")
    
    return all_criteria_met


if __name__ == "__main__":
    success = validate_usa_china_analysis()
    sys.exit(0 if success else 1)
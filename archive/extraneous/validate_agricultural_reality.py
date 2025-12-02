#!/usr/bin/env python3
"""
Agricultural Reality Validation Script

This script validates that multi-tier envelope results align with agricultural reality
by checking yield ranges, production totals, spatial patterns, and tier stratification
against known agricultural benchmarks.

Implements validation requirements V2.1-V2.4 from the multi-tier envelope integration spec.
"""

import sys
import logging
from pathlib import Path
import json
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer
from agririchter.validation.agricultural_reality_validator import (
    AgriculturalRealityValidator, 
    validate_agricultural_reality_comprehensive
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agricultural_reality_validation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_agricultural_reality_validation():
    """Run comprehensive agricultural reality validation."""
    logger = setup_logging()
    logger.info("Starting Agricultural Reality Validation")
    
    # Output directory
    output_dir = Path('agricultural_reality_validation_output')
    output_dir.mkdir(exist_ok=True)
    
    # Countries and crops to validate
    countries = ['USA', 'CHN']
    crops = ['wheat', 'maize', 'rice']
    
    validation_results = {}
    analysis_results = {}
    
    for crop in crops:
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATING AGRICULTURAL REALITY: {crop.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize components with correct root directory
            config = Config(crop_type=crop, root_dir='.')
            grid_manager = GridDataManager(config)
            
            if not grid_manager.is_loaded():
                logger.info(f"Loading SPAM data for {crop}...")
                grid_manager.load_spam_data()
            
            spatial_mapper = SpatialMapper(config, grid_manager)
            country_manager = CountryBoundaryManager(config, spatial_mapper, grid_manager)
            multi_tier_engine = MultiTierEnvelopeEngine(config)
            national_analyzer = NationalEnvelopeAnalyzer(config, country_manager, multi_tier_engine)
            
            for country in countries:
                analysis_key = f"{country}_{crop}"
                logger.info(f"\nValidating {country} {crop} analysis...")
                
                try:
                    # Check data coverage first
                    coverage_validation = country_manager.validate_country_data_coverage(country)
                    
                    if not coverage_validation['valid']:
                        logger.warning(f"Insufficient data coverage for {country} {crop}: {coverage_validation}")
                        continue
                    
                    # Run national analysis
                    start_time = time.time()
                    national_results = national_analyzer.analyze_national_capacity(country)
                    
                    # Get multi-tier results from the national results
                    multi_tier_results = national_results.multi_tier_results
                    
                    analysis_time = time.time() - start_time
                    
                    # Store results for validation
                    analysis_results[analysis_key] = (national_results, multi_tier_results)
                    
                    # Log key metrics
                    logger.info(f"  Analysis completed in {analysis_time:.1f}s")
                    logger.info(f"  Total production: {national_results.national_statistics['total_production_mt']/1e6:.1f} MMT")
                    logger.info(f"  Average yield: {national_results.national_statistics['average_yield_mt_per_ha']:.2f} MT/ha")
                    
                    width_reduction = multi_tier_results.get_width_reduction('commercial')
                    if width_reduction is not None:
                        logger.info(f"  Commercial tier width reduction: {width_reduction:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {country} {crop}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to process {crop}: {e}")
            continue
    
    # Run comprehensive agricultural reality validation
    logger.info(f"\n{'='*60}")
    logger.info("RUNNING AGRICULTURAL REALITY VALIDATION")
    logger.info(f"{'='*60}")
    
    if analysis_results:
        validation_results = validate_agricultural_reality_comprehensive(
            analysis_results, output_dir
        )
        
        # Log validation summary
        total_analyses = len(validation_results)
        valid_analyses = sum(1 for report in validation_results.values() if report.overall_valid)
        
        logger.info(f"\nValidation Summary:")
        logger.info(f"  Total analyses: {total_analyses}")
        logger.info(f"  Passed validation: {valid_analyses}")
        logger.info(f"  Success rate: {valid_analyses/total_analyses*100:.1f}%")
        
        # Log detailed results
        for analysis_key, report in validation_results.items():
            country, crop = analysis_key.split('_', 1)
            status = "PASS" if report.overall_valid else "REVIEW"
            logger.info(f"  {country} {crop}: {status}")
            
            if report.warnings:
                for warning in report.warnings:
                    logger.warning(f"    - {warning}")
        
        # Save validation results to JSON
        json_results = {}
        for analysis_key, report in validation_results.items():
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            json_results[analysis_key] = convert_numpy_types({
                'overall_valid': report.overall_valid,
                'yield_validation': report.yield_validation,
                'production_validation': report.production_validation,
                'spatial_validation': report.spatial_validation,
                'tier_validation': report.tier_validation,
                'warnings': report.warnings,
                'recommendations': report.recommendations
            })
        
        json_path = output_dir / 'agricultural_reality_validation_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"\nValidation results saved to:")
        logger.info(f"  Report: {output_dir / 'agricultural_reality_validation_report.md'}")
        logger.info(f"  Data: {json_path}")
        
        # Overall status
        if valid_analyses == total_analyses:
            logger.info("\n✅ ALL ANALYSES ALIGN WITH AGRICULTURAL REALITY")
            logger.info("The multi-tier envelope system produces results consistent with known agricultural patterns.")
        else:
            logger.warning(f"\n⚠️ {total_analyses - valid_analyses} ANALYSES NEED REVIEW")
            logger.warning("Some results may not fully align with agricultural reality benchmarks.")
        
        return validation_results
    
    else:
        logger.error("No analysis results available for validation")
        return {}


if __name__ == '__main__':
    try:
        validation_results = run_agricultural_reality_validation()
        
        # Exit with appropriate code
        if validation_results:
            all_valid = all(report.overall_valid for report in validation_results.values())
            sys.exit(0 if all_valid else 1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logging.error(f"Agricultural reality validation failed: {e}")
        sys.exit(3)
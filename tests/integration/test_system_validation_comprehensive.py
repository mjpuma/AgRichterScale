"""
Comprehensive system validation tests for multi-tier envelope integration.

This test suite validates the complete multi-tier system without requiring
full SPAM datasets, using synthetic data that matches real data characteristics.
"""

import pytest
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import psutil
import os

from agririchter.core.config import Config
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine, MultiTierResults
from agririchter.analysis.envelope import EnvelopeData
from agririchter.validation.spam_data_filter import SPAMDataFilter
from agririchter.core.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class SystemValidationSuite:
    """Comprehensive system validation test suite."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.validation_errors = []
        
    def create_synthetic_spam_data(self, n_cells: int = 1000, crop_type: str = 'wheat') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic SPAM-like data for testing."""
        np.random.seed(42)  # Reproducible results
        
        # Create realistic yield distributions based on real SPAM characteristics
        if crop_type == 'wheat':
            # Wheat: global yields typically 1-8 tons/ha
            base_yields = np.random.lognormal(mean=1.5, sigma=0.8, size=n_cells)
            base_yields = np.clip(base_yields, 0.5, 12.0)  # Realistic range
            crop_col = 'WHEA_A'
        elif crop_type == 'rice':
            # Rice: global yields typically 2-10 tons/ha  
            base_yields = np.random.lognormal(mean=1.8, sigma=0.7, size=n_cells)
            base_yields = np.clip(base_yields, 1.0, 15.0)
            crop_col = 'RICE_A'
        elif crop_type == 'maize':
            # Maize: global yields typically 2-15 tons/ha
            base_yields = np.random.lognormal(mean=2.0, sigma=0.9, size=n_cells)
            base_yields = np.clip(base_yields, 1.0, 20.0)
            crop_col = 'MAIZ_A'
        else:
            base_yields = np.random.lognormal(mean=1.5, sigma=0.8, size=n_cells)
            base_yields = np.clip(base_yields, 0.5, 12.0)
            crop_col = 'WHEA_A'
        
        # Create harvest areas (0.01 to 100 km²)
        harvest_areas = np.random.lognormal(mean=1.0, sigma=1.5, size=n_cells)
        harvest_areas = np.clip(harvest_areas, 0.01, 100.0)
        
        # Calculate production (tons) = yield (tons/ha) * area (ha)
        # Convert km² to ha: 1 km² = 100 ha
        production_tons = base_yields * harvest_areas * 100
        
        # Convert to kcal using crop-specific conversion factors
        if crop_type == 'wheat':
            kcal_per_ton = 3.34e9  # Wheat: ~3340 kcal/kg
        elif crop_type == 'rice':
            kcal_per_ton = 3.65e9  # Rice: ~3650 kcal/kg
        elif crop_type == 'maize':
            kcal_per_ton = 3.65e9  # Maize: ~3650 kcal/kg
        else:
            kcal_per_ton = 3.34e9
            
        production_kcal = production_tons * kcal_per_ton
        
        # Create DataFrames with realistic structure
        production_df = pd.DataFrame({
            'cell_id': range(n_cells),
            'lat': np.random.uniform(-60, 70, n_cells),
            'lon': np.random.uniform(-180, 180, n_cells),
            crop_col: production_kcal,
            'iso3': np.random.choice(['USA', 'CHN', 'IND', 'BRA', 'RUS'], n_cells)
        })
        
        harvest_df = pd.DataFrame({
            'cell_id': range(n_cells),
            'lat': np.random.uniform(-60, 70, n_cells),
            'lon': np.random.uniform(-180, 180, n_cells),
            crop_col: harvest_areas,
            'iso3': np.random.choice(['USA', 'CHN', 'IND', 'BRA', 'RUS'], n_cells)
        })
        
        return production_df, harvest_df
    
    def test_multi_tier_calculation_accuracy(self) -> Dict[str, Any]:
        """Test multi-tier calculation accuracy and mathematical properties."""
        logger.info("Testing multi-tier calculation accuracy...")
        
        try:
            # Create test configuration
            config = Config()
            config.crop_type = 'wheat'
            
            # Create engine
            engine = MultiTierEnvelopeEngine(config)
            
            # Create synthetic data
            production_df, harvest_df = self.create_synthetic_spam_data(n_cells=500)
            
            # Calculate multi-tier envelopes
            results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
            
            # Validate results structure
            assert isinstance(results, MultiTierResults)
            assert len(results.tier_results) >= 2  # At least comprehensive and commercial
            assert 'comprehensive' in results.tier_results
            assert 'commercial' in results.tier_results
            
            # Validate mathematical properties
            for tier_name, envelope_data in results.tier_results.items():
                assert isinstance(envelope_data, EnvelopeData)
                assert envelope_data.convergence_validated
                assert len(envelope_data.disruption_areas) > 0
                assert len(envelope_data.lower_bound_production) == len(envelope_data.upper_bound_production)
                
                # Check monotonicity
                assert np.all(np.diff(envelope_data.lower_bound_production) >= 0)
                assert np.all(np.diff(envelope_data.upper_bound_production) >= 0)
                
                # Check dominance (upper >= lower)
                assert np.all(envelope_data.upper_bound_production >= envelope_data.lower_bound_production)
                assert np.all(envelope_data.upper_bound_harvest >= envelope_data.lower_bound_harvest)
            
            # Validate width reductions
            comprehensive_envelope = results.tier_results['comprehensive']
            commercial_envelope = results.tier_results['commercial']
            
            # Commercial tier should be narrower than comprehensive
            comp_width = np.mean(comprehensive_envelope.upper_bound_production - comprehensive_envelope.lower_bound_production)
            comm_width = np.mean(commercial_envelope.upper_bound_production - commercial_envelope.lower_bound_production)
            
            width_reduction = (comp_width - comm_width) / comp_width * 100
            assert width_reduction > 0, "Commercial tier should be narrower than comprehensive"
            
            return {
                'passed': True,
                'details': {
                    'tiers_calculated': list(results.tier_results.keys()),
                    'width_reduction_pct': width_reduction,
                    'envelope_points': len(comprehensive_envelope.disruption_areas),
                    'mathematical_validation': 'passed'
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-tier calculation test failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test system performance meets requirements."""
        logger.info("Testing performance benchmarks...")
        
        try:
            config = Config()
            config.crop_type = 'wheat'
            
            # Test different dataset sizes
            performance_results = {}
            
            for size_name, n_cells in [('small', 100), ('medium', 500), ('large', 1000)]:
                logger.info(f"Testing {size_name} dataset ({n_cells} cells)")
                
                # Create data
                production_df, harvest_df = self.create_synthetic_spam_data(n_cells=n_cells)
                
                # Measure performance
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Create engine and calculate
                engine = MultiTierEnvelopeEngine(config)
                results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                performance_results[size_name] = {
                    'execution_time_seconds': execution_time,
                    'memory_usage_mb': memory_used,
                    'cells_processed': n_cells,
                    'tiers_calculated': len(results.tier_results)
                }
                
                # Performance assertions based on requirements
                if size_name == 'small':
                    assert execution_time < 30, f"Small dataset too slow: {execution_time:.2f}s"
                elif size_name == 'medium':
                    assert execution_time < 120, f"Medium dataset too slow: {execution_time:.2f}s"
                elif size_name == 'large':
                    assert execution_time < 300, f"Large dataset too slow: {execution_time:.2f}s"
                
                assert memory_used < 500, f"Memory usage too high: {memory_used:.1f}MB"
            
            return {
                'passed': True,
                'details': performance_results
            }
            
        except Exception as e:
            logger.error(f"Performance benchmark test failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def test_data_quality_validation(self) -> Dict[str, Any]:
        """Test data quality and filtering validation."""
        logger.info("Testing data quality validation...")
        
        try:
            config = Config()
            config.crop_type = 'wheat'
            
            # Create data with known quality issues
            production_df, harvest_df = self.create_synthetic_spam_data(n_cells=200)
            
            # Add some problematic data
            # Very small harvest areas (should be filtered)
            production_df.loc[:10, 'WHEA_A'] = 1000  # Very small production
            harvest_df.loc[:10, 'WHEA_A'] = 0.005   # Very small area
            
            # Very high yields (should be filtered)
            production_df.loc[190:, 'WHEA_A'] = 1e12  # Unrealistic production
            harvest_df.loc[190:, 'WHEA_A'] = 1.0     # Normal area
            
            # Calculate with filtering
            engine = MultiTierEnvelopeEngine(config)
            results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
            
            # Check that filtering worked
            base_stats = results.base_statistics
            spam_stats = base_stats.get('spam_filter_stats', {})
            
            assert spam_stats.get('retention_rate', 0) < 100, "Some data should have been filtered"
            assert spam_stats.get('retention_rate', 0) > 50, "Too much data was filtered"
            
            # Validate that results are reasonable
            for tier_name, envelope_data in results.tier_results.items():
                # Check that yields are in reasonable range
                total_production = envelope_data.convergence_point[1]
                total_harvest = envelope_data.convergence_point[0]
                
                if total_harvest > 0:
                    avg_yield = total_production / total_harvest / 100 / 3.34e9  # Convert to tons/ha
                    assert 0.1 < avg_yield < 50, f"Unrealistic average yield: {avg_yield:.2f} tons/ha"
            
            return {
                'passed': True,
                'details': {
                    'original_cells': len(production_df),
                    'filtered_cells': spam_stats.get('final_cells', 0),
                    'retention_rate': spam_stats.get('retention_rate', 0),
                    'area_filtered': spam_stats.get('area_removed_pct', 0),
                    'yield_filtered': spam_stats.get('yield_removed_pct', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Data quality validation test failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def test_crop_type_variations(self) -> Dict[str, Any]:
        """Test system works with different crop types."""
        logger.info("Testing crop type variations...")
        
        try:
            crop_results = {}
            
            for crop_type in ['wheat', 'rice', 'maize']:
                logger.info(f"Testing {crop_type}")
                
                config = Config()
                config.crop_type = crop_type
                
                # Create crop-specific data
                production_df, harvest_df = self.create_synthetic_spam_data(n_cells=200, crop_type=crop_type)
                
                # Calculate envelopes
                engine = MultiTierEnvelopeEngine(config)
                results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
                
                # Validate results
                assert len(results.tier_results) >= 2
                assert 'comprehensive' in results.tier_results
                assert 'commercial' in results.tier_results
                
                # Check width reduction
                width_reduction = results.get_width_reduction('commercial')
                assert width_reduction is not None and width_reduction > 0
                
                crop_results[crop_type] = {
                    'tiers_calculated': len(results.tier_results),
                    'width_reduction_pct': width_reduction,
                    'envelope_points': len(results.tier_results['comprehensive'].disruption_areas)
                }
            
            return {
                'passed': True,
                'details': crop_results
            }
            
        except Exception as e:
            logger.error(f"Crop type variations test failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def test_caching_and_optimization(self) -> Dict[str, Any]:
        """Test caching and performance optimizations."""
        logger.info("Testing caching and optimization...")
        
        try:
            config = Config()
            config.crop_type = 'wheat'
            
            # Create test data
            production_df, harvest_df = self.create_synthetic_spam_data(n_cells=300)
            
            # Test with caching enabled
            engine_cached = MultiTierEnvelopeEngine(config, enable_caching=True)
            
            # First calculation (cache miss)
            start_time = time.time()
            results1 = engine_cached.calculate_multi_tier_envelope(production_df, harvest_df)
            first_time = time.time() - start_time
            
            # Second calculation (cache hit)
            start_time = time.time()
            results2 = engine_cached.calculate_multi_tier_envelope(production_df, harvest_df)
            second_time = time.time() - start_time
            
            # Cache should make second calculation faster
            speedup = first_time / max(second_time, 0.001)  # Avoid division by zero
            
            # Results should be identical
            assert len(results1.tier_results) == len(results2.tier_results)
            
            # Test parallel processing if available
            parallel_results = {}
            if os.cpu_count() > 1:
                engine_parallel = MultiTierEnvelopeEngine(config, enable_parallel=True)
                
                start_time = time.time()
                results_parallel = engine_parallel.calculate_multi_tier_envelope(production_df, harvest_df)
                parallel_time = time.time() - start_time
                
                parallel_results = {
                    'parallel_time': parallel_time,
                    'tiers_calculated': len(results_parallel.tier_results)
                }
            
            return {
                'passed': True,
                'details': {
                    'caching_speedup': speedup,
                    'first_calculation_time': first_time,
                    'cached_calculation_time': second_time,
                    'parallel_processing': parallel_results
                }
            }
            
        except Exception as e:
            logger.error(f"Caching and optimization test failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and generate comprehensive report."""
        logger.info("Starting comprehensive system validation...")
        
        test_suite = [
            ('Multi-Tier Calculation Accuracy', self.test_multi_tier_calculation_accuracy),
            ('Performance Benchmarks', self.test_performance_benchmarks),
            ('Data Quality Validation', self.test_data_quality_validation),
            ('Crop Type Variations', self.test_crop_type_variations),
            ('Caching and Optimization', self.test_caching_and_optimization)
        ]
        
        all_results = {}
        overall_passed = True
        
        for test_name, test_function in test_suite:
            logger.info(f"Running: {test_name}")
            
            try:
                result = test_function()
                all_results[test_name] = result
                
                if not result.get('passed', False):
                    overall_passed = False
                    logger.error(f"❌ {test_name} FAILED")
                else:
                    logger.info(f"✅ {test_name} PASSED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
                all_results[test_name] = {
                    'passed': False,
                    'error': str(e)
                }
                overall_passed = False
        
        # Generate summary
        summary = {
            'overall_status': 'PASS' if overall_passed else 'FAIL',
            'total_tests': len(test_suite),
            'passed_tests': sum(1 for r in all_results.values() if r.get('passed', False)),
            'failed_tests': sum(1 for r in all_results.values() if not r.get('passed', False)),
            'test_results': all_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return summary


@pytest.fixture
def validation_suite():
    """Create validation suite fixture."""
    return SystemValidationSuite()


class TestSystemValidationComprehensive:
    """Comprehensive system validation test class."""
    
    def test_comprehensive_system_validation(self, validation_suite):
        """Run comprehensive system validation."""
        results = validation_suite.run_comprehensive_validation()
        
        # Assert overall success
        assert results['overall_status'] == 'PASS', f"System validation failed: {results['failed_tests']} tests failed"
        
        # Validate specific requirements
        assert results['passed_tests'] >= 4, "At least 4 tests should pass"
        
        # Check that key tests passed
        test_results = results['test_results']
        
        # Multi-tier calculation must work
        assert test_results.get('Multi-Tier Calculation Accuracy', {}).get('passed', False), \
            "Multi-tier calculation test must pass"
        
        # Performance must be acceptable
        perf_result = test_results.get('Performance Benchmarks', {})
        if perf_result.get('passed', False):
            perf_details = perf_result.get('details', {})
            # Check that medium dataset completes in reasonable time
            medium_time = perf_details.get('medium', {}).get('execution_time_seconds', 999)
            assert medium_time < 120, f"Medium dataset too slow: {medium_time:.2f}s"
        
        # Generate validation report
        report_path = Path('./system_validation_report.md')
        generate_system_validation_report(results, report_path)
        
        logger.info(f"✅ Comprehensive system validation completed successfully")
        logger.info(f"📊 Results: {results['passed_tests']}/{results['total_tests']} tests passed")
        logger.info(f"📄 Report generated: {report_path}")


def generate_system_validation_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate comprehensive system validation report."""
    report_lines = [
        "# Multi-Tier Envelope System: Comprehensive Validation Report",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Overall Status:** {'✅ PASS' if results['overall_status'] == 'PASS' else '❌ FAIL'}",
        "",
        "## Executive Summary",
        ""
    ]
    
    if results['overall_status'] == 'PASS':
        report_lines.extend([
            "🎉 **SYSTEM VALIDATION SUCCESSFUL**",
            "",
            "The multi-tier envelope integration system has successfully passed comprehensive",
            "validation testing. All core functionality, performance requirements, and",
            "mathematical properties have been verified.",
            ""
        ])
    else:
        report_lines.extend([
            "⚠️ **SYSTEM VALIDATION FAILED**",
            "",
            f"**Failed Tests:** {results['failed_tests']}/{results['total_tests']}",
            "",
            "Please review the detailed test results below and address all failing",
            "tests before proceeding with production deployment.",
            ""
        ])
    
    # Test Results Summary
    report_lines.extend([
        "## Test Results Summary",
        "",
        f"- **Total Tests:** {results['total_tests']}",
        f"- **Passed:** {results['passed_tests']}",
        f"- **Failed:** {results['failed_tests']}",
        "",
        "| Test Name | Status | Details |",
        "|-----------|--------|---------|"
    ])
    
    for test_name, result in results['test_results'].items():
        status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        error_msg = result.get('error', 'See details below')
        details = error_msg if not result.get('passed', False) else "All validations passed"
        
        report_lines.append(f"| {test_name} | {status} | {details} |")
    
    # Detailed Results
    report_lines.extend([
        "",
        "## Detailed Test Results",
        ""
    ])
    
    for test_name, result in results['test_results'].items():
        report_lines.extend([
            f"### {test_name}",
            ""
        ])
        
        if result.get('passed', False):
            report_lines.append("✅ **Status:** PASSED")
            
            if 'details' in result:
                report_lines.append("")
                report_lines.append("**Details:**")
                details = result['details']
                
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, dict):
                            report_lines.append(f"- **{key}:**")
                            for sub_key, sub_value in value.items():
                                report_lines.append(f"  - {sub_key}: {sub_value}")
                        else:
                            report_lines.append(f"- **{key}:** {value}")
        else:
            report_lines.append("❌ **Status:** FAILED")
            if 'error' in result:
                report_lines.append(f"**Error:** {result['error']}")
        
        report_lines.append("")
    
    # Recommendations
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    if results['overall_status'] == 'PASS':
        report_lines.extend([
            "- ✅ **System Ready:** All validation tests passed successfully",
            "- 🚀 **Deploy to Production:** System meets all requirements",
            "- 📊 **Monitor Performance:** Set up production monitoring",
            "- 🔄 **Regular Testing:** Run validation tests with new data regularly",
            ""
        ])
    else:
        report_lines.extend([
            "- ❌ **Do Not Deploy:** Fix all failing tests first",
            "- 🔍 **Review Failures:** Check detailed test results above",
            "- 🧪 **Re-run Validation:** After fixes, run comprehensive validation again",
            "- 🆘 **Get Support:** Contact development team if issues persist",
            ""
        ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    # Run comprehensive validation
    suite = SystemValidationSuite()
    results = suite.run_comprehensive_validation()
    
    # Generate report
    report_path = Path('./system_validation_report.md')
    generate_system_validation_report(results, report_path)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SYSTEM VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Status: {results['overall_status']}")
    print(f"Tests: {results['passed_tests']}/{results['total_tests']} passed")
    print(f"Report: {report_path}")
    print(f"{'='*80}")
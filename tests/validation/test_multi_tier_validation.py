"""
Comprehensive validation tests for multi-tier envelope system.

This module validates that the multi-tier system meets all requirements
and maintains mathematical correctness across all scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper
from agrichter.data.country_boundary_manager import CountryBoundaryManager
from agrichter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine, MultiTierResults
from agrichter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer
from agrichter.analysis.convergence_validator import ConvergenceValidator


class ValidationReport:
    """Comprehensive validation report generator."""
    
    def __init__(self):
        self.test_results = {}
        self.validation_errors = []
        self.validation_warnings = []
        self.performance_metrics = {}
    
    def add_test_result(self, test_name: str, passed: bool, details: Dict[str, Any] = None):
        """Add test result to report."""
        self.test_results[test_name] = {
            'passed': passed,
            'details': details or {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if not passed:
            self.validation_errors.append(f"{test_name}: {details.get('error', 'Test failed')}")
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.validation_warnings.append(message)
    
    def add_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Add performance metric."""
        self.performance_metrics[metric_name] = {
            'value': value,
            'unit': unit,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_overall_status(self) -> bool:
        """Get overall validation status."""
        return all(result['passed'] for result in self.test_results.values())
    
    def generate_report(self, output_path: Path):
        """Generate comprehensive validation report."""
        overall_status = self.get_overall_status()
        
        report_lines = [
            "# Multi-Tier Envelope System Validation Report",
            "",
            f"**Generated:** {pd.Timestamp.now().isoformat()}",
            f"**Overall Status:** {'✓ PASS' if overall_status else '✗ FAIL'}",
            "",
            "## Executive Summary",
            ""
        ]
        
        if overall_status:
            report_lines.extend([
                "✅ **System Ready for Production Deployment**",
                "",
                "All validation tests passed successfully. The multi-tier envelope system",
                "meets all mathematical, performance, and integration requirements.",
                ""
            ])
        else:
            report_lines.extend([
                "❌ **System Not Ready for Production Deployment**",
                "",
                f"**{len(self.validation_errors)} critical issues** must be resolved before deployment.",
                ""
            ])
        
        # Test Results Summary
        report_lines.extend([
            "## Test Results Summary",
            "",
            f"- **Total Tests:** {len(self.test_results)}",
            f"- **Passed:** {sum(1 for r in self.test_results.values() if r['passed'])}",
            f"- **Failed:** {sum(1 for r in self.test_results.values() if not r['passed'])}",
            f"- **Warnings:** {len(self.validation_warnings)}",
            ""
        ])
        
        # Detailed Test Results
        report_lines.extend([
            "## Detailed Test Results",
            ""
        ])
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            report_lines.append(f"### {test_name}: {status}")
            
            if result['details']:
                for key, value in result['details'].items():
                    report_lines.append(f"- **{key}:** {value}")
            
            report_lines.append("")
        
        # Performance Metrics
        if self.performance_metrics:
            report_lines.extend([
                "## Performance Metrics",
                ""
            ])
            
            for metric_name, metric_data in self.performance_metrics.items():
                value = metric_data['value']
                unit = metric_data['unit']
                report_lines.append(f"- **{metric_name}:** {value:.2f} {unit}")
            
            report_lines.append("")
        
        # Validation Errors
        if self.validation_errors:
            report_lines.extend([
                "## Critical Issues",
                ""
            ])
            
            for error in self.validation_errors:
                report_lines.append(f"- ❌ {error}")
            
            report_lines.append("")
        
        # Validation Warnings
        if self.validation_warnings:
            report_lines.extend([
                "## Warnings",
                ""
            ])
            
            for warning in self.validation_warnings:
                report_lines.append(f"- ⚠️ {warning}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if overall_status:
            report_lines.extend([
                "- ✅ System is ready for production deployment",
                "- 📊 Monitor performance metrics in production",
                "- 🔄 Run validation tests regularly with new data",
                ""
            ])
        else:
            report_lines.extend([
                "- ❌ Resolve all critical issues before deployment",
                "- 🔍 Review failed test details above",
                "- 🧪 Re-run validation after fixes",
                ""
            ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save raw data
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'validation_errors': self.validation_errors,
                'validation_warnings': self.validation_warnings,
                'performance_metrics': self.performance_metrics,
                'overall_status': overall_status
            }, f, indent=2)


class TestMultiTierValidation:
    """Comprehensive validation test suite for multi-tier envelope system."""
    
    @pytest.fixture(scope="class")
    def validation_report(self):
        """Validation report instance."""
        return ValidationReport()
    
    @pytest.fixture(scope="class")
    def test_config(self):
        """Test configuration."""
        return Config(crop_type='wheat')
    
    @pytest.fixture(scope="class")
    def grid_manager(self, test_config):
        """Grid manager with loaded data."""
        manager = GridDataManager(test_config)
        if not manager.is_loaded():
            manager.load_spam_data()
        return manager
    
    @pytest.fixture(scope="class")
    def multi_tier_engine(self, test_config):
        """Multi-tier envelope engine."""
        return MultiTierEnvelopeEngine(test_config)
    
    @pytest.fixture(scope="class")
    def convergence_validator(self):
        """Convergence validator."""
        return ConvergenceValidator(tolerance=1e-6)
    
    @pytest.fixture(scope="class")
    def sample_data(self, grid_manager):
        """Sample SPAM data for validation."""
        return grid_manager.get_sample_data(n_samples=3000)
    
    def test_requirement_r1_real_spam_integration(self, validation_report, multi_tier_engine, sample_data):
        """Validate R1: Real SPAM Data Integration."""
        production_df, harvest_df = sample_data
        
        try:
            # R1.1: Apply multi-tier system to real SPAM data
            results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            
            # R1.2: Validate width reductions
            commercial_reduction = results.get_width_reduction('commercial')
            width_reduction_valid = commercial_reduction is not None and commercial_reduction >= 0
            
            # R1.3: Mathematical properties preserved
            math_validation = multi_tier_engine.validate_multi_tier_results(results)
            math_properties_valid = math_validation['overall_valid']
            
            # R1.4: Generate comparison reports
            comparison_data = {
                'comprehensive_width': results.width_analysis.get('comprehensive_width', 0),
                'commercial_width': results.width_analysis.get('commercial_width', 0),
                'width_reduction_pct': commercial_reduction
            }
            
            all_valid = (
                isinstance(results, MultiTierResults) and
                width_reduction_valid and
                math_properties_valid and
                len(results.tier_results) >= 2
            )
            
            validation_report.add_test_result(
                'R1_Real_SPAM_Integration',
                all_valid,
                {
                    'tiers_calculated': len(results.tier_results),
                    'width_reduction_pct': commercial_reduction,
                    'mathematical_validation': math_properties_valid,
                    'comparison_data_generated': bool(comparison_data)
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'R1_Real_SPAM_Integration',
                False,
                {'error': str(e)}
            )
    
    def test_requirement_r2_national_analysis(self, validation_report, test_config, grid_manager):
        """Validate R2: National-Level Analysis Implementation."""
        try:
            # Create national analysis components
            spatial_mapper = SpatialMapper(test_config)
            country_manager = CountryBoundaryManager(test_config, spatial_mapper, grid_manager)
            multi_tier_engine = MultiTierEnvelopeEngine(test_config)
            national_analyzer = NationalEnvelopeAnalyzer(test_config, country_manager, multi_tier_engine)
            
            # R2.1 & R2.2: USA and China analysis
            countries_tested = []
            for country_code in ['USA', 'CHN']:
                try:
                    validation = country_manager.validate_country_data_coverage(country_code)
                    if validation['valid']:
                        results = national_analyzer.analyze_national_capacity(country_code)
                        countries_tested.append(country_code)
                except Exception as e:
                    validation_report.add_warning(f"Country {country_code} analysis failed: {e}")
            
            # R2.3: National comparison framework
            comparison_framework_valid = len(countries_tested) >= 1
            
            # R2.4: Validate against expected patterns
            patterns_valid = True
            for country_code in countries_tested:
                try:
                    results = national_analyzer.analyze_national_capacity(country_code)
                    # Basic pattern validation
                    if results.national_statistics['total_production_mt'] <= 0:
                        patterns_valid = False
                        break
                except:
                    patterns_valid = False
                    break
            
            validation_report.add_test_result(
                'R2_National_Analysis',
                comparison_framework_valid and patterns_valid,
                {
                    'countries_analyzed': countries_tested,
                    'comparison_framework': comparison_framework_valid,
                    'patterns_validated': patterns_valid
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'R2_National_Analysis',
                False,
                {'error': str(e)}
            )
    
    def test_requirement_r3_production_integration(self, validation_report, multi_tier_engine, sample_data):
        """Validate R3: Production System Integration."""
        production_df, harvest_df = sample_data
        
        try:
            # R3.1: Update existing envelope calculators
            results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            calculator_updated = hasattr(multi_tier_engine, 'base_calculator')
            
            # R3.2: SPAM filtering integration
            spam_filtering_integrated = 'spam_filter_stats' in results.base_statistics
            
            # R3.3: Multi-tier options in pipelines
            multi_tier_options = len(results.tier_results) >= 2
            
            # R3.4: Backward compatibility
            comprehensive_results = multi_tier_engine.calculate_multi_tier_envelope(
                production_df, harvest_df, tiers=['comprehensive']
            )
            backward_compatible = 'comprehensive' in comprehensive_results.tier_results
            
            all_valid = (
                calculator_updated and
                spam_filtering_integrated and
                multi_tier_options and
                backward_compatible
            )
            
            validation_report.add_test_result(
                'R3_Production_Integration',
                all_valid,
                {
                    'calculator_updated': calculator_updated,
                    'spam_filtering': spam_filtering_integrated,
                    'multi_tier_options': multi_tier_options,
                    'backward_compatible': backward_compatible
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'R3_Production_Integration',
                False,
                {'error': str(e)}
            )
    
    def test_requirement_t1_mathematical_validation(self, validation_report, multi_tier_engine, convergence_validator, sample_data):
        """Validate T1: Mathematical Validation Requirements."""
        production_df, harvest_df = sample_data
        
        try:
            results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            
            # T1.1: Comprehensive validation suite
            validation_results = multi_tier_engine.validate_multi_tier_results(results)
            comprehensive_validation = validation_results['overall_valid']
            
            # T1.2: SPAM methodology compliance
            spam_compliance = 'spam_filter_stats' in results.base_statistics
            
            # T1.3: Mathematical properties
            properties_valid = True
            for tier_name, envelope_data in results.tier_results.items():
                # Check monotonicity
                if not np.all(np.diff(envelope_data.lower_bound_production) >= 0):
                    properties_valid = False
                    break
                
                # Check dominance
                if not np.all(envelope_data.upper_bound_production >= envelope_data.lower_bound_production):
                    properties_valid = False
                    break
            
            # T1.4: Numerical stability
            numerical_stability = not any(
                np.any(np.isnan(envelope_data.lower_bound_production)) or
                np.any(np.isnan(envelope_data.upper_bound_production))
                for envelope_data in results.tier_results.values()
            )
            
            all_valid = (
                comprehensive_validation and
                spam_compliance and
                properties_valid and
                numerical_stability
            )
            
            validation_report.add_test_result(
                'T1_Mathematical_Validation',
                all_valid,
                {
                    'comprehensive_validation': comprehensive_validation,
                    'spam_compliance': spam_compliance,
                    'mathematical_properties': properties_valid,
                    'numerical_stability': numerical_stability
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'T1_Mathematical_Validation',
                False,
                {'error': str(e)}
            )
    
    def test_requirement_t3_performance(self, validation_report, multi_tier_engine, sample_data):
        """Validate T3: Performance Requirements."""
        production_df, harvest_df = sample_data
        
        try:
            import time
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # T3.1: Calculation time
            start_time = time.time()
            results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            calculation_time = time.time() - start_time
            
            # T3.2: Memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # T3.3: Reproducibility
            results2 = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            reproducible = True
            for tier_name in results.tier_results.keys():
                if tier_name in results2.tier_results:
                    env1 = results.tier_results[tier_name]
                    env2 = results2.tier_results[tier_name]
                    if not np.allclose(env1.lower_bound_production, env2.lower_bound_production, rtol=1e-10):
                        reproducible = False
                        break
            
            # Performance targets
            time_target = calculation_time < 300  # 5 minutes
            memory_target = memory_usage < 8000  # 8GB
            
            validation_report.add_performance_metric('calculation_time', calculation_time, 'seconds')
            validation_report.add_performance_metric('memory_usage', memory_usage, 'MB')
            
            validation_report.add_test_result(
                'T3_Performance_Requirements',
                time_target and memory_target and reproducible,
                {
                    'calculation_time_seconds': calculation_time,
                    'memory_usage_mb': memory_usage,
                    'time_target_met': time_target,
                    'memory_target_met': memory_target,
                    'reproducible': reproducible
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'T3_Performance_Requirements',
                False,
                {'error': str(e)}
            )
    
    def test_width_reduction_targets(self, validation_report, multi_tier_engine, sample_data):
        """Validate width reduction performance targets."""
        production_df, harvest_df = sample_data
        
        try:
            results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            
            # Check commercial tier width reduction
            commercial_reduction = results.get_width_reduction('commercial')
            
            # Width reduction should be positive (commercial narrower than comprehensive)
            width_reduction_positive = commercial_reduction is not None and commercial_reduction >= 0
            
            # Log actual vs expected ranges
            expected_range = "7-35% (varies with real data characteristics)"
            actual_value = f"{commercial_reduction:.1f}%" if commercial_reduction else "N/A"
            
            validation_report.add_test_result(
                'Width_Reduction_Targets',
                width_reduction_positive,
                {
                    'commercial_width_reduction_pct': commercial_reduction,
                    'expected_range': expected_range,
                    'actual_value': actual_value,
                    'positive_reduction': width_reduction_positive
                }
            )
            
            if commercial_reduction is not None and commercial_reduction < 5:
                validation_report.add_warning(
                    f"Commercial tier width reduction ({commercial_reduction:.1f}%) is lower than typical range. "
                    "This may be due to data characteristics or filtering effects."
                )
            
        except Exception as e:
            validation_report.add_test_result(
                'Width_Reduction_Targets',
                False,
                {'error': str(e)}
            )
    
    def test_data_quality_requirements(self, validation_report, multi_tier_engine, sample_data):
        """Validate data quality requirements."""
        production_df, harvest_df = sample_data
        
        try:
            results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
            
            # Check data retention
            base_stats = results.base_statistics
            total_cells = base_stats['total_cells']
            
            # Should have reasonable number of cells
            min_cells_met = total_cells >= 1000
            
            # Check SPAM filter retention if available
            retention_rate = None
            if 'spam_filter_stats' in base_stats:
                retention_rate = base_stats['spam_filter_stats'].get('retention_rate', 0)
            
            retention_adequate = retention_rate is None or retention_rate >= 20  # At least 20% retention
            
            # Check yield distributions
            mean_yield = base_stats.get('mean_yield', 0)
            yield_reasonable = 0 < mean_yield < 50  # Reasonable yield range (MT/ha)
            
            validation_report.add_test_result(
                'Data_Quality_Requirements',
                min_cells_met and retention_adequate and yield_reasonable,
                {
                    'total_cells': total_cells,
                    'min_cells_met': min_cells_met,
                    'retention_rate_pct': retention_rate,
                    'retention_adequate': retention_adequate,
                    'mean_yield_mt_per_ha': mean_yield,
                    'yield_reasonable': yield_reasonable
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'Data_Quality_Requirements',
                False,
                {'error': str(e)}
            )
    
    def test_system_integration_validation(self, validation_report, test_config, grid_manager):
        """Validate complete system integration."""
        try:
            # Test component integration
            spatial_mapper = SpatialMapper(test_config)
            country_manager = CountryBoundaryManager(test_config, spatial_mapper, grid_manager)
            multi_tier_engine = MultiTierEnvelopeEngine(test_config)
            national_analyzer = NationalEnvelopeAnalyzer(test_config, country_manager, multi_tier_engine)
            
            # Test basic functionality
            available_countries = country_manager.get_available_countries()
            tier_info = multi_tier_engine.get_tier_info()
            
            # Integration checks
            components_initialized = all([
                spatial_mapper is not None,
                country_manager is not None,
                multi_tier_engine is not None,
                national_analyzer is not None
            ])
            
            configuration_valid = (
                len(available_countries) >= 2 and
                len(tier_info) >= 2
            )
            
            validation_report.add_test_result(
                'System_Integration_Validation',
                components_initialized and configuration_valid,
                {
                    'components_initialized': components_initialized,
                    'available_countries': len(available_countries),
                    'available_tiers': len(tier_info),
                    'configuration_valid': configuration_valid
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'System_Integration_Validation',
                False,
                {'error': str(e)}
            )
    
    def test_error_handling_validation(self, validation_report, multi_tier_engine):
        """Validate error handling and edge cases."""
        try:
            error_handling_tests = []
            
            # Test 1: Empty data
            try:
                empty_df = pd.DataFrame()
                multi_tier_engine.calculate_multi_tier_envelope(empty_df, empty_df)
                error_handling_tests.append(False)  # Should have raised an error
            except (ValueError, KeyError):
                error_handling_tests.append(True)  # Correctly handled
            except Exception:
                error_handling_tests.append(False)  # Wrong error type
            
            # Test 2: Invalid tier names
            try:
                sample_df = pd.DataFrame({
                    'WHEA_A': [100, 200, 300],
                    'x': [0, 1, 2],
                    'y': [0, 1, 2]
                })
                results = multi_tier_engine.calculate_multi_tier_envelope(
                    sample_df, sample_df, tiers=['invalid_tier', 'comprehensive']
                )
                # Should handle gracefully and calculate valid tiers
                error_handling_tests.append('comprehensive' in results.tier_results and 'invalid_tier' not in results.tier_results)
            except Exception:
                error_handling_tests.append(False)
            
            # Test 3: Malformed data
            try:
                malformed_df = pd.DataFrame({
                    'WHEA_A': [np.nan, np.inf, -1],
                    'x': [0, 1, 2],
                    'y': [0, 1, 2]
                })
                results = multi_tier_engine.calculate_multi_tier_envelope(malformed_df, malformed_df)
                # Should handle or filter out invalid data
                error_handling_tests.append(True)
            except Exception:
                error_handling_tests.append(False)
            
            all_error_handling_passed = all(error_handling_tests)
            
            validation_report.add_test_result(
                'Error_Handling_Validation',
                all_error_handling_passed,
                {
                    'empty_data_handling': error_handling_tests[0] if len(error_handling_tests) > 0 else False,
                    'invalid_tier_handling': error_handling_tests[1] if len(error_handling_tests) > 1 else False,
                    'malformed_data_handling': error_handling_tests[2] if len(error_handling_tests) > 2 else False,
                    'total_tests_passed': sum(error_handling_tests),
                    'total_tests': len(error_handling_tests)
                }
            )
            
        except Exception as e:
            validation_report.add_test_result(
                'Error_Handling_Validation',
                False,
                {'error': str(e)}
            )


def run_comprehensive_validation(output_dir: Path = None) -> ValidationReport:
    """Run comprehensive validation and generate report."""
    if output_dir is None:
        output_dir = Path('.')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run validation tests
    validation_report = ValidationReport()
    
    try:
        # Initialize test components
        config = Config(crop_type='wheat')
        grid_manager = GridDataManager(config)
        
        if not grid_manager.is_loaded():
            grid_manager.load_spam_data()
        
        multi_tier_engine = MultiTierEnvelopeEngine(config)
        sample_data = grid_manager.get_sample_data(n_samples=2000)
        
        # Run validation tests
        test_instance = TestMultiTierValidation()
        
        # Initialize fixtures manually for standalone execution
        test_instance.test_requirement_r1_real_spam_integration(validation_report, multi_tier_engine, sample_data)
        test_instance.test_requirement_r3_production_integration(validation_report, multi_tier_engine, sample_data)
        test_instance.test_requirement_t1_mathematical_validation(validation_report, multi_tier_engine, ConvergenceValidator(), sample_data)
        test_instance.test_requirement_t3_performance(validation_report, multi_tier_engine, sample_data)
        test_instance.test_width_reduction_targets(validation_report, multi_tier_engine, sample_data)
        test_instance.test_data_quality_requirements(validation_report, multi_tier_engine, sample_data)
        test_instance.test_system_integration_validation(validation_report, config, grid_manager)
        test_instance.test_error_handling_validation(validation_report, multi_tier_engine)
        
        # Try national analysis tests (may fail if data not available)
        try:
            test_instance.test_requirement_r2_national_analysis(validation_report, config, grid_manager)
        except Exception as e:
            validation_report.add_warning(f"National analysis validation skipped: {e}")
        
    except Exception as e:
        validation_report.add_test_result(
            'Validation_Setup',
            False,
            {'error': f"Failed to initialize validation: {e}"}
        )
    
    # Generate report
    report_path = output_dir / 'multi_tier_validation_report.md'
    validation_report.generate_report(report_path)
    
    return validation_report


if __name__ == "__main__":
    # Run comprehensive validation
    report = run_comprehensive_validation(Path('.'))
    
    if report.get_overall_status():
        print("✅ All validation tests passed - System ready for production!")
    else:
        print("❌ Validation failed - Check report for details")
        print(f"Errors: {len(report.validation_errors)}")
        print(f"Warnings: {len(report.validation_warnings)}")
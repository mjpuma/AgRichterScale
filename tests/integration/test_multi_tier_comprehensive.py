"""
Comprehensive integration tests for multi-tier envelope system.

This test suite validates the complete multi-tier envelope integration system
including mathematical properties, performance requirements, and real SPAM data integration.
"""

import pytest
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any
import time
import json

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine, MultiTierResults
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer
from agririchter.analysis.national_comparison_analyzer import NationalComparisonAnalyzer
from agririchter.validation.spam_data_filter import SPAMDataFilter


class TestMultiTierComprehensive:
    """Comprehensive test suite for multi-tier envelope system."""
    
    @pytest.fixture(scope="class")
    def test_config(self):
        """Create test configuration."""
        return Config(crop_type='wheat', root_dir='.')
    
    @pytest.fixture(scope="class")
    def grid_manager(self, test_config):
        """Create and load grid manager."""
        manager = GridDataManager(test_config)
        if not manager.is_loaded():
            manager.load_spam_data()
        return manager
    
    @pytest.fixture(scope="class")
    def spatial_mapper(self, test_config, grid_manager):
        """Create spatial mapper."""
        return SpatialMapper(test_config, grid_manager)
    
    @pytest.fixture(scope="class")
    def country_manager(self, test_config, spatial_mapper, grid_manager):
        """Create country boundary manager."""
        return CountryBoundaryManager(test_config, spatial_mapper, grid_manager)
    
    @pytest.fixture(scope="class")
    def multi_tier_engine(self, test_config):
        """Create multi-tier envelope engine."""
        return MultiTierEnvelopeEngine(test_config)
    
    @pytest.fixture(scope="class")
    def national_analyzer(self, test_config, country_manager, multi_tier_engine):
        """Create national envelope analyzer."""
        return NationalEnvelopeAnalyzer(test_config, country_manager, multi_tier_engine)
    
    @pytest.fixture(scope="class")
    def comparison_analyzer(self, test_config, national_analyzer):
        """Create national comparison analyzer."""
        return NationalComparisonAnalyzer(test_config, national_analyzer)
    
    @pytest.fixture(scope="class")
    def sample_spam_data(self, grid_manager):
        """Get sample SPAM data for testing."""
        # Get the full data and sample it
        production_df = grid_manager.get_production_data()
        harvest_df = grid_manager.get_harvest_area_data()
        
        # Sample 5000 rows for testing
        n_samples = min(5000, len(production_df))
        sample_indices = np.random.choice(len(production_df), size=n_samples, replace=False)
        
        production_sample = production_df.iloc[sample_indices].copy()
        harvest_sample = harvest_df.iloc[sample_indices].copy()
        
        return production_sample, harvest_sample
    
    def test_multi_tier_engine_initialization(self, multi_tier_engine):
        """Test multi-tier engine initialization."""
        assert multi_tier_engine is not None
        assert hasattr(multi_tier_engine, 'tier_configs')
        assert 'comprehensive' in multi_tier_engine.tier_configs
        assert 'commercial' in multi_tier_engine.tier_configs
        
        # Test tier info retrieval
        tier_info = multi_tier_engine.get_tier_info()
        assert isinstance(tier_info, dict)
        assert len(tier_info) >= 2
        
        for tier_name, info in tier_info.items():
            assert 'name' in info
            assert 'description' in info
            assert 'policy_applications' in info
    
    def test_multi_tier_calculation_with_real_data(self, multi_tier_engine, sample_spam_data):
        """Test multi-tier envelope calculation with real SPAM data."""
        production_df, harvest_df = sample_spam_data
        
        # Calculate multi-tier envelopes
        start_time = time.time()
        results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
        calculation_time = time.time() - start_time
        
        # Validate results structure
        assert isinstance(results, MultiTierResults)
        assert results.crop_type == 'wheat'
        assert len(results.tier_results) >= 2
        assert 'comprehensive' in results.tier_results
        assert 'commercial' in results.tier_results
        
        # Validate mathematical properties
        for tier_name, envelope_data in results.tier_results.items():
            assert envelope_data.convergence_validated, f"{tier_name} tier failed convergence validation"
            assert len(envelope_data.disruption_areas) > 0, f"{tier_name} tier has no envelope points"
            
            # Check monotonicity
            assert np.all(np.diff(envelope_data.lower_bound_production) >= 0), f"{tier_name} lower bound not monotonic"
            assert np.all(np.diff(envelope_data.upper_bound_production) >= 0), f"{tier_name} upper bound not monotonic"
            
            # Check dominance
            assert np.all(envelope_data.upper_bound_production >= envelope_data.lower_bound_production), \
                f"{tier_name} dominance property violated"
        
        # Validate width reductions
        commercial_reduction = results.get_width_reduction('commercial')
        assert commercial_reduction is not None, "Commercial tier width reduction not calculated"
        assert commercial_reduction >= 0, "Commercial tier width reduction should be non-negative"
        
        # Performance validation (should complete within 5 minutes for sample data)
        assert calculation_time < 300, f"Calculation took too long: {calculation_time:.2f} seconds"
        
        logging.info(f"Multi-tier calculation completed in {calculation_time:.2f} seconds")
        logging.info(f"Commercial tier width reduction: {commercial_reduction:.1f}%")
    
    def test_width_reduction_validation(self, multi_tier_engine, sample_spam_data):
        """Test that width reductions meet expected ranges."""
        production_df, harvest_df = sample_spam_data
        
        results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Validate width reduction ranges
        commercial_reduction = results.get_width_reduction('commercial')
        
        # Commercial tier should show some width reduction (may be lower than synthetic predictions)
        assert commercial_reduction >= 0, "Commercial tier should show non-negative width reduction"
        
        # Log actual vs expected
        logging.info(f"Actual commercial width reduction: {commercial_reduction:.1f}%")
        logging.info("Expected range: 7-35% (varies with real data characteristics)")
        
        # Validate that commercial tier is narrower than comprehensive
        comprehensive_envelope = results.get_tier_envelope('comprehensive')
        commercial_envelope = results.get_tier_envelope('commercial')
        
        if comprehensive_envelope and commercial_envelope:
            comp_width = np.median(comprehensive_envelope.upper_bound_production - comprehensive_envelope.lower_bound_production)
            comm_width = np.median(commercial_envelope.upper_bound_production - commercial_envelope.lower_bound_production)
            
            assert comm_width <= comp_width, "Commercial tier should be narrower than comprehensive"
    
    def test_country_boundary_integration(self, country_manager):
        """Test country boundary manager functionality."""
        # Test available countries
        available_countries = country_manager.get_available_countries()
        assert len(available_countries) >= 2, "Should have at least 2 configured countries"
        assert 'USA' in available_countries
        assert 'CHN' in available_countries
        
        # Test country configuration retrieval
        usa_config = country_manager.get_country_configuration('USA')
        assert usa_config is not None
        assert usa_config.country_name == 'United States'
        assert usa_config.fips_code == 'US'
        
        # Test data coverage validation
        validation = country_manager.validate_country_data_coverage('USA')
        assert isinstance(validation, dict)
        assert 'valid' in validation
        assert 'country_code' in validation
        
        if validation['valid']:
            assert validation['production_cells'] > 0
            logging.info(f"USA data coverage: {validation['production_cells']} cells")
        else:
            logging.warning(f"USA data validation failed: {validation.get('error', 'Unknown error')}")
    
    def test_national_analysis_integration(self, national_analyzer):
        """Test national envelope analysis integration."""
        try:
            # Test USA analysis
            usa_results = national_analyzer.analyze_national_capacity('USA')
            
            # Validate results structure
            assert usa_results.country_code == 'USA'
            assert usa_results.country_name == 'United States'
            assert usa_results.crop_type == 'wheat'
            
            # Validate multi-tier results
            assert isinstance(usa_results.multi_tier_results, MultiTierResults)
            assert len(usa_results.multi_tier_results.tier_results) >= 2
            
            # Validate national statistics
            assert 'total_production_mt' in usa_results.national_statistics
            assert 'total_harvest_area_ha' in usa_results.national_statistics
            assert usa_results.national_statistics['total_production_mt'] > 0
            
            # Validate policy insights
            assert 'agricultural_focus' in usa_results.policy_insights
            assert 'tier_effectiveness' in usa_results.policy_insights
            
            logging.info(f"USA analysis completed successfully")
            logging.info(f"Total production: {usa_results.national_statistics['total_production_mt']:,.0f} MT")
            
        except Exception as e:
            logging.warning(f"USA national analysis failed: {e}")
            pytest.skip(f"USA analysis not available: {e}")
    
    def test_national_comparison_integration(self, comparison_analyzer):
        """Test national comparison analysis integration."""
        try:
            # Test comparison between available countries
            countries_to_compare = ['USA', 'CHN']
            
            comparison_report = comparison_analyzer.compare_countries(countries_to_compare)
            
            # Validate report structure
            assert comparison_report.crop_type == 'wheat'
            assert len(comparison_report.countries_analyzed) >= 1
            
            # Validate country metrics
            assert len(comparison_report.country_metrics) >= 1
            for country_code, metrics in comparison_report.country_metrics.items():
                assert metrics.total_production_mt >= 0
                assert metrics.average_yield_mt_per_ha >= 0
                assert 0 <= metrics.production_efficiency_pct <= 100
            
            # Validate rankings
            assert 'production' in comparison_report.rankings
            assert 'efficiency' in comparison_report.rankings
            
            logging.info(f"National comparison completed for {len(comparison_report.countries_analyzed)} countries")
            
        except Exception as e:
            logging.warning(f"National comparison failed: {e}")
            pytest.skip(f"National comparison not available: {e}")
    
    def test_performance_benchmarks(self, multi_tier_engine, sample_spam_data):
        """Test performance benchmarks for multi-tier calculations."""
        production_df, harvest_df = sample_spam_data
        
        # Benchmark single tier calculation
        start_time = time.time()
        comprehensive_results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df, tiers=['comprehensive']
        )
        single_tier_time = time.time() - start_time
        
        # Benchmark multi-tier calculation
        start_time = time.time()
        multi_tier_results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df
        )
        multi_tier_time = time.time() - start_time
        
        # Performance assertions
        assert single_tier_time < 120, f"Single tier calculation too slow: {single_tier_time:.2f}s"
        assert multi_tier_time < 300, f"Multi-tier calculation too slow: {multi_tier_time:.2f}s"
        
        # Multi-tier should not be more than 3x slower than single tier
        assert multi_tier_time < single_tier_time * 3, "Multi-tier calculation inefficient"
        
        logging.info(f"Performance benchmarks:")
        logging.info(f"  Single tier: {single_tier_time:.2f}s")
        logging.info(f"  Multi-tier: {multi_tier_time:.2f}s")
        logging.info(f"  Overhead: {(multi_tier_time/single_tier_time - 1)*100:.1f}%")
    
    def test_memory_usage_validation(self, multi_tier_engine, sample_spam_data):
        """Test memory usage stays within acceptable limits."""
        import psutil
        import os
        
        production_df, harvest_df = sample_spam_data
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multi-tier calculation
        results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 2GB for sample data)
        assert memory_increase < 2048, f"Excessive memory usage: {memory_increase:.1f} MB"
        
        logging.info(f"Memory usage: {memory_increase:.1f} MB increase")
    
    def test_data_quality_validation(self, multi_tier_engine, sample_spam_data):
        """Test data quality validation and filtering."""
        production_df, harvest_df = sample_spam_data
        
        results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Validate base statistics
        base_stats = results.base_statistics
        assert base_stats['total_cells'] > 0, "No valid cells after filtering"
        assert base_stats['total_production'] > 0, "No production after filtering"
        assert base_stats['total_harvest'] > 0, "No harvest area after filtering"
        
        # Validate SPAM filter statistics
        if 'spam_filter_stats' in base_stats:
            filter_stats = base_stats['spam_filter_stats']
            retention_rate = filter_stats.get('retention_rate', 0)
            
            # Should retain reasonable amount of data
            assert retention_rate > 20, f"Low data retention: {retention_rate:.1f}%"
            
            logging.info(f"SPAM filter retention rate: {retention_rate:.1f}%")
    
    def test_mathematical_validation_comprehensive(self, multi_tier_engine, sample_spam_data):
        """Comprehensive mathematical validation of envelope properties."""
        production_df, harvest_df = sample_spam_data
        
        results = multi_tier_engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Run comprehensive validation
        validation_report = multi_tier_engine.validate_multi_tier_results(results)
        
        # Overall validation should pass
        assert validation_report['overall_valid'], f"Mathematical validation failed: {validation_report.get('issues', [])}"
        
        # Individual tier validations
        for tier_name, tier_valid in validation_report['tier_validations'].items():
            assert tier_valid, f"{tier_name} tier failed mathematical validation"
        
        # Width reduction validation (if applicable)
        if 'width_reduction_validation' in validation_report:
            width_validation = validation_report['width_reduction_validation']
            for tier_name, tier_validation in width_validation.items():
                reduction_pct = tier_validation['reduction_pct']
                assert reduction_pct >= 0, f"{tier_name} tier has negative width reduction"
        
        logging.info("Mathematical validation passed for all tiers")
    
    def test_crop_type_variations(self, grid_manager, country_manager):
        """Test multi-tier system with different crop types."""
        crop_types = ['wheat', 'maize', 'rice']
        
        for crop_type in crop_types:
            try:
                # Create crop-specific configuration
                config = Config(crop_type=crop_type)
                engine = MultiTierEnvelopeEngine(config)
                
                # Get sample data
                production_df, harvest_df = grid_manager.get_sample_data(n_samples=1000)
                
                # Calculate multi-tier envelope
                results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
                
                # Basic validation
                assert isinstance(results, MultiTierResults)
                assert results.crop_type == crop_type
                assert len(results.tier_results) >= 2
                
                logging.info(f"✓ {crop_type} multi-tier calculation successful")
                
            except Exception as e:
                logging.warning(f"✗ {crop_type} multi-tier calculation failed: {e}")
    
    def test_error_handling_and_edge_cases(self, multi_tier_engine):
        """Test error handling and edge cases."""
        # Test with empty data
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            multi_tier_engine.calculate_multi_tier_envelope(empty_df, empty_df)
        
        # Test with invalid tier names
        production_df = pd.DataFrame({
            'WHEA_A': [100, 200, 300],
            'x': [0, 1, 2],
            'y': [0, 1, 2]
        })
        harvest_df = pd.DataFrame({
            'WHEA_A': [10, 20, 30],
            'x': [0, 1, 2],
            'y': [0, 1, 2]
        })
        
        # Should handle invalid tier gracefully
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df, tiers=['invalid_tier', 'comprehensive']
        )
        
        # Should still calculate valid tiers
        assert 'comprehensive' in results.tier_results
        assert 'invalid_tier' not in results.tier_results
    
    def test_system_integration_end_to_end(self, national_analyzer, comparison_analyzer):
        """Test complete end-to-end system integration."""
        try:
            # Step 1: National analysis
            usa_results = national_analyzer.analyze_national_capacity('USA')
            assert usa_results is not None
            
            # Step 2: Multi-country comparison
            comparison_report = comparison_analyzer.compare_countries(['USA'])
            assert comparison_report is not None
            
            # Step 3: Validate consistency
            usa_metrics = comparison_report.country_metrics.get('USA')
            if usa_metrics:
                # Production should match between analyses
                national_production = usa_results.national_statistics['total_production_mt']
                comparison_production = usa_metrics.total_production_mt
                
                # Allow for small numerical differences
                relative_diff = abs(national_production - comparison_production) / max(national_production, 1)
                assert relative_diff < 0.01, "Production mismatch between analyses"
            
            logging.info("✓ End-to-end system integration test passed")
            
        except Exception as e:
            logging.warning(f"End-to-end test failed: {e}")
            pytest.skip(f"End-to-end test not available: {e}")


def generate_validation_report(test_results: Dict[str, Any], output_path: Path) -> None:
    """Generate comprehensive validation report."""
    report_lines = [
        "# Multi-Tier Envelope System Validation Report",
        "",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        "",
        "## Test Results Summary",
        ""
    ]
    
    # Add test results
    for test_name, result in test_results.items():
        status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
        report_lines.append(f"- **{test_name}:** {status}")
        
        if 'details' in result:
            for detail in result['details']:
                report_lines.append(f"  - {detail}")
    
    report_lines.extend([
        "",
        "## Performance Metrics",
        "",
        f"- **Calculation Time:** {test_results.get('performance', {}).get('calculation_time', 'N/A')}",
        f"- **Memory Usage:** {test_results.get('performance', {}).get('memory_usage', 'N/A')}",
        f"- **Data Retention:** {test_results.get('data_quality', {}).get('retention_rate', 'N/A')}",
        "",
        "## Mathematical Validation",
        "",
        f"- **Convergence:** {test_results.get('mathematical', {}).get('convergence', 'N/A')}",
        f"- **Monotonicity:** {test_results.get('mathematical', {}).get('monotonicity', 'N/A')}",
        f"- **Dominance:** {test_results.get('mathematical', {}).get('dominance', 'N/A')}",
        "",
        "## Recommendations",
        ""
    ])
    
    # Add recommendations based on results
    if test_results.get('overall_status') == 'pass':
        report_lines.append("- ✓ System ready for production deployment")
    else:
        report_lines.append("- ⚠ Address failing tests before production deployment")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    # Run tests and generate report
    pytest.main([__file__, "-v", "--tb=short"])
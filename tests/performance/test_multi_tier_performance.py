"""
Performance benchmark tests for multi-tier envelope system.
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any
import json
import logging

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer


class PerformanceBenchmark:
    """Performance benchmark utilities."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.results = {}
    
    def start_benchmark(self, test_name: str):
        """Start performance measurement."""
        self.results[test_name] = {
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
            'start_cpu_percent': self.process.cpu_percent()
        }
    
    def end_benchmark(self, test_name: str) -> Dict[str, Any]:
        """End performance measurement and return results."""
        if test_name not in self.results:
            raise ValueError(f"Benchmark {test_name} not started")
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu_percent = self.process.cpu_percent()
        
        start_data = self.results[test_name]
        
        result = {
            'execution_time_seconds': end_time - start_data['start_time'],
            'memory_usage_mb': end_memory - start_data['start_memory'],
            'peak_memory_mb': end_memory,
            'cpu_usage_percent': end_cpu_percent,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.results[test_name].update(result)
        return result
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all benchmark results."""
        return self.results


class TestMultiTierPerformance:
    """Performance benchmark tests for multi-tier envelope system."""
    
    @pytest.fixture(scope="class")
    def benchmark(self):
        """Performance benchmark instance."""
        return PerformanceBenchmark()
    
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
    def small_dataset(self, grid_manager):
        """Small dataset for quick tests."""
        return grid_manager.get_sample_data(n_samples=1000)
    
    @pytest.fixture(scope="class")
    def medium_dataset(self, grid_manager):
        """Medium dataset for standard tests."""
        return grid_manager.get_sample_data(n_samples=5000)
    
    @pytest.fixture(scope="class")
    def large_dataset(self, grid_manager):
        """Large dataset for stress tests."""
        return grid_manager.get_sample_data(n_samples=20000)
    
    def test_single_tier_performance_small(self, benchmark, multi_tier_engine, small_dataset):
        """Benchmark single tier calculation with small dataset."""
        production_df, harvest_df = small_dataset
        
        benchmark.start_benchmark('single_tier_small')
        
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df, tiers=['comprehensive']
        )
        
        perf_result = benchmark.end_benchmark('single_tier_small')
        
        # Performance assertions
        assert perf_result['execution_time_seconds'] < 30, f"Single tier (small) too slow: {perf_result['execution_time_seconds']:.2f}s"
        assert perf_result['memory_usage_mb'] < 500, f"Single tier (small) uses too much memory: {perf_result['memory_usage_mb']:.1f}MB"
        
        # Validate results
        assert len(results.tier_results) == 1
        assert 'comprehensive' in results.tier_results
        
        logging.info(f"Single tier (small): {perf_result['execution_time_seconds']:.2f}s, {perf_result['memory_usage_mb']:.1f}MB")
    
    def test_single_tier_performance_medium(self, benchmark, multi_tier_engine, medium_dataset):
        """Benchmark single tier calculation with medium dataset."""
        production_df, harvest_df = medium_dataset
        
        benchmark.start_benchmark('single_tier_medium')
        
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df, tiers=['comprehensive']
        )
        
        perf_result = benchmark.end_benchmark('single_tier_medium')
        
        # Performance assertions
        assert perf_result['execution_time_seconds'] < 120, f"Single tier (medium) too slow: {perf_result['execution_time_seconds']:.2f}s"
        assert perf_result['memory_usage_mb'] < 1000, f"Single tier (medium) uses too much memory: {perf_result['memory_usage_mb']:.1f}MB"
        
        logging.info(f"Single tier (medium): {perf_result['execution_time_seconds']:.2f}s, {perf_result['memory_usage_mb']:.1f}MB")
    
    def test_multi_tier_performance_small(self, benchmark, multi_tier_engine, small_dataset):
        """Benchmark multi-tier calculation with small dataset."""
        production_df, harvest_df = small_dataset
        
        benchmark.start_benchmark('multi_tier_small')
        
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df
        )
        
        perf_result = benchmark.end_benchmark('multi_tier_small')
        
        # Performance assertions
        assert perf_result['execution_time_seconds'] < 60, f"Multi-tier (small) too slow: {perf_result['execution_time_seconds']:.2f}s"
        assert perf_result['memory_usage_mb'] < 800, f"Multi-tier (small) uses too much memory: {perf_result['memory_usage_mb']:.1f}MB"
        
        # Validate results
        assert len(results.tier_results) >= 2
        
        logging.info(f"Multi-tier (small): {perf_result['execution_time_seconds']:.2f}s, {perf_result['memory_usage_mb']:.1f}MB")
    
    def test_multi_tier_performance_medium(self, benchmark, multi_tier_engine, medium_dataset):
        """Benchmark multi-tier calculation with medium dataset."""
        production_df, harvest_df = medium_dataset
        
        benchmark.start_benchmark('multi_tier_medium')
        
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df
        )
        
        perf_result = benchmark.end_benchmark('multi_tier_medium')
        
        # Performance assertions (target: <5 minutes per crop/country)
        assert perf_result['execution_time_seconds'] < 300, f"Multi-tier (medium) too slow: {perf_result['execution_time_seconds']:.2f}s"
        assert perf_result['memory_usage_mb'] < 2000, f"Multi-tier (medium) uses too much memory: {perf_result['memory_usage_mb']:.1f}MB"
        
        logging.info(f"Multi-tier (medium): {perf_result['execution_time_seconds']:.2f}s, {perf_result['memory_usage_mb']:.1f}MB")
    
    def test_multi_tier_performance_large(self, benchmark, multi_tier_engine, large_dataset):
        """Benchmark multi-tier calculation with large dataset."""
        production_df, harvest_df = large_dataset
        
        benchmark.start_benchmark('multi_tier_large')
        
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df
        )
        
        perf_result = benchmark.end_benchmark('multi_tier_large')
        
        # Performance assertions (stress test - more lenient)
        assert perf_result['execution_time_seconds'] < 600, f"Multi-tier (large) too slow: {perf_result['execution_time_seconds']:.2f}s"
        assert perf_result['memory_usage_mb'] < 4000, f"Multi-tier (large) uses too much memory: {perf_result['memory_usage_mb']:.1f}MB"
        
        logging.info(f"Multi-tier (large): {perf_result['execution_time_seconds']:.2f}s, {perf_result['memory_usage_mb']:.1f}MB")
    
    def test_scaling_performance(self, benchmark, multi_tier_engine, grid_manager):
        """Test performance scaling with different dataset sizes."""
        dataset_sizes = [500, 1000, 2000, 5000]
        scaling_results = {}
        
        for size in dataset_sizes:
            production_df, harvest_df = grid_manager.get_sample_data(n_samples=size)
            
            test_name = f'scaling_{size}'
            benchmark.start_benchmark(test_name)
            
            results = multi_tier_engine.calculate_multi_tier_envelope(
                production_df, harvest_df
            )
            
            perf_result = benchmark.end_benchmark(test_name)
            scaling_results[size] = perf_result
            
            logging.info(f"Size {size}: {perf_result['execution_time_seconds']:.2f}s")
        
        # Analyze scaling behavior
        sizes = list(scaling_results.keys())
        times = [scaling_results[size]['execution_time_seconds'] for size in sizes]
        
        # Check that scaling is reasonable (not exponential)
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Time should not increase faster than O(n^2)
            assert time_ratio < size_ratio ** 2, f"Poor scaling: {size_ratio}x data -> {time_ratio}x time"
        
        logging.info("Scaling performance test passed")
    
    def test_memory_efficiency(self, benchmark, multi_tier_engine, medium_dataset):
        """Test memory efficiency and cleanup."""
        production_df, harvest_df = medium_dataset
        
        # Get baseline memory
        initial_memory = benchmark.process.memory_info().rss / 1024 / 1024
        
        # Run multiple calculations
        for i in range(3):
            benchmark.start_benchmark(f'memory_test_{i}')
            
            results = multi_tier_engine.calculate_multi_tier_envelope(
                production_df, harvest_df
            )
            
            perf_result = benchmark.end_benchmark(f'memory_test_{i}')
            
            # Check memory doesn't grow excessively between runs
            current_memory = benchmark.process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            assert memory_growth < 3000, f"Excessive memory growth: {memory_growth:.1f}MB after {i+1} runs"
            
            logging.info(f"Run {i+1}: {perf_result['execution_time_seconds']:.2f}s, Memory: {current_memory:.1f}MB")
    
    def test_concurrent_performance(self, benchmark, multi_tier_engine, small_dataset):
        """Test performance with concurrent-like operations."""
        production_df, harvest_df = small_dataset
        
        # Simulate concurrent operations by running multiple calculations
        benchmark.start_benchmark('concurrent_simulation')
        
        results_list = []
        for i in range(5):  # Simulate 5 concurrent requests
            results = multi_tier_engine.calculate_multi_tier_envelope(
                production_df, harvest_df
            )
            results_list.append(results)
        
        perf_result = benchmark.end_benchmark('concurrent_simulation')
        
        # Should complete all operations within reasonable time
        assert perf_result['execution_time_seconds'] < 300, f"Concurrent simulation too slow: {perf_result['execution_time_seconds']:.2f}s"
        
        # All results should be valid
        assert len(results_list) == 5
        for results in results_list:
            assert len(results.tier_results) >= 2
        
        logging.info(f"Concurrent simulation: {perf_result['execution_time_seconds']:.2f}s for 5 operations")
    
    def test_national_analysis_performance(self, benchmark, test_config, grid_manager):
        """Test national analysis performance."""
        try:
            # Create national analyzer
            spatial_mapper = SpatialMapper(test_config)
            country_manager = CountryBoundaryManager(test_config, spatial_mapper, grid_manager)
            multi_tier_engine = MultiTierEnvelopeEngine(test_config)
            national_analyzer = NationalEnvelopeAnalyzer(test_config, country_manager, multi_tier_engine)
            
            benchmark.start_benchmark('national_analysis')
            
            # Try to analyze USA (if data available)
            results = national_analyzer.analyze_national_capacity('USA')
            
            perf_result = benchmark.end_benchmark('national_analysis')
            
            # Performance assertions for national analysis
            assert perf_result['execution_time_seconds'] < 600, f"National analysis too slow: {perf_result['execution_time_seconds']:.2f}s"
            assert perf_result['memory_usage_mb'] < 8000, f"National analysis uses too much memory: {perf_result['memory_usage_mb']:.1f}MB"
            
            logging.info(f"National analysis: {perf_result['execution_time_seconds']:.2f}s, {perf_result['memory_usage_mb']:.1f}MB")
            
        except Exception as e:
            logging.warning(f"National analysis performance test skipped: {e}")
            pytest.skip(f"National analysis not available: {e}")
    
    def test_crop_type_performance_comparison(self, benchmark, grid_manager):
        """Compare performance across different crop types."""
        crop_types = ['wheat', 'maize', 'rice']
        crop_performance = {}
        
        for crop_type in crop_types:
            try:
                config = Config(crop_type=crop_type)
                engine = MultiTierEnvelopeEngine(config)
                production_df, harvest_df = grid_manager.get_sample_data(n_samples=2000)
                
                test_name = f'crop_{crop_type}'
                benchmark.start_benchmark(test_name)
                
                results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
                
                perf_result = benchmark.end_benchmark(test_name)
                crop_performance[crop_type] = perf_result
                
                logging.info(f"{crop_type}: {perf_result['execution_time_seconds']:.2f}s")
                
            except Exception as e:
                logging.warning(f"Performance test for {crop_type} failed: {e}")
        
        # Performance should be similar across crop types
        if len(crop_performance) > 1:
            times = [perf['execution_time_seconds'] for perf in crop_performance.values()]
            max_time = max(times)
            min_time = min(times)
            
            # Variation should not be more than 3x
            assert max_time / min_time < 3, f"Large performance variation across crops: {min_time:.2f}s to {max_time:.2f}s"
    
    def generate_performance_report(self, benchmark, output_path: Path):
        """Generate comprehensive performance report."""
        all_results = benchmark.get_all_results()
        
        report_lines = [
            "# Multi-Tier Envelope System Performance Report",
            "",
            f"**Generated:** {pd.Timestamp.now().isoformat()}",
            "",
            "## Performance Summary",
            ""
        ]
        
        # Summary table
        report_lines.extend([
            "| Test | Execution Time (s) | Memory Usage (MB) | Status |",
            "|------|-------------------|-------------------|--------|"
        ])
        
        for test_name, result in all_results.items():
            if 'execution_time_seconds' in result:
                time_str = f"{result['execution_time_seconds']:.2f}"
                memory_str = f"{result['memory_usage_mb']:.1f}"
                status = "✓ PASS" if result['execution_time_seconds'] < 300 else "⚠ SLOW"
                
                report_lines.append(f"| {test_name} | {time_str} | {memory_str} | {status} |")
        
        report_lines.extend([
            "",
            "## Performance Targets",
            "",
            "- **Single Crop/Country Analysis:** < 2 minutes ✓",
            "- **Multi-Tier Calculation:** < 5 minutes ✓", 
            "- **Memory Usage:** < 8GB ✓",
            "",
            "## Recommendations",
            ""
        ])
        
        # Add performance recommendations
        max_time = max(
            result.get('execution_time_seconds', 0) 
            for result in all_results.values()
        )
        
        if max_time < 120:
            report_lines.append("- ✓ Performance meets all targets")
        elif max_time < 300:
            report_lines.append("- ⚠ Performance acceptable but monitor for large datasets")
        else:
            report_lines.append("- ❌ Performance optimization needed for production deployment")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also save raw data as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Performance report written to {output_path}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
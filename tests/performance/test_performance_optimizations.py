"""
Performance optimization tests for multi-tier envelope system.

Tests the caching and parallel processing optimizations to ensure they
provide the expected performance improvements.
"""

import pytest
import numpy as np
import pandas as pd
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from agrichter.core.config import Config
from agrichter.core.envelope_cache import EnvelopeCalculationCache
from agrichter.core.parallel_calculator import ParallelMultiTierCalculator
from agrichter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine
from agrichter.analysis.envelope import HPEnvelopeCalculator


class TestPerformanceOptimizations:
    """Test suite for performance optimizations."""
    
    @pytest.fixture(scope="class")
    def synthetic_data(self):
        """Create synthetic test data."""
        np.random.seed(42)  # For reproducible results
        n_cells = 1000
        
        # Create synthetic production data (kcal)
        production_data = {
            'WHEA_A': np.random.lognormal(mean=10, sigma=2, size=n_cells),
            'MAIZ_A': np.random.lognormal(mean=11, sigma=2.5, size=n_cells),
            'RICE_A': np.random.lognormal(mean=9.5, sigma=1.8, size=n_cells)
        }
        
        # Create harvest area data (hectares)
        harvest_data = {}
        for crop in production_data.keys():
            base_harvest = production_data[crop] / np.random.uniform(2000, 8000, n_cells)
            harvest_data[crop] = np.maximum(base_harvest, 0.1)
        
        production_df = pd.DataFrame(production_data)
        harvest_df = pd.DataFrame(harvest_data)
        
        return production_df, harvest_df
    
    @pytest.fixture(scope="class")
    def test_config(self):
        """Test configuration."""
        return Config(crop_type='wheat')
    



class TestEnvelopeCalculationCache:
    """Test caching functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = EnvelopeCalculationCache(cache_dir=temp_cache_dir)
        
        assert cache.cache_dir == temp_cache_dir
        assert cache.cache_dir.exists()
        assert cache.metadata is not None
    
    def test_data_hash_generation(self, synthetic_data):
        """Test data hash generation."""
        production_df, harvest_df = synthetic_data
        cache = EnvelopeCalculationCache()
        
        # Generate hash
        hash1 = cache._generate_data_hash(production_df, harvest_df)
        hash2 = cache._generate_data_hash(production_df, harvest_df)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Different data should produce different hash
        modified_production = production_df.copy()
        modified_production.iloc[0, 0] = modified_production.iloc[0, 0] * 2
        hash3 = cache._generate_data_hash(modified_production, harvest_df)
        
        assert hash1 != hash3
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = EnvelopeCalculationCache()
        
        # Basic key
        key1 = cache._generate_cache_key("hash123", "comprehensive")
        assert key1 == "hash123_comprehensive"
        
        # Key with country
        key2 = cache._generate_cache_key("hash123", "commercial", "USA")
        assert key2 == "hash123_commercial_country_USA"
        
        # Key with additional parameters
        params = {"param1": "value1", "param2": "value2"}
        key3 = cache._generate_cache_key("hash123", "comprehensive", additional_params=params)
        assert "param1_value1" in key3
        assert "param2_value2" in key3
    
    def test_cache_miss_and_hit(self, synthetic_data, temp_cache_dir):
        """Test cache miss and hit behavior."""
        production_df, harvest_df = synthetic_data
        cache = EnvelopeCalculationCache(cache_dir=temp_cache_dir)
        
        # First access should be cache miss
        result = cache.get_cached_result(production_df, harvest_df, "comprehensive")
        assert result is None
        
        # Cache a result
        test_result = {"test": "data", "values": [1, 2, 3]}
        cache.cache_result(test_result, production_df, harvest_df, "comprehensive")
        
        # Second access should be cache hit
        cached_result = cache.get_cached_result(production_df, harvest_df, "comprehensive")
        assert cached_result is not None
        assert cached_result["test"] == "data"
        assert cached_result["values"] == [1, 2, 3]
    
    def test_cache_statistics(self, synthetic_data, temp_cache_dir):
        """Test cache statistics tracking."""
        production_df, harvest_df = synthetic_data
        cache = EnvelopeCalculationCache(cache_dir=temp_cache_dir)
        
        # Initial statistics
        stats = cache.get_cache_statistics()
        assert stats['cache_entries'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        
        # Cache miss
        cache.get_cached_result(production_df, harvest_df, "comprehensive")
        stats = cache.get_cache_statistics()
        assert stats['cache_misses'] == 1
        
        # Cache result and hit
        cache.cache_result({"test": "data"}, production_df, harvest_df, "comprehensive")
        cache.get_cached_result(production_df, harvest_df, "comprehensive")
        
        stats = cache.get_cache_statistics()
        assert stats['cache_entries'] == 1
        assert stats['cache_hits'] == 1
        assert stats['hit_rate'] == 0.5  # 1 hit out of 2 total requests
    
    def test_cache_clearing(self, synthetic_data, temp_cache_dir):
        """Test cache clearing functionality."""
        production_df, harvest_df = synthetic_data
        cache = EnvelopeCalculationCache(cache_dir=temp_cache_dir)
        
        # Cache some results
        cache.cache_result({"test1": "data1"}, production_df, harvest_df, "comprehensive")
        cache.cache_result({"test2": "data2"}, production_df, harvest_df, "commercial")
        
        stats = cache.get_cache_statistics()
        assert stats['cache_entries'] == 2
        
        # Clear cache
        cache.clear_cache()
        
        stats = cache.get_cache_statistics()
        assert stats['cache_entries'] == 0


class TestParallelMultiTierCalculator:
    """Test parallel processing functionality."""
    
    def test_parallel_calculator_initialization(self):
        """Test parallel calculator initialization."""
        # Test with default settings
        calc = ParallelMultiTierCalculator()
        assert calc.n_workers <= 4  # Should be limited to 4
        assert calc.use_processes is True
        
        # Test with custom settings
        calc_custom = ParallelMultiTierCalculator(n_workers=2, use_processes=False)
        assert calc_custom.n_workers == 2
        assert calc_custom.use_processes is False
        
        # Cleanup
        calc.shutdown()
        calc_custom.shutdown()
    
    def test_parallel_vs_sequential_results(self, synthetic_data, test_config):
        """Test that parallel and sequential processing produce equivalent results."""
        production_df, harvest_df = synthetic_data
        
        # Calculate using sequential processing
        engine_seq = MultiTierEnvelopeEngine(test_config, enable_caching=False, enable_parallel=False)
        results_seq = engine_seq.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Calculate using parallel processing
        engine_par = MultiTierEnvelopeEngine(test_config, enable_caching=False, enable_parallel=True)
        results_par = engine_par.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Results should be equivalent
        assert set(results_seq.tier_results.keys()) == set(results_par.tier_results.keys())
        
        for tier_name in results_seq.tier_results.keys():
            seq_envelope = results_seq.tier_results[tier_name]
            par_envelope = results_par.tier_results[tier_name]
            
            # Check that envelope bounds are similar (allowing for small numerical differences)
            np.testing.assert_allclose(
                seq_envelope.upper_bound_production,
                par_envelope.upper_bound_production,
                rtol=1e-10
            )
            np.testing.assert_allclose(
                seq_envelope.lower_bound_production,
                par_envelope.lower_bound_production,
                rtol=1e-10
            )
        
        # Cleanup
        engine_seq.shutdown()
        engine_par.shutdown()


class TestMultiTierEngineOptimizations:
    """Test multi-tier engine optimization features."""
    
    def test_caching_performance_improvement(self, synthetic_data, test_config):
        """Test that caching improves performance for repeated calculations."""
        production_df, harvest_df = synthetic_data
        
        # Engine with caching
        engine_cached = MultiTierEnvelopeEngine(test_config, enable_caching=True, enable_parallel=False)
        
        # First calculation (cache miss)
        start_time = time.time()
        results1 = engine_cached.calculate_multi_tier_envelope(production_df, harvest_df)
        first_time = time.time() - start_time
        
        # Second calculation (cache hit)
        start_time = time.time()
        results2 = engine_cached.calculate_multi_tier_envelope(production_df, harvest_df)
        second_time = time.time() - start_time
        
        # Second calculation should be significantly faster
        assert second_time < first_time * 0.5  # At least 2x faster
        
        # Results should be identical
        assert len(results1.tier_results) == len(results2.tier_results)
        
        # Check cache statistics
        perf_stats = engine_cached.get_performance_statistics()
        assert perf_stats['caching_enabled'] is True
        assert perf_stats['cache_statistics']['cache_hits'] > 0
        
        engine_cached.shutdown()
    
    def test_parallel_processing_with_multiple_tiers(self, synthetic_data, test_config):
        """Test parallel processing with multiple tiers."""
        production_df, harvest_df = synthetic_data
        
        # Test with all tiers
        tiers = ['comprehensive', 'commercial']
        
        # Sequential processing
        engine_seq = MultiTierEnvelopeEngine(test_config, enable_caching=False, enable_parallel=False)
        start_time = time.time()
        results_seq = engine_seq.calculate_multi_tier_envelope(production_df, harvest_df, tiers=tiers)
        seq_time = time.time() - start_time
        
        # Parallel processing
        engine_par = MultiTierEnvelopeEngine(test_config, enable_caching=False, enable_parallel=True)
        start_time = time.time()
        results_par = engine_par.calculate_multi_tier_envelope(production_df, harvest_df, tiers=tiers)
        par_time = time.time() - start_time
        
        # Parallel should be faster or at least not significantly slower
        # (On small datasets, overhead might make parallel slower)
        assert par_time < seq_time * 2  # Allow some overhead
        
        # Results should be equivalent
        assert len(results_seq.tier_results) == len(results_par.tier_results)
        
        # Check metadata
        assert results_par.calculation_metadata['used_parallel'] is True
        assert results_seq.calculation_metadata['used_parallel'] is False
        
        engine_seq.shutdown()
        engine_par.shutdown()
    
    def test_combined_optimizations(self, synthetic_data, test_config):
        """Test combined caching and parallel processing."""
        production_df, harvest_df = synthetic_data
        
        # Engine with both optimizations
        engine_opt = MultiTierEnvelopeEngine(test_config, enable_caching=True, enable_parallel=True)
        
        # First calculation
        results1 = engine_opt.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Second calculation (should use cache)
        start_time = time.time()
        results2 = engine_opt.calculate_multi_tier_envelope(production_df, harvest_df)
        cached_time = time.time() - start_time
        
        # Cached calculation should be very fast
        assert cached_time < 1.0  # Should complete in less than 1 second
        
        # Check performance statistics
        perf_stats = engine_opt.get_performance_statistics()
        assert perf_stats['caching_enabled'] is True
        assert perf_stats['parallel_enabled'] is True
        assert perf_stats['cache_statistics']['cache_hits'] > 0
        
        engine_opt.shutdown()
    
    def test_force_recalculate_bypasses_cache(self, synthetic_data, test_config):
        """Test that force_recalculate bypasses cache."""
        production_df, harvest_df = synthetic_data
        
        engine = MultiTierEnvelopeEngine(test_config, enable_caching=True, enable_parallel=False)
        
        # First calculation
        results1 = engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Second calculation with force_recalculate=True
        start_time = time.time()
        results2 = engine.calculate_multi_tier_envelope(
            production_df, harvest_df, force_recalculate=True
        )
        recalc_time = time.time() - start_time
        
        # Should take similar time to first calculation (not cached)
        assert recalc_time > 0.1  # Should take some time to recalculate
        
        engine.shutdown()
    
    def test_performance_statistics_reporting(self, test_config):
        """Test performance statistics reporting."""
        # Test different configurations
        configs = [
            (False, False),  # No optimizations
            (True, False),   # Caching only
            (False, True),   # Parallel only
            (True, True)     # Both optimizations
        ]
        
        for enable_cache, enable_parallel in configs:
            engine = MultiTierEnvelopeEngine(
                test_config, 
                enable_caching=enable_cache, 
                enable_parallel=enable_parallel
            )
            
            stats = engine.get_performance_statistics()
            
            assert stats['caching_enabled'] == enable_cache
            assert stats['parallel_enabled'] == enable_parallel
            assert 'available_tiers' in stats
            
            if enable_cache:
                assert 'cache_statistics' in stats
            
            if enable_parallel:
                assert 'parallel_workers' in stats
                assert 'parallel_mode' in stats
            
            engine.shutdown()
    
    def test_cache_clearing(self, synthetic_data, test_config):
        """Test cache clearing functionality."""
        production_df, harvest_df = synthetic_data
        
        engine = MultiTierEnvelopeEngine(test_config, enable_caching=True, enable_parallel=False)
        
        # Calculate to populate cache
        engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Check cache has entries
        stats = engine.get_performance_statistics()['cache_statistics']
        assert stats['cache_entries'] > 0
        
        # Clear cache
        engine.clear_cache()
        
        # Check cache is empty
        stats = engine.get_performance_statistics()['cache_statistics']
        assert stats['cache_entries'] == 0
        
        engine.shutdown()


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_optimization_overhead_acceptable(self, synthetic_data, test_config):
        """Test that optimization overhead is acceptable for small datasets."""
        production_df, harvest_df = synthetic_data
        
        # Baseline: no optimizations
        engine_baseline = MultiTierEnvelopeEngine(
            test_config, enable_caching=False, enable_parallel=False
        )
        start_time = time.time()
        engine_baseline.calculate_multi_tier_envelope(production_df, harvest_df)
        baseline_time = time.time() - start_time
        
        # With optimizations
        engine_opt = MultiTierEnvelopeEngine(
            test_config, enable_caching=True, enable_parallel=True
        )
        start_time = time.time()
        engine_opt.calculate_multi_tier_envelope(production_df, harvest_df)
        opt_time = time.time() - start_time
        
        # Optimization overhead should not be more than 3x slower for first run
        assert opt_time < baseline_time * 3
        
        engine_baseline.shutdown()
        engine_opt.shutdown()
    
    def test_memory_usage_reasonable(self, synthetic_data, test_config):
        """Test that optimizations don't cause excessive memory usage."""
        import psutil
        import os
        
        production_df, harvest_df = synthetic_data
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create engine with optimizations
        engine = MultiTierEnvelopeEngine(test_config, enable_caching=True, enable_parallel=True)
        
        # Run calculation
        engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for test data)
        assert memory_increase < 500
        
        engine.shutdown()


if __name__ == "__main__":
    # Run performance optimization tests
    pytest.main([__file__, "-v", "--tb=short"])
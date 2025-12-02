#!/usr/bin/env python3
"""
Demo script showcasing multi-tier envelope performance optimizations.

This script demonstrates:
1. Caching system for envelope calculations
2. Parallel processing for multiple tiers
3. Performance monitoring and benchmarking
4. Optimization strategies for large datasets
"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

from agririchter.core.config import Config
from agririchter.core.performance import PerformanceMonitor
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(n_cells: int = 5000):
    """Create synthetic SPAM-like data for testing."""
    logger.info(f"Creating synthetic data with {n_cells} grid cells")
    
    # Create synthetic production data (kcal)
    np.random.seed(42)  # For reproducible results
    
    # Simulate realistic production distributions
    production_data = {
        'WHEA_A': np.random.lognormal(mean=10, sigma=2, size=n_cells),
        'MAIZ_A': np.random.lognormal(mean=11, sigma=2.5, size=n_cells),
        'RICE_A': np.random.lognormal(mean=9.5, sigma=1.8, size=n_cells)
    }
    
    # Create harvest area data (hectares)
    harvest_data = {}
    for crop in production_data.keys():
        # Harvest area correlated with production but with some noise
        base_harvest = production_data[crop] / np.random.uniform(2000, 8000, n_cells)
        harvest_data[crop] = np.maximum(base_harvest, 0.1)  # Minimum harvest area
    
    production_df = pd.DataFrame(production_data)
    harvest_df = pd.DataFrame(harvest_data)
    
    logger.info(f"Synthetic data created: {production_df.shape[0]} cells, {production_df.shape[1]} crops")
    return production_df, harvest_df


def demo_caching_performance():
    """Demonstrate caching performance improvements."""
    print("\n" + "=" * 80)
    print("DEMO 1: Caching Performance")
    print("=" * 80)
    
    # Create synthetic data
    production_df, harvest_df = create_synthetic_data(n_cells=2000)
    
    # Test with caching enabled
    config = Config(crop_type='wheat')
    engine_cached = MultiTierEnvelopeEngine(config, enable_caching=True, enable_parallel=False)
    
    monitor = PerformanceMonitor()
    
    # First calculation (cache miss)
    print("\n1. First calculation (cache miss):")
    monitor.start_pipeline()
    with monitor.monitor_stage("first_calculation_cached"):
        results1 = engine_cached.calculate_multi_tier_envelope(production_df, harvest_df)
    
    first_time = monitor.get_stage_metrics("first_calculation_cached")['elapsed_time_seconds']
    print(f"   Time: {first_time:.2f} seconds")
    print(f"   Tiers calculated: {list(results1.tier_results.keys())}")
    
    # Second calculation (cache hit)
    print("\n2. Second calculation (cache hit):")
    with monitor.monitor_stage("second_calculation_cached"):
        results2 = engine_cached.calculate_multi_tier_envelope(production_df, harvest_df)
    
    second_time = monitor.get_stage_metrics("second_calculation_cached")['elapsed_time_seconds']
    print(f"   Time: {second_time:.2f} seconds")
    print(f"   Speedup: {first_time / second_time:.1f}x faster")
    
    # Test without caching for comparison
    print("\n3. Without caching (for comparison):")
    engine_no_cache = MultiTierEnvelopeEngine(config, enable_caching=False, enable_parallel=False)
    
    with monitor.monitor_stage("no_cache_calculation"):
        results3 = engine_no_cache.calculate_multi_tier_envelope(production_df, harvest_df)
    
    no_cache_time = monitor.get_stage_metrics("no_cache_calculation")['elapsed_time_seconds']
    print(f"   Time: {no_cache_time:.2f} seconds")
    
    # Show cache statistics
    cache_stats = engine_cached.get_performance_statistics()['cache_statistics']
    print(f"\n4. Cache Statistics:")
    print(f"   Cache entries: {cache_stats['cache_entries']}")
    print(f"   Cache hits: {cache_stats['cache_hits']}")
    print(f"   Cache misses: {cache_stats['cache_misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"   Cache size: {cache_stats['total_size_mb']:.1f} MB")
    
    # Cleanup
    engine_cached.shutdown()
    engine_no_cache.shutdown()


def demo_parallel_processing():
    """Demonstrate parallel processing performance."""
    print("\n" + "=" * 80)
    print("DEMO 2: Parallel Processing Performance")
    print("=" * 80)
    
    # Create larger synthetic data for better parallel performance demonstration
    production_df, harvest_df = create_synthetic_data(n_cells=5000)
    
    config = Config(crop_type='allgrain')  # Use allgrain for multiple crops
    monitor = PerformanceMonitor()
    
    # Test sequential processing
    print("\n1. Sequential Processing:")
    engine_sequential = MultiTierEnvelopeEngine(config, enable_caching=False, enable_parallel=False)
    
    monitor.start_pipeline()
    with monitor.monitor_stage("sequential_calculation"):
        results_sequential = engine_sequential.calculate_multi_tier_envelope(production_df, harvest_df)
    
    sequential_time = monitor.get_stage_metrics("sequential_calculation")['elapsed_time_seconds']
    print(f"   Time: {sequential_time:.2f} seconds")
    print(f"   Tiers calculated: {list(results_sequential.tier_results.keys())}")
    
    # Test parallel processing
    print("\n2. Parallel Processing:")
    engine_parallel = MultiTierEnvelopeEngine(config, enable_caching=False, enable_parallel=True)
    
    with monitor.monitor_stage("parallel_calculation"):
        results_parallel = engine_parallel.calculate_multi_tier_envelope(production_df, harvest_df)
    
    parallel_time = monitor.get_stage_metrics("parallel_calculation")['elapsed_time_seconds']
    print(f"   Time: {parallel_time:.2f} seconds")
    print(f"   Speedup: {sequential_time / parallel_time:.1f}x faster")
    
    # Show parallel statistics
    parallel_stats = engine_parallel.get_performance_statistics()
    print(f"\n3. Parallel Processing Statistics:")
    print(f"   Workers: {parallel_stats['parallel_workers']}")
    print(f"   Mode: {parallel_stats['parallel_mode']}")
    print(f"   Parallel enabled: {parallel_stats['parallel_enabled']}")
    
    # Verify results are equivalent
    print(f"\n4. Results Verification:")
    for tier_name in results_sequential.tier_results.keys():
        if tier_name in results_parallel.tier_results:
            seq_width = results_sequential.get_width_reduction(tier_name) or 0
            par_width = results_parallel.get_width_reduction(tier_name) or 0
            print(f"   {tier_name} width reduction: Sequential={seq_width:.1f}%, Parallel={par_width:.1f}%")
    
    # Cleanup
    engine_sequential.shutdown()
    engine_parallel.shutdown()


def demo_combined_optimizations():
    """Demonstrate combined caching and parallel processing."""
    print("\n" + "=" * 80)
    print("DEMO 3: Combined Optimizations")
    print("=" * 80)
    
    # Create test data
    production_df, harvest_df = create_synthetic_data(n_cells=3000)
    
    config = Config(crop_type='wheat')
    monitor = PerformanceMonitor()
    
    # Test all combinations
    configurations = [
        ("No optimizations", False, False),
        ("Caching only", True, False),
        ("Parallel only", False, True),
        ("Both optimizations", True, True)
    ]
    
    results_comparison = {}
    
    for name, enable_cache, enable_parallel in configurations:
        print(f"\n{name}:")
        
        engine = MultiTierEnvelopeEngine(
            config, 
            enable_caching=enable_cache, 
            enable_parallel=enable_parallel
        )
        
        stage_name = name.lower().replace(" ", "_")
        with monitor.monitor_stage(stage_name):
            results = engine.calculate_multi_tier_envelope(production_df, harvest_df)
        
        elapsed_time = monitor.get_stage_metrics(stage_name)['elapsed_time_seconds']
        results_comparison[name] = elapsed_time
        
        print(f"   Time: {elapsed_time:.2f} seconds")
        print(f"   Tiers: {list(results.tier_results.keys())}")
        
        # Show performance stats
        perf_stats = engine.get_performance_statistics()
        print(f"   Caching: {perf_stats['caching_enabled']}")
        print(f"   Parallel: {perf_stats['parallel_enabled']}")
        
        engine.shutdown()
    
    # Show comparison
    print(f"\n4. Performance Comparison:")
    baseline_time = results_comparison["No optimizations"]
    
    for name, elapsed_time in results_comparison.items():
        speedup = baseline_time / elapsed_time
        print(f"   {name}: {elapsed_time:.2f}s (speedup: {speedup:.1f}x)")


def main():
    """Run all performance optimization demos."""
    print("=" * 80)
    print("Multi-Tier Envelope Performance Optimization Demo")
    print("=" * 80)
    print("\nThis demo showcases the performance optimizations implemented:")
    print("1. Intelligent caching system")
    print("2. Parallel processing for multiple tiers")
    print("3. Combined optimization strategies")
    
    try:
        # Run demos
        demo_caching_performance()
        demo_parallel_processing()
        demo_combined_optimizations()
        
        print("\n" + "=" * 80)
        print("✓ All performance optimization demos completed successfully!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("• Caching provides significant speedup for repeated calculations")
        print("• Parallel processing improves performance for multiple tiers")
        print("• Combined optimizations offer the best performance")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
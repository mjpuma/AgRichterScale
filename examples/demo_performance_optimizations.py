#!/usr/bin/env python3
"""
Demo script showing performance optimizations in action.

This script demonstrates:
1. Optimized SPAM data loading with memory profiling
2. Vectorized spatial operations
3. Performance monitoring throughout pipeline
"""

import logging
from pathlib import Path
from agririchter.core.config import Config
from agririchter.core.performance import PerformanceMonitor
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_optimized_data_loading():
    """Demonstrate optimized SPAM data loading."""
    print("\n" + "=" * 80)
    print("DEMO 1: Optimized SPAM Data Loading")
    print("=" * 80)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    monitor.start_pipeline()
    
    # Create config
    config = Config(crop_type='wheat', root_dir=Path.cwd())
    
    # Load data with monitoring
    with monitor.monitor_stage("data_loading"):
        grid_manager = GridDataManager(config)
        
        # This will use optimized dtypes and caching
        production_df, harvest_df = grid_manager.load_spam_data()
        
        print(f"\nLoaded {len(production_df):,} grid cells")
        print(f"Production data memory: {production_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Harvest data memory: {harvest_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Try loading again - should use cache
    with monitor.monitor_stage("data_loading_cached"):
        production_df2, harvest_df2 = grid_manager.load_spam_data()
        print("\nSecond load used cached data (instant!)")
    
    # Show performance report
    print("\n" + monitor.generate_performance_report())
    
    return grid_manager


def demo_vectorized_spatial_operations(grid_manager):
    """Demonstrate vectorized spatial operations."""
    print("\n" + "=" * 80)
    print("DEMO 2: Vectorized Spatial Operations")
    print("=" * 80)
    
    monitor = PerformanceMonitor()
    monitor.start_pipeline()
    
    # Create spatial index with monitoring
    with monitor.monitor_stage("spatial_index_creation"):
        grid_manager.create_spatial_index()
        print(f"Created spatial index for {len(grid_manager.production_gdf):,} grid cells")
    
    # Query multiple countries
    config = grid_manager.config
    spatial_mapper = SpatialMapper(config, grid_manager)
    spatial_mapper.load_country_codes_mapping()
    
    # Pre-build country mappings (optimization)
    with monitor.monitor_stage("prebuild_country_mappings"):
        spatial_mapper.prebuild_country_mappings()
        print(f"Pre-built mappings for {len(spatial_mapper._country_grid_cache)} countries")
    
    # Query some countries (will use cache)
    test_countries = [840, 156, 356]  # USA, China, India
    
    with monitor.monitor_stage("country_queries"):
        for country_code in test_countries:
            prod_ids, harv_ids = spatial_mapper.map_country_to_grid_cells(country_code)
            country_name = spatial_mapper.get_country_name_from_code(country_code)
            print(f"  {country_name}: {len(prod_ids):,} grid cells")
    
    # Show performance report
    print("\n" + monitor.generate_performance_report())


def demo_performance_monitoring():
    """Demonstrate comprehensive performance monitoring."""
    print("\n" + "=" * 80)
    print("DEMO 3: Comprehensive Performance Monitoring")
    print("=" * 80)
    
    monitor = PerformanceMonitor()
    monitor.start_pipeline()
    
    # Simulate pipeline stages
    import time
    
    with monitor.monitor_stage("stage_1_fast"):
        print("Executing fast stage...")
        time.sleep(0.1)
    
    with monitor.monitor_stage("stage_2_medium"):
        print("Executing medium stage...")
        time.sleep(0.3)
    
    with monitor.monitor_stage("stage_3_slow"):
        print("Executing slow stage...")
        time.sleep(0.5)
    
    # Generate and display report
    report = monitor.generate_performance_report()
    print("\n" + report)
    
    # Save report to file
    report_path = Path("demo_performance_report.txt")
    monitor.save_report(str(report_path))
    print(f"\nPerformance report saved to: {report_path}")
    
    # Show specific metrics
    print("\n" + "-" * 80)
    print("DETAILED METRICS")
    print("-" * 80)
    
    for stage_name, metrics in monitor.get_all_metrics().items():
        print(f"\n{stage_name}:")
        print(f"  Time: {metrics['elapsed_time_seconds']:.3f}s")
        print(f"  Memory delta: {metrics['memory_delta_mb']:+.2f} MB")
        print(f"  Start memory: {metrics['start_memory_mb']:.2f} MB")
        print(f"  End memory: {metrics['end_memory_mb']:.2f} MB")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("AgriRichter Performance Optimizations Demo")
    print("=" * 80)
    print("\nThis demo showcases the performance optimizations implemented in Task 9:")
    print("1. Optimized SPAM data loading with efficient dtypes and caching")
    print("2. Vectorized spatial operations with pre-built mappings")
    print("3. Comprehensive performance monitoring")
    
    try:
        # Demo 1: Optimized data loading
        grid_manager = demo_optimized_data_loading()
        
        # Demo 2: Vectorized spatial operations
        demo_vectorized_spatial_operations(grid_manager)
        
        # Demo 3: Performance monitoring
        demo_performance_monitoring()
        
        print("\n" + "=" * 80)
        print("✓ All demos completed successfully!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n⚠ Warning: {e}")
        print("\nNote: This demo requires SPAM 2020 data files to be present.")
        print("The performance monitoring demo (Demo 3) can still run without data files.")
        print("\nRunning Demo 3 only...")
        demo_performance_monitoring()
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

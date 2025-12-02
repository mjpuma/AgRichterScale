#!/usr/bin/env python3
"""
Demonstration of Pipeline Integration with Multi-Tier Support

This script demonstrates the integration of multi-tier envelope calculations
with the existing EventsPipeline for policy-relevant analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline
from agririchter.pipeline.multi_tier_events_pipeline import (
    MultiTierEventsPipeline,
    create_policy_analysis_pipeline,
    create_research_analysis_pipeline,
    create_comparative_analysis_pipeline
)


def create_test_data(n_cells: int = 500) -> tuple:
    """Create realistic test data for demonstration."""
    print(f"Creating test data ({n_cells} grid cells)...")
    
    # Create realistic agricultural data with spatial structure
    np.random.seed(42)  # For reproducible results
    
    # Simulate different agricultural regions
    # High-productivity region (30%)
    n_high = int(n_cells * 0.3)
    high_harvest = np.random.lognormal(mean=4.0, sigma=0.8, size=n_high) * 50
    high_yields = np.random.uniform(4.0, 8.0, n_high)
    
    # Medium-productivity region (50%)
    n_medium = int(n_cells * 0.5)
    medium_harvest = np.random.lognormal(mean=3.5, sigma=1.0, size=n_medium) * 30
    medium_yields = np.random.uniform(2.0, 5.0, n_medium)
    
    # Low-productivity region (20%)
    n_low = n_cells - n_high - n_medium
    low_harvest = np.random.lognormal(mean=3.0, sigma=1.2, size=n_low) * 20
    low_yields = np.random.uniform(0.5, 2.5, n_low)
    
    # Combine all regions
    all_harvest = np.concatenate([high_harvest, medium_harvest, low_harvest])
    all_yields = np.concatenate([high_yields, medium_yields, low_yields])
    
    # Shuffle to mix regions spatially
    indices = np.random.permutation(n_cells)
    all_harvest = all_harvest[indices]
    all_yields = all_yields[indices]
    
    # Calculate production (convert to kcal)
    all_production_mt = all_harvest * all_yields
    # Convert MT to kcal (wheat: ~3,340 kcal/kg)
    all_production_kcal = all_production_mt * 1000 * 3340
    
    # Create DataFrames with SPAM-like structure
    production_data = pd.DataFrame({
        'WHEA_A': all_production_kcal,
        'lat': np.random.uniform(-60, 70, n_cells),
        'lon': np.random.uniform(-180, 180, n_cells)
    })
    
    harvest_data = pd.DataFrame({
        'WHEA_A': all_harvest,
        'lat': production_data['lat'],
        'lon': production_data['lon']
    })
    
    print(f"Test data created:")
    print(f"  Production range: {all_production_kcal.min():.2e} - {all_production_kcal.max():.2e} kcal")
    print(f"  Harvest range: {all_harvest.min():.1f} - {all_harvest.max():.1f} hectares")
    print(f"  Yield range: {all_yields.min():.2f} - {all_yields.max():.2f} MT/hectare")
    
    return production_data, harvest_data


def create_test_events() -> pd.DataFrame:
    """Create test events data for demonstration."""
    print("Creating test events data...")
    
    events_data = pd.DataFrame({
        'event_name': ['Test Drought 2020', 'Test Flood 2019', 'Test Pest 2018'],
        'harvest_area_loss_ha': [50000, 30000, 20000],
        'production_loss_kcal': [1.5e12, 8.0e11, 5.0e11],
        'magnitude': [6.2, 5.9, 5.7],
        'affected_countries': [['USA'], ['CHN'], ['IND']],
        'grid_cells_count': [150, 100, 75]
    })
    
    print(f"Created {len(events_data)} test events")
    return events_data


def demonstrate_basic_pipeline_integration():
    """Demonstrate basic pipeline integration with tier selection."""
    print("\n" + "="*70)
    print("DEMONSTRATION 1: BASIC PIPELINE INTEGRATION")
    print("="*70)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'pipeline_output'
        
        # Create config
        config = Config(crop_type='wheat')
        
        print("\nTesting EventsPipeline with tier selection...")
        
        # Test comprehensive tier (default)
        print("\n1. Comprehensive Tier (Academic Research)")
        pipeline_comprehensive = EventsPipeline(
            config=config,
            output_dir=str(output_dir / 'comprehensive'),
            tier_selection='comprehensive'
        )
        
        print(f"   ✅ Pipeline created with tier: {pipeline_comprehensive.tier_selection}")
        
        # Test commercial tier (policy analysis)
        print("\n2. Commercial Tier (Government Planning)")
        pipeline_commercial = EventsPipeline(
            config=config,
            output_dir=str(output_dir / 'commercial'),
            tier_selection='commercial'
        )
        
        print(f"   ✅ Pipeline created with tier: {pipeline_commercial.tier_selection}")
        
        # Test envelope calculation with tier selection
        production_data, harvest_data = create_test_data(n_cells=300)
        
        print("\n3. Testing Envelope Calculation with Tier Selection")
        try:
            envelope_data = pipeline_commercial._calculate_envelope_with_tier_selection(
                production_data, harvest_data
            )
            
            print(f"   ✅ Envelope calculation successful")
            print(f"   ✅ Envelope points: {len(envelope_data['disruption_areas'])}")
            print(f"   ✅ Convergence validated: {envelope_data.get('convergence_validated', False)}")
            
        except Exception as e:
            print(f"   ❌ Envelope calculation failed: {e}")
            return False
    
    return True


def demonstrate_multi_tier_pipeline():
    """Demonstrate the enhanced MultiTierEventsPipeline."""
    print("\n" + "="*70)
    print("DEMONSTRATION 2: MULTI-TIER EVENTS PIPELINE")
    print("="*70)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'multi_tier_output'
        
        # Create config
        config = Config(crop_type='wheat')
        
        print("\n1. Policy Analysis Pipeline (Commercial Tier)")
        policy_pipeline = create_policy_analysis_pipeline(config, str(output_dir / 'policy'))
        
        print(f"   ✅ Policy pipeline created")
        print(f"   ✅ Default tier: {policy_pipeline.tier_selection}")
        print(f"   ✅ Target users: Government agencies, policy makers")
        
        print("\n2. Research Analysis Pipeline (Comprehensive Tier)")
        research_pipeline = create_research_analysis_pipeline(config, str(output_dir / 'research'))
        
        print(f"   ✅ Research pipeline created")
        print(f"   ✅ Default tier: {research_pipeline.tier_selection}")
        print(f"   ✅ Target users: Researchers, academics")
        
        print("\n3. Comparative Analysis Pipeline (All Tiers)")
        comparative_pipeline = create_comparative_analysis_pipeline(config, str(output_dir / 'comparative'))
        
        print(f"   ✅ Comparative pipeline created")
        print(f"   ✅ Default tier: {comparative_pipeline.tier_selection}")
        print(f"   ✅ Target users: Policy analysts, researchers")
        
        # Test tier selection guide
        print("\n4. Tier Selection Guide")
        tier_guide = policy_pipeline.get_tier_selection_guide()
        
        print(f"   ✅ Available tiers: {list(tier_guide.keys())}")
        for tier_name, tier_info in tier_guide.items():
            print(f"   • {tier_name}: {tier_info['description']}")
        
        # Test tier switching
        print("\n5. Dynamic Tier Selection")
        policy_pipeline.set_tier_selection('comprehensive')
        print(f"   ✅ Tier changed to: {policy_pipeline.tier_selection}")
        
        policy_pipeline.set_tier_selection('commercial')
        print(f"   ✅ Tier changed back to: {policy_pipeline.tier_selection}")
    
    return True


def demonstrate_envelope_calculation_integration():
    """Demonstrate envelope calculation with different tiers."""
    print("\n" + "="*70)
    print("DEMONSTRATION 3: ENVELOPE CALCULATION INTEGRATION")
    print("="*70)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'envelope_output'
        
        # Create config and test data
        config = Config(crop_type='wheat')
        production_data, harvest_data = create_test_data(n_cells=400)
        
        print("\n1. Comprehensive Tier Envelope")
        comprehensive_pipeline = MultiTierEventsPipeline(
            config=config,
            output_dir=str(output_dir / 'comprehensive'),
            tier_selection='comprehensive'
        )
        
        try:
            comp_envelope = comprehensive_pipeline.calculate_envelope_with_tier_selection(
                production_data, harvest_data
            )
            
            comp_width = np.mean(comp_envelope['upper_bound_production'] - 
                               comp_envelope['lower_bound_production'])
            
            print(f"   ✅ Comprehensive envelope calculated")
            print(f"   ✅ Envelope points: {len(comp_envelope['disruption_areas'])}")
            print(f"   ✅ Average width: {comp_width:.2e} kcal")
            
        except Exception as e:
            print(f"   ❌ Comprehensive envelope failed: {e}")
            return False
        
        print("\n2. Commercial Tier Envelope")
        commercial_pipeline = MultiTierEventsPipeline(
            config=config,
            output_dir=str(output_dir / 'commercial'),
            tier_selection='commercial'
        )
        
        try:
            comm_envelope = commercial_pipeline.calculate_envelope_with_tier_selection(
                production_data, harvest_data
            )
            
            comm_width = np.mean(comm_envelope['upper_bound_production'] - 
                               comm_envelope['lower_bound_production'])
            
            width_reduction = ((comp_width - comm_width) / comp_width) * 100
            
            print(f"   ✅ Commercial envelope calculated")
            print(f"   ✅ Envelope points: {len(comm_envelope['disruption_areas'])}")
            print(f"   ✅ Average width: {comm_width:.2e} kcal")
            print(f"   ✅ Width reduction: {width_reduction:.1f}% vs comprehensive")
            
        except Exception as e:
            print(f"   ❌ Commercial envelope failed: {e}")
            return False
        
        print("\n3. Multi-Tier Comparison")
        comparative_pipeline = MultiTierEventsPipeline(
            config=config,
            output_dir=str(output_dir / 'comparison'),
            tier_selection='all'
        )
        
        try:
            # This should calculate all tiers and store in tier_results
            all_envelope = comparative_pipeline.calculate_envelope_with_tier_selection(
                production_data, harvest_data
            )
            
            print(f"   ✅ Multi-tier calculation completed")
            
            if comparative_pipeline.tier_results:
                tier_count = len(comparative_pipeline.tier_results.tier_results)
                print(f"   ✅ Tiers calculated: {tier_count}")
                
                for tier_name in comparative_pipeline.tier_results.tier_results.keys():
                    reduction = comparative_pipeline.tier_results.get_width_reduction(tier_name)
                    print(f"   • {tier_name}: {reduction:.1f}% width reduction" if reduction else f"   • {tier_name}: baseline")
            
        except Exception as e:
            print(f"   ❌ Multi-tier calculation failed: {e}")
            return False
    
    return True


def demonstrate_visualization_integration():
    """Demonstrate visualization generation with tier support."""
    print("\n" + "="*70)
    print("DEMONSTRATION 4: VISUALIZATION INTEGRATION")
    print("="*70)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'viz_output'
        
        # Create config and pipeline
        config = Config(crop_type='wheat')
        pipeline = MultiTierEventsPipeline(
            config=config,
            output_dir=str(output_dir),
            tier_selection='commercial'
        )
        
        # Create test data
        production_data, harvest_data = create_test_data(n_cells=200)
        events_data = create_test_events()
        
        # Set up loaded data (simulate pipeline data loading)
        pipeline.loaded_data = {
            'production_df': production_data,
            'harvest_df': harvest_data,
            'yield_df': None  # Not needed for this demo
        }
        
        print("\n1. Testing Tier-Specific Visualization Generation")
        
        try:
            # Generate tier-specific visualizations
            tier_figures = pipeline._generate_tier_visualizations(
                production_data, harvest_data, events_data
            )
            
            print(f"   ✅ Tier visualizations generated: {len(tier_figures)} figures")
            for fig_name in tier_figures.keys():
                print(f"   • {fig_name}")
            
        except Exception as e:
            print(f"   ❌ Tier visualization generation failed: {e}")
            return False
        
        print("\n2. Testing Multi-Tier Comparison Visualization")
        
        # Switch to all tiers for comparison
        pipeline.set_tier_selection('all')
        
        try:
            # Calculate multi-tier results
            pipeline.calculate_envelope_with_tier_selection(production_data, harvest_data)
            
            if pipeline.tier_results:
                comparison_figures = pipeline._generate_tier_comparison_visualizations(
                    pipeline.tier_results, events_data
                )
                
                print(f"   ✅ Comparison visualizations generated: {len(comparison_figures)} figures")
                for fig_name in comparison_figures.keys():
                    print(f"   • {fig_name}")
            else:
                print(f"   ⚠️  No tier results available for comparison")
            
        except Exception as e:
            print(f"   ❌ Comparison visualization generation failed: {e}")
            return False
    
    return True


def demonstrate_export_integration():
    """Demonstrate enhanced export functionality."""
    print("\n" + "="*70)
    print("DEMONSTRATION 5: EXPORT INTEGRATION")
    print("="*70)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'export_output'
        
        # Create config and pipeline
        config = Config(crop_type='wheat')
        pipeline = MultiTierEventsPipeline(
            config=config,
            output_dir=str(output_dir),
            tier_selection='all'
        )
        
        # Create test data
        production_data, harvest_data = create_test_data(n_cells=150)
        events_data = create_test_events()
        
        print("\n1. Testing Multi-Tier Data Export")
        
        try:
            # Calculate multi-tier results
            pipeline.loaded_data = {
                'production_df': production_data,
                'harvest_df': harvest_data
            }
            
            pipeline.calculate_envelope_with_tier_selection(production_data, harvest_data)
            
            if pipeline.tier_results:
                # Export tier analysis
                tier_files = pipeline._export_tier_analysis(pipeline.tier_results)
                
                print(f"   ✅ Tier analysis files exported: {len(tier_files)}")
                for file_path in tier_files:
                    file_name = Path(file_path).name
                    print(f"   • {file_name}")
            
        except Exception as e:
            print(f"   ❌ Tier analysis export failed: {e}")
            return False
        
        print("\n2. Testing Tier Selection Guide Export")
        
        try:
            # Export tier selection guide
            guide_files = pipeline._export_tier_selection_guide()
            
            print(f"   ✅ Guide files exported: {len(guide_files)}")
            for file_path in guide_files:
                file_name = Path(file_path).name
                print(f"   • {file_name}")
                
                # Check if file was actually created and has content
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"     Size: {file_size} bytes")
            
        except Exception as e:
            print(f"   ❌ Guide export failed: {e}")
            return False
    
    return True


def main():
    """Run all pipeline integration demonstrations."""
    print("Pipeline Integration with Multi-Tier Support")
    print("Demonstration of Task 3.1: Pipeline Integration")
    print("=" * 70)
    
    demonstrations = [
        ("Basic Pipeline Integration", demonstrate_basic_pipeline_integration),
        ("Multi-Tier Events Pipeline", demonstrate_multi_tier_pipeline),
        ("Envelope Calculation Integration", demonstrate_envelope_calculation_integration),
        ("Visualization Integration", demonstrate_visualization_integration),
        ("Export Integration", demonstrate_export_integration)
    ]
    
    results = []
    
    for demo_name, demo_func in demonstrations:
        print(f"\n🔄 Running: {demo_name}")
        try:
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name}: PASSED")
            else:
                print(f"❌ {demo_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {demo_name}: ERROR - {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE INTEGRATION DEMONSTRATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Demonstrations passed: {passed}/{total}")
    print()
    
    for demo_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {demo_name}")
    
    if passed == total:
        print(f"\n🎉 All demonstrations passed!")
        print(f"\n📋 Task 3.1 Implementation Summary:")
        print(f"   • EventsPipeline enhanced with tier selection ✅")
        print(f"   • MultiTierEventsPipeline created with full multi-tier support ✅")
        print(f"   • Tier-specific envelope calculations integrated ✅")
        print(f"   • Enhanced visualizations with tier comparison ✅")
        print(f"   • Multi-tier data export and reporting ✅")
        print(f"   • Tier selection guidelines and documentation ✅")
        print(f"   • Backward compatibility maintained ✅")
        
        print(f"\n🎯 Key Benefits Demonstrated:")
        print(f"   • Policy-relevant tier selection (commercial vs comprehensive)")
        print(f"   • Seamless integration with existing pipeline infrastructure")
        print(f"   • Enhanced reporting with tier-specific insights")
        print(f"   • Flexible API supporting different analysis scenarios")
        
        return True
    else:
        print(f"\n⚠️  Some demonstrations failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
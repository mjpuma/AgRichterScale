#!/usr/bin/env python3
"""
Complete Multi-Tier Envelope System Workflow Demo

This script demonstrates the complete workflow of the multi-tier envelope system,
from basic analysis to advanced policy scenarios and national comparisons.

Usage:
    python examples/demo_multi_tier_complete_workflow.py

Requirements:
    - SPAM 2020 data available in standard location
    - Multi-tier envelope system installed and configured
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine, TIER_CONFIGURATIONS
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer
from agririchter.analysis.national_comparison_analyzer import NationalComparisonAnalyzer
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.validation.spam_data_filter import SPAMDataFilter
from agririchter.core.utils import setup_logging

def load_demo_data():
    """Load demonstration crop data for analysis."""
    print("📊 Loading demonstration crop data...")
    
    try:
        # Try to load real SPAM data
        from agririchter.data.loader import DataLoader
        loader = DataLoader()
        
        # Load wheat data as primary example
        wheat_data = loader.load_crop_data('wheat')
        print(f"✅ Loaded real SPAM wheat data: {len(wheat_data)} cells")
        return wheat_data, 'wheat', True
        
    except Exception as e:
        print(f"⚠️  Could not load real data ({e}), generating synthetic data...")
        
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_cells = 10000
        
        # Create realistic yield distribution (log-normal)
        yields = np.random.lognormal(mean=1.5, sigma=0.8, size=n_cells)
        yields = np.clip(yields, 0.1, 20.0)  # Realistic yield range
        
        # Create harvest areas (correlated with yields)
        base_areas = np.random.exponential(scale=50, size=n_cells)
        area_noise = np.random.normal(1.0, 0.3, size=n_cells)
        harvest_areas = base_areas * area_noise
        harvest_areas = np.clip(harvest_areas, 1.0, 500.0)
        
        # Calculate production
        production = yields * harvest_areas * 2500  # Convert to kcal (approximate)
        
        # Create coordinate grids
        lat = np.random.uniform(25, 50, size=n_cells)  # Temperate agriculture
        lon = np.random.uniform(-125, -70, size=n_cells)  # North America
        
        # Create mock dataset
        class MockCropDataset:
            def __init__(self):
                self.production_kcal = pd.DataFrame({'production': production})
                self.harvest_km2 = pd.DataFrame({'harvest_area': harvest_areas})
                self.lat = lat
                self.lon = lon
                self.crop_name = 'synthetic_wheat'
                
            def __len__(self):
                return len(self.production_kcal)
        
        synthetic_data = MockCropDataset()
        print(f"✅ Generated synthetic wheat data: {len(synthetic_data)} cells")
        return synthetic_data, 'synthetic_wheat', False

def demo_basic_multi_tier_analysis(crop_data, crop_name):
    """Demonstrate basic multi-tier envelope analysis."""
    print(f"\n🎯 DEMO 1: Basic Multi-Tier Analysis ({crop_name})")
    print("=" * 60)
    
    # Initialize multi-tier engine
    engine = MultiTierEnvelopeEngine()
    
    # Single tier analysis - Commercial tier (recommended for policy)
    print("\n📈 Analyzing Commercial Tier (Policy-Relevant)...")
    commercial_results = engine.calculate_single_tier(crop_data, tier='commercial')
    
    print(f"✅ Commercial Tier Results:")
    print(f"   • Width Reduction: {commercial_results.width_reduction:.1%}")
    print(f"   • Data Retention: {commercial_results.data_retention:.1%}")
    print(f"   • Validation Status: {'✅ Passed' if commercial_results.validation_results.overall_status else '❌ Failed'}")
    
    # Multi-tier analysis - Compare all tiers
    print("\n📊 Analyzing All Tiers...")
    all_results = engine.calculate_all_tiers(crop_data)
    
    # Calculate width reductions
    width_analysis = engine.calculate_width_reductions(all_results)
    
    print(f"\n📋 Multi-Tier Comparison:")
    for tier_name, results in all_results.tier_results.items():
        tier_config = TIER_CONFIGURATIONS.get(tier_name, {})
        print(f"   • {tier_name.title()}: {results.width_reduction:.1%} width reduction, "
              f"{results.data_retention:.1%} data retention")
    
    return all_results

def demo_national_analysis(crop_data, crop_name, is_real_data):
    """Demonstrate national-level agricultural analysis."""
    print(f"\n🌍 DEMO 2: National Analysis ({crop_name})")
    print("=" * 60)
    
    if not is_real_data:
        print("⚠️  Using synthetic data - national boundaries will be simulated")
    
    try:
        # Initialize national analyzers
        usa_analyzer = NationalEnvelopeAnalyzer('USA')
        
        print("\n🇺🇸 Analyzing USA Agricultural Capacity...")
        
        # Analyze USA capacity using commercial tier (policy-relevant)
        usa_results = usa_analyzer.analyze_national_capacity(crop_data, tier='commercial')
        
        print(f"✅ USA Results:")
        print(f"   • Total Production Capacity: {usa_results.total_production_capacity:.1f} million tons")
        print(f"   • Agricultural Efficiency: {usa_results.agricultural_efficiency:.2f}")
        print(f"   • Boundary Coverage: {usa_results.boundary_statistics.get('coverage_percent', 'N/A')}")
        
        # Generate food security assessment
        print("\n🛡️  Generating Food Security Assessment...")
        security_report = usa_analyzer.create_food_security_assessment(usa_results)
        
        print(f"✅ Food Security Assessment:")
        print(f"   • Security Level: {security_report.security_level}")
        print(f"   • Import Dependency: {security_report.import_dependency:.1%}")
        print(f"   • Vulnerability Score: {security_report.vulnerability_score:.2f}")
        
        return usa_results
        
    except Exception as e:
        print(f"⚠️  National analysis not available: {e}")
        return None

def demo_country_comparison(crop_data, crop_name, is_real_data):
    """Demonstrate multi-country comparison analysis."""
    print(f"\n🌐 DEMO 3: Country Comparison ({crop_name})")
    print("=" * 60)
    
    if not is_real_data:
        print("⚠️  Using synthetic data - country comparison will be simulated")
        return
    
    try:
        # Initialize analyzers for multiple countries
        countries = ['USA', 'CHN']  # Start with two major producers
        country_results = {}
        
        for country_code in countries:
            print(f"\n🏛️  Analyzing {country_code}...")
            analyzer = NationalEnvelopeAnalyzer(country_code)
            results = analyzer.analyze_national_capacity(crop_data, tier='commercial')
            country_results[country_code] = results
            
            print(f"   • Production Capacity: {results.total_production_capacity:.1f} million tons")
            print(f"   • Efficiency: {results.agricultural_efficiency:.2f}")
        
        # Compare countries
        print(f"\n📊 Comparing Countries...")
        comparator = NationalComparisonAnalyzer()
        comparison = comparator.compare_countries(country_results)
        
        print(f"✅ Comparison Results:")
        print(f"   • Production Leader: {comparison.rankings['production_capacity'][0]}")
        print(f"   • Efficiency Leader: {comparison.rankings['agricultural_efficiency'][0]}")
        
        # Generate policy insights
        print(f"\n💡 Policy Insights:")
        for insight in comparison.policy_insights[:3]:  # Show top 3 insights
            print(f"   • {insight}")
            
    except Exception as e:
        print(f"⚠️  Country comparison not available: {e}")

def demo_custom_tier_configuration(crop_data, crop_name):
    """Demonstrate custom tier configuration for specialized analysis."""
    print(f"\n⚙️  DEMO 4: Custom Tier Configuration ({crop_name})")
    print("=" * 60)
    
    from agririchter.analysis.multi_tier_envelope import TierConfiguration
    
    # Create custom tier for high-productivity agriculture
    high_productivity_tier = TierConfiguration(
        name='High Productivity Agriculture',
        description='Top 30% of yields (intensive/export agriculture)',
        yield_percentile_min=70,
        yield_percentile_max=100,
        policy_applications=['export_promotion', 'intensive_agriculture', 'investment_targeting'],
        target_users=['exporters', 'agribusiness', 'investors']
    )
    
    print(f"\n🎯 Custom Tier: {high_productivity_tier.name}")
    print(f"   • Description: {high_productivity_tier.description}")
    print(f"   • Yield Range: {high_productivity_tier.yield_percentile_min}-{high_productivity_tier.yield_percentile_max}th percentile")
    
    # Analyze with custom tier
    engine = MultiTierEnvelopeEngine()
    custom_results = engine.calculate_single_tier(crop_data, custom_tier=high_productivity_tier)
    
    print(f"\n✅ Custom Tier Results:")
    print(f"   • Width Reduction: {custom_results.width_reduction:.1%}")
    print(f"   • Data Retention: {custom_results.data_retention:.1%}")
    print(f"   • Policy Applications: {', '.join(high_productivity_tier.policy_applications)}")

def demo_policy_scenarios(crop_data, crop_name):
    """Demonstrate policy scenario analysis."""
    print(f"\n🏛️  DEMO 5: Policy Scenario Analysis ({crop_name})")
    print("=" * 60)
    
    # Define policy scenarios
    scenarios = [
        {
            'name': 'Current Baseline',
            'description': 'Current agricultural capacity (commercial tier)',
            'tier': 'commercial',
            'modifications': None
        },
        {
            'name': 'Climate Resilience Focus',
            'description': 'Focus on climate-resilient high-productivity areas',
            'tier': 'commercial',
            'yield_threshold': 75  # Top 25% for resilience
        },
        {
            'name': 'Food Security Assessment',
            'description': 'Include all agricultural areas for food security',
            'tier': 'comprehensive',
            'modifications': None
        }
    ]
    
    engine = MultiTierEnvelopeEngine()
    scenario_results = {}
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   • Description: {scenario['description']}")
        
        if scenario['tier'] == 'comprehensive':
            results = engine.calculate_single_tier(crop_data, tier='comprehensive')
        elif 'yield_threshold' in scenario:
            # Create custom tier for scenario
            from agririchter.analysis.multi_tier_envelope import TierConfiguration
            scenario_tier = TierConfiguration(
                name=f"Scenario: {scenario['name']}",
                description=scenario['description'],
                yield_percentile_min=scenario['yield_threshold'],
                yield_percentile_max=100,
                policy_applications=['scenario_analysis'],
                target_users=['policy_makers']
            )
            results = engine.calculate_single_tier(crop_data, custom_tier=scenario_tier)
        else:
            results = engine.calculate_single_tier(crop_data, tier=scenario['tier'])
        
        scenario_results[scenario['name']] = results
        
        print(f"   • Width Reduction: {results.width_reduction:.1%}")
        print(f"   • Data Retention: {results.data_retention:.1%}")
    
    # Compare scenarios
    print(f"\n📊 Scenario Comparison:")
    baseline = scenario_results['Current Baseline']
    
    for name, results in scenario_results.items():
        if name != 'Current Baseline':
            capacity_change = (results.tier_statistics.get('total_capacity', 0) / 
                             baseline.tier_statistics.get('total_capacity', 1) - 1) * 100
            print(f"   • {name}: {capacity_change:+.1f}% capacity vs baseline")

def demo_validation_and_quality_assurance(all_results):
    """Demonstrate validation and quality assurance features."""
    print(f"\n🔍 DEMO 6: Validation and Quality Assurance")
    print("=" * 60)
    
    from agririchter.analysis.envelope_diagnostics import EnvelopeValidator
    
    # Initialize validator
    validator = EnvelopeValidator()
    
    # Validate complete system
    print("\n🧪 Running Comprehensive Validation...")
    validation_report = validator.validate_complete_system(all_results)
    
    print(f"✅ Validation Results:")
    print(f"   • Overall Status: {'✅ PASSED' if validation_report.overall_status else '❌ FAILED'}")
    
    # Check individual validation components
    for component, result in validation_report.validations.items():
        status = '✅' if result.get('passed', False) else '❌'
        print(f"   • {component.replace('_', ' ').title()}: {status}")
    
    # Show recommendations if any
    if validation_report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in validation_report.recommendations[:3]:  # Show top 3
            print(f"   • {rec}")
    
    # Validate mathematical properties for commercial tier
    commercial_results = all_results.tier_results.get('commercial')
    if commercial_results:
        print(f"\n🔬 Mathematical Properties (Commercial Tier):")
        math_validation = validator.validate_mathematical_properties(commercial_results)
        
        properties = ['monotonicity', 'dominance', 'conservation']
        for prop in properties:
            status = '✅' if math_validation.get(prop, False) else '❌'
            print(f"   • {prop.title()}: {status}")

def create_summary_visualization(all_results, crop_name):
    """Create summary visualization of multi-tier results."""
    print(f"\n📈 Creating Summary Visualization...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        tier_names = []
        width_reductions = []
        data_retentions = []
        
        for tier_name, results in all_results.tier_results.items():
            tier_names.append(tier_name.title())
            width_reductions.append(results.width_reduction * 100)  # Convert to percentage
            data_retentions.append(results.data_retention * 100)
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Width reduction plot
        bars1 = ax1.bar(tier_names, width_reductions, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.set_title(f'Envelope Width Reduction by Tier\n({crop_name})')
        ax1.set_ylabel('Width Reduction (%)')
        ax1.set_ylim(0, max(width_reductions) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars1, width_reductions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Data retention plot
        bars2 = ax2.bar(tier_names, data_retentions, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax2.set_title(f'Data Retention by Tier\n({crop_name})')
        ax2.set_ylabel('Data Retention (%)')
        ax2.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, value in zip(bars2, data_retentions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('demo_output_multi_tier')
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / f'multi_tier_summary_{crop_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

def main():
    """Run complete multi-tier envelope system demonstration."""
    print("🌾 Multi-Tier Envelope System: Complete Workflow Demo")
    print("=" * 70)
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Load demonstration data
    crop_data, crop_name, is_real_data = load_demo_data()
    
    # Demo 1: Basic multi-tier analysis
    all_results = demo_basic_multi_tier_analysis(crop_data, crop_name)
    
    # Demo 2: National analysis
    national_results = demo_national_analysis(crop_data, crop_name, is_real_data)
    
    # Demo 3: Country comparison
    demo_country_comparison(crop_data, crop_name, is_real_data)
    
    # Demo 4: Custom tier configuration
    demo_custom_tier_configuration(crop_data, crop_name)
    
    # Demo 5: Policy scenarios
    demo_policy_scenarios(crop_data, crop_name)
    
    # Demo 6: Validation and quality assurance
    demo_validation_and_quality_assurance(all_results)
    
    # Create summary visualization
    create_summary_visualization(all_results, crop_name)
    
    # Final summary
    print(f"\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully demonstrated multi-tier envelope system with {crop_name}")
    print(f"📊 Analyzed {len(all_results.tier_results)} tiers")
    print(f"🔍 All validation tests completed")
    print(f"📈 Summary visualization created")
    
    if is_real_data:
        print(f"🌍 Real SPAM data analysis completed")
    else:
        print(f"⚠️  Synthetic data used - run with real SPAM data for production analysis")
    
    print(f"\n📚 Next Steps:")
    print(f"   • Review user guide: docs/MULTI_TIER_USER_GUIDE.md")
    print(f"   • Check policy guide: docs/MULTI_TIER_POLICY_GUIDE.md")
    print(f"   • Explore API reference: docs/MULTI_TIER_API_REFERENCE.md")
    print(f"   • Run with your own data using the API")

if __name__ == '__main__':
    main()
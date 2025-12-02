#!/usr/bin/env python3
"""
Demonstration of National Comparison Analyzer functionality.

This script demonstrates the comprehensive national comparison and reporting
capabilities for cross-country agricultural capacity analysis.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.data.country_boundary_manager import CountryBoundaryManager
from agririchter.analysis.national_envelope_analyzer import NationalEnvelopeAnalyzer
from agririchter.analysis.national_comparison_analyzer import NationalComparisonAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate national comparison analyzer functionality."""
    
    print("=" * 80)
    print("NATIONAL COMPARISON ANALYZER DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize configuration for wheat analysis
        config = Config('wheat')
        config.spam_data_dir = Path('spam2020V2r0_global_production/spam2020V2r0_global_production')
        config.harvest_data_dir = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area')
        
        print(f"\n1. Configuration Setup")
        print(f"   Crop Type: {config.crop_type}")
        print(f"   SPAM Data Directory: {config.spam_data_dir}")
        print(f"   Harvest Data Directory: {config.harvest_data_dir}")
        
        # Initialize components
        print(f"\n2. Initializing Analysis Components")
        
        # Initialize required components for CountryBoundaryManager
        from agririchter.data.spatial_mapper import SpatialMapper
        from agririchter.data.grid_manager import GridDataManager
        
        grid_manager = GridDataManager(config)
        print(f"   ✓ Grid Manager initialized")
        
        spatial_mapper = SpatialMapper(config, grid_manager)
        print(f"   ✓ Spatial Mapper initialized")
        
        country_manager = CountryBoundaryManager(config, spatial_mapper, grid_manager)
        print(f"   ✓ Country Boundary Manager initialized")
        
        national_analyzer = NationalEnvelopeAnalyzer(
            config=config,
            country_boundary_manager=country_manager
        )
        print(f"   ✓ National Envelope Analyzer initialized")
        
        comparison_analyzer = NationalComparisonAnalyzer(
            config=config,
            national_analyzer=national_analyzer
        )
        print(f"   ✓ National Comparison Analyzer initialized")
        
        # Define countries for comparison
        countries_to_compare = ['USA', 'CHN']  # Start with USA and China
        print(f"\n3. Countries Selected for Comparison: {countries_to_compare}")
        
        # Create output directory
        output_dir = Path('demo_output_national_comparison')
        output_dir.mkdir(exist_ok=True)
        print(f"   Output Directory: {output_dir}")
        
        # Perform national comparison analysis
        print(f"\n4. Performing National Comparison Analysis")
        print(f"   This may take several minutes for real SPAM data processing...")
        
        comparison_report = comparison_analyzer.compare_countries(
            country_codes=countries_to_compare,
            output_dir=output_dir
        )
        
        print(f"   ✓ Comparison analysis completed successfully")
        
        # Display key results
        print(f"\n5. Key Comparison Results")
        print(f"   Report Date: {comparison_report.report_date}")
        print(f"   Countries Analyzed: {comparison_report.countries_analyzed}")
        
        # Show rankings
        print(f"\n   Rankings Summary:")
        for metric, ranking in comparison_report.rankings.items():
            if ranking:
                top_country, top_value = ranking[0]
                print(f"   - {metric.replace('_', ' ').title()}: {top_country} ({top_value:.2f})")
        
        # Show strategic insights
        print(f"\n6. Strategic Insights:")
        for i, insight in enumerate(comparison_report.strategic_insights, 1):
            print(f"   {i}. {insight}")
        
        # Show policy recommendations summary
        print(f"\n7. Policy Recommendations Summary:")
        for country_code in comparison_report.countries_analyzed:
            recs = comparison_report.policy_recommendations.get(country_code, [])
            high_priority_recs = [r for r in recs if r.priority == 'high']
            
            print(f"\n   {country_code} - High Priority Recommendations:")
            for rec in high_priority_recs:
                print(f"   - {rec.category.title()}: {rec.recommendation}")
        
        # Show country metrics
        print(f"\n8. Detailed Country Metrics:")
        for country_code, metrics in comparison_report.country_metrics.items():
            print(f"\n   {country_code} ({metrics.country_name}):")
            print(f"   - Total Production: {metrics.total_production_mt:,.0f} MT")
            print(f"   - Average Yield: {metrics.average_yield_mt_per_ha:.2f} MT/ha")
            print(f"   - Production Efficiency: {metrics.production_efficiency_pct:.1f}%")
            print(f"   - Food Security Score: {metrics.food_security_score:.1f}")
            print(f"   - Export Potential Score: {metrics.export_potential_score:.1f}")
            print(f"   - Commercial Width Reduction: {metrics.commercial_width_reduction_pct:.1f}%")
        
        # Generate executive summary
        print(f"\n9. Generating Executive Summary")
        summary_path = output_dir / f"executive_summary_{config.crop_type}.md"
        comparison_analyzer.generate_executive_summary(comparison_report, summary_path)
        print(f"   ✓ Executive summary written to: {summary_path}")
        
        # Show visualization files created
        print(f"\n10. Visualizations Created:")
        for viz_name, viz_path in comparison_report.visualization_paths.items():
            print(f"   - {viz_name.replace('_', ' ').title()}: {viz_path}")
        
        # Show output files
        print(f"\n11. Output Files Generated:")
        output_files = list(output_dir.glob('*'))
        for file_path in sorted(output_files):
            print(f"   - {file_path.name}")
        
        print(f"\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"All outputs saved to: {output_dir.absolute()}")
        
        return comparison_report
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\nDemonstration completed successfully!")
        print(f"Check the demo_output_national_comparison directory for all generated files.")
    else:
        print(f"\nDemonstration failed. Check the logs for details.")
        sys.exit(1)
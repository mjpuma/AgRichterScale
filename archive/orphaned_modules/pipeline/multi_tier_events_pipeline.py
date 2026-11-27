"""
Multi-Tier Events Pipeline

Enhanced events pipeline with multi-tier envelope support for policy-relevant analysis.
Integrates with existing EventsPipeline while adding tier selection capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from .events_pipeline import EventsPipeline
from ..core.config import Config
from ..core.performance import PerformanceMonitor
from ..analysis.multi_tier_envelope import MultiTierEnvelopeEngine, MultiTierResults, TIER_CONFIGURATIONS
from ..analysis.envelope import HPEnvelopeCalculator


class MultiTierEventsPipeline(EventsPipeline):
    """
    Enhanced events pipeline with multi-tier envelope support.
    
    Extends the base EventsPipeline to support:
    - Multi-tier envelope calculation
    - Policy-relevant tier selection
    - Tier comparison analysis
    - Enhanced reporting with tier insights
    """
    
    def __init__(self, config: Config, output_dir: str, 
                 tier_selection: str = 'commercial',
                 enable_performance_monitoring: bool = True):
        """
        Initialize multi-tier events pipeline.
        
        Args:
            config: Configuration object with data paths and settings
            output_dir: Directory for saving outputs
            tier_selection: Default tier for envelope calculations ('comprehensive', 'commercial', 'all')
            enable_performance_monitoring: If True, enable performance monitoring
        """
        super().__init__(config, output_dir, tier_selection, enable_performance_monitoring)
        
        # Multi-tier specific configuration
        self.tier_selection = tier_selection
        self.multi_tier_engine: Optional[MultiTierEnvelopeEngine] = None
        self.tier_results: Optional[MultiTierResults] = None
        
        # Enhanced logging
        self.logger.info(f"Multi-tier events pipeline initialized")
        self.logger.info(f"Default tier selection: {tier_selection}")
        
        # Validate tier selection
        if tier_selection not in ['comprehensive', 'commercial', 'all']:
            raise ValueError(f"Invalid tier selection: {tier_selection}. "
                           f"Must be one of: {list(TIER_CONFIGURATIONS.keys()) + ['all']}")
    
    def initialize_multi_tier_engine(self) -> None:
        """Initialize the multi-tier envelope engine."""
        if self.multi_tier_engine is None:
            self.logger.info("Initializing multi-tier envelope engine")
            self.multi_tier_engine = MultiTierEnvelopeEngine(self.config)
            self.logger.info("Multi-tier engine initialized successfully")
    
    def calculate_envelope_with_tier_selection(self, 
                                             production_df: pd.DataFrame,
                                             harvest_df: pd.DataFrame,
                                             tier: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate envelope bounds with tier selection.
        
        Args:
            production_df: Production data DataFrame
            harvest_df: Harvest area data DataFrame
            tier: Tier to calculate (uses instance default if None)
        
        Returns:
            Envelope data dictionary or multi-tier results
        """
        tier = tier or self.tier_selection
        
        self.logger.info(f"Calculating envelope bounds for tier: {tier}")
        
        if tier == 'all':
            # Calculate all tiers using multi-tier engine
            self.initialize_multi_tier_engine()
            self.tier_results = self.multi_tier_engine.calculate_multi_tier_envelope(
                production_df, harvest_df
            )
            
            # Return comprehensive tier for backward compatibility with visualization
            return self.tier_results.get_tier_envelope('comprehensive').to_dict()
        
        elif tier in TIER_CONFIGURATIONS:
            # Calculate specific tier
            if tier == 'comprehensive':
                # Use existing calculator for comprehensive (backward compatibility)
                calculator = HPEnvelopeCalculator(self.config)
                return calculator.calculate_hp_envelope(production_df, harvest_df, tier='comprehensive')
            else:
                # Use multi-tier engine for other tiers
                self.initialize_multi_tier_engine()
                self.tier_results = self.multi_tier_engine.calculate_multi_tier_envelope(
                    production_df, harvest_df, tiers=[tier]
                )
                return self.tier_results.get_tier_envelope(tier).to_dict()
        
        else:
            raise ValueError(f"Unknown tier: {tier}")
    
    def generate_visualizations(self, events_df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate visualizations with multi-tier envelope support.
        
        Extends base visualization generation to include tier-specific envelope plots
        and tier comparison visualizations.
        
        Args:
            events_df: DataFrame with calculated event results
            
        Returns:
            Dictionary of figure objects including tier-specific visualizations
        """
        self.logger.info("=== Stage 3: Multi-Tier Visualization Generation ===")
        
        if self.performance_monitor:
            self.performance_monitor.start_stage("multi_tier_visualization_generation")
        
        try:
            # Start with base visualizations
            figures = super().generate_visualizations(events_df)
            
            # Add multi-tier specific visualizations
            production_df = self.loaded_data.get('production_df')
            harvest_df = self.loaded_data.get('harvest_df')
            
            if production_df is not None and harvest_df is not None:
                # Generate tier-specific envelope plots
                tier_figures = self._generate_tier_visualizations(
                    production_df, harvest_df, events_df
                )
                figures.update(tier_figures)
                
                # Generate tier comparison plots if multiple tiers calculated
                if self.tier_results is not None and len(self.tier_results.tier_results) > 1:
                    comparison_figures = self._generate_tier_comparison_visualizations(
                        self.tier_results, events_df
                    )
                    figures.update(comparison_figures)
            
            self.logger.info(f"Multi-tier visualization generation completed: {len(figures)} figures created")
            
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            
            return figures
            
        except Exception as e:
            self.logger.error(f"Multi-tier visualization generation failed: {e}")
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            raise
    
    def _generate_tier_visualizations(self, 
                                    production_df: pd.DataFrame,
                                    harvest_df: pd.DataFrame,
                                    events_df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate tier-specific envelope visualizations.
        
        Args:
            production_df: Production data
            harvest_df: Harvest area data
            events_df: Events data
        
        Returns:
            Dictionary of tier-specific figures
        """
        tier_figures = {}
        
        try:
            from ..visualization.hp_envelope import HPEnvelopeVisualizer
            
            # Calculate envelope for selected tier
            envelope_data = self.calculate_envelope_with_tier_selection(
                production_df, harvest_df, self.tier_selection
            )
            
            # Create tier-specific envelope plot
            hp_viz = HPEnvelopeVisualizer(self.config)
            
            # Customize title based on tier
            tier_config = TIER_CONFIGURATIONS.get(self.tier_selection)
            if tier_config:
                tier_title = f"H-P Envelope ({tier_config.name})"
            else:
                tier_title = f"H-P Envelope ({self.tier_selection.title()})"
            
            fig_tier_envelope = hp_viz.create_hp_envelope_plot(
                envelope_data, events_df, title=tier_title
            )
            
            tier_figures[f'hp_envelope_{self.tier_selection}'] = fig_tier_envelope
            
            self.logger.info(f"Generated {self.tier_selection} tier envelope visualization")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate tier-specific visualizations: {e}")
        
        return tier_figures
    
    def _generate_tier_comparison_visualizations(self, 
                                               tier_results: MultiTierResults,
                                               events_df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate tier comparison visualizations.
        
        Args:
            tier_results: Multi-tier calculation results
            events_df: Events data
        
        Returns:
            Dictionary of comparison figures
        """
        comparison_figures = {}
        
        try:
            # Create tier comparison plot
            fig_comparison = self._create_tier_comparison_plot(tier_results)
            comparison_figures['tier_comparison'] = fig_comparison
            
            # Create width reduction analysis plot
            fig_width_analysis = self._create_width_analysis_plot(tier_results)
            comparison_figures['width_analysis'] = fig_width_analysis
            
            self.logger.info("Generated tier comparison visualizations")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate tier comparison visualizations: {e}")
        
        return comparison_figures
    
    def _create_tier_comparison_plot(self, tier_results: MultiTierResults) -> plt.Figure:
        """
        Create a plot comparing envelope bounds across tiers.
        
        Args:
            tier_results: Multi-tier calculation results
        
        Returns:
            Matplotlib figure with tier comparison
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'comprehensive': 'blue', 'commercial': 'red'}
        alphas = {'comprehensive': 0.3, 'commercial': 0.5}
        
        for tier_name, envelope_data in tier_results.tier_results.items():
            color = colors.get(tier_name, 'gray')
            alpha = alphas.get(tier_name, 0.4)
            
            # Plot envelope bounds
            ax.fill_between(
                envelope_data.lower_bound_harvest,
                envelope_data.lower_bound_production,
                envelope_data.upper_bound_production,
                alpha=alpha,
                color=color,
                label=f'{tier_name.title()} Tier'
            )
            
            # Plot envelope lines
            ax.plot(envelope_data.lower_bound_harvest, envelope_data.lower_bound_production,
                   color=color, linestyle='-', linewidth=2)
            ax.plot(envelope_data.upper_bound_harvest, envelope_data.upper_bound_production,
                   color=color, linestyle='-', linewidth=2)
        
        ax.set_xlabel('Harvest Area (km²)')
        ax.set_ylabel('Production (kcal)')
        ax.set_title(f'Multi-Tier Envelope Comparison - {tier_results.crop_type.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add width reduction annotations
        if 'commercial' in tier_results.tier_results:
            reduction = tier_results.get_width_reduction('commercial')
            if reduction is not None:
                ax.text(0.02, 0.98, f'Commercial Tier Width Reduction: {reduction:.1f}%',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _create_width_analysis_plot(self, tier_results: MultiTierResults) -> plt.Figure:
        """
        Create a plot showing width reduction analysis.
        
        Args:
            tier_results: Multi-tier calculation results
        
        Returns:
            Matplotlib figure with width analysis
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Width comparison
        tier_names = []
        widths = []
        reductions = []
        
        for tier_name in tier_results.tier_results.keys():
            tier_names.append(tier_name.title())
            
            # Calculate representative width
            envelope_data = tier_results.get_tier_envelope(tier_name)
            production_widths = envelope_data.upper_bound_production - envelope_data.lower_bound_production
            width = np.median(production_widths)
            widths.append(width)
            
            # Get reduction percentage
            reduction = tier_results.get_width_reduction(tier_name) or 0.0
            reductions.append(reduction)
        
        # Bar plot of widths
        bars1 = ax1.bar(tier_names, widths, color=['blue', 'red'][:len(tier_names)])
        ax1.set_ylabel('Median Envelope Width (kcal)')
        ax1.set_title('Envelope Width by Tier')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, width in zip(bars1, widths):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{width:.2e}', ha='center', va='bottom')
        
        # Bar plot of reductions
        bars2 = ax2.bar(tier_names, reductions, color=['blue', 'red'][:len(tier_names)])
        ax2.set_ylabel('Width Reduction (%)')
        ax2.set_title('Width Reduction vs Comprehensive Tier')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, reduction in zip(bars2, reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reduction:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def export_results(self, events_df: pd.DataFrame, figures: Dict[str, plt.Figure]) -> Dict[str, List[str]]:
        """
        Export results with multi-tier analysis data.
        
        Extends base export to include tier-specific data and analysis reports.
        
        Args:
            events_df: DataFrame with event results
            figures: Dictionary of figure objects
            
        Returns:
            Dictionary of exported file paths
        """
        self.logger.info("=== Stage 4: Multi-Tier Results Export ===")
        
        if self.performance_monitor:
            self.performance_monitor.start_stage("multi_tier_results_export")
        
        try:
            # Start with base export
            exported_files = super().export_results(events_df, figures)
            
            # Add multi-tier specific exports
            if self.tier_results is not None:
                tier_files = self._export_tier_analysis(self.tier_results)
                exported_files['tier_analysis_files'] = tier_files
            
            # Export tier selection guide
            guide_files = self._export_tier_selection_guide()
            exported_files['guide_files'] = guide_files
            
            self.logger.info("Multi-tier results export completed")
            
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Multi-tier results export failed: {e}")
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            raise
    
    def _export_tier_analysis(self, tier_results: MultiTierResults) -> List[str]:
        """
        Export tier analysis data and reports.
        
        Args:
            tier_results: Multi-tier calculation results
        
        Returns:
            List of exported file paths
        """
        tier_files = []
        
        try:
            # Create tier analysis directory
            tier_dir = self.output_dir / 'tier_analysis'
            tier_dir.mkdir(exist_ok=True)
            
            # Export tier summary statistics
            summary_path = tier_dir / f'tier_summary_{self.config.crop_type}.csv'
            summary_stats = tier_results.get_summary_statistics()
            
            # Convert to DataFrame for export
            summary_rows = []
            for tier_name, tier_data in summary_stats.items():
                if tier_name.endswith('_tier'):
                    tier_base_name = tier_name.replace('_tier', '')
                    summary_rows.append({
                        'tier_name': tier_base_name,
                        'description': tier_data.get('description', ''),
                        'envelope_points': tier_data.get('envelope_points', 0),
                        'convergence_validated': tier_data.get('convergence_validated', False),
                        'width_reduction_pct': tier_data.get('width_reduction', 0.0)
                    })
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(summary_path, index=False)
                tier_files.append(str(summary_path))
                self.logger.info(f"Exported tier summary: {summary_path}")
            
            # Export width analysis
            width_path = tier_dir / f'width_analysis_{self.config.crop_type}.csv'
            width_df = pd.DataFrame([tier_results.width_analysis])
            width_df.to_csv(width_path, index=False)
            tier_files.append(str(width_path))
            
            # Export individual tier envelope data
            for tier_name, envelope_data in tier_results.tier_results.items():
                tier_envelope_path = tier_dir / f'envelope_{tier_name}_{self.config.crop_type}.csv'
                
                envelope_df = pd.DataFrame({
                    'disruption_area': envelope_data.disruption_areas,
                    'lower_bound_harvest': envelope_data.lower_bound_harvest,
                    'lower_bound_production': envelope_data.lower_bound_production,
                    'upper_bound_harvest': envelope_data.upper_bound_harvest,
                    'upper_bound_production': envelope_data.upper_bound_production
                })
                
                envelope_df.to_csv(tier_envelope_path, index=False)
                tier_files.append(str(tier_envelope_path))
                self.logger.info(f"Exported {tier_name} envelope data: {tier_envelope_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to export tier analysis: {e}")
        
        return tier_files
    
    def _export_tier_selection_guide(self) -> List[str]:
        """
        Export tier selection guide for users.
        
        Returns:
            List of exported guide file paths
        """
        guide_files = []
        
        try:
            # Create guides directory
            guides_dir = self.output_dir / 'guides'
            guides_dir.mkdir(exist_ok=True)
            
            # Export tier selection guide
            guide_path = guides_dir / 'tier_selection_guide.md'
            
            guide_content = self._generate_tier_selection_guide_content()
            
            with open(guide_path, 'w') as f:
                f.write(guide_content)
            
            guide_files.append(str(guide_path))
            self.logger.info(f"Exported tier selection guide: {guide_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to export tier selection guide: {e}")
        
        return guide_files
    
    def _generate_tier_selection_guide_content(self) -> str:
        """Generate content for tier selection guide."""
        
        guide_lines = [
            "# Multi-Tier Envelope Analysis - Tier Selection Guide",
            "",
            "## Overview",
            "",
            "This guide helps you select the appropriate productivity tier for your agricultural capacity analysis.",
            "",
            "## Available Tiers",
            ""
        ]
        
        for tier_name, tier_config in TIER_CONFIGURATIONS.items():
            guide_lines.extend([
                f"### {tier_config.name}",
                "",
                f"**Description:** {tier_config.description}",
                f"**Yield Range:** {tier_config.yield_percentile_min}-{tier_config.yield_percentile_max}th percentile",
                f"**Expected Width Reduction:** {tier_config.expected_width_reduction}",
                "",
                "**Policy Applications:**",
                ""
            ])
            
            for app in tier_config.policy_applications:
                guide_lines.append(f"- {app.replace('_', ' ').title()}")
            
            guide_lines.extend([
                "",
                "**Target Users:**",
                ""
            ])
            
            for user in tier_config.target_users:
                guide_lines.append(f"- {user.replace('_', ' ').title()}")
            
            guide_lines.extend(["", "---", ""])
        
        guide_lines.extend([
            "## Tier Selection Recommendations",
            "",
            "### For Government Planning and Policy Analysis",
            "**Recommended Tier:** Commercial Agriculture",
            "- Focuses on economically viable agricultural land",
            "- Excludes marginal areas with very low productivity",
            "- Provides realistic capacity estimates for policy planning",
            "",
            "### For Academic Research and Theoretical Analysis",
            "**Recommended Tier:** Comprehensive (All Lands)",
            "- Includes all agricultural land regardless of productivity",
            "- Provides theoretical maximum bounds",
            "- Useful for baseline comparisons and academic studies",
            "",
            "### For Investment and Development Planning",
            "**Recommended Tier:** Commercial Agriculture",
            "- Targets economically viable agricultural areas",
            "- Helps identify high-potential regions for investment",
            "- Provides realistic return-on-investment estimates",
            "",
            "## Usage Examples",
            "",
            "```python",
            "# Government planning scenario",
            "pipeline = MultiTierEventsPipeline(config, output_dir, tier_selection='commercial')",
            "",
            "# Academic research scenario", 
            "pipeline = MultiTierEventsPipeline(config, output_dir, tier_selection='comprehensive')",
            "",
            "# Comparative analysis",
            "pipeline = MultiTierEventsPipeline(config, output_dir, tier_selection='all')",
            "```",
            "",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Crop Type: {self.config.crop_type}",
            f"Current Selection: {self.tier_selection}"
        ])
        
        return "\n".join(guide_lines)
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate enhanced summary report with multi-tier analysis.
        
        Extends base summary report to include tier-specific insights and recommendations.
        
        Args:
            results: Dictionary with pipeline results
                
        Returns:
            Enhanced summary report as string
        """
        # Start with base report
        base_report = super().generate_summary_report(results)
        
        # Add multi-tier specific sections
        tier_sections = []
        
        if self.tier_results is not None:
            tier_sections.extend([
                "",
                "-" * 80,
                "MULTI-TIER ANALYSIS RESULTS",
                "-" * 80,
                f"Selected Tier: {self.tier_selection}",
                f"Tiers Calculated: {', '.join(self.tier_results.tier_results.keys())}",
                ""
            ])
            
            # Add tier statistics
            for tier_name, envelope_data in self.tier_results.tier_results.items():
                tier_config = TIER_CONFIGURATIONS.get(tier_name)
                reduction = self.tier_results.get_width_reduction(tier_name)
                
                tier_sections.extend([
                    f"{tier_name.upper()} TIER:",
                    f"  Description: {tier_config.description if tier_config else 'Unknown'}",
                    f"  Envelope Points: {len(envelope_data.disruption_areas)}",
                    f"  Convergence Validated: {envelope_data.convergence_validated}",
                    f"  Width Reduction: {reduction:.1f}%" if reduction is not None else "  Width Reduction: 0.0% (baseline)",
                    ""
                ])
            
            # Add policy recommendations
            tier_sections.extend([
                "POLICY RECOMMENDATIONS:",
                ""
            ])
            
            if self.tier_selection == 'commercial':
                tier_sections.extend([
                    "✓ Commercial tier selected - suitable for government planning",
                    "✓ Envelope bounds exclude marginal agricultural areas",
                    "✓ Results represent economically viable production capacity",
                    "✓ Recommended for policy analysis and investment decisions",
                    ""
                ])
            elif self.tier_selection == 'comprehensive':
                tier_sections.extend([
                    "✓ Comprehensive tier selected - includes all agricultural land",
                    "✓ Provides theoretical maximum production bounds",
                    "✓ Suitable for academic research and baseline comparisons",
                    "✓ May include economically marginal areas",
                    ""
                ])
        
        # Combine base report with tier sections
        return base_report + "\n".join(tier_sections)
    
    def get_tier_selection_guide(self) -> Dict[str, Dict[str, Any]]:
        """
        Get tier selection guide for API users.
        
        Returns:
            Dictionary with tier information and selection guidance
        """
        if self.multi_tier_engine is None:
            self.initialize_multi_tier_engine()
        
        return self.multi_tier_engine.get_tier_info()
    
    def set_tier_selection(self, tier: str) -> None:
        """
        Update tier selection for subsequent calculations.
        
        Args:
            tier: New tier selection ('comprehensive', 'commercial', 'all')
        """
        if tier not in ['comprehensive', 'commercial', 'all']:
            raise ValueError(f"Invalid tier: {tier}")
        
        self.tier_selection = tier
        self.logger.info(f"Tier selection updated to: {tier}")
    
    def run_analysis(self, disruption_scenarios: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Run complete multi-tier analysis pipeline.
        
        Extends base pipeline to use tier-specific envelope calculations.
        
        Args:
            disruption_scenarios: Optional disruption scenarios for analysis
        
        Returns:
            Complete pipeline results with multi-tier data
        """
        self.logger.info(f"Starting multi-tier analysis pipeline with tier: {self.tier_selection}")
        
        try:
            # Load data
            loaded_data = self.load_all_data()
            
            # Calculate events
            events_df = self.calculate_events()
            
            # Generate visualizations (includes tier-specific plots)
            figures = self.generate_visualizations(events_df)
            
            # Export results (includes tier analysis)
            exported_files = self.export_results(events_df, figures)
            
            # Compile results
            results = {
                'events_df': events_df,
                'figures': figures,
                'exported_files': exported_files,
                'loaded_data': loaded_data,
                'tier_selection': self.tier_selection,
                'tier_results': self.tier_results
            }
            
            # Generate summary report
            summary_report = self.generate_summary_report(results)
            
            # Save summary report
            report_path = self.output_dir / 'reports' / f'pipeline_summary_{self.config.crop_type}.txt'
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(summary_report)
            
            results['summary_report'] = summary_report
            results['summary_report_path'] = str(report_path)
            
            self.logger.info("Multi-tier analysis pipeline completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-tier analysis pipeline failed: {e}")
            raise


# Convenience functions for common use cases
def create_policy_analysis_pipeline(config: Config, output_dir: str) -> MultiTierEventsPipeline:
    """
    Create pipeline configured for policy analysis (commercial tier).
    
    Args:
        config: Configuration instance
        output_dir: Output directory path
    
    Returns:
        MultiTierEventsPipeline configured for policy analysis
    """
    return MultiTierEventsPipeline(
        config=config,
        output_dir=output_dir,
        tier_selection='commercial'
    )


def create_research_analysis_pipeline(config: Config, output_dir: str) -> MultiTierEventsPipeline:
    """
    Create pipeline configured for research analysis (comprehensive tier).
    
    Args:
        config: Configuration instance
        output_dir: Output directory path
    
    Returns:
        MultiTierEventsPipeline configured for research analysis
    """
    return MultiTierEventsPipeline(
        config=config,
        output_dir=output_dir,
        tier_selection='comprehensive'
    )


def create_comparative_analysis_pipeline(config: Config, output_dir: str) -> MultiTierEventsPipeline:
    """
    Create pipeline configured for comparative analysis (all tiers).
    
    Args:
        config: Configuration instance
        output_dir: Output directory path
    
    Returns:
        MultiTierEventsPipeline configured for comparative analysis
    """
    return MultiTierEventsPipeline(
        config=config,
        output_dir=output_dir,
        tier_selection='all'
    )
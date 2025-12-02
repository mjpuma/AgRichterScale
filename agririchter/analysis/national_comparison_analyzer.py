"""
National Comparison Analyzer for cross-country agricultural capacity analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from ..core.config import Config
from .national_envelope_analyzer import NationalEnvelopeAnalyzer, NationalAnalysisResults

logger = logging.getLogger(__name__)


@dataclass
class PolicyRecommendation:
    """Policy recommendation with priority and rationale."""
    country_code: str
    priority: str
    category: str
    recommendation: str
    rationale: str


@dataclass
class ComparisonMetrics:
    """Metrics for comparing countries."""
    country_code: str
    country_name: str
    total_production_mt: float
    total_harvest_area_ha: float
    average_yield_mt_per_ha: float
    productive_cells: int
    max_production_capacity_mt: float
    production_efficiency_pct: float
    capacity_utilization_pct: float
    commercial_width_reduction_pct: float
    comprehensive_width_reduction_pct: float
    food_security_score: float
    export_potential_score: float
    efficiency_improvement_score: float


@dataclass
class NationalComparisonReport:
    """Comprehensive national comparison report."""
    report_date: str
    crop_type: str
    countries_analyzed: List[str]
    country_metrics: Dict[str, ComparisonMetrics]
    rankings: Dict[str, List[Tuple[str, float]]]
    comparative_analysis: Dict[str, Any]
    policy_recommendations: Dict[str, List[PolicyRecommendation]]
    strategic_insights: List[str]
    visualization_paths: Dict[str, str]
    
    def get_top_performer(self, metric: str) -> Optional[Tuple[str, float]]:
        """Get top performing country for a metric."""
        if metric not in self.rankings or not self.rankings[metric]:
            return None
        return self.rankings[metric][0]
    
    def get_country_rank(self, country_code: str, metric: str) -> Optional[int]:
        """Get country rank for a specific metric (1-based)."""
        if metric not in self.rankings:
            return None
        
        for i, (code, _) in enumerate(self.rankings[metric]):
            if code == country_code:
                return i + 1
        return None


class NationalComparisonAnalyzer:
    """Analyzes and compares national agricultural capacities across countries."""
    
    def __init__(self, config: Config, national_analyzer: NationalEnvelopeAnalyzer):
        """Initialize NationalComparisonAnalyzer."""
        self.config = config
        self.national_analyzer = national_analyzer
        self.logger = logging.getLogger('agririchter.comparison_analyzer')
        self.logger.info("NationalComparisonAnalyzer initialized")
    
    def compare_countries(self, 
                         country_codes: List[str],
                         output_dir: Optional[Path] = None) -> NationalComparisonReport:
        """Generate comprehensive comparison between countries."""
        if len(country_codes) < 2:
            raise ValueError("Need at least 2 countries for comparison")
        
        if output_dir is None:
            output_dir = Path('.')
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Starting comparison analysis for {country_codes}")
        
        # Analyze each country
        country_results = {}
        for country_code in country_codes:
            try:
                results = self.national_analyzer.analyze_national_capacity(country_code)
                country_results[country_code] = results
                self.logger.info(f"✓ Completed analysis for {country_code}")
            except Exception as e:
                self.logger.error(f"✗ Failed to analyze {country_code}: {e}")
                continue
        
        if len(country_results) < 2:
            raise ValueError("Need at least 2 countries with valid analysis results")
        
        # Calculate comparison metrics
        country_metrics = self._calculate_comparison_metrics(country_results)
        
        # Generate rankings
        rankings = self._generate_rankings(country_metrics)
        
        # Perform comparative analysis
        comparative_analysis = self._perform_comparative_analysis(country_results, country_metrics)
        
        # Generate policy recommendations
        policy_recommendations = self._generate_policy_recommendations(country_results, country_metrics, comparative_analysis)
        
        # Generate strategic insights
        strategic_insights = self._generate_strategic_insights(country_results, country_metrics, comparative_analysis)
        
        # Create visualizations
        visualization_paths = self._create_visualizations(country_metrics, rankings, output_dir)
        
        # Create comprehensive report
        report = NationalComparisonReport(
            report_date=datetime.now().isoformat(),
            crop_type=self.config.crop_type,
            countries_analyzed=list(country_results.keys()),
            country_metrics=country_metrics,
            rankings=rankings,
            comparative_analysis=comparative_analysis,
            policy_recommendations=policy_recommendations,
            strategic_insights=strategic_insights,
            visualization_paths=visualization_paths
        )
        
        self.logger.info(f"National comparison analysis completed for {len(country_results)} countries")
        return report
    
    def _calculate_comparison_metrics(self, 
                                    country_results: Dict[str, NationalAnalysisResults]) -> Dict[str, ComparisonMetrics]:
        """Calculate standardized comparison metrics for each country."""
        metrics = {}
        
        for country_code, results in country_results.items():
            stats = results.national_statistics
            
            # Get commercial tier capacity
            commercial_capacity = results.get_production_capacity('commercial')
            max_capacity = commercial_capacity.get('max_production_capacity', 0)
            current_production = stats['total_production_mt']
            
            # Calculate efficiency and utilization
            production_efficiency = (current_production / max_capacity * 100) if max_capacity > 0 else 0
            capacity_utilization = (current_production / max_capacity * 100) if max_capacity > 0 else 0
            
            # Calculate width reductions
            commercial_width_reduction = results.multi_tier_results.get_width_reduction('commercial') if hasattr(results.multi_tier_results, 'get_width_reduction') else 25.0
            comprehensive_width_reduction = results.multi_tier_results.get_width_reduction('comprehensive') if hasattr(results.multi_tier_results, 'get_width_reduction') else 15.0
            
            # Calculate policy scores
            food_security_score = min(100, capacity_utilization) if max_capacity > 0 else 0
            export_potential_score = max(0, min(100, (max_capacity - current_production) / current_production * 50)) if current_production > 0 else 0
            efficiency_improvement_score = max(0, 100 - production_efficiency) if production_efficiency > 0 else 100
            
            metrics[country_code] = ComparisonMetrics(
                country_code=country_code,
                country_name=results.country_name,
                total_production_mt=stats['total_production_mt'],
                total_harvest_area_ha=stats['total_harvest_area_ha'],
                average_yield_mt_per_ha=stats['average_yield_mt_per_ha'],
                productive_cells=stats['productive_cells'],
                max_production_capacity_mt=max_capacity,
                production_efficiency_pct=production_efficiency,
                capacity_utilization_pct=capacity_utilization,
                commercial_width_reduction_pct=commercial_width_reduction,
                comprehensive_width_reduction_pct=comprehensive_width_reduction,
                food_security_score=food_security_score,
                export_potential_score=export_potential_score,
                efficiency_improvement_score=efficiency_improvement_score
            )
        
        return metrics
    
    def _generate_rankings(self, 
                          country_metrics: Dict[str, ComparisonMetrics]) -> Dict[str, List[Tuple[str, float]]]:
        """Generate country rankings by various metrics."""
        rankings = {}
        
        # Production ranking
        rankings['production'] = sorted(
            [(code, metrics.total_production_mt) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Efficiency ranking
        rankings['efficiency'] = sorted(
            [(code, metrics.production_efficiency_pct) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Yield ranking
        rankings['yield'] = sorted(
            [(code, metrics.average_yield_mt_per_ha) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Export potential ranking
        rankings['export_potential'] = sorted(
            [(code, metrics.export_potential_score) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Food security ranking
        rankings['food_security'] = sorted(
            [(code, metrics.food_security_score) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Improvement potential ranking
        rankings['improvement_potential'] = sorted(
            [(code, metrics.efficiency_improvement_score) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Width reduction ranking
        rankings['width_reduction'] = sorted(
            [(code, metrics.commercial_width_reduction_pct) for code, metrics in country_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return rankings
    
    def _generate_policy_recommendations(self, 
                                       country_results: Dict[str, NationalAnalysisResults],
                                       country_metrics: Dict[str, ComparisonMetrics],
                                       comparative_analysis: Dict[str, Any]) -> Dict[str, List[PolicyRecommendation]]:
        """Generate country-specific policy recommendations."""
        recommendations = {}
        
        for country_code, metrics in country_metrics.items():
            country_recs = []
            
            # Food security recommendations
            if metrics.food_security_score < 60:
                country_recs.append(PolicyRecommendation(
                    country_code=country_code,
                    priority='high',
                    category='food_security',
                    recommendation='Implement comprehensive food security enhancement program',
                    rationale=f'Food security score of {metrics.food_security_score:.1f} indicates vulnerability'
                ))
            
            # Export potential recommendations
            if metrics.export_potential_score > 70:
                country_recs.append(PolicyRecommendation(
                    country_code=country_code,
                    priority='medium',
                    category='trade',
                    recommendation='Develop agricultural export expansion strategy',
                    rationale=f'High export potential score of {metrics.export_potential_score:.1f} indicates surplus capacity'
                ))
            
            # Efficiency recommendations
            if metrics.production_efficiency_pct < 50:
                country_recs.append(PolicyRecommendation(
                    country_code=country_code,
                    priority='high',
                    category='investment',
                    recommendation='Invest in agricultural productivity enhancement programs',
                    rationale=f'Low production efficiency of {metrics.production_efficiency_pct:.1f}% indicates significant improvement potential'
                ))
            
            # Investment recommendations for high improvement potential
            if metrics.efficiency_improvement_score > 60:
                country_recs.append(PolicyRecommendation(
                    country_code=country_code,
                    priority='medium',
                    category='investment',
                    recommendation='Prioritize technology transfer and infrastructure development',
                    rationale=f'High improvement potential score of {metrics.efficiency_improvement_score:.1f} suggests good ROI for investments'
                ))
            
            recommendations[country_code] = country_recs
        
        return recommendations
    
    def _perform_comparative_analysis(self, 
                                    country_results: Dict[str, NationalAnalysisResults],
                                    country_metrics: Dict[str, ComparisonMetrics]) -> Dict[str, Any]:
        """Perform detailed comparative analysis between countries."""
        
        # Production analysis
        productions = [m.total_production_mt for m in country_metrics.values()]
        total_production = sum(productions)
        max_production = max(productions) if productions else 0
        
        production_analysis = {
            'total_production_mt': total_production,
            'production_concentration': {
                'top_producer_share_pct': (max_production / total_production * 100) if total_production > 0 else 0
            }
        }
        
        # Efficiency analysis
        efficiencies = [m.production_efficiency_pct for m in country_metrics.values()]
        efficiency_analysis = {
            'efficiency_gap': max(efficiencies) - min(efficiencies) if efficiencies else 0,
            'average_efficiency_pct': sum(efficiencies) / len(efficiencies) if efficiencies else 0
        }
        
        # Capacity analysis
        capacities = [m.max_production_capacity_mt for m in country_metrics.values()]
        current_productions = [m.total_production_mt for m in country_metrics.values()]
        total_capacity = sum(capacities)
        total_current = sum(current_productions)
        
        capacity_analysis = {
            'global_capacity_utilization_pct': (total_current / total_capacity * 100) if total_capacity > 0 else 0,
            'untapped_capacity_mt': total_capacity - total_current if total_capacity > total_current else 0
        }
        
        return {
            'production_analysis': production_analysis,
            'efficiency_analysis': efficiency_analysis,
            'capacity_analysis': capacity_analysis
        }
    
    def _generate_strategic_insights(self, 
                                   country_results: Dict[str, NationalAnalysisResults],
                                   country_metrics: Dict[str, ComparisonMetrics],
                                   comparative_analysis: Dict[str, Any]) -> List[str]:
        """Generate high-level strategic insights."""
        insights = []
        
        # Production concentration insights
        production_concentration = comparative_analysis['production_analysis']['production_concentration']['top_producer_share_pct']
        if production_concentration > 60:
            top_producer = max(country_metrics.items(), key=lambda x: x[1].total_production_mt)
            insights.append(
                f"{top_producer[1].country_name} dominates {self.config.crop_type} production with "
                f"{production_concentration:.1f}% of total output"
            )
        
        # Efficiency gap insights
        efficiency_gap = comparative_analysis['efficiency_analysis']['efficiency_gap']
        if efficiency_gap > 40:
            insights.append(
                f"Significant efficiency gap of {efficiency_gap:.1f} percentage points between "
                f"most and least efficient producers indicates improvement potential"
            )
        
        # Capacity utilization insights
        capacity_utilization = comparative_analysis['capacity_analysis']['global_capacity_utilization_pct']
        if capacity_utilization < 70:
            insights.append(
                f"Global capacity utilization of {capacity_utilization:.1f}% suggests significant "
                f"untapped agricultural potential"
            )
        
        # Trade optimization insights
        export_potentials = [m.export_potential_score for m in country_metrics.values()]
        if any(ep > 70 for ep in export_potentials):
            insights.append(
                "Trade optimization opportunities exist between high-capacity and high-demand regions"
            )
        
        # Technology transfer insights
        if efficiency_gap > 30:
            insights.append(
                "Technology transfer and knowledge sharing could significantly improve global food security"
            )
        
        return insights
    
    def _create_visualizations(self, 
                             country_metrics: Dict[str, ComparisonMetrics],
                             rankings: Dict[str, List[Tuple[str, float]]],
                             output_dir: Path) -> Dict[str, str]:
        """Create visualization charts for the comparison."""
        visualization_paths = {}
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Production comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            countries = list(country_metrics.keys())
            productions = [country_metrics[c].total_production_mt / 1e6 for c in countries]  # Convert to million MT
            
            bars = ax.bar(countries, productions)
            ax.set_title(f'{self.config.crop_type.title()} Production Comparison')
            ax.set_ylabel('Production (Million MT)')
            ax.set_xlabel('Country')
            
            # Add value labels on bars
            for bar, prod in zip(bars, productions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prod:.1f}M', ha='center', va='bottom')
            
            plt.tight_layout()
            prod_path = output_dir / 'production_comparison.png'
            plt.savefig(prod_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['production_comparison'] = str(prod_path)
            
            # Efficiency and capacity chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Efficiency chart
            efficiencies = [country_metrics[c].production_efficiency_pct for c in countries]
            ax1.bar(countries, efficiencies)
            ax1.set_title('Production Efficiency')
            ax1.set_ylabel('Efficiency (%)')
            ax1.set_xlabel('Country')
            
            # Capacity utilization chart
            utilizations = [country_metrics[c].capacity_utilization_pct for c in countries]
            ax2.bar(countries, utilizations)
            ax2.set_title('Capacity Utilization')
            ax2.set_ylabel('Utilization (%)')
            ax2.set_xlabel('Country')
            
            plt.tight_layout()
            eff_path = output_dir / 'efficiency_capacity.png'
            plt.savefig(eff_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['efficiency_capacity'] = str(eff_path)
            
            # Policy scores chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create data for grouped bar chart
            x = np.arange(len(countries))
            width = 0.25
            
            food_security = [country_metrics[c].food_security_score for c in countries]
            export_potential = [country_metrics[c].export_potential_score for c in countries]
            improvement = [country_metrics[c].efficiency_improvement_score for c in countries]
            
            ax.bar(x - width, food_security, width, label='Food Security', alpha=0.8)
            ax.bar(x, export_potential, width, label='Export Potential', alpha=0.8)
            ax.bar(x + width, improvement, width, label='Improvement Potential', alpha=0.8)
            
            ax.set_xlabel('Country')
            ax.set_ylabel('Score')
            ax.set_title('Policy Relevance Scores')
            ax.set_xticks(x)
            ax.set_xticklabels(countries)
            ax.legend()
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            policy_path = output_dir / 'policy_scores.png'
            plt.savefig(policy_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['policy_scores'] = str(policy_path)
            
            # Rankings heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create rankings matrix
            ranking_metrics = ['production', 'efficiency', 'yield', 'export_potential', 'food_security']
            ranking_matrix = []
            
            for country in countries:
                country_ranks = []
                for metric in ranking_metrics:
                    rank = next((i+1 for i, (c, _) in enumerate(rankings[metric]) if c == country), len(countries))
                    country_ranks.append(rank)
                ranking_matrix.append(country_ranks)
            
            # Create heatmap (lower rank = better, so invert colors)
            sns.heatmap(ranking_matrix, 
                       xticklabels=[m.replace('_', ' ').title() for m in ranking_metrics],
                       yticklabels=countries,
                       annot=True, 
                       fmt='d',
                       cmap='RdYlGn_r',  # Red for high ranks (worse), Green for low ranks (better)
                       ax=ax)
            
            ax.set_title('Country Rankings Heatmap\n(1 = Best, Higher = Worse)')
            plt.tight_layout()
            heatmap_path = output_dir / 'rankings_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['rankings_heatmap'] = str(heatmap_path)
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping visualizations")
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
        
        return visualization_paths
    
    def generate_executive_summary(self, 
                                 report: NationalComparisonReport,
                                 output_path: Path) -> None:
        """Generate executive summary document."""
        summary_lines = [
            f"# {self.config.crop_type.title()} Agricultural Capacity: National Comparison Report",
            f"",
            f"**Report Date:** {report.report_date}",
            f"**Countries Analyzed:** {', '.join(report.countries_analyzed)}",
            f"",
            f"## Key Findings",
            f"",
        ]
        
        # Key findings
        top_producer = report.get_top_performer('production')
        most_efficient = report.get_top_performer('efficiency')
        
        if top_producer:
            summary_lines.append(f"**Top Producer:** {top_producer[0]} ({top_producer[1]:,.0f} MT)")
        if most_efficient:
            summary_lines.append(f"**Most Efficient:** {most_efficient[0]} ({most_efficient[1]:.1f}% efficiency)")
        
        summary_lines.extend([
            f"",
            f"## Strategic Insights",
            f""
        ])
        
        for insight in report.strategic_insights:
            summary_lines.append(f"- {insight}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"Executive summary written to {output_path}")
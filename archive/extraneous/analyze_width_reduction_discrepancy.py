#!/usr/bin/env python3
"""
Analysis script to understand width reduction discrepancy between synthetic and real SPAM data.

This script investigates why real SPAM 2020 data produces lower width reductions
(7-14%) compared to synthetic predictions (22-35%).
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.analysis.multi_tier_envelope import MultiTierEnvelopeEngine
from agririchter.validation.spam_data_filter import SPAMDataFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('width_analysis')


class WidthReductionAnalyzer:
    """Analyze width reduction characteristics in real vs synthetic data."""
    
    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.results = {}
    
    def analyze_crop_characteristics(self, crop_type: str) -> Dict[str, Any]:
        """Analyze yield distribution and tier characteristics for a crop."""
        logger.info(f"Analyzing {crop_type} characteristics...")
        
        # Load data
        config = Config(crop_type=crop_type, root_dir=self.root_dir, spam_version='2020')
        data_loader = DataLoader(config)
        
        production_df = data_loader.load_spam_production()
        harvest_df = data_loader.load_spam_harvest_area()
        
        # Align datasets
        production_df, harvest_df = self._align_datasets(production_df, harvest_df)
        
        # Initialize multi-tier engine
        spam_filter = SPAMDataFilter(preset='standard')
        multi_tier_engine = MultiTierEnvelopeEngine(config, spam_filter)
        
        # Get base data after SPAM filtering
        base_data = multi_tier_engine._prepare_base_data(production_df, harvest_df)
        
        # Analyze yield distribution
        yields = base_data['yields']
        yield_analysis = self._analyze_yield_distribution(yields)
        
        # Analyze tier filtering effects
        tier_analysis = self._analyze_tier_effects(base_data, multi_tier_engine.tier_configs)
        
        # Calculate envelope characteristics
        envelope_analysis = self._analyze_envelope_characteristics(
            multi_tier_engine, production_df, harvest_df
        )
        
        return {
            'crop_type': crop_type,
            'yield_distribution': yield_analysis,
            'tier_effects': tier_analysis,
            'envelope_characteristics': envelope_analysis,
            'base_statistics': {
                'total_cells': len(yields),
                'total_production': float(base_data['production'].sum()),
                'total_harvest': float(base_data['harvest'].sum()),
                'mean_yield': float(yields.mean()),
                'median_yield': float(np.median(yields))
            }
        }
    
    def _align_datasets(self, production_df: pd.DataFrame, 
                       harvest_df: pd.DataFrame) -> tuple:
        """Align production and harvest datasets."""
        production_df['coord_key'] = production_df['x'].astype(str) + '_' + production_df['y'].astype(str)
        harvest_df['coord_key'] = harvest_df['x'].astype(str) + '_' + harvest_df['y'].astype(str)
        
        common_coords = set(production_df['coord_key']).intersection(set(harvest_df['coord_key']))
        
        aligned_production = production_df[production_df['coord_key'].isin(common_coords)].copy()
        aligned_harvest = harvest_df[harvest_df['coord_key'].isin(common_coords)].copy()
        
        aligned_production = aligned_production.sort_values('coord_key').reset_index(drop=True)
        aligned_harvest = aligned_harvest.sort_values('coord_key').reset_index(drop=True)
        
        aligned_production = aligned_production.drop('coord_key', axis=1)
        aligned_harvest = aligned_harvest.drop('coord_key', axis=1)
        
        return aligned_production, aligned_harvest
    
    def _analyze_yield_distribution(self, yields: np.ndarray) -> Dict[str, Any]:
        """Analyze yield distribution characteristics."""
        percentiles = [5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
        yield_percentiles = {f'p{p}': float(np.percentile(yields, p)) for p in percentiles}
        
        # Calculate yield concentration metrics
        p20_threshold = np.percentile(yields, 20)
        p80_threshold = np.percentile(yields, 80)
        
        low_yield_fraction = np.sum(yields <= p20_threshold) / len(yields)
        high_yield_fraction = np.sum(yields >= p80_threshold) / len(yields)
        
        # Calculate yield variability
        cv = np.std(yields) / np.mean(yields)  # Coefficient of variation
        
        return {
            'percentiles': yield_percentiles,
            'low_yield_fraction': float(low_yield_fraction),
            'high_yield_fraction': float(high_yield_fraction),
            'coefficient_of_variation': float(cv),
            'yield_range_ratio': float(yield_percentiles['p95'] / yield_percentiles['p5']),
            'interquartile_range': float(yield_percentiles['p75'] - yield_percentiles['p25']),
            'median_to_mean_ratio': float(yield_percentiles['p50'] / np.mean(yields))
        }
    
    def _analyze_tier_effects(self, base_data: Dict[str, np.ndarray], 
                            tier_configs: Dict) -> Dict[str, Any]:
        """Analyze the effects of tier filtering on data distribution."""
        yields = base_data['yields']
        production = base_data['production']
        harvest = base_data['harvest']
        
        tier_effects = {}
        
        for tier_name, tier_config in tier_configs.items():
            # Apply tier filtering
            yield_min = np.percentile(yields, tier_config.yield_percentile_min)
            yield_max = np.percentile(yields, tier_config.yield_percentile_max)
            
            tier_mask = (yields >= yield_min) & (yields <= yield_max)
            
            tier_yields = yields[tier_mask]
            tier_production = production[tier_mask]
            tier_harvest = harvest[tier_mask]
            
            # Calculate tier statistics
            retention_rate = len(tier_yields) / len(yields)
            production_retention = tier_production.sum() / production.sum()
            harvest_retention = tier_harvest.sum() / harvest.sum()
            
            # Calculate yield concentration in tier
            yield_cv_tier = np.std(tier_yields) / np.mean(tier_yields)
            yield_cv_original = np.std(yields) / np.mean(yields)
            
            tier_effects[tier_name] = {
                'cell_retention_rate': float(retention_rate),
                'production_retention_rate': float(production_retention),
                'harvest_retention_rate': float(harvest_retention),
                'yield_cv_reduction': float((yield_cv_original - yield_cv_tier) / yield_cv_original),
                'mean_yield_increase': float(np.mean(tier_yields) / np.mean(yields)),
                'yield_range_reduction': float(1 - (tier_yields.max() - tier_yields.min()) / (yields.max() - yields.min()))
            }
        
        return tier_effects
    
    def _analyze_envelope_characteristics(self, multi_tier_engine, 
                                        production_df: pd.DataFrame, 
                                        harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze envelope characteristics for different tiers."""
        
        # Calculate multi-tier envelopes
        results = multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_df, tiers=['comprehensive', 'commercial']
        )
        
        envelope_chars = {}
        
        for tier_name, envelope_data in results.tier_results.items():
            # Calculate envelope width statistics
            production_widths = envelope_data.upper_bound_production - envelope_data.lower_bound_production
            harvest_widths = envelope_data.upper_bound_harvest - envelope_data.lower_bound_harvest
            
            # Calculate relative widths (width / midpoint)
            production_midpoints = (envelope_data.upper_bound_production + envelope_data.lower_bound_production) / 2
            harvest_midpoints = (envelope_data.upper_bound_harvest + envelope_data.lower_bound_harvest) / 2
            
            relative_prod_widths = production_widths / production_midpoints
            relative_harvest_widths = harvest_widths / harvest_midpoints
            
            envelope_chars[tier_name] = {
                'mean_production_width': float(np.mean(production_widths)),
                'median_production_width': float(np.median(production_widths)),
                'mean_relative_production_width': float(np.mean(relative_prod_widths)),
                'median_relative_production_width': float(np.median(relative_prod_widths)),
                'mean_harvest_width': float(np.mean(harvest_widths)),
                'median_harvest_width': float(np.median(harvest_widths)),
                'envelope_points': len(envelope_data.disruption_areas),
                'convergence_validated': envelope_data.convergence_validated
            }
        
        # Calculate width reduction
        if 'comprehensive' in envelope_chars and 'commercial' in envelope_chars:
            comp_width = envelope_chars['comprehensive']['median_production_width']
            comm_width = envelope_chars['commercial']['median_production_width']
            width_reduction = (comp_width - comm_width) / comp_width * 100
            
            envelope_chars['width_reduction_analysis'] = {
                'absolute_width_reduction': float(comp_width - comm_width),
                'relative_width_reduction_pct': float(width_reduction),
                'comprehensive_median_width': float(comp_width),
                'commercial_median_width': float(comm_width)
            }
        
        return envelope_chars
    
    def analyze_all_crops(self) -> Dict[str, Any]:
        """Analyze all crops and generate comprehensive report."""
        crops = ['wheat', 'maize', 'rice']
        all_analyses = {}
        
        for crop in crops:
            try:
                crop_analysis = self.analyze_crop_characteristics(crop)
                all_analyses[crop] = crop_analysis
                
                # Log key findings
                width_reduction = crop_analysis['envelope_characteristics'].get('width_reduction_analysis', {}).get('relative_width_reduction_pct', 'N/A')
                yield_cv = crop_analysis['yield_distribution']['coefficient_of_variation']
                
                logger.info(f"{crop}: {width_reduction}% width reduction, CV={yield_cv:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {crop}: {str(e)}")
                all_analyses[crop] = {'error': str(e)}
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(all_analyses)
        all_analyses['comparative_analysis'] = comparative_analysis
        
        return all_analyses
    
    def _generate_comparative_analysis(self, all_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across crops."""
        successful_crops = [crop for crop, analysis in all_analyses.items() 
                          if 'error' not in analysis and crop != 'comparative_analysis']
        
        if not successful_crops:
            return {'error': 'No successful crop analyses'}
        
        # Collect width reductions
        width_reductions = []
        yield_cvs = []
        tier_retentions = []
        
        for crop in successful_crops:
            analysis = all_analyses[crop]
            
            # Width reduction
            width_red = analysis['envelope_characteristics'].get('width_reduction_analysis', {}).get('relative_width_reduction_pct')
            if width_red is not None:
                width_reductions.append(width_red)
            
            # Yield coefficient of variation
            yield_cv = analysis['yield_distribution']['coefficient_of_variation']
            yield_cvs.append(yield_cv)
            
            # Commercial tier retention rate
            comm_retention = analysis['tier_effects'].get('commercial', {}).get('cell_retention_rate')
            if comm_retention is not None:
                tier_retentions.append(comm_retention)
        
        comparative = {
            'successful_crops': successful_crops,
            'width_reduction_summary': {
                'mean': float(np.mean(width_reductions)) if width_reductions else None,
                'std': float(np.std(width_reductions)) if width_reductions else None,
                'min': float(np.min(width_reductions)) if width_reductions else None,
                'max': float(np.max(width_reductions)) if width_reductions else None,
                'all_values': width_reductions
            },
            'yield_variability_summary': {
                'mean_cv': float(np.mean(yield_cvs)) if yield_cvs else None,
                'cv_range': [float(np.min(yield_cvs)), float(np.max(yield_cvs))] if yield_cvs else None
            },
            'tier_retention_summary': {
                'mean_commercial_retention': float(np.mean(tier_retentions)) if tier_retentions else None,
                'retention_range': [float(np.min(tier_retentions)), float(np.max(tier_retentions))] if tier_retentions else None
            }
        }
        
        # Analysis insights
        insights = []
        
        if width_reductions:
            mean_reduction = np.mean(width_reductions)
            if mean_reduction < 22:
                insights.append(f"Width reductions ({mean_reduction:.1f}%) are below expected range (22-35%)")
                insights.append("This suggests real SPAM data has more uniform yield distributions than synthetic data")
        
        if yield_cvs:
            mean_cv = np.mean(yield_cvs)
            if mean_cv < 1.0:
                insights.append(f"Low yield coefficient of variation ({mean_cv:.3f}) indicates relatively uniform yields")
                insights.append("Uniform yields reduce the effectiveness of productivity-based filtering")
        
        comparative['insights'] = insights
        
        return comparative
    
    def save_analysis(self, results: Dict[str, Any], output_file: str = 'width_reduction_analysis.json'):
        """Save analysis results to JSON file."""
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {output_file}")
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print formatted summary report."""
        print("\n" + "="*80)
        print("WIDTH REDUCTION ANALYSIS SUMMARY")
        print("="*80)
        
        comparative = results.get('comparative_analysis', {})
        
        if 'error' in comparative:
            print(f"Analysis failed: {comparative['error']}")
            return
        
        # Width reduction summary
        width_summary = comparative.get('width_reduction_summary', {})
        if width_summary.get('mean') is not None:
            print(f"Width Reduction Summary:")
            print(f"  Mean: {width_summary['mean']:.1f}%")
            print(f"  Range: {width_summary['min']:.1f}% - {width_summary['max']:.1f}%")
            print(f"  Expected range: 22-35%")
            print(f"  Within expected range: {width_summary['min'] >= 22 and width_summary['max'] <= 35}")
        
        # Yield variability summary
        yield_summary = comparative.get('yield_variability_summary', {})
        if yield_summary.get('mean_cv') is not None:
            print(f"\nYield Variability Summary:")
            print(f"  Mean coefficient of variation: {yield_summary['mean_cv']:.3f}")
            print(f"  CV range: {yield_summary['cv_range'][0]:.3f} - {yield_summary['cv_range'][1]:.3f}")
        
        # Tier retention summary
        retention_summary = comparative.get('tier_retention_summary', {})
        if retention_summary.get('mean_commercial_retention') is not None:
            print(f"\nTier Retention Summary:")
            print(f"  Mean commercial tier retention: {retention_summary['mean_commercial_retention']:.1%}")
            print(f"  Retention range: {retention_summary['retention_range'][0]:.1%} - {retention_summary['retention_range'][1]:.1%}")
        
        # Individual crop results
        print(f"\nIndividual Crop Results:")
        for crop in ['wheat', 'maize', 'rice']:
            if crop in results and 'error' not in results[crop]:
                analysis = results[crop]
                width_red = analysis['envelope_characteristics'].get('width_reduction_analysis', {}).get('relative_width_reduction_pct', 'N/A')
                yield_cv = analysis['yield_distribution']['coefficient_of_variation']
                comm_retention = analysis['tier_effects'].get('commercial', {}).get('cell_retention_rate', 'N/A')
                
                print(f"  {crop.upper()}: {width_red}% width reduction, CV={yield_cv:.3f}, {comm_retention:.1%} commercial retention")
        
        # Insights
        insights = comparative.get('insights', [])
        if insights:
            print(f"\nKey Insights:")
            for insight in insights:
                print(f"  • {insight}")
        
        print("="*80)


def main():
    """Main analysis execution."""
    analyzer = WidthReductionAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_crops()
    
    # Print summary
    analyzer.print_summary_report(results)
    
    # Save detailed results
    analyzer.save_analysis(results)
    
    return results


if __name__ == '__main__':
    results = main()
"""
Task 2: Sensitivity Analysis Function

Compare H-P envelopes with and without multicropping area capping to assess
the impact on vulnerability rankings and production loss estimates.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Union, Tuple, Any, List
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def sensitivity_analysis_capping(original_data: Union[Dict, pd.DataFrame],
                                capped_data: Union[Dict, pd.DataFrame],
                                production_data: Union[Dict, pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compare H-P envelopes with/without capping.
    
    Parameters:
    -----------
    original_data : dict or DataFrame
        Original harvest area data (before capping)
    capped_data : dict or DataFrame
        Capped harvest area data (after capping)
    production_data : dict or DataFrame, optional
        Production data for envelope calculation
        
    Returns:
    --------
    comparison_results : dict
        - envelope_differences: production loss differences by magnitude
        - magnitude_impacts: percentage changes at each M_D level
        - affected_regions: spatial distribution of capping effects
        - summary_statistics: overall impact metrics
    """
    logger.info("Starting sensitivity analysis for multicropping capping")
    
    comparison_results = {
        'analysis_type': 'multicropping_capping_sensitivity',
        'crops_analyzed': [],
        'envelope_differences': {},
        'magnitude_impacts': {},
        'affected_regions': {},
        'summary_statistics': {},
        'methodology_notes': {
            'capping_threshold': '85.5 km² theoretical cell area',
            'envelope_calculation': 'H-P envelope bounds comparison',
            'magnitude_range': '0-6 AgriRichter scale',
            'assumptions': [
                'Production scales proportionally with capped harvest areas',
                'Spatial patterns preserved after capping',
                'Envelope monotonicity maintained'
            ]
        }
    }
    
    try:
        # Extract crop data for analysis
        crop_comparisons = _extract_crop_comparisons(original_data, capped_data, production_data)
        
        for crop_name, crop_data in crop_comparisons.items():
            logger.info(f"Analyzing sensitivity for {crop_name}")
            
            # Calculate envelopes for both original and capped data
            original_envelope = _calculate_envelope_bounds(
                crop_data['original_harvest'],
                crop_data['original_production'],
                crop_name + '_original'
            )
            
            capped_envelope = _calculate_envelope_bounds(
                crop_data['capped_harvest'],
                crop_data['capped_production'],
                crop_name + '_capped'
            )
            
            # Compare envelopes at different magnitude levels
            magnitude_comparison = _compare_envelope_magnitudes(
                original_envelope, capped_envelope, crop_name
            )
            
            # Analyze spatial distribution of effects
            spatial_analysis = _analyze_spatial_effects(
                crop_data['original_harvest'],
                crop_data['capped_harvest'],
                crop_name
            )
            
            # Store results
            comparison_results['crops_analyzed'].append(crop_name)
            comparison_results['envelope_differences'][crop_name] = {
                'original_envelope': original_envelope,
                'capped_envelope': capped_envelope
            }
            comparison_results['magnitude_impacts'][crop_name] = magnitude_comparison
            comparison_results['affected_regions'][crop_name] = spatial_analysis
        
        # Generate summary statistics
        comparison_results['summary_statistics'] = _generate_summary_statistics(
            comparison_results
        )
        
        logger.info("Sensitivity analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {str(e)}")
        comparison_results['error'] = str(e)
        raise
    
    return comparison_results


def _extract_crop_comparisons(original_data: Union[Dict, pd.DataFrame],
                             capped_data: Union[Dict, pd.DataFrame],
                             production_data: Union[Dict, pd.DataFrame] = None) -> Dict[str, Dict]:
    """Extract crop data for comparison analysis."""
    
    crop_comparisons = {}
    
    if isinstance(original_data, dict) and isinstance(capped_data, dict):
        # Dictionary format
        for crop_name in original_data.keys():
            if crop_name in capped_data:
                original_harvest = np.array(original_data[crop_name])
                capped_harvest = np.array(capped_data[crop_name])
                
                # Use production data if available, otherwise estimate from harvest
                if production_data and crop_name in production_data:
                    original_production = np.array(production_data[crop_name])
                    # Scale production proportionally with capped harvest
                    production_scale = np.divide(capped_harvest, original_harvest,
                                               out=np.ones_like(capped_harvest),
                                               where=original_harvest>0)
                    capped_production = original_production * production_scale
                else:
                    # Estimate production from harvest (assuming constant yield)
                    avg_yield = 500  # mt/km² typical yield
                    original_production = original_harvest / 100 * avg_yield  # Convert to mt
                    capped_production = capped_harvest / 100 * avg_yield
                
                crop_comparisons[crop_name] = {
                    'original_harvest': original_harvest,
                    'capped_harvest': capped_harvest,
                    'original_production': original_production,
                    'capped_production': capped_production
                }
    
    elif isinstance(original_data, pd.DataFrame) and isinstance(capped_data, pd.DataFrame):
        # DataFrame format
        crop_columns = [col for col in original_data.columns if col.endswith('_A')]
        
        for crop_col in crop_columns:
            if crop_col in capped_data.columns:
                crop_name = crop_col.replace('_A', '').lower()
                
                original_harvest = original_data[crop_col].values
                capped_harvest = capped_data[crop_col].values
                
                # Use production data if available
                if isinstance(production_data, pd.DataFrame):
                    prod_col = crop_col  # Assume same column naming
                    if prod_col in production_data.columns:
                        original_production = production_data[prod_col].values
                        # Scale production proportionally
                        production_scale = np.divide(capped_harvest, original_harvest,
                                                   out=np.ones_like(capped_harvest),
                                                   where=original_harvest>0)
                        capped_production = original_production * production_scale
                    else:
                        # Estimate from harvest
                        avg_yield = 500
                        original_production = original_harvest / 100 * avg_yield
                        capped_production = capped_harvest / 100 * avg_yield
                else:
                    # Estimate from harvest
                    avg_yield = 500
                    original_production = original_harvest / 100 * avg_yield
                    capped_production = capped_harvest / 100 * avg_yield
                
                crop_comparisons[crop_name] = {
                    'original_harvest': original_harvest,
                    'capped_harvest': capped_harvest,
                    'original_production': original_production,
                    'capped_production': capped_production
                }
    
    return crop_comparisons


def _calculate_envelope_bounds(harvest_areas: np.ndarray,
                              production_values: np.ndarray,
                              label: str) -> Dict[str, Any]:
    """Calculate simplified H-P envelope bounds for comparison."""
    
    # Convert harvest areas to km²
    harvest_km2 = harvest_areas / 100 if np.max(harvest_areas) > 1000 else harvest_areas
    
    # Filter valid data
    valid_mask = (harvest_km2 > 0) & (production_values > 0)
    valid_harvest = harvest_km2[valid_mask]
    valid_production = production_values[valid_mask]
    
    if len(valid_harvest) == 0:
        return {
            'label': label,
            'harvest_area': np.array([]),
            'upper_bound': np.array([]),
            'lower_bound': np.array([]),
            'total_harvest': 0,
            'total_production': 0
        }
    
    # Calculate yields for sorting
    yields = valid_production / valid_harvest
    
    # Sort by yield for envelope bounds
    yield_sort_indices = np.argsort(yields)
    
    # Upper bound: highest yields first
    upper_harvest_cumsum = np.cumsum(valid_harvest[yield_sort_indices[::-1]])
    upper_production_cumsum = np.cumsum(valid_production[yield_sort_indices[::-1]])
    
    # Lower bound: lowest yields first
    lower_harvest_cumsum = np.cumsum(valid_harvest[yield_sort_indices])
    lower_production_cumsum = np.cumsum(valid_production[yield_sort_indices])
    
    # Create common harvest area grid for comparison
    max_harvest = min(upper_harvest_cumsum[-1], lower_harvest_cumsum[-1])
    harvest_grid = np.linspace(0, max_harvest, 100)
    
    # Interpolate bounds to common grid
    upper_bound_interp = np.interp(harvest_grid, upper_harvest_cumsum, upper_production_cumsum)
    lower_bound_interp = np.interp(harvest_grid, lower_harvest_cumsum, lower_production_cumsum)
    
    return {
        'label': label,
        'harvest_area': harvest_grid,
        'upper_bound': upper_bound_interp,
        'lower_bound': lower_bound_interp,
        'total_harvest': float(valid_harvest.sum()),
        'total_production': float(valid_production.sum()),
        'envelope_width_pct': float((upper_bound_interp[-1] - lower_bound_interp[-1]) / 
                                   upper_bound_interp[-1] * 100) if upper_bound_interp[-1] > 0 else 0
    }


def _compare_envelope_magnitudes(original_envelope: Dict,
                                capped_envelope: Dict,
                                crop_name: str) -> Dict[str, Any]:
    """Compare envelope bounds at different magnitude levels."""
    
    magnitude_comparison = {
        'crop_name': crop_name,
        'magnitude_levels': [],
        'production_differences': [],
        'percentage_changes': [],
        'harvest_area_coverage': []
    }
    
    # Define magnitude levels (0-6 AgriRichter scale)
    magnitude_levels = np.arange(0, 7)
    
    for magnitude in magnitude_levels:
        # Calculate coverage percentage for this magnitude
        coverage_pct = magnitude / 6.0  # 0 to 100% coverage
        
        # Find corresponding harvest area
        max_harvest = min(original_envelope['total_harvest'], 
                         capped_envelope['total_harvest'])
        target_harvest = max_harvest * coverage_pct
        
        # Interpolate production values at this harvest level
        original_production = np.interp(target_harvest, 
                                      original_envelope['harvest_area'],
                                      original_envelope['upper_bound'])
        
        capped_production = np.interp(target_harvest,
                                    capped_envelope['harvest_area'],
                                    capped_envelope['upper_bound'])
        
        # Calculate differences
        production_diff = capped_production - original_production
        percentage_change = (production_diff / original_production * 100) if original_production > 0 else 0
        
        magnitude_comparison['magnitude_levels'].append(float(magnitude))
        magnitude_comparison['production_differences'].append(float(production_diff))
        magnitude_comparison['percentage_changes'].append(float(percentage_change))
        magnitude_comparison['harvest_area_coverage'].append(float(coverage_pct * 100))
    
    return magnitude_comparison


def _analyze_spatial_effects(original_harvest: np.ndarray,
                            capped_harvest: np.ndarray,
                            crop_name: str) -> Dict[str, Any]:
    """Analyze spatial distribution of capping effects."""
    
    # Calculate differences
    harvest_diff = capped_harvest - original_harvest
    cells_affected = np.sum(harvest_diff != 0)
    total_cells = len(original_harvest)
    
    # Calculate impact statistics
    total_area_reduced = np.sum(-harvest_diff[harvest_diff < 0])  # Only reductions
    max_reduction = np.min(harvest_diff) if np.any(harvest_diff < 0) else 0
    
    spatial_analysis = {
        'crop_name': crop_name,
        'total_cells': int(total_cells),
        'cells_affected': int(cells_affected),
        'cells_affected_pct': float(cells_affected / total_cells * 100),
        'total_area_reduced_ha': float(total_area_reduced),
        'max_area_reduction_ha': float(-max_reduction),
        'mean_area_reduction_ha': float(total_area_reduced / cells_affected) if cells_affected > 0 else 0,
        'spatial_distribution': {
            'uniform': cells_affected < total_cells * 0.01,  # Less than 1% affected = uniform
            'clustered': cells_affected > total_cells * 0.05,  # More than 5% affected = clustered
            'scattered': True  # Default assumption
        }
    }
    
    return spatial_analysis


def _generate_summary_statistics(comparison_results: Dict) -> Dict[str, Any]:
    """Generate overall summary statistics for sensitivity analysis."""
    
    summary = {
        'total_crops_analyzed': len(comparison_results['crops_analyzed']),
        'overall_impact': 'low',  # Will be determined based on analysis
        'max_magnitude_change_pct': 0,
        'total_cells_affected': 0,
        'total_area_reduced_ha': 0,
        'envelope_width_changes': {},
        'vulnerability_ranking_impact': 'minimal'
    }
    
    # Aggregate statistics across crops
    for crop_name in comparison_results['crops_analyzed']:
        # Magnitude impacts
        if crop_name in comparison_results['magnitude_impacts']:
            mag_data = comparison_results['magnitude_impacts'][crop_name]
            max_change = max(abs(x) for x in mag_data['percentage_changes'])
            summary['max_magnitude_change_pct'] = max(summary['max_magnitude_change_pct'], max_change)
        
        # Spatial effects
        if crop_name in comparison_results['affected_regions']:
            spatial_data = comparison_results['affected_regions'][crop_name]
            summary['total_cells_affected'] += spatial_data['cells_affected']
            summary['total_area_reduced_ha'] += spatial_data['total_area_reduced_ha']
        
        # Envelope width changes
        if crop_name in comparison_results['envelope_differences']:
            env_data = comparison_results['envelope_differences'][crop_name]
            original_width = env_data['original_envelope']['envelope_width_pct']
            capped_width = env_data['capped_envelope']['envelope_width_pct']
            width_change = capped_width - original_width
            summary['envelope_width_changes'][crop_name] = {
                'original_width_pct': original_width,
                'capped_width_pct': capped_width,
                'width_change_pct': width_change
            }
    
    # Determine overall impact level
    if summary['max_magnitude_change_pct'] > 10:
        summary['overall_impact'] = 'high'
        summary['vulnerability_ranking_impact'] = 'significant'
    elif summary['max_magnitude_change_pct'] > 5:
        summary['overall_impact'] = 'moderate'
        summary['vulnerability_ranking_impact'] = 'moderate'
    else:
        summary['overall_impact'] = 'low'
        summary['vulnerability_ranking_impact'] = 'minimal'
    
    return summary


def generate_sensitivity_report(comparison_results: Dict) -> str:
    """Generate comprehensive sensitivity analysis report."""
    
    report_lines = [
        "MULTICROPPING CAPPING SENSITIVITY ANALYSIS REPORT",
        "=" * 60,
        f"Analysis Type: {comparison_results['analysis_type']}",
        f"Crops Analyzed: {', '.join(comparison_results['crops_analyzed'])}",
        ""
    ]
    
    # Summary statistics
    if 'summary_statistics' in comparison_results:
        summary = comparison_results['summary_statistics']
        report_lines.extend([
            "OVERALL IMPACT ASSESSMENT:",
            "-" * 30,
            f"Overall Impact Level: {summary['overall_impact'].upper()}",
            f"Vulnerability Ranking Impact: {summary['vulnerability_ranking_impact']}",
            f"Maximum Magnitude Change: {summary['max_magnitude_change_pct']:.1f}%",
            f"Total Cells Affected: {summary['total_cells_affected']:,}",
            f"Total Area Reduced: {summary['total_area_reduced_ha']:.2f} ha",
            ""
        ])
    
    # Crop-specific results
    report_lines.append("CROP-SPECIFIC RESULTS:")
    report_lines.append("-" * 30)
    
    for crop_name in comparison_results['crops_analyzed']:
        report_lines.append(f"\n{crop_name.upper()}:")
        
        # Magnitude impacts
        if crop_name in comparison_results['magnitude_impacts']:
            mag_data = comparison_results['magnitude_impacts'][crop_name]
            max_change = max(abs(x) for x in mag_data['percentage_changes'])
            report_lines.append(f"  Maximum magnitude change: {max_change:.1f}%")
        
        # Spatial effects
        if crop_name in comparison_results['affected_regions']:
            spatial_data = comparison_results['affected_regions'][crop_name]
            report_lines.extend([
                f"  Cells affected: {spatial_data['cells_affected']:,} ({spatial_data['cells_affected_pct']:.2f}%)",
                f"  Area reduced: {spatial_data['total_area_reduced_ha']:.2f} ha",
                f"  Max reduction per cell: {spatial_data['max_area_reduction_ha']:.2f} ha"
            ])
        
        # Envelope width changes
        if crop_name in comparison_results.get('summary_statistics', {}).get('envelope_width_changes', {}):
            width_data = comparison_results['summary_statistics']['envelope_width_changes'][crop_name]
            report_lines.append(f"  Envelope width change: {width_data['original_width_pct']:.1f}% → {width_data['capped_width_pct']:.1f}%")
    
    # Methodology notes
    if 'methodology_notes' in comparison_results:
        notes = comparison_results['methodology_notes']
        report_lines.extend([
            "",
            "METHODOLOGY:",
            "-" * 15,
            f"Capping Threshold: {notes['capping_threshold']}",
            f"Envelope Calculation: {notes['envelope_calculation']}",
            f"Magnitude Range: {notes['magnitude_range']}",
            "",
            "ASSUMPTIONS:"
        ])
        
        for assumption in notes['assumptions']:
            report_lines.append(f"- {assumption}")
    
    return "\n".join(report_lines)


def save_sensitivity_results(comparison_results: Dict,
                           output_dir: str = "multicropping_fix/output_data") -> Dict[str, str]:
    """Save sensitivity analysis results to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Save comparison results
    import json
    
    def convert_numpy_types(obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_file = output_path / "sensitivity_comparison.json"
    json_compatible_results = convert_numpy_types(comparison_results)
    
    with open(results_file, 'w') as f:
        json.dump(json_compatible_results, f, indent=2)
    file_paths['results'] = str(results_file)
    
    # Save report
    report_file = output_path / "sensitivity_analysis_report.txt"
    report = generate_sensitivity_report(comparison_results)
    with open(report_file, 'w') as f:
        f.write(report)
    file_paths['report'] = str(report_file)
    
    # Save CSV summary for easy analysis
    csv_file = output_path / "sensitivity_summary.csv"
    _save_sensitivity_csv(comparison_results, csv_file)
    file_paths['csv_summary'] = str(csv_file)
    
    logger.info(f"Sensitivity analysis results saved to {output_dir}")
    
    return file_paths


def _save_sensitivity_csv(comparison_results: Dict, csv_file: Path):
    """Save sensitivity analysis summary as CSV."""
    
    rows = []
    
    for crop_name in comparison_results['crops_analyzed']:
        # Get magnitude impacts
        if crop_name in comparison_results['magnitude_impacts']:
            mag_data = comparison_results['magnitude_impacts'][crop_name]
            
            for i, magnitude in enumerate(mag_data['magnitude_levels']):
                row = {
                    'crop': crop_name,
                    'magnitude_level': magnitude,
                    'harvest_coverage_pct': mag_data['harvest_area_coverage'][i],
                    'production_difference': mag_data['production_differences'][i],
                    'percentage_change': mag_data['percentage_changes'][i]
                }
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"Sensitivity summary CSV saved to {csv_file}")


# Convenience function
def quick_sensitivity_analysis(original_data: Union[Dict, pd.DataFrame],
                              capped_data: Union[Dict, pd.DataFrame],
                              production_data: Union[Dict, pd.DataFrame] = None,
                              save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function for quick sensitivity analysis with automatic saving.
    
    Parameters:
    -----------
    original_data : dict or DataFrame
        Original harvest area data
    capped_data : dict or DataFrame
        Capped harvest area data
    production_data : dict or DataFrame, optional
        Production data
    save_results : bool
        Whether to save results to files
        
    Returns:
    --------
    comparison_results : dict
        Complete sensitivity analysis results
    """
    # Run sensitivity analysis
    comparison_results = sensitivity_analysis_capping(
        original_data, capped_data, production_data
    )
    
    # Save results if requested
    if save_results:
        file_paths = save_sensitivity_results(comparison_results)
        comparison_results['output_files'] = file_paths
    
    return comparison_results
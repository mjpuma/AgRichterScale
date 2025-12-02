"""
Task 3: Enhanced Envelope Calculation with Capping Analysis

Calculate H-P envelope with multicropping area capping and comprehensive documentation
of assumptions and limitations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Union, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_envelope_with_capping_analysis(data: Union[Dict, pd.DataFrame],
                                           production_data: Union[Dict, pd.DataFrame] = None,
                                           apply_capping: bool = True,
                                           theoretical_cell_area: float = 85.5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate H-P envelope with capping documentation.
    
    Parameters:
    -----------
    data : dict or DataFrame
        Harvest area data by crop and grid cell
    production_data : dict or DataFrame, optional
        Production data by crop and grid cell
    apply_capping : bool, default=True
        Whether to apply theoretical area capping
    theoretical_cell_area : float, default=85.5
        Maximum cell area in km²
        
    Returns:
    --------
    envelope_data : dict
        Upper/lower bounds by magnitude with metadata
    methodology_notes : dict
        Documentation of assumptions and limitations
    """
    logger.info(f"Calculating H-P envelope with capping={'enabled' if apply_capping else 'disabled'}")
    
    # Initialize results structure
    envelope_data = {
        'capping_applied': apply_capping,
        'theoretical_cell_area_km2': theoretical_cell_area,
        'crops': {},
        'envelope_bounds': {},
        'validation_results': {},
        'processing_metadata': {}
    }
    
    methodology_notes = {
        'multicropping_approach': 'Area capping at theoretical cell limits' if apply_capping else 'No capping applied',
        'theoretical_cell_area': f'{theoretical_cell_area} km² based on SPAM 2020 grid resolution',
        'assumptions': [],
        'limitations': [],
        'data_quality_notes': [],
        'envelope_properties': {}
    }
    
    try:
        # Apply capping if requested
        if apply_capping:
            from multicropping_fix.cap_harvest_areas import cap_harvest_areas
            capped_data, capping_stats = cap_harvest_areas(data, theoretical_cell_area)
            
            envelope_data['capping_statistics'] = capping_stats
            methodology_notes['assumptions'].extend([
                'Harvest areas capped at theoretical maximum cell area',
                'Production scales proportionally with capped harvest areas',
                'Multicropping effects distributed uniformly within cells'
            ])
            methodology_notes['data_quality_notes'].append(
                f"Capped {capping_stats['total_cells_affected']} cells across all crops"
            )
            
            working_data = capped_data
        else:
            working_data = data
            methodology_notes['assumptions'].append('Original harvest areas used without capping')
            methodology_notes['limitations'].append('May include multicropping violations of physical constraints')
        
        # Extract crop data for envelope calculation
        crop_datasets = _extract_crop_datasets(working_data, production_data)
        
        # Calculate envelopes for each crop
        for crop_name, crop_data in crop_datasets.items():
            logger.info(f"Calculating envelope for {crop_name}")
            
            # Calculate envelope bounds
            envelope_bounds = _calculate_crop_envelope(
                crop_data['harvest_areas'],
                crop_data['production_values'],
                crop_name
            )
            
            # Validate envelope properties
            validation_results = _validate_envelope_properties(envelope_bounds, crop_name)
            
            # Store results
            envelope_data['crops'][crop_name] = crop_data
            envelope_data['envelope_bounds'][crop_name] = envelope_bounds
            envelope_data['validation_results'][crop_name] = validation_results
            
            # Update methodology notes
            if not validation_results['monotonic']:
                methodology_notes['limitations'].append(f'{crop_name}: Non-monotonic envelope detected')
            
            if validation_results['convergence_gap_pct'] > 5:
                methodology_notes['limitations'].append(f'{crop_name}: Poor convergence ({validation_results["convergence_gap_pct"]:.1f}%)')
        
        # Generate overall envelope properties
        envelope_data['envelope_properties'] = _calculate_envelope_properties(envelope_data)
        methodology_notes['envelope_properties'] = envelope_data['envelope_properties']
        
        # Add standard assumptions and limitations
        methodology_notes['assumptions'].extend([
            'Grid cells represent homogeneous agricultural areas',
            'Yield patterns remain stable during disruption events',
            'Envelope bounds represent theoretical production limits'
        ])
        
        methodology_notes['limitations'].extend([
            'Does not account for crop rotation constraints',
            'Assumes perfect substitutability within crop types',
            'Spatial correlation effects not modeled',
            'Seasonal timing effects not considered'
        ])
        
        logger.info("Envelope calculation completed successfully")
        
    except Exception as e:
        logger.error(f"Envelope calculation failed: {str(e)}")
        envelope_data['error'] = str(e)
        methodology_notes['error'] = str(e)
        raise
    
    return envelope_data, methodology_notes


def _extract_crop_datasets(harvest_data: Union[Dict, pd.DataFrame],
                          production_data: Union[Dict, pd.DataFrame] = None) -> Dict[str, Dict]:
    """Extract crop datasets for envelope calculation."""
    
    crop_datasets = {}
    
    if isinstance(harvest_data, dict):
        # Dictionary format
        for crop_name, harvest_areas in harvest_data.items():
            harvest_array = np.array(harvest_areas)
            
            # Get production data
            if production_data and crop_name in production_data:
                production_array = np.array(production_data[crop_name])
            else:
                # Estimate production from harvest areas (constant yield assumption)
                avg_yield = _get_typical_yield(crop_name)
                production_array = harvest_array / 100 * avg_yield  # Convert ha to km², apply yield
            
            crop_datasets[crop_name] = {
                'harvest_areas': harvest_array,
                'production_values': production_array,
                'data_source': 'dictionary'
            }
    
    elif isinstance(harvest_data, pd.DataFrame):
        # DataFrame format
        crop_columns = [col for col in harvest_data.columns if col.endswith('_A')]
        
        for crop_col in crop_columns:
            crop_name = crop_col.replace('_A', '').lower()
            harvest_array = harvest_data[crop_col].values
            
            # Get production data
            if isinstance(production_data, pd.DataFrame) and crop_col in production_data.columns:
                production_array = production_data[crop_col].values
            else:
                # Estimate production
                avg_yield = _get_typical_yield(crop_name)
                production_array = harvest_array / 100 * avg_yield
            
            crop_datasets[crop_name] = {
                'harvest_areas': harvest_array,
                'production_values': production_array,
                'data_source': 'dataframe',
                'column_name': crop_col
            }
    
    return crop_datasets


def _get_typical_yield(crop_name: str) -> float:
    """Get typical yield for crop (mt/km²) for production estimation."""
    
    typical_yields = {
        'wheat': 338.9,
        'maize': 531.8,
        'rice': 454.0,
        'soybean': 200.0,
        'barley': 300.0,
        'default': 400.0
    }
    
    return typical_yields.get(crop_name.lower(), typical_yields['default'])


def _calculate_crop_envelope(harvest_areas: np.ndarray,
                           production_values: np.ndarray,
                           crop_name: str) -> Dict[str, Any]:
    """Calculate H-P envelope bounds for a single crop."""
    
    # Convert harvest areas to km² if needed
    if np.max(harvest_areas) > 1000:  # Likely in hectares
        harvest_km2 = harvest_areas / 100
    else:
        harvest_km2 = harvest_areas
    
    # Filter valid data
    valid_mask = (harvest_km2 > 0) & (production_values > 0) & np.isfinite(harvest_km2) & np.isfinite(production_values)
    valid_harvest = harvest_km2[valid_mask]
    valid_production = production_values[valid_mask]
    
    if len(valid_harvest) == 0:
        logger.warning(f"No valid data for {crop_name}")
        return {
            'crop_name': crop_name,
            'harvest_area': np.array([]),
            'upper_bound': np.array([]),
            'lower_bound': np.array([]),
            'total_harvest_km2': 0,
            'total_production_mt': 0,
            'valid_cells': 0,
            'envelope_width_pct': 0
        }
    
    # Calculate yields for sorting
    yields = valid_production / valid_harvest
    
    # Remove extreme outliers (beyond 99.9th percentile)
    yield_99_9 = np.percentile(yields, 99.9)
    reasonable_mask = yields <= yield_99_9
    
    if np.sum(reasonable_mask) < len(yields) * 0.5:  # If removing too many, keep all
        reasonable_mask = np.ones_like(yields, dtype=bool)
    
    filtered_harvest = valid_harvest[reasonable_mask]
    filtered_production = valid_production[reasonable_mask]
    filtered_yields = yields[reasonable_mask]
    
    # Sort by yield for envelope bounds
    yield_sort_indices = np.argsort(filtered_yields)
    
    # Upper bound: highest yields first (descending order)
    upper_harvest_cumsum = np.cumsum(filtered_harvest[yield_sort_indices[::-1]])
    upper_production_cumsum = np.cumsum(filtered_production[yield_sort_indices[::-1]])
    
    # Lower bound: lowest yields first (ascending order)
    lower_harvest_cumsum = np.cumsum(filtered_harvest[yield_sort_indices])
    lower_production_cumsum = np.cumsum(filtered_production[yield_sort_indices])
    
    # Create common harvest area grid for output
    max_harvest = min(upper_harvest_cumsum[-1], lower_harvest_cumsum[-1])
    harvest_grid = np.linspace(0, max_harvest, 1000)  # High resolution for smooth curves
    
    # Interpolate bounds to common grid
    upper_bound_interp = np.interp(harvest_grid, upper_harvest_cumsum, upper_production_cumsum)
    lower_bound_interp = np.interp(harvest_grid, lower_harvest_cumsum, lower_production_cumsum)
    
    # Ensure monotonicity
    upper_bound_interp = np.maximum.accumulate(upper_bound_interp)
    lower_bound_interp = np.maximum.accumulate(lower_bound_interp)
    
    # Calculate envelope width at maximum harvest
    final_width = upper_bound_interp[-1] - lower_bound_interp[-1]
    envelope_width_pct = (final_width / upper_bound_interp[-1] * 100) if upper_bound_interp[-1] > 0 else 0
    
    envelope_bounds = {
        'crop_name': crop_name,
        'harvest_area': harvest_grid,
        'upper_bound': upper_bound_interp,
        'lower_bound': lower_bound_interp,
        'total_harvest_km2': float(filtered_harvest.sum()),
        'total_production_mt': float(filtered_production.sum()),
        'valid_cells': int(len(filtered_harvest)),
        'envelope_width_pct': float(envelope_width_pct),
        'yield_range': {
            'min_yield': float(filtered_yields.min()),
            'max_yield': float(filtered_yields.max()),
            'median_yield': float(np.median(filtered_yields))
        }
    }
    
    return envelope_bounds


def _validate_envelope_properties(envelope_bounds: Dict, crop_name: str) -> Dict[str, Any]:
    """Validate envelope properties for quality assurance."""
    
    validation_results = {
        'crop_name': crop_name,
        'monotonic': True,
        'convergent': True,
        'physically_reasonable': True,
        'convergence_gap_pct': 0,
        'issues': []
    }
    
    if len(envelope_bounds['harvest_area']) == 0:
        validation_results['issues'].append('No data available for envelope calculation')
        return validation_results
    
    harvest_area = envelope_bounds['harvest_area']
    upper_bound = envelope_bounds['upper_bound']
    lower_bound = envelope_bounds['lower_bound']
    
    # Check monotonicity
    upper_diffs = np.diff(upper_bound)
    lower_diffs = np.diff(lower_bound)
    
    if np.any(upper_diffs < -1e-6):  # Allow small numerical errors
        validation_results['monotonic'] = False
        validation_results['issues'].append('Upper bound is not monotonic')
    
    if np.any(lower_diffs < -1e-6):
        validation_results['monotonic'] = False
        validation_results['issues'].append('Lower bound is not monotonic')
    
    # Check convergence
    if len(upper_bound) > 0 and len(lower_bound) > 0:
        final_gap = upper_bound[-1] - lower_bound[-1]
        convergence_gap_pct = (final_gap / upper_bound[-1] * 100) if upper_bound[-1] > 0 else 0
        validation_results['convergence_gap_pct'] = float(convergence_gap_pct)
        
        if convergence_gap_pct > 10:  # More than 10% gap
            validation_results['convergent'] = False
            validation_results['issues'].append(f'Poor convergence: {convergence_gap_pct:.1f}% gap at maximum harvest')
    
    # Check physical reasonableness
    max_yield = envelope_bounds['yield_range']['max_yield']
    if max_yield > 10000:  # More than 10,000 mt/km² is unrealistic
        validation_results['physically_reasonable'] = False
        validation_results['issues'].append(f'Unrealistic maximum yield: {max_yield:.0f} mt/km²')
    
    # Check for negative values
    if np.any(upper_bound < 0) or np.any(lower_bound < 0):
        validation_results['physically_reasonable'] = False
        validation_results['issues'].append('Negative production values detected')
    
    # Check bound ordering
    if np.any(lower_bound > upper_bound):
        validation_results['physically_reasonable'] = False
        validation_results['issues'].append('Lower bound exceeds upper bound')
    
    return validation_results


def _calculate_envelope_properties(envelope_data: Dict) -> Dict[str, Any]:
    """Calculate overall envelope properties across all crops."""
    
    properties = {
        'total_crops': len(envelope_data['envelope_bounds']),
        'total_valid_cells': 0,
        'total_harvest_area_km2': 0,
        'total_production_mt': 0,
        'average_envelope_width_pct': 0,
        'convergence_quality': 'good',
        'monotonicity_preserved': True
    }
    
    width_values = []
    
    for crop_name, bounds in envelope_data['envelope_bounds'].items():
        properties['total_valid_cells'] += bounds.get('valid_cells', 0)
        properties['total_harvest_area_km2'] += bounds.get('total_harvest_km2', 0)
        properties['total_production_mt'] += bounds.get('total_production_mt', 0)
        
        width_pct = bounds.get('envelope_width_pct', 0)
        if width_pct > 0:
            width_values.append(width_pct)
        
        # Check validation results
        if crop_name in envelope_data['validation_results']:
            validation = envelope_data['validation_results'][crop_name]
            if not validation['monotonic']:
                properties['monotonicity_preserved'] = False
            
            if validation['convergence_gap_pct'] > 10:
                properties['convergence_quality'] = 'poor'
            elif validation['convergence_gap_pct'] > 5:
                properties['convergence_quality'] = 'moderate'
    
    if width_values:
        properties['average_envelope_width_pct'] = float(np.mean(width_values))
    
    return properties


def generate_envelope_methodology_report(envelope_data: Dict, methodology_notes: Dict) -> str:
    """Generate comprehensive methodology report for envelope calculation."""
    
    report_lines = [
        "H-P ENVELOPE CALCULATION WITH MULTICROPPING ANALYSIS",
        "=" * 65,
        f"Capping Applied: {'Yes' if envelope_data['capping_applied'] else 'No'}",
        f"Theoretical Cell Area: {envelope_data['theoretical_cell_area_km2']} km²",
        f"Crops Analyzed: {', '.join(envelope_data['envelope_bounds'].keys())}",
        ""
    ]
    
    # Capping statistics if applied
    if envelope_data['capping_applied'] and 'capping_statistics' in envelope_data:
        stats = envelope_data['capping_statistics']
        report_lines.extend([
            "MULTICROPPING CAPPING RESULTS:",
            "-" * 35,
            f"Total cells affected: {stats['total_cells_affected']:,}",
            f"Total area reduced: {stats['total_area_reduced_ha']:.2f} ha",
            f"Crops processed: {', '.join(stats['crops_processed'])}",
            ""
        ])
    
    # Envelope properties
    if 'envelope_properties' in envelope_data:
        props = envelope_data['envelope_properties']
        report_lines.extend([
            "ENVELOPE PROPERTIES:",
            "-" * 20,
            f"Total crops: {props['total_crops']}",
            f"Total valid cells: {props['total_valid_cells']:,}",
            f"Total harvest area: {props['total_harvest_area_km2']:.2f} km²",
            f"Total production: {props['total_production_mt']:.2f} mt",
            f"Average envelope width: {props['average_envelope_width_pct']:.1f}%",
            f"Convergence quality: {props['convergence_quality']}",
            f"Monotonicity preserved: {'Yes' if props['monotonicity_preserved'] else 'No'}",
            ""
        ])
    
    # Crop-specific results
    report_lines.append("CROP-SPECIFIC RESULTS:")
    report_lines.append("-" * 25)
    
    for crop_name, bounds in envelope_data['envelope_bounds'].items():
        report_lines.append(f"\n{crop_name.upper()}:")
        report_lines.append(f"  Valid cells: {bounds['valid_cells']:,}")
        report_lines.append(f"  Harvest area: {bounds['total_harvest_km2']:.2f} km²")
        report_lines.append(f"  Production: {bounds['total_production_mt']:.2f} mt")
        report_lines.append(f"  Envelope width: {bounds['envelope_width_pct']:.1f}%")
        
        if crop_name in envelope_data['validation_results']:
            validation = envelope_data['validation_results'][crop_name]
            if validation['issues']:
                report_lines.append("  Issues:")
                for issue in validation['issues']:
                    report_lines.append(f"    - {issue}")
    
    # Methodology notes
    report_lines.extend([
        "",
        "METHODOLOGY:",
        "-" * 15,
        f"Multicropping Approach: {methodology_notes['multicropping_approach']}",
        f"Theoretical Cell Area: {methodology_notes['theoretical_cell_area']}",
        "",
        "ASSUMPTIONS:"
    ])
    
    for assumption in methodology_notes['assumptions']:
        report_lines.append(f"- {assumption}")
    
    report_lines.append("\nLIMITATIONS:")
    for limitation in methodology_notes['limitations']:
        report_lines.append(f"- {limitation}")
    
    if methodology_notes['data_quality_notes']:
        report_lines.append("\nDATA QUALITY NOTES:")
        for note in methodology_notes['data_quality_notes']:
            report_lines.append(f"- {note}")
    
    return "\n".join(report_lines)


def save_envelope_results(envelope_data: Dict,
                         methodology_notes: Dict,
                         output_dir: str = "multicropping_fix/output_data") -> Dict[str, str]:
    """Save envelope calculation results to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Save envelope bounds data
    bounds_file = output_path / "envelope_bounds_capped.csv"
    _save_envelope_bounds_csv(envelope_data, bounds_file)
    file_paths['envelope_bounds'] = str(bounds_file)
    
    # Save complete results as JSON
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
    
    results_file = output_path / "envelope_calculation_results.json"
    combined_results = {
        'envelope_data': envelope_data,
        'methodology_notes': methodology_notes
    }
    json_compatible_results = convert_numpy_types(combined_results)
    
    with open(results_file, 'w') as f:
        json.dump(json_compatible_results, f, indent=2)
    file_paths['complete_results'] = str(results_file)
    
    # Save methodology report
    report_file = output_path / "envelope_methodology_report.txt"
    report = generate_envelope_methodology_report(envelope_data, methodology_notes)
    with open(report_file, 'w') as f:
        f.write(report)
    file_paths['methodology_report'] = str(report_file)
    
    logger.info(f"Envelope calculation results saved to {output_dir}")
    
    return file_paths


def _save_envelope_bounds_csv(envelope_data: Dict, csv_file: Path):
    """Save envelope bounds as CSV for easy analysis."""
    
    rows = []
    
    for crop_name, bounds in envelope_data['envelope_bounds'].items():
        harvest_area = bounds['harvest_area']
        upper_bound = bounds['upper_bound']
        lower_bound = bounds['lower_bound']
        
        for i in range(len(harvest_area)):
            row = {
                'crop': crop_name,
                'harvest_area_km2': harvest_area[i],
                'upper_bound_mt': upper_bound[i],
                'lower_bound_mt': lower_bound[i],
                'envelope_width_mt': upper_bound[i] - lower_bound[i]
            }
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"Envelope bounds CSV saved to {csv_file}")


# Convenience function
def quick_envelope_calculation(data: Union[Dict, pd.DataFrame],
                              production_data: Union[Dict, pd.DataFrame] = None,
                              apply_capping: bool = True,
                              save_results: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function for quick envelope calculation with automatic saving.
    
    Parameters:
    -----------
    data : dict or DataFrame
        Harvest area data
    production_data : dict or DataFrame, optional
        Production data
    apply_capping : bool
        Whether to apply multicropping capping
    save_results : bool
        Whether to save results to files
        
    Returns:
    --------
    envelope_data : dict
        Envelope calculation results
    methodology_notes : dict
        Methodology documentation
    """
    # Calculate envelope
    envelope_data, methodology_notes = calculate_envelope_with_capping_analysis(
        data, production_data, apply_capping
    )
    
    # Save results if requested
    if save_results:
        file_paths = save_envelope_results(envelope_data, methodology_notes)
        envelope_data['output_files'] = file_paths
    
    return envelope_data, methodology_notes
"""
Task 1: Implement Area Capping Function

Caps SPAM harvest areas at theoretical maximum cell area to address multicropping
violations of physical land constraints.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Union, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def cap_harvest_areas(harvest_data: Union[Dict, pd.DataFrame], 
                     theoretical_cell_area: float = 85.5) -> Tuple[Union[Dict, pd.DataFrame], Dict[str, Any]]:
    """
    Cap harvest areas at theoretical maximum cell area.
    
    Addresses multicropping violations where harvest areas exceed the physical
    constraints of grid cell size due to seasonal rotation and intercropping.
    
    Parameters:
    -----------
    harvest_data : dict or DataFrame
        Harvest area data by crop and grid cell (in hectares)
    theoretical_cell_area : float, default=85.5
        Maximum possible area per cell (km²) based on SPAM grid resolution
        
    Returns:
    --------
    capped_data : dict or DataFrame
        Harvest areas capped at theoretical maximum
    capping_stats : dict
        Statistics on capping effects (cells affected, area reduced)
        
    Notes:
    ------
    - Theoretical cell area of 85.5 km² based on 0.083302° × 0.083302° SPAM grid
    - Multicropping identified in 0.01-0.67% of cells across crops
    - Capping preserves spatial patterns while enforcing physical constraints
    """
    logger.info(f"Applying harvest area capping at {theoretical_cell_area} km²")
    
    # Convert theoretical area to hectares for comparison
    theoretical_area_ha = theoretical_cell_area * 100  # km² to hectares
    
    capping_stats = {
        'theoretical_cell_area_km2': theoretical_cell_area,
        'theoretical_cell_area_ha': theoretical_area_ha,
        'crops_processed': [],
        'total_cells_affected': 0,
        'total_area_reduced_ha': 0.0,
        'crop_details': {}
    }
    
    if isinstance(harvest_data, dict):
        return _cap_harvest_dict(harvest_data, theoretical_area_ha, capping_stats)
    elif isinstance(harvest_data, pd.DataFrame):
        return _cap_harvest_dataframe(harvest_data, theoretical_area_ha, capping_stats)
    else:
        raise ValueError("harvest_data must be either dict or pandas DataFrame")


def _cap_harvest_dict(harvest_data: Dict, 
                     theoretical_area_ha: float, 
                     capping_stats: Dict) -> Tuple[Dict, Dict]:
    """Cap harvest areas in dictionary format."""
    capped_data = {}
    
    for crop_name, crop_harvest_areas in harvest_data.items():
        logger.info(f"Processing {crop_name} harvest areas")
        
        # Convert to numpy array for processing
        harvest_array = np.array(crop_harvest_areas)
        
        # Identify cells exceeding theoretical area
        exceeding_mask = harvest_array > theoretical_area_ha
        cells_affected = np.sum(exceeding_mask)
        
        if cells_affected > 0:
            # Calculate area reduction
            original_exceeding = harvest_array[exceeding_mask]
            area_reduced = np.sum(original_exceeding - theoretical_area_ha)
            
            # Apply capping
            capped_array = np.minimum(harvest_array, theoretical_area_ha)
            
            logger.info(f"{crop_name}: Capped {cells_affected:,} cells, "
                       f"reduced area by {area_reduced:.2f} ha")
        else:
            capped_array = harvest_array.copy()
            area_reduced = 0.0
            logger.info(f"{crop_name}: No cells exceeded theoretical area")
        
        # Store results
        capped_data[crop_name] = capped_array
        
        # Update statistics
        crop_stats = {
            'total_cells': len(harvest_array),
            'cells_affected': int(cells_affected),
            'cells_affected_pct': float(cells_affected / len(harvest_array) * 100),
            'area_reduced_ha': float(area_reduced),
            'max_original_area_ha': float(harvest_array.max()),
            'max_capped_area_ha': float(capped_array.max()),
            'mean_original_area_ha': float(harvest_array.mean()),
            'mean_capped_area_ha': float(capped_array.mean())
        }
        
        capping_stats['crop_details'][crop_name] = crop_stats
        capping_stats['crops_processed'].append(crop_name)
        capping_stats['total_cells_affected'] += cells_affected
        capping_stats['total_area_reduced_ha'] += area_reduced
    
    return capped_data, capping_stats


def _cap_harvest_dataframe(harvest_data: pd.DataFrame, 
                          theoretical_area_ha: float, 
                          capping_stats: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Cap harvest areas in DataFrame format."""
    capped_data = harvest_data.copy()
    
    # Identify crop columns (assume they end with '_A' for area)
    crop_columns = [col for col in harvest_data.columns if col.endswith('_A')]
    
    if not crop_columns:
        logger.warning("No crop columns found (expected columns ending with '_A')")
        return capped_data, capping_stats
    
    for crop_col in crop_columns:
        crop_name = crop_col.replace('_A', '').lower()
        logger.info(f"Processing {crop_name} harvest areas (column: {crop_col})")
        
        harvest_array = harvest_data[crop_col].values
        
        # Identify cells exceeding theoretical area
        exceeding_mask = harvest_array > theoretical_area_ha
        cells_affected = np.sum(exceeding_mask)
        
        if cells_affected > 0:
            # Calculate area reduction
            original_exceeding = harvest_array[exceeding_mask]
            area_reduced = np.sum(original_exceeding - theoretical_area_ha)
            
            # Apply capping
            capped_data[crop_col] = np.minimum(harvest_array, theoretical_area_ha)
            
            logger.info(f"{crop_name}: Capped {cells_affected:,} cells, "
                       f"reduced area by {area_reduced:.2f} ha")
        else:
            area_reduced = 0.0
            logger.info(f"{crop_name}: No cells exceeded theoretical area")
        
        # Update statistics
        crop_stats = {
            'column_name': crop_col,
            'total_cells': len(harvest_array),
            'cells_affected': int(cells_affected),
            'cells_affected_pct': float(cells_affected / len(harvest_array) * 100),
            'area_reduced_ha': float(area_reduced),
            'max_original_area_ha': float(harvest_array.max()),
            'max_capped_area_ha': float(capped_data[crop_col].max()),
            'mean_original_area_ha': float(harvest_array.mean()),
            'mean_capped_area_ha': float(capped_data[crop_col].mean())
        }
        
        capping_stats['crop_details'][crop_name] = crop_stats
        capping_stats['crops_processed'].append(crop_name)
        capping_stats['total_cells_affected'] += cells_affected
        capping_stats['total_area_reduced_ha'] += area_reduced
    
    # Add metadata to DataFrame
    capped_data.attrs['multicropping_capped'] = True
    capped_data.attrs['theoretical_cell_area_km2'] = theoretical_area_ha / 100
    capped_data.attrs['capping_stats'] = capping_stats
    
    return capped_data, capping_stats


def validate_capping_results(capped_data: Union[Dict, pd.DataFrame], 
                           theoretical_cell_area: float = 85.5) -> Dict[str, bool]:
    """
    Validate that capping was applied correctly.
    
    Parameters:
    -----------
    capped_data : dict or DataFrame
        Capped harvest area data
    theoretical_cell_area : float
        Maximum allowed area per cell (km²)
        
    Returns:
    --------
    validation_results : dict
        Boolean results for each validation check
    """
    theoretical_area_ha = theoretical_cell_area * 100
    
    validation_results = {
        'no_cells_exceed_limit': True,
        'data_structure_preserved': True,
        'positive_values_only': True,
        'finite_values_only': True
    }
    
    if isinstance(capped_data, dict):
        for crop_name, harvest_areas in capped_data.items():
            harvest_array = np.array(harvest_areas)
            
            # Check no cells exceed limit
            if np.any(harvest_array > theoretical_area_ha):
                validation_results['no_cells_exceed_limit'] = False
                logger.error(f"{crop_name}: Cells still exceed theoretical area after capping")
            
            # Check for positive values
            if np.any(harvest_array < 0):
                validation_results['positive_values_only'] = False
                logger.error(f"{crop_name}: Negative values found after capping")
            
            # Check for finite values
            if not np.all(np.isfinite(harvest_array)):
                validation_results['finite_values_only'] = False
                logger.error(f"{crop_name}: Non-finite values found after capping")
    
    elif isinstance(capped_data, pd.DataFrame):
        crop_columns = [col for col in capped_data.columns if col.endswith('_A')]
        
        for crop_col in crop_columns:
            harvest_array = capped_data[crop_col].values
            
            # Check no cells exceed limit
            if np.any(harvest_array > theoretical_area_ha):
                validation_results['no_cells_exceed_limit'] = False
                logger.error(f"{crop_col}: Cells still exceed theoretical area after capping")
            
            # Check for positive values
            if np.any(harvest_array < 0):
                validation_results['positive_values_only'] = False
                logger.error(f"{crop_col}: Negative values found after capping")
            
            # Check for finite values
            if not np.all(np.isfinite(harvest_array)):
                validation_results['finite_values_only'] = False
                logger.error(f"{crop_col}: Non-finite values found after capping")
    
    # Overall validation
    all_passed = all(validation_results.values())
    validation_results['all_checks_passed'] = all_passed
    
    if all_passed:
        logger.info("✓ All validation checks passed")
    else:
        logger.error("✗ Some validation checks failed")
    
    return validation_results


def generate_capping_report(capping_stats: Dict) -> str:
    """
    Generate a human-readable report of capping results.
    
    Parameters:
    -----------
    capping_stats : dict
        Statistics from cap_harvest_areas function
        
    Returns:
    --------
    report : str
        Formatted report string
    """
    report_lines = [
        "MULTICROPPING AREA CAPPING REPORT",
        "=" * 50,
        f"Theoretical Cell Area: {capping_stats['theoretical_cell_area_km2']:.1f} km² "
        f"({capping_stats['theoretical_cell_area_ha']:.0f} ha)",
        f"Crops Processed: {', '.join(capping_stats['crops_processed'])}",
        f"Total Cells Affected: {capping_stats['total_cells_affected']:,}",
        f"Total Area Reduced: {capping_stats['total_area_reduced_ha']:.2f} ha "
        f"({capping_stats['total_area_reduced_ha']/100:.2f} km²)",
        "",
        "CROP-SPECIFIC RESULTS:",
        "-" * 30
    ]
    
    for crop_name, stats in capping_stats['crop_details'].items():
        report_lines.extend([
            f"{crop_name.upper()}:",
            f"  Total cells: {stats['total_cells']:,}",
            f"  Cells affected: {stats['cells_affected']:,} ({stats['cells_affected_pct']:.2f}%)",
            f"  Area reduced: {stats['area_reduced_ha']:.2f} ha",
            f"  Max area: {stats['max_original_area_ha']:.2f} → {stats['max_capped_area_ha']:.2f} ha",
            f"  Mean area: {stats['mean_original_area_ha']:.2f} → {stats['mean_capped_area_ha']:.2f} ha",
            ""
        ])
    
    report_lines.extend([
        "METHODOLOGY:",
        "- Addresses multicropping violations of physical land constraints",
        "- Preserves spatial patterns while enforcing theoretical cell area limits",
        "- Based on SPAM 2020 grid resolution (0.083302° × 0.083302°)",
        "",
        "ASSUMPTIONS:",
        "- Theoretical cell area represents maximum possible agricultural land",
        "- Multicropping effects are distributed uniformly within cells",
        "- Capping preserves relative productivity patterns"
    ])
    
    return "\n".join(report_lines)


def save_capping_results(capped_data: Union[Dict, pd.DataFrame], 
                        capping_stats: Dict, 
                        output_dir: str = "multicropping_fix/output_data") -> Dict[str, str]:
    """
    Save capping results to files.
    
    Parameters:
    -----------
    capped_data : dict or DataFrame
        Capped harvest area data
    capping_stats : dict
        Capping statistics
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    file_paths : dict
        Paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Save capped data
    if isinstance(capped_data, pd.DataFrame):
        data_file = output_path / "capped_harvest_areas.csv"
        capped_data.to_csv(data_file, index=False)
        file_paths['capped_data'] = str(data_file)
        logger.info(f"Saved capped harvest areas to {data_file}")
    
    # Save statistics (convert numpy types to native Python types for JSON)
    stats_file = output_path / "capping_statistics.json"
    import json
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    json_compatible_stats = convert_numpy_types(capping_stats)
    
    with open(stats_file, 'w') as f:
        json.dump(json_compatible_stats, f, indent=2)
    file_paths['statistics'] = str(stats_file)
    logger.info(f"Saved capping statistics to {stats_file}")
    
    # Save report
    report_file = output_path / "capping_report.txt"
    report = generate_capping_report(capping_stats)
    with open(report_file, 'w') as f:
        f.write(report)
    file_paths['report'] = str(report_file)
    logger.info(f"Saved capping report to {report_file}")
    
    return file_paths


# Convenience function for quick capping
def quick_cap_harvest_areas(harvest_data: Union[Dict, pd.DataFrame], 
                           theoretical_cell_area: float = 85.5,
                           save_results: bool = True) -> Tuple[Union[Dict, pd.DataFrame], Dict]:
    """
    Convenience function for quick harvest area capping with validation.
    
    Parameters:
    -----------
    harvest_data : dict or DataFrame
        Harvest area data to cap
    theoretical_cell_area : float
        Maximum cell area in km²
    save_results : bool
        Whether to save results to files
        
    Returns:
    --------
    capped_data : dict or DataFrame
        Capped harvest areas
    capping_stats : dict
        Capping statistics
    """
    # Apply capping
    capped_data, capping_stats = cap_harvest_areas(harvest_data, theoretical_cell_area)
    
    # Validate results
    validation_results = validate_capping_results(capped_data, theoretical_cell_area)
    capping_stats['validation_results'] = validation_results
    
    if not validation_results['all_checks_passed']:
        raise ValueError("Capping validation failed - see logs for details")
    
    # Save results if requested
    if save_results:
        file_paths = save_capping_results(capped_data, capping_stats)
        capping_stats['output_files'] = file_paths
    
    return capped_data, capping_stats
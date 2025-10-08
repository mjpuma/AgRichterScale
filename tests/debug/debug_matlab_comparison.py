#!/usr/bin/env python3
"""Debug script to compare Python implementation with MATLAB code."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AgriRichter modules
from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.analysis.envelope import HPEnvelopeCalculator
from agririchter.visualization.plots import AgriRichterPlotter, EnvelopePlotter


def debug_envelope_calculation():
    """Debug the H-P envelope calculation by comparing with MATLAB algorithm."""
    logger.info("Debugging H-P envelope calculation...")
    
    # Use wheat for testing
    config = Config(crop_type='wheat')
    
    # Load data
    loader = DataLoader(config)
    try:
        production_df = loader.load_spam_production()
        harvest_df = loader.load_spam_harvest_area()
        logger.info(f"Loaded {len(production_df)} production records")
        logger.info(f"Loaded {len(harvest_df)} harvest records")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return
    
    # Get crop columns (wheat = column 1 in MATLAB, but Python uses 0-based indexing)
    crop_columns = [col for col in production_df.columns if col.endswith('_A')]
    logger.info(f"Found crop columns: {crop_columns}")
    
    # For wheat, we want the first crop column (wheat_A)
    wheat_columns = [col for col in crop_columns if 'whea' in col.lower()]
    logger.info(f"Wheat columns: {wheat_columns}")
    
    if not wheat_columns:
        logger.error("No wheat columns found!")
        return
    
    # Calculate total production and harvest for wheat
    total_production = production_df[wheat_columns].sum(axis=1)
    total_harvest = harvest_df[wheat_columns].sum(axis=1)
    
    logger.info(f"Total production range: {total_production.min():.2e} - {total_production.max():.2e} kcal")
    logger.info(f"Total harvest range: {total_harvest.min():.2f} - {total_harvest.max():.2f} km²")
    
    # Create H-P matrix following MATLAB algorithm
    # HPmatrix = [TotalHarvest(:),TotalProduction(:),TotalProduction(:)./TotalHarvest(:)];
    yields = total_production / total_harvest
    yields = yields.replace([np.inf, -np.inf], np.nan)
    
    # Create matrix: [harvest_area, production, yield]
    hp_matrix = np.column_stack([total_harvest.values, total_production.values, yields.values])
    
    # Remove NaNs and zeros (following MATLAB code)
    valid_mask = (~np.isnan(hp_matrix).any(axis=1)) & (hp_matrix[:, 1] > 0)
    hp_matrix_clean = hp_matrix[valid_mask]
    
    logger.info(f"H-P matrix shape after cleaning: {hp_matrix_clean.shape}")
    logger.info(f"Yield range: {hp_matrix_clean[:, 2].min():.2e} - {hp_matrix_clean[:, 2].max():.2e}")
    
    # Sort by yield (column 2) - MATLAB: sortrows(HPmatrix,3)
    sort_indices = np.argsort(hp_matrix_clean[:, 2])
    hp_matrix_sorted = hp_matrix_clean[sort_indices]
    
    logger.info(f"Sorted yield range: {hp_matrix_sorted[0, 2]:.2e} to {hp_matrix_sorted[-1, 2]:.2e}")
    
    # Calculate cumulative sums - MATLAB algorithm
    # HPmatrix_cumsum_SmallLarge = cumsum(HPmatrix_sorted);
    # HPmatrix_cumsum_LargeSmall = cumsum(flipud(HPmatrix_sorted));
    cumsum_small_large = np.cumsum(hp_matrix_sorted, axis=0)
    cumsum_large_small = np.cumsum(np.flipud(hp_matrix_sorted), axis=0)
    
    logger.info(f"Cumsum small->large harvest range: {cumsum_small_large[0, 0]:.0f} - {cumsum_small_large[-1, 0]:.0f}")
    logger.info(f"Cumsum large->small harvest range: {cumsum_large_small[0, 0]:.0f} - {cumsum_large_small[-1, 0]:.0f}")
    
    # Define disruption areas (from MATLAB for wheat)
    # Disturb_HA = [1 10 50 100 200 400 600 800 1000 4000 6000 8000 10000:2000:2205000];
    disturb_ha = np.concatenate([
        [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000],
        np.arange(10000, 2205000, 2000)
    ])
    
    logger.info(f"Disruption areas: {len(disturb_ha)} points from {disturb_ha[0]} to {disturb_ha[-1]} km²")
    
    # Calculate envelope bounds (MATLAB algorithm)
    n_points = len(disturb_ha)
    lower_bound_harvest = np.zeros(n_points)
    lower_bound_production = np.zeros(n_points)
    upper_bound_harvest = np.zeros(n_points)
    upper_bound_production = np.zeros(n_points)
    
    for i, target_area in enumerate(disturb_ha):
        # Lower bound (least productive first)
        # index_temp = min(find(HPmatrix_cumsum_SmallLarge(:,1)>Disturb_HA(i_disturb)));
        indices_lower = np.where(cumsum_small_large[:, 0] > target_area)[0]
        if len(indices_lower) > 0:
            idx = indices_lower[0]
            lower_bound_harvest[i] = cumsum_small_large[idx, 0]
            lower_bound_production[i] = cumsum_small_large[idx, 1]
        
        # Upper bound (most productive first)
        indices_upper = np.where(cumsum_large_small[:, 0] > target_area)[0]
        if len(indices_upper) > 0:
            idx = indices_upper[0]
            upper_bound_harvest[i] = cumsum_large_small[idx, 0]
            upper_bound_production[i] = cumsum_large_small[idx, 1]
    
    # Create envelope data
    envelope_data = {
        'disruption_areas': disturb_ha,
        'lower_bound_harvest': lower_bound_harvest,
        'lower_bound_production': lower_bound_production,
        'upper_bound_harvest': upper_bound_harvest,
        'upper_bound_production': upper_bound_production,
        'crop_type': 'wheat'
    }
    
    logger.info(f"Envelope calculated with {n_points} points")
    logger.info(f"Lower production range: {lower_bound_production[lower_bound_production > 0].min():.2e} - {lower_bound_production.max():.2e}")
    logger.info(f"Upper production range: {upper_bound_production[upper_bound_production > 0].min():.2e} - {upper_bound_production.max():.2e}")
    
    return envelope_data


def debug_agririchter_scale():
    """Debug the AgriRichter scale calculation."""
    logger.info("Debugging AgriRichter scale calculation...")
    
    config = Config(crop_type='wheat')
    
    # From MATLAB code for wheat:
    # calories_cropprod = calories_cropprod_dict.Wheat; (3.34 kcal/g)
    # gramsperMetricTon = 1000000;
    # threshold = 1*gramsperMetricTon*calories_cropprod;
    
    grams_per_metric_ton = 1000000
    calories_wheat = 3.34  # kcal/g
    threshold = 1 * grams_per_metric_ton * calories_wheat  # 3.34e6 kcal
    
    logger.info(f"Threshold: {threshold:.2e} kcal")
    
    # MATLAB: prod_range = logspace(10, 15.2, 10000);
    prod_range = np.logspace(10, 15.2, 10000)
    
    # MATLAB: X=log10(prod_range/threshold); Y=prod_range;
    X = np.log10(prod_range / threshold)
    Y = prod_range
    
    logger.info(f"X (magnitude) range: {X.min():.2f} - {X.max():.2f}")
    logger.info(f"Y (production) range: {Y.min():.2e} - {Y.max():.2e}")
    
    # Create scale data
    scale_data = {
        'magnitudes': X,
        'production_kcal': Y,
        'threshold': threshold
    }
    
    return scale_data


def plot_debug_results():
    """Plot the debug results."""
    logger.info("Creating debug plots...")
    
    # Calculate envelope and scale
    envelope_data = debug_envelope_calculation()
    scale_data = debug_agririchter_scale()
    
    if envelope_data is None or scale_data is None:
        logger.error("Failed to calculate envelope or scale data")
        return
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: H-P Envelope
    ax1.set_title('H-P Envelope (Debug)', fontsize=14, fontweight='bold')
    
    # Plot envelope area
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    # Convert to log10 for plotting
    log_lower_prod = np.log10(np.maximum(lower_production, 1))
    log_upper_prod = np.log10(np.maximum(upper_production, 1))
    
    # Fill between bounds
    ax1.fill_between(lower_harvest, log_lower_prod, log_upper_prod,
                     alpha=0.3, color='lightblue', label='H-P Envelope')
    
    # Plot boundary lines
    ax1.plot(lower_harvest, log_lower_prod, 'b-', linewidth=2, 
             label='Lower Bound (Least Productive)')
    ax1.plot(upper_harvest, log_upper_prod, 'r-', linewidth=2,
             label='Upper Bound (Most Productive)')
    
    ax1.set_xlabel('Harvest Area (km²)')
    ax1.set_ylabel('Production (log₁₀ kcal)')
    ax1.set_xscale('log')
    ax1.set_xlim(1, 1e7)
    ax1.set_ylim(10, 16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AgriRichter Scale
    ax2.set_title('AgriRichter Scale (Debug)', fontsize=14, fontweight='bold')
    
    magnitudes = scale_data['magnitudes']
    production = scale_data['production_kcal']
    
    ax2.plot(magnitudes, production, 'b-', linewidth=2, label='AgriRichter Scale')
    ax2.set_xlabel('Magnitude (log₁₀ production/threshold)')
    ax2.set_ylabel('Production Loss (kcal)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save debug plot
    output_path = Path('debug_matlab_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Debug plot saved to {output_path}")
    
    plt.show()


def main():
    """Main function to run debug comparison."""
    try:
        plot_debug_results()
        logger.info("Debug comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Debug comparison failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
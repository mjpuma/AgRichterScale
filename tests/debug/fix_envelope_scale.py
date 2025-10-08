#!/usr/bin/env python3
"""
Fix the envelope and scale calculations based on MATLAB code and mathematical formulation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data that matches SPAM structure for testing."""
    logger.info("Creating sample SPAM-like data for testing...")
    
    # Create sample grid (smaller for testing)
    n_cells = 10000
    np.random.seed(42)
    
    # Sample coordinates
    x_coords = np.random.uniform(-180, 180, n_cells)
    y_coords = np.random.uniform(-60, 80, n_cells)
    
    # Sample wheat production (metric tons) - log-normal distribution
    wheat_production_mt = np.random.lognormal(mean=2, sigma=2, size=n_cells)
    
    # Sample wheat harvest area (hectares) - log-normal distribution  
    wheat_harvest_ha = np.random.lognormal(mean=3, sigma=1.5, size=n_cells)
    
    # Convert units following MATLAB code:
    # Production: metric tons -> grams -> kcal
    grams_per_metric_ton = 1000000
    wheat_kcal_per_gram = 3.34  # From MATLAB
    wheat_production_kcal = wheat_production_mt * grams_per_metric_ton * wheat_kcal_per_gram
    
    # Harvest area: hectares -> km²
    wheat_harvest_km2 = wheat_harvest_ha * 0.01
    
    # Create DataFrames
    production_df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'WHEA_A': wheat_production_kcal
    })
    
    harvest_df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'WHEA_A': wheat_harvest_km2
    })
    
    logger.info(f"Created sample data with {n_cells} grid cells")
    logger.info(f"Production range: {wheat_production_kcal.min():.2e} - {wheat_production_kcal.max():.2e} kcal")
    logger.info(f"Harvest area range: {wheat_harvest_km2.min():.3f} - {wheat_harvest_km2.max():.3f} km²")
    
    return production_df, harvest_df


def calculate_hp_envelope_corrected(production_df, harvest_df, crop_columns=None, loss_factor=1.0):
    """
    Calculate H-P envelope following the exact MATLAB algorithm and mathematical formulation.
    
    Based on MATLAB code:
    HPmatrix = [TotalHarvest(:),TotalProduction(:),TotalProduction(:)./TotalHarvest(:)];
    HPmatrix_sorted = sortrows(HPmatrix,3); % sort by grid-cell *yield*
    """
    logger.info("Calculating H-P envelope using corrected MATLAB algorithm...")
    
    if crop_columns is None:
        crop_columns = [col for col in production_df.columns if col.endswith('_A')]
    
    # Calculate totals for selected crops
    total_production = production_df[crop_columns].sum(axis=1)
    total_harvest = harvest_df[crop_columns].sum(axis=1)
    
    # Calculate yield (production per unit area) - this is the key sorting criterion
    yields = total_production / total_harvest
    yields = yields.replace([np.inf, -np.inf], np.nan)
    
    # Create H-P matrix: [harvest_area, production, yield]
    # Following MATLAB: HPmatrix = [TotalHarvest(:),TotalProduction(:),TotalProduction(:)./TotalHarvest(:)];
    hp_matrix = np.column_stack([
        total_harvest.values,
        total_production.values, 
        yields.values
    ])
    
    # Remove NaNs and zeros (following MATLAB code)
    valid_mask = (~np.isnan(hp_matrix).any(axis=1)) & (hp_matrix[:, 1] > 0) & (hp_matrix[:, 0] > 0)
    hp_matrix_clean = hp_matrix[valid_mask]
    
    logger.info(f"H-P matrix shape after cleaning: {hp_matrix_clean.shape}")
    logger.info(f"Yield range: {hp_matrix_clean[:, 2].min():.2e} - {hp_matrix_clean[:, 2].max():.2e} kcal/km²")
    
    # Sort by yield (column 2) - MATLAB: sortrows(HPmatrix,3)
    sort_indices = np.argsort(hp_matrix_clean[:, 2])
    hp_matrix_sorted = hp_matrix_clean[sort_indices]
    
    # Calculate cumulative sums - MATLAB algorithm
    # HPmatrix_cumsum_SmallLarge = cumsum(HPmatrix_sorted); (least productive first)
    # HPmatrix_cumsum_LargeSmall = cumsum(flipud(HPmatrix_sorted)); (most productive first)
    cumsum_small_large = np.cumsum(hp_matrix_sorted, axis=0)  # Lower bound
    cumsum_large_small = np.cumsum(np.flipud(hp_matrix_sorted), axis=0)  # Upper bound
    
    # Define disruption areas (from MATLAB for wheat)
    # Disturb_HA = [1 10 50 100 200 400 600 800 1000 4000 6000 8000 10000:2000:2205000];
    max_harvest = total_harvest.sum()
    disturb_ha = np.concatenate([
        [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000],
        np.arange(10000, min(2205000, max_harvest), 2000)
    ])
    
    logger.info(f"Disruption areas: {len(disturb_ha)} points from {disturb_ha[0]} to {disturb_ha[-1]} km²")
    logger.info(f"Total available harvest area: {max_harvest:.0f} km²")
    
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
            lower_bound_production[i] = cumsum_small_large[idx, 1] * loss_factor
        
        # Upper bound (most productive first)
        indices_upper = np.where(cumsum_large_small[:, 0] > target_area)[0]
        if len(indices_upper) > 0:
            idx = indices_upper[0]
            upper_bound_harvest[i] = cumsum_large_small[idx, 0]
            upper_bound_production[i] = cumsum_large_small[idx, 1] * loss_factor
    
    # Create envelope data
    envelope_data = {
        'disruption_areas': disturb_ha,
        'lower_bound_harvest': lower_bound_harvest,
        'lower_bound_production': lower_bound_production,
        'upper_bound_harvest': upper_bound_harvest,
        'upper_bound_production': upper_bound_production
    }
    
    logger.info(f"Envelope calculated with {n_points} points")
    valid_lower = lower_bound_production[lower_bound_production > 0]
    valid_upper = upper_bound_production[upper_bound_production > 0]
    if len(valid_lower) > 0:
        logger.info(f"Lower production range: {valid_lower.min():.2e} - {valid_lower.max():.2e} kcal")
    if len(valid_upper) > 0:
        logger.info(f"Upper production range: {valid_upper.min():.2e} - {valid_upper.max():.2e} kcal")
    
    return envelope_data


def calculate_agririchter_scale_corrected():
    """
    Calculate AgriRichter scale following MATLAB algorithm and mathematical formulation.
    
    From mathematical formulation:
    M_D = log10(A_H) where A_H is disrupted harvest area
    The scale relates magnitude to production loss through upper/lower bounds
    """
    logger.info("Calculating AgriRichter scale using corrected algorithm...")
    
    # Constants from MATLAB code
    grams_per_metric_ton = 1000000
    wheat_kcal_per_gram = 3.34  # kcal/g for wheat
    threshold = 1 * grams_per_metric_ton * wheat_kcal_per_gram  # 3.34e6 kcal
    
    logger.info(f"Threshold: {threshold:.2e} kcal (1 metric ton of wheat)")
    
    # MATLAB: prod_range = logspace(10, 15.2, 10000);
    prod_range = np.logspace(10, 15.2, 10000)
    
    # MATLAB algorithm:
    # X = log10(prod_range/threshold); (magnitude)
    # Y = prod_range; (production loss in kcal)
    magnitudes = np.log10(prod_range / threshold)
    production_kcal = prod_range
    
    logger.info(f"Magnitude range: {magnitudes.min():.2f} - {magnitudes.max():.2f}")
    logger.info(f"Production range: {production_kcal.min():.2e} - {production_kcal.max():.2e} kcal")
    
    # Create scale data
    scale_data = {
        'magnitudes': magnitudes,
        'production_kcal': production_kcal,
        'threshold': threshold
    }
    
    return scale_data


def create_sample_events():
    """Create sample historical events for testing."""
    logger.info("Creating sample historical events...")
    
    # Sample events with realistic values
    events_data = {
        'event_name': ['Drought_2012', 'Flood_2010', 'Pest_2008', 'Heat_2003', 'Storm_2015'],
        'harvest_area_km2': [50000, 25000, 75000, 40000, 30000],
        'production_loss_kcal': [1e12, 5e11, 1.5e12, 8e11, 6e11],
        'magnitude': [4.7, 4.4, 4.9, 4.6, 4.5]
    }
    
    events_df = pd.DataFrame(events_data)
    logger.info(f"Created {len(events_df)} sample events")
    
    return events_df


def plot_corrected_results():
    """Plot the corrected envelope and scale results."""
    logger.info("Creating corrected plots...")
    
    # Create sample data
    production_df, harvest_df = create_sample_data()
    
    # Calculate corrected envelope and scale
    envelope_data = calculate_hp_envelope_corrected(production_df, harvest_df, ['WHEA_A'])
    scale_data = calculate_agririchter_scale_corrected()
    events_df = create_sample_events()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: H-P Envelope (Corrected)
    ax1.set_title('H-P Envelope (Corrected Algorithm)', fontsize=14, fontweight='bold')
    
    # Extract envelope data
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    # Convert to log10 for plotting (following MATLAB)
    log_lower_prod = np.log10(np.maximum(lower_production, 1))
    log_upper_prod = np.log10(np.maximum(upper_production, 1))
    
    # Fill envelope area
    valid_mask = (lower_harvest > 0) & (upper_harvest > 0)
    if np.any(valid_mask):
        ax1.fill_between(lower_harvest[valid_mask], log_lower_prod[valid_mask], log_upper_prod[valid_mask],
                         alpha=0.4, color='lightblue', label='H-P Envelope')
        
        # Plot boundary lines
        ax1.plot(lower_harvest[valid_mask], log_lower_prod[valid_mask], 'b-', linewidth=2, 
                 label='Lower Bound (Least Productive)')
        ax1.plot(upper_harvest[valid_mask], log_upper_prod[valid_mask], 'r-', linewidth=2,
                 label='Upper Bound (Most Productive)')
    
    # Plot sample events
    for _, event in events_df.iterrows():
        log_loss = np.log10(event['production_loss_kcal'])
        ax1.scatter(event['harvest_area_km2'], log_loss, 
                   s=80, alpha=0.8, edgecolors='black', linewidths=1)
        ax1.annotate(event['event_name'], 
                    (event['harvest_area_km2'], log_loss),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Disrupted Harvest Area (km²)', fontsize=12)
    ax1.set_ylabel('Production Loss (log₁₀ kcal)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_xlim(1, 1e7)
    ax1.set_ylim(9, 16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AgriRichter Scale (Corrected)
    ax2.set_title('AgriRichter Scale (Corrected Algorithm)', fontsize=14, fontweight='bold')
    
    magnitudes = scale_data['magnitudes']
    production = scale_data['production_kcal']
    
    # Plot scale line
    ax2.plot(magnitudes, production, 'b-', linewidth=2, label='AgriRichter Scale')
    
    # Plot sample events on scale
    for _, event in events_df.iterrows():
        # Calculate magnitude for event
        event_magnitude = np.log10(event['production_loss_kcal'] / scale_data['threshold'])
        ax2.scatter(event_magnitude, event['production_loss_kcal'],
                   s=80, alpha=0.8, edgecolors='black', linewidths=1, color='red')
        ax2.annotate(event['event_name'],
                    (event_magnitude, event['production_loss_kcal']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Magnitude (log₁₀ production loss / threshold)', fontsize=12)
    ax2.set_ylabel('Production Loss (kcal)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save corrected plot
    output_path = Path('corrected_envelope_scale.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Corrected plot saved to {output_path}")
    
    plt.show()
    
    return envelope_data, scale_data


def main():
    """Main function to test corrected algorithms."""
    try:
        envelope_data, scale_data = plot_corrected_results()
        
        print("\n" + "="*60)
        print("CORRECTED ALGORITHM TEST RESULTS")
        print("="*60)
        print(f"✓ H-P Envelope calculated with {len(envelope_data['disruption_areas'])} points")
        print(f"✓ AgriRichter Scale calculated with {len(scale_data['magnitudes'])} points")
        print(f"✓ Threshold: {scale_data['threshold']:.2e} kcal")
        
        # Validate results
        valid_envelope = np.sum(envelope_data['lower_bound_production'] > 0)
        print(f"✓ Valid envelope points: {valid_envelope}")
        
        print("\n✓ Corrected algorithms implemented successfully!")
        
    except Exception as e:
        logger.error(f"Corrected algorithm test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
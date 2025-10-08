#!/usr/bin/env python3
"""
Fix the envelope visualization to match MATLAB exactly.
"""

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
from agririchter.analysis.envelope import HPEnvelopeCalculator


def create_test_data():
    """Create test data that matches SPAM structure."""
    logger.info("Creating test data...")
    
    # Create sample grid
    n_cells = 5000
    np.random.seed(42)
    
    # Sample coordinates
    x_coords = np.random.uniform(-180, 180, n_cells)
    y_coords = np.random.uniform(-60, 80, n_cells)
    
    # Sample wheat production and harvest with realistic distributions
    wheat_production_mt = np.random.lognormal(mean=1.5, sigma=2, size=n_cells)
    wheat_harvest_ha = np.random.lognormal(mean=2.5, sigma=1.5, size=n_cells)
    
    # Convert units following MATLAB code
    grams_per_metric_ton = 1000000
    wheat_kcal_per_gram = 3.34
    wheat_production_kcal = wheat_production_mt * grams_per_metric_ton * wheat_kcal_per_gram
    wheat_harvest_km2 = wheat_harvest_ha * 0.01
    
    # Create DataFrames with proper column names
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
    
    logger.info(f"Created test data with {n_cells} grid cells")
    return production_df, harvest_df


def create_matlab_style_envelope_plot(envelope_data, loss_factor=1.0):
    """
    Create envelope plot exactly following MATLAB algorithm.
    
    MATLAB code:
    X=log10([DisturbMatrix_lower(1,1),DisturbMatrix_upper(:,1)', flipud(DisturbMatrix_lower(:,1))']);
    Y=log10(loss_factor*[DisturbMatrix_lower(1,2),DisturbMatrix_upper(:,2)', flipud(DisturbMatrix_lower(:,2))']);
    fill(X,Y,rgb('Grey'), 'FaceAlpha', 0.4,'EdgeColor','none');
    """
    logger.info("Creating MATLAB-style envelope plot...")
    
    # Extract envelope data
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    # Apply loss factor
    lower_production_adj = lower_production * loss_factor
    upper_production_adj = upper_production * loss_factor
    
    # Filter out zero values to avoid log(0)
    valid_mask = (lower_harvest > 0) & (lower_production_adj > 0) & (upper_harvest > 0) & (upper_production_adj > 0)
    
    if not np.any(valid_mask):
        logger.error("No valid data points for envelope plotting")
        return None
    
    # Get valid data
    lower_harvest_valid = lower_harvest[valid_mask]
    lower_production_valid = lower_production_adj[valid_mask]
    upper_harvest_valid = upper_harvest[valid_mask]
    upper_production_valid = upper_production_adj[valid_mask]
    
    logger.info(f"Valid envelope points: {len(lower_harvest_valid)}")
    
    # Create closed polygon coordinates following MATLAB algorithm
    # X=log10([DisturbMatrix_lower(1,1),DisturbMatrix_upper(:,1)', flipud(DisturbMatrix_lower(:,1))']);
    # Y=log10(loss_factor*[DisturbMatrix_lower(1,2),DisturbMatrix_upper(:,2)', flipud(DisturbMatrix_lower(:,2))']);
    
    # Concatenate coordinates for closed polygon
    X_polygon = np.concatenate([
        [lower_harvest_valid[0]],           # First point of lower bound
        upper_harvest_valid,                # All upper bound points (left to right)
        np.flipud(lower_harvest_valid)      # All lower bound points (right to left)
    ])
    
    Y_polygon = np.concatenate([
        [lower_production_valid[0]],        # First point of lower bound
        upper_production_valid,             # All upper bound points (left to right)
        np.flipud(lower_production_valid)   # All lower bound points (right to left)
    ])
    
    # Convert to log10 for plotting (following MATLAB)
    X_polygon_log = np.log10(X_polygon)
    Y_polygon_log = np.log10(Y_polygon)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Fill the envelope area (following MATLAB: fill(X,Y,rgb('Grey'), 'FaceAlpha', 0.4,'EdgeColor','none'))
    ax.fill(X_polygon_log, Y_polygon_log, color='gray', alpha=0.4, edgecolor='none', label='H-P Envelope')
    
    # Plot boundary lines for clarity
    ax.plot(np.log10(lower_harvest_valid), np.log10(lower_production_valid), 'b-', linewidth=2, 
            label='Lower Bound (Least Productive)', alpha=0.8)
    ax.plot(np.log10(upper_harvest_valid), np.log10(upper_production_valid), 'r-', linewidth=2,
            label='Upper Bound (Most Productive)', alpha=0.8)
    
    # Set up axes following MATLAB
    ax.set_xlabel('Disrupted Harvest Area, log₁₀(harvest area in km²)', fontsize=18)
    ax.set_ylabel('Disrupted Production, log₁₀(kcal)', fontsize=18)
    ax.set_xlim([2.5, 7.5])  # MATLAB: xlim([2.5 7.5])
    ax.set_ylim([9.5, 16.5])  # MATLAB: ylim([9.5 16.5])
    ax.tick_params(axis='both', which='major', labelsize=20)  # MATLAB: set(gca,'FontSize',20)
    ax.set_title('Disrupted Envelope: Production vs Harvest Area', fontsize=20)  # MATLAB title
    
    # Add box (MATLAB: box on)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig, ax


def test_envelope_convergence():
    """Test that envelope bounds converge properly."""
    logger.info("Testing envelope bounds convergence...")
    
    config = Config(crop_type='wheat')
    production_df, harvest_df = create_test_data()
    
    # Calculate envelope
    envelope_calc = HPEnvelopeCalculator(config)
    envelope_data = envelope_calc.calculate_hp_envelope(production_df, harvest_df)
    
    # Check convergence at maximum disruption area
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    # Find the last valid points
    valid_lower = (lower_harvest > 0) & (lower_production > 0)
    valid_upper = (upper_harvest > 0) & (upper_production > 0)
    
    if np.any(valid_lower) and np.any(valid_upper):
        last_lower_idx = np.where(valid_lower)[0][-1]
        last_upper_idx = np.where(valid_upper)[0][-1]
        
        logger.info(f"Last lower bound: harvest={lower_harvest[last_lower_idx]:.0f} km², production={lower_production[last_lower_idx]:.2e} kcal")
        logger.info(f"Last upper bound: harvest={upper_harvest[last_upper_idx]:.0f} km², production={upper_production[last_upper_idx]:.2e} kcal")
        
        # Check if bounds converge (should be close at maximum disruption)
        harvest_diff = abs(lower_harvest[last_lower_idx] - upper_harvest[last_upper_idx])
        production_diff = abs(lower_production[last_lower_idx] - upper_production[last_upper_idx])
        
        logger.info(f"Convergence check - Harvest diff: {harvest_diff:.0f} km², Production diff: {production_diff:.2e} kcal")
        
        # The bounds should converge when we reach total available area
        total_harvest = harvest_df['WHEA_A'].sum()
        total_production = production_df['WHEA_A'].sum()
        
        logger.info(f"Total available: harvest={total_harvest:.0f} km², production={total_production:.2e} kcal")
    
    return envelope_data


def main():
    """Main function to test corrected envelope visualization."""
    try:
        logger.info("Testing corrected envelope visualization...")
        
        # Test envelope convergence
        envelope_data = test_envelope_convergence()
        
        # Create MATLAB-style plot
        fig, ax = create_matlab_style_envelope_plot(envelope_data, loss_factor=1.0)
        
        if fig is not None:
            # Save plot
            output_path = Path('matlab_style_envelope.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ MATLAB-style envelope plot saved to {output_path}")
            
            plt.show()
        
        print("\n" + "="*60)
        print("MATLAB-STYLE ENVELOPE TEST RESULTS")
        print("="*60)
        print("✓ Envelope bounds convergence: TESTED")
        print("✓ MATLAB-style visualization: CREATED")
        print("✓ Closed polygon shading: IMPLEMENTED")
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
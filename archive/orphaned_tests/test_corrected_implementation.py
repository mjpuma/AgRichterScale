#!/usr/bin/env python3
"""Test the corrected envelope and scale implementation."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AgRichter modules
from agrichter.core.config import Config
from agrichter.analysis.envelope import HPEnvelopeCalculator
from agrichter.analysis.agrichter import AgRichterAnalyzer
from agrichter.visualization.plots import AgRichterPlotter, EnvelopePlotter


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


def test_envelope_calculation():
    """Test the corrected H-P envelope calculation."""
    logger.info("Testing H-P envelope calculation...")
    
    config = Config(crop_type='wheat')
    production_df, harvest_df = create_test_data()
    
    # Calculate envelope
    envelope_calc = HPEnvelopeCalculator(config)
    envelope_data = envelope_calc.calculate_hp_envelope(production_df, harvest_df)
    
    # Validate results
    assert 'disrupted_areas' in envelope_data
    assert 'lower_bound_harvest' in envelope_data
    assert 'lower_bound_production' in envelope_data
    assert 'upper_bound_harvest' in envelope_data
    assert 'upper_bound_production' in envelope_data
    
    logger.info(f"✓ Envelope calculated with {len(envelope_data['disrupted_areas'])} points")
    
    # Check that upper bounds >= lower bounds
    valid_mask = (envelope_data['lower_bound_production'] > 0) & (envelope_data['upper_bound_production'] > 0)
    if np.any(valid_mask):
        lower_valid = envelope_data['lower_bound_production'][valid_mask]
        upper_valid = envelope_data['upper_bound_production'][valid_mask]
        assert np.all(upper_valid >= lower_valid), "Upper bounds should be >= lower bounds"
        logger.info("✓ Upper bounds >= lower bounds validation passed")
    
    return envelope_data


def test_scale_calculation():
    """Test the corrected AgRichter scale calculation."""
    logger.info("Testing AgRichter scale calculation...")
    
    config = Config(crop_type='wheat')
    analyzer = AgRichterAnalyzer(config)
    
    # Calculate scale data
    scale_data = analyzer.create_richter_scale_data(
        min_magnitude=0.0,
        max_magnitude=8.0,
        n_points=10000
    )
    
    # Validate results
    assert 'magnitudes' in scale_data
    assert 'production_kcal' in scale_data
    assert 'threshold_kcal' in scale_data
    
    logger.info(f"✓ Scale calculated with {len(scale_data['magnitudes'])} points")
    logger.info(f"✓ Magnitude range: {scale_data['magnitudes'].min():.2f} - {scale_data['magnitudes'].max():.2f}")
    logger.info(f"✓ Threshold: {scale_data['threshold_kcal']:.2e} kcal")
    
    # Check that production increases with magnitude
    magnitudes = scale_data['magnitudes']
    production = scale_data['production_kcal']
    assert np.all(np.diff(magnitudes) >= 0), "Magnitudes should be non-decreasing"
    assert np.all(np.diff(production) >= 0), "Production should be non-decreasing"
    logger.info("✓ Scale monotonicity validation passed")
    
    return scale_data


def test_visualization():
    """Test the visualization with corrected data."""
    logger.info("Testing visualization...")
    
    config = Config(crop_type='wheat')
    production_df, harvest_df = create_test_data()
    
    # Calculate data
    envelope_data = test_envelope_calculation()
    scale_data = test_scale_calculation()
    
    # Create sample events
    events_df = pd.DataFrame({
        'event_name': ['Test_Event_1', 'Test_Event_2'],
        'harvest_area_km2': [10000, 50000],
        'production_loss_kcal': [1e12, 5e12],
        'magnitude': [4.0, 4.7]
    })
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: H-P Envelope
    envelope_plotter = EnvelopePlotter(config)
    
    # Extract and plot envelope data
    lower_harvest = envelope_data['lower_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_harvest = envelope_data['upper_bound_harvest']
    upper_production = envelope_data['upper_bound_production']
    
    # Convert to log10 for plotting
    log_lower_prod = np.log10(np.maximum(lower_production, 1))
    log_upper_prod = np.log10(np.maximum(upper_production, 1))
    
    # Plot envelope
    valid_mask = (lower_harvest > 0) & (upper_harvest > 0)
    if np.any(valid_mask):
        ax1.fill_between(lower_harvest[valid_mask], log_lower_prod[valid_mask], log_upper_prod[valid_mask],
                         alpha=0.4, color='lightblue', label='H-P Envelope')
        ax1.plot(lower_harvest[valid_mask], log_lower_prod[valid_mask], 'b-', linewidth=2, 
                 label='Lower Bound')
        ax1.plot(upper_harvest[valid_mask], log_upper_prod[valid_mask], 'r-', linewidth=2,
                 label='Upper Bound')
    
    # Plot events
    for _, event in events_df.iterrows():
        log_loss = np.log10(event['production_loss_kcal'])
        ax1.scatter(event['harvest_area_km2'], log_loss, s=80, alpha=0.8)
        ax1.annotate(event['event_name'], (event['harvest_area_km2'], log_loss),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_title('H-P Envelope (Corrected)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Harvest Area (km²)')
    ax1.set_ylabel('Production (log₁₀ kcal)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AgRichter Scale
    scale_plotter = AgRichterPlotter(config)
    
    magnitudes = scale_data['magnitudes']
    production = scale_data['production_kcal']
    
    ax2.plot(magnitudes, production, 'b-', linewidth=2, label='AgRichter Scale')
    
    # Plot events on scale
    for _, event in events_df.iterrows():
        event_magnitude = np.log10(event['production_loss_kcal'] / scale_data['threshold_kcal'])
        ax2.scatter(event_magnitude, event['production_loss_kcal'], s=80, alpha=0.8, color='red')
        ax2.annotate(event['event_name'], (event_magnitude, event['production_loss_kcal']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_title('AgRichter Scale (Corrected)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('Production Loss (kcal)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path('test_corrected_implementation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Test plot saved to {output_path}")
    
    plt.show()


def main():
    """Main test function."""
    try:
        logger.info("Starting corrected implementation tests...")
        
        # Test individual components
        envelope_data = test_envelope_calculation()
        scale_data = test_scale_calculation()
        
        # Test visualization
        test_visualization()
        
        print("\n" + "="*60)
        print("CORRECTED IMPLEMENTATION TEST RESULTS")
        print("="*60)
        print("✓ H-P Envelope calculation: PASSED")
        print("✓ AgRichter Scale calculation: PASSED")
        print("✓ Visualization: PASSED")
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
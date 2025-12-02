"""
Demo script comparing V1 (original) and V2 (robust builder) envelope calculators.

This script demonstrates:
1. How to use both calculators with the same API
2. Visual comparison of results
3. QA metrics comparison
4. Performance comparison

Usage:
    python examples/demo_envelope_v1_v2_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agririchter.core.config import Config
from agririchter.analysis.envelope import HPEnvelopeCalculator
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2
from agririchter.data.loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(config, sample_size=1000):
    """Load sample SPAM data."""
    logger.info(f"Loading sample data for {config.crop_type}...")
    
    loader = DataLoader(config)
    production_df = loader.load_spam_production()
    harvest_df = loader.load_spam_harvest()
    
    # Take sample for faster demo
    production_sample = production_df.head(sample_size)
    harvest_sample = harvest_df.head(sample_size)
    
    logger.info(f"Loaded {len(production_sample)} grid cells")
    return production_sample, harvest_sample


def compare_calculators(config, production_df, harvest_df):
    """Compare V1 and V2 calculators."""
    logger.info("\n" + "=" * 70)
    logger.info("ENVELOPE CALCULATOR COMPARISON")
    logger.info("=" * 70)
    
    # Initialize calculators
    v1 = HPEnvelopeCalculator(config)
    v2 = HPEnvelopeCalculatorV2(config)
    
    # Calculate envelopes with timing
    logger.info("\nCalculating V1 envelope...")
    start = time.time()
    envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
    v1_time = time.time() - start
    logger.info(f"V1 completed in {v1_time:.3f}s")
    
    logger.info("\nCalculating V2 envelope...")
    start = time.time()
    envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
    v2_time = time.time() - start
    logger.info(f"V2 completed in {v2_time:.3f}s")
    
    # Performance comparison
    logger.info("\n" + "-" * 70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("-" * 70)
    logger.info(f"V1 time: {v1_time:.3f}s")
    logger.info(f"V2 time: {v2_time:.3f}s")
    logger.info(f"Ratio (V2/V1): {v2_time/v1_time:.2f}x")
    
    # Output size comparison
    logger.info("\n" + "-" * 70)
    logger.info("OUTPUT SIZE COMPARISON")
    logger.info("-" * 70)
    logger.info(f"V1 data points: {len(envelope_v1['disruption_areas'])}")
    logger.info(f"V2 data points: {len(envelope_v2['disruption_areas'])}")
    
    # Statistics comparison
    logger.info("\n" + "-" * 70)
    logger.info("STATISTICS COMPARISON")
    logger.info("-" * 70)
    
    stats_v1 = v1.get_envelope_statistics(envelope_v1)
    stats_v2 = v2.get_envelope_statistics(envelope_v2)
    
    logger.info(f"\nV1 Production Range:")
    logger.info(f"  Lower: {stats_v1['lower_bound_stats']['min_production']:.2e} - "
               f"{stats_v1['lower_bound_stats']['max_production']:.2e} kcal")
    logger.info(f"  Upper: {stats_v1['upper_bound_stats']['min_production']:.2e} - "
               f"{stats_v1['upper_bound_stats']['max_production']:.2e} kcal")
    
    logger.info(f"\nV2 Production Range:")
    logger.info(f"  Lower: {stats_v2['lower_bound_stats']['min_production']:.2e} - "
               f"{stats_v2['lower_bound_stats']['max_production']:.2e} kcal")
    logger.info(f"  Upper: {stats_v2['upper_bound_stats']['min_production']:.2e} - "
               f"{stats_v2['upper_bound_stats']['max_production']:.2e} kcal")
    
    # Numerical similarity
    v1_max_prod = stats_v1['upper_bound_stats']['max_production']
    v2_max_prod = stats_v2['upper_bound_stats']['max_production']
    rel_diff = abs(v1_max_prod - v2_max_prod) / max(v1_max_prod, v2_max_prod)
    
    logger.info(f"\nNumerical Similarity:")
    logger.info(f"  Max production relative difference: {rel_diff:.2%}")
    
    # Quality checks
    logger.info("\n" + "-" * 70)
    logger.info("QUALITY CHECKS")
    logger.info("-" * 70)
    
    # Check dominance
    v1_violations = np.sum(
        envelope_v1['upper_bound_production'] < envelope_v1['lower_bound_production']
    )
    v2_violations = np.sum(
        envelope_v2['upper_bound_production'] < envelope_v2['lower_bound_production']
    )
    
    logger.info(f"\nDominance violations (upper < lower):")
    logger.info(f"  V1: {v1_violations} violations")
    logger.info(f"  V2: {v2_violations} violations")
    
    if v2_violations == 0:
        logger.info("  ✓ V2 enforces dominance constraint")
    
    # Check monotonicity
    v1_lower_mono = np.all(np.diff(envelope_v1['lower_bound_production']) >= 0)
    v1_upper_mono = np.all(np.diff(envelope_v1['upper_bound_production']) >= 0)
    v2_lower_mono = np.all(np.diff(envelope_v2['lower_bound_production']) >= 0)
    v2_upper_mono = np.all(np.diff(envelope_v2['upper_bound_production']) >= 0)
    
    logger.info(f"\nMonotonicity:")
    logger.info(f"  V1 lower: {'✓' if v1_lower_mono else '✗'}, "
               f"upper: {'✓' if v1_upper_mono else '✗'}")
    logger.info(f"  V2 lower: {'✓' if v2_lower_mono else '✗'}, "
               f"upper: {'✓' if v2_upper_mono else '✗'}")
    
    # V2 QA summary
    if 'v2_summary' in envelope_v2:
        logger.info("\n" + "-" * 70)
        logger.info("V2 QA SUMMARY")
        logger.info("-" * 70)
        summary = envelope_v2['v2_summary']
        logger.info(f"  QA Status: {summary.get('qa_status', 'UNKNOWN')}")
        logger.info(f"  Valid Cells: {summary.get('n_valid_cells', 'N/A')}")
        logger.info(f"  Total Production: {summary.get('total_P_mt', 'N/A'):.2e}")
        logger.info(f"  Total Harvest: {summary.get('total_H_km2', 'N/A'):.2f} km²")
    
    # Overall recommendation
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATION")
    logger.info("=" * 70)
    
    if v2_violations == 0 and v2_lower_mono and v2_upper_mono and rel_diff < 0.10:
        logger.info("✓ V2 passes all checks and is numerically similar to V1")
        logger.info("✓ V2 provides better mathematical guarantees")
        logger.info("✓ V2 is READY for production use")
    else:
        logger.info("⚠ V2 needs further investigation before production use")
    
    logger.info("=" * 70 + "\n")
    
    return envelope_v1, envelope_v2, stats_v1, stats_v2


def plot_comparison(envelope_v1, envelope_v2, config, output_path='envelope_comparison.png'):
    """Create comparison plot."""
    logger.info(f"Creating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Envelope Calculator Comparison - {config.crop_type.upper()}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: V1 envelope
    ax1 = axes[0, 0]
    ax1.fill_between(
        envelope_v1['lower_bound_harvest'],
        envelope_v1['lower_bound_production'],
        envelope_v1['upper_bound_production'],
        alpha=0.3, color='blue', label='Envelope'
    )
    ax1.plot(envelope_v1['lower_bound_harvest'], envelope_v1['lower_bound_production'],
            'b-', linewidth=1.5, label='Lower bound')
    ax1.plot(envelope_v1['upper_bound_harvest'], envelope_v1['upper_bound_production'],
            'b--', linewidth=1.5, label='Upper bound')
    ax1.set_xlabel('Cumulative Harvest Area (km²)')
    ax1.set_ylabel('Cumulative Production (kcal)')
    ax1.set_title('V1 (Original Calculator)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: V2 envelope
    ax2 = axes[0, 1]
    ax2.fill_between(
        envelope_v2['lower_bound_harvest'],
        envelope_v2['lower_bound_production'],
        envelope_v2['upper_bound_production'],
        alpha=0.3, color='green', label='Envelope'
    )
    ax2.plot(envelope_v2['lower_bound_harvest'], envelope_v2['lower_bound_production'],
            'g-', linewidth=1.5, label='Lower bound')
    ax2.plot(envelope_v2['upper_bound_harvest'], envelope_v2['upper_bound_production'],
            'g--', linewidth=1.5, label='Upper bound')
    ax2.set_xlabel('Cumulative Harvest Area (km²)')
    ax2.set_ylabel('Cumulative Production (kcal)')
    ax2.set_title('V2 (Robust Builder)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overlay comparison
    ax3 = axes[1, 0]
    ax3.plot(envelope_v1['lower_bound_harvest'], envelope_v1['lower_bound_production'],
            'b-', linewidth=1.5, alpha=0.7, label='V1 Lower')
    ax3.plot(envelope_v1['upper_bound_harvest'], envelope_v1['upper_bound_production'],
            'b--', linewidth=1.5, alpha=0.7, label='V1 Upper')
    ax3.plot(envelope_v2['lower_bound_harvest'], envelope_v2['lower_bound_production'],
            'g-', linewidth=1.5, alpha=0.7, label='V2 Lower')
    ax3.plot(envelope_v2['upper_bound_harvest'], envelope_v2['upper_bound_production'],
            'g--', linewidth=1.5, alpha=0.7, label='V2 Upper')
    ax3.set_xlabel('Cumulative Harvest Area (km²)')
    ax3.set_ylabel('Cumulative Production (kcal)')
    ax3.set_title('Overlay Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Difference plot
    ax4 = axes[1, 1]
    
    # Interpolate V1 onto V2 grid for comparison
    v2_harvest = envelope_v2['lower_bound_harvest']
    v1_lower_interp = np.interp(v2_harvest, envelope_v1['lower_bound_harvest'], 
                                 envelope_v1['lower_bound_production'])
    v1_upper_interp = np.interp(v2_harvest, envelope_v1['upper_bound_harvest'], 
                                 envelope_v1['upper_bound_production'])
    
    lower_diff = envelope_v2['lower_bound_production'] - v1_lower_interp
    upper_diff = envelope_v2['upper_bound_production'] - v1_upper_interp
    
    ax4.plot(v2_harvest, lower_diff, 'b-', linewidth=1.5, label='Lower bound diff')
    ax4.plot(v2_harvest, upper_diff, 'r-', linewidth=1.5, label='Upper bound diff')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Cumulative Harvest Area (km²)')
    ax4.set_ylabel('Production Difference (V2 - V1) (kcal)')
    ax4.set_title('Difference (V2 - V1)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_path}")
    
    return fig


def main():
    """Main demo function."""
    logger.info("Starting Envelope Calculator V1 vs V2 Comparison Demo")
    
    # Configure for wheat analysis
    config = Config(crop_type='wheat', spam_version='2020')
    
    try:
        # Load sample data
        production_df, harvest_df = load_sample_data(config, sample_size=1000)
        
        # Compare calculators
        envelope_v1, envelope_v2, stats_v1, stats_v2 = compare_calculators(
            config, production_df, harvest_df
        )
        
        # Create comparison plot
        plot_comparison(envelope_v1, envelope_v2, config)
        
        # Print reports
        logger.info("\n" + "=" * 70)
        logger.info("V1 REPORT")
        logger.info("=" * 70)
        print(HPEnvelopeCalculator(config).create_envelope_report(envelope_v1))
        
        logger.info("\n" + "=" * 70)
        logger.info("V2 REPORT")
        logger.info("=" * 70)
        print(HPEnvelopeCalculatorV2(config).create_envelope_report(envelope_v2))
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

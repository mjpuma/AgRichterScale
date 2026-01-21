#!/usr/bin/env python3
"""
Figure 4: Global Systemic Risk (The Fragility Gap)

Calculates the annual exceedance probability of global agricultural disruptions
based on historical production variability (USDA PSD 1961-2021) coupled with 
the AgRichter H-P Envelope.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to allow importing from agririchter
sys.path.append(str(Path(__file__).parent.parent))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.usda import USDADataLoader
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_shocks():
    """Calculate historical production shocks from USDA data."""
    loader = USDADataLoader()
    crops = ['wheat', 'rice', 'maize']
    
    # Load and aggregate production
    total_prod_df = None
    for crop in crops:
        df = loader.load_crop_data(crop)
        # Select Year and Production
        crop_prod = df[['Year', 'Production']].copy()
        crop_prod = crop_prod.rename(columns={'Production': f'Prod_{crop}'})
        
        if total_prod_df is None:
            total_prod_df = crop_prod
        else:
            total_prod_df = pd.merge(total_prod_df, crop_prod, on='Year', how='inner')
            
    # Sum the production columns
    prod_cols = [f'Prod_{crop}' for crop in crops]
    total_prod_df['Production'] = total_prod_df[prod_cols].sum(axis=1)
    
    # Sort by year
    total_prod_df = total_prod_df.sort_values('Year')
    
    # Calculate detrended shocks (as fraction of production)
    # Use a simple linear trend for detrending global production
    z = np.polyfit(total_prod_df['Year'], total_prod_df['Production'], 1)
    p = np.poly1d(z)
    total_prod_df['Trend'] = p(total_prod_df['Year'])
    total_prod_df['Shock_Val'] = total_prod_df['Production'] - total_prod_df['Trend']
    total_prod_df['Shock_Frac'] = total_prod_df['Shock_Val'] / total_prod_df['Trend']
    
    # Calculate basic stats
    shock_std = total_prod_df['Shock_Frac'].std()
    
    return shock_std, total_prod_df['Trend'].iloc[-1]

def generate_risk_figure():
    """Generate Figure 4: Systemic Risk Curve."""
    logger.info("Generating Figure 4: Predictive Systemic Risk...")
    
    # 1. Get Historical Variability
    shock_std, current_trend_prod_mt = calculate_shocks()
    
    # Convert trend production to kcal
    config = Config(crop_type='allgrain', root_dir='.')
    caloric_content = config.get_caloric_content() # kcal/g
    kcal_per_mt = 1_000_000.0 * caloric_content
    current_prod_kcal = current_trend_prod_mt * kcal_per_mt
    
    logger.info(f"Global variability (std dev): {shock_std:.4f}")
    logger.info(f"Current trend production: {current_prod_kcal:.2e} kcal")
    
    # 2. Map variability to AgRichter using Global Envelope
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)
    global_envelope = calculator.calculate_hp_envelope(prod_df, harv_df)
    
    H_km2 = global_envelope['upper_bound_harvest']
    P_up = global_envelope['upper_bound_production']
    M_scale = np.log10(H_km2)
    
    # 3. Model Exceedance Probability
    # Use a Fat-Tailed (Exponential) tail model for predictive risk
    # P(Loss > x) = exp(-x / scale)
    # where scale is related to historical volatility
    
    # We calibrate the scale to the 1-standard-deviation point
    # P(Loss > sigma) approx 0.32 for Normal, but we use Exponential for the tail
    # exp(-sigma / scale) = 0.32  =>  scale = -sigma / ln(0.32)
    # sigma = shock_std * current_prod_kcal
    sigma_kcal = shock_std * current_prod_kcal
    tail_scale = sigma_kcal / (-np.log(0.32))
    
    # Range of magnitudes to plot
    mags_plot = np.linspace(3, 8, 100)
    # Map Magnitude back to production loss using Envelope Upper Bound
    losses_at_mags = np.interp(mags_plot, M_scale, P_up)
    
    # Calculate exceedance probability for each magnitude
    probs = np.exp(-losses_at_mags / tail_scale)
    probs = np.clip(probs, 1e-6, 1) # Cap at 1 and floor for plotting
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Background: Gray fill for historical observation range (up to ~Mag 6)
    ax.axvspan(3, 6, color='gray', alpha=0.05, label='Historical Observation Range')
    
    # The Risk Curve
    ax.plot(mags_plot, probs, color='firebrick', linewidth=4, label='Systemic Risk Curve (USDA Tail Model)')
    
    # Fill "Zone of Insecurity"
    ax.fill_between(mags_plot, probs, 1e-6, color='firebrick', alpha=0.1)
    
    # Buffer Thresholds
    thresholds = config.get_thresholds()
    # Ensure they are sorted for better annotation
    sorted_thresh = sorted(thresholds.items(), key=lambda x: x[1])
    
    colors = {'1 Month': '#FFD700', '3 Months': '#FF4500', 'Total Stocks': '#800080'}
    
    for label, val in sorted_thresh:
        color = colors.get(label, 'black')
        # Map threshold kcal to Magnitude
        idx = np.searchsorted(P_up, val)
        if idx < len(M_scale):
            mag = M_scale[idx]
            ax.axvline(mag, color=color, linestyle='--', linewidth=2.5, alpha=0.9)
            
            # Find probability at this magnitude
            prob_at_thresh = np.exp(-val / tail_scale)
            return_period = 1/prob_at_thresh
            
            # Formatting for Return Period
            if return_period >= 1000:
                rp_text = f">1,000 yr"
            else:
                rp_text = f"~{return_period:.0f} yr"
                
            ax.text(mag + 0.05, 0.01, f'{label}\n(RP {rp_text})', 
                    color=color, fontweight='bold', fontsize=11, rotation=90, va='bottom')

    # Aesthetics
    ax.set_yscale('log')
    ax.set_xlim(3, 7.5)
    ax.set_ylim(1e-4, 1)
    
    ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Exceedance Probability', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4: The Fragility Gap\nPredictive Risk of Global Food System Breach', fontsize=16, fontweight='bold')
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add Return Period axis on right
    ax_rp = ax.twinx()
    ax_rp.set_yscale('log')
    ax_rp.set_ylim(ax.get_ylim())
    y_ticks = [1, 0.1, 0.01, 0.001, 0.0001]
    ax_rp.set_yticks(y_ticks)
    ax_rp.set_yticklabels([f'{1/y:.0f}' for y in y_ticks])
    ax_rp.set_ylabel('Return Period (Years)', fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    output_path = Path('results/figure4_risk_probability.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
    
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_risk_figure()

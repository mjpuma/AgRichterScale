#!/usr/bin/env python3
"""
Figure S3: Methodology Validation - Historical Volatility & Risk Calibration
Justifies the detrending and exponential distribution assumptions for Figure 4.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

# Add parent directory to path to allow importing from agrichter
sys.path.append(str(Path(__file__).parent.parent))

from agrichter.core.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_usda_wide(data_type="production"):
    """Load USDA data in wide format."""
    file_map = {
        "production": "grains_world_usdapsd_production_jan212026.csv",
        "consumption": "grains_world_usdapsd_consumption_jan212026.csv",
        "endingstocks": "grains_world_usdapsd_endingstocks_jan212026.csv"
    }
    file_path = Path("USDAdata") / file_map[data_type]
    if not file_path.exists():
        # Fallback to general file search
        available_files = list(Path("USDAdata").glob(f"grains_world_usdapsd_{data_type}_*.csv"))
        if available_files:
            file_path = available_files[0]
        else:
            raise FileNotFoundError(f"Missing required USDA file in USDAdata/ for {data_type}")
        
    logger.info(f"Loading {data_type} data from {file_path}")
    df = pd.read_csv(file_path)
    year_cols = [c for c in df.columns if '/' in c and len(c) == 9]
    df_long = df.melt(
        id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'],
        value_vars=year_cols,
        var_name='Year_Range',
        value_name='Value'
    )
    df_long['Year'] = df_long['Year_Range'].str[:4].astype(int)
    if df_long['Value'].dtype == object:
        df_long['Value'] = df_long['Value'].str.replace(',', '').str.replace('"', '').astype(float)
    return df_long

def durbin_watson(residuals):
    """Calculate Durbin-Watson statistic."""
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

def main():
    logger.info("Generating Figure S3: Methodology Validation...")
    
    # 1. Load Data
    try:
        prod_df = load_usda_wide("production")
    except FileNotFoundError as e:
        logger.error(e)
        return 1

    allgrain_prod = prod_df[
        (prod_df['Commodity'] == 'Allgrain') & 
        (prod_df['Country'] == 'World')
    ].sort_values('Year').copy()
    
    if allgrain_prod.empty:
        logger.warning("Allgrain row not found, aggregating Wheat, Rice, Maize...")
        target_crops = ['Corn', '"Rice, Milled"', 'Wheat']
        allgrain_prod = prod_df[
            (prod_df['Commodity'].isin(target_crops)) & 
            (prod_df['Country'] == 'World')
        ].groupby('Year')['Value'].sum().reset_index()

    years = allgrain_prod['Year'].values
    vals = allgrain_prod['Value'].values
    
    # 2. FIND OPTIMAL STRUCTURAL BREAK
    # Search for break year that minimizes total RSS
    # Exclude first and last 10 years to ensure stable trends
    possible_breaks = np.arange(years[10], years[-10])
    best_rss = float('inf')
    best_break = None
    
    for b_year in possible_breaks:
        m1 = years < b_year
        m2 = years >= b_year
        
        z1 = np.polyfit(years[m1], vals[m1], 1)
        z2 = np.polyfit(years[m2], vals[m2], 1)
        
        rss1 = np.sum((vals[m1] - np.poly1d(z1)(years[m1]))**2)
        rss2 = np.sum((vals[m2] - np.poly1d(z2)(years[m2]))**2)
        
        total_rss = rss1 + rss2
        if total_rss < best_rss:
            best_rss = total_rss
            best_break = b_year
            
    logger.info(f"Objective Structural Break detected at year: {best_break}")
    break_year = best_break
    
    # Final piecewise fit with best break
    mask1 = years < break_year
    z1 = np.polyfit(years[mask1], vals[mask1], 1)
    p1 = np.poly1d(z1)
    
    mask2 = years >= break_year
    z2 = np.polyfit(years[mask2], vals[mask2], 1)
    p2 = np.poly1d(z2)
    
    trend_vals = np.zeros_like(vals)
    trend_vals[mask1] = p1(years[mask1])
    trend_vals[mask2] = p2(years[mask2])
    
    shocks_frac = (vals - trend_vals) / trend_vals
    
    # Statistics
    sigma = np.std(shocks_frac)
    dw_stat = durbin_watson(shocks_frac)
    
    # Manual R-squared
    y_mean = np.mean(vals)
    ss_tot = np.sum((vals - y_mean)**2)
    ss_res = np.sum((vals - trend_vals)**2)
    r_squared_piecewise = 1 - (ss_res / ss_tot)
    
    # Single linear trend for comparison
    z_single = np.polyfit(years, vals, 1)
    p_single = np.poly1d(z_single)
    trend_single = p_single(years)
    ss_res_single = np.sum((vals - trend_single)**2)
    r_squared_single = 1 - (ss_res_single / ss_tot)
    
    # Era Comparison
    early_shocks = shocks_frac[years < break_year]
    late_shocks = shocks_frac[years >= break_year]
    vol_early = np.std(early_shocks)
    vol_late = np.std(late_shocks)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # A) Raw Production + Piecewise Trend
    ax = axes[0, 0]
    ax.plot(years, vals / 1e6, 'ko-', label='Observed Production', markersize=4, alpha=0.7)
    ax.plot(years, trend_vals / 1e6, 'r-', lw=2.5, label=f'Optimal Piecewise Trend\n(Break @ {break_year}, $R^2={r_squared_piecewise:.3f}$)')
    ax.plot(years, trend_single / 1e6, 'k--', lw=1, alpha=0.5, label=f'Single Linear Trend ($R^2={r_squared_single:.3f}$)')
    ax.axvline(break_year, color='gray', linestyle=':', alpha=0.8)
    ax.set_title(f'A) Global Grain Production: Break @ {break_year}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Production (Million Metric Tons)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # B) Fractional Shocks
    ax = axes[0, 1]
    ax.bar(years, shocks_frac * 100, color='steelblue', alpha=0.8, label='Fractional Shock')
    ax.axhline(0, color='black', lw=1)
    ax.axhline(sigma*100, color='red', linestyle='--', alpha=0.5, label=f'Volatility ($\\sigma$ = {sigma*100:.2f}%)')
    ax.axhline(-sigma*100, color='red', linestyle='--', alpha=0.5)
    ax.set_title('B) Annual Production Shocks (Detrended)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Deviation from Trend (%)', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # C) Distribution of Shocks (Histogram + Fit)
    ax = axes[1, 0]
    # Filter for losses (negative shocks)
    losses = -shocks_frac[shocks_frac < 0]
    
    # 1. Plot histogram of all shocks
    bins = np.linspace(-0.18, 0.18, 25)
    ax.hist(shocks_frac * 100, bins=bins*100, density=True, color='gray', alpha=0.3, label='Historical Shocks (USDA)')
    
    # 2. Plot Normal Distribution (The "Standard" Assumption)
    x_full = np.linspace(-0.20, 0.20, 200)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_full / sigma)**2)
    ax.plot(x_full * 100, y_norm / 100, 'k--', lw=1.5, label=f'Normal Fit ($\sigma = {sigma*100:.1f}$%)')
    
    # 3. Plot the Exponential Risk Tail (The "Conservative" Assumption)
    # We only plot this for the loss side (negative shocks)
    lam = sigma / (-np.log(0.32))
    x_tail = np.linspace(0, 0.20, 100)
    # The exponential tail is used in Fig 4 to bound the probability of loss > L
    # Here we show its PDF to compare density
    y_exp_pdf = (1/lam) * np.exp(-x_tail/lam)
    
    # We plot it on the negative side to show how it models the "Loss" distribution
    # Note: We scale it by 0.5 because it's modeling the tail of a 2-sided distribution
    ax.plot(-x_tail * 100, 0.5 * y_exp_pdf / 100, 'r-', lw=3, label=f'Exponential Risk Tail\n($\lambda = {lam*100:.2f}$%)')
    
    ax.set_title('C) Distribution: Normal vs. Exponential Tail', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annual Production Shock (%)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-18, 18)

    # D) Empirical vs. Theoretical Exceedance (Semi-log)
    ax = axes[1, 1]
    total_years = len(shocks_frac)
    sorted_losses = np.sort(losses)[::-1]
    ranks = np.arange(1, len(sorted_losses) + 1)
    annual_empirical_p = ranks / total_years
    
    ax.semilogy(sorted_losses * 100, annual_empirical_p, 'ko', label='Historical Losses (USDA)', markersize=6)
    
    # Comparison of Exceedance Probabilities
    x_exceed = np.linspace(0, 0.20, 100)
    # Manual erfc approximation or simplified Normal Tail (for visualization)
    # p = 0.5 * exp(-0.5 * (x/sigma)^2) is a simple approximation for the exceedance tail
    p_norm_exceed = 0.5 * np.exp(-0.5 * (x_exceed / sigma)**2)
    
    # Exponential Exceedance (as used in Figure 4)
    # We use p=1.0 at x=0 for the risk boundary, or p=0.5 if modeling historical frequency
    p_exp_model = np.exp(-x_exceed / lam) # The "Conservative Boundary" version
    p_exp_fit = 0.5 * np.exp(-x_exceed / lam) # The "Frequency Fit" version
    
    ax.semilogy(x_exceed * 100, p_norm_exceed, 'k--', label='Normal Exceedance (Unsafe)')
    ax.semilogy(x_exceed * 100, p_exp_model, 'r-', lw=2.5, label='AgRichter Risk Model (Conservative)')
    ax.semilogy(x_exceed * 100, p_exp_fit, 'r:', lw=1.5, label='Exp. Frequency Fit')
    
    ax.set_title('D) The "Safety Margin" Justification', fontsize=14, fontweight='bold')
    ax.set_xlabel('Production Loss Magnitude (%)', fontsize=12)
    ax.set_ylabel('Annual Exceedance Probability', fontsize=12)
    ax.set_ylim(1e-4, 1.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.2)
    
    # Highlight the "Safety Gap"
    ax.fill_between(x_exceed * 100, p_norm_exceed, p_exp_model, color='red', alpha=0.1, label='Safety Margin')

    plt.tight_layout()
    output_path = Path('results/figureS3_methodology_validation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved validation figure to {output_path}")

    # Print Summary Statistics
    print("\n" + "="*50)
    print("STATISTICAL VALIDATION SUMMARY (USDA Allgrain World)")
    print("="*50)
    print(f"Period:              {years[0]} - {years[-1]}")
    print(f"Observations:        {len(years)} years")
    print(f"Break Year:          {break_year}")
    print(f"Trend 1 (1960-1997): {z1[0]/1e6:,.2f} MMT/year")
    print(f"Trend 2 (1998-2025): {z2[0]/1e6:,.2f} MMT/year")
    print(f"Piecewise R-squared: {r_squared_piecewise:.4f} (vs Single: {r_squared_single:.4f})")
    print(f"Residual Std Dev:    {sigma*100:.2f}% (Volatility)")
    print(f"Durbin-Watson:       {dw_stat:.4f} (Ideal: 2.0)")
    print("-" * 50)
    print("STATIONARITY CHECK (PIECEWISE)")
    print(f"Volatility 1960-1997: {vol_early*100:.2f}%")
    print(f"Volatility 1998-2025: {vol_late*100:.2f}%")
    print(f"Change Factor:       {vol_late/vol_early:.2f}x increase")
    print("-" * 50)
    print("CALIBRATION (EXPONENTIAL)")
    print(f"Scale Parameter (lambda): {lam*100:.4f}% of trend production")
    print(f"Prob(|shock| > sigma):   0.32 (Anchor point)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

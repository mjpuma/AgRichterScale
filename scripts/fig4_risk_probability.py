#!/usr/bin/env python3
"""
Figure 4: Global Systemic Risk (The Fragility Gap)

This script calculates the annual exceedance probability of global agricultural disruptions.
It integrates historical production variability with the AgRichter H-P Envelope.

DATA PEDIGREE & PROVENANCE:
- Production Data: USDAdata/grains_world_usdapsd_production_jan212026.csv
- Consumption Data: USDAdata/grains_world_usdapsd_consumption_jan212026.csv
- Stocks Data: USDAdata/grains_world_usdapsd_endingstocks_jan212026.csv
- Commodity: 'Allgrain' (Aggregate of global grain production/consumption/stocks)
- Period: 1960/61 to 2023/24

STATISTICAL METHODS:
1. Detrending: Linear trend removal from historical aggregate production.
2. Shocks: Deviations from trend as a fraction of current production.
3. Volatility: Standard deviation of historical fractional shocks.
4. Exceedance Probability: Exponential tail model calibrated to historical volatility.
5. Uncertainty: 1000-iteration bootstrap resampling of historical shocks for 95% Confidence Intervals.
6. Diagnostics: Durbin-Watson statistic for autocorrelation; split-era volatility comparison (1960-1990 vs 1991-2023).
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
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_usda_wide(data_type="production"):
    """
    Load USDA wide-format data (years as columns) and convert to long format.
    """
    # Use the refreshed Jan 2026 data
    file_map = {
        "production": "grains_world_usdapsd_production_jan212026.csv",
        "consumption": "grains_world_usdapsd_consumption_jan212026.csv",
        "endingstocks": "grains_world_usdapsd_endingstocks_jan212026.csv"
    }
    
    file_path = Path("USDAdata") / file_map[data_type]
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required USDA file: {file_path}")
        
    logger.info(f"Loading {data_type} data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Melt years into long format
    # Identify year columns (e.g., "1960/1961")
    year_cols = [c for c in df.columns if '/' in c and len(c) == 9]
    
    df_long = df.melt(
        id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'],
        value_vars=year_cols,
        var_name='Year_Range',
        value_name='Value'
    )
    
    # 2. Clean Year (take start year as in user's old code)
    df_long['Year'] = df_long['Year_Range'].str[:4].astype(int)
    
    # 3. Clean Value (remove commas, handle quotes)
    if df_long['Value'].dtype == object:
        df_long['Value'] = df_long['Value'].str.replace(',', '').str.replace('"', '').astype(float)
    
    return df_long

def durbin_watson(residuals):
    """Calculate Durbin-Watson statistic to check for autocorrelation."""
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

def calculate_shocks():
    """Calculate historical production shocks and perform diagnostics."""
    prod_df = load_usda_wide("production")
    
    # Filter for Allgrain, World
    allgrain_prod = prod_df[
        (prod_df['Commodity'] == 'Allgrain') & 
        (prod_df['Country'] == 'World')
    ].sort_values('Year').copy()
    
    if allgrain_prod.empty:
        # Fallback to sum of Corn, Rice, Wheat if Allgrain not found
        logger.warning("Allgrain row not found, aggregating Corn, Rice, Milled, and Wheat...")
        # Note: "Rice, Milled" is the name in the file
        target_crops = ['Corn', '"Rice, Milled"', 'Wheat']
        components = prod_df[
            (prod_df['Commodity'].isin(target_crops)) & 
            (prod_df['Country'] == 'World')
        ]
        allgrain_prod = components.groupby('Year')['Value'].sum().reset_index()
    
    # Detrending (Linear)
    years = allgrain_prod['Year'].values
    vals = allgrain_prod['Value'].values
    
    z = np.polyfit(years, vals, 1)
    p = np.poly1d(z)
    trend_vals = p(years)
    
    shocks_val = vals - trend_vals
    shocks_frac = shocks_val / trend_vals
    
    # --- Diagnostics ---
    # 1. Autocorrelation (Durbin-Watson)
    dw_stat = durbin_watson(shocks_frac)
    
    # 2. Nonstationarity (Era Split)
    split_year = 1990
    early_shocks = shocks_frac[years <= split_year]
    late_shocks = shocks_frac[years > split_year]
    
    vol_early = np.std(early_shocks)
    vol_late = np.std(late_shocks)
    vol_full = np.std(shocks_frac)
    
    # Use full record for the main scale, but track recent shift
    logger.info(f"Durbin-Watson Statistic: {dw_stat:.4f} (Ideal: ~2.0)")
    logger.info(f"Volatility Comparison: Early ({vol_early:.4f}) vs Recent ({vol_late:.4f})")
    logger.info(f"Full Record Volatility (sigma): {vol_full:.4f}")
    
    # --- Thresholds from wide format ---
    cons_df = load_usda_wide("consumption")
    stock_df = load_usda_wide("endingstocks")
    
    # Use 2019-2021 mean (align with SPAM 2020 reference period)
    ref_years = [2019, 2020, 2021]
    
    def get_ref_val(df, commodity):
        subset = df[
            (df['Commodity'] == commodity) & 
            (df['Country'] == 'World') & 
            (df['Year'].isin(ref_years))
        ]
        return subset['Value'].mean()
    
    avg_cons = get_ref_val(cons_df, 'Allgrain')
    avg_stocks = get_ref_val(stock_df, 'Allgrain')
    
    return {
        'sigma': vol_full,
        'sigma_recent': vol_late,
        'shocks': shocks_frac,
        'current_trend_prod': trend_vals[-1],
        'avg_cons': avg_cons,
        'avg_stocks': avg_stocks,
        'dw_stat': dw_stat,
        'vol_ratio': vol_late / vol_early if vol_early > 0 else 1.0
    }

def generate_all_figures():
    """Main entry point to generate both versions of Figure 4 efficiently."""
    logger.info("Generating Figure 4: The Fragility Gap (Dual Versions)...")
    
    # 1. Historical Analysis
    stats = calculate_shocks()
    config = Config(crop_type='allgrain', root_dir='.')
    caloric_content = config.get_caloric_content()
    kcal_per_tmt = 1e9 * caloric_content
    current_prod_kcal = stats['current_trend_prod'] * kcal_per_tmt
    
    # 2. Envelope Mapping (Load ONCE)
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)
    global_envelope = calculator.calculate_hp_envelope(prod_df, harv_df)
    
    # Clear large dataframes immediately
    del prod_df, harv_df
    import gc
    gc.collect()
    
    H_km2 = global_envelope['upper_bound_harvest']
    P_up = global_envelope['upper_bound_production']
    M_scale = np.log10(H_km2)
    
    # 3. Risk Modeling with Bootstrapping (Compute ONCE)
    mags_plot = np.linspace(3, 8, 200)
    losses_at_mags = np.interp(mags_plot, M_scale, P_up)
    
    n_boot = 1000
    boot_probs = np.zeros((n_boot, len(mags_plot)))
    logger.info(f"Performing {n_boot} bootstrap iterations...")
    shocks = stats['shocks']
    
    for i in range(n_boot):
        sample = np.random.choice(shocks, size=len(shocks), replace=True)
        sigma_boot = np.std(sample)
        scale_boot = (sigma_boot * current_prod_kcal) / (-np.log(0.32))
        boot_probs[i, :] = np.exp(-losses_at_mags / scale_boot)
    
    prob_median = np.percentile(boot_probs, 50, axis=0)
    prob_low = np.percentile(boot_probs, 2.5, axis=0)
    prob_high = np.percentile(boot_probs, 97.5, axis=0)
    
    # Thresholds setup
    thresh_kcal = {
        '1 Month': (stats['avg_cons'] / 12.0) * kcal_per_tmt,
        '3 Months': (stats['avg_cons'] / 4.0) * kcal_per_tmt,
        'Total Stocks': stats['avg_stocks'] * kcal_per_tmt
    }
    thresh_mags = {}
    for label, val in thresh_kcal.items():
        idx = np.searchsorted(P_up, val)
        thresh_mags[label] = M_scale[idx] if idx < len(M_scale) else 8.0

    # 4. Generate Both Plots
    for mode in ['zonal', 'exposure']:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if mode == 'zonal':
            resilient_mask = (mags_plot <= thresh_mags['1 Month'])
            absorption_mask = (mags_plot > thresh_mags['1 Month']) & (mags_plot <= thresh_mags['Total Stocks'])
            failure_mask = (mags_plot > thresh_mags['Total Stocks'])
            
            plt.rcParams['hatch.linewidth'] = 1.2
            ax.fill_between(mags_plot, prob_median, 1e-4, where=resilient_mask, 
                            color='seagreen', alpha=0.2, label='Zone of Resilience (Buffered)')
            ax.fill_between(mags_plot, prob_median, 1e-4, where=absorption_mask, 
                            color='goldenrod', alpha=0.2, label='Zone of Absorption (Reserves)')
            
            # Shading + Hatching only for Failure zone
            ax.fill_between(mags_plot, prob_median, 1e-4, where=failure_mask, 
                            color='crimson', alpha=0.15)
            ax.fill_between(mags_plot, prob_median, 1e-4, where=failure_mask, 
                            facecolor='none', edgecolor='crimson', alpha=0.4, hatch='///',
                            label='The Fragility Gap (Systemic Breach)')
            
            ax.axvspan(3, 6, color='gray', alpha=0.03, linestyle=':', label='Historical Obs. Range')
            
        elif mode == 'exposure':
            ax.axvspan(3, 6, color='gray', alpha=0.04, label='Historical Observation Range')
            mask = mags_plot >= thresh_mags['Total Stocks']
            plt.rcParams['hatch.linewidth'] = 1.5
            ax.fill_between(mags_plot, prob_median, 1e-4, where=mask, color='crimson', alpha=0.1)
            ax.fill_between(mags_plot, prob_median, 1e-4, where=mask, 
                            facecolor='none', edgecolor='crimson', alpha=0.7, hatch='//', 
                            label='The Fragility Gap (Unbuffered Risk)')

        # Shared elements
        ax.fill_between(mags_plot, prob_low, prob_high, color='gray', alpha=0.2, label='95% Confidence Interval')
        ax.plot(mags_plot, prob_median, color='firebrick', linewidth=4, label='Annual Exceedance Probability')
        
        colors = {'1 Month': '#FFD700', '3 Months': '#FF4500', 'Total Stocks': '#800080'}
        main_scale = (stats['sigma'] * current_prod_kcal) / (-np.log(0.32))
        for label, val in sorted(thresh_kcal.items(), key=lambda x: x[1]):
            mag = thresh_mags[label]
            color = colors.get(label, 'black')
            ax.axvline(mag, color=color, linestyle='--', linewidth=3, alpha=0.8)
            prob_at_thresh = np.exp(-val / main_scale)
            rp = 1.0 / prob_at_thresh
            rp_text = f">1,000 yr" if rp > 1000 else f"~{rp:.0f} yr"
            ax.text(mag + 0.04, 2e-4, f'{label}\n(RP {rp_text})', 
                    color=color, fontweight='bold', fontsize=11, rotation=90, va='bottom')

        diag_text = (f"DIAGNOSTICS:\n"
                    f"Durbin-Watson: {stats['dw_stat']:.2f}\n"
                    f"Volatility Shift: {stats['vol_ratio']:.1%} increase\n"
                    f"$\sigma_{{full}}$: {stats['sigma']:.2%}\n"
                    f"$\sigma_{{recent}}$: {stats['sigma_recent']:.2%}")
        ax.text(0.02, 0.05, diag_text, transform=ax.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        ax.set_yscale('log')
        ax.set_xlim(3, 7.5)
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annual Exceedance Probability', fontsize=14, fontweight='bold')
        ax.set_title(f'Figure 4: The Fragility Gap\nPredictive Risk of Global Food System Breach', fontsize=16, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.15)
        
        ax_rp = ax.twinx()
        ax_rp.set_yscale('log')
        ax_rp.set_ylim(ax.get_ylim())
        y_ticks = [1, 0.1, 0.01, 0.001, 0.0001]
        ax_rp.set_yticks(y_ticks)
        ax_rp.set_yticklabels([f'{1/y:.0f}' for y in y_ticks])
        ax_rp.set_ylabel('Return Period (Years)', fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        plt.tight_layout()
        output_path = Path(f'results/figure4_risk_probability_{mode}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved {output_path}")
        plt.close()

if __name__ == "__main__":
    generate_all_figures()



#!/usr/bin/env python3
"""
Figure 4: Global Systemic Risk (The Fragility Gap) - ULTRA-BOLD VERSION
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, FixedLocator
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper
from agrichter.data.events import EventsProcessor
from agrichter.analysis.event_calculator import EventCalculator
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fig4_v4')

# ULTRA-BOLD STYLE SETTINGS
mpl.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 48,
    'axes.labelsize': 42,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'axes.linewidth': 5.0,
    'lines.linewidth': 8.0,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.width': 5.0,
    'ytick.major.width': 5.0,
    'xtick.major.size': 12,
    'ytick.major.size': 12,
    'svg.fonttype': 'path'
})

def load_usda_wide(data_type="production"):
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

def calculate_shocks():
    prod_df = load_usda_wide("production")
    allgrain_prod = prod_df[(prod_df['Commodity'] == 'Allgrain') & (prod_df['Country'] == 'World')].sort_values('Year').copy()
    years, vals = allgrain_prod['Year'].values, allgrain_prod['Value'].values
    z = np.polyfit(years, vals, 1)
    p = np.poly1d(z)
    trend_vals = p(years)
    shocks_frac = (vals - trend_vals) / trend_vals
    
    cons_df, stock_df = load_usda_wide("consumption"), load_usda_wide("endingstocks")
    ref_years = [2019, 2020, 2021]
    def get_ref_val(df, commodity):
        subset = df[(df['Commodity'] == commodity) & (df['Country'] == 'World') & (df['Year'].isin(ref_years))]
        return subset['Value'].mean()
    
    return {
        'sigma': np.std(shocks_frac),
        'shocks': shocks_frac,
        'current_trend_prod': trend_vals[-1],
        'avg_cons': get_ref_val(cons_df, 'Allgrain'),
        'avg_stocks': get_ref_val(stock_df, 'Allgrain')
    }

def generate_figure_4():
    logger.info("Generating Figure 4 (Ultra-Bold)...")
    stats = calculate_shocks()
    config = Config(crop_type='allgrain', root_dir='.')
    kcal_per_tmt = 1e9 * config.get_caloric_content()
    current_prod_kcal = stats['current_trend_prod'] * kcal_per_tmt
    
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)
    global_envelope = calculator.calculate_hp_envelope(prod_df, harv_df)
    
    H_km2 = global_envelope['upper_bound_harvest']
    P_up = global_envelope['upper_bound_production']
    M_scale = np.log10(H_km2)
    
    mags_plot = np.linspace(3, 7.5, 200)
    losses_at_mags = np.interp(mags_plot, M_scale, P_up)
    
    # Risk Modeling
    n_boot = 1000
    boot_probs = np.zeros((n_boot, len(mags_plot)))
    shocks = stats['shocks']
    for i in range(n_boot):
        sample = np.random.choice(shocks, size=len(shocks), replace=True)
        sigma_boot = np.std(sample)
        scale_boot = (sigma_boot * current_prod_kcal) / (-np.log(0.32))
        boot_probs[i, :] = np.exp(-losses_at_mags / scale_boot)
    
    prob_median = np.percentile(boot_probs, 50, axis=0)
    prob_low = np.percentile(boot_probs, 2.5, axis=0)
    prob_high = np.percentile(boot_probs, 97.5, axis=0)
    
    thresh_kcal = {
        '3-Month Reserve': (stats['avg_cons'] / 4.0) * kcal_per_tmt,
        'Total Stocks': stats['avg_stocks'] * kcal_per_tmt
    }
    thresh_mags = {}
    for label, val in thresh_kcal.items():
        idx = np.searchsorted(P_up, val)
        thresh_mags[label] = M_scale[idx] if idx < len(M_scale) else 7.5

    # Main Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # The Fragility Gap Shading
    mask = mags_plot >= thresh_mags['Total Stocks']
    ax.fill_between(mags_plot, prob_median, 1e-4, where=mask, 
                    color='crimson', alpha=0.15)
    ax.fill_between(mags_plot, prob_median, 1e-4, where=mask, 
                    facecolor='none', edgecolor='crimson', alpha=0.6, hatch='///', lw=0)
    
    # Confidence Interval
    ax.fill_between(mags_plot, prob_low, prob_high, color='gray', alpha=0.2)
    
    # Main Risk Curve
    ax.plot(mags_plot, prob_median, color='firebrick', lw=10)
    
    # Threshold Verticals
    colors = {'3-Month Reserve': 'orange', 'Total Stocks': 'purple'}
    main_scale = (stats['sigma'] * current_prod_kcal) / (-np.log(0.32))
    
    for label, val in thresh_kcal.items():
        mag = thresh_mags[label]
        color = colors[label]
        ax.axvline(mag, color=color, ls='--', lw=6, alpha=0.9)
        prob_at = np.exp(-val / main_scale)
        rp = 1.0 / prob_at
        rp_text = f">1,000 yr" if rp > 1000 else f"~{rp:.0f} yr"
        ax.text(mag + 0.05, 1.5e-4, f'{label}\n(RP {rp_text})', 
                color=color, fontweight='bold', fontsize=28, rotation=90, va='bottom')

    # Styling
    ax.set_yscale('log')
    ax.set_xlim(3, 7.5)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel('Magnitude ($M_D$)', labelpad=25)
    ax.set_ylabel('Annual Probability', labelpad=25)
    ax.set_title('The Fragility Gap', pad=40)
    ax.grid(True, which="both", ls="-", alpha=0.2, lw=2)
    
    # Twin axis for Return Period
    ax_rp = ax.twinx()
    ax_rp.set_yscale('log')
    ax_rp.set_ylim(ax.get_ylim())
    y_ticks = [1, 0.1, 0.01, 0.001, 0.0001]
    ax_rp.set_yticks(y_ticks)
    ax_rp.set_yticklabels([f'{1/y:.0f}' if y < 1 else '1' for y in y_ticks])
    ax_rp.set_ylabel('Return Period (Years)', labelpad=25)
    
    # Remove top spines
    ax.spines['top'].set_visible(False)
    ax_rp.spines['top'].set_visible(False)
    
    plt.tight_layout()
    results_dir = Path('results/fig4_components')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(results_dir / 'figure4_risk_v4.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'figure4_risk_v4.svg', format='svg', bbox_inches='tight')
    
    # STANDALONE LEGEND
    import matplotlib.patches as mpatches
    fig_leg = plt.figure(figsize=(18, 4))
    
    # Create proxies
    # Annual Prob (Line)
    p1 = plt.Line2D([0], [0], color='firebrick', lw=10, label='Annual Exceedance Probability')
    # CI (Wide line/shading)
    p2 = plt.Line2D([0], [0], color='gray', alpha=0.3, lw=15, label='95% Confidence Interval')
    # 3-Month (Dashed)
    p3 = plt.Line2D([0], [0], color='orange', ls='--', lw=6, label='3-Month Reserve')
    # Total Stocks (Dashed)
    p4 = plt.Line2D([0], [0], color='purple', ls='--', lw=6, label='Total Stocks')
    
    # Fragility Gap - Using diagonal lines to match figure
    # We use a denser hatch '//////' to ensure it's visible in the small legend box
    gap_proxy = mpatches.Patch(facecolor='#FFCCCC', hatch='//////', edgecolor='crimson', 
                               linewidth=2, label='The Fragility Gap')

    proxies = [p1, p2, p3, p4, gap_proxy]
    
    fig_leg.legend(handles=proxies, loc='center', ncol=3, frameon=True, fontsize=26, 
                   handleheight=3.0, handlelength=4.0)
    plt.axis('off')
    plt.savefig(results_dir / 'figure4_legend_v4.svg', format='svg', bbox_inches='tight')
    plt.savefig(results_dir / 'figure4_legend_v4.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    logger.info(f"🎉 Figure 4 Overhaul Complete! Files in {results_dir}")

if __name__ == "__main__":
    generate_figure_4()

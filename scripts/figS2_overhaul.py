#!/usr/bin/env python3
"""
Figure S2 Overhaul: 4-Panel Crop Comparison (Ultra-Bold)
Wheat, Maize, Rice, All Grains
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
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figS2_v4')

# ULTRA-BOLD STYLE SETTINGS
mpl.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 42,
    'axes.labelsize': 36,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'axes.linewidth': 4.0,
    'lines.linewidth': 8.0,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.width': 4.0,
    'ytick.major.width': 4.0,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'svg.fonttype': 'path'
})

def load_usda_wide(data_type="production"):
    file_map = {
        "production": "grains_world_usdapsd_production_jan212026.csv",
        "consumption": "grains_world_usdapsd_consumption_jan212026.csv"
    }
    file_path = Path("USDAdata") / file_map[data_type]
    df = pd.read_csv(file_path)
    year_cols = [c for c in df.columns if '/' in c and len(c) == 9]
    df_long = df.melt(id_vars=['Commodity', 'Country'], value_vars=year_cols, var_name='Year_Range', value_name='Value')
    df_long['Year'] = df_long['Year_Range'].str[:4].astype(int)
    if df_long['Value'].dtype == object:
        df_long['Value'] = df_long['Value'].str.replace(',', '').str.replace('"', '').astype(float)
    return df_long

def generate_figure_s2():
    logger.info("Generating Figure S2: Crop Comparison (Ultra-Bold Aligned)...")
    
    crops = ['allgrain', 'wheat', 'maize', 'rice']
    crop_labels = {'allgrain': 'All Grains', 'wheat': 'Wheat', 'maize': 'Maize', 'rice': 'Rice'}
    
    # Load USDA consumption for thresholds
    cons_df = load_usda_wide("consumption")
    ref_years = [2019, 2020, 2021]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for i, crop in enumerate(crops):
        ax = axes[i]
        logger.info(f"  Processing {crop}...")
        
        config = Config(crop_type=crop, root_dir='.')
        grid_manager = GridDataManager(config)
        prod_df, harv_df = grid_manager.load_spam_data()
        
        calculator = HPEnvelopeCalculatorV2(config)
        envelope = calculator.calculate_hp_envelope(prod_df, harv_df)
        
        lower_harvest = np.array(envelope['lower_bound_harvest'])
        upper_harvest = np.array(envelope['upper_bound_harvest'])
        lower_bound = np.array(envelope['lower_bound_production'])
        upper_bound = np.array(envelope['upper_bound_production'])
        
        # Plot fill
        ax.fill(np.concatenate([np.log10(lower_harvest), np.flip(np.log10(upper_harvest))]),
                np.concatenate([lower_bound, np.flip(upper_bound)]),
                color='gray', alpha=0.2)
        
        # Plot bounds
        ax.plot(np.log10(upper_harvest), upper_bound, color='black', lw=8)
        ax.plot(np.log10(lower_harvest), lower_bound, color='blue', lw=8)
        
        # --- THRESHOLDS (1-Month, 3-Month) ---
        usda_crop = 'Allgrain' if crop == 'allgrain' else crop.title()
        if crop == 'rice': usda_crop = 'Rice, Milled' # Standardize for USDA
        if crop == 'maize': usda_crop = 'Corn'        # USDA uses 'Corn' for Maize
        
        avg_cons = cons_df[(cons_df['Commodity'] == usda_crop) & 
                           (cons_df['Country'] == 'World') & 
                           (cons_df['Year'].isin(ref_years))]['Value'].mean()
        
        kcal_per_tmt = 1e9 * config.get_caloric_content()
        thresh_1m = (avg_cons / 12.0) * kcal_per_tmt
        thresh_3m = (avg_cons / 4.0) * kcal_per_tmt
        
        # Plot aligned thresholds (matching Fig 1 & 2 colors)
        ax.axhline(thresh_1m, color='#FFD700', ls='--', lw=6, alpha=0.9)
        ax.axhline(thresh_3m, color='#FF4500', ls='--', lw=6, alpha=0.9)
        
        # Text labels for thresholds
        # 1-Month: Below the line
        ax.text(7.3, thresh_1m / 1.15, '1-Month', color='#FFD700', fontsize=24, ha='right', va='top', fontweight='bold')
        # 3-Month: Above the line
        ax.text(7.3, thresh_3m * 1.15, '3-Month', color='#FF4500', fontsize=24, ha='right', va='bottom', fontweight='bold')
        
        # Convergence point
        total_prod = envelope['upper_bound_production'][-1]
        total_harv = envelope['upper_bound_harvest'][-1]
        ax.plot(np.log10(total_harv), total_prod, 'go', ms=25, mec='darkgreen', mew=3, zorder=10)
        
        # Style
        ax.set_yscale('log')
        ax.set_xlim([1, 7.5])
        ax.set_ylim([1e8, 1.62e16])
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5, 6, 7]))
        ax.yaxis.set_major_locator(LogLocator(base=100.0, numticks=5))
        
        ax.set_title(crop_labels[crop], pad=20)
        if i >= 2: ax.set_xlabel('Magnitude ($M_D$)')
        if i % 2 == 0: ax.set_ylabel('Production Loss (kcal)')
        
        ax.grid(True, alpha=0.2, lw=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    results_dir = Path('results/supplement_components')
    results_dir.mkdir(exist_ok=True, parents=True)

    # Create a single legend for the figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=8, label='Upper Bound'),
        Line2D([0], [0], color='blue', lw=8, label='Lower Bound'),
        Line2D([0], [0], color='#FFD700', ls='--', lw=6, label='1-Month Reserve'),
        Line2D([0], [0], color='#FF4500', ls='--', lw=6, label='3-Month Reserve'),
        Line2D([0], [0], marker='o', color='w', label='Global Convergence',
               markerfacecolor='green', markersize=25, markeredgecolor='darkgreen', markeredgewidth=3)
    ]
    
    # Create a separate figure for the legend
    fig_leg = plt.figure(figsize=(24, 2))
    fig_leg.legend(handles=legend_elements, loc='center', ncol=5, frameon=False, fontsize=32)
    fig_leg.savefig(results_dir / 'figureS2_legend_v4.svg', format='svg', bbox_inches='tight')
    plt.close(fig_leg)
    
    plt.savefig(results_dir / 'figureS2_comparative_v4.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'figureS2_comparative_v4.svg', format='svg', bbox_inches='tight')
    
    plt.close('all')
    logger.info(f"🎉 Figure S2 Overhaul Complete! Files in {results_dir}")

if __name__ == "__main__":
    generate_figure_s2()

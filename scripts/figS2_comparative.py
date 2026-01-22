
import sys
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing from agrichter
sys.path.append(str(Path(__file__).parent.parent))

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Country definitions - Full Spectrum
COUNTRY_CONFIGS = [
    {'key': 'Global', 'label': 'Global (Aggregate)', 'spam_name': None, 'color': 'black', 'linewidth': 3.0},
    {'key': 'USA', 'label': 'USA', 'spam_name': 'United States of America', 'color': '#1f77b4', 'linewidth': 2.0},
    {'key': 'China', 'label': 'China', 'spam_name': 'China', 'color': '#d62728', 'linewidth': 2.0},
    {'key': 'India', 'label': 'India', 'spam_name': 'India', 'color': '#ff7f0e', 'linewidth': 2.0},
    {'key': 'Brazil', 'label': 'Brazil', 'spam_name': 'Brazil', 'color': '#2ca02c', 'linewidth': 2.0},
    {'key': 'Egypt', 'label': 'Egypt (Stiff)', 'spam_name': 'Egypt', 'color': '#9467bd', 'linewidth': 2.0},
    {'code': 'FRA', 'label': 'France', 'spam_name': 'France', 'color': '#8c564b', 'linewidth': 1.5},
    {'code': 'AUS', 'label': 'Australia', 'spam_name': 'Australia', 'color': '#e377c2', 'linewidth': 1.5},
    {'code': 'ARG', 'label': 'Argentina', 'spam_name': 'Argentina', 'color': '#7f7f7f', 'linewidth': 1.5},
    {'code': 'DEU', 'label': 'Germany', 'spam_name': 'Germany', 'color': '#bcbd22', 'linewidth': 1.5},
    {'code': 'NGA', 'label': 'Nigeria', 'spam_name': 'Nigeria', 'color': '#17becf', 'linewidth': 1.5},
]

def generate_comparative_figure():
    """Generate comparative H-P envelope figure with all countries."""
    logger.info("Generating Figure S2: National Fingerprint Gallery...")
    
    # Initialize config
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')

    # Load Data
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    # Pre-calculate envelopes
    envelopes = {}
    calculator = HPEnvelopeCalculatorV2(config)
    
    for country in COUNTRY_CONFIGS:
        key = country.get('key') or country.get('code')
        spam_name = country['spam_name']
        
        logger.info(f"Calculating {key}...")
        
        if key == 'Global':
            c_prod = prod_df
            c_harv = harv_df
        else:
            mask = prod_df['ADM0_NAME'] == spam_name
            if mask.sum() == 0:
                logger.warning(f"No data for {key}")
                continue
            c_prod = prod_df[mask]
            c_harv = harv_df[mask]
            
        try:
            envelope = calculator.calculate_hp_envelope(c_prod, c_harv)
            envelopes[key] = envelope
        except Exception as e:
            logger.error(f"Failed to calculate {key}: {e}")

    # Plotting Logic
    scales = [('log', 'log', 'loglog'), ('log', 'linear', 'semilogx')]
    
    for x_scale, y_scale, suffix in scales:
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xscale('linear')
        ax.set_yscale(y_scale)
        ax.set_xlim(3, 8)
        
        if y_scale == 'linear':
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(1e-2, 110)

        # Plot Global first as background
        global_env = envelopes['Global']
        magnitude_global = np.log10(global_env['upper_bound_harvest'])
        P_up_pct_global = (global_env['upper_bound_production'] / global_env['upper_bound_production'][-1]) * 100
        ax.plot(magnitude_global, P_up_pct_global, color='black', linewidth=3, label='Global Aggregate', zorder=10)

        # Plot other countries
        for country in COUNTRY_CONFIGS:
            key = country.get('key') or country.get('code')
            if key == 'Global' or key not in envelopes: continue
            
            env = envelopes[key]
            magnitude = np.log10(env['upper_bound_harvest'])
            P_up_pct = (env['upper_bound_production'] / env['upper_bound_production'][-1]) * 100
            
            ax.plot(magnitude, P_up_pct, color=country['color'], linewidth=country['linewidth'], 
                    label=country['label'], alpha=0.8)

        ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Production Loss (% of Total)', fontsize=14, fontweight='bold')
        ax.set_title(f'Figure S2: National Agricultural Fingerprint Gallery\n(Normalized Envelopes - {suffix} version)', fontsize=18, fontweight='bold')
        
        ax.grid(True, which="both", ls="-", alpha=0.15)
        ax.legend(fontsize=10, loc='upper left', ncol=2, framealpha=0.9)
        
        plt.tight_layout()
        output_path = Path(f'results/figureS2_comparative_envelopes_{suffix}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
        logger.info(f"Saved {output_path}")
        plt.close()

if __name__ == "__main__":
    generate_comparative_figure()

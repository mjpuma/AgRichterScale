
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

# Country definitions
COUNTRY_CONFIGS = [
    {'key': 'Global', 'label': 'Global', 'spam_name': None, 'color': 'black', 'linewidth': 2.5},
    {'key': 'USA', 'label': 'USA', 'spam_name': 'United States of America', 'color': 'blue', 'linewidth': 2.0},
    {'key': 'China', 'label': 'China', 'spam_name': 'China', 'color': 'red', 'linewidth': 2.0},
    {'key': 'India', 'label': 'India', 'spam_name': 'India', 'color': 'orange', 'linewidth': 2.0},
    {'key': 'Brazil', 'label': 'Brazil', 'spam_name': 'Brazil', 'color': 'green', 'linewidth': 2.0},
]

def generate_comparative_figure():
    """Generate comparative H-P envelope figure."""
    logger.info("Generating Comparative National Vulnerability Figure...")
    
    # Initialize config
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    config.data_files['country_codes'] = Path('ancillary/CountryCode_Convert.xls')

    # Load Data
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    # Pre-calculate envelopes
    envelopes = {}
    calculator = HPEnvelopeCalculatorV2(config)
    
    for country in COUNTRY_CONFIGS:
        key = country['key']
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

    # Define scales to generate
    scales = [
        ('log', 'log', 'loglog'),
        ('log', 'linear', 'semilogx')
    ]
    
    for x_scale, y_scale, suffix in scales:
        logger.info(f"Plotting {suffix} version...")
        # Setup Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.set_xscale('linear')
        ax.set_yscale(y_scale)
        
        # Use scientific notation for Y axis if linear
        if y_scale == 'linear':
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        # Set X-axis limits for AgRichter Magnitude
        ax.set_xlim(3, 8)
        
        # Plot each country
        for country in COUNTRY_CONFIGS:
            key = country['key']
            if key not in envelopes:
                continue
                
            envelope = envelopes[key]
            label = country['label']
            color = country['color']
            lw = country['linewidth']
            
            H_km2 = envelope['upper_bound_harvest']
            P_up = envelope['upper_bound_production']
            P_low = envelope['lower_bound_production']
            
            # Convert harvest area to AgRichter Magnitude
            magnitude = np.log10(H_km2)
            
            # Calculate total production for normalization
            total_P = P_up[-1]
            
            # Normalize to percentage
            P_up_pct = (P_up / total_P) * 100
            P_low_pct = (P_low / total_P) * 100
            
            # Plot shaded band
            ax.fill_between(magnitude, P_low_pct, P_up_pct, color=color, alpha=0.2, label=f"{label} range")
            
            # Plot upper bound line for emphasis
            ax.plot(magnitude, P_up_pct, color=color, linewidth=lw, alpha=0.8)
            
            # Add marker at total (100%)
            total_mag = magnitude[-1]
            ax.plot(total_mag, 100, marker='o', color=color, markersize=6)

        # Formatting
        ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Production Loss (% of Total)', fontsize=13, fontweight='bold')
        ax.set_title(f'Figure S2: Comparative National Vulnerability Fingerprints\n(H-P Envelopes Normalized - {suffix})', fontsize=16)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=10, loc='upper left', ncol=2)
        
        # Set Y-axis limits for percentage
        if y_scale == 'linear':
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(1e-2, 110) # Log scale needs positive floor
            
        plt.tight_layout()
        
        output_path = Path(f'results/figureS2_comparative_envelopes_{suffix}.png')
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.savefig(output_path.with_suffix('.svg'), format='svg')
        logger.info(f"Saved to {output_path}")
        
        plt.close(fig)

if __name__ == "__main__":
    generate_comparative_figure()

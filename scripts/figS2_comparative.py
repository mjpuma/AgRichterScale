
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt

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
        
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        
        # Use scientific notation for Y axis if linear
        if y_scale == 'linear':
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        # Set X-axis limits to start from Mag 3 (10^3)
        ax.set_xlim(left=1e3)
        
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
            P_kcal = envelope['upper_bound_production']
            
            # Plot
            ax.plot(H_km2, P_kcal, label=f"{label}", color=color, linewidth=lw)
            
            # Add marker at total
            total_H = H_km2[-1]
            total_P = P_kcal[-1]
            ax.plot(total_H, total_P, marker='o', color=color, markersize=8)
            
            # Annotate Total
            ax.annotate(f"{label} Total", 
                        xy=(total_H, total_P), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, color=color)

        # Formatting
        ax.set_xlabel('Disrupted Harvest Area (km²)', fontsize=14)
        ax.set_ylabel('Production Loss (kcal)', fontsize=14)
        ax.set_title(f'Comparative National Agricultural Vulnerability\n(Upper Bound H-P Envelopes - {suffix})', fontsize=16)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=12, loc='upper left')
        
        # Add AgriRichter magnitudes
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        
        for mag in range(4, 8):
            val = 10**mag
            if val > x_min and val < x_max:
                ax.axvline(val, color='gray', linestyle='--', alpha=0.5)
                # Adjust text position based on scale
                if y_scale == 'log':
                    text_y = y_min * (y_max/y_min)**0.05 # 5% up in log scale
                else:
                    text_y = y_min + (y_max - y_min) * 0.05
                    
                ax.text(val, text_y, f"Mag {mag}\n({val:,.0f} km²)", 
                        rotation=90, verticalalignment='bottom', color='gray')

        plt.tight_layout()
        
        output_path = Path(f'results/figureS2_comparative_envelopes_{suffix}.png')
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.savefig(output_path.with_suffix('.svg'), format='svg')
        logger.info(f"Saved to {output_path}")
        
        plt.close(fig)

if __name__ == "__main__":
    generate_comparative_figure()

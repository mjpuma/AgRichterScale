#!/usr/bin/env python3
"""
Figure S3: Agricultural Vulnerability Typologies (Stiff vs. Resilient Envelopes)

Compares normalized H-P envelopes for Global, Egypt (Stiff), and USA (Resilient).
This supports the claim that envelope width/shape is a fingerprint of national
agricultural topology.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

# Country definitions for typology comparison
TYPOLOGY_CONFIGS = [
    {'key': 'Global', 'label': 'Global (Aggregate)', 'spam_name': None, 'color': 'black', 'linewidth': 2.5},
    {'key': 'Egypt', 'label': 'Egypt (Stiff/Concentrated)', 'spam_name': 'Egypt', 'color': 'red', 'linewidth': 2.0},
    {'key': 'USA', 'label': 'USA (Resilient/Dispersed)', 'spam_name': 'United States of America', 'color': 'blue', 'linewidth': 2.0},
]

def _load_spam_data(config: Config):
    config.data_files['production'] = Path(
        'spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv'
    )
    config.data_files['harvest_area'] = Path(
        'spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv'
    )

    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    return prod_df, harv_df

def _compute_envelopes(prod_df, harv_df, calculator):
    envelopes = {}
    for country in TYPOLOGY_CONFIGS:
        key = country['key']
        spam_name = country['spam_name']
        logger.info(f"Calculating envelope for {key}...")

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
            envelopes[key] = calculator.calculate_hp_envelope(c_prod, c_harv)
        except Exception as e:
            logger.error(f"Failed to calculate {key}: {e}")

    return envelopes

def generate_typology_figure():
    """Generate normalized typology comparison figure."""
    logger.info("Generating Figure S3: Resilience Typologies...")

    config = Config(crop_type='allgrain', root_dir='.')
    prod_df, harv_df = _load_spam_data(config)

    calculator = HPEnvelopeCalculatorV2(config)
    envelopes = _compute_envelopes(prod_df, harv_df, calculator)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(2, 8)
    ax.set_ylim(1e-2, 110)

    for country in TYPOLOGY_CONFIGS:
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
        
        # Convert to magnitude
        magnitude = np.log10(H_km2)
        
        # Total production for normalization
        total_P = P_up[-1]
        
        # Normalize to percentage
        P_up_pct = (P_up / total_P) * 100
        P_low_pct = (P_low / total_P) * 100

        # Plot shaded band
        ax.fill_between(magnitude, P_low_pct, P_up_pct, color=color, alpha=0.15, label=f"{label} Range")
        
        # Plot upper bound line for clarity
        ax.plot(magnitude, P_up_pct, color=color, linewidth=lw, alpha=0.8)
        
        # Mark total point
        total_mag = magnitude[-1]
        ax.plot(total_mag, 100, marker='o', color=color, markersize=6)

    ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Production Loss (% of Total)', fontsize=13, fontweight='bold')
    ax.set_title('Figure S3: Agricultural Resilience Typologies\n(Concentrated vs. Dispersed Production Fingerprints)', fontsize=16)
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()

    output_path = Path('results/figureS3_comparative_envelopes.png')
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_typology_figure()

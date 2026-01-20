#!/usr/bin/env python3
"""
Figure 4: Historical Exceedance Probability (Risk Curve)

Calculates the annual exceedance probability of agricultural disruptions
based on the historical record since 1315.
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
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.data.events import EventsProcessor
from agririchter.analysis.event_calculator import EventCalculator
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for risk analysis
START_YEAR = 1315
CURRENT_YEAR = 2026
TOTAL_YEARS = CURRENT_YEAR - START_YEAR

def load_real_events(crop: str, config: Config, grid_manager: GridDataManager):
    """Load and calculate real historical events."""
    logger.info(f"Loading real historical events for {crop}...")
    
    country_file = Path('ancillary/DisruptionCountry.xls')
    state_file = Path('ancillary/DisruptionStateProvince.xls')
    
    if not country_file.exists() or not state_file.exists():
        logger.error("Event Excel files not found!")
        return pd.DataFrame()
    
    country_sheets = pd.read_excel(country_file, sheet_name=None, engine='xlrd')
    state_sheets = pd.read_excel(state_file, sheet_name=None, engine='xlrd')
    
    events_processor = EventsProcessor(config)
    raw_events_data = {'country': country_sheets, 'state': state_sheets}
    events_data = events_processor.process_event_sheets(raw_events_data)
    
    spatial_mapper = SpatialMapper(config, grid_manager)
    spatial_mapper.load_country_codes_mapping()
    
    event_calculator = EventCalculator(config, grid_manager, spatial_mapper)
    events_df = event_calculator.calculate_all_events(events_data)
    
    return events_df

def generate_risk_figure():
    """Generate the risk curve figure."""
    logger.info("Generating Figure 4: Risk Probability Curve...")
    
    # Initialize components
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    
    grid_manager = GridDataManager(config)
    
    # Load and calculate events
    events_df = load_real_events('allgrain', config, grid_manager)
    
    # Filter valid events
    valid_events = events_df[events_df['magnitude'].notna()].copy()
    valid_events = valid_events.sort_values('magnitude', ascending=False)
    
    # Calculate Exceedance Probability
    # P(M >= Mi) = m / (N + 1)
    valid_events['rank'] = range(1, len(valid_events) + 1)
    valid_events['probability'] = valid_events['rank'] / (TOTAL_YEARS + 1)
    valid_events['return_period'] = 1 / valid_events['probability']
    
    # Calculate global envelope to find threshold magnitudes
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)
    global_envelope = calculator.calculate_hp_envelope(prod_df, harv_df)
    
    H_km2 = global_envelope['upper_bound_harvest']
    P_up = global_envelope['upper_bound_production']
    M_scale = np.log10(H_km2)
    
    thresholds = config.thresholds
    buffer_mags = {}
    for label, val in thresholds.items():
        # Find M where Upper Bound P hits threshold
        idx = np.searchsorted(P_up, val)
        if idx < len(M_scale):
            buffer_mags[label] = M_scale[idx]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot historical events
    ax.scatter(valid_events['magnitude'], valid_events['probability'], 
               color='black', zorder=5, label='Historical Events (1315-2025)')
    
    # Step plot for empirical distribution
    ax.step(valid_events['magnitude'], valid_events['probability'], 
            where='post', color='gray', alpha=0.5, linestyle='--')
    
    # Annotate key events
    top_events = ['GreatFamine', 'DustBowl', 'ChineseFamine1960', 'NoSummer', 'Drought18761878']
    for _, row in valid_events.iterrows():
        if row['event_name'] in top_events:
            ax.annotate(row['event_name'], (row['magnitude'], row['probability']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Vertical lines for buffers
    colors = {'1 Month': 'gold', '3 Months': 'orange', 'Total Stocks': 'purple'}
    for label, mag in buffer_mags.items():
        color = colors.get(label, 'red')
        ax.axvline(mag, color=color, linestyle='-', alpha=0.7, label=f'{label} Breach ($M_D \\approx {mag:.1f}$)')
        
    # Aesthetics
    ax.set_yscale('log')
    ax.set_xlim(3, 8)
    ax.set_ylim(5e-4, 1)
    
    ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Annual Exceedance Probability', fontsize=13, fontweight='bold')
    ax.set_title('Figure 4: Risk-Exceedance Curve for Global Agricultural Disruptions', fontsize=15, fontweight='bold')
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add Return Period axis on top
    ax2 = ax.twiny()
    ax2.set_xscale(ax.get_xscale()) # No, magnitude is linear, prob is log on Y. 
    # Return period is 1/Prob. 
    # Wait, twinX would be for secondary Y. I want secondary X? No, Return Period maps to Probability.
    # So I want a secondary Y axis for Return Period.
    ax_rp = ax.twinx()
    ax_rp.set_yscale('log')
    ax_rp.set_ylim(ax.get_ylim())
    # Prob 10^-1 -> RP 10^1
    # Prob 10^-3 -> RP 10^3
    y_ticks = ax.get_yticks()
    ax_rp.set_yticks(y_ticks)
    ax_rp.set_yticklabels([f'{1/y:.0f}' if y > 0 else '' for y in y_ticks])
    ax_rp.set_ylabel('Return Period (Years)', fontsize=13, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path('results/figure4_risk_probability.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
    
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_risk_figure()

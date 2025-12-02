#!/usr/bin/env python3
"""
Figure 3: H-P Envelopes (4 panels)

Generates H-P envelope figures for wheat, maize, rice, and all grains.
Shows harvest area disruption vs production loss with historical events.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figure3')

# Set journal-quality fonts
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'font.weight': 'normal',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
})

from lib.config import Config
from lib.grid_manager import GridDataManager
from lib.spatial_mapper import SpatialMapper
from lib.events import EventsProcessor
from lib.event_calculator import EventCalculator
from lib.envelope_calculator import HPEnvelopeCalculatorV2
from lib.envelope_viz import HPEnvelopeVisualizer
import pandas as pd


def load_real_events(crop: str, config: Config, grid_manager: GridDataManager):
    """Load and calculate real historical events."""
    logger.info(f"  Loading real historical events for {crop}...")
    
    try:
        # Load event definitions from Excel files
        country_file = Path('ancillary/DisruptionCountry.xls')
        state_file = Path('ancillary/DisruptionStateProvince.xls')
        
        if not country_file.exists() or not state_file.exists():
            logger.warning("  Event Excel files not found, using empty DataFrame")
            return pd.DataFrame()
        
        # Load Excel sheets
        country_sheets = pd.read_excel(country_file, sheet_name=None, engine='xlrd')
        state_sheets = pd.read_excel(state_file, sheet_name=None, engine='xlrd')
        
        # Process events
        events_processor = EventsProcessor(config)
        raw_events_data = {'country': country_sheets, 'state': state_sheets}
        events_data = events_processor.process_event_sheets(raw_events_data)
        
        logger.info(f"    Loaded {len(events_data)} event definitions")
        
        # Calculate event losses using EventCalculator
        spatial_mapper = SpatialMapper(config, grid_manager)
        spatial_mapper.load_country_codes_mapping()
        
        event_calculator = EventCalculator(config, grid_manager, spatial_mapper)
        events_df = event_calculator.calculate_all_events(events_data)
        
        logger.info(f"    Calculated losses for {len(events_df)} events")
        return events_df
        
    except Exception as e:
        logger.warning(f"  Failed to load real events: {e}")
        return pd.DataFrame()


def calculate_envelope(crop: str):
    """Calculate H-P envelope for a crop."""
    logger.info(f"  Calculating envelope for {crop}...")
    
    # Initialize config
    config = Config(crop_type=crop, root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    
    # Load data
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    # Calculate envelope
    calculator = HPEnvelopeCalculatorV2(config)
    envelope_data = calculator.calculate_hp_envelope(prod_df, harv_df)
    
    logger.info(f"    ✅ Envelope calculated: {len(envelope_data.get('disrupted_areas', []))} points")
    return envelope_data, config, grid_manager


def main():
    """Generate Figure 3: H-P Envelopes."""
    logger.info("=" * 60)
    logger.info("FIGURE 3: H-P Envelopes (4 panels)")
    logger.info("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        crops = ['wheat', 'maize', 'rice', 'allgrain']
        crop_names = {'wheat': 'Wheat', 'maize': 'Maize', 'rice': 'Rice', 'allgrain': 'All Grains'}
        
        # Generate individual figures for each crop
        logger.info("Generating individual H-P envelope figures...")
        for crop in crops:
            logger.info(f"Processing {crop}...")
            
            # Calculate envelope
            envelope_data, config, grid_manager = calculate_envelope(crop)
            
            # Load REAL events (not sample data)
            events_data = load_real_events(crop, config, grid_manager)
            
            # Create visualizer
            visualizer = HPEnvelopeVisualizer(config)
            
            # Generate figure
            fig = visualizer.create_hp_envelope_plot(
                envelope_data, 
                events_data,
                save_path=results_dir / f'figure3_{crop}_individual.png'
            )
            plt.close(fig)
            
            logger.info(f"  ✅ {crop} saved")
        
        # Create 4-panel combined figure with proper plots
        logger.info("Creating 4-panel combined figure with real plots...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Store envelope data and events for all crops
        all_data = {}
        for crop in crops:
            logger.info(f"  Loading data for {crop} panel...")
            envelope_data, config, grid_manager = calculate_envelope(crop)
            events_data = load_real_events(crop, config, grid_manager)
            all_data[crop] = {'envelope': envelope_data, 'events': events_data, 'config': config}
        
        # Determine axis ranges
        # Allgrain gets its own scale (panel 0)
        # Wheat, Maize, Rice share a scale (panels 1-3)
        
        # Find max values for individual crops (wheat, maize, rice)
        individual_max_h = max([
            all_data['wheat']['envelope']['lower_bound_harvest'][-1],
            all_data['maize']['envelope']['lower_bound_harvest'][-1],
            all_data['rice']['envelope']['lower_bound_harvest'][-1]
        ])
        individual_max_p = max([
            all_data['wheat']['envelope']['upper_bound_production'][-1],
            all_data['maize']['envelope']['upper_bound_production'][-1],
            all_data['rice']['envelope']['upper_bound_production'][-1]
        ])
        
        # Plot order: allgrain, wheat, maize, rice
        plot_order = ['allgrain', 'wheat', 'maize', 'rice']
        
        for i, crop in enumerate(plot_order):
            ax = axes[i]
            data = all_data[crop]
            envelope = data['envelope']
            events = data['events']
            
            # Plot envelope fill
            lower_h = envelope['lower_bound_harvest']
            upper_h = envelope['upper_bound_harvest']
            lower_p = envelope['lower_bound_production']
            upper_p = envelope['upper_bound_production']
            
            # Convert to magnitude
            lower_mag = np.log10(lower_h)
            upper_mag = np.log10(upper_h)
            
            # Fill envelope
            X_env = np.concatenate([lower_mag, np.flip(upper_mag)])
            Y_env = np.concatenate([lower_p, np.flip(upper_p)])
            ax.fill(X_env, Y_env, color='gray', alpha=0.3, label='H-P Envelope')
            
            # Plot boundary lines
            ax.plot(upper_mag, upper_p, 'k-', linewidth=2, label='Upper Bound', alpha=0.8)
            ax.plot(lower_mag, lower_p, 'b-', linewidth=2, label='Lower Bound', alpha=0.8)
            
            # Plot events if available
            if not events.empty and 'harvest_area_km2' in events.columns:
                events_mag = np.log10(events['harvest_area_km2'])
                ax.scatter(events_mag, events['production_loss_kcal'], 
                          c='red', s=80, alpha=0.7, edgecolors='darkred', 
                          linewidths=1.5, zorder=5, label='Historical Events')
            
            # Set scale
            ax.set_yscale('log')
            
            # Set axis limits based on crop
            if crop == 'allgrain':
                # Allgrain gets its own scale
                ax.set_xlim([2, 7])
                ax.set_ylim([1e10, upper_p[-1] * 1.2])
            else:
                # Individual crops share scale
                ax.set_xlim([2, 6.5])
                ax.set_ylim([1e10, individual_max_p * 1.2])
            
            # Labels and title
            ax.set_xlabel(r'Magnitude $M_D = \log_{10}(A_H)$ [km²]', fontsize=14)
            ax.set_ylabel('Production Loss [kcal]', fontsize=14)
            ax.set_title(crop_names[crop], fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Legend only on first panel
            if i == 0:
                ax.legend(loc='lower right', fontsize=11)
        
        plt.tight_layout(pad=2.0)
        output_path = results_dir / 'figure3_hp_envelopes.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("")
        logger.info("🎉 SUCCESS!")
        logger.info(f"✅ Individual figures saved: figure3_{{crop}}_individual.png")
        logger.info(f"✅ Combined figure saved: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

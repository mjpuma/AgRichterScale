#!/usr/bin/env python3
"""
Figure 1: H-P Envelopes (4 panels)

Generates H-P envelope figures for wheat, maize, rice, and all grains.
Shows harvest area disruption vs production loss with historical events.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Add parent directory to path to allow importing from agririchter
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figure1')

# Set journal-quality fonts (Nature style - Larger)
mpl.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.weight': 'normal',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
})

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.data.events import EventsProcessor
from agririchter.analysis.event_calculator import EventCalculator
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2
from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer
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
    """Generate Figure 1: H-P Envelopes."""
    logger.info("=" * 60)
    logger.info("FIGURE 1: H-P Envelopes (4 panels)")
    logger.info("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Updated order: All Grains first
        crops = ['allgrain', 'wheat', 'maize', 'rice']
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
                save_path=results_dir / f'figure1_{crop}_individual.png'
            )
            plt.close(fig)
            
            logger.info(f"  ✅ {crop} saved")
        
        # Create 4-panel combined figure
        logger.info("Creating 4-panel combined figure...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, crop in enumerate(crops):
            logger.info(f"  Adding {crop} to combined figure...")
            
            # Calculate envelope
            envelope_data, config, grid_manager = calculate_envelope(crop)
            
            # Load REAL events
            events_data = load_real_events(crop, config, grid_manager)
            
            # Create visualizer
            visualizer = HPEnvelopeVisualizer(config)
            
            # Plot on subplot
            visualizer.create_hp_envelope_plot(
                envelope_data, 
                events_data,
                ax=axes[i]
            )
            
            # Set subplot title
            axes[i].set_title(crop_names[crop], fontsize=20, fontweight='bold')
        
        plt.tight_layout(pad=2.0)
        output_path = results_dir / 'figure1_hp_envelopes.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("")
        logger.info("🎉 SUCCESS!")
        logger.info(f"✅ Individual figures saved: figure1_{{crop}}_individual.png")
        logger.info(f"✅ Combined figure saved: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

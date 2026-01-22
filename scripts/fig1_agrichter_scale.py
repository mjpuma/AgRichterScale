#!/usr/bin/env python3
"""
Figure 1: AgRichter Scale (4 panels)

Generates AgRichter Scale figures for wheat, maize, rice, and all grains.
Shows magnitude (M_D = log10(A_H)) vs production loss with historical events.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add parent directory to path to allow importing from agrichter
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

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper
from agrichter.data.events import EventsProcessor
from agrichter.analysis.event_calculator import EventCalculator
from agrichter.visualization.agrichter_scale import AgRichterScaleVisualizer
import pandas as pd


def load_real_events(crop: str, config: Config, grid_manager: GridDataManager):
    """Load and calculate real historical events."""
    logger.info(f"  Loading real historical events for {crop}...")
    
    try:
        # Load event definitions from Excel files
        country_file = Path('ancillary/DisruptionCountry.xls')
        state_file = Path('ancillary/DisruptionStateProvince.xls')
        
        if not country_file.exists() or not state_file.exists():
            logger.error("  ❌ Event Excel files not found! Cannot load real events.")
            logger.error(f"  Looking for: {country_file} and {state_file}")
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
        logger.error(f"  ❌ Failed to load real events: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def main():
    """Generate Figure 1: AgRichter Scale."""
    logger.info("=" * 60)
    logger.info("FIGURE 1: AgRichter Scale (4 panels)")
    logger.info("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Ensure individual plots directory exists
        individual_dir = results_dir / 'individual_plots'
        individual_dir.mkdir(exist_ok=True)
        
        # Updated order: All Grains first
        crops = ['allgrain', 'wheat', 'maize', 'rice']
        crop_names = {'wheat': 'Wheat', 'maize': 'Maize', 'rice': 'Rice', 'allgrain': 'All Grains'}
        
        # Generate individual figures for each crop
        logger.info("Generating individual AgRichter figures...")
        for crop in crops:
            logger.info(f"  Processing {crop}...")
            
            # Initialize config - using dynamic thresholds automatically
            # NOTE: We set use_dynamic_thresholds=False for Fig 2 because it shows Area vs Magnitude,
            # and production thresholds (in kcal) don't map directly without assuming yields.
            config = Config(crop_type=crop, root_dir='.', use_dynamic_thresholds=False)
            
            # Explicitly clear thresholds to ensure NO arbitrary lines appear on Figure 3
            config.thresholds = {}
            
            config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
            config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
            
            # Load SPAM data
            grid_manager = GridDataManager(config)
            grid_manager.load_spam_data()
            
            # Load REAL events (not sample data)
            events_data = load_real_events(crop, config, grid_manager)
            
            # Create visualizer
            visualizer = AgRichterScaleVisualizer(config, use_event_types=True)
            
            # Generate figure
            fig = visualizer.create_agrichter_scale_plot(
                events_data, 
                save_path=individual_dir / f'figure1_{crop}_individual.png'
            )
            # Also save as SVG
            fig.savefig(individual_dir / f'figure1_{crop}_individual.svg', format='svg', bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"    ✅ {crop} saved to {individual_dir}")
        
        # Create 4-panel combined figure
        logger.info("Creating 4-panel combined figure...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, crop in enumerate(crops):
            logger.info(f"  Adding {crop} to combined figure...")
            
            # Initialize config
            # NOTE: We set use_dynamic_thresholds=False for Fig 2 because it shows Area vs Magnitude,
            # and production thresholds (in kcal) don't map directly without assuming yields.
            config = Config(crop_type=crop, root_dir='.', use_dynamic_thresholds=False)
            
            # Explicitly clear thresholds to ensure NO arbitrary lines appear on Figure 3
            config.thresholds = {}
            
            config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
            config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
            
            # Load SPAM data
            grid_manager = GridDataManager(config)
            grid_manager.load_spam_data()
            
            # Load REAL events
            events_data = load_real_events(crop, config, grid_manager)
            
            # Create visualizer and plot on subplot
            visualizer = AgRichterScaleVisualizer(config, use_event_types=True)
            visualizer.create_agrichter_scale_plot(
                events_data,
                ax=axes[i]
            )
            
            # Set subplot title
            axes[i].set_title(crop_names[crop], fontsize=20, fontweight='bold')

        plt.tight_layout(pad=2.0)
        output_path = results_dir / 'figure1_agrichter_scale.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        # Also save as SVG
        plt.savefig(results_dir / 'figure1_agrichter_scale.svg', format='svg', bbox_inches='tight', facecolor='white')
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
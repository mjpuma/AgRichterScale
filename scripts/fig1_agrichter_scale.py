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
import numpy as np
import pandas as pd

# Add parent directory to path to allow importing from agrichter
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figure1')

# Set journal-quality fonts (Nature style)
mpl.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'font.weight': 'normal',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
})

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper
from agrichter.data.events import EventsProcessor
from agrichter.analysis.event_calculator import EventCalculator
from agrichter.visualization.agrichter_scale import AgRichterScaleVisualizer

def load_real_events(crop: str, config: Config, grid_manager: GridDataManager):
    """Load and calculate real historical events."""
    logger.info(f"  Loading real historical events for {crop}...")
    try:
        country_file = Path('ancillary/DisruptionCountry.xls')
        state_file = Path('ancillary/DisruptionStateProvince.xls')
        if not country_file.exists() or not state_file.exists():
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
    except Exception as e:
        logger.error(f"  ❌ Failed to load real events: {e}")
        return pd.DataFrame()

def main():
    """Generate Figure 1: AgRichter Scale."""
    logger.info("=" * 60)
    logger.info("FIGURE 1: AgRichter Scale (4 panels)")
    logger.info("=" * 60)
    
    try:
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        individual_dir = results_dir / 'individual_plots'
        individual_dir.mkdir(exist_ok=True)
        
        crops = ['allgrain', 'wheat', 'maize', 'rice']
        
        # Combined figure
        logger.info("Creating 4-panel combined figure...")
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        axes = axes.flatten()
        
        for i, crop in enumerate(crops):
            config = Config(crop_type=crop, root_dir='/Users/mjp38/GitHub/AgRichter2025', use_dynamic_thresholds=True)
            grid_manager = GridDataManager(config)
            grid_manager.load_spam_data()
            events_data = load_real_events(crop, config, grid_manager)
            
            visualizer = AgRichterScaleVisualizer(config, use_event_types=True)
            visualizer.create_agrichter_scale_plot(
                events_data,
                ax=axes[i],
                is_combined=True,
                show_labels=True,  # Show labels for expanded priority events
                show_axis_labels=False,
                show_legend=False
            )

        fig.supxlabel('AgRichter Magnitude ($M_D = \log_{10}(A_H/\mathrm{km}^2)$)', 
                       fontsize=14, fontweight='bold', y=0.02)
        fig.supylabel('Harvest Area Disrupted (km$^2$)', 
                       fontsize=14, fontweight='bold', x=0.02)

        import matplotlib.lines as mlines
        legend_elements = [
            mlines.Line2D([0], [0], marker='o', color='w', label='Climate & Weather',
                          markerfacecolor='#1f77b4', markersize=10, markeredgecolor='black'),
            mlines.Line2D([0], [0], marker='s', color='w', label='Conflict & Policy',
                          markerfacecolor='#d62728', markersize=10, markeredgecolor='black'),
            mlines.Line2D([0], [0], marker='D', color='w', label='Pest & Disease',
                          markerfacecolor='#2ca02c', markersize=10, markeredgecolor='black'),
            mlines.Line2D([0], [0], marker='^', color='w', label='Geophysical',
                          markerfacecolor='#ff7f0e', markersize=10, markeredgecolor='black'),
            mlines.Line2D([0], [0], marker='*', color='w', label='Compound/Other',
                          markerfacecolor='#7f7f7f', markersize=10, markeredgecolor='black'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.06),
                   ncol=3, fontsize=10, frameon=False)

        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.25, hspace=0.25)
        plt.savefig(results_dir / 'figure1_agrichter_scale.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(results_dir / 'figure1_agrichter_scale.svg', format='svg', bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("🎉 SUCCESS!")
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

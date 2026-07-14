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
        all_events_for_export = []
        
        # Combined figure
        logger.info("Creating 4-panel combined figure...")
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        axes = axes.flatten()
        
        for i, crop in enumerate(crops):
            config = Config(crop_type=crop, root_dir='.', use_dynamic_thresholds=True)
            grid_manager = GridDataManager(config)
            grid_manager.load_spam_data()
            events_data = load_real_events(crop, config, grid_manager)
            
            # Prepare data for export
            if not events_data.empty:
                export_df = events_data.copy()
                export_df['crop_type'] = crop
                # Add display name
                export_df['display_name'] = export_df['event_name'].apply(config.get_event_label)
                # Ensure harvest_area_km2 exists
                if 'harvest_area_km2' not in export_df.columns:
                    export_df['harvest_area_km2'] = export_df['harvest_area_loss_ha'] * 0.01
                # Rename magnitude to magnitude_MD
                export_df = export_df.rename(columns={'magnitude': 'magnitude_MD'})
                all_events_for_export.append(export_df)

            visualizer = AgRichterScaleVisualizer(config, use_event_types=True)
            visualizer.create_agrichter_scale_plot(
                events_data,
                ax=axes[i],
                is_combined=True,
                show_labels=True,  # Show labels for expanded priority events
                show_axis_labels=False,
                show_legend=False
            )

            # Also generate individual plots for each crop (PNG and SVG)
            logger.info(f"  Generating individual plots for {crop}...")
            ind_fig = visualizer.create_agrichter_scale_plot(
                events_data,
                show_labels=True,
                show_axis_labels=True,
                show_legend=True
            )
            ind_fig.savefig(individual_dir / f'figure1_{crop}_individual.png', dpi=300, bbox_inches='tight')
            ind_fig.savefig(individual_dir / f'figure1_{crop}_individual.svg', format='svg', bbox_inches='tight')
            plt.close(ind_fig)

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
        
        # Final Export of computed events to CSV
        if all_events_for_export:
            final_export_df = pd.concat(all_events_for_export, ignore_index=True)
            
            # Merge with metadata from food_disruptions.csv
            metadata_path = Path('ancillary/food_disruptions.csv')
            if metadata_path.exists():
                metadata_df = pd.read_csv(metadata_path)
                # Select only the columns we need from metadata
                meta_cols = ['event_name', 'year_start', 'year_end', 'region', 'event_type']
                # Drop event_type from computed data to use the one from metadata (driver category)
                final_export_df = final_export_df.drop(columns=['event_type'], errors='ignore')
                final_export_df = pd.merge(final_export_df, metadata_df[meta_cols], on='event_name', how='left')
            
            # Select and order final columns
            column_order = [
                'event_name', 'display_name', 'crop_type', 'magnitude_MD',
                'harvest_area_km2', 'production_loss_kcal', 'event_type',
                'year_start', 'year_end', 'region'
            ]
            # Filter to only requested columns that exist
            final_export_df = final_export_df[[c for c in column_order if c in final_export_df.columns]]
            
            csv_path = results_dir / 'historical_events_computed.csv'
            final_export_df.to_csv(csv_path, index=False)
            logger.info(f"✅ Exported computed events to {csv_path}")

        logger.info("🎉 SUCCESS!")
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

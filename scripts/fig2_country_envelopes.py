#!/usr/bin/env python3
"""
Figure 2: Country H-P Envelopes

Generates country-level H-P envelope figures for major producers and additional countries
to show varying envelope shapes.
"""

import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Add parent directory to path to allow importing from agririchter
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figure2')

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


# GDAM Country Codes for filtering events
COUNTRY_GDAM_CODES = {
    'USA': 240.0,
    'China': 48.0,
    'India': 105.0,
    'Brazil': 32.0,
    'France': 79.0,
    'Egypt': 69.0,
    'Australia': 14.0,
    'Argentina': 11.0,
    'Germany': 86.0,
    'Nigeria': 163.0
}

# Country names from SPAM data (ADM0_NAME)
COUNTRY_NAMES = {
    'USA': 'United States of America',
    'China': 'China',
    'India': 'India',
    'Brazil': 'Brazil',
    'France': 'France',
    'Egypt': 'Egypt',
    'Australia': 'Australia',
    'Argentina': 'Argentina',
    'Germany': 'Germany',
    'Nigeria': 'Nigeria'
}


def load_real_events(country_key: str, config: Config, grid_manager: GridDataManager):
    """Load and calculate real historical events for a specific country."""
    logger.info(f"  Loading real historical events for {country_key}...")
    
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
        
        # Calculate event losses using EventCalculator with country filtering
        spatial_mapper = SpatialMapper(config, grid_manager)
        spatial_mapper.load_country_codes_mapping()
        
        event_calculator = EventCalculator(config, grid_manager, spatial_mapper)
        
        # Filter events to only include impacts within the specific country
        target_gdam_code = COUNTRY_GDAM_CODES.get(country_key)
        if target_gdam_code:
            logger.info(f"    Filtering event losses for {country_key} (GDAM: {target_gdam_code})")
            events_df = event_calculator.calculate_all_events(events_data, limit_to_country_code=target_gdam_code)
        else:
            logger.warning(f"    ⚠️ No GDAM code found for {country_key}, calculating global losses")
            events_df = event_calculator.calculate_all_events(events_data)
        
        logger.info(f"    Calculated filtered losses for {len(events_df)} events")
        return events_df
        
    except Exception as e:
        logger.warning(f"  Failed to load real events: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def calculate_country_envelope(country_key: str, country_name_spam: str, global_production: float = None):
    """Calculate H-P envelope for a specific country and scale thresholds."""
    logger.info(f"  Calculating envelope for {country_key} (SPAM name='{country_name_spam}')...")
    
    # Initialize config for allgrain with dynamic thresholds
    import inspect
    logger.info(f"Imported Config from: {inspect.getfile(Config)}")
    config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=True)
    
    # WORKAROUND for mysterious AttributeError
    if not hasattr(config, 'get_ipc_colors'):
        logger.warning("Patching missing get_ipc_colors method on Config object")
        def get_ipc_colors_patch():
            if config.use_dynamic_thresholds:
                 return {
                     '1 Month': '#FFD700',       # Gold/Yellow
                     '3 Months': '#FF4500',      # OrangeRed
                     'Total Stocks': '#800080'   # Purple
                 }
            return {
                1: '#00FF00', 2: '#FFFF00', 3: '#FFA500', 4: '#FF0000', 5: '#800080'
            }
        config.get_ipc_colors = get_ipc_colors_patch
        
    if not hasattr(config, 'get_crop_indices'):
        logger.warning("Patching missing get_crop_indices method on Config object")
        def get_crop_indices_patch():
            return config.crop_indices
        config.get_crop_indices = get_crop_indices_patch

    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    
    # Load data
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    # Filter by country name (ADM0_NAME)
    if 'ADM0_NAME' in prod_df.columns:
        prod_df_country = prod_df[prod_df['ADM0_NAME'] == country_name_spam]
        harv_df_country = harv_df[harv_df['ADM0_NAME'] == country_name_spam]
    else:
        logger.warning(f"    ⚠️  No suitable country column found (ADM0_NAME), using all data")
        prod_df_country = prod_df
        harv_df_country = harv_df
    
    if len(prod_df_country) == 0:
        logger.error(f"    ❌ No data found for country '{country_name_spam}'")
        return {}, config, grid_manager

    # Calculate country total production for scaling thresholds
    country_production = grid_manager.get_crop_production(prod_df_country, config.get_crop_indices(), convert_to_kcal=True)
    country_harvest = grid_manager.get_crop_harvest_area(harv_df_country, config.get_crop_indices()) * config.get_unit_conversions()['hectares_to_km2']
    
    logger.info(f"    Stats for {country_key}: Prod={country_production:.2e} kcal, Harv={country_harvest:.2f} km²")
    
    # Scale thresholds if global production is provided
    if global_production and global_production > 0:
        scaling_factor = country_production / global_production
        logger.info(f"    Scaling global thresholds by factor {scaling_factor:.4f} (Country/Global Production)")
        
        new_thresholds = {}
        for key, value in config.thresholds.items():
            new_thresholds[key] = value * scaling_factor
        config.thresholds = new_thresholds
        logger.info(f"    Scaled Thresholds: {config.thresholds}")
    
    # Calculate envelope
    calculator = HPEnvelopeCalculatorV2(config)
    envelope_data = calculator.calculate_hp_envelope(prod_df_country, harv_df_country)
    
    return envelope_data, config, grid_manager, country_production, country_harvest


def get_global_stats():
    """Calculate global production and harvest for scaling."""
    logger.info("Calculating Global Statistics for Scaling...")
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    grid_manager = GridDataManager(config)
    prod_df, _ = grid_manager.load_spam_data()
    
    global_production = grid_manager.get_crop_production(prod_df, config.get_crop_indices(), convert_to_kcal=True)
    logger.info(f"  Global Production (All Grains): {global_production:.2e} kcal")
    return global_production


def main():
    """Generate Figure 2: Country H-P Envelopes."""
    logger.info("=" * 60)
    logger.info("FIGURE 2: Country H-P Envelopes")
    logger.info("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Get global stats for scaling thresholds
        global_production = get_global_stats()
        
        # Core countries for the main combined figure
        core_countries = ['USA', 'China', 'India', 'Brazil']
        
        # Additional countries to explore typology
        additional_countries = ['France', 'Egypt', 'Australia', 'Argentina']
        
        all_countries = core_countries + additional_countries
        
        # Generate individual figures for all countries
        logger.info("Generating individual country H-P envelope figures...")
        
        # Store data for combined plots
        plot_data = {}
        
        for country_key in all_countries:
            logger.info(f"Processing {country_key}...")
            
            try:
                country_name_spam = COUNTRY_NAMES[country_key]
                envelope_data, config, grid_manager, country_prod, country_harv = calculate_country_envelope(
                    country_key, country_name_spam, global_production
                )
                
                if not envelope_data:
                    logger.warning(f"  ⚠️ Skipping {country_key} due to empty envelope data")
                    continue

                # Load REAL events
                events_data = load_real_events(country_key, config, grid_manager)
                
                # Store for later
                plot_data[country_key] = {
                    'envelope': envelope_data,
                    'events': events_data,
                    'config': config,
                    'prod': country_prod,
                    'harv': country_harv
                }
                
                # Create visualizer
                visualizer = HPEnvelopeVisualizer(config)
                
                # Generate figure
                fig = visualizer.create_hp_envelope_plot(
                    envelope_data, 
                    events_data,
                    save_path=results_dir / f'figure2_{country_key.lower()}_individual.png',
                    title=f'H-P Envelope - {country_key}',
                    total_production=country_prod,
                    total_harvest=country_harv
                )
                # Also save as SVG
                fig.savefig(results_dir / f'figure2_{country_key.lower()}_individual.svg', format='svg', bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"  ✅ {country_key} saved")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process {country_key}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Combine all countries into a single 6-panel figure (Core + selected Additional)
        # Selection: USA, China, India, Brazil, France, Argentina
        # (Egypt and Australia moved to supplementary/individual)
        selected_countries = ['USA', 'China', 'India', 'Brazil', 'France', 'Argentina']
        
        logger.info("Creating 6-panel combined figure (Selected Countries)...")
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        # Debug: Print plot_data keys
        logger.info(f"Available countries in plot_data: {list(plot_data.keys())}")
        
        for i, country_key in enumerate(selected_countries):
            logger.info(f"  Plotting subplot {i}: {country_key}")
            if country_key not in plot_data:
                logger.warning(f"    ⚠️ {country_key} missing from plot_data!")
                continue
                
            data = plot_data[country_key]
            
            # Create visualizer with country-specific config (scaled thresholds)
            if not hasattr(data['config'], 'get_ipc_colors'):
                logger.error(f"Config object missing get_ipc_colors! Dir: {dir(data['config'])}")
            
            visualizer = HPEnvelopeVisualizer(data['config'])
            
            visualizer.create_hp_envelope_plot(
                data['envelope'], 
                data['events'],
                ax=axes[i],
                total_production=data['prod'],
                total_harvest=data['harv']
            )
            
            axes[i].set_title(country_key, fontsize=24, fontweight='bold')
            logger.info(f"    ✓ Added subplot for {country_key}")
        
        plt.tight_layout(pad=2.0)
        output_path = results_dir / 'figure2_country_envelopes.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        # Also save as SVG
        plt.savefig(results_dir / 'figure2_country_envelopes.svg', format='svg', bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"✅ Combined 6-panel figure saved: {output_path}")
        
        # Export country statistics table
        stats_list = []
        for key, data in plot_data.items():
            # Extract thresholds
            thresholds = data['config'].thresholds
            stats = {
                'Country': key,
                'Annual Production (kcal)': f"{data['prod']:.2e}",
                'Harvest Area (km2)': f"{data['harv']:,.0f}",
                '1-Month Supply (kcal)': f"{thresholds.get('1 Month', 0):.2e}",
                '3-Month Supply (kcal)': f"{thresholds.get('3 Months', 0):.2e}",
                'Total Stocks (kcal)': f"{thresholds.get('Total Stocks', 0):.2e}"
            }
            stats_list.append(stats)
            
        stats_df = pd.DataFrame(stats_list)
        stats_path = results_dir / 'country_stocks_verification_table.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"✅ Verification table saved: {stats_path}")

        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

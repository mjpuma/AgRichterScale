#!/usr/bin/env python3
"""
Figure 4: Country H-P Envelopes (4 panels)

Generates country-level H-P envelope figures for USA, China, India, and Brazil.
Shows country-specific harvest area disruption vs production loss with historical events.
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
logger = logging.getLogger('figure4')

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


def load_real_events(country_name: str, config: Config, grid_manager: GridDataManager):
    """Load and calculate real historical events for a specific country."""
    logger.info(f"  Loading real historical events for {country_name}...")
    
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
        
        # TODO: Filter events by country if possible
        logger.info(f"    Calculated losses for {len(events_df)} events (all countries)")
        return events_df
        
    except Exception as e:
        logger.warning(f"  Failed to load real events: {e}")
        return pd.DataFrame()


# Country FIPS codes from SPAM data (2-letter codes)
COUNTRY_CODES = {
    'USA': 'US',
    'China': 'CH',
    'India': 'IN',
    'Brazil': 'BR'
}


def calculate_country_metrics(prod_df_country, harv_df_country, envelope_data, country_name):
    """Calculate country-specific metrics for text box display."""
    import numpy as np
    
    # Get allgrain columns
    allgrain_cols = [col for col in prod_df_country.columns if col.endswith('_A') and 
                    col.startswith(('WHEA', 'RICE', 'MAIZ', 'BARL', 'MILL', 'SORG', 'OCER'))]
    
    # Calculate totals
    total_production_mt = prod_df_country[allgrain_cols].sum().sum() / 1000  # Convert to Mt
    total_harvest_ha = harv_df_country[allgrain_cols].sum().sum()  # hectares
    total_harvest_km2 = total_harvest_ha * 0.01  # convert to km²
    
    # Calculate vulnerability metric: envelope width at 50% of total harvest
    target_harvest = total_harvest_km2 * 0.5
    
    # Find envelope bounds at 50% disruption
    lower_harvest = envelope_data['lower_bound_harvest']
    upper_harvest = envelope_data['upper_bound_harvest']
    lower_production = envelope_data['lower_bound_production']
    upper_production = envelope_data['upper_bound_production']
    
    # Interpolate to find production values at 50% harvest disruption
    try:
        lower_prod_50 = np.interp(target_harvest, lower_harvest, lower_production)
        upper_prod_50 = np.interp(target_harvest, upper_harvest, upper_production)
        vulnerability_width = upper_prod_50 - lower_prod_50
        vulnerability_pct = (vulnerability_width / upper_prod_50) * 100 if upper_prod_50 > 0 else 0
    except:
        vulnerability_pct = 0
    
    return {
        'total_production_mt': total_production_mt,
        'total_harvest_km2': total_harvest_km2,
        'vulnerability_pct': vulnerability_pct,
        'country_name': country_name
    }


def add_country_metrics_textbox(ax, metrics):
    """Add a professional metrics text box to the plot."""
    
    # Format the metrics text
    text = (
        f"{metrics['country_name']} Agricultural Profile\n"
        f"{'─' * 35}\n"
        f"Total Production: {metrics['total_production_mt']:.1f} Mt/yr\n"
        f"Harvest Area: {metrics['total_harvest_km2']:.0f} km²\n"
        f"Vulnerability*: {metrics['vulnerability_pct']:.1f}%\n\n"
        f"*Envelope width at 50% disruption"
    )
    
    # Add text box in upper left corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace')


def calculate_country_envelope(country_name: str, country_code: str):
    """Calculate H-P envelope for a specific country."""
    logger.info(f"  Calculating envelope for {country_name} (FIPS={country_code})...")
    
    # Initialize config for allgrain
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    
    # Load data
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    # Filter by FIPS0 country code (SPAM 2020 uses FIPS codes)
    if 'FIPS0' in prod_df.columns:
        prod_df_country = prod_df[prod_df['FIPS0'] == country_code]
        harv_df_country = harv_df[harv_df['FIPS0'] == country_code]
        logger.info(f"    Filtered by FIPS0={country_code}: {len(prod_df_country):,} cells")
    elif 'ADM0_NAME' in prod_df.columns:
        # Fallback to country name matching
        prod_df_country = prod_df[prod_df['ADM0_NAME'].str.contains(country_name, case=False, na=False)]
        harv_df_country = harv_df[harv_df['ADM0_NAME'].str.contains(country_name, case=False, na=False)]
        logger.info(f"    Filtered by name={country_name}: {len(prod_df_country):,} cells")
    else:
        logger.error(f"    ❌ No country code column found!")
        prod_df_country = prod_df
        harv_df_country = harv_df
    
    if len(prod_df_country) == 0:
        logger.error(f"    ❌ No cells found for {country_name}! Using all data as fallback.")
        prod_df_country = prod_df
        harv_df_country = harv_df
    
    # Calculate envelope
    calculator = HPEnvelopeCalculatorV2(config)
    envelope_data = calculator.calculate_hp_envelope(prod_df_country, harv_df_country)
    
    # Calculate country metrics
    metrics = calculate_country_metrics(prod_df_country, harv_df_country, envelope_data, country_name)
    
    logger.info(f"    ✅ Envelope calculated: {len(envelope_data.get('disrupted_areas', []))} points")
    logger.info(f"    📊 Metrics: {metrics['total_production_mt']:.1f} Mt, {metrics['total_harvest_km2']:.0f} km², {metrics['vulnerability_pct']:.1f}% vulnerable")
    
    return envelope_data, config, grid_manager, metrics


def main():
    """Generate Figure 4: Country H-P Envelopes."""
    logger.info("=" * 60)
    logger.info("FIGURE 4: Country H-P Envelopes (4 panels)")
    logger.info("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        countries = ['USA', 'China', 'India', 'Brazil']
        
        # Generate individual figures for each country
        logger.info("Generating individual country H-P envelope figures...")
        for country in countries:
            logger.info(f"Processing {country}...")
            
            # Calculate country envelope and metrics
            country_code = COUNTRY_CODES[country]
            envelope_data, config, grid_manager, metrics = calculate_country_envelope(country, country_code)
            
            # For country plots, show envelopes only (no events)
            # Country-specific event filtering is complex and events may not be relevant
            events_data = pd.DataFrame()
            
            # Create visualizer
            visualizer = HPEnvelopeVisualizer(config)
            
            # Generate figure
            fig = visualizer.create_hp_envelope_plot(
                envelope_data, 
                events_data,
                save_path=None  # Don't save yet, we'll add metrics first
            )
            
            # Add metrics text box to the figure
            ax = fig.gca()
            add_country_metrics_textbox(ax, metrics)
            
            # Save with metrics
            fig.savefig(results_dir / f'figure4_{country.lower()}_individual.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            logger.info(f"  ✅ {country} saved with metrics")
        
        # Create 4-panel combined figure with real plots
        logger.info("Creating 4-panel combined figure with real plots...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Store envelope data for all countries
        all_data = {}
        for country in countries:
            logger.info(f"  Loading data for {country} panel...")
            country_code = COUNTRY_CODES[country]
            envelope_data, config, grid_manager, metrics = calculate_country_envelope(country, country_code)
            all_data[country] = {'envelope': envelope_data, 'config': config, 'metrics': metrics}
        
        # All countries share the same scale for comparison
        max_h = max([data['envelope']['lower_bound_harvest'][-1] for data in all_data.values()])
        max_p = max([data['envelope']['upper_bound_production'][-1] for data in all_data.values()])
        
        for i, country in enumerate(countries):
            ax = axes[i]
            data = all_data[country]
            envelope = data['envelope']
            
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
            
            # Set scale (shared across all countries)
            ax.set_yscale('log')
            ax.set_xlim([2, 6.5])
            ax.set_ylim([1e10, max_p * 1.2])
            
            # Labels and title
            ax.set_xlabel(r'Magnitude $M_D = \log_{10}(A_H)$ [km²]', fontsize=14)
            ax.set_ylabel('Production Loss [kcal]', fontsize=14)
            ax.set_title(country, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add metrics text box
            metrics = data['metrics']
            add_country_metrics_textbox(ax, metrics)
            
            # Legend only on first panel
            if i == 0:
                ax.legend(loc='lower right', fontsize=11)
        
        plt.tight_layout(pad=2.0)
        output_path = results_dir / 'figure4_country_envelopes.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("")
        logger.info("🎉 SUCCESS!")
        logger.info(f"✅ Individual figures saved: figure4_{{country}}_individual.png")
        logger.info(f"✅ Combined figure saved: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Verification script for state/province level event filtering."""

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_event_data_from_excel(config: Config) -> dict:
    """
    Load event data from Excel files.
    
    Returns:
        Dictionary with event definitions
    """
    file_paths = config.get_file_paths()
    
    # Load country-level events
    country_file = file_paths['disruption_country']
    state_file = file_paths['disruption_state']
    
    if not country_file.exists():
        logger.error(f"Country disruption file not found: {country_file}")
        return {}
    
    logger.info(f"Loading country-level events from {country_file}")
    
    # Read all sheets from the Excel file
    country_sheets = pd.read_excel(country_file, sheet_name=None)
    logger.info(f"Found {len(country_sheets)} sheets in country file")
    
    # Load state-level events if available
    state_sheets = {}
    if state_file.exists():
        logger.info(f"Loading state-level events from {state_file}")
        state_sheets = pd.read_excel(state_file, sheet_name=None)
        logger.info(f"Found {len(state_sheets)} sheets in state file")
    
    # Process events
    events_definitions = {}
    
    for sheet_name, df in country_sheets.items():
        if sheet_name.startswith('Sheet') or len(df) == 0:
            continue
        
        logger.info(f"\nProcessing event: {sheet_name}")
        logger.info(f"  Country sheet shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        
        # Parse country data (skip header row)
        data_df = df.iloc[1:] if len(df) > 1 else df
        
        country_codes = []
        state_flags = []
        country_names = []
        
        for _, row in data_df.iterrows():
            if len(row) >= 3:
                country_name = row.iloc[0] if pd.notna(row.iloc[0]) else ""
                country_code = row.iloc[1] if pd.notna(row.iloc[1]) else 0
                state_flag = row.iloc[2] if pd.notna(row.iloc[2]) else 0
                
                if country_code and country_code != 0:
                    country_names.append(str(country_name))
                    country_codes.append(float(country_code))
                    state_flags.append(float(state_flag))
        
        # Parse state data if available
        state_codes = []
        state_names = []
        
        if sheet_name in state_sheets:
            state_df = state_sheets[sheet_name]
            logger.info(f"  State sheet shape: {state_df.shape}")
            
            state_data_df = state_df.iloc[1:] if len(state_df) > 1 else state_df
            
            for _, row in state_data_df.iterrows():
                if len(row) >= 2:
                    state_name = row.iloc[0] if pd.notna(row.iloc[0]) else ""
                    state_code = row.iloc[1] if pd.notna(row.iloc[1]) else 0
                    
                    if state_name or state_code:
                        state_names.append(str(state_name))
                        state_codes.append(float(state_code) if pd.notna(state_code) else 0)
        
        events_definitions[sheet_name] = {
            'country_names': country_names,
            'country_codes': country_codes,
            'state_flags': state_flags,
            'state_codes': state_codes,
            'state_names': state_names
        }
        
        logger.info(f"  Countries: {len(country_codes)}")
        logger.info(f"  State-level countries: {sum(1 for f in state_flags if f == 1)}")
        logger.info(f"  States/provinces: {len(state_codes)}")
    
    return events_definitions


def verify_state_level_event(
    event_name: str,
    event_data: dict,
    event_calculator: EventCalculator,
    spatial_mapper: SpatialMapper,
    grid_manager: GridDataManager
) -> dict:
    """
    Verify state-level filtering for a specific event.
    
    Returns:
        Dictionary with verification results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Verifying event: {event_name}")
    logger.info(f"{'='*70}")
    
    results = {
        'event_name': event_name,
        'has_state_data': any(f == 1 for f in event_data.get('state_flags', [])),
        'countries': [],
        'verification_passed': True,
        'issues': []
    }
    
    # Process each country
    for i, country_code in enumerate(event_data.get('country_codes', [])):
        state_flag = event_data['state_flags'][i] if i < len(event_data['state_flags']) else 0
        country_name = event_data['country_names'][i] if i < len(event_data['country_names']) else 'Unknown'
        
        # Get ISO3 code
        iso3_code = spatial_mapper.get_iso3_from_country_code(country_code, code_system='GDAM ')
        
        logger.info(f"\nCountry: {country_name} (GDAM: {country_code}, ISO3: {iso3_code})")
        logger.info(f"  State-level processing: {'Yes' if state_flag == 1 else 'No'}")
        
        country_result = {
            'name': country_name,
            'code': country_code,
            'iso3': iso3_code,
            'state_level': state_flag == 1,
            'grid_cells': 0,
            'states_matched': []
        }
        
        if state_flag == 1:
            # State-level processing
            state_codes = event_data.get('state_codes', [])
            state_names = event_data.get('state_names', [])
            
            logger.info(f"  States to match: {len(state_codes)}")
            for j, state_name in enumerate(state_names[:5]):  # Show first 5
                logger.info(f"    - {state_name}")
            if len(state_names) > 5:
                logger.info(f"    ... and {len(state_names) - 5} more")
            
            # Get grid cells for states
            try:
                production_cells, harvest_cells = spatial_mapper.get_state_grid_cells_dataframe(
                    country_code, state_codes, code_system='GDAM '
                )
                
                country_result['grid_cells'] = len(production_cells)
                
                if len(production_cells) > 0:
                    # Check which states were matched
                    matched_states = production_cells['ADM1_NAME'].unique().tolist()
                    country_result['states_matched'] = matched_states
                    
                    logger.info(f"  ✓ Matched {len(production_cells)} grid cells")
                    logger.info(f"  ✓ Matched states: {matched_states[:5]}")
                    if len(matched_states) > 5:
                        logger.info(f"    ... and {len(matched_states) - 5} more")
                else:
                    logger.warning(f"  ⚠ No grid cells matched for states")
                    results['issues'].append(
                        f"{country_name}: No grid cells matched for {len(state_codes)} states"
                    )
                    results['verification_passed'] = False
                    
            except Exception as e:
                logger.error(f"  ✗ Error processing states: {e}")
                results['issues'].append(f"{country_name}: Error - {e}")
                results['verification_passed'] = False
        else:
            # Country-level processing
            try:
                production_cells, harvest_cells = grid_manager.get_grid_cells_by_iso3(iso3_code)
                country_result['grid_cells'] = len(production_cells)
                
                if len(production_cells) > 0:
                    logger.info(f"  ✓ Matched {len(production_cells)} grid cells (country-level)")
                else:
                    logger.warning(f"  ⚠ No grid cells found for country")
                    results['issues'].append(f"{country_name}: No grid cells found")
                    
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                results['issues'].append(f"{country_name}: Error - {e}")
        
        results['countries'].append(country_result)
    
    return results


def generate_verification_map(
    event_name: str,
    verification_results: dict,
    grid_manager: GridDataManager,
    spatial_mapper: SpatialMapper,
    output_dir: Path
):
    """
    Generate a map showing the affected regions for verification.
    """
    logger.info(f"\nGenerating verification map for {event_name}...")
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Collect all grid cells for this event
    all_production_cells = []
    
    for country in verification_results['countries']:
        iso3 = country['iso3']
        if not iso3:
            continue
        
        if country['state_level']:
            # Get state-level cells (would need to re-query)
            # For now, just mark that it's state-level
            pass
        else:
            # Get country-level cells
            try:
                prod_cells, _ = grid_manager.get_grid_cells_by_iso3(iso3)
                if len(prod_cells) > 0:
                    all_production_cells.append(prod_cells)
            except:
                pass
    
    if all_production_cells:
        combined_cells = pd.concat(all_production_cells, ignore_index=True)
        
        # Plot grid cells
        ax.scatter(
            combined_cells['x'], 
            combined_cells['y'],
            c='red',
            s=1,
            alpha=0.5,
            label='Affected grid cells'
        )
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Event: {event_name}\nAffected Regions ({len(combined_cells)} grid cells)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save map
        output_file = output_dir / f'verification_map_{event_name.replace(" ", "_")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved map to {output_file}")
        plt.close()
    else:
        logger.warning(f"  No grid cells to plot for {event_name}")
        plt.close()


def main():
    """Main verification workflow."""
    
    logger.info("="*70)
    logger.info("State/Province Level Event Verification")
    logger.info("="*70)
    
    # 1. Initialize system
    logger.info("\n1. Initializing system...")
    root_dir = Path.cwd()
    config = Config(crop_type='wheat', root_dir=root_dir, spam_version='2020')
    
    grid_manager = GridDataManager(config)
    grid_manager.load_spam_data()
    
    spatial_mapper = SpatialMapper(config, grid_manager)
    spatial_mapper.load_country_codes_mapping()
    
    event_calculator = EventCalculator(config, grid_manager, spatial_mapper)
    
    # 2. Load event data from Excel files
    logger.info("\n2. Loading event data from Excel files...")
    events_definitions = load_event_data_from_excel(config)
    
    if not events_definitions:
        logger.error("No event data loaded. Exiting.")
        return
    
    logger.info(f"\nLoaded {len(events_definitions)} events")
    
    # 3. Identify events with state-level data
    state_level_events = {
        name: data for name, data in events_definitions.items()
        if any(f == 1 for f in data.get('state_flags', []))
    }
    
    logger.info(f"\nFound {len(state_level_events)} events with state-level data:")
    for event_name in state_level_events.keys():
        logger.info(f"  - {event_name}")
    
    # 4. Verify each state-level event
    logger.info("\n3. Verifying state-level events...")
    
    verification_results = []
    output_dir = Path('verification_output')
    output_dir.mkdir(exist_ok=True)
    
    for event_name, event_data in state_level_events.items():
        result = verify_state_level_event(
            event_name,
            event_data,
            event_calculator,
            spatial_mapper,
            grid_manager
        )
        verification_results.append(result)
        
        # Generate verification map
        try:
            generate_verification_map(
                event_name,
                result,
                grid_manager,
                spatial_mapper,
                output_dir
            )
        except Exception as e:
            logger.error(f"Error generating map for {event_name}: {e}")
    
    # 5. Generate summary report
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*70)
    
    total_events = len(verification_results)
    passed = sum(1 for r in verification_results if r['verification_passed'])
    failed = total_events - passed
    
    logger.info(f"\nTotal state-level events verified: {total_events}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.info("\nEvents with issues:")
        for result in verification_results:
            if not result['verification_passed']:
                logger.info(f"\n  {result['event_name']}:")
                for issue in result['issues']:
                    logger.info(f"    - {issue}")
    
    # 6. Save detailed results to CSV
    summary_data = []
    for result in verification_results:
        for country in result['countries']:
            summary_data.append({
                'event': result['event_name'],
                'country': country['name'],
                'country_code': country['code'],
                'iso3': country['iso3'],
                'state_level': country['state_level'],
                'grid_cells': country['grid_cells'],
                'states_matched': ', '.join(country['states_matched'][:5]) if country['states_matched'] else ''
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'state_level_verification_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved detailed summary to {summary_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Verification complete!")
    logger.info("="*70)


if __name__ == '__main__':
    main()

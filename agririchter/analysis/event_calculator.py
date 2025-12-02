"""Event Calculator for historical agricultural disruption events."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from ..core.config import Config
from ..data.grid_manager import GridDataManager
from ..data.spatial_mapper import SpatialMapper


logger = logging.getLogger(__name__)


class EventCalculator:
    """
    Calculates production losses and magnitudes for historical events.
    
    Processes event definitions to compute actual harvest area losses
    and production losses by mapping geographic regions to SPAM grid cells.
    """
    
    def __init__(
        self, 
        config: Config, 
        grid_manager: GridDataManager, 
        spatial_mapper: SpatialMapper
    ):
        """
        Initialize EventCalculator.
        
        Args:
            config: Configuration object with crop parameters
            grid_manager: GridDataManager for querying grid cells
            spatial_mapper: SpatialMapper for geographic mapping
        """
        self.config = config
        self.grid_manager = grid_manager
        self.spatial_mapper = spatial_mapper
        
        # Ensure dependencies are initialized
        if not self.grid_manager.is_loaded():
            logger.info("Grid data not loaded, loading now...")
            self.grid_manager.load_spam_data()
        
        if self.spatial_mapper.country_codes_mapping is None:
            logger.info("Country codes mapping not loaded, loading now...")
            self.spatial_mapper.load_country_codes_mapping()
        
        # Storage for event results
        self.event_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"Initialized EventCalculator for {config.crop_type} "
            f"(crop indices: {config.get_crop_indices()})"
        )
    
    def calculate_single_event(
        self, 
        event_name: str, 
        event_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate losses for a single historical event.
        
        Processes event definition to determine affected regions and
        aggregates losses across all affected countries/states.
        
        Args:
            event_name: Name of the event
            event_data: Dictionary with event definition containing:
                - country_codes: List of numeric country codes
                - state_flags: List of flags (0=country-level, 1=state-level)
                - state_codes: List of state codes (for state-level events)
        
        Returns:
            Dictionary with:
                - harvest_area_loss_ha: Total harvest area loss in hectares
                - production_loss_kcal: Total production loss in kcal
                - affected_countries: Number of affected countries
                - affected_states: Number of affected states
                - grid_cells_count: Total grid cells affected
        """
        logger.info(f"Calculating losses for event: {event_name}")
        
        # Initialize totals
        total_harvest_loss_ha = 0.0
        total_production_loss_kcal = 0.0
        total_grid_cells = 0
        affected_countries_count = 0
        affected_states_count = 0
        
        # Get event parameters
        country_codes = event_data.get('country_codes', [])
        state_flags = event_data.get('state_flags', [])
        state_codes = event_data.get('state_codes', [])
        
        if not country_codes:
            logger.warning(f"Event {event_name} has no country codes")
            return {
                'harvest_area_loss_ha': 0.0,
                'production_loss_kcal': 0.0,
                'affected_countries': 0,
                'affected_states': 0,
                'grid_cells_count': 0
            }
        
        # Process each affected country
        for i, country_code in enumerate(country_codes):
            # Skip invalid country codes
            if pd.isna(country_code) or country_code == 0:
                continue
            
            # Determine if state-level processing is needed
            state_flag = state_flags[i] if i < len(state_flags) else 0
            
            if state_flag == 1:
                # State-level processing
                logger.debug(f"  Processing country {country_code} at state level")
                harvest_loss, production_loss, grid_cells = self.calculate_state_level_loss(
                    country_code, state_codes
                )
                affected_states_count += len(state_codes) if state_codes else 0
            else:
                # Country-level processing
                logger.debug(f"  Processing country {country_code} at country level")
                harvest_loss, production_loss, grid_cells = self.calculate_country_level_loss(
                    country_code
                )
            
            # Aggregate losses
            total_harvest_loss_ha += harvest_loss
            total_production_loss_kcal += production_loss
            total_grid_cells += grid_cells
            
            if harvest_loss > 0 or production_loss > 0:
                affected_countries_count += 1
        
        # Store results
        result = {
            'harvest_area_loss_ha': total_harvest_loss_ha,
            'production_loss_kcal': total_production_loss_kcal,
            'affected_countries': affected_countries_count,
            'affected_states': affected_states_count,
            'grid_cells_count': total_grid_cells
        }
        
        self.event_results[event_name] = result
        
        logger.info(
            f"Event {event_name}: {total_harvest_loss_ha:.2f} ha, "
            f"{total_production_loss_kcal:.2e} kcal, "
            f"{affected_countries_count} countries, {total_grid_cells} grid cells"
        )
        
        return result
    
    def calculate_country_level_loss(
        self, 
        country_code: float
    ) -> Tuple[float, float, int]:
        """
        Calculate losses at country level.
        
        Maps country code to FIPS, queries grid cells, and aggregates
        production and harvest area for selected crops.
        
        Args:
            country_code: Numeric country code (GDAM system)
        
        Returns:
            Tuple of (harvest_area_loss_ha, production_loss_kcal, grid_cells_count)
        """
        # Map country code to FIPS using SpatialMapper (SPAM uses FIPS codes)
        fips_code = self.spatial_mapper.get_fips_from_country_code(
            country_code, code_system='GDAM '
        )
        
        if fips_code is None:
            country_name = self.spatial_mapper.get_country_name_from_code(
                country_code, code_system='GDAM '
            )
            logger.warning(
                f"Could not map country code {country_code} "
                f"({country_name or 'Unknown'}) to FIPS"
            )
            return 0.0, 0.0, 0
        
        # Query GridDataManager for country's grid cells (using FIPS code)
        try:
            production_cells, harvest_area_cells = self.grid_manager.get_grid_cells_by_iso3(
                fips_code
            )
        except Exception as e:
            logger.error(f"Error querying grid cells for FIPS {fips_code}: {e}")
            return 0.0, 0.0, 0
        
        if len(production_cells) == 0:
            logger.warning(f"No grid cells found for FIPS: {fips_code}")
            return 0.0, 0.0, 0
        
        # Get crop indices for current analysis
        crop_indices = self.config.get_crop_indices()
        
        # Sum production and harvest area for selected crops
        harvest_area_ha = self.grid_manager.get_crop_harvest_area(
            harvest_area_cells, crop_indices
        )
        
        # Get production in metric tons, then convert to kcal
        production_mt = self.grid_manager.get_crop_production(
            production_cells, crop_indices, convert_to_kcal=False
        )
        
        # Convert units: metric tons → grams → kcal
        unit_conversions = self.config.get_unit_conversions()
        caloric_content = self.config.get_caloric_content()
        
        grams = production_mt * unit_conversions['grams_per_metric_ton']
        production_kcal = grams * caloric_content
        
        grid_cells_count = len(production_cells)
        
        logger.debug(
            f"Country {fips_code}: {harvest_area_ha:.2f} ha, "
            f"{production_mt:.2f} MT = {production_kcal:.2e} kcal, "
            f"{grid_cells_count} cells"
        )
        
        return harvest_area_ha, production_kcal, grid_cells_count
    
    def calculate_state_level_loss(
        self, 
        country_code: float, 
        state_codes: List[float]
    ) -> Tuple[float, float, int]:
        """
        Calculate losses at state/province level.
        
        Maps state codes to grid cells and aggregates losses across
        all affected states within the country.
        
        Args:
            country_code: Numeric country code (GDAM system)
            state_codes: List of numeric state codes or state names
        
        Returns:
            Tuple of (harvest_area_loss_ha, production_loss_kcal, grid_cells_count)
        """
        if not state_codes:
            logger.warning(
                f"No state codes provided for country {country_code}, "
                f"falling back to country-level calculation"
            )
            return self.calculate_country_level_loss(country_code)
        
        logger.debug(
            f"Calculating state-level losses for country {country_code}, "
            f"{len(state_codes)} states"
        )
        
        # Map state codes to grid cells using SpatialMapper
        try:
            production_ids, harvest_area_ids = self.spatial_mapper.map_state_to_grid_cells(
                country_code, state_codes, code_system='GDAM '
            )
        except Exception as e:
            logger.error(
                f"Error mapping states to grid cells for country {country_code}: {e}"
            )
            return 0.0, 0.0, 0
        
        if not production_ids:
            country_name = self.spatial_mapper.get_country_name_from_code(
                country_code, code_system='GDAM '
            )
            logger.warning(
                f"No grid cells found for states {state_codes} in "
                f"country {country_code} ({country_name or 'Unknown'})"
            )
            return 0.0, 0.0, 0
        
        # Get full DataFrames for the state grid cells
        production_cells, harvest_area_cells = self.spatial_mapper.get_state_grid_cells_dataframe(
            country_code, state_codes, code_system='GDAM '
        )
        
        if len(production_cells) == 0:
            logger.warning(f"No production data found for states {state_codes}")
            return 0.0, 0.0, 0
        
        # Get crop indices for current analysis
        crop_indices = self.config.get_crop_indices()
        
        # Sum production and harvest area for selected crops
        harvest_area_ha = self.grid_manager.get_crop_harvest_area(
            harvest_area_cells, crop_indices
        )
        
        # Get production in metric tons, then convert to kcal
        production_mt = self.grid_manager.get_crop_production(
            production_cells, crop_indices, convert_to_kcal=False
        )
        
        # Convert units: metric tons → grams → kcal
        unit_conversions = self.config.get_unit_conversions()
        caloric_content = self.config.get_caloric_content()
        
        grams = production_mt * unit_conversions['grams_per_metric_ton']
        production_kcal = grams * caloric_content
        
        grid_cells_count = len(production_cells)
        
        logger.debug(
            f"States in country {country_code}: {harvest_area_ha:.2f} ha, "
            f"{production_mt:.2f} MT = {production_kcal:.2e} kcal, "
            f"{grid_cells_count} cells across {len(state_codes)} states"
        )
        
        return harvest_area_ha, production_kcal, grid_cells_count
    
    def calculate_magnitude(self, harvest_area_ha: float) -> float:
        """
        Calculate AgriRichter magnitude for an event.
        
        Applies the formula: M_D = log10(disrupted harvest area in km²)
        
        Args:
            harvest_area_ha: Harvest area loss in hectares
        
        Returns:
            Magnitude value (log10 scale), or NaN if harvest area is zero
        """
        # Convert hectares to km² (multiply by 0.01)
        unit_conversions = self.config.get_unit_conversions()
        harvest_area_km2 = harvest_area_ha * unit_conversions['hectares_to_km2']
        
        # Handle zero values by replacing with NaN
        if harvest_area_km2 <= 0:
            logger.debug(
                f"Harvest area is zero or negative ({harvest_area_km2:.2f} km²), "
                f"returning NaN for magnitude"
            )
            return np.nan
        
        # Apply log10 transformation
        magnitude = np.log10(harvest_area_km2)
        
        # Validate magnitude range (typically 2-7 for historical events)
        if magnitude < 1 or magnitude > 8:
            logger.warning(
                f"Magnitude {magnitude:.2f} is outside typical range [2, 7]. "
                f"Harvest area: {harvest_area_ha:.2f} ha = {harvest_area_km2:.2f} km²"
            )
        
        logger.debug(
            f"Magnitude calculation: {harvest_area_ha:.2f} ha = "
            f"{harvest_area_km2:.2f} km² → M_D = {magnitude:.2f}"
        )
        
        return magnitude
    
    def _get_event_type(self, event_name: str) -> str:
        """Get event type from food_disruptions.csv."""
        try:
            # Load the CSV file
            csv_path = self.config.root_dir / 'ancillary' / 'food_disruptions.csv'
            if csv_path.exists():
                events_df = pd.read_csv(csv_path)
                # Find matching event
                match = events_df[events_df['event_name'] == event_name]
                if not match.empty:
                    return match.iloc[0]['event_type']
            return 'Unknown'
        except Exception:
            return 'Unknown'
    
    def calculate_all_events(
        self, 
        events_definitions: Dict[str, Dict[str, Any]],
        limit_to_country_code: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculate losses for all historical events.
        
        Processes all 21 events sequentially with progress logging,
        handling errors gracefully to continue with remaining events.
        
        Args:
            events_definitions: Dictionary mapping event names to event data
                Each event_data should contain:
                - country_codes: List of numeric country codes
                - state_flags: List of flags (0=country, 1=state level)
                - state_codes: List of state codes (for state-level events)
            limit_to_country_code: Optional GDAM country code to filter results.
                If provided, only losses for this specific country will be calculated.
        
        Returns:
            DataFrame with columns:
                - event_name: Name of the event
                - harvest_area_loss_ha: Harvest area loss in hectares
                - production_loss_kcal: Production loss in kcal
                - magnitude: AgriRichter magnitude (log10 scale)
                - affected_countries: Number of affected countries
                - affected_states: Number of affected states
                - grid_cells_count: Total grid cells affected
        """
        if limit_to_country_code is not None:
            logger.info(f"Calculating losses strictly for country code: {limit_to_country_code}")
        else:
            logger.info(f"Calculating losses for {len(events_definitions)} historical events (Global)")
        
        results_list = []
        total_events = len(events_definitions)
        
        for idx, (event_name, event_data) in enumerate(events_definitions.items(), 1):
            # If filtering by country, we need to modify the event_data on the fly
            if limit_to_country_code is not None:
                country_codes = event_data.get('country_codes', [])
                
                # Check if target country is in this event
                if limit_to_country_code not in country_codes:
                    # Country not affected by this event -> 0 loss
                    logger.debug(f"Skipping event {event_name} (target country not affected)")
                    results_list.append({
                        'event_name': event_name,
                        'harvest_area_loss_ha': 0.0,
                        'production_loss_kcal': 0.0,
                        'magnitude': np.nan,
                        'affected_countries': 0,
                        'affected_states': 0,
                        'grid_cells_count': 0,
                        'event_type': self._get_event_type(event_name)
                    })
                    continue
                
                # Create filtered event definition containing ONLY the target country
                # We need to find the index of the country code to get corresponding state flags/codes
                indices = [i for i, x in enumerate(country_codes) if x == limit_to_country_code]
                
                filtered_event_data = {
                    'country_codes': [country_codes[i] for i in indices],
                    'state_flags': [event_data.get('state_flags', [])[i] for i in indices],
                    # State codes are a bit trickier as they are a flat list corresponding to state_flags=1
                    # But the current structure seems to be parallel lists? 
                    # Let's check process_event_sheets output structure again.
                    # It seems country_codes and state_flags are parallel.
                    # State codes might be handled differently in process_event_sheets.
                    # Let's assume for now we pass the whole state_codes list and calculate_single_event handles it?
                    # Actually calculate_single_event calls calculate_state_level_loss which takes ALL state_codes.
                    # This implies state_codes are specific to the event, not per country in the data structure?
                    # Reviewing process_event_sheets: it returns 'state_codes': [list of all state codes for event]
                    # And calculate_state_level_loss iterates through ALL state_codes and maps them to the country.
                    # So passing the full list is safe because map_state_to_grid_cells filters by country ISO3.
                    'state_codes': event_data.get('state_codes', [])
                }
                
                # Use filtered data
                event_data_to_use = filtered_event_data
            else:
                event_data_to_use = event_data

            logger.info(f"Processing event {idx}/{total_events}: {event_name}")
            
            try:
                # Calculate single event losses
                result = self.calculate_single_event(event_name, event_data_to_use)
                
                # Calculate magnitude
                magnitude = self.calculate_magnitude(result['harvest_area_loss_ha'])
                
                # Load event type from food_disruptions.csv
                event_type = self._get_event_type(event_name)
                
                # Compile results
                event_result = {
                    'event_name': event_name,
                    'harvest_area_loss_ha': result['harvest_area_loss_ha'],
                    'production_loss_kcal': result['production_loss_kcal'],
                    'magnitude': magnitude,
                    'affected_countries': result['affected_countries'],
                    'affected_states': result['affected_states'],
                    'grid_cells_count': result['grid_cells_count'],
                    'event_type': event_type
                }
                
                results_list.append(event_result)
                
                # Log progress
                if result['harvest_area_loss_ha'] > 0:
                    logger.info(
                        f"  ✓ {event_name}: {result['harvest_area_loss_ha']:.2f} ha, "
                        f"{result['production_loss_kcal']:.2e} kcal, M={magnitude:.2f}"
                    )
                else:
                    logger.debug( # Demoted to debug to reduce noise when filtering
                        f"  ⚠ {event_name}: No losses calculated (zero grid cells matched)"
                    )
                
            except Exception as e:
                logger.error(
                    f"  ✗ Error processing event {event_name}: {e}", 
                    exc_info=True
                )
                # Add placeholder result to continue with other events
                results_list.append({
                    'event_name': event_name,
                    'harvest_area_loss_ha': 0.0,
                    'production_loss_kcal': 0.0,
                    'magnitude': np.nan,
                    'affected_countries': 0,
                    'affected_states': 0,
                    'grid_cells_count': 0
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Log summary statistics
        total_harvest_loss = results_df['harvest_area_loss_ha'].sum()
        total_production_loss = results_df['production_loss_kcal'].sum()
        events_with_data = (results_df['harvest_area_loss_ha'] > 0).sum()
        # Avoid mean of empty slice warning
        if len(results_df['magnitude'].dropna()) > 0:
            avg_magnitude = results_df['magnitude'].mean()
        else:
            avg_magnitude = 0.0
        
        logger.info(
            f"\nBatch processing complete: {events_with_data}/{total_events} events with data"
        )
        logger.info(f"Total harvest area loss: {total_harvest_loss:.2f} ha")
        logger.info(f"Total production loss: {total_production_loss:.2e} kcal")
        logger.info(f"Average magnitude: {avg_magnitude:.2f}")
        
        return results_df
    
    def validate_event_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate event results for data quality and reasonableness.
        
        Checks:
        - No event has losses exceeding global production
        - All events have non-zero losses (or documents why zero)
        - Magnitude ranges are reasonable (typically 2-7)
        - Flags events with suspicious values for review
        
        Args:
            results_df: DataFrame with calculated event results
        
        Returns:
            Dictionary with validation results:
                - valid: Overall validation status
                - errors: List of error messages
                - warnings: List of warning messages
                - metrics: Dictionary of validation metrics
                - suspicious_events: List of event names flagged for review
        """
        logger.info("Validating event results...")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'suspicious_events': []
        }
        
        # 1. Calculate global production for comparison
        crop_indices = self.config.get_crop_indices()
        production_df = self.grid_manager.get_production_data()
        harvest_df = self.grid_manager.get_harvest_area_data()
        
        global_production_kcal = self.grid_manager.get_crop_production(
            production_df, crop_indices, convert_to_kcal=True
        )
        global_harvest_area_ha = self.grid_manager.get_crop_harvest_area(
            harvest_df, crop_indices
        )
        
        validation_results['metrics']['global_production_kcal'] = global_production_kcal
        validation_results['metrics']['global_harvest_area_ha'] = global_harvest_area_ha
        
        logger.info(
            f"Global totals: {global_production_kcal:.2e} kcal, "
            f"{global_harvest_area_ha:.2f} ha"
        )
        
        # 2. Check no event exceeds global production
        for _, row in results_df.iterrows():
            event_name = row['event_name']
            production_loss = row['production_loss_kcal']
            harvest_loss = row['harvest_area_loss_ha']
            
            if production_loss > global_production_kcal:
                validation_results['errors'].append(
                    f"Event {event_name} production loss ({production_loss:.2e} kcal) "
                    f"exceeds global production ({global_production_kcal:.2e} kcal)"
                )
                validation_results['valid'] = False
                validation_results['suspicious_events'].append(event_name)
            
            if harvest_loss > global_harvest_area_ha:
                validation_results['errors'].append(
                    f"Event {event_name} harvest area loss ({harvest_loss:.2f} ha) "
                    f"exceeds global harvest area ({global_harvest_area_ha:.2f} ha)"
                )
                validation_results['valid'] = False
                validation_results['suspicious_events'].append(event_name)
        
        # 3. Check for zero losses
        zero_loss_events = results_df[results_df['harvest_area_loss_ha'] == 0]
        if len(zero_loss_events) > 0:
            validation_results['warnings'].append(
                f"{len(zero_loss_events)} events have zero losses: "
                f"{zero_loss_events['event_name'].tolist()}"
            )
            validation_results['metrics']['zero_loss_events'] = len(zero_loss_events)
        
        # 4. Validate magnitude ranges (typically 2-7 for historical events)
        valid_magnitudes = results_df['magnitude'].dropna()
        if len(valid_magnitudes) > 0:
            min_magnitude = valid_magnitudes.min()
            max_magnitude = valid_magnitudes.max()
            avg_magnitude = valid_magnitudes.mean()
            
            validation_results['metrics']['magnitude_range'] = (
                float(min_magnitude), float(max_magnitude)
            )
            validation_results['metrics']['avg_magnitude'] = float(avg_magnitude)
            
            # Flag events with unusual magnitudes
            for _, row in results_df.iterrows():
                magnitude = row['magnitude']
                event_name = row['event_name']
                
                if pd.notna(magnitude):
                    if magnitude < 2:
                        validation_results['warnings'].append(
                            f"Event {event_name} has unusually low magnitude: {magnitude:.2f}"
                        )
                        if event_name not in validation_results['suspicious_events']:
                            validation_results['suspicious_events'].append(event_name)
                    elif magnitude > 7:
                        validation_results['warnings'].append(
                            f"Event {event_name} has unusually high magnitude: {magnitude:.2f}"
                        )
                        if event_name not in validation_results['suspicious_events']:
                            validation_results['suspicious_events'].append(event_name)
        
        # 5. Check for NaN magnitudes
        nan_magnitudes = results_df['magnitude'].isna().sum()
        if nan_magnitudes > 0:
            validation_results['warnings'].append(
                f"{nan_magnitudes} events have NaN magnitudes (likely zero harvest area)"
            )
            validation_results['metrics']['nan_magnitudes'] = int(nan_magnitudes)
        
        # 6. Calculate percentage of global production affected
        total_production_loss = results_df['production_loss_kcal'].sum()
        total_harvest_loss = results_df['harvest_area_loss_ha'].sum()
        
        if global_production_kcal > 0:
            production_loss_pct = (total_production_loss / global_production_kcal) * 100
            validation_results['metrics']['total_production_loss_pct'] = production_loss_pct
            
            if production_loss_pct > 100:
                validation_results['errors'].append(
                    f"Total production loss ({production_loss_pct:.1f}%) exceeds 100% of global production"
                )
                validation_results['valid'] = False
        
        if global_harvest_area_ha > 0:
            harvest_loss_pct = (total_harvest_loss / global_harvest_area_ha) * 100
            validation_results['metrics']['total_harvest_loss_pct'] = harvest_loss_pct
            
            if harvest_loss_pct > 100:
                validation_results['errors'].append(
                    f"Total harvest area loss ({harvest_loss_pct:.1f}%) exceeds 100% of global harvest area"
                )
                validation_results['valid'] = False
        
        # 7. Summary statistics
        validation_results['metrics']['total_events'] = len(results_df)
        validation_results['metrics']['events_with_data'] = (
            results_df['harvest_area_loss_ha'] > 0
        ).sum()
        validation_results['metrics']['total_production_loss_kcal'] = total_production_loss
        validation_results['metrics']['total_harvest_loss_ha'] = total_harvest_loss
        
        # Log summary
        if validation_results['valid']:
            logger.info("Event results validation PASSED")
        else:
            logger.error(
                f"Event results validation FAILED with {len(validation_results['errors'])} errors"
            )
        
        if validation_results['warnings']:
            logger.warning(
                f"Event results validation has {len(validation_results['warnings'])} warnings"
            )
        
        if validation_results['suspicious_events']:
            logger.warning(
                f"Flagged {len(validation_results['suspicious_events'])} events for review: "
                f"{validation_results['suspicious_events']}"
            )
        
        return validation_results
    
    def generate_validation_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            results_df: DataFrame with calculated event results
        
        Returns:
            Formatted validation report string
        """
        validation_results = self.validate_event_results(results_df)
        
        report_lines = [
            "=" * 70,
            "EVENT RESULTS VALIDATION REPORT",
            "=" * 70,
            "",
            f"Status: {'PASSED' if validation_results['valid'] else 'FAILED'}",
            f"Crop Type: {self.config.crop_type}",
            f"SPAM Version: {self.config.get_spam_version()}",
            "",
            "SUMMARY METRICS:",
            f"  Total Events: {validation_results['metrics'].get('total_events', 0)}",
            f"  Events with Data: {validation_results['metrics'].get('events_with_data', 0)}",
            f"  Zero Loss Events: {validation_results['metrics'].get('zero_loss_events', 0)}",
            f"  NaN Magnitudes: {validation_results['metrics'].get('nan_magnitudes', 0)}",
            "",
            "GLOBAL COMPARISON:",
            f"  Global Production: {validation_results['metrics'].get('global_production_kcal', 0):.2e} kcal",
            f"  Global Harvest Area: {validation_results['metrics'].get('global_harvest_area_ha', 0):.2f} ha",
            f"  Total Production Loss: {validation_results['metrics'].get('total_production_loss_kcal', 0):.2e} kcal "
            f"({validation_results['metrics'].get('total_production_loss_pct', 0):.2f}%)",
            f"  Total Harvest Loss: {validation_results['metrics'].get('total_harvest_loss_ha', 0):.2f} ha "
            f"({validation_results['metrics'].get('total_harvest_loss_pct', 0):.2f}%)",
            "",
            "MAGNITUDE STATISTICS:",
        ]
        
        if 'magnitude_range' in validation_results['metrics']:
            min_mag, max_mag = validation_results['metrics']['magnitude_range']
            avg_mag = validation_results['metrics']['avg_magnitude']
            report_lines.extend([
                f"  Range: [{min_mag:.2f}, {max_mag:.2f}]",
                f"  Average: {avg_mag:.2f}",
                f"  Expected Range: [2.0, 7.0]",
            ])
        else:
            report_lines.append("  No valid magnitudes calculated")
        
        if validation_results['errors']:
            report_lines.extend([
                "",
                f"ERRORS ({len(validation_results['errors'])}):",
            ])
            for error in validation_results['errors']:
                report_lines.append(f"  ✗ {error}")
        
        if validation_results['warnings']:
            report_lines.extend([
                "",
                f"WARNINGS ({len(validation_results['warnings'])}):",
            ])
            for warning in validation_results['warnings']:
                report_lines.append(f"  ⚠ {warning}")
        
        if validation_results['suspicious_events']:
            report_lines.extend([
                "",
                f"SUSPICIOUS EVENTS ({len(validation_results['suspicious_events'])}):",
            ])
            for event_name in validation_results['suspicious_events']:
                event_row = results_df[results_df['event_name'] == event_name].iloc[0]
                report_lines.append(
                    f"  • {event_name}: {event_row['harvest_area_loss_ha']:.2f} ha, "
                    f"{event_row['production_loss_kcal']:.2e} kcal, M={event_row['magnitude']:.2f}"
                )
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def get_event_grid_cells(
        self,
        event_name: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all grid cells affected by an event for verification/visualization.
        
        Useful for debugging and creating verification maps.
        
        Args:
            event_name: Name of the event
            event_data: Dictionary with event definition
        
        Returns:
            Dictionary with:
                - production_cells: DataFrame of production grid cells
                - harvest_cells: DataFrame of harvest area grid cells
                - by_country: Dict mapping country names to their grid cells
        """
        logger.info(f"Retrieving grid cells for event: {event_name}")
        
        all_production_cells = []
        all_harvest_cells = []
        by_country = {}
        
        country_codes = event_data.get('country_codes', [])
        state_flags = event_data.get('state_flags', [])
        state_codes = event_data.get('state_codes', [])
        
        for i, country_code in enumerate(country_codes):
            if pd.isna(country_code) or country_code == 0:
                continue
            
            # Get country name for labeling
            country_name = self.spatial_mapper.get_country_name_from_code(
                country_code, code_system='GDAM '
            )
            
            state_flag = state_flags[i] if i < len(state_flags) else 0
            
            if state_flag == 1:
                # State-level
                production_cells, harvest_cells = self.spatial_mapper.get_state_grid_cells_dataframe(
                    country_code, state_codes, code_system='GDAM '
                )
            else:
                # Country-level
                iso3_code = self.spatial_mapper.get_iso3_from_country_code(
                    country_code, code_system='GDAM '
                )
                if iso3_code:
                    production_cells, harvest_cells = self.grid_manager.get_grid_cells_by_iso3(
                        iso3_code
                    )
                else:
                    production_cells = pd.DataFrame()
                    harvest_cells = pd.DataFrame()
            
            if len(production_cells) > 0:
                all_production_cells.append(production_cells)
                all_harvest_cells.append(harvest_cells)
                by_country[country_name or f"Country_{country_code}"] = {
                    'production': production_cells,
                    'harvest': harvest_cells
                }
        
        # Combine all cells
        combined_production = pd.concat(all_production_cells, ignore_index=True) if all_production_cells else pd.DataFrame()
        combined_harvest = pd.concat(all_harvest_cells, ignore_index=True) if all_harvest_cells else pd.DataFrame()
        
        logger.info(
            f"Retrieved {len(combined_production)} production cells, "
            f"{len(combined_harvest)} harvest cells for {event_name}"
        )
        
        return {
            'production_cells': combined_production,
            'harvest_cells': combined_harvest,
            'by_country': by_country
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EventCalculator(crop_type='{self.config.crop_type}', "
            f"events_calculated={len(self.event_results)})"
        )

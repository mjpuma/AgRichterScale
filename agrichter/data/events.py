"""Historical events data processing for AgRichter framework."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.constants import HISTORICAL_EVENTS


class EventsProcessor:
    """Processor for historical disruption events data."""
    
    def __init__(self, config: Config):
        """
        Initialize events processor.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agrichter.events')
        self.events_list = HISTORICAL_EVENTS
        
        # Initialize storage for processed events
        self.country_events = {}
        self.state_events = {}
        self.processed_events = {}
    
    def process_event_sheets(self, events_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Process historical events data from Excel sheets.
        
        Args:
            events_data: Dictionary with 'country' and 'state' sheet data
        
        Returns:
            Dictionary with processed event data
        """
        self.logger.info("Processing historical disruption events")
        
        country_sheets = events_data.get('country', {})
        state_sheets = events_data.get('state', {})
        
        processed_events = {}
        
        # Process each historical event
        for event_name in self.events_list:
            self.logger.debug(f"Processing event: {event_name}")
            
            event_data = {
                'name': event_name,
                'countries': [],
                'country_codes': [],
                'state_flags': [],
                'state_codes': [],
                'state_names': []
            }
            
            # Process country-level data
            if event_name in country_sheets:
                country_df = country_sheets[event_name]
                event_data.update(self._process_country_sheet(country_df))
            
            # Process state-level data
            if event_name in state_sheets:
                state_df = state_sheets[event_name]
                state_data = self._process_state_sheet(state_df)
                event_data.update(state_data)
            
            processed_events[event_name] = event_data
        
        self.logger.info(f"Processed {len(processed_events)} historical events")
        return processed_events
    
    def _process_country_sheet(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Process country-level event data sheet.
        
        Args:
            df: Country event DataFrame
        
        Returns:
            Dictionary with processed country data
        """
        if df.empty:
            return {'countries': [], 'country_codes': [], 'state_flags': []}
        
        # Expected columns: Country Name, Country Code, State Flag
        # Skip first row if it contains headers
        data_df = df.iloc[1:] if len(df) > 1 else df
        
        countries = []
        country_codes = []
        state_flags = []
        
        for _, row in data_df.iterrows():
            if len(row) >= 3:
                # Extract country name, code, and state flag
                country_name = row.iloc[0] if pd.notna(row.iloc[0]) else ""
                country_code = row.iloc[1] if pd.notna(row.iloc[1]) else 0
                state_flag = row.iloc[2] if pd.notna(row.iloc[2]) else 0
                
                if country_name or country_code:
                    countries.append(str(country_name))
                    country_codes.append(float(country_code) if pd.notna(country_code) else 0)
                    state_flags.append(float(state_flag) if pd.notna(state_flag) else 0)
        
        return {
            'countries': countries,
            'country_codes': country_codes,
            'state_flags': state_flags
        }
    
    def _process_state_sheet(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Process state/province-level event data sheet.
        
        Args:
            df: State event DataFrame
        
        Returns:
            Dictionary with processed state data
        """
        if df.empty:
            return {'state_names': [], 'state_codes': []}
        
        # Expected columns: ISO 3166-2, Country Code, Subdivision name
        # Note: We use Subdivision name (col 2) as the 'code' for lookup 
        # because SpatialMapper supports name matching on ADM1_NAME
        data_df = df  # Don't skip first row if headers are already parsed by read_excel
        
        state_names = []
        state_codes = []
        
        for _, row in data_df.iterrows():
            if len(row) >= 3:
                # ISO code (e.g. US.CO)
                iso_sub = row.iloc[0] if pd.notna(row.iloc[0]) else ""
                # Country code (e.g. 240) - redundant but present
                country_code = row.iloc[1] if pd.notna(row.iloc[1]) else 0
                # Subdivision name (e.g. Colorado) - CRITICAL for matching
                subdivision_name = row.iloc[2] if pd.notna(row.iloc[2]) else ""
                
                if subdivision_name:
                    # We use the name as the 'code' because we match against names
                    state_names.append(str(subdivision_name))
                    state_codes.append(str(subdivision_name))
            elif len(row) >= 2:
                # Fallback for sheets with fewer columns (unlikely based on inspection)
                val = row.iloc[0]
                if pd.notna(val):
                    state_names.append(str(val))
                    state_codes.append(str(val))
        
        return {
            'state_names': state_names,
            'state_codes': state_codes
        }
    
    def calculate_event_losses(self, events: Dict[str, Any], 
                             production_df: pd.DataFrame, 
                             harvest_df: pd.DataFrame,
                             country_codes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate production and harvest area losses for each historical event.
        
        Args:
            events: Processed events data
            production_df: SPAM production data
            harvest_df: SPAM harvest area data
            country_codes_df: Country code mapping
        
        Returns:
            DataFrame with event losses
        """
        self.logger.info("Calculating historical event losses")
        
        # Get crop indices for current analysis
        crop_indices = self.config.get_crop_indices()
        
        # Initialize results
        event_results = []
        
        for event_name, event_data in events.items():
            self.logger.debug(f"Calculating losses for {event_name}")
            
            total_harvest_loss = 0.0
            total_production_loss = 0.0
            
            # Process each affected country
            for i, country_code in enumerate(event_data.get('country_codes', [])):
                if country_code == 0:
                    continue
                
                # Check if this country needs state-level processing
                state_flag = event_data.get('state_flags', [0])[i] if i < len(event_data.get('state_flags', [])) else 0
                
                if state_flag == 1:
                    # State-level processing
                    harvest_loss, production_loss = self._calculate_state_level_losses(
                        event_data, country_code, production_df, harvest_df, crop_indices
                    )
                else:
                    # Country-level processing
                    harvest_loss, production_loss = self._calculate_country_level_losses(
                        country_code, production_df, harvest_df, crop_indices, country_codes_df
                    )
                
                total_harvest_loss += harvest_loss
                total_production_loss += production_loss
            
            event_results.append({
                'event': event_name,
                'harvest_area_loss_ha': total_harvest_loss,
                'production_loss_kcal': total_production_loss,
                'affected_countries': len(event_data.get('country_codes', [])),
                'has_state_data': any(flag == 1 for flag in event_data.get('state_flags', []))
            })
        
        results_df = pd.DataFrame(event_results)
        self.logger.info(f"Calculated losses for {len(results_df)} events")
        
        return results_df
    
    def _calculate_country_level_losses(self, country_code: float, 
                                      production_df: pd.DataFrame,
                                      harvest_df: pd.DataFrame,
                                      crop_indices: List[int],
                                      country_codes_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate losses at country level.
        
        Args:
            country_code: Country code from events data
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
            crop_indices: List of crop indices to include
            country_codes_df: Country code mapping
        
        Returns:
            Tuple of (harvest_loss_ha, production_loss_kcal)
        """
        # Find ISO3 code for this country
        iso3_code = self._get_iso3_from_country_code(country_code, country_codes_df)
        
        if not iso3_code:
            self.logger.warning(f"Could not find ISO3 code for country code {country_code}")
            return 0.0, 0.0
        
        # Filter data for this country
        country_production = production_df[production_df['iso3'] == iso3_code]
        country_harvest = harvest_df[harvest_df['iso3'] == iso3_code]
        
        if country_production.empty or country_harvest.empty:
            self.logger.warning(f"No data found for country {iso3_code}")
            return 0.0, 0.0
        
        # Calculate losses for selected crops
        harvest_loss = 0.0
        production_loss = 0.0
        
        # Get crop column names (1-based indices converted to column names)
        crop_columns = [f"crop_{i}" for i in crop_indices]  # This needs to be mapped to actual column names
        
        # Map to actual SPAM column names
        spam_crop_cols = self._get_spam_crop_columns(crop_indices)
        
        for col in spam_crop_cols:
            if col in country_production.columns and col in country_harvest.columns:
                harvest_loss += country_harvest[col].sum()
                production_loss += country_production[col].sum()
        
        # Convert production to kcal
        caloric_content = self.config.get_caloric_content()
        unit_conversions = self.config.get_unit_conversions()
        
        production_loss_kcal = production_loss * unit_conversions['grams_per_metric_ton'] * caloric_content
        
        return harvest_loss, production_loss_kcal
    
    def _calculate_state_level_losses(self, event_data: Dict[str, Any], 
                                    country_code: float,
                                    production_df: pd.DataFrame,
                                    harvest_df: pd.DataFrame,
                                    crop_indices: List[int]) -> Tuple[float, float]:
        """
        Calculate losses at state/province level.
        
        Args:
            event_data: Event data dictionary
            country_code: Country code
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
            crop_indices: List of crop indices
        
        Returns:
            Tuple of (harvest_loss_ha, production_loss_kcal)
        """
        # For state-level processing, we need to match state codes
        # This is more complex and depends on the specific data structure
        
        harvest_loss = 0.0
        production_loss = 0.0
        
        state_codes = event_data.get('state_codes', [])
        
        for state_code in state_codes:
            if state_code == 0:
                continue
            
            # Filter data for this state (using name_adm1 column)
            state_production = production_df[production_df['name_adm1'] == state_code]
            state_harvest = harvest_df[harvest_df['name_adm1'] == state_code]
            
            if not state_production.empty and not state_harvest.empty:
                spam_crop_cols = self._get_spam_crop_columns(crop_indices)
                
                for col in spam_crop_cols:
                    if col in state_production.columns and col in state_harvest.columns:
                        harvest_loss += state_harvest[col].sum()
                        production_loss += state_production[col].sum()
        
        # Convert production to kcal
        caloric_content = self.config.get_caloric_content()
        unit_conversions = self.config.get_unit_conversions()
        
        production_loss_kcal = production_loss * unit_conversions['grams_per_metric_ton'] * caloric_content
        
        return harvest_loss, production_loss_kcal
    
    def _get_spam_crop_columns(self, crop_indices: List[int]) -> List[str]:
        """
        Convert crop indices to SPAM column names.
        
        Args:
            crop_indices: List of 1-based crop indices
        
        Returns:
            List of SPAM column names
        """
        # SPAM column mapping (1-based index to column name)
        spam_columns = [
            'whea_a', 'rice_a', 'maiz_a', 'barl_a', 'pmil_a', 'smil_a', 
            'sorg_a', 'ocer_a', 'pota_a', 'swpo_a', 'yams_a', 'cass_a', 
            'orts_a', 'bean_a', 'chic_a', 'cowp_a', 'pige_a', 'lent_a', 
            'opul_a', 'soyb_a', 'grou_a', 'cnut_a', 'oilp_a', 'sunf_a', 
            'rape_a', 'sesa_a', 'ooil_a', 'sugc_a', 'sugb_a', 'cott_a', 
            'ofib_a', 'acof_a', 'rcof_a', 'coco_a', 'teas_a', 'toba_a', 
            'bana_a', 'plnt_a', 'trof_a', 'temf_a', 'vege_a', 'rest_a'
        ]
        
        # Convert 1-based indices to 0-based and get column names
        return [spam_columns[i-1] for i in crop_indices if 1 <= i <= len(spam_columns)]
    
    def _get_iso3_from_country_code(self, country_code: float, 
                                   country_codes_df: pd.DataFrame) -> Optional[str]:
        """
        Get ISO3 code from country code using mapping table.
        
        Args:
            country_code: Numeric country code
            country_codes_df: Country codes mapping DataFrame
        
        Returns:
            ISO3 code string or None if not found
        """
        # Look up in the GDAM column (assuming this matches the event country codes)
        matching_rows = country_codes_df[country_codes_df['GDAM'] == country_code]
        
        if not matching_rows.empty:
            # Return the ISO3Alpha code
            iso3_code = matching_rows.iloc[0]['ISO3Alpha']
            return iso3_code if pd.notna(iso3_code) else None
        
        return None
    
    def create_events_summary(self, events_losses: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for historical events.
        
        Args:
            events_losses: DataFrame with calculated event losses
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_events': len(events_losses),
            'total_harvest_loss_ha': events_losses['harvest_area_loss_ha'].sum(),
            'total_production_loss_kcal': events_losses['production_loss_kcal'].sum(),
            'avg_harvest_loss_ha': events_losses['harvest_area_loss_ha'].mean(),
            'avg_production_loss_kcal': events_losses['production_loss_kcal'].mean(),
            'max_harvest_loss_event': events_losses.loc[events_losses['harvest_area_loss_ha'].idxmax(), 'event'],
            'max_production_loss_event': events_losses.loc[events_losses['production_loss_kcal'].idxmax(), 'event'],
            'events_with_state_data': events_losses['has_state_data'].sum()
        }
        
        # Add magnitude calculations
        unit_conversions = self.config.get_unit_conversions()
        events_losses['harvest_area_loss_km2'] = events_losses['harvest_area_loss_ha'] * unit_conversions['hectares_to_km2']
        events_losses['magnitude'] = np.log10(events_losses['harvest_area_loss_km2'].replace(0, np.nan))
        
        summary['avg_magnitude'] = events_losses['magnitude'].mean()
        summary['max_magnitude'] = events_losses['magnitude'].max()
        summary['min_magnitude'] = events_losses['magnitude'].min()
        
        return summary
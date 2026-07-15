"""Historical events data processing for AgRichter framework."""

import logging
from typing import Dict, List, Any
import pandas as pd

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
        
        # Expected columns: Country Name, Country Code, State Flag.
        # pd.read_excel already consumed the header row, so every row here is
        # data. (A former `iloc[1:]` skip silently dropped the FIRST country
        # of every multi-country event, e.g. Australia from ENSO 2015-16.)
        data_df = df
        
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

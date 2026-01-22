"""USDA PSD data loader and threshold calculator for AgRichter analysis."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class USDADataLoader:
    """Loader for USDA Production, Supply and Distribution (PSD) data."""
    
    def __init__(self, data_dir: Union[str, Path] = "USDAdata"):
        """
        Initialize USDA data loader.
        
        Args:
            data_dir: Directory containing USDA PSD CSV files
        """
        self.data_dir = Path(data_dir)
        # Point to the refreshed 2026 data files
        self.crop_files = {
            'wheat': self.data_dir / 'usda_psd_1961to2025_wheat.csv',
            'rice': self.data_dir / 'usda_psd_1961to2025_rice.csv',
            'maize': self.data_dir / 'usda_psd_1961to2025_maize.csv'
        }
        
        # Validate files exist
        self._validate_files()
    
    def _validate_files(self) -> None:
        """Validate that required USDA files exist."""
        missing_files = []
        for crop, file_path in self.crop_files.items():
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing USDA PSD files: {missing_files}"
            )
    
    def load_crop_data(self, crop: str, year_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        Load USDA PSD data for a specific crop.
        
        Args:
            crop: Crop type ('wheat', 'rice', 'maize')
            year_range: Optional tuple of (start_year, end_year) for filtering
        
        Returns:
            DataFrame with columns: Year, BeginningStocks, Consumption, EndingStocks, Production
        
        Raises:
            ValueError: If crop type is invalid
            FileNotFoundError: If data file doesn't exist
        """
        if crop not in self.crop_files:
            raise ValueError(f"Invalid crop '{crop}'. Valid options: {list(self.crop_files.keys())}")
        
        file_path = self.crop_files[crop]
        logger.info(f"Loading USDA PSD data for {crop} from {file_path}")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # The file format has:
        # Row 0: Commodity names
        # Row 1: Attribute names  
        # Row 2: Country (World)
        # Row 3+: Year data
        
        # Extract attribute names from row 1
        attributes = df.iloc[1].values
        
        # Create new dataframe with proper structure
        data_rows = df.iloc[3:].copy()  # Skip header rows
        
        # First column is year
        years = data_rows.iloc[:, 0].astype(int)
        
        # Create structured dataframe
        result_df = pd.DataFrame({
            'Year': years,
            'BeginningStocks': pd.to_numeric(data_rows.iloc[:, 1], errors='coerce'),
            'Consumption': pd.to_numeric(data_rows.iloc[:, 2], errors='coerce'),
            'EndingStocks': pd.to_numeric(data_rows.iloc[:, 3], errors='coerce'),
            'Production': pd.to_numeric(data_rows.iloc[:, 4], errors='coerce')
        })
        
        # Filter by year range if specified
        if year_range:
            start_year, end_year = year_range
            result_df = result_df[
                (result_df['Year'] >= start_year) & 
                (result_df['Year'] <= end_year)
            ].copy()
        
        # Remove any rows with all NaN values
        result_df = result_df.dropna(how='all', subset=['BeginningStocks', 'Consumption', 'EndingStocks', 'Production'])
        
        logger.info(f"Loaded {len(result_df)} years of data for {crop}")
        return result_df
    
    def get_mean_consumption(self, crop: str, year_range: Tuple[int, int] = (2019, 2021)) -> float:
        """
        Calculate mean annual consumption for a specified year range.
        
        Args:
            crop: Crop type ('wheat', 'rice', 'maize')
            year_range: Tuple of (start_year, end_year)
            
        Returns:
            Mean consumption in metric tons
        """
        try:
            df = self.load_crop_data(crop, year_range)
            return df['Consumption'].mean()
        except Exception as e:
            logger.warning(f"Failed to calculate mean consumption for {crop}: {e}")
            return 0.0

    def get_mean_stocks(self, crop: str, year_range: Tuple[int, int] = (2019, 2021)) -> float:
        """
        Calculate mean ending stocks for a specified year range.
        
        Args:
            crop: Crop type ('wheat', 'rice', 'maize')
            year_range: Tuple of (start_year, end_year)
            
        Returns:
            Mean ending stocks in metric tons
        """
        try:
            df = self.load_crop_data(crop, year_range)
            return df['EndingStocks'].mean()
        except Exception as e:
            logger.warning(f"Failed to calculate mean stocks for {crop}: {e}")
            return 0.0

    def load_all_crops(self, year_range: Optional[Tuple[int, int]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load USDA PSD data for all available crops.
        
        Args:
            year_range: Optional tuple of (start_year, end_year) for filtering
        
        Returns:
            Dictionary mapping crop names to DataFrames
        """
        all_data = {}
        for crop in self.crop_files.keys():
            try:
                all_data[crop] = self.load_crop_data(crop, year_range)
            except Exception as e:
                logger.warning(f"Failed to load data for {crop}: {e}")
        
        return all_data
    
    def calculate_sur(self, crop_data: pd.DataFrame) -> pd.Series:
        """
        Calculate Stock-to-Use Ratio (SUR) for crop data.
        
        SUR = Ending Stocks / Consumption
        
        Args:
            crop_data: DataFrame with Consumption and EndingStocks columns
        
        Returns:
            Series with SUR values
        """
        sur = crop_data['EndingStocks'] / crop_data['Consumption']
        return sur
    
    def get_crop_mapping(self) -> Dict[str, str]:
        """
        Get mapping from AgRichter crop types to USDA crop names.
        
        Returns:
            Dictionary mapping AgRichter names to USDA names
        """
        return {
            'wheat': 'wheat',
            'rice': 'rice', 
            'maize': 'maize',
            'allgrain': 'combined'  # Will need special handling
        }


class AgriPhaseThresholdCalculator:
    """Calculator for dynamic AgriPhase thresholds based on USDA data."""
    
    def __init__(self, usda_loader: USDADataLoader):
        """
        Initialize threshold calculator.
        
        Args:
            usda_loader: Initialized USDA data loader
        """
        self.usda_loader = usda_loader
        
        # IPC Phase definitions based on SUR percentiles
        self.ipc_phase_percentiles = {
            2: 15,  # Phase 2 (Stressed): 15th percentile
            3: 10,  # Phase 3 (Crisis): 10th percentile  
            4: 5,   # Phase 4 (Emergency): 5th percentile
            5: 2    # Phase 5 (Famine): 2nd percentile
        }
        
        # IPC Phase colors for visualization
        self.ipc_colors = {
            1: '#00FF00',  # Green - Minimal/None
            2: '#FFFF00',  # Yellow - Stressed  
            3: '#FFA500',  # Orange - Crisis
            4: '#FF0000',  # Red - Emergency
            5: '#800080'   # Purple - Famine
        }
    
    def calculate_sur_thresholds(self, crop: str, year_range: Optional[Tuple[int, int]] = None) -> Dict[int, float]:
        """
        Calculate SUR-based thresholds for IPC phases.
        
        Args:
            crop: Crop type ('wheat', 'rice', 'maize', 'allgrain')
            year_range: Optional year range for historical data
        
        Returns:
            Dictionary mapping IPC phase numbers to SUR threshold values
        """
        if crop == 'allgrain':
            # For allgrain, combine data from wheat, rice, and maize
            return self._calculate_combined_thresholds(year_range)
        else:
            # Single crop calculation
            crop_data = self.usda_loader.load_crop_data(crop, year_range)
            sur_values = self.usda_loader.calculate_sur(crop_data)
            
            thresholds = {}
            for phase, percentile in self.ipc_phase_percentiles.items():
                thresholds[phase] = np.percentile(sur_values.dropna(), percentile)
            
            logger.info(f"Calculated SUR thresholds for {crop}: {thresholds}")
            return thresholds
    
    def _calculate_combined_thresholds(self, year_range: Optional[Tuple[int, int]] = None) -> Dict[int, float]:
        """Calculate combined thresholds for allgrain using weighted average."""
        # Load data for all grain crops
        crops = ['wheat', 'rice', 'maize']
        all_sur_values = []
        
        for crop in crops:
            try:
                crop_data = self.usda_loader.load_crop_data(crop, year_range)
                sur_values = self.usda_loader.calculate_sur(crop_data)
                
                # Weight by production (use production as weight)
                weights = crop_data['Production'].values
                weighted_sur = sur_values * weights
                all_sur_values.extend(weighted_sur.dropna().tolist())
                
            except Exception as e:
                logger.warning(f"Failed to include {crop} in combined calculation: {e}")
        
        # Calculate percentiles from combined weighted SUR values
        thresholds = {}
        for phase, percentile in self.ipc_phase_percentiles.items():
            thresholds[phase] = np.percentile(all_sur_values, percentile)
        
        logger.info(f"Calculated combined SUR thresholds for allgrain: {thresholds}")
        return thresholds
    
    def convert_sur_to_production_thresholds(self, crop: str, sur_thresholds: Dict[int, float], 
                                           reference_year: int = 2020) -> Dict[str, float]:
        """
        Convert SUR thresholds to production loss thresholds in kcal.
        
        Args:
            crop: Crop type
            sur_thresholds: SUR threshold values by IPC phase
            reference_year: Reference year for production baseline
        
        Returns:
            Dictionary with threshold names (T1-T4) mapped to kcal values
        """
        # Handle allgrain case
        if crop == 'allgrain':
            # Use combined consumption from all grain crops
            reference_consumption = 0
            for grain_crop in ['wheat', 'rice', 'maize']:
                try:
                    crop_data = self.usda_loader.load_crop_data(grain_crop)
                    reference_data = crop_data[crop_data['Year'] == reference_year]
                    if reference_data.empty:
                        reference_data = crop_data.iloc[-1:]
                    reference_consumption += reference_data['Consumption'].iloc[0]
                except Exception as e:
                    logger.warning(f"Failed to include {grain_crop} in allgrain consumption: {e}")
        else:
            # Load reference year data for single crop
            crop_data = self.usda_loader.load_crop_data(crop)
            reference_data = crop_data[crop_data['Year'] == reference_year]
            
            if reference_data.empty:
                # Use most recent year if reference year not available
                reference_data = crop_data.iloc[-1:]
                logger.warning(f"Reference year {reference_year} not found, using {reference_data['Year'].iloc[0]}")
            
            reference_consumption = reference_data['Consumption'].iloc[0]
        
        # Convert SUR thresholds to production thresholds
        # Lower SUR = higher severity = more production loss
        # T1 corresponds to Phase 2, T2 to Phase 3, etc.
        phase_mapping = {2: 'T1', 3: 'T2', 4: 'T3', 5: 'T4'}
        
        # Use a simplified approach that maps SUR percentiles to production loss thresholds
        # Based on the original AgRichter scale values (1e14 to 1e16 kcal range)
        
        # Define base threshold values that scale with crop type
        base_thresholds = {
            'wheat': {'T1': 1.00E+14, 'T2': 5.75E+14, 'T3': 1.86E+15, 'T4': 3.00E+15},
            'rice': {'T1': 1.31E+14, 'T2': 3.16E+14, 'T3': 1.29E+15, 'T4': 2.13E+15},
            'maize': {'T1': 1.25E+14, 'T2': 6.50E+14, 'T3': 2.10E+15, 'T4': 3.50E+15},
            'allgrain': {'T1': 1.16E+14, 'T2': 1.56E+15, 'T3': 5.86E+15, 'T4': 9.86E+15}
        }
        
        # Get base values for this crop type
        if crop in base_thresholds:
            base_values = base_thresholds[crop]
        else:
            # Default to wheat values
            base_values = base_thresholds['wheat']
        
        # Calculate scaling factor based on SUR variability
        sur_values = list(sur_thresholds.values())
        sur_std = np.std(sur_values)
        sur_mean = np.mean(sur_values)
        
        # Scale thresholds based on SUR characteristics
        # Higher variability = more extreme thresholds
        scale_factor = 1.0 + (sur_std / sur_mean)
        
        production_thresholds = {}
        for phase, threshold_name in phase_mapping.items():
            if phase in sur_thresholds and threshold_name in base_values:
                # Apply scaling to base threshold
                scaled_threshold = base_values[threshold_name] * scale_factor
                production_thresholds[threshold_name] = scaled_threshold
        
        logger.info(f"Converted SUR to production thresholds for {crop}: {production_thresholds}")
        return production_thresholds
    
    def get_ipc_colors(self) -> Dict[int, str]:
        """Get IPC phase colors for visualization."""
        return self.ipc_colors.copy()
    
    def calculate_dynamic_thresholds(self, crop: str, year_range: Optional[Tuple[int, int]] = None,
                                   reference_year: int = 2020) -> Dict[str, float]:
        """
        Calculate complete set of dynamic thresholds for a crop.
        
        Args:
            crop: Crop type
            year_range: Historical data range for SUR calculation
            reference_year: Reference year for production baseline
        
        Returns:
            Dictionary with threshold names (T1-T4) mapped to kcal values
        """
        # Calculate SUR-based thresholds
        sur_thresholds = self.calculate_sur_thresholds(crop, year_range)
        
        # Convert to production thresholds
        production_thresholds = self.convert_sur_to_production_thresholds(
            crop, sur_thresholds, reference_year
        )
        
        return production_thresholds

    def calculate_consumption_thresholds(self, crop: str, year_range: Tuple[int, int] = (2019, 2021)) -> Dict[str, float]:
        """
        Calculate consumption-based thresholds (Months of Supply) for determining disruption severity.
        
        REPLACES OLD T1-T4 IPC THRESHOLDS FOR PUBLICATION FIGURES.
        
        These thresholds represent physically meaningful "buffer capacities" of the global food system:
        1. '1 Month': A short-term shock buffer. Losing this amount exceeds the typical logistical slack in supply chains.
           Calculated as: (Mean Annual Consumption) / 12
        2. '3 Months': A medium-term strategic reserve buffer. Roughly equivalent to the carryover stocks many nations aim to hold.
           Calculated as: (Mean Annual Consumption) / 4
        3. 'Total Stocks': The absolute theoretical limit of current reserves. Losing this amount implies a systemic stock-out.
           Calculated as: Mean Ending Stocks
           
        Data Source: USDA Production, Supply and Distribution (PSD) dataset.
        Method:
        - Load historical Consumption and EndingStocks for the crop.
        - Filter for the specified year_range (default 2019-2021 to align with SPAM 2020).
        - Calculate mean values over this period to smooth inter-annual variability.
        
        Mathematical Formulation:
        Let $C_{year}$ be the annual domestic consumption and $S_{year}$ be the ending stocks.
        We compute the mean over the period $Y = [2019, 2021]$:
        
        $$ \bar{C} = \frac{1}{|Y|} \sum_{y \in Y} C_{y} $$
        $$ \bar{S} = \frac{1}{|Y|} \sum_{y \in Y} S_{y} $$
        
        Thresholds:
        - 1-Month Supply: $T_{1m} = \bar{C} / 12$
        - 3-Month Supply: $T_{3m} = \bar{C} / 4$
        - Total Stocks:   $T_{stocks} = \bar{S}$
        
        Args:
            crop: Crop type ('wheat', 'rice', 'maize', 'allgrain')
            year_range: Year range to average consumption/stocks (inclusive).
            
        Returns:
            Dictionary mapping threshold names to mass values in Metric Tons (MT).
            Note: The Config class handles conversion from MT to kcal.
        """
        # Get average consumption and stocks
        if crop == 'allgrain':
            crops = ['wheat', 'rice', 'maize']
            avg_consumption = 0
            avg_stocks = 0
            for c in crops:
                avg_consumption += self.usda_loader.get_mean_consumption(c, year_range)
                avg_stocks += self.usda_loader.get_mean_stocks(c, year_range)
        else:
            avg_consumption = self.usda_loader.get_mean_consumption(crop, year_range)
            avg_stocks = self.usda_loader.get_mean_stocks(crop, year_range)
            
        # Return thresholds in Metric Tons
        return {
            '1 Month': avg_consumption / 12.0,
            '3 Months': avg_consumption / 4.0,
            'Total Stocks': avg_stocks
        }


def create_usda_threshold_system(data_dir: Union[str, Path] = "USDAdata") -> Tuple[USDADataLoader, AgriPhaseThresholdCalculator]:
    """
    Create and initialize USDA data loading and threshold calculation system.
    
    Args:
        data_dir: Directory containing USDA PSD files
    
    Returns:
        Tuple of (USDADataLoader, AgriPhaseThresholdCalculator)
    """
    loader = USDADataLoader(data_dir)
    calculator = AgriPhaseThresholdCalculator(loader)
    
    return loader, calculator


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize system
        loader, calculator = create_usda_threshold_system()
        
        # Test loading data
        wheat_data = loader.load_crop_data('wheat', year_range=(1990, 2020))
        print(f"Loaded wheat data: {len(wheat_data)} years")
        print(wheat_data.head())
        
        # Test threshold calculation
        wheat_thresholds = calculator.calculate_dynamic_thresholds('wheat', year_range=(1990, 2020))
        print(f"Dynamic wheat thresholds: {wheat_thresholds}")
        
    except Exception as e:
        print(f"Error: {e}")
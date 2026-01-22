"""Data validation module for AgRichter framework."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from ..core.base import BaseValidator
from ..core.utils import validate_coordinate_ranges, validate_numeric_range


class DataValidator(BaseValidator):
    """Comprehensive data validator for AgRichter data."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logging.getLogger('agrichter.validator')
        
        # Define validation thresholds
        self.coord_tolerance = 1e-6
        self.max_production_per_cell = 1e8  # metric tons
        self.max_harvest_area_per_cell = 1e6  # hectares
        self.min_reasonable_yield = 0.1  # metric tons per hectare
        self.max_reasonable_yield = 50.0  # metric tons per hectare
    
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """
        Main validation entry point.
        
        Args:
            data: Data to validate (DataFrame or dict of DataFrames)
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if isinstance(data, pd.DataFrame):
            is_valid, df_errors = self.validate_dataframe(data)
            errors.extend(df_errors)
        elif isinstance(data, dict):
            is_valid = True
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df_valid, df_errors = self.validate_dataframe(df, name)
                    if not df_valid:
                        is_valid = False
                    errors.extend([f"{name}: {error}" for error in df_errors])
        else:
            return False, ["Unsupported data type for validation"]
        
        return len(errors) == 0, errors
    
    def validate_dataframe(self, df: pd.DataFrame, name: str = "DataFrame") -> Tuple[bool, List[str]]:
        """
        Validate a single DataFrame.
        
        Args:
            df: DataFrame to validate
            name: Name of the DataFrame for error reporting
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Basic structure validation
        if df.empty:
            errors.append(f"{name} is empty")
            return False, errors
        
        # Check for required columns based on DataFrame type
        if 'x' in df.columns and 'y' in df.columns:
            coord_errors = self.validate_coordinates(df)
            errors.extend(coord_errors)
        
        # Validate crop data if present
        crop_columns = [col for col in df.columns if col.endswith('_a')]
        if crop_columns:
            crop_errors = self.validate_crop_data(df, crop_columns)
            errors.extend(crop_errors)
        
        # Check for data quality issues
        quality_errors = self.validate_data_quality(df, name)
        errors.extend(quality_errors)
        
        return len(errors) == 0, errors
    
    def validate_coordinates(self, df: pd.DataFrame) -> List[str]:
        """
        Validate coordinate data.
        
        Args:
            df: DataFrame with 'x' and 'y' columns
        
        Returns:
            List of error messages
        """
        errors = []
        
        # Check coordinate ranges
        invalid_lat = (~df['y'].between(-90, 90, inclusive='both')).sum()
        invalid_lon = (~df['x'].between(-180, 180, inclusive='both')).sum()
        
        if invalid_lat > 0:
            errors.append(f"Found {invalid_lat} rows with invalid latitude values")
        
        if invalid_lon > 0:
            errors.append(f"Found {invalid_lon} rows with invalid longitude values")
        
        # Check for missing coordinates
        missing_coords = df[['x', 'y']].isna().any(axis=1).sum()
        if missing_coords > 0:
            errors.append(f"Found {missing_coords} rows with missing coordinates")
        
        # Check coordinate precision (should be consistent with 5-minute grid)
        expected_precision = 0.0833333333333333  # 5 arcminutes
        
        # Check if coordinates align with expected grid
        x_remainder = (df['x'] % expected_precision).abs()
        y_remainder = (df['y'] % expected_precision).abs()
        
        misaligned_x = (x_remainder > self.coord_tolerance).sum()
        misaligned_y = (y_remainder > self.coord_tolerance).sum()
        
        if misaligned_x > 0:
            self.logger.warning(f"Found {misaligned_x} rows with non-standard longitude grid alignment")
        
        if misaligned_y > 0:
            self.logger.warning(f"Found {misaligned_y} rows with non-standard latitude grid alignment")
        
        return errors
    
    def validate_crop_data(self, df: pd.DataFrame, crop_columns: List[str]) -> List[str]:
        """
        Validate crop production or harvest area data.
        
        Args:
            df: DataFrame with crop data
            crop_columns: List of crop column names
        
        Returns:
            List of error messages
        """
        errors = []
        
        # Check for negative values
        for col in crop_columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative values in {col}")
        
        # Check for unreasonably large values
        for col in crop_columns:
            if 'P_' in col or col.endswith('_a'):  # Production data
                extreme_count = (df[col] > self.max_production_per_cell).sum()
                if extreme_count > 0:
                    errors.append(f"Found {extreme_count} extremely high production values in {col}")
            elif 'H_' in col:  # Harvest area data
                extreme_count = (df[col] > self.max_harvest_area_per_cell).sum()
                if extreme_count > 0:
                    errors.append(f"Found {extreme_count} extremely high harvest area values in {col}")
        
        # Check data coverage
        total_cells = len(df)
        cells_with_data = (df[crop_columns] > 0).any(axis=1).sum()
        coverage = cells_with_data / total_cells * 100
        
        if coverage < 10:  # Less than 10% coverage
            errors.append(f"Very low data coverage: only {coverage:.1f}% of cells have crop data")
        
        self.logger.info(f"Crop data coverage: {coverage:.1f}% ({cells_with_data}/{total_cells} cells)")
        
        return errors
    
    def validate_data_quality(self, df: pd.DataFrame, name: str) -> List[str]:
        """
        Validate general data quality metrics.
        
        Args:
            df: DataFrame to validate
            name: Name for error reporting
        
        Returns:
            List of error messages
        """
        errors = []
        
        # Check for excessive missing data
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct > 50:
                    errors.append(f"{name}.{col}: {missing_pct:.1f}% missing values")
        
        # Check for duplicate coordinates (if applicable)
        if 'x' in df.columns and 'y' in df.columns:
            duplicates = df.duplicated(subset=['x', 'y']).sum()
            if duplicates > 0:
                errors.append(f"{name}: Found {duplicates} duplicate coordinate pairs")
        
        # Check for data type consistency
        for col in df.columns:
            if col.endswith('_a') and df[col].dtype not in ['float64', 'int64']:
                errors.append(f"{name}.{col}: Expected numeric data type, got {df[col].dtype}")
        
        return errors
    
    def validate_production_harvest_consistency(self, production_df: pd.DataFrame, 
                                              harvest_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate consistency between production and harvest area data.
        
        Args:
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check coordinate alignment
        if len(production_df) != len(harvest_df):
            errors.append(f"Length mismatch: production={len(production_df)}, harvest={len(harvest_df)}")
        else:
            # Check coordinate consistency
            coord_diff_x = (production_df['x'] - harvest_df['x']).abs().max()
            coord_diff_y = (production_df['y'] - harvest_df['y']).abs().max()
            
            if coord_diff_x > self.coord_tolerance or coord_diff_y > self.coord_tolerance:
                errors.append(f"Coordinate mismatch: max diff x={coord_diff_x:.2e}, y={coord_diff_y:.2e}")
        
        # Validate yield calculations (production/harvest area)
        crop_columns = [col for col in production_df.columns if col.endswith('_a')]
        
        for col in crop_columns:
            if col in harvest_df.columns:
                # Calculate yields where both production and harvest area > 0
                mask = (production_df[col] > 0) & (harvest_df[col] > 0)
                if mask.sum() > 0:
                    yields = production_df.loc[mask, col] / harvest_df.loc[mask, col]
                    
                    # Check for unreasonable yields
                    low_yield = (yields < self.min_reasonable_yield).sum()
                    high_yield = (yields > self.max_reasonable_yield).sum()
                    
                    if low_yield > 0:
                        self.logger.warning(f"{col}: {low_yield} cells with very low yields (<{self.min_reasonable_yield} t/ha)")
                    
                    if high_yield > 0:
                        self.logger.warning(f"{col}: {high_yield} cells with very high yields (>{self.max_reasonable_yield} t/ha)")
        
        return len(errors) == 0, errors
    
    def validate_historical_events(self, events_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate historical events data structure.
        
        Args:
            events_data: Dictionary containing historical events data
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required keys
        required_keys = ['country', 'state']
        for key in required_keys:
            if key not in events_data:
                errors.append(f"Missing required key: {key}")
        
        # Validate each event sheet
        if 'country' in events_data:
            country_sheets = events_data['country']
            if not isinstance(country_sheets, dict):
                errors.append("Country events data should be a dictionary of sheets")
            else:
                for sheet_name, sheet_data in country_sheets.items():
                    if not isinstance(sheet_data, pd.DataFrame):
                        errors.append(f"Country sheet '{sheet_name}' is not a DataFrame")
                    elif sheet_data.empty:
                        self.logger.warning(f"Country sheet '{sheet_name}' is empty")
        
        if 'state' in events_data:
            state_sheets = events_data['state']
            if not isinstance(state_sheets, dict):
                errors.append("State events data should be a dictionary of sheets")
            else:
                for sheet_name, sheet_data in state_sheets.items():
                    if not isinstance(sheet_data, pd.DataFrame):
                        errors.append(f"State sheet '{sheet_name}' is not a DataFrame")
                    elif sheet_data.empty:
                        self.logger.warning(f"State sheet '{sheet_name}' is empty")
        
        return len(errors) == 0, errors
    
    def generate_validation_report(self, data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            data: Dictionary of data to validate
        
        Returns:
            Formatted validation report string
        """
        report_lines = ["=== AgRichter Data Validation Report ===\n"]
        
        overall_valid = True
        
        for name, dataset in data.items():
            report_lines.append(f"Dataset: {name}")
            report_lines.append("-" * (len(name) + 10))
            
            if isinstance(dataset, pd.DataFrame):
                is_valid, errors = self.validate_dataframe(dataset, name)
                
                if is_valid:
                    report_lines.append("✓ PASSED")
                else:
                    report_lines.append("✗ FAILED")
                    overall_valid = False
                    for error in errors:
                        report_lines.append(f"  - {error}")
                
                # Add summary statistics
                report_lines.append(f"  Rows: {len(dataset)}")
                report_lines.append(f"  Columns: {len(dataset.columns)}")
                
                if 'x' in dataset.columns and 'y' in dataset.columns:
                    coord_coverage = (~dataset[['x', 'y']].isna().any(axis=1)).sum()
                    report_lines.append(f"  Coordinate coverage: {coord_coverage}/{len(dataset)} ({coord_coverage/len(dataset)*100:.1f}%)")
            
            report_lines.append("")
        
        # Overall status
        report_lines.insert(1, f"Overall Status: {'PASSED' if overall_valid else 'FAILED'}\n")
        
        return "\n".join(report_lines)
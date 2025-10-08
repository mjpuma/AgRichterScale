"""SPAM data validation utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class SPAMValidator:
    """Validator for SPAM data files."""
    
    # Expected columns for SPAM 2020 data
    EXPECTED_METADATA_COLS = [
        'grid_code', 'x', 'y', 'FIPS0', 'FIPS1', 'FIPS2',
        'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
        'rec_type', 'tech_type', 'unit', 'year_data'
    ]
    
    # Key crop columns (uppercase with _A suffix)
    KEY_CROP_COLS = [
        'WHEA_A',  # Wheat
        'RICE_A',  # Rice
        'MAIZ_A',  # Maize
        'BARL_A',  # Barley
        'SORG_A',  # Sorghum
        'MILL_A',  # Millet
        'OCER_A',  # Other cereals
        'SOYB_A',  # Soybeans
    ]
    
    def __init__(self, spam_version: str = '2020'):
        """
        Initialize SPAM validator.
        
        Args:
            spam_version: SPAM data version ('2010' or '2020')
        """
        self.spam_version = spam_version
        self.logger = logging.getLogger(__name__)
    
    def validate_file_structure(self, file_path: Path, 
                                file_type: str = 'production') -> Dict[str, any]:
        """
        Validate SPAM file structure and columns.
        
        Args:
            file_path: Path to SPAM CSV file
            file_type: Type of file ('production' or 'harvest_area')
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'file_exists': file_path.exists(),
            'file_readable': False,
            'columns_valid': False,
            'metadata_cols_present': [],
            'metadata_cols_missing': [],
            'crop_cols_present': [],
            'crop_cols_missing': [],
            'coordinate_cols_valid': False,
            'total_columns': 0,
            'sample_row_count': 0,
            'errors': []
        }
        
        if not results['file_exists']:
            results['errors'].append(f"File not found: {file_path}")
            return results
        
        try:
            # Read first few rows to validate structure
            df = pd.read_csv(file_path, nrows=100)
            results['file_readable'] = True
            results['total_columns'] = len(df.columns)
            results['sample_row_count'] = len(df)
            
            # Check metadata columns
            for col in self.EXPECTED_METADATA_COLS:
                if col in df.columns:
                    results['metadata_cols_present'].append(col)
                else:
                    results['metadata_cols_missing'].append(col)
            
            # Check key crop columns
            for col in self.KEY_CROP_COLS:
                if col in df.columns:
                    results['crop_cols_present'].append(col)
                else:
                    results['crop_cols_missing'].append(col)
            
            # Validate coordinate columns
            if 'x' in df.columns and 'y' in df.columns:
                results['coordinate_cols_valid'] = True
                
                # Check coordinate ranges
                x_min, x_max = df['x'].min(), df['x'].max()
                y_min, y_max = df['y'].min(), df['y'].max()
                
                results['x_range'] = (float(x_min), float(x_max))
                results['y_range'] = (float(y_min), float(y_max))
                
                # Validate ranges (longitude: -180 to 180, latitude: -90 to 90)
                if not (-180 <= x_min <= 180 and -180 <= x_max <= 180):
                    results['errors'].append(
                        f"Invalid x (longitude) range: {x_min} to {x_max}"
                    )
                if not (-90 <= y_min <= 90 and -90 <= y_max <= 90):
                    results['errors'].append(
                        f"Invalid y (latitude) range: {y_min} to {y_max}"
                    )
            
            # Overall validation
            results['columns_valid'] = (
                len(results['metadata_cols_missing']) == 0 and
                len(results['crop_cols_missing']) == 0 and
                results['coordinate_cols_valid']
            )
            
            self.logger.info(
                f"Validated {file_type} file: {file_path.name} - "
                f"{results['total_columns']} columns, "
                f"{len(results['crop_cols_present'])} crop columns found"
            )
            
        except Exception as e:
            results['errors'].append(f"Error reading file: {str(e)}")
            self.logger.error(f"Failed to validate {file_path}: {e}")
        
        return results
    
    def validate_spam_data_pair(self, production_path: Path, 
                               harvest_area_path: Path) -> Dict[str, any]:
        """
        Validate both production and harvest area files.
        
        Args:
            production_path: Path to production CSV file
            harvest_area_path: Path to harvest area CSV file
        
        Returns:
            Dictionary with validation results for both files
        """
        results = {
            'production': self.validate_file_structure(
                production_path, 'production'
            ),
            'harvest_area': self.validate_file_structure(
                harvest_area_path, 'harvest_area'
            ),
            'files_consistent': False,
            'spam_version': self.spam_version
        }
        
        # Check if both files have same structure
        if (results['production']['columns_valid'] and 
            results['harvest_area']['columns_valid']):
            
            prod_cols = set(results['production']['metadata_cols_present'] + 
                          results['production']['crop_cols_present'])
            harvest_cols = set(results['harvest_area']['metadata_cols_present'] + 
                             results['harvest_area']['crop_cols_present'])
            
            results['files_consistent'] = (prod_cols == harvest_cols)
            
            if not results['files_consistent']:
                self.logger.warning(
                    "Production and harvest area files have different column structures"
                )
        
        return results
    
    def get_crop_column_mapping(self) -> Dict[str, str]:
        """
        Get mapping of crop names to column names.
        
        Returns:
            Dictionary mapping crop names to SPAM column names
        """
        return {
            'wheat': 'WHEA_A',
            'rice': 'RICE_A',
            'maize': 'MAIZ_A',
            'barley': 'BARL_A',
            'sorghum': 'SORG_A',
            'millet': 'MILL_A',
            'other_cereals': 'OCER_A',
            'soybeans': 'SOYB_A',
        }
    
    def document_spam_structure(self, file_path: Path) -> str:
        """
        Generate documentation of SPAM file structure.
        
        Args:
            file_path: Path to SPAM CSV file
        
        Returns:
            String with formatted documentation
        """
        if not file_path.exists():
            return f"File not found: {file_path}"
        
        try:
            df = pd.read_csv(file_path, nrows=5)
            
            doc = f"SPAM {self.spam_version} File Structure\n"
            doc += "=" * 50 + "\n\n"
            doc += f"File: {file_path.name}\n"
            doc += f"Total Columns: {len(df.columns)}\n\n"
            
            doc += "Metadata Columns:\n"
            for col in self.EXPECTED_METADATA_COLS:
                if col in df.columns:
                    doc += f"  ✓ {col}\n"
                else:
                    doc += f"  ✗ {col} (MISSING)\n"
            
            doc += "\nKey Crop Columns:\n"
            for col in self.KEY_CROP_COLS:
                if col in df.columns:
                    doc += f"  ✓ {col}\n"
                else:
                    doc += f"  ✗ {col} (MISSING)\n"
            
            doc += f"\nSample Data (first row):\n"
            doc += f"  grid_code: {df.iloc[0]['grid_code']}\n"
            doc += f"  x (lon): {df.iloc[0]['x']}\n"
            doc += f"  y (lat): {df.iloc[0]['y']}\n"
            doc += f"  FIPS0 (ISO3): {df.iloc[0]['FIPS0']}\n"
            doc += f"  ADM0_NAME: {df.iloc[0]['ADM0_NAME']}\n"
            
            return doc
            
        except Exception as e:
            return f"Error generating documentation: {str(e)}"

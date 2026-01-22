"""Configuration management for AgRichter analysis."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .constants import (
    CROP_INDICES, CALORIC_CONTENT, THRESHOLDS, DISRUPTION_RANGES,
    PRODUCTION_RANGES, EVENT_COLORS, EVENT_MARKERS, GRAMS_PER_METRIC_TON,
    HECTARES_TO_KM2, GRID_PARAMS
)
from .utils import (
    validate_file_permissions, validate_directory_structure,
    create_directory_structure, check_disk_space
)
# USDA imports will be done conditionally to avoid circular imports


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class Config:
    """Configuration manager for AgRichter analysis."""
    
    def __init__(self, crop_type: str, root_dir: Optional[Union[str, Path]] = None, 
                 use_dynamic_thresholds: bool = True, usda_year_range: Optional[tuple] = None,
                 spam_version: str = '2020', convergence_validation: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            crop_type: Type of crop analysis ('allgrain', 'wheat', 'rice', 'maize')
            root_dir: Root directory for data files (default: './data')
            use_dynamic_thresholds: Whether to use USDA-based dynamic thresholds
            usda_year_range: Year range for USDA threshold calculation (default: 1990-2020)
            spam_version: SPAM data version to use ('2010' or '2020', default: '2020')
            convergence_validation: Configuration for envelope convergence validation
        
        Raises:
            ConfigError: If crop_type is invalid or spam_version is invalid
        """
        self.crop_type = self._validate_crop_type(crop_type)
        self.root_dir = Path(root_dir) if root_dir else Path('./data')
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.usda_year_range = usda_year_range or (1990, 2020)
        self.spam_version = self._validate_spam_version(spam_version)
        self.convergence_validation = self._setup_convergence_validation(convergence_validation)
        
        # Initialize USDA system if using dynamic thresholds
        self.usda_loader = None
        self.threshold_calculator = None
        if self.use_dynamic_thresholds:
            try:
                # Import here to avoid circular imports
                from ..data.usda import create_usda_threshold_system
                self.usda_loader, self.threshold_calculator = create_usda_threshold_system()
            except Exception as e:
                logging.warning(f"Failed to initialize USDA system, falling back to static thresholds: {e}")
                self.use_dynamic_thresholds = False
        
        # Initialize derived parameters
        self._setup_paths()
        self._setup_crop_parameters()
    
    def _validate_crop_type(self, crop_type: str) -> str:
        """Validate crop type parameter."""
        valid_crops = list(CROP_INDICES.keys())
        if crop_type not in valid_crops:
            raise ConfigError(
                f"Invalid crop type '{crop_type}'. "
                f"Valid options: {valid_crops}"
            )
        return crop_type
    
    def _validate_spam_version(self, spam_version: str) -> str:
        """Validate SPAM version parameter."""
        valid_versions = ['2010', '2020']
        if spam_version not in valid_versions:
            raise ConfigError(
                f"Invalid SPAM version '{spam_version}'. "
                f"Valid options: {valid_versions}"
            )
        return spam_version
    
    def _setup_convergence_validation(self, convergence_validation: Optional[Dict]) -> Dict:
        """Set up convergence validation configuration with defaults."""
        default_config = {
            'enabled': True,
            'enforce_convergence': True,
            'fallback_on_failure': True,
            'tolerance': 1e-6,
            'max_iterations': 10,
            'log_validation_details': False,
            'backward_compatible': True
        }
        
        if convergence_validation is None:
            return default_config
        
        # Merge user config with defaults
        config = default_config.copy()
        config.update(convergence_validation)
        
        # Validate configuration values
        if not isinstance(config['enabled'], bool):
            raise ConfigError("convergence_validation.enabled must be boolean")
        if not isinstance(config['enforce_convergence'], bool):
            raise ConfigError("convergence_validation.enforce_convergence must be boolean")
        if not isinstance(config['fallback_on_failure'], bool):
            raise ConfigError("convergence_validation.fallback_on_failure must be boolean")
        if not isinstance(config['tolerance'], (int, float)) or config['tolerance'] <= 0:
            raise ConfigError("convergence_validation.tolerance must be positive number")
        if not isinstance(config['max_iterations'], int) or config['max_iterations'] <= 0:
            raise ConfigError("convergence_validation.max_iterations must be positive integer")
        
        return config
    
    def _setup_paths(self) -> None:
        """Set up file paths for data inputs and outputs."""
        self.paths = {
            'inputs': self.root_dir / 'inputs',
            'ancillary': self.root_dir / 'ancillary',
            'outputs': self.root_dir / 'outputs',
            'media': self.root_dir / 'outputs' / 'media'
        }
        
        # SPAM data files - version-specific paths
        if self.spam_version == '2020':
            spam_prod_dir = self.root_dir / 'spam2020V2r0_global_production' / 'spam2020V2r0_global_production'
            spam_harvest_dir = self.root_dir / 'spam2020V2r0_global_harvested_area' / 'spam2020V2r0_global_harvested_area'
            spam_yield_dir = self.root_dir / 'spam2020V2r0_global_yield' / 'spam2020V2r0_global_yield'
            production_file = spam_prod_dir / 'spam2020V2r0_global_P_TA.csv'
            harvest_area_file = spam_harvest_dir / 'spam2020V2r0_global_H_TA.csv'
            yield_file = spam_yield_dir / 'spam2020V2r0_global_Y_TA.csv'
        else:  # 2010
            spam_prod_dir = self.root_dir / 'spam2010_global_production'
            spam_harvest_dir = self.root_dir / 'spam2010_global_harvested_area'
            spam_yield_dir = self.root_dir / 'spam2010_global_yield'
            production_file = spam_prod_dir / 'spam2010_global_P_TA.csv'
            harvest_area_file = spam_harvest_dir / 'spam2010_global_H_TA.csv'
            yield_file = spam_yield_dir / 'spam2010_global_Y_TA.csv'
        
        self.data_files = {
            'production': production_file,
            'harvest_area': harvest_area_file,
            'yield': yield_file,
            'nutrition': self.paths['ancillary'] / 'Nutrition_SPAMcrops.xls',
            'food_codes': self.paths['ancillary'] / 'Foodcodes_SPAMtoFAOSTAT.xls',
            'country_codes': self.paths['ancillary'] / 'CountryCode_Convert.xls',
            'disruption_country': self.paths['ancillary'] / 'DisruptionCountry.xls',
            'disruption_state': self.paths['ancillary'] / 'DisruptionStateProvince.xls'
        }
    
    def _setup_crop_parameters(self) -> None:
        """Set up crop-specific parameters."""
        self.crop_indices = CROP_INDICES[self.crop_type]
        
        # Set caloric content based on crop type
        if self.crop_type == 'allgrain':
            self.caloric_content = CALORIC_CONTENT['Allgrain']
        elif self.crop_type == 'wheat':
            self.caloric_content = CALORIC_CONTENT['Wheat']
        elif self.crop_type == 'rice':
            self.caloric_content = CALORIC_CONTENT['Rice']
        elif self.crop_type == 'maize':
            self.caloric_content = CALORIC_CONTENT['Corn']
            
        # Set thresholds (dynamic consumption-based or static)
        if self.use_dynamic_thresholds and self.threshold_calculator:
            try:
                # Calculate consumption-based thresholds (Months of Supply)
                # These are returned in Metric Tons
                # NOTE: This replaces the old T1-T4 thresholds with physically meaningful values:
                # 1 Month Supply, 3 Months Supply, and Total Ending Stocks
                raw_thresholds = self.threshold_calculator.calculate_consumption_thresholds(
                    self.crop_type, self.usda_year_range
                )
                
                # Convert MT to kcal: MT * 10^6 g/MT * kcal/g
                # This conversion is critical for comparing against production losses which are in kcal
                conversion_factor = 1_000_000.0 * self.caloric_content
                
                self.thresholds = {
                    k: v * conversion_factor 
                    for k, v in raw_thresholds.items()
                }
                logging.info(f"Using consumption-based thresholds for {self.crop_type}: {self.thresholds}")
            except Exception as e:
                logging.warning(f"Failed to calculate dynamic thresholds, using static: {e}")
                self.thresholds = THRESHOLDS[self.crop_type]
        else:
            self.thresholds = THRESHOLDS[self.crop_type]
        
        self.disruption_range = DISRUPTION_RANGES[self.crop_type]
        self.production_range = PRODUCTION_RANGES[self.crop_type]
        self.event_colors = EVENT_COLORS[self.crop_type]
        
    def get_ipc_colors(self) -> Dict[Union[int, str], str]:
        """Get threshold colors for visualization."""
        if self.use_dynamic_thresholds:
             # Colors for consumption-based thresholds
             return {
                 '1 Month': '#FFD700',       # Gold/Yellow
                 '3 Months': '#FF4500',      # OrangeRed (more visible than DarkOrange)
                 'Total Stocks': '#800080'   # Purple (Systemic)
             }
        elif self.threshold_calculator:
            return self.threshold_calculator.get_ipc_colors()
        else:
            # Fallback colors
            return {
                1: '#00FF00',  # Green
                2: '#FFFF00',  # Yellow  
                3: '#FFA500',  # Orange
                4: '#FF0000',  # Red
                5: '#800080'   # Purple
            }
    
    def get_crop_indices(self) -> List[int]:
        """Get crop indices for current crop type."""
        return self.crop_indices
    
    def get_caloric_content(self) -> float:
        """Get caloric content for current crop type."""
        return self.caloric_content
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get AgRichter thresholds for current crop type."""
        return self.thresholds
    
    def get_threshold_log10(self) -> Dict[str, float]:
        """Get log10 of thresholds for plotting."""
        return {k: float(f"{v:.2e}".split('e')[0]) * 10**int(f"{v:.2e}".split('e')[1]) 
                for k, v in self.thresholds.items()}
    
    def get_file_paths(self) -> Dict[str, Path]:
        """Get dictionary of all file paths."""
        return self.data_files
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get output file paths for current crop type."""
        media_dir = self.paths['media']
        outputs_dir = self.paths['outputs']
        
        return {
            'production_map': media_dir / f'ProductionMap_{self.crop_type}.svg',
            'richter_scale_eps': media_dir / f'RichterScale_{self.crop_type}.eps',
            'richter_scale_jpg': media_dir / f'RichterScale_{self.crop_type}.jpg',
            'hp_envelope_eps': media_dir / f'Production_vs_HarvestArea_{self.crop_type}.eps',
            'hp_envelope_jpg': media_dir / f'Production_vs_HarvestArea_{self.crop_type}.jpg',
            'event_losses_csv': outputs_dir / f'LostProd_Events_spam2020_{self.crop_type}.csv',
            'envelope_data_csv': outputs_dir / f'HP_Envelope_Data_{self.crop_type}.csv'
        }
    
    def get_map_title(self) -> str:
        """Get title for production map."""
        titles = {
            'allgrain': 'Grains (log₁₀ kcal)',
            'wheat': 'Wheat (log₁₀ kcal)',
            'rice': 'Rice (log₁₀ kcal)',
            'maize': 'Maize (log₁₀ kcal)'
        }
        return titles[self.crop_type]
    
    def get_event_style(self, event_name: str) -> Dict[str, str]:
        """Get color and marker style for an event."""
        color = self.event_colors.get(event_name, 'green')
        marker = EVENT_MARKERS.get(color, 'd')
        return {'color': color, 'marker': marker}
    
    def get_unit_conversions(self) -> Dict[str, float]:
        """Get unit conversion constants."""
        return {
            'grams_per_metric_ton': GRAMS_PER_METRIC_TON,
            'hectares_to_km2': HECTARES_TO_KM2
        }
    
    def get_grid_params(self) -> Dict[str, float]:
        """Get grid parameters for coordinate system."""
        return GRID_PARAMS.copy()
    
    def get_spam_version(self) -> str:
        """Get SPAM data version being used."""
        return self.spam_version
    
    def validate_files_exist(self) -> List[str]:
        """
        Validate that required data files exist.
        
        Returns:
            List of missing file paths
        """
        missing_files = []
        for name, path in self.data_files.items():
            if not path.exists():
                missing_files.append(str(path))
        return missing_files
    
    def validate_spam_files(self) -> Dict[str, bool]:
        """
        Validate that SPAM data files exist and are readable.
        
        Returns:
            Dictionary with validation results for SPAM files
        """
        from .spam_validator import SPAMValidator
        
        validator = SPAMValidator(spam_version=self.spam_version)
        
        # Perform comprehensive validation
        validation_results = validator.validate_spam_data_pair(
            self.data_files['production'],
            self.data_files['harvest_area']
        )
        
        # Add basic existence checks
        validation_results['production_exists'] = self.data_files['production'].exists()
        validation_results['harvest_area_exists'] = self.data_files['harvest_area'].exists()
        
        return validation_results
    
    def get_spam_structure_documentation(self) -> str:
        """
        Get documentation of SPAM file structure.
        
        Returns:
            String with formatted documentation
        """
        from .spam_validator import SPAMValidator
        
        validator = SPAMValidator(spam_version=self.spam_version)
        
        doc = validator.document_spam_structure(self.data_files['production'])
        doc += "\n\n" + "=" * 50 + "\n\n"
        doc += validator.document_spam_structure(self.data_files['harvest_area'])
        
        return doc
    
    def validate_file_permissions(self) -> Dict[str, bool]:
        """
        Validate file permissions for all required files.
        
        Returns:
            Dictionary with permission validation results
        """
        results = {}
        for name, path in self.data_files.items():
            results[name] = validate_file_permissions(path, 'r')
        return results
    
    def validate_output_permissions(self) -> bool:
        """
        Validate write permissions for output directories.
        
        Returns:
            True if all output directories are writable
        """
        for path in self.paths.values():
            if path.exists() and not validate_file_permissions(path, 'w'):
                return False
        return True
    
    def validate_directory_structure(self) -> Dict[str, bool]:
        """
        Validate expected directory structure.
        
        Returns:
            Dictionary with directory validation results
        """
        return validate_directory_structure(self.root_dir)
    
    def create_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        create_directory_structure(self.root_dir)
        
        # Also create crop-specific subdirectories
        crop_dir = self.paths['outputs'] / self.crop_type
        crop_dir.mkdir(exist_ok=True)
    
    def check_disk_space(self, required_gb: float = 2.0) -> bool:
        """
        Check if sufficient disk space is available for outputs.
        
        Args:
            required_gb: Required space in GB
        
        Returns:
            True if sufficient space is available
        """
        return check_disk_space(self.paths['outputs'], required_gb)
    
    def get_system_info(self) -> Dict[str, str]:
        """Get system information for debugging."""
        return {
            'root_dir': str(self.root_dir.absolute()),
            'crop_type': self.crop_type,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'working_directory': str(Path.cwd()),
            'user': os.getenv('USER', 'unknown')
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (f"Config(crop_type='{self.crop_type}', "
                f"root_dir='{self.root_dir}', "
                f"spam_version='{self.spam_version}', "
                f"crop_indices={self.crop_indices})")
    
    def get_usda_data(self, year_range: Optional[tuple] = None) -> Optional[Dict]:
        """
        Get USDA PSD data for current crop type.
        
        Args:
            year_range: Optional year range filter
        
        Returns:
            Dictionary with USDA data or None if not available
        """
        if not self.usda_loader:
            return None
        
        try:
            if self.crop_type == 'allgrain':
                return self.usda_loader.load_all_crops(year_range)
            else:
                crop_name = self.crop_type if self.crop_type != 'maize' else 'maize'
                return {crop_name: self.usda_loader.load_crop_data(crop_name, year_range)}
        except Exception as e:
            logging.warning(f"Failed to load USDA data: {e}")
            return None
    
    def get_sur_thresholds(self) -> Optional[Dict[int, float]]:
        """
        Get SUR-based thresholds for IPC phases.
        
        Returns:
            Dictionary mapping IPC phase to SUR threshold or None
        """
        if not self.threshold_calculator:
            return None
        
        try:
            return self.threshold_calculator.calculate_sur_thresholds(
                self.crop_type, self.usda_year_range
            )
        except Exception as e:
            logging.warning(f"Failed to calculate SUR thresholds: {e}")
            return None
    
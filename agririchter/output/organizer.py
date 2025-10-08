"""File organization utilities for AgriRichter output management."""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class FileOrganizer:
    """Organizes output files with consistent naming and directory structure."""
    
    def __init__(self, base_output_dir: Union[str, Path] = "outputs"):
        """
        Initialize file organizer.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define standard directory structure
        self.directories = {
            'data': self.base_dir / 'data',
            'figures': self.base_dir / 'figures',
            'reports': self.base_dir / 'reports',
            'media': self.base_dir / 'media',
            'analysis': self.base_dir / 'analysis',
            'temp': self.base_dir / 'temp'
        }
        
        # Create directories
        self._create_directory_structure()
    
    def _create_directory_structure(self) -> None:
        """Create the standard output directory structure."""
        for dir_name, dir_path in self.directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def get_output_path(self, file_type: str, crop_type: str, 
                       file_name: str, include_timestamp: bool = False) -> Path:
        """
        Generate standardized output file path.
        
        Args:
            file_type: Type of file ('data', 'figures', 'reports', 'media', 'analysis')
            crop_type: Crop type for organization
            file_name: Base file name
            include_timestamp: Whether to include timestamp in filename
        
        Returns:
            Path object for the output file
        """
        if file_type not in self.directories:
            raise ValueError(f"Invalid file type '{file_type}'. Valid types: {list(self.directories.keys())}")
        
        # Create crop-specific subdirectory
        crop_dir = self.directories[file_type] / crop_type
        crop_dir.mkdir(exist_ok=True)
        
        # Add timestamp if requested
        if include_timestamp:
            name_parts = file_name.split('.')
            if len(name_parts) > 1:
                base_name = '.'.join(name_parts[:-1])
                extension = name_parts[-1]
                file_name = f"{base_name}_{self.timestamp}.{extension}"
            else:
                file_name = f"{file_name}_{self.timestamp}"
        
        return crop_dir / file_name
    
    def get_figure_path(self, crop_type: str, figure_type: str, 
                       format: str = 'png') -> Path:
        """
        Generate standardized figure file path.
        
        Args:
            crop_type: Crop type ('wheat', 'rice', 'maize', 'allgrain')
            figure_type: Type of figure ('agririchter_scale', 'hp_envelope', 'production_map', 'balance_timeseries')
            format: File format ('png', 'svg', 'eps', 'jpg')
        
        Returns:
            Path object for the figure file
        """
        # Standardize naming convention
        figure_names = {
            'agririchter_scale': f'AgriRichterScale_{crop_type.title()}',
            'hp_envelope': f'HP_Envelope_{crop_type.title()}',
            'production_map': f'ProductionMap_{crop_type.title()}',
            'balance_timeseries': f'BalanceTimeSeries_{crop_type.title()}'
        }
        
        if figure_type not in figure_names:
            raise ValueError(f"Invalid figure type '{figure_type}'. Valid types: {list(figure_names.keys())}")
        
        file_name = f"{figure_names[figure_type]}.{format}"
        return self.get_output_path('figures', crop_type, file_name)
    
    def get_data_path(self, crop_type: str, data_type: str, 
                     format: str = 'csv') -> Path:
        """
        Generate standardized data file path.
        
        Args:
            crop_type: Crop type
            data_type: Type of data ('event_losses', 'envelope_data', 'threshold_data', 'analysis_results')
            format: File format ('csv', 'json', 'xlsx')
        
        Returns:
            Path object for the data file
        """
        # Standardize naming convention
        data_names = {
            'event_losses': f'EventLosses_{crop_type.title()}',
            'envelope_data': f'HP_Envelope_Data_{crop_type.title()}',
            'threshold_data': f'Thresholds_{crop_type.title()}',
            'analysis_results': f'AnalysisResults_{crop_type.title()}',
            'usda_data': f'USDA_PSD_Data_{crop_type.title()}',
            'sur_analysis': f'SUR_Analysis_{crop_type.title()}'
        }
        
        if data_type not in data_names:
            raise ValueError(f"Invalid data type '{data_type}'. Valid types: {list(data_names.keys())}")
        
        file_name = f"{data_names[data_type]}.{format}"
        return self.get_output_path('data', crop_type, file_name)
    
    def get_report_path(self, crop_type: str, report_type: str = 'analysis') -> Path:
        """
        Generate standardized report file path.
        
        Args:
            crop_type: Crop type
            report_type: Type of report ('analysis', 'summary', 'validation')
        
        Returns:
            Path object for the report file
        """
        report_names = {
            'analysis': f'AnalysisReport_{crop_type.title()}',
            'summary': f'Summary_{crop_type.title()}',
            'validation': f'ValidationReport_{crop_type.title()}'
        }
        
        if report_type not in report_names:
            raise ValueError(f"Invalid report type '{report_type}'. Valid types: {list(report_names.keys())}")
        
        file_name = f"{report_names[report_type]}.txt"
        return self.get_output_path('reports', crop_type, file_name, include_timestamp=True)
    
    def organize_existing_files(self, source_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Organize existing files from a source directory into the standard structure.
        
        Args:
            source_dir: Directory containing files to organize
        
        Returns:
            Dictionary mapping file types to lists of organized file paths
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_path}")
        
        organized_files = {
            'figures': [],
            'data': [],
            'reports': [],
            'other': []
        }
        
        # Define file patterns for organization
        patterns = {
            'figures': ['.png', '.svg', '.eps', '.jpg', '.pdf'],
            'data': ['.csv', '.json', '.xlsx', '.xls'],
            'reports': ['.txt', '.md', '.html', '.pdf']
        }
        
        # Process all files in source directory
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                file_type = self._classify_file(file_path, patterns)
                
                if file_type != 'other':
                    # Try to extract crop type from filename
                    crop_type = self._extract_crop_type(file_path.name)
                    
                    # Generate new path
                    new_path = self.get_output_path(file_type, crop_type, file_path.name)
                    
                    # Copy file to new location
                    try:
                        shutil.copy2(file_path, new_path)
                        organized_files[file_type].append(new_path)
                        logger.info(f"Organized file: {file_path} -> {new_path}")
                    except Exception as e:
                        logger.warning(f"Failed to organize file {file_path}: {e}")
                        organized_files['other'].append(file_path)
                else:
                    organized_files['other'].append(file_path)
        
        return organized_files
    
    def _classify_file(self, file_path: Path, patterns: Dict[str, List[str]]) -> str:
        """Classify file based on extension."""
        extension = file_path.suffix.lower()
        
        for file_type, extensions in patterns.items():
            if extension in extensions:
                return file_type
        
        return 'other'
    
    def _extract_crop_type(self, filename: str) -> str:
        """Extract crop type from filename."""
        filename_lower = filename.lower()
        
        # Check for crop type keywords
        if 'wheat' in filename_lower:
            return 'wheat'
        elif 'rice' in filename_lower:
            return 'rice'
        elif 'maize' in filename_lower or 'corn' in filename_lower:
            return 'maize'
        elif 'allgrain' in filename_lower or 'all_grain' in filename_lower:
            return 'allgrain'
        else:
            return 'general'
    
    def clean_temp_files(self) -> int:
        """
        Clean temporary files and return count of files removed.
        
        Returns:
            Number of files removed
        """
        temp_dir = self.directories['temp']
        if not temp_dir.exists():
            return 0
        
        files_removed = 0
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    files_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {file_path}: {e}")
        
        logger.info(f"Cleaned {files_removed} temporary files")
        return files_removed
    
    def get_directory_info(self) -> Dict[str, Dict[str, Union[int, str]]]:
        """
        Get information about the directory structure.
        
        Returns:
            Dictionary with directory information
        """
        info = {}
        
        for dir_name, dir_path in self.directories.items():
            if dir_path.exists():
                # Count files in directory
                file_count = sum(1 for _ in dir_path.rglob('*') if _.is_file())
                
                # Calculate directory size
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                
                info[dir_name] = {
                    'path': str(dir_path),
                    'file_count': file_count,
                    'size_bytes': total_size,
                    'size_mb': round(total_size / (1024 * 1024), 2)
                }
            else:
                info[dir_name] = {
                    'path': str(dir_path),
                    'file_count': 0,
                    'size_bytes': 0,
                    'size_mb': 0.0
                }
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the file organizer."""
        return f"FileOrganizer(base_dir='{self.base_dir}', timestamp='{self.timestamp}')"


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create file organizer
    organizer = FileOrganizer("test_outputs")
    
    # Test path generation
    figure_path = organizer.get_figure_path('wheat', 'agririchter_scale', 'png')
    print(f"Figure path: {figure_path}")
    
    data_path = organizer.get_data_path('wheat', 'event_losses', 'csv')
    print(f"Data path: {data_path}")
    
    report_path = organizer.get_report_path('wheat', 'analysis')
    print(f"Report path: {report_path}")
    
    # Get directory info
    info = organizer.get_directory_info()
    print(f"Directory info: {info}")
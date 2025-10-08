"""Data and figure export utilities for AgriRichter analysis."""

import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DataExporter:
    """Exports analysis data in various formats."""
    
    def __init__(self, organizer):
        """
        Initialize data exporter.
        
        Args:
            organizer: FileOrganizer instance for path management
        """
        self.organizer = organizer
    
    def export_event_losses(self, events_data: pd.DataFrame, crop_type: str,
                           format: str = 'csv') -> Path:
        """
        Export event loss data to CSV/Excel format.
        
        Args:
            events_data: DataFrame with event loss data
            crop_type: Crop type for file naming
            format: Export format ('csv', 'xlsx', 'json')
        
        Returns:
            Path to exported file
        """
        output_path = self.organizer.get_data_path(crop_type, 'event_losses', format)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata columns
        export_data = events_data.copy()
        export_data['export_timestamp'] = datetime.now().isoformat()
        export_data['crop_type'] = crop_type
        
        # Export based on format
        if format == 'csv':
            export_data.to_csv(output_path, index=False)
        elif format == 'xlsx':
            export_data.to_excel(output_path, index=False, engine='openpyxl')
        elif format == 'json':
            export_data.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported event losses data to: {output_path}")
        return output_path
    
    def export_envelope_data(self, envelope_data: Dict, crop_type: str,
                           format: str = 'csv') -> Path:
        """
        Export H-P envelope data.
        
        Args:
            envelope_data: Dictionary with envelope data (disrupted_areas, upper_bound, lower_bound)
            crop_type: Crop type for file naming
            format: Export format ('csv', 'xlsx', 'json')
        
        Returns:
            Path to exported file
        """
        output_path = self.organizer.get_data_path(crop_type, 'envelope_data', format)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'disrupted_area_km2': envelope_data['disrupted_areas'],
            'upper_bound_kcal': envelope_data['upper_bound'],
            'lower_bound_kcal': envelope_data['lower_bound']
        })
        
        # Add metadata
        df['export_timestamp'] = datetime.now().isoformat()
        df['crop_type'] = crop_type
        
        # Calculate additional metrics
        df['magnitude'] = np.log10(df['disrupted_area_km2'])
        df['envelope_width_kcal'] = df['upper_bound_kcal'] - df['lower_bound_kcal']
        
        # Export based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'xlsx':
            df.to_excel(output_path, index=False, engine='openpyxl')
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported envelope data to: {output_path}")
        return output_path
    
    def export_threshold_data(self, thresholds: Dict[str, float], 
                            sur_thresholds: Optional[Dict[int, float]], 
                            crop_type: str, format: str = 'csv') -> Path:
        """
        Export threshold data including both production and SUR thresholds.
        
        Args:
            thresholds: Production thresholds (T1-T4)
            sur_thresholds: SUR thresholds by IPC phase
            crop_type: Crop type for file naming
            format: Export format ('csv', 'xlsx', 'json')
        
        Returns:
            Path to exported file
        """
        output_path = self.organizer.get_data_path(crop_type, 'threshold_data', format)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare threshold data
        threshold_records = []
        
        # Production thresholds
        for threshold_name, threshold_value in thresholds.items():
            threshold_records.append({
                'threshold_type': 'production',
                'threshold_name': threshold_name,
                'threshold_value': threshold_value,
                'unit': 'kcal',
                'ipc_phase': {'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}.get(threshold_name, None)
            })
        
        # SUR thresholds
        if sur_thresholds:
            for ipc_phase, sur_value in sur_thresholds.items():
                threshold_records.append({
                    'threshold_type': 'sur',
                    'threshold_name': f'Phase_{ipc_phase}',
                    'threshold_value': sur_value,
                    'unit': 'ratio',
                    'ipc_phase': ipc_phase
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(threshold_records)
        df['export_timestamp'] = datetime.now().isoformat()
        df['crop_type'] = crop_type
        
        # Export based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'xlsx':
            df.to_excel(output_path, index=False, engine='openpyxl')
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported threshold data to: {output_path}")
        return output_path
    
    def export_usda_data(self, usda_data: Dict, crop_type: str,
                        format: str = 'csv') -> Path:
        """
        Export USDA PSD data.
        
        Args:
            usda_data: Dictionary with USDA data by crop
            crop_type: Crop type for file naming
            format: Export format ('csv', 'xlsx', 'json')
        
        Returns:
            Path to exported file
        """
        output_path = self.organizer.get_data_path(crop_type, 'usda_data', format)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine all USDA data
        combined_data = []
        for crop_name, crop_data in usda_data.items():
            if not crop_data.empty:
                crop_data_copy = crop_data.copy()
                crop_data_copy['crop_name'] = crop_name
                crop_data_copy['export_timestamp'] = datetime.now().isoformat()
                
                # Calculate SUR
                crop_data_copy['SUR'] = crop_data_copy['EndingStocks'] / crop_data_copy['Consumption']
                
                combined_data.append(crop_data_copy)
        
        if combined_data:
            df = pd.concat(combined_data, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Export based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'xlsx':
            df.to_excel(output_path, index=False, engine='openpyxl')
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported USDA data to: {output_path}")
        return output_path
    
    def export_analysis_summary(self, analysis_results: Dict[str, Any], 
                              crop_type: str) -> Path:
        """
        Export comprehensive analysis summary.
        
        Args:
            analysis_results: Dictionary with analysis results
            crop_type: Crop type for file naming
        
        Returns:
            Path to exported file
        """
        output_path = self.organizer.get_data_path(crop_type, 'analysis_results', 'json')
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        summary = {
            'metadata': {
                'crop_type': crop_type,
                'export_timestamp': datetime.now().isoformat(),
                'analysis_version': '1.0'
            },
            'results': analysis_results
        }
        
        # Convert numpy arrays to lists for JSON serialization
        summary = self._convert_numpy_for_json(summary)
        
        # Export as JSON
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Exported analysis summary to: {output_path}")
        return output_path
    
    def _convert_numpy_for_json(self, obj: Any) -> Any:
        """Convert numpy arrays and types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        else:
            return obj


class FigureExporter:
    """Exports figures in multiple formats with consistent styling."""
    
    def __init__(self, organizer):
        """
        Initialize figure exporter.
        
        Args:
            organizer: FileOrganizer instance for path management
        """
        self.organizer = organizer
        
        # Default export settings
        self.default_formats = ['png', 'svg', 'eps']
        self.default_dpi = 300
        self.default_bbox_inches = 'tight'
        self.default_facecolor = 'white'
        self.default_edgecolor = 'none'
    
    def export_figure(self, fig: plt.Figure, crop_type: str, figure_type: str,
                     formats: Optional[List[str]] = None, **kwargs) -> List[Path]:
        """
        Export figure in multiple formats.
        
        Args:
            fig: Matplotlib figure object
            crop_type: Crop type for file naming
            figure_type: Type of figure
            formats: List of formats to export (default: png, svg, eps)
            **kwargs: Additional arguments for savefig
        
        Returns:
            List of paths to exported files
        """
        if formats is None:
            formats = self.default_formats
        
        exported_paths = []
        
        # Set default savefig parameters
        savefig_params = {
            'dpi': self.default_dpi,
            'bbox_inches': self.default_bbox_inches,
            'facecolor': self.default_facecolor,
            'edgecolor': self.default_edgecolor
        }
        savefig_params.update(kwargs)
        
        for format in formats:
            try:
                output_path = self.organizer.get_figure_path(crop_type, figure_type, format)
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Export figure
                fig.savefig(output_path, format=format, **savefig_params)
                exported_paths.append(output_path)
                
                logger.info(f"Exported {figure_type} figure to: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to export figure in {format} format: {e}")
        
        return exported_paths
    
    def export_all_figures(self, figures: Dict[str, plt.Figure], crop_type: str,
                          formats: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        Export multiple figures at once.
        
        Args:
            figures: Dictionary mapping figure types to figure objects
            crop_type: Crop type for file naming
            formats: List of formats to export
        
        Returns:
            Dictionary mapping figure types to lists of exported paths
        """
        exported_files = {}
        
        for figure_type, fig in figures.items():
            try:
                paths = self.export_figure(fig, crop_type, figure_type, formats)
                exported_files[figure_type] = paths
            except Exception as e:
                logger.error(f"Failed to export {figure_type} figure: {e}")
                exported_files[figure_type] = []
        
        return exported_files


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from agririchter.output.organizer import FileOrganizer
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create organizer and exporter
    organizer = FileOrganizer("test_outputs")
    exporter = DataExporter(organizer)
    
    # Test data export
    sample_events = pd.DataFrame({
        'event_name': ['Test Event 1', 'Test Event 2'],
        'harvest_area_km2': [100000, 200000],
        'production_loss_kcal': [1e14, 2e14]
    })
    
    # Export test data
    csv_path = exporter.export_event_losses(sample_events, 'wheat', 'csv')
    print(f"Exported CSV: {csv_path}")
    
    # Test envelope data export
    sample_envelope = {
        'disrupted_areas': np.array([100, 1000, 10000]),
        'upper_bound': np.array([1e12, 1e13, 1e14]),
        'lower_bound': np.array([5e11, 5e12, 5e13])
    }
    
    envelope_path = exporter.export_envelope_data(sample_envelope, 'wheat', 'csv')
    print(f"Exported envelope data: {envelope_path}")
    
    print("Data export test completed!")
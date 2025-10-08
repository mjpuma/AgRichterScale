"""Main output management system for AgriRichter analysis."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from .organizer import FileOrganizer
from .exporter import DataExporter, FigureExporter
from .reporter import AnalysisReporter

logger = logging.getLogger(__name__)


class OutputManager:
    """Comprehensive output management for AgriRichter analysis."""
    
    def __init__(self, base_output_dir: Union[str, Path] = "outputs",
                 auto_organize: bool = True):
        """
        Initialize output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
            auto_organize: Whether to automatically organize files
        """
        self.organizer = FileOrganizer(base_output_dir)
        self.data_exporter = DataExporter(self.organizer)
        self.figure_exporter = FigureExporter(self.organizer)
        self.reporter = AnalysisReporter(self.organizer)
        self.auto_organize = auto_organize
        
        # Track exported files
        self.exported_files = {
            'data': {},
            'figures': {},
            'reports': {}
        }
        
        logger.info(f"OutputManager initialized with base directory: {base_output_dir}")
    
    def export_complete_analysis(self, analysis_data: Dict[str, Any], 
                               crop_type: str) -> Dict[str, List[Path]]:
        """
        Export complete analysis results including data, figures, and reports.
        
        Args:
            analysis_data: Dictionary containing all analysis results
            crop_type: Crop type for organization
        
        Returns:
            Dictionary mapping export types to lists of file paths
        """
        exported_paths = {
            'data': [],
            'figures': [],
            'reports': []
        }
        
        logger.info(f"Starting complete analysis export for {crop_type}")
        
        try:
            # Export data files
            if 'events_data' in analysis_data:
                path = self.export_event_losses(analysis_data['events_data'], crop_type)
                exported_paths['data'].append(path)
            
            if 'envelope_data' in analysis_data:
                path = self.export_envelope_data(analysis_data['envelope_data'], crop_type)
                exported_paths['data'].append(path)
            
            if 'thresholds' in analysis_data:
                sur_thresholds = analysis_data.get('sur_thresholds')
                path = self.export_threshold_data(analysis_data['thresholds'], 
                                                sur_thresholds, crop_type)
                exported_paths['data'].append(path)
            
            if 'usda_data' in analysis_data:
                path = self.export_usda_data(analysis_data['usda_data'], crop_type)
                exported_paths['data'].append(path)
            
            # Export analysis summary
            path = self.export_analysis_summary(analysis_data, crop_type)
            exported_paths['data'].append(path)
            
            # Export figures
            if 'figures' in analysis_data:
                figure_paths = self.export_all_figures(analysis_data['figures'], crop_type)
                for paths in figure_paths.values():
                    exported_paths['figures'].extend(paths)
            
            # Generate and export reports
            comprehensive_report = self.reporter.generate_comprehensive_report(analysis_data, crop_type)
            summary_report = self.reporter.generate_summary_report(analysis_data, crop_type)
            validation_report = self.reporter.generate_validation_report(analysis_data, crop_type)
            
            exported_paths['reports'].extend([comprehensive_report, summary_report, validation_report])
            
            # Update tracking
            self.exported_files['data'][crop_type] = exported_paths['data']
            self.exported_files['figures'][crop_type] = exported_paths['figures']
            self.exported_files['reports'][crop_type] = exported_paths['reports']
            
            logger.info(f"Complete analysis export finished for {crop_type}")
            
        except Exception as e:
            logger.error(f"Error during complete analysis export: {e}")
            raise
        
        return exported_paths
    
    def export_event_losses(self, events_data: pd.DataFrame, crop_type: str,
                          formats: List[str] = ['csv', 'xlsx']) -> List[Path]:
        """Export event losses data in multiple formats."""
        paths = []
        for format in formats:
            try:
                path = self.data_exporter.export_event_losses(events_data, crop_type, format)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to export event losses in {format}: {e}")
        return paths[0] if len(paths) == 1 else paths
    
    def export_envelope_data(self, envelope_data: Dict, crop_type: str,
                           formats: List[str] = ['csv']) -> List[Path]:
        """Export H-P envelope data in multiple formats."""
        paths = []
        for format in formats:
            try:
                path = self.data_exporter.export_envelope_data(envelope_data, crop_type, format)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to export envelope data in {format}: {e}")
        return paths[0] if len(paths) == 1 else paths
    
    def export_threshold_data(self, thresholds: Dict[str, float], 
                            sur_thresholds: Optional[Dict[int, float]], 
                            crop_type: str, formats: List[str] = ['csv']) -> List[Path]:
        """Export threshold data in multiple formats."""
        paths = []
        for format in formats:
            try:
                path = self.data_exporter.export_threshold_data(thresholds, sur_thresholds, 
                                                              crop_type, format)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to export threshold data in {format}: {e}")
        return paths[0] if len(paths) == 1 else paths
    
    def export_usda_data(self, usda_data: Dict, crop_type: str,
                        formats: List[str] = ['csv']) -> List[Path]:
        """Export USDA PSD data in multiple formats."""
        paths = []
        for format in formats:
            try:
                path = self.data_exporter.export_usda_data(usda_data, crop_type, format)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to export USDA data in {format}: {e}")
        return paths[0] if len(paths) == 1 else paths
    
    def export_analysis_summary(self, analysis_results: Dict[str, Any], 
                              crop_type: str) -> Path:
        """Export comprehensive analysis summary."""
        return self.data_exporter.export_analysis_summary(analysis_results, crop_type)
    
    def export_all_figures(self, figures: Dict[str, plt.Figure], crop_type: str,
                          formats: List[str] = ['png', 'svg', 'eps']) -> Dict[str, List[Path]]:
        """Export all figures in multiple formats."""
        return self.figure_exporter.export_all_figures(figures, crop_type, formats)
    
    def generate_analysis_report(self, analysis_data: Dict[str, Any], 
                               crop_type: str) -> Path:
        """
        Generate comprehensive analysis report.
        
        Args:
            analysis_data: Dictionary containing analysis results
            crop_type: Crop type for the report
        
        Returns:
            Path to generated report file
        """
        report_path = self.organizer.get_report_path(crop_type, 'analysis')
        
        # Ensure output directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report content
        report_content = self._generate_report_content(analysis_data, crop_type)
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated analysis report: {report_path}")
        return report_path
    
    def _generate_report_content(self, analysis_data: Dict[str, Any], 
                               crop_type: str) -> str:
        """Generate the content for the analysis report."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"AGRIRICHTER ANALYSIS REPORT - {crop_type.upper()}")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Analysis Overview
        lines.append("ANALYSIS OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"Crop Type: {crop_type}")
        
        if 'thresholds' in analysis_data:
            lines.append("Production Thresholds (kcal):")
            for threshold, value in analysis_data['thresholds'].items():
                lines.append(f"  {threshold}: {value:.2e}")
        
        if 'sur_thresholds' in analysis_data:
            lines.append("SUR Thresholds:")
            for phase, value in analysis_data['sur_thresholds'].items():
                lines.append(f"  IPC Phase {phase}: {value:.3f}")
        
        lines.append("")
        
        # Events Analysis
        if 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            lines.append("HISTORICAL EVENTS ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Number of events analyzed: {len(events_df)}")
            
            if not events_df.empty:
                lines.append(f"Total harvest area affected: {events_df['harvest_area_km2'].sum():,.0f} km²")
                lines.append(f"Total production loss: {events_df['production_loss_kcal'].sum():.2e} kcal")
                lines.append(f"Average event magnitude: {np.log10(events_df['harvest_area_km2']).mean():.2f}")
                
                # Top events by impact
                lines.append("\nTop 5 events by production loss:")
                top_events = events_df.nlargest(5, 'production_loss_kcal')
                for _, event in top_events.iterrows():
                    lines.append(f"  {event['event_name']}: {event['production_loss_kcal']:.2e} kcal")
        
        lines.append("")
        
        # Envelope Analysis
        if 'envelope_data' in analysis_data:
            envelope = analysis_data['envelope_data']
            lines.append("H-P ENVELOPE ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Disruption area range: {min(envelope['disrupted_areas']):,.0f} - {max(envelope['disrupted_areas']):,.0f} km²")
            lines.append(f"Upper bound range: {min(envelope['upper_bound']):.2e} - {max(envelope['upper_bound']):.2e} kcal")
            lines.append(f"Lower bound range: {min(envelope['lower_bound']):.2e} - {max(envelope['lower_bound']):.2e} kcal")
        
        lines.append("")
        
        # USDA Data Analysis
        if 'usda_data' in analysis_data:
            lines.append("USDA PSD DATA ANALYSIS")
            lines.append("-" * 40)
            for crop_name, crop_data in analysis_data['usda_data'].items():
                if not crop_data.empty:
                    lines.append(f"{crop_name.title()} ({len(crop_data)} years):")
                    lines.append(f"  Production range: {crop_data['Production'].min():,.0f} - {crop_data['Production'].max():,.0f} MT")
                    lines.append(f"  Consumption range: {crop_data['Consumption'].min():,.0f} - {crop_data['Consumption'].max():,.0f} MT")
                    
                    # Calculate SUR statistics
                    sur = crop_data['EndingStocks'] / crop_data['Consumption']
                    lines.append(f"  SUR mean: {sur.mean():.3f}, std: {sur.std():.3f}")
        
        lines.append("")
        
        # File Exports
        lines.append("EXPORTED FILES")
        lines.append("-" * 40)
        
        if crop_type in self.exported_files['data']:
            lines.append("Data files:")
            for path in self.exported_files['data'][crop_type]:
                lines.append(f"  {path}")
        
        if crop_type in self.exported_files['figures']:
            lines.append("Figure files:")
            for path in self.exported_files['figures'][crop_type]:
                lines.append(f"  {path}")
        
        lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def organize_existing_outputs(self, source_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """Organize existing output files into the standard structure."""
        return self.organizer.organize_existing_files(source_dir)
    
    def clean_temporary_files(self) -> int:
        """Clean temporary files and return count removed."""
        return self.organizer.clean_temp_files()
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get summary of all outputs and directory structure.
        
        Returns:
            Dictionary with output summary information
        """
        directory_info = self.organizer.get_directory_info()
        
        summary = {
            'base_directory': str(self.organizer.base_dir),
            'timestamp': self.organizer.timestamp,
            'directory_structure': directory_info,
            'exported_files_by_crop': self.exported_files,
            'total_files': sum(info['file_count'] for info in directory_info.values()),
            'total_size_mb': sum(info['size_mb'] for info in directory_info.values())
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the output manager."""
        return f"OutputManager(base_dir='{self.organizer.base_dir}')"


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output manager
    manager = OutputManager("test_outputs")
    
    # Create sample analysis data
    sample_analysis = {
        'events_data': pd.DataFrame({
            'event_name': ['Test Event 1', 'Test Event 2'],
            'harvest_area_km2': [100000, 200000],
            'production_loss_kcal': [1e14, 2e14]
        }),
        'envelope_data': {
            'disrupted_areas': np.array([100, 1000, 10000]),
            'upper_bound': np.array([1e12, 1e13, 1e14]),
            'lower_bound': np.array([5e11, 5e12, 5e13])
        },
        'thresholds': {
            'T1': 1e14,
            'T2': 5e14,
            'T3': 1.5e15,
            'T4': 3e15
        },
        'sur_thresholds': {
            2: 0.25,
            3: 0.20,
            4: 0.15,
            5: 0.10
        }
    }
    
    # Export complete analysis
    exported_paths = manager.export_complete_analysis(sample_analysis, 'wheat')
    print(f"Exported paths: {exported_paths}")
    
    # Get output summary
    summary = manager.get_output_summary()
    print(f"Output summary: {summary}")
    
    print("Output management test completed!")
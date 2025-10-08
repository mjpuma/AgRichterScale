"""End-to-end pipeline orchestrator for AgriRichter events analysis."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from agririchter.core.config import Config
from agririchter.core.performance import PerformanceMonitor
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator


class EventsPipeline:
    """
    Orchestrates the complete AgriRichter events analysis workflow.
    
    This class coordinates data loading, event calculation, visualization generation,
    and results export for historical agricultural disruption events.
    """
    
    def __init__(self, config: Config, output_dir: str, enable_performance_monitoring: bool = True):
        """
        Initialize the events pipeline.
        
        Args:
            config: Configuration object with data paths and settings
            output_dir: Directory for saving outputs
            enable_performance_monitoring: If True, enable performance monitoring
        """
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize performance monitor
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Initialize component placeholders
        self.grid_manager: Optional[GridDataManager] = None
        self.spatial_mapper: Optional[SpatialMapper] = None
        self.event_calculator: Optional[EventCalculator] = None
        
        # Data storage
        self.loaded_data: Dict[str, Any] = {}
        self.events_df: Optional[pd.DataFrame] = None
        self.figures: Dict[str, plt.Figure] = {}
        
        self.logger.info(f"EventsPipeline initialized with output directory: {self.output_dir}")
        if enable_performance_monitoring:
            self.logger.info("Performance monitoring enabled")
    
    def _setup_logging(self) -> None:
        """Configure logging for pipeline stages."""
        # Create console handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all required data for events analysis.
        
        Loads:
        - SPAM 2020 production and harvest area data
        - Event definitions from Excel files
        - Country code mapping
        - Boundary data (if available)
        
        Returns:
            Dictionary containing loaded data with keys:
            - 'grid_manager': Initialized GridDataManager
            - 'spatial_mapper': Initialized SpatialMapper
            - 'events_data': Dictionary with event definitions
        
        Raises:
            Exception: If critical data files cannot be loaded
        """
        self.logger.info("=== Stage 1: Data Loading ===")
        
        if self.performance_monitor:
            self.performance_monitor.start_stage("data_loading")
        
        try:
            # Initialize GridDataManager and load SPAM data
            self.logger.info("Loading SPAM 2020 grid data...")
            self.grid_manager = GridDataManager(self.config)
            production_df, harvest_df = self.grid_manager.load_spam_data()
            self.logger.info(f"Loaded {len(production_df)} grid cells from SPAM 2020")
            
            # Create spatial index for efficient queries
            self.logger.info("Creating spatial index...")
            self.grid_manager.create_spatial_index()
            
            # Initialize SpatialMapper
            self.logger.info("Initializing spatial mapper...")
            self.spatial_mapper = SpatialMapper(self.config, self.grid_manager)
            
            # Load country code mapping
            self.logger.info("Loading country code mapping...")
            country_mapping = self.spatial_mapper.load_country_codes_mapping()
            self.logger.info(f"Loaded {len(country_mapping)} country code mappings")
            
            # Pre-build country mappings for better performance
            self.logger.info("Pre-building country-to-grid-cell mappings...")
            self.spatial_mapper.prebuild_country_mappings()
            self.logger.info("Country mappings pre-built successfully")
            
            # Load boundary data if available
            try:
                self.logger.info("Loading boundary data...")
                self.spatial_mapper.load_boundary_data()
                self.logger.info("Boundary data loaded successfully")
            except Exception as e:
                self.logger.warning(f"Boundary data not available, will use ISO3 matching only: {e}")
            
            # Load event definitions
            self.logger.info("Loading event definitions...")
            from agririchter.data.events import EventsProcessor
            events_processor = EventsProcessor(self.config)
            
            # Load event Excel files
            from agririchter.data.loader import DataLoader
            data_loader = DataLoader(self.config)
            
            try:
                country_file = self.config.root_dir / 'ancillary' / 'DisruptionCountry.xls'
                state_file = self.config.root_dir / 'ancillary' / 'DisruptionStateProvince.xls'
                
                # Load all sheets from both files
                country_sheets = pd.read_excel(country_file, sheet_name=None, engine='xlrd')
                state_sheets = pd.read_excel(state_file, sheet_name=None, engine='xlrd')
                
                raw_events_data = {
                    'country': country_sheets,
                    'state': state_sheets
                }
                
                self.logger.info(f"Loaded {len(country_sheets)} country event sheets")
                self.logger.info(f"Loaded {len(state_sheets)} state event sheets")
                
                # Process event sheets into structured format
                self.logger.info("Processing event definitions...")
                events_data = events_processor.process_event_sheets(raw_events_data)
                self.logger.info(f"Processed {len(events_data)} events")
                
            except Exception as e:
                self.logger.error(f"Failed to load event definition files: {e}")
                raise
            
            # Store loaded data
            self.loaded_data = {
                'grid_manager': self.grid_manager,
                'spatial_mapper': self.spatial_mapper,
                'events_data': events_data,
                'production_df': production_df,
                'harvest_df': harvest_df,
                'country_mapping': country_mapping
            }
            
            self.logger.info("Data loading completed successfully")
            
            if self.performance_monitor:
                self.performance_monitor.end_stage()
                self.performance_monitor.log_memory_usage("After data loading")
            
            return self.loaded_data
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            raise
    
    def calculate_events(self) -> pd.DataFrame:
        """
        Calculate losses for all historical events.
        
        Initializes EventCalculator and processes all 21 historical events
        to calculate harvest area losses, production losses, and magnitudes.
        
        Returns:
            DataFrame with event results containing columns:
            - event_name: Name of the historical event
            - harvest_area_loss_ha: Harvest area loss in hectares
            - production_loss_kcal: Production loss in kcal
            - magnitude: AgriRichter magnitude (log10 scale)
            - affected_countries: List of affected countries
            - grid_cells_count: Number of grid cells affected
        
        Raises:
            Exception: If event calculation fails
        """
        self.logger.info("=== Stage 2: Event Calculation ===")
        
        if self.performance_monitor:
            self.performance_monitor.start_stage("event_calculation")
        
        try:
            # Ensure data is loaded
            if not self.grid_manager or not self.spatial_mapper:
                self.logger.info("Data not loaded, loading now...")
                self.load_all_data()
            
            # Initialize EventCalculator
            self.logger.info("Initializing event calculator...")
            self.event_calculator = EventCalculator(
                self.config,
                self.grid_manager,
                self.spatial_mapper
            )
            
            # Get events data
            events_data = self.loaded_data.get('events_data', {})
            if not events_data:
                raise ValueError("Events data not loaded")
            
            # Calculate all events
            self.logger.info("Calculating losses for all historical events...")
            self.events_df = self.event_calculator.calculate_all_events(events_data)
            
            # Log summary statistics
            total_events = len(self.events_df)
            total_harvest_loss = self.events_df['harvest_area_loss_ha'].sum()
            total_production_loss = self.events_df['production_loss_kcal'].sum()
            
            self.logger.info(f"Event calculation completed:")
            self.logger.info(f"  - Total events processed: {total_events}")
            self.logger.info(f"  - Total harvest area loss: {total_harvest_loss:,.0f} ha")
            self.logger.info(f"  - Total production loss: {total_production_loss:,.0f} kcal")
            self.logger.info(f"  - Magnitude range: {self.events_df['magnitude'].min():.2f} - {self.events_df['magnitude'].max():.2f}")
            
            if self.performance_monitor:
                self.performance_monitor.end_stage()
                self.performance_monitor.log_memory_usage("After event calculation")
            
            return self.events_df
            
        except Exception as e:
            self.logger.error(f"Event calculation failed: {e}")
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            raise
    
    def generate_visualizations(self, events_df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate all publication-quality visualizations.
        
        Creates three key figures:
        1. Global production map
        2. H-P Envelope with real events
        3. AgriRichter Scale with real events
        
        Args:
            events_df: DataFrame with calculated event results
            
        Returns:
            Dictionary of figure objects with keys:
            - 'production_map': Global production map figure
            - 'hp_envelope': H-P envelope figure
            - 'agririchter_scale': AgriRichter scale figure
        
        Raises:
            Exception: If visualization generation fails
        """
        self.logger.info("=== Stage 3: Visualization Generation ===")
        
        if self.performance_monitor:
            self.performance_monitor.start_stage("visualization_generation")
        
        try:
            figures = {}
            
            # Import visualization modules
            from agririchter.visualization.maps import GlobalProductionMapper
            from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer
            from agririchter.visualization.agririchter_scale import AgriRichterScaleVisualizer
            from agririchter.analysis.envelope import HPEnvelopeCalculator
            
            # 1. Generate global production map
            self.logger.info("Generating global production map...")
            try:
                mapper = GlobalProductionMapper(self.config)
                production_df = self.loaded_data.get('production_df')
                if production_df is not None:
                    fig_map = mapper.create_global_map(
                        production_df,
                        title=f"Global {self.config.crop_type.title()} Production"
                    )
                    figures['production_map'] = fig_map
                    self.logger.info("Global production map created successfully")
                else:
                    self.logger.warning("Production data not available, skipping map")
            except Exception as e:
                self.logger.warning(f"Failed to create production map: {e}")
            
            # 2. Generate H-P Envelope with real events
            self.logger.info("Generating H-P Envelope with real events...")
            try:
                # Calculate envelope
                envelope_calc = HPEnvelopeCalculator(self.config)
                envelope_data = envelope_calc.calculate_hp_envelope(
                    self.loaded_data.get('production_df'),
                    self.loaded_data.get('harvest_df')
                )
                
                # Create visualization
                hp_viz = HPEnvelopeVisualizer(self.config)
                fig_hp = hp_viz.create_hp_envelope_plot(envelope_data, events_df)
                figures['hp_envelope'] = fig_hp
                self.logger.info("H-P Envelope created successfully")
            except Exception as e:
                self.logger.warning(f"Failed to create H-P Envelope: {e}")
            
            # 3. Generate AgriRichter Scale with real events
            self.logger.info("Generating AgriRichter Scale with real events...")
            try:
                scale_viz = AgriRichterScaleVisualizer(self.config)
                fig_scale = scale_viz.create_agririchter_scale_plot(events_df)
                figures['agririchter_scale'] = fig_scale
                self.logger.info("AgriRichter Scale created successfully")
            except Exception as e:
                self.logger.warning(f"Failed to create AgriRichter Scale: {e}")
            
            # Store figures
            self.figures = figures
            
            self.logger.info(f"Visualization generation completed: {len(figures)} figures created")
            
            if self.performance_monitor:
                self.performance_monitor.end_stage()
                self.performance_monitor.log_memory_usage("After visualization generation")
            
            return figures
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            raise
    
    def export_results(self, events_df: pd.DataFrame, figures: Dict[str, plt.Figure]) -> Dict[str, List[str]]:
        """
        Export results to organized directory structure.
        
        Creates organized output structure:
        - data/: CSV files with event results
        - figures/: Visualization files in multiple formats
        - reports/: Summary reports
        
        Args:
            events_df: DataFrame with event results
            figures: Dictionary of figure objects
            
        Returns:
            Dictionary of exported file paths with keys:
            - 'csv_files': List of CSV file paths
            - 'figure_files': List of figure file paths
            - 'report_files': List of report file paths
        
        Raises:
            Exception: If export fails
        """
        self.logger.info("=== Stage 4: Results Export ===")
        
        if self.performance_monitor:
            self.performance_monitor.start_stage("results_export")
        
        try:
            exported_files = {
                'csv_files': [],
                'figure_files': [],
                'report_files': []
            }
            
            # Create output directory structure
            self.output_dir.mkdir(parents=True, exist_ok=True)
            data_dir = self.output_dir / 'data'
            figures_dir = self.output_dir / 'figures'
            reports_dir = self.output_dir / 'reports'
            
            data_dir.mkdir(exist_ok=True)
            figures_dir.mkdir(exist_ok=True)
            reports_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Created output directory structure at {self.output_dir}")
            
            # Export events DataFrame to CSV
            if events_df is not None and len(events_df) > 0:
                crop_type = self.config.crop_type
                csv_filename = f"events_{crop_type}_spam2020.csv"
                csv_path = data_dir / csv_filename
                
                self.logger.info(f"Exporting events data to {csv_path}")
                events_df.to_csv(csv_path, index=False)
                exported_files['csv_files'].append(str(csv_path))
                self.logger.info(f"Saved events data: {csv_path}")
            
            # Export figures in multiple formats
            figure_formats = ['svg', 'eps', 'jpg', 'png']
            
            for fig_name, fig in figures.items():
                if fig is not None:
                    for fmt in figure_formats:
                        fig_filename = f"{fig_name}_{self.config.crop_type}.{fmt}"
                        fig_path = figures_dir / fig_filename
                        
                        try:
                            # Set appropriate DPI for raster formats
                            dpi = 300 if fmt in ['jpg', 'png'] else None
                            
                            self.logger.info(f"Saving {fig_name} as {fmt}...")
                            fig.savefig(fig_path, format=fmt, dpi=dpi, bbox_inches='tight')
                            exported_files['figure_files'].append(str(fig_path))
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to save {fig_name} as {fmt}: {e}")
                    
                    self.logger.info(f"Saved {fig_name} in {len(figure_formats)} formats")
            
            # Log export summary
            total_files = sum(len(files) for files in exported_files.values())
            self.logger.info(f"Results export completed:")
            self.logger.info(f"  - CSV files: {len(exported_files['csv_files'])}")
            self.logger.info(f"  - Figure files: {len(exported_files['figure_files'])}")
            self.logger.info(f"  - Total files exported: {total_files}")
            
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Results export failed: {e}")
            if self.performance_monitor:
                self.performance_monitor.end_stage()
            raise
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive summary report.
        
        Creates a detailed report including:
        - Pipeline execution summary
        - Event statistics (total events, losses, magnitude ranges)
        - Generated files list
        - Data quality metrics
        - Validation results
        
        Args:
            results: Dictionary with pipeline results containing:
                - events_df: DataFrame with event results
                - exported_files: Dictionary of exported file paths
                - figures: Dictionary of figure objects
                
        Returns:
            Summary report as string
        
        Raises:
            Exception: If report generation fails
        """
        self.logger.info("=== Stage 5: Summary Report Generation ===")
        
        try:
            from datetime import datetime
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("AgriRichter Events Analysis Pipeline - Summary Report")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Crop Type: {self.config.crop_type}")
            report_lines.append(f"SPAM Version: {self.config.spam_version}")
            report_lines.append("")
            
            # Event Statistics
            events_df = results.get('events_df')
            if events_df is not None and len(events_df) > 0:
                report_lines.append("-" * 80)
                report_lines.append("EVENT STATISTICS")
                report_lines.append("-" * 80)
                report_lines.append(f"Total Events Processed: {len(events_df)}")
                report_lines.append("")
                
                # Harvest area statistics
                total_harvest_loss = events_df['harvest_area_loss_ha'].sum()
                mean_harvest_loss = events_df['harvest_area_loss_ha'].mean()
                max_harvest_loss = events_df['harvest_area_loss_ha'].max()
                min_harvest_loss = events_df['harvest_area_loss_ha'].min()
                
                report_lines.append("Harvest Area Loss (hectares):")
                report_lines.append(f"  Total:   {total_harvest_loss:,.0f} ha")
                report_lines.append(f"  Mean:    {mean_harvest_loss:,.0f} ha")
                report_lines.append(f"  Maximum: {max_harvest_loss:,.0f} ha")
                report_lines.append(f"  Minimum: {min_harvest_loss:,.0f} ha")
                report_lines.append("")
                
                # Production loss statistics
                total_production_loss = events_df['production_loss_kcal'].sum()
                mean_production_loss = events_df['production_loss_kcal'].mean()
                max_production_loss = events_df['production_loss_kcal'].max()
                min_production_loss = events_df['production_loss_kcal'].min()
                
                report_lines.append("Production Loss (kcal):")
                report_lines.append(f"  Total:   {total_production_loss:,.0f} kcal")
                report_lines.append(f"  Mean:    {mean_production_loss:,.0f} kcal")
                report_lines.append(f"  Maximum: {max_production_loss:,.0f} kcal")
                report_lines.append(f"  Minimum: {min_production_loss:,.0f} kcal")
                report_lines.append("")
                
                # Magnitude statistics
                mean_magnitude = events_df['magnitude'].mean()
                max_magnitude = events_df['magnitude'].max()
                min_magnitude = events_df['magnitude'].min()
                
                report_lines.append("AgriRichter Magnitude:")
                report_lines.append(f"  Mean:    {mean_magnitude:.2f}")
                report_lines.append(f"  Maximum: {max_magnitude:.2f}")
                report_lines.append(f"  Minimum: {min_magnitude:.2f}")
                report_lines.append("")
            
            # Generated Files
            exported_files = results.get('exported_files', {})
            if exported_files:
                report_lines.append("-" * 80)
                report_lines.append("GENERATED FILES")
                report_lines.append("-" * 80)
                
                csv_files = exported_files.get('csv_files', [])
                if csv_files:
                    report_lines.append(f"CSV Files ({len(csv_files)}):")
                    for file_path in csv_files:
                        report_lines.append(f"  - {file_path}")
                    report_lines.append("")
                
                figure_files = exported_files.get('figure_files', [])
                if figure_files:
                    report_lines.append(f"Figure Files ({len(figure_files)}):")
                    for file_path in figure_files:
                        report_lines.append(f"  - {file_path}")
                    report_lines.append("")
                
                report_files = exported_files.get('report_files', [])
                if report_files:
                    report_lines.append(f"Report Files ({len(report_files)}):")
                    for file_path in report_files:
                        report_lines.append(f"  - {file_path}")
                    report_lines.append("")
            
            # Data Quality Metrics
            if events_df is not None and len(events_df) > 0:
                report_lines.append("-" * 80)
                report_lines.append("DATA QUALITY METRICS")
                report_lines.append("-" * 80)
                
                # Check for events with zero losses
                zero_harvest = (events_df['harvest_area_loss_ha'] == 0).sum()
                zero_production = (events_df['production_loss_kcal'] == 0).sum()
                
                report_lines.append(f"Events with zero harvest area loss: {zero_harvest}")
                report_lines.append(f"Events with zero production loss: {zero_production}")
                
                # Check for NaN values
                nan_harvest = events_df['harvest_area_loss_ha'].isna().sum()
                nan_production = events_df['production_loss_kcal'].isna().sum()
                nan_magnitude = events_df['magnitude'].isna().sum()
                
                report_lines.append(f"Events with NaN harvest area loss: {nan_harvest}")
                report_lines.append(f"Events with NaN production loss: {nan_production}")
                report_lines.append(f"Events with NaN magnitude: {nan_magnitude}")
                report_lines.append("")
            
            # Performance Metrics
            if self.performance_monitor:
                report_lines.append("-" * 80)
                report_lines.append("PERFORMANCE METRICS")
                report_lines.append("-" * 80)
                
                total_time = self.performance_monitor.get_total_pipeline_time()
                report_lines.append(f"Total Pipeline Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                report_lines.append("")
                
                # Stage timings
                all_metrics = self.performance_monitor.get_all_metrics()
                if all_metrics:
                    report_lines.append("Stage Timings:")
                    for stage_name, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['timestamp']):
                        elapsed = metrics['elapsed_time_seconds']
                        memory_delta = metrics['memory_delta_mb']
                        report_lines.append(f"  {stage_name}:")
                        report_lines.append(f"    Time:   {elapsed:.2f}s ({elapsed/60:.2f}m)")
                        report_lines.append(f"    Memory: {memory_delta:+.2f} MB")
                    report_lines.append("")
                
                # Performance assessment
                if total_time < 600:
                    report_lines.append("✓ Pipeline completed within target time (< 10 minutes)")
                else:
                    report_lines.append("⚠ Pipeline exceeded target time of 10 minutes")
                report_lines.append("")
            
            # Pipeline Summary
            report_lines.append("-" * 80)
            report_lines.append("PIPELINE EXECUTION SUMMARY")
            report_lines.append("-" * 80)
            report_lines.append(f"Output Directory: {self.output_dir}")
            report_lines.append(f"Status: Completed Successfully")
            report_lines.append("")
            
            report_lines.append("=" * 80)
            
            # Join all lines into report string
            report = "\n".join(report_lines)
            
            # Save report to file
            reports_dir = self.output_dir / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"pipeline_summary_{self.config.crop_type}.txt"
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Summary report saved to: {report_path}")
            
            # Also log the report
            self.logger.info("\n" + report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Summary report generation failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete pipeline from data loading to export.
        
        Orchestrates all pipeline stages in sequence:
        1. Load all required data (SPAM, events, boundaries)
        2. Calculate event losses and magnitudes
        3. Generate visualizations
        4. Export results to organized directory structure
        5. Generate summary report
        
        Returns:
            Comprehensive results dictionary with keys:
            - 'events_df': DataFrame with event results
            - 'figures': Dictionary of figure objects
            - 'exported_files': Dictionary of exported file paths
            - 'summary_report': Summary report string
            - 'performance_report': Performance report string (if monitoring enabled)
            - 'status': Pipeline execution status
            - 'errors': List of non-critical errors encountered
        
        Raises:
            Exception: If critical pipeline stage fails
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting AgriRichter Events Analysis Pipeline")
        self.logger.info("=" * 60)
        
        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_pipeline()
            self.performance_monitor.log_memory_usage("Pipeline start")
        
        results = {
            'events_df': None,
            'figures': {},
            'exported_files': {},
            'summary_report': '',
            'performance_report': '',
            'status': 'running',
            'errors': []
        }
        
        try:
            # Stage 1: Load all data
            try:
                self.load_all_data()
                self.logger.info("✓ Data loading completed successfully")
            except Exception as e:
                self.logger.error(f"✗ Data loading failed: {e}")
                raise
            
            # Stage 2: Calculate events
            try:
                events_df = self.calculate_events()
                results['events_df'] = events_df
                self.logger.info("✓ Event calculation completed successfully")
            except Exception as e:
                self.logger.error(f"✗ Event calculation failed: {e}")
                raise
            
            # Stage 3: Generate visualizations
            try:
                figures = self.generate_visualizations(events_df)
                results['figures'] = figures
                self.logger.info("✓ Visualization generation completed successfully")
            except Exception as e:
                self.logger.warning(f"⚠ Visualization generation encountered errors: {e}")
                results['errors'].append(f"Visualization: {e}")
                # Continue with partial results
            
            # Stage 4: Export results
            try:
                exported_files = self.export_results(events_df, results['figures'])
                results['exported_files'] = exported_files
                self.logger.info("✓ Results export completed successfully")
            except Exception as e:
                self.logger.warning(f"⚠ Results export encountered errors: {e}")
                results['errors'].append(f"Export: {e}")
                # Continue to generate report
            
            # Stage 5: Generate summary report
            try:
                summary_report = self.generate_summary_report(results)
                results['summary_report'] = summary_report
                self.logger.info("✓ Summary report generated successfully")
            except Exception as e:
                self.logger.warning(f"⚠ Summary report generation encountered errors: {e}")
                results['errors'].append(f"Report: {e}")
            
            # Generate and save performance report
            if self.performance_monitor:
                try:
                    self.performance_monitor.log_memory_usage("Pipeline end")
                    performance_report = self.performance_monitor.generate_performance_report()
                    results['performance_report'] = performance_report
                    
                    # Save performance report to file
                    reports_dir = self.output_dir / 'reports'
                    reports_dir.mkdir(parents=True, exist_ok=True)
                    perf_report_path = reports_dir / f"performance_{self.config.crop_type}.txt"
                    self.performance_monitor.save_report(str(perf_report_path))
                    
                    # Also log it
                    self.logger.info("\n" + performance_report)
                    
                except Exception as e:
                    self.logger.warning(f"⚠ Performance report generation encountered errors: {e}")
            
            # Update status
            if len(results['errors']) == 0:
                results['status'] = 'completed'
                self.logger.info("=" * 60)
                self.logger.info("✓ Pipeline completed successfully!")
                self.logger.info("=" * 60)
            else:
                results['status'] = 'completed_with_warnings'
                self.logger.warning("=" * 60)
                self.logger.warning(f"⚠ Pipeline completed with {len(results['errors'])} warnings")
                self.logger.warning("=" * 60)
            
            return results
            
        except Exception as e:
            results['status'] = 'failed'
            self.logger.error("=" * 60)
            self.logger.error(f"✗ Pipeline failed: {e}")
            self.logger.error("=" * 60)
            
            # Still try to generate performance report on failure
            if self.performance_monitor:
                try:
                    performance_report = self.performance_monitor.generate_performance_report()
                    results['performance_report'] = performance_report
                    self.logger.info("\n" + performance_report)
                except:
                    pass
            
            raise

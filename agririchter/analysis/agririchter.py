"""AgriRichter Scale analysis and magnitude calculations."""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np

from ..core.base import BaseAnalyzer
from ..core.config import Config
from ..processing.converters import UnitConverter
from .envelope import HPEnvelopeCalculator


class AgriRichterError(Exception):
    """Exception raised for AgriRichter analysis errors."""
    pass


class AgriRichterAnalyzer(BaseAnalyzer):
    """Main analyzer for AgriRichter Scale calculations."""
    
    def __init__(self, config: Config):
        """
        Initialize AgriRichter analyzer.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agririchter.analyzer')
        self.converter = UnitConverter(config)
        self.envelope_calculator = HPEnvelopeCalculator(config)
        
        # Get thresholds for current crop type
        self.thresholds = config.get_thresholds()
        self.threshold_log10 = {k: np.log10(v) for k, v in self.thresholds.items()}
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main analysis entry point.
        
        Args:
            data: Input data for analysis
        
        Returns:
            Analysis results dictionary
        """
        # This is a generic interface - specific analysis methods below
        return {}
    
    def calculate_magnitude(self, harvest_area_km2: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate AgriRichter magnitude from disrupted harvest area.
        
        The AgriRichter magnitude is defined as:
        M_D = log10(disrupted harvest area in km²)
        
        Args:
            harvest_area_km2: Disrupted harvest area in km²
        
        Returns:
            AgriRichter magnitude (log10 scale)
        """
        try:
            magnitude = self.converter.convert_harvest_area_to_magnitude(harvest_area_km2)
            
            if isinstance(magnitude, (np.ndarray, pd.Series)):
                self.logger.debug(f"Calculated magnitude for {len(magnitude)} events")
            else:
                self.logger.debug(f"Calculated magnitude: {magnitude:.2f}")
            
            return magnitude
            
        except Exception as e:
            raise AgriRichterError(f"Failed to calculate magnitude: {str(e)}")
    
    def classify_event_severity(self, production_loss_kcal: Union[float, np.ndarray, pd.Series]) -> Union[str, List[str]]:
        """
        Classify event severity based on production loss and thresholds.
        
        Args:
            production_loss_kcal: Production loss in kcal
        
        Returns:
            Severity classification ('T1', 'T2', 'T3', 'T4', or 'Below T1')
        """
        try:
            if isinstance(production_loss_kcal, (np.ndarray, pd.Series)):
                # Handle arrays
                classifications = []
                for loss in production_loss_kcal:
                    classifications.append(self._classify_single_event(loss))
                return classifications
            else:
                # Handle single value
                return self._classify_single_event(production_loss_kcal)
                
        except Exception as e:
            raise AgriRichterError(f"Failed to classify event severity: {str(e)}")
    
    def _classify_single_event(self, production_loss: float) -> str:
        """Classify a single event's severity."""
        if production_loss >= self.thresholds['T4']:
            return 'T4'
        elif production_loss >= self.thresholds['T3']:
            return 'T3'
        elif production_loss >= self.thresholds['T2']:
            return 'T2'
        elif production_loss >= self.thresholds['T1']:
            return 'T1'
        else:
            return 'Below T1'
    
    def process_historical_events(self, events_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process historical events data for AgriRichter analysis.
        
        Args:
            events_data: DataFrame with historical event data
                        Expected columns: 'harvest_area_loss_ha', 'production_loss_kcal'
        
        Returns:
            DataFrame with AgriRichter analysis results
        """
        try:
            self.logger.info("Processing historical events for AgriRichter analysis")
            
            # Validate input data
            required_columns = ['harvest_area_loss_ha', 'production_loss_kcal']
            missing_columns = [col for col in required_columns if col not in events_data.columns]
            
            if missing_columns:
                raise AgriRichterError(f"Missing required columns: {missing_columns}")
            
            # Create results DataFrame
            results = events_data.copy()
            
            # Convert harvest area to km²
            results['harvest_area_loss_km2'] = self.converter.convert_harvest_area_to_km2(
                results['harvest_area_loss_ha']
            )
            
            # Calculate AgriRichter magnitude
            results['magnitude'] = self.calculate_magnitude(results['harvest_area_loss_km2'])
            
            # Classify severity
            results['severity_class'] = self.classify_event_severity(results['production_loss_kcal'])
            
            # Add threshold information
            for threshold_name, threshold_value in self.thresholds.items():
                results[f'{threshold_name}_threshold'] = threshold_value
                results[f'{threshold_name}_log10'] = self.threshold_log10[threshold_name]
            
            # Calculate relative magnitude (compared to threshold)
            results['magnitude_relative_T1'] = results['magnitude'] - self.threshold_log10['T1']
            
            # Add crop type information
            results['crop_type'] = self.config.crop_type
            
            self.logger.info(f"Processed {len(results)} historical events")
            
            return results
            
        except Exception as e:
            raise AgriRichterError(f"Failed to process historical events: {str(e)}")
    
    def create_richter_scale_data(self, min_magnitude: float = 0.0, 
                                 max_magnitude: float = 8.0, 
                                 n_points: int = 10000) -> Dict[str, np.ndarray]:
        """
        Create data for plotting the AgriRichter scale curve following MATLAB algorithm.
        
        MATLAB algorithm:
        prod_range = logspace(10, 15.2, 10000);
        threshold = 1*gramsperMetricTon*calories_cropprod;
        X = log10(prod_range/threshold);
        Y = prod_range;
        
        Args:
            min_magnitude: Minimum magnitude for scale
            max_magnitude: Maximum magnitude for scale  
            n_points: Number of points in the scale
        
        Returns:
            Dictionary with scale data
        """
        try:
            # Get crop-specific parameters
            caloric_content = self.config.get_caloric_content()
            unit_conversions = self.config.get_unit_conversions()
            
            # Calculate threshold (1 metric ton of crop in kcal)
            # MATLAB: threshold = 1*gramsperMetricTon*calories_cropprod;
            threshold_mt = 1.0  # 1 metric ton baseline
            threshold_kcal = threshold_mt * unit_conversions['grams_per_metric_ton'] * caloric_content
            
            # Create production range following MATLAB
            # MATLAB: prod_range = logspace(10, 15.2, 10000); (for wheat)
            if self.config.crop_type == 'wheat':
                prod_range = np.logspace(10, 15.2, n_points)
            elif self.config.crop_type == 'rice':
                prod_range = np.logspace(10, 15.0, n_points)
            elif self.config.crop_type == 'allgrain':
                prod_range = np.logspace(10, 15.9, n_points)
            else:
                prod_range = np.logspace(10, 15.2, n_points)  # Default
            
            # Calculate magnitudes following MATLAB
            # MATLAB: X = log10(prod_range/threshold);
            magnitudes = np.log10(prod_range / threshold_kcal)
            
            # Production values are the range itself
            # MATLAB: Y = prod_range;
            production_kcal = prod_range
            
            self.logger.info(f"Created AgriRichter scale: {len(magnitudes)} points")
            self.logger.info(f"Magnitude range: {magnitudes.min():.2f} - {magnitudes.max():.2f}")
            self.logger.info(f"Production range: {production_kcal.min():.2e} - {production_kcal.max():.2e} kcal")
            self.logger.info(f"Threshold: {threshold_kcal:.2e} kcal")
            
            return {
                'magnitudes': magnitudes,
                'production_kcal': production_kcal,
                'threshold_kcal': threshold_kcal,
                'crop_type': self.config.crop_type
            }
            
        except Exception as e:
            raise AgriRichterError(f"Failed to create Richter scale data: {str(e)}")
    
    def analyze_event_distribution(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of historical events across severity classes.
        
        Args:
            events_df: DataFrame with processed historical events
        
        Returns:
            Dictionary with distribution analysis
        """
        try:
            analysis = {
                'total_events': len(events_df),
                'crop_type': self.config.crop_type,
                'severity_distribution': {},
                'magnitude_statistics': {},
                'production_statistics': {}
            }
            
            # Severity class distribution
            severity_counts = events_df['severity_class'].value_counts()
            for severity, count in severity_counts.items():
                analysis['severity_distribution'][severity] = {
                    'count': int(count),
                    'percentage': float(count / len(events_df) * 100)
                }
            
            # Magnitude statistics
            magnitudes = events_df['magnitude'].dropna()
            if len(magnitudes) > 0:
                analysis['magnitude_statistics'] = {
                    'min': float(magnitudes.min()),
                    'max': float(magnitudes.max()),
                    'mean': float(magnitudes.mean()),
                    'median': float(magnitudes.median()),
                    'std': float(magnitudes.std())
                }
            
            # Production loss statistics
            production_losses = events_df['production_loss_kcal'].dropna()
            if len(production_losses) > 0:
                analysis['production_statistics'] = {
                    'min': float(production_losses.min()),
                    'max': float(production_losses.max()),
                    'mean': float(production_losses.mean()),
                    'median': float(production_losses.median()),
                    'total': float(production_losses.sum())
                }
            
            # Threshold analysis
            analysis['threshold_analysis'] = {}
            for threshold_name, threshold_value in self.thresholds.items():
                above_threshold = (events_df['production_loss_kcal'] >= threshold_value).sum()
                analysis['threshold_analysis'][threshold_name] = {
                    'threshold_value': float(threshold_value),
                    'events_above': int(above_threshold),
                    'percentage_above': float(above_threshold / len(events_df) * 100)
                }
            
            return analysis
            
        except Exception as e:
            raise AgriRichterError(f"Failed to analyze event distribution: {str(e)}")
    
    def run_complete_analysis(self, production_kcal: pd.DataFrame,
                             harvest_km2: pd.DataFrame,
                             events_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run complete AgriRichter analysis pipeline.
        
        Args:
            production_kcal: Production data in kcal
            harvest_km2: Harvest area data in km²
            events_data: Optional historical events data
        
        Returns:
            Complete analysis results
        """
        try:
            self.logger.info("Starting complete AgriRichter analysis")
            
            results = {
                'crop_type': self.config.crop_type,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 1. Calculate H-P envelope
            self.logger.info("Calculating H-P envelope...")
            envelope_data = self.envelope_calculator.calculate_hp_envelope(
                production_kcal, harvest_km2
            )
            results['envelope'] = envelope_data
            
            # 2. Create Richter scale data
            self.logger.info("Creating Richter scale data...")
            production_range = self.config.production_range
            scale_data = self.create_richter_scale_data(
                min_magnitude=production_range[0] - 10,  # Extend range
                max_magnitude=production_range[1] - 10,
                n_points=1000
            )
            results['richter_scale'] = scale_data
            
            # 3. Process historical events (if provided)
            if events_data is not None:
                self.logger.info("Processing historical events...")
                processed_events = self.process_historical_events(events_data)
                results['historical_events'] = processed_events
                
                # 4. Analyze event distribution
                event_analysis = self.analyze_event_distribution(processed_events)
                results['event_analysis'] = event_analysis
            
            # 5. Generate summary statistics
            results['summary'] = self._generate_analysis_summary(results)
            
            self.logger.info("Complete AgriRichter analysis finished successfully")
            
            return results
            
        except Exception as e:
            raise AgriRichterError(f"Complete analysis failed: {str(e)}")
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        summary = {
            'crop_type': results['crop_type'],
            'analysis_components': list(results.keys())
        }
        
        # Envelope summary
        if 'envelope' in results:
            envelope_stats = self.envelope_calculator.get_envelope_statistics(results['envelope'])
            summary['envelope_summary'] = {
                'disruption_points': envelope_stats['n_disruption_points'],
                'max_disruption_area': envelope_stats['max_disruption_area'],
                'production_range': [
                    envelope_stats['lower_bound_stats']['min_production'],
                    envelope_stats['upper_bound_stats']['max_production']
                ]
            }
        
        # Events summary
        if 'historical_events' in results:
            events_df = results['historical_events']
            summary['events_summary'] = {
                'total_events': len(events_df),
                'magnitude_range': [
                    float(events_df['magnitude'].min()),
                    float(events_df['magnitude'].max())
                ],
                'severity_classes': events_df['severity_class'].value_counts().to_dict()
            }
        
        # Thresholds summary
        summary['thresholds'] = self.thresholds
        summary['threshold_log10'] = self.threshold_log10
        
        return summary
    
    def create_analysis_report(self, results: Dict[str, Any]) -> str:
        """
        Create comprehensive analysis report.
        
        Args:
            results: Complete analysis results
        
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 70,
            "AgriRichter Scale Analysis Report",
            "=" * 70,
            f"Crop Type: {results['crop_type'].upper()}",
            f"Analysis Date: {results['analysis_timestamp']}",
            ""
        ]
        
        # Thresholds section
        report_lines.extend([
            "AGRIRICHTER THRESHOLDS",
            "-" * 23,
        ])
        
        for threshold_name, threshold_value in self.thresholds.items():
            log_value = self.threshold_log10[threshold_name]
            report_lines.append(f"{threshold_name}: {threshold_value:.2e} kcal (log10: {log_value:.2f})")
        
        report_lines.append("")
        
        # Envelope section
        if 'envelope' in results:
            envelope_stats = self.envelope_calculator.get_envelope_statistics(results['envelope'])
            report_lines.extend([
                "H-P ENVELOPE ANALYSIS",
                "-" * 20,
                f"Disruption Points: {envelope_stats['n_disruption_points']}",
                f"Area Range: {envelope_stats['min_disruption_area']:,.0f} - {envelope_stats['max_disruption_area']:,.0f} km²",
                f"Production Range: {envelope_stats['lower_bound_stats']['min_production']:.2e} - {envelope_stats['upper_bound_stats']['max_production']:.2e} kcal",
                ""
            ])
        
        # Historical events section
        if 'historical_events' in results and 'event_analysis' in results:
            event_analysis = results['event_analysis']
            report_lines.extend([
                "HISTORICAL EVENTS ANALYSIS",
                "-" * 27,
                f"Total Events: {event_analysis['total_events']}",
                ""
            ])
            
            # Severity distribution
            report_lines.append("Severity Distribution:")
            for severity, data in event_analysis['severity_distribution'].items():
                report_lines.append(f"  {severity}: {data['count']} events ({data['percentage']:.1f}%)")
            
            report_lines.append("")
            
            # Magnitude statistics
            if 'magnitude_statistics' in event_analysis:
                mag_stats = event_analysis['magnitude_statistics']
                report_lines.extend([
                    "Magnitude Statistics:",
                    f"  Range: {mag_stats['min']:.2f} - {mag_stats['max']:.2f}",
                    f"  Mean: {mag_stats['mean']:.2f} ± {mag_stats['std']:.2f}",
                    f"  Median: {mag_stats['median']:.2f}",
                    ""
                ])
        
        return "\n".join(report_lines)
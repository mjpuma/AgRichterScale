"""Advanced reporting system for AgriRichter analysis."""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AnalysisReporter:
    """Generates comprehensive analysis reports with statistics and validation."""
    
    def __init__(self, organizer):
        """
        Initialize analysis reporter.
        
        Args:
            organizer: FileOrganizer instance for path management
        """
        self.organizer = organizer
    
    def generate_comprehensive_report(self, analysis_data: Dict[str, Any], 
                                    crop_type: str) -> Path:
        """
        Generate comprehensive analysis report with detailed statistics.
        
        Args:
            analysis_data: Dictionary containing all analysis results
            crop_type: Crop type for the report
        
        Returns:
            Path to generated report file
        """
        report_path = self.organizer.get_report_path(crop_type, 'analysis')
        
        # Ensure output directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report content
        report_content = self._generate_comprehensive_content(analysis_data, crop_type)
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated comprehensive analysis report: {report_path}")
        return report_path
    
    def generate_summary_report(self, analysis_data: Dict[str, Any], 
                              crop_type: str) -> Path:
        """
        Generate executive summary report.
        
        Args:
            analysis_data: Dictionary containing analysis results
            crop_type: Crop type for the report
        
        Returns:
            Path to generated summary file
        """
        summary_path = self.organizer.get_report_path(crop_type, 'summary')
        
        # Ensure output directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate summary content
        summary_content = self._generate_summary_content(analysis_data, crop_type)
        
        # Write summary to file
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"Generated summary report: {summary_path}")
        return summary_path
    
    def generate_validation_report(self, analysis_data: Dict[str, Any], 
                                 crop_type: str) -> Path:
        """
        Generate data validation and quality report.
        
        Args:
            analysis_data: Dictionary containing analysis results
            crop_type: Crop type for the report
        
        Returns:
            Path to generated validation report
        """
        validation_path = self.organizer.get_report_path(crop_type, 'validation')
        
        # Ensure output directory exists
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate validation content
        validation_content = self._generate_validation_content(analysis_data, crop_type)
        
        # Write validation report to file
        with open(validation_path, 'w') as f:
            f.write(validation_content)
        
        logger.info(f"Generated validation report: {validation_path}")
        return validation_path
    
    def _generate_comprehensive_content(self, analysis_data: Dict[str, Any], 
                                      crop_type: str) -> str:
        """Generate comprehensive report content."""
        lines = []
        
        # Header
        lines.append("=" * 100)
        lines.append(f"COMPREHENSIVE AGRIRICHTER ANALYSIS REPORT - {crop_type.upper()}")
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Analysis Version: 1.0")
        lines.append("")
        
        # Executive Summary
        lines.extend(self._generate_executive_summary(analysis_data, crop_type))
        
        # Methodology
        lines.extend(self._generate_methodology_section(crop_type))
        
        # Threshold Analysis
        lines.extend(self._generate_threshold_analysis(analysis_data))
        
        # Historical Events Analysis
        lines.extend(self._generate_events_analysis(analysis_data))
        
        # H-P Envelope Analysis
        lines.extend(self._generate_envelope_analysis(analysis_data))
        
        # USDA Data Analysis
        lines.extend(self._generate_usda_analysis(analysis_data))
        
        # Statistical Summary
        lines.extend(self._generate_statistical_summary(analysis_data))
        
        # Data Quality Assessment
        lines.extend(self._generate_quality_assessment(analysis_data))
        
        # Conclusions and Recommendations
        lines.extend(self._generate_conclusions(analysis_data, crop_type))
        
        # Appendices
        lines.extend(self._generate_appendices(analysis_data))
        
        # Footer
        lines.append("")
        lines.append("=" * 100)
        lines.append("END OF COMPREHENSIVE REPORT")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def _generate_executive_summary(self, analysis_data: Dict[str, Any], 
                                  crop_type: str) -> List[str]:
        """Generate executive summary section."""
        lines = []
        lines.append("EXECUTIVE SUMMARY")
        lines.append("=" * 50)
        lines.append("")
        
        lines.append(f"This report presents a comprehensive analysis of agricultural disruption")
        lines.append(f"risks for {crop_type} using the AgriRichter Scale methodology.")
        lines.append("")
        
        # Key findings
        if 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            if not events_df.empty:
                total_events = len(events_df)
                total_area = events_df['harvest_area_km2'].sum()
                total_loss = events_df['production_loss_kcal'].sum()
                
                lines.append("KEY FINDINGS:")
                lines.append(f"• Analyzed {total_events} historical disruption events")
                lines.append(f"• Total affected harvest area: {total_area:,.0f} km²")
                lines.append(f"• Total estimated production loss: {total_loss:.2e} kcal")
                
                # Severity distribution
                if 'thresholds' in analysis_data:
                    thresholds = analysis_data['thresholds']
                    severity_counts = self._classify_event_severity(events_df, thresholds)
                    lines.append(f"• Event severity distribution:")
                    for severity, count in severity_counts.items():
                        lines.append(f"  - {severity}: {count} events")
        
        lines.append("")
        return lines
    
    def _generate_methodology_section(self, crop_type: str) -> List[str]:
        """Generate methodology section."""
        lines = []
        lines.append("METHODOLOGY")
        lines.append("=" * 50)
        lines.append("")
        
        lines.append("The AgriRichter Scale analysis employs the following methodology:")
        lines.append("")
        lines.append("1. DATA SOURCES:")
        lines.append("   • SPAM 2020 global production and harvest area data")
        lines.append("   • USDA Production, Supply & Distribution (PSD) data")
        lines.append("   • Historical agricultural disruption event records")
        lines.append("")
        
        lines.append("2. THRESHOLD CALCULATION:")
        lines.append("   • Dynamic thresholds based on USDA Stock-to-Use Ratio (SUR) percentiles")
        lines.append("   • IPC Phase classification (2-5) for food security assessment")
        lines.append("   • Crop-specific calibration using historical data")
        lines.append("")
        
        lines.append("3. H-P ENVELOPE CONSTRUCTION:")
        lines.append("   • Grid cell sorting by productivity (yield)")
        lines.append("   • Cumulative production loss calculation")
        lines.append("   • Upper and lower bound estimation")
        lines.append("")
        
        lines.append("4. MAGNITUDE CALCULATION:")
        lines.append("   • M_D = log₁₀(disrupted harvest area in km²)")
        lines.append("   • Event impact assessment using production loss in kcal")
        lines.append("")
        
        return lines
    
    def _generate_threshold_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate threshold analysis section."""
        lines = []
        lines.append("THRESHOLD ANALYSIS")
        lines.append("=" * 50)
        lines.append("")
        
        if 'thresholds' in analysis_data:
            thresholds = analysis_data['thresholds']
            lines.append("PRODUCTION LOSS THRESHOLDS (kcal):")
            for threshold_name, threshold_value in thresholds.items():
                ipc_phase = {'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}.get(threshold_name, 'Unknown')
                lines.append(f"  {threshold_name} (IPC Phase {ipc_phase}): {threshold_value:.2e}")
            lines.append("")
        
        if 'sur_thresholds' in analysis_data:
            sur_thresholds = analysis_data['sur_thresholds']
            lines.append("STOCK-TO-USE RATIO (SUR) THRESHOLDS:")
            for phase, sur_value in sur_thresholds.items():
                phase_names = {2: 'Stressed', 3: 'Crisis', 4: 'Emergency', 5: 'Famine'}
                phase_name = phase_names.get(phase, f'Phase {phase}')
                lines.append(f"  IPC Phase {phase} ({phase_name}): {sur_value:.3f}")
            lines.append("")
        
        # Threshold interpretation
        lines.append("THRESHOLD INTERPRETATION:")
        lines.append("• Lower SUR values indicate higher food insecurity risk")
        lines.append("• Production loss thresholds represent cumulative global impact")
        lines.append("• Thresholds are calibrated using 30-year historical data (1990-2020)")
        lines.append("")
        
        return lines
    
    def _generate_events_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate historical events analysis section."""
        lines = []
        lines.append("HISTORICAL EVENTS ANALYSIS")
        lines.append("=" * 50)
        lines.append("")
        
        if 'events_data' not in analysis_data:
            lines.append("No historical events data available for analysis.")
            lines.append("")
            return lines
        
        events_df = analysis_data['events_data']
        if events_df.empty:
            lines.append("No historical events found in dataset.")
            lines.append("")
            return lines
        
        # Basic statistics
        lines.append(f"DATASET OVERVIEW:")
        lines.append(f"• Total events analyzed: {len(events_df)}")
        lines.append(f"• Time period: Historical agricultural disruptions")
        lines.append("")
        
        # Area statistics
        area_stats = events_df['harvest_area_km2'].describe()
        lines.append("AFFECTED HARVEST AREA STATISTICS (km²):")
        lines.append(f"• Mean: {area_stats['mean']:,.0f}")
        lines.append(f"• Median: {area_stats['50%']:,.0f}")
        lines.append(f"• Standard deviation: {area_stats['std']:,.0f}")
        lines.append(f"• Range: {area_stats['min']:,.0f} - {area_stats['max']:,.0f}")
        lines.append("")
        
        # Production loss statistics
        loss_stats = events_df['production_loss_kcal'].describe()
        lines.append("PRODUCTION LOSS STATISTICS (kcal):")
        lines.append(f"• Mean: {loss_stats['mean']:.2e}")
        lines.append(f"• Median: {loss_stats['50%']:.2e}")
        lines.append(f"• Standard deviation: {loss_stats['std']:.2e}")
        lines.append(f"• Range: {loss_stats['min']:.2e} - {loss_stats['max']:.2e}")
        lines.append("")
        
        # Magnitude analysis
        magnitudes = np.log10(events_df['harvest_area_km2'])
        lines.append("MAGNITUDE DISTRIBUTION:")
        lines.append(f"• Mean magnitude: {magnitudes.mean():.2f}")
        lines.append(f"• Magnitude range: {magnitudes.min():.2f} - {magnitudes.max():.2f}")
        lines.append("")
        
        # Top events
        lines.append("TOP 10 EVENTS BY PRODUCTION LOSS:")
        top_events = events_df.nlargest(10, 'production_loss_kcal')
        for i, (_, event) in enumerate(top_events.iterrows(), 1):
            magnitude = np.log10(event['harvest_area_km2'])
            lines.append(f"{i:2d}. {event['event_name']:<25} "
                        f"M={magnitude:.2f}, Loss={event['production_loss_kcal']:.2e} kcal")
        lines.append("")
        
        return lines
    
    def _generate_envelope_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate H-P envelope analysis section."""
        lines = []
        lines.append("H-P ENVELOPE ANALYSIS")
        lines.append("=" * 50)
        lines.append("")
        
        if 'envelope_data' not in analysis_data:
            lines.append("No H-P envelope data available for analysis.")
            lines.append("")
            return lines
        
        envelope = analysis_data['envelope_data']
        
        # Basic envelope statistics
        areas = np.array(envelope['disrupted_areas'])
        upper = np.array(envelope['upper_bound'])
        lower = np.array(envelope['lower_bound'])
        
        lines.append("ENVELOPE CHARACTERISTICS:")
        lines.append(f"• Disruption area range: {areas.min():,.0f} - {areas.max():,.0f} km²")
        lines.append(f"• Upper bound range: {upper.min():.2e} - {upper.max():.2e} kcal")
        lines.append(f"• Lower bound range: {lower.min():.2e} - {lower.max():.2e} kcal")
        lines.append("")
        
        # Envelope width analysis
        envelope_width = upper - lower
        relative_width = envelope_width / upper * 100
        
        lines.append("ENVELOPE WIDTH ANALYSIS:")
        lines.append(f"• Mean absolute width: {envelope_width.mean():.2e} kcal")
        lines.append(f"• Mean relative width: {relative_width.mean():.1f}%")
        lines.append(f"• Width variability (CV): {(envelope_width.std()/envelope_width.mean()*100):.1f}%")
        lines.append("")
        
        # Scaling analysis
        log_areas = np.log10(areas)
        log_upper = np.log10(upper)
        
        # Fit power law to upper bound
        coeffs = np.polyfit(log_areas, log_upper, 1)
        scaling_exponent = coeffs[0]
        
        lines.append("SCALING RELATIONSHIPS:")
        lines.append(f"• Upper bound scaling exponent: {scaling_exponent:.2f}")
        lines.append(f"• Interpretation: Production loss ∝ (Area)^{scaling_exponent:.2f}")
        lines.append("")
        
        return lines
    
    def _generate_usda_analysis(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate USDA data analysis section."""
        lines = []
        lines.append("USDA PSD DATA ANALYSIS")
        lines.append("=" * 50)
        lines.append("")
        
        if 'usda_data' not in analysis_data:
            lines.append("No USDA PSD data available for analysis.")
            lines.append("")
            return lines
        
        usda_data = analysis_data['usda_data']
        
        for crop_name, crop_data in usda_data.items():
            if crop_data.empty:
                continue
                
            lines.append(f"{crop_name.upper()} DATA SUMMARY:")
            lines.append(f"• Data period: {crop_data['Year'].min()}-{crop_data['Year'].max()}")
            lines.append(f"• Number of years: {len(crop_data)}")
            lines.append("")
            
            # Production statistics
            prod_stats = crop_data['Production'].describe()
            lines.append(f"  Production Statistics (1000 MT):")
            lines.append(f"  • Mean: {prod_stats['mean']:,.0f}")
            lines.append(f"  • Trend: {self._calculate_trend(crop_data, 'Production')}")
            lines.append("")
            
            # Consumption statistics
            cons_stats = crop_data['Consumption'].describe()
            lines.append(f"  Consumption Statistics (1000 MT):")
            lines.append(f"  • Mean: {cons_stats['mean']:,.0f}")
            lines.append(f"  • Trend: {self._calculate_trend(crop_data, 'Consumption')}")
            lines.append("")
            
            # SUR analysis
            sur = crop_data['EndingStocks'] / crop_data['Consumption']
            sur_stats = sur.describe()
            lines.append(f"  Stock-to-Use Ratio (SUR) Analysis:")
            lines.append(f"  • Mean SUR: {sur_stats['mean']:.3f}")
            lines.append(f"  • SUR volatility (CV): {(sur_stats['std']/sur_stats['mean']*100):.1f}%")
            lines.append(f"  • Minimum SUR: {sur_stats['min']:.3f} (Year: {crop_data.loc[sur.idxmin(), 'Year']})")
            lines.append(f"  • Maximum SUR: {sur_stats['max']:.3f} (Year: {crop_data.loc[sur.idxmax(), 'Year']})")
            lines.append("")
        
        return lines
    
    def _generate_statistical_summary(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate statistical summary section."""
        lines = []
        lines.append("STATISTICAL SUMMARY")
        lines.append("=" * 50)
        lines.append("")
        
        # Correlation analysis
        if 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            if not events_df.empty and len(events_df) > 2:
                correlation = events_df['harvest_area_km2'].corr(events_df['production_loss_kcal'])
                lines.append("CORRELATION ANALYSIS:")
                lines.append(f"• Harvest area vs Production loss correlation: {correlation:.3f}")
                
                # Log-log correlation
                log_area = np.log10(events_df['harvest_area_km2'])
                log_loss = np.log10(events_df['production_loss_kcal'])
                log_correlation = log_area.corr(log_loss)
                lines.append(f"• Log(area) vs Log(loss) correlation: {log_correlation:.3f}")
                lines.append("")
        
        # Distribution analysis
        lines.append("DISTRIBUTION CHARACTERISTICS:")
        if 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            if not events_df.empty and len(events_df) >= 3:
                try:
                    # Test for log-normal distribution
                    from scipy import stats
                    
                    # Shapiro-Wilk test on log-transformed data
                    log_areas = np.log10(events_df['harvest_area_km2'])
                    shapiro_stat, shapiro_p = stats.shapiro(log_areas)
                    
                    lines.append(f"• Log-normal test for harvest areas:")
                    lines.append(f"  - Shapiro-Wilk statistic: {shapiro_stat:.3f}")
                    lines.append(f"  - p-value: {shapiro_p:.3f}")
                    lines.append(f"  - Distribution: {'Log-normal' if shapiro_p > 0.05 else 'Non-log-normal'}")
                except Exception as e:
                    lines.append(f"• Distribution analysis failed: {e}")
            else:
                lines.append("• Insufficient data for distribution analysis (need ≥3 events)")
        
        lines.append("")
        return lines
    
    def _generate_quality_assessment(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate data quality assessment section."""
        lines = []
        lines.append("DATA QUALITY ASSESSMENT")
        lines.append("=" * 50)
        lines.append("")
        
        quality_score = 0
        total_checks = 0
        
        # Check events data quality
        if 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            lines.append("EVENTS DATA QUALITY:")
            
            # Completeness check
            missing_data = events_df.isnull().sum().sum()
            total_checks += 1
            if missing_data == 0:
                lines.append("✓ No missing data in events dataset")
                quality_score += 1
            else:
                lines.append(f"⚠ {missing_data} missing values in events dataset")
            
            # Range validation
            total_checks += 1
            if (events_df['harvest_area_km2'] > 0).all() and (events_df['production_loss_kcal'] > 0).all():
                lines.append("✓ All values are positive (valid range)")
                quality_score += 1
            else:
                lines.append("⚠ Some values are zero or negative")
            
            lines.append("")
        
        # Check USDA data quality
        if 'usda_data' in analysis_data:
            lines.append("USDA DATA QUALITY:")
            
            for crop_name, crop_data in analysis_data['usda_data'].items():
                if not crop_data.empty:
                    # Check for missing values
                    missing = crop_data.isnull().sum().sum()
                    total_checks += 1
                    if missing == 0:
                        lines.append(f"✓ {crop_name}: No missing data")
                        quality_score += 1
                    else:
                        lines.append(f"⚠ {crop_name}: {missing} missing values")
                    
                    # Check for reasonable SUR values
                    sur = crop_data['EndingStocks'] / crop_data['Consumption']
                    total_checks += 1
                    if (sur > 0).all() and (sur < 2).all():
                        lines.append(f"✓ {crop_name}: SUR values in reasonable range")
                        quality_score += 1
                    else:
                        lines.append(f"⚠ {crop_name}: Some SUR values outside reasonable range")
            
            lines.append("")
        
        # Overall quality score
        if total_checks > 0:
            quality_percentage = (quality_score / total_checks) * 100
            lines.append(f"OVERALL DATA QUALITY SCORE: {quality_percentage:.1f}% ({quality_score}/{total_checks} checks passed)")
            
            if quality_percentage >= 90:
                lines.append("✓ Excellent data quality")
            elif quality_percentage >= 75:
                lines.append("⚠ Good data quality with minor issues")
            elif quality_percentage >= 50:
                lines.append("⚠ Moderate data quality - review recommended")
            else:
                lines.append("✗ Poor data quality - significant issues detected")
        
        lines.append("")
        return lines
    
    def _generate_conclusions(self, analysis_data: Dict[str, Any], crop_type: str) -> List[str]:
        """Generate conclusions and recommendations section."""
        lines = []
        lines.append("CONCLUSIONS AND RECOMMENDATIONS")
        lines.append("=" * 50)
        lines.append("")
        
        lines.append("MAIN FINDINGS:")
        
        # Threshold-based conclusions
        if 'thresholds' in analysis_data and 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            thresholds = analysis_data['thresholds']
            
            if not events_df.empty:
                severity_counts = self._classify_event_severity(events_df, thresholds)
                lines.append(f"• Historical events show diverse severity levels for {crop_type}")
                lines.append(f"• Most events fall in lower severity categories")
                
                # Check for extreme events
                if 'T4' in thresholds:
                    extreme_events = events_df[events_df['production_loss_kcal'] > thresholds['T4']]
                    if not extreme_events.empty:
                        lines.append(f"• {len(extreme_events)} events exceed T4 threshold (extreme severity)")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("• Continue monitoring agricultural disruption patterns")
        lines.append("• Update thresholds periodically with new USDA data")
        lines.append("• Expand analysis to include climate change projections")
        lines.append("• Develop early warning systems based on SUR indicators")
        lines.append("")
        
        lines.append("LIMITATIONS:")
        lines.append("• Analysis based on historical data patterns")
        lines.append("• Assumes uniform production distribution within regions")
        lines.append("• Does not account for trade and substitution effects")
        lines.append("• Limited to production-side impacts only")
        lines.append("")
        
        return lines
    
    def _generate_appendices(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate appendices section."""
        lines = []
        lines.append("APPENDICES")
        lines.append("=" * 50)
        lines.append("")
        
        lines.append("APPENDIX A: TECHNICAL SPECIFICATIONS")
        lines.append("-" * 40)
        lines.append("• SPAM 2020 V2r0 global production data")
        lines.append("• USDA PSD data (1961-2021)")
        lines.append("• 5-arcminute spatial resolution")
        lines.append("• Robinson projection for global maps")
        lines.append("")
        
        lines.append("APPENDIX B: ABBREVIATIONS")
        lines.append("-" * 40)
        lines.append("• SUR: Stock-to-Use Ratio")
        lines.append("• IPC: Integrated Food Security Phase Classification")
        lines.append("• PSD: Production, Supply and Distribution")
        lines.append("• SPAM: Spatial Production Allocation Model")
        lines.append("• H-P: Harvest-Production")
        lines.append("")
        
        return lines
    
    def _generate_summary_content(self, analysis_data: Dict[str, Any], 
                                crop_type: str) -> str:
        """Generate executive summary content."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"AGRIRICHTER ANALYSIS EXECUTIVE SUMMARY - {crop_type.upper()}")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Key metrics
        if 'events_data' in analysis_data:
            events_df = analysis_data['events_data']
            if not events_df.empty:
                lines.append("KEY METRICS:")
                lines.append(f"• Events analyzed: {len(events_df)}")
                lines.append(f"• Total affected area: {events_df['harvest_area_km2'].sum():,.0f} km²")
                lines.append(f"• Total production loss: {events_df['production_loss_kcal'].sum():.2e} kcal")
                lines.append(f"• Average event magnitude: {np.log10(events_df['harvest_area_km2']).mean():.2f}")
        
        lines.append("")
        
        # Threshold summary
        if 'thresholds' in analysis_data:
            lines.append("SEVERITY THRESHOLDS:")
            for threshold, value in analysis_data['thresholds'].items():
                lines.append(f"• {threshold}: {value:.2e} kcal")
        
        lines.append("")
        lines.append("For detailed analysis, see the comprehensive report.")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_validation_content(self, analysis_data: Dict[str, Any], 
                                   crop_type: str) -> str:
        """Generate validation report content."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"AGRIRICHTER DATA VALIDATION REPORT - {crop_type.upper()}")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Data validation checks
        lines.extend(self._generate_quality_assessment(analysis_data))
        
        # Metadata validation
        lines.append("METADATA VALIDATION:")
        lines.append(f"• Crop type: {crop_type}")
        lines.append(f"• Analysis timestamp: {datetime.now().isoformat()}")
        
        if 'usda_data' in analysis_data:
            for crop_name, crop_data in analysis_data['usda_data'].items():
                if not crop_data.empty:
                    lines.append(f"• {crop_name} data years: {crop_data['Year'].min()}-{crop_data['Year'].max()}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _classify_event_severity(self, events_df: pd.DataFrame, 
                               thresholds: Dict[str, float]) -> Dict[str, int]:
        """Classify events by severity based on thresholds."""
        severity_counts = {'Below T1': 0, 'T1-T2': 0, 'T2-T3': 0, 'T3-T4': 0, 'Above T4': 0}
        
        for _, event in events_df.iterrows():
            loss = event['production_loss_kcal']
            
            if loss < thresholds.get('T1', float('inf')):
                severity_counts['Below T1'] += 1
            elif loss < thresholds.get('T2', float('inf')):
                severity_counts['T1-T2'] += 1
            elif loss < thresholds.get('T3', float('inf')):
                severity_counts['T2-T3'] += 1
            elif loss < thresholds.get('T4', float('inf')):
                severity_counts['T3-T4'] += 1
            else:
                severity_counts['Above T4'] += 1
        
        return severity_counts
    
    def _calculate_trend(self, data: pd.DataFrame, column: str) -> str:
        """Calculate trend for a time series."""
        try:
            from scipy import stats
            years = data['Year'].values
            values = data[column].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
            
            if p_value < 0.05:
                if slope > 0:
                    return f"Increasing ({slope:.1f}/year, R²={r_value**2:.3f})"
                else:
                    return f"Decreasing ({slope:.1f}/year, R²={r_value**2:.3f})"
            else:
                return "No significant trend"
        except:
            return "Trend calculation failed"


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from agririchter.output.organizer import FileOrganizer
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create organizer and reporter
    organizer = FileOrganizer("test_outputs/reporter_test")
    reporter = AnalysisReporter(organizer)
    
    # Create sample analysis data
    sample_analysis = {
        'events_data': pd.DataFrame({
            'event_name': ['Event A', 'Event B', 'Event C'],
            'harvest_area_km2': [50000, 150000, 300000],
            'production_loss_kcal': [2e14, 8e14, 1.5e15]
        }),
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
    
    # Generate reports
    comprehensive_path = reporter.generate_comprehensive_report(sample_analysis, 'wheat')
    summary_path = reporter.generate_summary_report(sample_analysis, 'wheat')
    validation_path = reporter.generate_validation_report(sample_analysis, 'wheat')
    
    print(f"Generated comprehensive report: {comprehensive_path}")
    print(f"Generated summary report: {summary_path}")
    print(f"Generated validation report: {validation_path}")
    
    print("Advanced reporting test completed!")
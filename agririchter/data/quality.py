"""Data quality assessment and reporting for AgriRichter framework."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..core.config import Config


class DataQualityAssessor:
    """Comprehensive data quality assessment for AgriRichter data."""
    
    def __init__(self, config: Config):
        """
        Initialize data quality assessor.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = logging.getLogger('agririchter.quality')
        
        # Quality thresholds
        self.thresholds = {
            'min_coverage_pct': 5.0,  # Minimum data coverage percentage
            'max_missing_pct': 90.0,  # Maximum missing data percentage
            'max_zero_pct': 95.0,     # Maximum zero values percentage
            'min_yield_t_ha': 0.01,   # Minimum reasonable yield (t/ha)
            'max_yield_t_ha': 100.0,  # Maximum reasonable yield (t/ha)
            'coord_precision_tolerance': 1e-6
        }
    
    def assess_spam_data_quality(self, production_df: pd.DataFrame, 
                                harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive quality assessment of SPAM data.
        
        Args:
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
        
        Returns:
            Dictionary with quality assessment results
        """
        self.logger.info("Starting comprehensive SPAM data quality assessment")
        
        assessment = {
            'production': self._assess_dataframe_quality(production_df, 'production'),
            'harvest_area': self._assess_dataframe_quality(harvest_df, 'harvest_area'),
            'consistency': self._assess_data_consistency(production_df, harvest_df),
            'spatial': self._assess_spatial_quality(production_df),
            'crop_specific': self._assess_crop_specific_quality(production_df, harvest_df),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall quality score
        assessment['overall_score'] = self._calculate_overall_score(assessment)
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_recommendations(assessment)
        
        self.logger.info(f"Data quality assessment complete. Overall score: {assessment['overall_score']:.2f}/10")
        
        return assessment
    
    def _assess_dataframe_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Assess quality of a single DataFrame.
        
        Args:
            df: DataFrame to assess
            data_type: Type of data ('production' or 'harvest_area')
        
        Returns:
            Quality assessment dictionary
        """
        crop_columns = [col for col in df.columns if col.endswith('_a')]
        
        assessment = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'crop_columns': len(crop_columns),
            'missing_data': {},
            'zero_values': {},
            'coverage': {},
            'outliers': {},
            'data_ranges': {}
        }
        
        # Assess missing data
        for col in crop_columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            assessment['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # Assess zero values
        for col in crop_columns:
            zero_count = (df[col] == 0).sum()
            zero_pct = (zero_count / len(df)) * 100
            assessment['zero_values'][col] = {
                'count': zero_count,
                'percentage': zero_pct
            }
        
        # Assess data coverage (non-zero, non-null values)
        for col in crop_columns:
            valid_data = (df[col] > 0) & df[col].notna()
            coverage_count = valid_data.sum()
            coverage_pct = (coverage_count / len(df)) * 100
            assessment['coverage'][col] = {
                'count': coverage_count,
                'percentage': coverage_pct
            }
        
        # Assess data ranges and outliers
        for col in crop_columns:
            valid_data = df[col][df[col] > 0]
            if len(valid_data) > 0:
                q1 = valid_data.quantile(0.25)
                q3 = valid_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
                
                assessment['data_ranges'][col] = {
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'median': float(valid_data.median()),
                    'std': float(valid_data.std())
                }
                
                assessment['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(valid_data)) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        return assessment
    
    def _assess_data_consistency(self, production_df: pd.DataFrame, 
                               harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess consistency between production and harvest area data.
        
        Args:
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
        
        Returns:
            Consistency assessment dictionary
        """
        assessment = {
            'coordinate_consistency': True,
            'dimension_consistency': True,
            'yield_analysis': {},
            'issues': []
        }
        
        # Check dimensions
        if len(production_df) != len(harvest_df):
            assessment['dimension_consistency'] = False
            assessment['issues'].append(f"Dimension mismatch: {len(production_df)} vs {len(harvest_df)} rows")
        
        # Check coordinate consistency
        if len(production_df) == len(harvest_df):
            coord_diff_x = (production_df['x'] - harvest_df['x']).abs().max()
            coord_diff_y = (production_df['y'] - harvest_df['y']).abs().max()
            
            if coord_diff_x > self.thresholds['coord_precision_tolerance'] or \
               coord_diff_y > self.thresholds['coord_precision_tolerance']:
                assessment['coordinate_consistency'] = False
                assessment['issues'].append(f"Coordinate mismatch: max diff x={coord_diff_x:.2e}, y={coord_diff_y:.2e}")
        
        # Analyze yields (production/harvest area)
        crop_columns = [col for col in production_df.columns if col.endswith('_a')]
        
        for col in crop_columns:
            if col in harvest_df.columns:
                # Calculate yields where both values > 0
                mask = (production_df[col] > 0) & (harvest_df[col] > 0)
                
                if mask.sum() > 0:
                    yields = production_df.loc[mask, col] / harvest_df.loc[mask, col]
                    
                    # Assess yield reasonableness
                    low_yield_count = (yields < self.thresholds['min_yield_t_ha']).sum()
                    high_yield_count = (yields > self.thresholds['max_yield_t_ha']).sum()
                    
                    assessment['yield_analysis'][col] = {
                        'total_cells': mask.sum(),
                        'mean_yield': float(yields.mean()),
                        'median_yield': float(yields.median()),
                        'min_yield': float(yields.min()),
                        'max_yield': float(yields.max()),
                        'low_yield_count': low_yield_count,
                        'high_yield_count': high_yield_count,
                        'low_yield_pct': (low_yield_count / len(yields)) * 100,
                        'high_yield_pct': (high_yield_count / len(yields)) * 100
                    }
        
        return assessment
    
    def _assess_spatial_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess spatial data quality.
        
        Args:
            df: DataFrame with spatial data
        
        Returns:
            Spatial quality assessment dictionary
        """
        assessment = {
            'coordinate_coverage': 0.0,
            'grid_alignment': {},
            'spatial_distribution': {},
            'issues': []
        }
        
        # Check coordinate coverage
        valid_coords = (~df[['x', 'y']].isna().any(axis=1)).sum()
        assessment['coordinate_coverage'] = (valid_coords / len(df)) * 100
        
        # Check grid alignment (5-minute resolution)
        expected_resolution = 0.0833333333333333
        
        x_remainder = (df['x'] % expected_resolution).abs()
        y_remainder = (df['y'] % expected_resolution).abs()
        
        x_aligned = (x_remainder < self.thresholds['coord_precision_tolerance']).sum()
        y_aligned = (y_remainder < self.thresholds['coord_precision_tolerance']).sum()
        
        assessment['grid_alignment'] = {
            'x_aligned_pct': (x_aligned / len(df)) * 100,
            'y_aligned_pct': (y_aligned / len(df)) * 100,
            'expected_resolution': expected_resolution
        }
        
        # Spatial distribution analysis
        assessment['spatial_distribution'] = {
            'lat_range': [float(df['y'].min()), float(df['y'].max())],
            'lon_range': [float(df['x'].min()), float(df['x'].max())],
            'unique_coordinates': len(df[['x', 'y']].drop_duplicates()),
            'duplicate_coordinates': len(df) - len(df[['x', 'y']].drop_duplicates())
        }
        
        return assessment
    
    def _assess_crop_specific_quality(self, production_df: pd.DataFrame, 
                                    harvest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess quality for specific crops relevant to current analysis.
        
        Args:
            production_df: Production DataFrame
            harvest_df: Harvest area DataFrame
        
        Returns:
            Crop-specific quality assessment
        """
        crop_indices = self.config.get_crop_indices()
        spam_columns = [
            'whea_a', 'rice_a', 'maiz_a', 'barl_a', 'pmil_a', 'smil_a', 
            'sorg_a', 'ocer_a'
        ]
        
        # Get relevant crop columns
        relevant_columns = [spam_columns[i-1] for i in crop_indices if 1 <= i <= len(spam_columns)]
        
        assessment = {
            'crop_type': self.config.crop_type,
            'relevant_crops': relevant_columns,
            'crop_coverage': {},
            'crop_statistics': {}
        }
        
        for col in relevant_columns:
            if col in production_df.columns and col in harvest_df.columns:
                # Production statistics
                prod_data = production_df[col][production_df[col] > 0]
                harvest_data = harvest_df[col][harvest_df[col] > 0]
                
                assessment['crop_coverage'][col] = {
                    'production_cells': len(prod_data),
                    'harvest_cells': len(harvest_data),
                    'production_coverage_pct': (len(prod_data) / len(production_df)) * 100,
                    'harvest_coverage_pct': (len(harvest_data) / len(harvest_df)) * 100
                }
                
                if len(prod_data) > 0 and len(harvest_data) > 0:
                    assessment['crop_statistics'][col] = {
                        'total_production': float(prod_data.sum()),
                        'total_harvest_area': float(harvest_data.sum()),
                        'avg_production_per_cell': float(prod_data.mean()),
                        'avg_harvest_per_cell': float(harvest_data.mean()),
                        'production_std': float(prod_data.std()),
                        'harvest_std': float(harvest_data.std())
                    }
        
        return assessment
    
    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score (0-10 scale).
        
        Args:
            assessment: Complete assessment dictionary
        
        Returns:
            Overall quality score
        """
        scores = []
        
        # Score based on data coverage
        prod_coverage = np.mean([
            crop['percentage'] for crop in assessment['production']['coverage'].values()
        ])
        harvest_coverage = np.mean([
            crop['percentage'] for crop in assessment['harvest_area']['coverage'].values()
        ])
        
        coverage_score = min(10, (prod_coverage + harvest_coverage) / 2 / 10)
        scores.append(coverage_score)
        
        # Score based on consistency
        consistency_score = 10.0
        if not assessment['consistency']['coordinate_consistency']:
            consistency_score -= 3.0
        if not assessment['consistency']['dimension_consistency']:
            consistency_score -= 2.0
        
        scores.append(max(0, consistency_score))
        
        # Score based on spatial quality
        spatial_score = min(10, assessment['spatial']['coordinate_coverage'] / 10)
        scores.append(spatial_score)
        
        # Score based on missing data
        avg_missing = np.mean([
            crop['percentage'] for crop in assessment['production']['missing_data'].values()
        ])
        missing_score = max(0, 10 - (avg_missing / 10))
        scores.append(missing_score)
        
        return np.mean(scores)
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """
        Generate data quality improvement recommendations.
        
        Args:
            assessment: Complete assessment dictionary
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Coverage recommendations
        avg_coverage = np.mean([
            crop['percentage'] for crop in assessment['production']['coverage'].values()
        ])
        
        if avg_coverage < self.thresholds['min_coverage_pct']:
            recommendations.append(
                f"Data coverage is very low ({avg_coverage:.1f}%). "
                "Consider using additional data sources or interpolation methods."
            )
        
        # Missing data recommendations
        high_missing_crops = [
            crop for crop, data in assessment['production']['missing_data'].items()
            if data['percentage'] > self.thresholds['max_missing_pct']
        ]
        
        if high_missing_crops:
            recommendations.append(
                f"High missing data in crops: {', '.join(high_missing_crops)}. "
                "Consider data imputation or exclusion from analysis."
            )
        
        # Consistency recommendations
        if not assessment['consistency']['coordinate_consistency']:
            recommendations.append(
                "Coordinate inconsistency detected between production and harvest data. "
                "Verify data alignment and consider spatial interpolation."
            )
        
        # Yield recommendations
        for crop, yield_data in assessment['consistency']['yield_analysis'].items():
            if yield_data['low_yield_pct'] > 10:
                recommendations.append(
                    f"High percentage of unreasonably low yields in {crop} ({yield_data['low_yield_pct']:.1f}%). "
                    "Review data quality or adjust yield thresholds."
                )
            
            if yield_data['high_yield_pct'] > 5:
                recommendations.append(
                    f"High percentage of unreasonably high yields in {crop} ({yield_data['high_yield_pct']:.1f}%). "
                    "Check for data entry errors or outliers."
                )
        
        return recommendations
    
    def generate_quality_report(self, assessment: Dict[str, Any], 
                              output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive quality report.
        
        Args:
            assessment: Quality assessment results
            output_path: Optional path to save report
        
        Returns:
            Formatted quality report string
        """
        report_lines = [
            "=" * 60,
            "AgriRichter Data Quality Assessment Report",
            "=" * 60,
            "",
            f"Overall Quality Score: {assessment['overall_score']:.2f}/10.0",
            f"Crop Type: {assessment.get('crop_specific', {}).get('crop_type', 'Unknown')}",
            "",
            "SUMMARY STATISTICS",
            "-" * 20,
            f"Production Data Rows: {assessment['production']['total_rows']:,}",
            f"Harvest Area Data Rows: {assessment['harvest_area']['total_rows']:,}",
            f"Crop Columns: {assessment['production']['crop_columns']}",
            f"Coordinate Coverage: {assessment['spatial']['coordinate_coverage']:.1f}%",
            ""
        ]
        
        # Add coverage summary
        report_lines.extend([
            "DATA COVERAGE",
            "-" * 15,
        ])
        
        for crop, data in assessment['production']['coverage'].items():
            report_lines.append(f"{crop}: {data['percentage']:.1f}% ({data['count']:,} cells)")
        
        report_lines.append("")
        
        # Add consistency summary
        report_lines.extend([
            "DATA CONSISTENCY",
            "-" * 17,
            f"Coordinate Consistency: {'✓' if assessment['consistency']['coordinate_consistency'] else '✗'}",
            f"Dimension Consistency: {'✓' if assessment['consistency']['dimension_consistency'] else '✗'}",
            ""
        ])
        
        # Add recommendations
        if assessment['recommendations']:
            report_lines.extend([
                "RECOMMENDATIONS",
                "-" * 16,
            ])
            for i, rec in enumerate(assessment['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Add detailed statistics for current crop type
        if 'crop_specific' in assessment:
            report_lines.extend([
                f"CROP-SPECIFIC ANALYSIS ({assessment['crop_specific']['crop_type'].upper()})",
                "-" * 30,
            ])
            
            for crop, stats in assessment['crop_specific'].get('crop_statistics', {}).items():
                report_lines.extend([
                    f"{crop}:",
                    f"  Total Production: {stats['total_production']:,.0f} metric tons",
                    f"  Total Harvest Area: {stats['total_harvest_area']:,.0f} hectares",
                    f"  Average Yield: {stats['total_production']/stats['total_harvest_area']:.2f} t/ha",
                    ""
                ])
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Quality report saved to {output_path}")
        
        return report_text
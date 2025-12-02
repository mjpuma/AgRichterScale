"""
National Envelope Analyzer for country-specific multi-tier agricultural analysis.

This module provides comprehensive national-level agricultural capacity analysis
using multi-tier envelope bounds with country boundary filtering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

from ..core.config import Config
from ..data.country_boundary_manager import CountryBoundaryManager, CountryConfiguration
from .multi_tier_envelope import MultiTierEnvelopeEngine, MultiTierResults, TIER_CONFIGURATIONS
from .envelope import EnvelopeData

logger = logging.getLogger(__name__)


@dataclass
class NationalAnalysisResults:
    """Results from national agricultural capacity analysis."""
    
    country_code: str
    country_name: str
    crop_type: str
    multi_tier_results: MultiTierResults
    national_statistics: Dict[str, Any]
    policy_insights: Dict[str, Any]
    validation_results: Dict[str, Any]
    
    def get_tier_envelope(self, tier_name: str) -> Optional[EnvelopeData]:
        """Get envelope data for a specific tier."""
        return self.multi_tier_results.get_tier_envelope(tier_name)
    
    def get_width_reduction(self, tier_name: str) -> Optional[float]:
        """Get width reduction percentage for a tier."""
        return self.multi_tier_results.get_width_reduction(tier_name)
    
    def get_production_capacity(self, tier_name: str = 'commercial') -> Dict[str, float]:
        """Get production capacity metrics for a tier."""
        envelope_data = self.get_tier_envelope(tier_name)
        if envelope_data is None:
            return {}
        
        return {
            'max_production_capacity': float(np.max(envelope_data.upper_bound_production)),
            'min_production_capacity': float(np.max(envelope_data.lower_bound_production)),
            'total_harvest_area': float(np.max(envelope_data.upper_bound_harvest)),
            'production_range': float(np.max(envelope_data.upper_bound_production) - np.max(envelope_data.lower_bound_production))
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report."""
        return {
            'country_info': {
                'country_code': self.country_code,
                'country_name': self.country_name,
                'crop_type': self.crop_type
            },
            'tier_analysis': {
                tier_name: {
                    'width_reduction_pct': self.get_width_reduction(tier_name),
                    'production_capacity': self.get_production_capacity(tier_name)
                }
                for tier_name in self.multi_tier_results.tier_results.keys()
            },
            'national_statistics': self.national_statistics,
            'policy_insights': self.policy_insights,
            'validation_status': self.validation_results.get('overall_valid', False)
        }


@dataclass
class NationalComparisonResults:
    """Results from comparing multiple countries."""
    
    countries: List[str]
    crop_type: str
    country_results: Dict[str, NationalAnalysisResults]
    comparative_metrics: Dict[str, Any]
    policy_recommendations: Dict[str, Any]
    
    def get_country_ranking(self, metric: str, tier: str = 'commercial') -> List[Tuple[str, float]]:
        """Get country ranking by a specific metric."""
        rankings = []
        
        for country_code, results in self.country_results.items():
            if metric == 'production_capacity':
                capacity = results.get_production_capacity(tier)
                value = capacity.get('max_production_capacity', 0)
            elif metric == 'width_reduction':
                value = results.get_width_reduction(tier) or 0
            elif metric == 'harvest_area':
                capacity = results.get_production_capacity(tier)
                value = capacity.get('total_harvest_area', 0)
            else:
                value = 0
            
            rankings.append((country_code, value))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def generate_comparison_summary(self) -> Dict[str, Any]:
        """Generate comprehensive comparison summary."""
        return {
            'countries_analyzed': self.countries,
            'crop_type': self.crop_type,
            'rankings': {
                'by_production_capacity': self.get_country_ranking('production_capacity'),
                'by_width_reduction': self.get_country_ranking('width_reduction'),
                'by_harvest_area': self.get_country_ranking('harvest_area')
            },
            'comparative_metrics': self.comparative_metrics,
            'policy_recommendations': self.policy_recommendations
        }


class NationalEnvelopeAnalyzer:
    """
    Handles country-specific envelope analysis with multi-tier support.
    
    Provides comprehensive national agricultural capacity analysis including:
    - Multi-tier envelope calculation for individual countries
    - Cross-country comparison and ranking
    - Policy-relevant insights and recommendations
    - Validation against known agricultural patterns
    """
    
    def __init__(self, 
                 config: Config,
                 country_boundary_manager: CountryBoundaryManager,
                 multi_tier_engine: Optional[MultiTierEnvelopeEngine] = None):
        """
        Initialize NationalEnvelopeAnalyzer.
        
        Args:
            config: Configuration object
            country_boundary_manager: CountryBoundaryManager instance
            multi_tier_engine: Optional MultiTierEnvelopeEngine (creates default if None)
        """
        self.config = config
        self.country_manager = country_boundary_manager
        self.multi_tier_engine = multi_tier_engine or MultiTierEnvelopeEngine(config)
        self.logger = logging.getLogger('agririchter.national_analyzer')
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, NationalAnalysisResults] = {}
        
        self.logger.info("NationalEnvelopeAnalyzer initialized")
    
    def analyze_national_capacity(self, 
                                country_code: str,
                                tiers: Optional[List[str]] = None,
                                validate_results: bool = True) -> NationalAnalysisResults:
        """
        Analyze agricultural capacity for a specific country.
        
        Args:
            country_code: Country code (e.g., 'USA', 'CHN')
            tiers: List of tiers to analyze (default: all tiers)
            validate_results: Whether to validate results against known patterns
        
        Returns:
            NationalAnalysisResults with comprehensive analysis
        """
        country_code = country_code.upper()
        cache_key = f"{country_code}_{self.config.crop_type}_{tiers}"
        
        # Check cache first
        if cache_key in self._analysis_cache:
            self.logger.debug(f"Returning cached analysis for {country_code}")
            return self._analysis_cache[cache_key]
        
        self.logger.info(f"Starting national capacity analysis for {country_code} ({self.config.crop_type})")
        
        # Get country configuration
        country_config = self.country_manager.get_country_configuration(country_code)
        if country_config is None:
            raise ValueError(f"Country {country_code} not supported")
        
        # Validate country data coverage
        validation = self.country_manager.validate_country_data_coverage(country_code)
        if not validation['valid']:
            raise ValueError(f"Insufficient data coverage for {country_code}: {validation.get('error', 'Unknown error')}")
        
        # Get country-specific SPAM data
        production_df, harvest_area_df = self.country_manager.get_country_data(country_code)
        
        self.logger.info(f"Loaded {len(production_df)} production cells for {country_config.country_name}")
        
        # Calculate multi-tier envelopes
        multi_tier_results = self.multi_tier_engine.calculate_multi_tier_envelope(
            production_df, harvest_area_df, tiers
        )
        
        # Calculate national statistics
        national_statistics = self._calculate_national_statistics(
            production_df, harvest_area_df, country_config
        )
        
        # Generate policy insights
        policy_insights = self._generate_policy_insights(
            multi_tier_results, country_config, national_statistics
        )
        
        # Validate results if requested
        validation_results = {}
        if validate_results:
            validation_results = self._validate_national_results(
                multi_tier_results, country_config, national_statistics
            )
        
        # Create results object
        results = NationalAnalysisResults(
            country_code=country_code,
            country_name=country_config.country_name,
            crop_type=self.config.crop_type,
            multi_tier_results=multi_tier_results,
            national_statistics=national_statistics,
            policy_insights=policy_insights,
            validation_results=validation_results
        )
        
        # Cache results
        self._analysis_cache[cache_key] = results
        
        self.logger.info(f"National capacity analysis completed for {country_config.country_name}")
        return results
    
    def compare_countries(self, 
                         country_codes: List[str],
                         tiers: Optional[List[str]] = None) -> NationalComparisonResults:
        """
        Compare agricultural capacity between multiple countries.
        
        Args:
            country_codes: List of country codes to compare
            tiers: List of tiers to analyze (default: all tiers)
        
        Returns:
            NationalComparisonResults with comparative analysis
        """
        self.logger.info(f"Starting national comparison for {len(country_codes)} countries: {country_codes}")
        
        # Analyze each country
        country_results = {}
        for country_code in country_codes:
            try:
                results = self.analyze_national_capacity(country_code, tiers)
                country_results[country_code] = results
                self.logger.info(f"✓ Completed analysis for {country_code}")
            except Exception as e:
                self.logger.error(f"✗ Failed to analyze {country_code}: {e}")
                continue
        
        if len(country_results) < 2:
            raise ValueError("Need at least 2 countries with valid data for comparison")
        
        # Calculate comparative metrics
        comparative_metrics = self._calculate_comparative_metrics(country_results)
        
        # Generate policy recommendations
        policy_recommendations = self._generate_comparative_policy_recommendations(
            country_results, comparative_metrics
        )
        
        results = NationalComparisonResults(
            countries=list(country_results.keys()),
            crop_type=self.config.crop_type,
            country_results=country_results,
            comparative_metrics=comparative_metrics,
            policy_recommendations=policy_recommendations
        )
        
        self.logger.info(f"National comparison completed for {len(country_results)} countries")
        return results
    
    def _calculate_national_statistics(self, 
                                     production_df: pd.DataFrame,
                                     harvest_area_df: pd.DataFrame,
                                     country_config: CountryConfiguration) -> Dict[str, Any]:
        """Calculate comprehensive national statistics."""
        # Get crop columns for this crop type
        crop_columns = self._get_crop_columns_for_type(production_df)
        
        # Calculate total production and harvest area
        total_production = production_df[crop_columns].sum().sum()
        total_harvest_area = harvest_area_df[crop_columns].sum().sum()
        
        # Calculate yields
        production_by_cell = production_df[crop_columns].sum(axis=1)
        harvest_by_cell = harvest_area_df[crop_columns].sum(axis=1)
        
        # Calculate yields where harvest area > 0
        valid_cells = harvest_by_cell > 0
        yields = np.divide(
            production_by_cell[valid_cells],
            harvest_by_cell[valid_cells],
            out=np.zeros_like(production_by_cell[valid_cells]),
            where=(harvest_by_cell[valid_cells] > 0)
        )
        
        statistics = {
            'total_production_mt': float(total_production),
            'total_harvest_area_ha': float(total_harvest_area),
            'average_yield_mt_per_ha': float(yields.mean()) if len(yields) > 0 else 0,
            'median_yield_mt_per_ha': float(np.median(yields)) if len(yields) > 0 else 0,
            'yield_std_mt_per_ha': float(yields.std()) if len(yields) > 0 else 0,
            'total_cells': len(production_df),
            'productive_cells': int(valid_cells.sum()),
            'productivity_coverage_pct': float((valid_cells.sum() / len(production_df)) * 100) if len(production_df) > 0 else 0,
            'geographic_extent': {
                'lat_min': float(production_df['y'].min()),
                'lat_max': float(production_df['y'].max()),
                'lon_min': float(production_df['x'].min()),
                'lon_max': float(production_df['x'].max())
            },
            'crop_breakdown': {}
        }
        
        # Calculate crop-specific statistics
        for crop_col in crop_columns:
            crop_production = production_df[crop_col].sum()
            crop_harvest = harvest_area_df[crop_col].sum()
            crop_cells = (production_df[crop_col] > 0).sum()
            
            statistics['crop_breakdown'][crop_col] = {
                'production_mt': float(crop_production),
                'harvest_area_ha': float(crop_harvest),
                'cells_with_production': int(crop_cells),
                'share_of_total_production_pct': float((crop_production / total_production) * 100) if total_production > 0 else 0,
                'share_of_total_area_pct': float((crop_harvest / total_harvest_area) * 100) if total_harvest_area > 0 else 0
            }
        
        return statistics
    
    def _get_crop_columns_for_type(self, production_df: pd.DataFrame) -> List[str]:
        """Get crop columns based on current crop type configuration."""
        all_crop_columns = [col for col in production_df.columns if col.endswith('_A')]
        
        if self.config.crop_type == 'wheat':
            return [col for col in all_crop_columns if 'WHEA' in col.upper()]
        elif self.config.crop_type == 'rice':
            return [col for col in all_crop_columns if 'RICE' in col.upper()]
        elif self.config.crop_type == 'maize':
            return [col for col in all_crop_columns if 'MAIZ' in col.upper()]
        elif self.config.crop_type == 'allgrain':
            grain_crops = ['WHEA_A', 'RICE_A', 'MAIZ_A', 'BARL_A', 'PMIL_A', 'SMIL_A', 'SORG_A', 'OCER_A']
            return [col for col in all_crop_columns if col in grain_crops]
        else:
            return all_crop_columns
    
    def _generate_policy_insights(self, 
                                multi_tier_results: MultiTierResults,
                                country_config: CountryConfiguration,
                                national_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate policy-relevant insights from analysis results."""
        insights = {
            'agricultural_focus': country_config.agricultural_focus,
            'tier_effectiveness': {},
            'capacity_assessment': {},
            'policy_recommendations': []
        }
        
        # Analyze tier effectiveness
        for tier_name in multi_tier_results.tier_results.keys():
            width_reduction = multi_tier_results.get_width_reduction(tier_name)
            tier_config = TIER_CONFIGURATIONS.get(tier_name)
            
            if tier_config:
                insights['tier_effectiveness'][tier_name] = {
                    'width_reduction_pct': width_reduction,
                    'policy_applications': tier_config.policy_applications,
                    'target_users': tier_config.target_users,
                    'effectiveness_rating': self._rate_tier_effectiveness(width_reduction)
                }
        
        # Capacity assessment based on agricultural focus
        if country_config.agricultural_focus == 'food_security':
            insights['capacity_assessment'] = self._assess_food_security_capacity(
                multi_tier_results, national_statistics
            )
        elif country_config.agricultural_focus == 'export_capacity':
            insights['capacity_assessment'] = self._assess_export_capacity(
                multi_tier_results, national_statistics
            )
        elif country_config.agricultural_focus == 'efficiency':
            insights['capacity_assessment'] = self._assess_efficiency_potential(
                multi_tier_results, national_statistics
            )
        
        # Generate policy recommendations
        insights['policy_recommendations'] = self._generate_policy_recommendations(
            multi_tier_results, country_config, national_statistics
        )
        
        return insights
    
    def _rate_tier_effectiveness(self, width_reduction: Optional[float]) -> str:
        """Rate the effectiveness of a tier based on width reduction."""
        if width_reduction is None:
            return 'unknown'
        elif width_reduction >= 30:
            return 'excellent'
        elif width_reduction >= 20:
            return 'good'
        elif width_reduction >= 10:
            return 'moderate'
        else:
            return 'limited'
    
    def _assess_food_security_capacity(self, 
                                     multi_tier_results: MultiTierResults,
                                     national_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess food security capacity using comprehensive tier."""
        comprehensive_envelope = multi_tier_results.get_tier_envelope('comprehensive')
        if comprehensive_envelope is None:
            return {'error': 'Comprehensive tier not available'}
        
        max_production = np.max(comprehensive_envelope.upper_bound_production)
        current_production = national_statistics['total_production_mt']
        
        return {
            'current_production_mt': current_production,
            'maximum_potential_mt': float(max_production),
            'production_gap_mt': float(max_production - current_production),
            'capacity_utilization_pct': float((current_production / max_production) * 100) if max_production > 0 else 0,
            'food_security_rating': self._rate_food_security(current_production, max_production)
        }
    
    def _assess_export_capacity(self, 
                              multi_tier_results: MultiTierResults,
                              national_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess export capacity using commercial tier."""
        commercial_envelope = multi_tier_results.get_tier_envelope('commercial')
        if commercial_envelope is None:
            return {'error': 'Commercial tier not available'}
        
        max_commercial_production = np.max(commercial_envelope.upper_bound_production)
        current_production = national_statistics['total_production_mt']
        
        return {
            'current_production_mt': current_production,
            'commercial_potential_mt': float(max_commercial_production),
            'export_potential_mt': float(max_commercial_production - current_production),
            'commercial_efficiency_pct': float((current_production / max_commercial_production) * 100) if max_commercial_production > 0 else 0,
            'export_capacity_rating': self._rate_export_capacity(current_production, max_commercial_production)
        }
    
    def _assess_efficiency_potential(self, 
                                   multi_tier_results: MultiTierResults,
                                   national_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess efficiency improvement potential."""
        comprehensive_envelope = multi_tier_results.get_tier_envelope('comprehensive')
        commercial_envelope = multi_tier_results.get_tier_envelope('commercial')
        
        if comprehensive_envelope is None or commercial_envelope is None:
            return {'error': 'Required tiers not available'}
        
        max_comprehensive = np.max(comprehensive_envelope.upper_bound_production)
        max_commercial = np.max(commercial_envelope.upper_bound_production)
        current_production = national_statistics['total_production_mt']
        
        return {
            'current_production_mt': current_production,
            'efficiency_gain_potential_mt': float(max_commercial - current_production),
            'total_potential_mt': float(max_comprehensive),
            'efficiency_improvement_pct': float(((max_commercial - current_production) / current_production) * 100) if current_production > 0 else 0,
            'efficiency_rating': self._rate_efficiency_potential(current_production, max_commercial)
        }
    
    def _rate_food_security(self, current: float, maximum: float) -> str:
        """Rate food security based on capacity utilization."""
        if maximum == 0:
            return 'unknown'
        
        utilization = (current / maximum) * 100
        if utilization >= 80:
            return 'high_utilization'
        elif utilization >= 60:
            return 'moderate_utilization'
        elif utilization >= 40:
            return 'low_utilization'
        else:
            return 'very_low_utilization'
    
    def _rate_export_capacity(self, current: float, commercial_max: float) -> str:
        """Rate export capacity based on commercial potential."""
        if commercial_max == 0:
            return 'unknown'
        
        efficiency = (current / commercial_max) * 100
        if efficiency >= 90:
            return 'limited_export_potential'
        elif efficiency >= 70:
            return 'moderate_export_potential'
        elif efficiency >= 50:
            return 'good_export_potential'
        else:
            return 'excellent_export_potential'
    
    def _rate_efficiency_potential(self, current: float, commercial_max: float) -> str:
        """Rate efficiency improvement potential."""
        if current == 0:
            return 'unknown'
        
        improvement_potential = ((commercial_max - current) / current) * 100
        if improvement_potential >= 100:
            return 'very_high_potential'
        elif improvement_potential >= 50:
            return 'high_potential'
        elif improvement_potential >= 25:
            return 'moderate_potential'
        else:
            return 'limited_potential'
    
    def _generate_policy_recommendations(self, 
                                       multi_tier_results: MultiTierResults,
                                       country_config: CountryConfiguration,
                                       national_statistics: Dict[str, Any]) -> List[str]:
        """Generate specific policy recommendations."""
        recommendations = []
        
        # Tier-specific recommendations
        commercial_reduction = multi_tier_results.get_width_reduction('commercial')
        if commercial_reduction and commercial_reduction >= 25:
            recommendations.append(
                f"Commercial agriculture tier shows {commercial_reduction:.1f}% envelope width reduction - "
                "focus policy on economically viable agricultural areas for maximum impact"
            )
        
        # Focus-specific recommendations
        if country_config.agricultural_focus == 'food_security':
            capacity_assessment = self._assess_food_security_capacity(multi_tier_results, national_statistics)
            if capacity_assessment.get('capacity_utilization_pct', 0) < 60:
                recommendations.append(
                    "Low capacity utilization detected - consider policies to increase agricultural productivity "
                    "and land use efficiency for food security"
                )
        
        elif country_config.agricultural_focus == 'export_capacity':
            capacity_assessment = self._assess_export_capacity(multi_tier_results, national_statistics)
            if capacity_assessment.get('commercial_efficiency_pct', 0) < 70:
                recommendations.append(
                    "Significant export potential identified - consider policies to support commercial "
                    "agricultural expansion and market access"
                )
        
        # Productivity-based recommendations
        avg_yield = national_statistics.get('average_yield_mt_per_ha', 0)
        if avg_yield > 0:
            # Compare with global averages (simplified heuristic)
            if self.config.crop_type == 'wheat' and avg_yield < 3.0:
                recommendations.append("Wheat yields below global average - consider productivity enhancement programs")
            elif self.config.crop_type == 'maize' and avg_yield < 5.0:
                recommendations.append("Maize yields below global average - consider improved varieties and practices")
            elif self.config.crop_type == 'rice' and avg_yield < 4.0:
                recommendations.append("Rice yields below global average - consider intensification programs")
        
        return recommendations
    
    def _validate_national_results(self, 
                                 multi_tier_results: MultiTierResults,
                                 country_config: CountryConfiguration,
                                 national_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate national results against known agricultural patterns."""
        validation = {
            'overall_valid': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Mathematical validation
        math_validation = self.multi_tier_engine.validate_multi_tier_results(multi_tier_results)
        validation['checks']['mathematical_properties'] = math_validation['overall_valid']
        if not math_validation['overall_valid']:
            validation['overall_valid'] = False
            validation['errors'].extend(math_validation.get('issues', []))
        
        # Production totals validation (basic sanity checks)
        total_production = national_statistics['total_production_mt']
        if total_production <= 0:
            validation['overall_valid'] = False
            validation['errors'].append("Zero or negative total production detected")
        elif total_production > 1e9:  # 1 billion MT seems unreasonably high
            validation['warnings'].append(f"Very high total production: {total_production:.0f} MT")
        
        # Yield validation
        avg_yield = national_statistics.get('average_yield_mt_per_ha', 0)
        if avg_yield <= 0:
            validation['warnings'].append("Zero or negative average yield")
        elif avg_yield > 20:  # 20 MT/ha seems very high for most crops
            validation['warnings'].append(f"Very high average yield: {avg_yield:.2f} MT/ha")
        
        # Coverage validation
        coverage = national_statistics.get('productivity_coverage_pct', 0)
        if coverage < 50:
            validation['warnings'].append(f"Low productivity coverage: {coverage:.1f}%")
        
        # Geographic extent validation
        geo_extent = national_statistics.get('geographic_extent', {})
        if geo_extent:
            lat_range = geo_extent['lat_max'] - geo_extent['lat_min']
            lon_range = geo_extent['lon_max'] - geo_extent['lon_min']
            
            if lat_range < 0.5 or lon_range < 0.5:
                validation['warnings'].append(
                    f"Small geographic extent: {lat_range:.2f}° lat × {lon_range:.2f}° lon"
                )
        
        validation['checks']['production_totals'] = total_production > 0
        validation['checks']['yield_reasonableness'] = 0 < avg_yield < 20
        validation['checks']['coverage_adequacy'] = coverage >= 50
        
        return validation
    
    def _calculate_comparative_metrics(self, 
                                    country_results: Dict[str, NationalAnalysisResults]) -> Dict[str, Any]:
        """Calculate comparative metrics between countries."""
        metrics = {
            'production_comparison': {},
            'efficiency_comparison': {},
            'tier_effectiveness_comparison': {},
            'summary_statistics': {}
        }
        
        # Production comparison
        for country_code, results in country_results.items():
            stats = results.national_statistics
            metrics['production_comparison'][country_code] = {
                'total_production_mt': stats['total_production_mt'],
                'total_harvest_area_ha': stats['total_harvest_area_ha'],
                'average_yield_mt_per_ha': stats['average_yield_mt_per_ha'],
                'productive_cells': stats['productive_cells']
            }
        
        # Efficiency comparison (commercial tier)
        for country_code, results in country_results.items():
            commercial_capacity = results.get_production_capacity('commercial')
            current_production = results.national_statistics['total_production_mt']
            
            efficiency_pct = (current_production / commercial_capacity.get('max_production_capacity', 1)) * 100 if commercial_capacity.get('max_production_capacity', 0) > 0 else 0
            
            metrics['efficiency_comparison'][country_code] = {
                'current_production_mt': current_production,
                'commercial_potential_mt': commercial_capacity.get('max_production_capacity', 0),
                'efficiency_pct': efficiency_pct,
                'production_gap_mt': commercial_capacity.get('max_production_capacity', 0) - current_production
            }
        
        # Tier effectiveness comparison
        for country_code, results in country_results.items():
            metrics['tier_effectiveness_comparison'][country_code] = {
                tier_name: results.get_width_reduction(tier_name)
                for tier_name in results.multi_tier_results.tier_results.keys()
            }
        
        # Summary statistics
        all_productions = [r.national_statistics['total_production_mt'] for r in country_results.values()]
        all_yields = [r.national_statistics['average_yield_mt_per_ha'] for r in country_results.values()]
        
        metrics['summary_statistics'] = {
            'total_countries': len(country_results),
            'total_production_all_countries_mt': sum(all_productions),
            'average_production_per_country_mt': np.mean(all_productions),
            'average_yield_all_countries_mt_per_ha': np.mean(all_yields),
            'production_coefficient_of_variation': np.std(all_productions) / np.mean(all_productions) if np.mean(all_productions) > 0 else 0
        }
        
        return metrics
    
    def _generate_comparative_policy_recommendations(self, 
                                                   country_results: Dict[str, NationalAnalysisResults],
                                                   comparative_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate policy recommendations based on country comparisons."""
        recommendations = {
            'cross_country_insights': [],
            'country_specific_priorities': {},
            'regional_cooperation_opportunities': []
        }
        
        # Identify high and low performers
        efficiency_data = comparative_metrics['efficiency_comparison']
        high_efficiency_countries = [
            country for country, data in efficiency_data.items()
            if data['efficiency_pct'] >= 70
        ]
        low_efficiency_countries = [
            country for country, data in efficiency_data.items()
            if data['efficiency_pct'] < 50
        ]
        
        # Cross-country insights
        if high_efficiency_countries and low_efficiency_countries:
            recommendations['cross_country_insights'].append(
                f"Efficiency gap identified: {', '.join(high_efficiency_countries)} show high efficiency "
                f"while {', '.join(low_efficiency_countries)} have significant improvement potential"
            )
        
        # Country-specific priorities
        for country_code, results in country_results.items():
            country_recommendations = []
            
            # Based on agricultural focus
            country_config = self.country_manager.get_country_configuration(country_code)
            if country_config:
                if country_config.agricultural_focus == 'export_capacity':
                    export_assessment = results.policy_insights.get('capacity_assessment', {})
                    if export_assessment.get('commercial_efficiency_pct', 0) < 70:
                        country_recommendations.append("Focus on commercial agricultural expansion for export growth")
                
                elif country_config.agricultural_focus == 'food_security':
                    food_security_assessment = results.policy_insights.get('capacity_assessment', {})
                    if food_security_assessment.get('capacity_utilization_pct', 0) < 60:
                        country_recommendations.append("Prioritize food security through productivity improvements")
            
            # Based on tier effectiveness
            commercial_reduction = results.get_width_reduction('commercial')
            if commercial_reduction and commercial_reduction >= 25:
                country_recommendations.append("Leverage commercial tier analysis for targeted policy interventions")
            
            recommendations['country_specific_priorities'][country_code] = country_recommendations
        
        # Regional cooperation opportunities
        production_data = comparative_metrics['production_comparison']
        high_production_countries = [
            country for country, data in production_data.items()
            if data['total_production_mt'] > comparative_metrics['summary_statistics']['average_production_per_country_mt']
        ]
        
        if len(high_production_countries) >= 2:
            recommendations['regional_cooperation_opportunities'].append(
                f"High-production countries ({', '.join(high_production_countries)}) could collaborate on "
                "technology transfer and best practices sharing"
            )
        
        return recommendations
    
    def export_analysis_results(self, 
                              results: NationalAnalysisResults,
                              output_dir: Path,
                              include_detailed_data: bool = True) -> Dict[str, Path]:
        """
        Export analysis results to files.
        
        Args:
            results: Analysis results to export
            output_dir: Output directory
            include_detailed_data: Whether to include detailed envelope data
        
        Returns:
            Dictionary mapping file types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export summary report
        summary_file = output_dir / f"{results.country_code}_{results.crop_type}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results.get_summary_report(), f, indent=2, default=str)
        exported_files['summary'] = summary_file
        
        # Export national statistics
        stats_file = output_dir / f"{results.country_code}_{results.crop_type}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(results.national_statistics, f, indent=2, default=str)
        exported_files['statistics'] = stats_file
        
        # Export policy insights
        policy_file = output_dir / f"{results.country_code}_{results.crop_type}_policy_insights.json"
        with open(policy_file, 'w') as f:
            json.dump(results.policy_insights, f, indent=2, default=str)
        exported_files['policy_insights'] = policy_file
        
        # Export detailed envelope data if requested
        if include_detailed_data:
            for tier_name, envelope_data in results.multi_tier_results.tier_results.items():
                envelope_file = output_dir / f"{results.country_code}_{results.crop_type}_{tier_name}_envelope.csv"
                
                # Create DataFrame with envelope data
                envelope_df = pd.DataFrame({
                    'disruption_areas': envelope_data.disruption_areas,
                    'lower_bound_harvest': envelope_data.lower_bound_harvest,
                    'lower_bound_production': envelope_data.lower_bound_production,
                    'upper_bound_harvest': envelope_data.upper_bound_harvest,
                    'upper_bound_production': envelope_data.upper_bound_production
                })
                
                envelope_df.to_csv(envelope_file, index=False)
                exported_files[f'{tier_name}_envelope'] = envelope_file
        
        self.logger.info(f"Exported {len(exported_files)} files to {output_dir}")
        return exported_files
    
    def generate_national_report(self, results: NationalAnalysisResults) -> str:
        """Generate a comprehensive text report for national analysis."""
        lines = [
            "=" * 80,
            f"NATIONAL AGRICULTURAL CAPACITY ANALYSIS: {results.country_name.upper()}",
            "=" * 80,
            "",
            "BASIC INFORMATION:",
            f"  Country: {results.country_name} ({results.country_code})",
            f"  Crop Type: {results.crop_type.title()}",
            f"  Analysis Date: {results.multi_tier_results.calculation_metadata.get('timestamp', 'Unknown')}",
            "",
            "NATIONAL STATISTICS:",
            f"  Total Production: {results.national_statistics['total_production_mt']:,.0f} MT",
            f"  Total Harvest Area: {results.national_statistics['total_harvest_area_ha']:,.0f} ha",
            f"  Average Yield: {results.national_statistics['average_yield_mt_per_ha']:.2f} MT/ha",
            f"  Productive Cells: {results.national_statistics['productive_cells']:,}",
            f"  Coverage: {results.national_statistics['productivity_coverage_pct']:.1f}%",
            ""
        ]
        
        # Multi-tier analysis
        lines.extend([
            "MULTI-TIER ANALYSIS:",
        ])
        
        for tier_name in results.multi_tier_results.tier_results.keys():
            width_reduction = results.get_width_reduction(tier_name)
            capacity = results.get_production_capacity(tier_name)
            
            lines.extend([
                f"  {tier_name.title()} Tier:",
                f"    Width Reduction: {width_reduction:.1f}%" if width_reduction else "    Width Reduction: N/A",
                f"    Max Production Capacity: {capacity.get('max_production_capacity', 0):,.0f} MT",
                f"    Total Harvest Area: {capacity.get('total_harvest_area', 0):,.0f} km²",
            ])
        
        lines.append("")
        
        # Policy insights
        if results.policy_insights.get('policy_recommendations'):
            lines.extend([
                "POLICY RECOMMENDATIONS:",
            ])
            for i, rec in enumerate(results.policy_insights['policy_recommendations'], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        # Validation status
        if results.validation_results:
            lines.extend([
                "VALIDATION STATUS:",
                f"  Overall Valid: {'✓' if results.validation_results.get('overall_valid', False) else '✗'}",
            ])
            
            if results.validation_results.get('warnings'):
                lines.append("  Warnings:")
                for warning in results.validation_results['warnings']:
                    lines.append(f"    - {warning}")
            
            if results.validation_results.get('errors'):
                lines.append("  Errors:")
                for error in results.validation_results['errors']:
                    lines.append(f"    - {error}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def clear_cache(self) -> None:
        """Clear analysis cache to free memory."""
        self._analysis_cache.clear()
        self.logger.info("NationalEnvelopeAnalyzer cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'analysis_cache_size': len(self._analysis_cache)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NationalEnvelopeAnalyzer("
            f"crop_type={self.config.crop_type}, "
            f"cached_analyses={len(self._analysis_cache)})"
        )
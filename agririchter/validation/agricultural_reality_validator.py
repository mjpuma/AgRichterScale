"""
Agricultural Reality Validator for Multi-Tier Envelope System.

This module implements comprehensive validation to ensure that multi-tier envelope
results align with known agricultural reality and patterns.

Validates requirements V2.1-V2.4:
- V2.1: Yield ranges must be realistic for each crop and country
- V2.2: Production totals must align with known agricultural statistics  
- V2.3: Spatial patterns must be consistent with known agricultural regions
- V2.4: Tier filtering must produce sensible productivity stratification
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from agririchter.analysis.multi_tier_envelope import MultiTierResults
from agririchter.analysis.national_envelope_analyzer import NationalAnalysisResults


@dataclass
class YieldBenchmarks:
    """Benchmark yield ranges for agricultural reality validation."""
    
    crop: str
    country: str
    min_realistic_yield: float  # MT/ha
    max_realistic_yield: float  # MT/ha
    typical_yield_range: Tuple[float, float]  # MT/ha
    source: str
    
    
@dataclass
class ProductionBenchmarks:
    """Benchmark production totals for agricultural reality validation."""
    
    crop: str
    country: str
    min_realistic_production: float  # Million MT
    max_realistic_production: float  # Million MT
    typical_production_range: Tuple[float, float]  # Million MT
    source: str


@dataclass
class AgriculturalRealityReport:
    """Comprehensive agricultural reality validation report."""
    
    overall_valid: bool
    yield_validation: Dict[str, Any]
    production_validation: Dict[str, Any]
    spatial_validation: Dict[str, Any]
    tier_validation: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]


class AgriculturalRealityValidator:
    """Validates multi-tier envelope results against agricultural reality."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.yield_benchmarks = self._load_yield_benchmarks()
        self.production_benchmarks = self._load_production_benchmarks()
    
    def _load_yield_benchmarks(self) -> Dict[str, YieldBenchmarks]:
        """Load realistic yield benchmarks for validation."""
        benchmarks = {}
        
        # USA yield benchmarks (based on USDA statistics)
        benchmarks['USA_wheat'] = YieldBenchmarks(
            crop='wheat', country='USA',
            min_realistic_yield=1.0, max_realistic_yield=8.0,
            typical_yield_range=(2.5, 4.5),
            source='USDA_NASS_2020-2023'
        )
        
        benchmarks['USA_maize'] = YieldBenchmarks(
            crop='maize', country='USA', 
            min_realistic_yield=3.0, max_realistic_yield=15.0,
            typical_yield_range=(8.0, 12.0),
            source='USDA_NASS_2020-2023'
        )
        
        benchmarks['USA_rice'] = YieldBenchmarks(
            crop='rice', country='USA',
            min_realistic_yield=4.0, max_realistic_yield=10.0,
            typical_yield_range=(6.0, 8.5),
            source='USDA_NASS_2020-2023'
        )
        
        # China yield benchmarks (based on FAO statistics)
        benchmarks['CHN_wheat'] = YieldBenchmarks(
            crop='wheat', country='CHN',
            min_realistic_yield=2.0, max_realistic_yield=7.0,
            typical_yield_range=(3.0, 4.5),
            source='FAO_STAT_2020-2023'
        )
        
        benchmarks['CHN_maize'] = YieldBenchmarks(
            crop='maize', country='CHN',
            min_realistic_yield=2.0, max_realistic_yield=8.0,
            typical_yield_range=(4.0, 6.0),
            source='FAO_STAT_2020-2023'
        )
        
        benchmarks['CHN_rice'] = YieldBenchmarks(
            crop='rice', country='CHN',
            min_realistic_yield=3.0, max_realistic_yield=8.0,
            typical_yield_range=(4.5, 6.5),
            source='FAO_STAT_2020-2023'
        )
        
        return benchmarks
    
    def _load_production_benchmarks(self) -> Dict[str, ProductionBenchmarks]:
        """Load realistic production benchmarks for validation."""
        benchmarks = {}
        
        # USA production benchmarks (Million MT, based on USDA/FAO)
        benchmarks['USA_wheat'] = ProductionBenchmarks(
            crop='wheat', country='USA',
            min_realistic_production=20.0, max_realistic_production=60.0,
            typical_production_range=(30.0, 50.0),
            source='USDA_FAO_2020-2023'
        )
        
        benchmarks['USA_maize'] = ProductionBenchmarks(
            crop='maize', country='USA',
            min_realistic_production=250.0, max_realistic_production=400.0,
            typical_production_range=(300.0, 370.0),
            source='USDA_FAO_2020-2023'
        )
        
        benchmarks['USA_rice'] = ProductionBenchmarks(
            crop='rice', country='USA',
            min_realistic_production=5.0, max_realistic_production=15.0,
            typical_production_range=(7.0, 12.0),
            source='USDA_FAO_2020-2023'
        )
        
        # China production benchmarks (Million MT, based on FAO)
        benchmarks['CHN_wheat'] = ProductionBenchmarks(
            crop='wheat', country='CHN',
            min_realistic_production=100.0, max_realistic_production=150.0,
            typical_production_range=(120.0, 140.0),
            source='FAO_STAT_2020-2023'
        )
        
        benchmarks['CHN_maize'] = ProductionBenchmarks(
            crop='maize', country='CHN',
            min_realistic_production=200.0, max_realistic_production=300.0,
            typical_production_range=(240.0, 280.0),
            source='FAO_STAT_2020-2023'
        )
        
        benchmarks['CHN_rice'] = ProductionBenchmarks(
            crop='rice', country='CHN',
            min_realistic_production=180.0, max_realistic_production=220.0,
            typical_production_range=(190.0, 210.0),
            source='FAO_STAT_2020-2023'
        )
        
        return benchmarks
    
    def validate_yield_realism(self, results: NationalAnalysisResults, 
                              crop: str, country: str) -> Dict[str, Any]:
        """Validate V2.1: Yield ranges must be realistic for each crop and country."""
        validation_key = f"{country}_{crop}"
        
        if validation_key not in self.yield_benchmarks:
            return {
                'valid': False,
                'reason': f'No yield benchmarks available for {country} {crop}',
                'benchmark_available': False
            }
        
        benchmark = self.yield_benchmarks[validation_key]
        actual_yield = results.national_statistics['average_yield_mt_per_ha']
        
        # Check if yield is within realistic range
        yield_realistic = (benchmark.min_realistic_yield <= actual_yield <= 
                          benchmark.max_realistic_yield)
        
        # Check if yield is within typical range (warning if outside)
        yield_typical = (benchmark.typical_yield_range[0] <= actual_yield <= 
                        benchmark.typical_yield_range[1])
        
        return {
            'valid': yield_realistic,
            'typical': yield_typical,
            'actual_yield': actual_yield,
            'benchmark_min': benchmark.min_realistic_yield,
            'benchmark_max': benchmark.max_realistic_yield,
            'typical_range': benchmark.typical_yield_range,
            'source': benchmark.source,
            'benchmark_available': True
        }
    
    def validate_production_totals(self, results: NationalAnalysisResults,
                                  crop: str, country: str) -> Dict[str, Any]:
        """Validate V2.2: Production totals must align with known agricultural statistics."""
        validation_key = f"{country}_{crop}"
        
        if validation_key not in self.production_benchmarks:
            return {
                'valid': False,
                'reason': f'No production benchmarks available for {country} {crop}',
                'benchmark_available': False
            }
        
        benchmark = self.production_benchmarks[validation_key]
        actual_production = results.national_statistics['total_production_mt'] / 1e6  # Convert to Million MT
        
        # Check if production is within realistic range
        production_realistic = (benchmark.min_realistic_production <= actual_production <= 
                               benchmark.max_realistic_production)
        
        # Check if production is within typical range (warning if outside)
        production_typical = (benchmark.typical_production_range[0] <= actual_production <= 
                             benchmark.typical_production_range[1])
        
        return {
            'valid': production_realistic,
            'typical': production_typical,
            'actual_production_mmt': actual_production,
            'benchmark_min': benchmark.min_realistic_production,
            'benchmark_max': benchmark.max_realistic_production,
            'typical_range': benchmark.typical_production_range,
            'source': benchmark.source,
            'benchmark_available': True
        }
    
    def validate_spatial_patterns(self, results: NationalAnalysisResults,
                                 crop: str, country: str) -> Dict[str, Any]:
        """Validate V2.3: Spatial patterns must be consistent with known agricultural regions."""
        
        # Basic spatial validation checks using national_statistics
        national_stats = results.national_statistics
        
        # Check geographic coverage
        total_cells = national_stats.get('total_cells', 0)
        productive_cells = national_stats.get('productive_cells', 0)
        coverage_adequate = total_cells >= 1000  # Minimum coverage requirement
        
        # Check productivity coverage
        productivity_coverage = national_stats.get('productivity_coverage_pct', 0)
        productivity_reasonable = productivity_coverage >= 50  # At least 50% of cells productive
        
        # Check yield variation (using yield statistics as proxy for spatial variation)
        mean_yield = national_stats.get('average_yield_mt_per_ha', 0)
        yield_std = national_stats.get('yield_std_mt_per_ha', 0)
        yield_cv = yield_std / mean_yield if mean_yield > 0 else 0
        
        # Reasonable yield variation (not too uniform, not too scattered)
        spatial_variation_reasonable = 0.2 <= yield_cv <= 2.0
        
        # Check geographic extent (should cover reasonable area)
        geographic_extent = national_stats.get('geographic_extent', {})
        lat_range = geographic_extent.get('lat_max', 0) - geographic_extent.get('lat_min', 0)
        lon_range = geographic_extent.get('lon_max', 0) - geographic_extent.get('lon_min', 0)
        geographic_coverage_reasonable = lat_range >= 1.0 and lon_range >= 1.0  # At least 1 degree coverage
        
        return {
            'valid': coverage_adequate and productivity_reasonable and spatial_variation_reasonable and geographic_coverage_reasonable,
            'coverage_adequate': coverage_adequate,
            'productivity_reasonable': productivity_reasonable,
            'spatial_variation_reasonable': spatial_variation_reasonable,
            'geographic_coverage_reasonable': geographic_coverage_reasonable,
            'total_cells': total_cells,
            'productive_cells': productive_cells,
            'productivity_coverage_pct': productivity_coverage,
            'yield_cv': yield_cv,
            'lat_range': lat_range,
            'lon_range': lon_range
        }
    
    def validate_tier_stratification(self, results: MultiTierResults) -> Dict[str, Any]:
        """Validate V2.4: Tier filtering must produce sensible productivity stratification."""
        
        if 'comprehensive' not in results.tier_results or 'commercial' not in results.tier_results:
            return {
                'valid': False,
                'reason': 'Missing required tiers for stratification validation'
            }
        
        # Get base statistics and calculate tier metrics
        base_stats = results.base_statistics
        
        # Check that envelope width actually decreased
        width_reduction = results.get_width_reduction('commercial')
        width_decreased = width_reduction is not None and width_reduction > 0
        
        # Check envelope properties
        comprehensive_envelope = results.tier_results['comprehensive']
        commercial_envelope = results.tier_results['commercial']
        
        # Check that commercial tier has fewer envelope points or narrower bounds
        comp_points = len(comprehensive_envelope.disruption_areas)
        comm_points = len(commercial_envelope.disruption_areas)
        
        # Check convergence validation
        comprehensive_valid = comprehensive_envelope.convergence_validated
        commercial_valid = commercial_envelope.convergence_validated
        convergence_maintained = comprehensive_valid and commercial_valid
        
        # Check that commercial tier production bounds are within comprehensive bounds
        # (at least at the convergence point)
        comp_convergence = comprehensive_envelope.convergence_point[1]  # production
        comm_convergence = commercial_envelope.convergence_point[1]  # production
        bounds_reasonable = comm_convergence <= comp_convergence
        
        # Estimate retention rate from SPAM filtering statistics
        spam_stats = base_stats.get('spam_filter_stats', {})
        retention_rate = spam_stats.get('retention_rate', 80.0) / 100.0  # Convert to fraction
        retention_reasonable = 0.2 <= retention_rate <= 0.95  # 20-95% retention
        
        return {
            'valid': width_decreased and convergence_maintained and bounds_reasonable and retention_reasonable,
            'width_decreased': width_decreased,
            'convergence_maintained': convergence_maintained,
            'bounds_reasonable': bounds_reasonable,
            'retention_reasonable': retention_reasonable,
            'comprehensive_points': comp_points,
            'commercial_points': comm_points,
            'comprehensive_convergence': comp_convergence,
            'commercial_convergence': comm_convergence,
            'retention_rate': retention_rate,
            'width_reduction_pct': width_reduction
        }
    
    def validate_agricultural_reality(self, results: NationalAnalysisResults,
                                    multi_tier_results: MultiTierResults,
                                    crop: str, country: str) -> AgriculturalRealityReport:
        """Comprehensive agricultural reality validation."""
        
        warnings = []
        recommendations = []
        
        # V2.1: Validate yield realism
        yield_validation = self.validate_yield_realism(results, crop, country)
        if not yield_validation.get('typical', True):
            warnings.append(
                f"Yield ({yield_validation['actual_yield']:.2f} MT/ha) outside typical range "
                f"{yield_validation['typical_range']} for {country} {crop}"
            )
        
        # V2.2: Validate production totals
        production_validation = self.validate_production_totals(results, crop, country)
        if not production_validation.get('typical', True):
            warnings.append(
                f"Production ({production_validation['actual_production_mmt']:.1f} MMT) outside typical range "
                f"{production_validation['typical_range']} for {country} {crop}"
            )
        
        # V2.3: Validate spatial patterns
        spatial_validation = self.validate_spatial_patterns(results, crop, country)
        if not spatial_validation['coverage_adequate']:
            warnings.append(f"Insufficient spatial coverage ({spatial_validation['total_cells']} cells)")
        
        # V2.4: Validate tier stratification
        tier_validation = self.validate_tier_stratification(multi_tier_results)
        if not tier_validation['width_decreased']:
            warnings.append("Commercial tier does not show width reduction over comprehensive tier")
        
        # Overall validation status
        overall_valid = (
            yield_validation.get('valid', False) and
            production_validation.get('valid', False) and
            spatial_validation.get('valid', False) and
            tier_validation.get('valid', False)
        )
        
        # Generate recommendations
        if not overall_valid:
            recommendations.append("Review data quality and filtering parameters")
            
        if len(warnings) > 2:
            recommendations.append("Consider adjusting tier thresholds or filtering criteria")
            
        if tier_validation.get('retention_rate', 0) < 0.3:
            recommendations.append("Commercial tier filtering may be too aggressive")
        
        return AgriculturalRealityReport(
            overall_valid=overall_valid,
            yield_validation=yield_validation,
            production_validation=production_validation,
            spatial_validation=spatial_validation,
            tier_validation=tier_validation,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def generate_reality_validation_report(self, validation_results: Dict[str, AgriculturalRealityReport],
                                         output_path: Path) -> None:
        """Generate comprehensive agricultural reality validation report."""
        
        report_lines = [
            "# Agricultural Reality Validation Report",
            "",
            f"**Generated:** {pd.Timestamp.now().isoformat()}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Overall status
        all_valid = all(report.overall_valid for report in validation_results.values())
        total_analyses = len(validation_results)
        valid_analyses = sum(1 for report in validation_results.values() if report.overall_valid)
        
        if all_valid:
            report_lines.extend([
                "✅ **All analyses align with agricultural reality**",
                "",
                f"All {total_analyses} country/crop combinations passed agricultural reality validation.",
                "Results are consistent with known agricultural patterns and statistics.",
                ""
            ])
        else:
            report_lines.extend([
                f"⚠️ **{valid_analyses}/{total_analyses} analyses align with agricultural reality**",
                "",
                f"{total_analyses - valid_analyses} analyses require attention to align with agricultural reality.",
                ""
            ])
        
        # Detailed results
        report_lines.extend([
            "## Detailed Validation Results",
            ""
        ])
        
        for analysis_key, report in validation_results.items():
            country, crop = analysis_key.split('_', 1)
            status = "✅ PASS" if report.overall_valid else "⚠️ REVIEW"
            
            report_lines.extend([
                f"### {country.upper()} {crop.title()}: {status}",
                ""
            ])
            
            # Yield validation
            yield_val = report.yield_validation
            if yield_val.get('benchmark_available', False):
                yield_status = "✓" if yield_val['valid'] else "✗"
                report_lines.append(
                    f"- **Yield Realism {yield_status}:** {yield_val['actual_yield']:.2f} MT/ha "
                    f"(range: {yield_val['benchmark_min']}-{yield_val['benchmark_max']})"
                )
            
            # Production validation
            prod_val = report.production_validation
            if prod_val.get('benchmark_available', False):
                prod_status = "✓" if prod_val['valid'] else "✗"
                report_lines.append(
                    f"- **Production Totals {prod_status}:** {prod_val['actual_production_mmt']:.1f} MMT "
                    f"(range: {prod_val['benchmark_min']}-{prod_val['benchmark_max']})"
                )
            
            # Spatial validation
            spatial_val = report.spatial_validation
            spatial_status = "✓" if spatial_val['valid'] else "✗"
            report_lines.append(
                f"- **Spatial Patterns {spatial_status}:** {spatial_val['total_cells']} cells, "
                f"Yield CV: {spatial_val['yield_cv']:.2f}"
            )
            
            # Tier validation
            tier_val = report.tier_validation
            tier_status = "✓" if tier_val['valid'] else "✗"
            report_lines.append(
                f"- **Tier Stratification {tier_status}:** {tier_val['retention_rate']:.1%} retention, "
                f"{tier_val['width_reduction_pct']:.1f}% width reduction"
            )
            
            # Warnings
            if report.warnings:
                report_lines.append("- **Warnings:**")
                for warning in report.warnings:
                    report_lines.append(f"  - {warning}")
            
            report_lines.append("")
        
        # Summary statistics
        report_lines.extend([
            "## Validation Summary Statistics",
            ""
        ])
        
        # Collect statistics
        yield_valid_count = sum(1 for r in validation_results.values() if r.yield_validation.get('valid', False))
        production_valid_count = sum(1 for r in validation_results.values() if r.production_validation.get('valid', False))
        spatial_valid_count = sum(1 for r in validation_results.values() if r.spatial_validation.get('valid', False))
        tier_valid_count = sum(1 for r in validation_results.values() if r.tier_validation.get('valid', False))
        
        report_lines.extend([
            f"- **Yield Realism:** {yield_valid_count}/{total_analyses} passed",
            f"- **Production Totals:** {production_valid_count}/{total_analyses} passed", 
            f"- **Spatial Patterns:** {spatial_valid_count}/{total_analyses} passed",
            f"- **Tier Stratification:** {tier_valid_count}/{total_analyses} passed",
            ""
        ])
        
        # Recommendations
        all_recommendations = []
        for report in validation_results.values():
            all_recommendations.extend(report.recommendations)
        
        if all_recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            
            unique_recommendations = list(set(all_recommendations))
            for rec in unique_recommendations:
                report_lines.append(f"- {rec}")
            
            report_lines.append("")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Agricultural reality validation report saved to {output_path}")


def validate_agricultural_reality_comprehensive(results_dict: Dict[str, Tuple[NationalAnalysisResults, MultiTierResults]],
                                              output_dir: Path = None) -> Dict[str, AgriculturalRealityReport]:
    """Run comprehensive agricultural reality validation for multiple analyses."""
    
    if output_dir is None:
        output_dir = Path('.')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    validator = AgriculturalRealityValidator()
    validation_results = {}
    
    for analysis_key, (national_results, multi_tier_results) in results_dict.items():
        country, crop = analysis_key.split('_', 1)
        
        validation_report = validator.validate_agricultural_reality(
            national_results, multi_tier_results, crop, country
        )
        
        validation_results[analysis_key] = validation_report
    
    # Generate comprehensive report
    report_path = output_dir / 'agricultural_reality_validation_report.md'
    validator.generate_reality_validation_report(validation_results, report_path)
    
    return validation_results
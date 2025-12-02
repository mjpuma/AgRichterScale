"""Diagnostic tools for H-P envelope convergence analysis."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .convergence_validator import ConvergenceValidator, ValidationResult


class EnvelopeDiagnostics:
    """Diagnostic tools for analyzing envelope convergence behavior."""
    
    def __init__(self, validator: Optional[ConvergenceValidator] = None):
        """
        Initialize envelope diagnostics.
        
        Args:
            validator: ConvergenceValidator instance (creates new one if None)
        """
        self.validator = validator or ConvergenceValidator()
        self.logger = logging.getLogger('agririchter.envelope_diagnostics')
    
    def analyze_convergence_behavior(self, envelope_data: Dict[str, np.ndarray],
                                   total_production: float,
                                   total_harvest: float,
                                   hp_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze envelope convergence properties.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
            hp_matrix: Optional H-P matrix for additional analysis
        
        Returns:
            Dictionary with comprehensive convergence analysis
        """
        analysis = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': self._analyze_data_summary(envelope_data),
            'convergence_metrics': self._analyze_convergence_metrics(
                envelope_data, total_production, total_harvest
            ),
            'envelope_properties': self._analyze_envelope_properties(envelope_data),
            'mathematical_validation': None,
            'recommendations': []
        }
        
        # Perform mathematical validation
        validation_result = self.validator.validate_mathematical_properties(
            envelope_data, total_production, total_harvest
        )
        analysis['mathematical_validation'] = {
            'is_valid': validation_result.is_valid,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'properties': validation_result.properties,
            'statistics': validation_result.statistics
        }
        
        # Add HP matrix analysis if provided
        if hp_matrix is not None:
            analysis['hp_matrix_analysis'] = self._analyze_hp_matrix(
                hp_matrix, total_production, total_harvest
            )
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        self.logger.info("Convergence behavior analysis completed")
        return analysis
    
    def generate_convergence_report(self, envelope_data: Dict[str, np.ndarray],
                                  total_production: float,
                                  total_harvest: float,
                                  hp_matrix: Optional[np.ndarray] = None) -> str:
        """
        Generate detailed convergence analysis report.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
            hp_matrix: Optional H-P matrix for additional analysis
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_convergence_behavior(
            envelope_data, total_production, total_harvest, hp_matrix
        )
        
        report_lines = [
            "=" * 80,
            "H-P ENVELOPE CONVERGENCE ANALYSIS REPORT",
            "=" * 80,
            f"Analysis Date: {analysis['timestamp']}",
            "",
            "DATA SUMMARY",
            "-" * 12,
            f"Envelope Points: {analysis['data_summary']['envelope_points']}",
            f"Harvest Area Range: {analysis['data_summary']['harvest_range'][0]:,.0f} - {analysis['data_summary']['harvest_range'][1]:,.0f} km²",
            f"Production Range: {analysis['data_summary']['production_range'][0]:.2e} - {analysis['data_summary']['production_range'][1]:.2e} kcal",
            "",
            "CONVERGENCE METRICS",
            "-" * 18,
            f"Harvest Coverage: {analysis['convergence_metrics']['harvest_coverage']:.1%}",
            f"Production Coverage: {analysis['convergence_metrics']['production_coverage']:.1%}",
            f"Convergence Distance: {analysis['convergence_metrics']['convergence_distance']:.2e}",
            f"Endpoint Error: {analysis['convergence_metrics']['endpoint_error']:.2%}",
            "",
            "ENVELOPE PROPERTIES",
            "-" * 18,
            f"Average Width: {analysis['envelope_properties']['avg_width']:.2e} kcal",
            f"Width Reduction: {analysis['envelope_properties']['width_reduction']:.1%}",
            f"Monotonic: {analysis['envelope_properties']['monotonic']}",
            f"Upper Dominance: {analysis['envelope_properties']['upper_dominance']}",
            "",
            "MATHEMATICAL VALIDATION",
            "-" * 22,
            f"Overall Valid: {analysis['mathematical_validation']['is_valid']}",
        ]
        
        # Add property details
        properties = analysis['mathematical_validation']['properties']
        for prop_name, prop_value in properties.items():
            formatted_name = prop_name.replace('_', ' ').title()
            report_lines.append(f"  {formatted_name}: {prop_value}")
        
        # Add errors and warnings
        if analysis['mathematical_validation']['errors']:
            report_lines.extend([
                "",
                "VALIDATION ERRORS",
                "-" * 17
            ])
            for error in analysis['mathematical_validation']['errors']:
                report_lines.append(f"  • {error}")
        
        if analysis['mathematical_validation']['warnings']:
            report_lines.extend([
                "",
                "VALIDATION WARNINGS",
                "-" * 19
            ])
            for warning in analysis['mathematical_validation']['warnings']:
                report_lines.append(f"  • {warning}")
        
        # Add recommendations
        if analysis['recommendations']:
            report_lines.extend([
                "",
                "RECOMMENDATIONS",
                "-" * 15
            ])
            for i, rec in enumerate(analysis['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
        
        # Add HP matrix analysis if available
        if 'hp_matrix_analysis' in analysis:
            hp_analysis = analysis['hp_matrix_analysis']
            report_lines.extend([
                "",
                "H-P MATRIX ANALYSIS",
                "-" * 19,
                f"Total Cells: {hp_analysis['total_cells']}",
                f"Valid Cells: {hp_analysis['valid_cells']}",
                f"Yield Range: {hp_analysis['yield_range'][0]:.2e} - {hp_analysis['yield_range'][1]:.2e} kcal/km²",
                f"Properly Sorted: {hp_analysis['properly_sorted']}",
                f"Conservation Valid: {hp_analysis['conservation_valid']}"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def plot_convergence_analysis(self, envelope_data: Dict[str, np.ndarray],
                                total_production: float,
                                total_harvest: float,
                                save_path: Optional[str] = None) -> Figure:
        """
        Create diagnostic plots showing convergence behavior.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
            save_path: Optional path to save the plot
        
        Returns:
            Matplotlib figure with convergence analysis plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('H-P Envelope Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        lower_harvest = envelope_data['lower_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_harvest = envelope_data['upper_bound_harvest']
        upper_production = envelope_data['upper_bound_production']
        
        # Plot 1: Full envelope with convergence point
        ax1 = axes[0, 0]
        self._plot_envelope_with_convergence(
            ax1, lower_harvest, lower_production, upper_harvest, upper_production,
            total_production, total_harvest
        )
        
        # Plot 2: Envelope width analysis
        ax2 = axes[0, 1]
        self._plot_envelope_width(
            ax2, lower_harvest, lower_production, upper_harvest, upper_production
        )
        
        # Plot 3: Convergence behavior near endpoint
        ax3 = axes[1, 0]
        self._plot_convergence_detail(
            ax3, lower_harvest, lower_production, upper_harvest, upper_production,
            total_production, total_harvest
        )
        
        # Plot 4: Validation summary
        ax4 = axes[1, 1]
        validation_result = self.validator.validate_mathematical_properties(
            envelope_data, total_production, total_harvest
        )
        self._plot_validation_summary(ax4, validation_result)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Convergence analysis plot saved to {save_path}")
        
        return fig
    
    def create_bounds_convergence_plot(self, envelope_data: Dict[str, np.ndarray],
                                     total_production: float, total_harvest: float,
                                     save_path: Optional[str] = None) -> Figure:
        """
        Create a specialized plot showing how bounds approach each other.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production from all cells
            total_harvest: Total harvest area from all cells
            save_path: Optional path to save the plot
        
        Returns:
            Matplotlib figure with bounds convergence analysis
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Envelope Bounds Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        lower_harvest = envelope_data['lower_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_harvest = envelope_data['upper_bound_harvest']
        upper_production = envelope_data['upper_bound_production']
        
        # Plot 1: Bounds approaching each other (full envelope)
        ax1.fill_between(lower_harvest, lower_production, upper_production, 
                        alpha=0.2, color='blue', label='Envelope Area')
        ax1.plot(lower_harvest, lower_production, 'b-', linewidth=2, label='Lower Bound')
        ax1.plot(upper_harvest, upper_production, 'r-', linewidth=2, label='Upper Bound')
        
        # Mark convergence point
        ax1.plot(total_harvest, total_production, 'go', markersize=10, 
                label=f'Expected Convergence')
        
        # Add convergence arrows showing bounds approaching
        if len(lower_harvest) > 10:
            # Add arrows at several points to show convergence direction
            arrow_indices = [len(lower_harvest)//4, len(lower_harvest)//2, 3*len(lower_harvest)//4]
            for idx in arrow_indices:
                if idx < len(lower_harvest):
                    x_pos = lower_harvest[idx]
                    y_lower = lower_production[idx]
                    y_upper = upper_production[idx]
                    y_mid = (y_lower + y_upper) / 2
                    
                    # Arrow from upper to lower (convergence)
                    ax1.annotate('', xy=(x_pos, y_lower), xytext=(x_pos, y_upper),
                               arrowprops=dict(arrowstyle='<->', color='purple', lw=2, alpha=0.7))
        
        ax1.set_xlabel('Harvest Area (km²)')
        ax1.set_ylabel('Production (kcal)')
        ax1.set_title('Full Envelope - Bounds Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Envelope width over harvest area
        envelope_width = upper_production - lower_production
        ax2.plot(lower_harvest, envelope_width, 'purple', linewidth=3, label='Envelope Width')
        ax2.fill_between(lower_harvest, 0, envelope_width, alpha=0.3, color='purple')
        
        # Add width reduction statistics
        if len(envelope_width) > 1:
            initial_width = envelope_width[0]
            final_width = envelope_width[-1]
            width_reduction = (initial_width - final_width) / initial_width * 100
            
            # Mark initial and final widths
            ax2.plot(lower_harvest[0], initial_width, 'ro', markersize=8, label=f'Initial Width: {initial_width:.2e}')
            ax2.plot(lower_harvest[-1], final_width, 'go', markersize=8, label=f'Final Width: {final_width:.2e}')
            
            # Add reduction annotation
            ax2.text(0.5, 0.8, f'Width Reduction: {width_reduction:.1f}%', 
                    transform=ax2.transAxes, ha='center', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        ax2.set_xlabel('Harvest Area (km²)')
        ax2.set_ylabel('Envelope Width (kcal)')
        ax2.set_title('Envelope Width - Convergence Behavior')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Bounds convergence plot saved to {save_path}")
        
        return fig
    
    def _analyze_data_summary(self, envelope_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze basic data summary."""
        lower_harvest = envelope_data['lower_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_production = envelope_data['upper_bound_production']
        
        return {
            'envelope_points': len(lower_harvest),
            'harvest_range': (float(np.min(lower_harvest)), float(np.max(lower_harvest))),
            'production_range': (float(np.min(lower_production)), float(np.max(upper_production)))
        }
    
    def _analyze_convergence_metrics(self, envelope_data: Dict[str, np.ndarray],
                                   total_production: float, total_harvest: float) -> Dict[str, float]:
        """Analyze convergence-specific metrics."""
        lower_harvest = envelope_data['lower_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_production = envelope_data['upper_bound_production']
        
        # Coverage metrics
        max_harvest = np.max(lower_harvest)
        max_production = np.max(lower_production)
        harvest_coverage = max_harvest / total_harvest
        production_coverage = max_production / total_production
        
        # Convergence distance (how close bounds are at the end)
        final_lower = lower_production[-1]
        final_upper = upper_production[-1]
        convergence_distance = abs(final_upper - final_lower)
        
        # Endpoint error (how far from expected total production)
        expected_final = total_production
        endpoint_error = abs(final_lower - expected_final) / expected_final
        
        return {
            'harvest_coverage': float(harvest_coverage),
            'production_coverage': float(production_coverage),
            'convergence_distance': float(convergence_distance),
            'endpoint_error': float(endpoint_error)
        }
    
    def _analyze_envelope_properties(self, envelope_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze envelope shape properties."""
        lower_harvest = envelope_data['lower_bound_harvest']
        lower_production = envelope_data['lower_bound_production']
        upper_harvest = envelope_data['upper_bound_harvest']
        upper_production = envelope_data['upper_bound_production']
        
        # Width analysis
        production_width = upper_production - lower_production
        avg_width = np.mean(production_width)
        initial_width = production_width[0] if len(production_width) > 0 else 0
        final_width = production_width[-1] if len(production_width) > 0 else 0
        width_reduction = (initial_width - final_width) / initial_width if initial_width > 0 else 0
        
        # Monotonicity
        lower_monotonic = np.all(np.diff(lower_harvest) >= 0)
        upper_monotonic = np.all(np.diff(upper_harvest) >= 0)
        monotonic = lower_monotonic and upper_monotonic
        
        # Upper dominance
        upper_dominance = np.all(upper_production >= lower_production)
        
        return {
            'avg_width': float(avg_width),
            'width_reduction': float(width_reduction),
            'monotonic': monotonic,
            'upper_dominance': upper_dominance
        }
    
    def _analyze_hp_matrix(self, hp_matrix: np.ndarray, 
                          total_production: float, total_harvest: float) -> Dict[str, Any]:
        """Analyze H-P matrix properties."""
        total_cells = len(hp_matrix)
        valid_cells = np.sum(~np.isnan(hp_matrix).any(axis=1))
        
        yields = hp_matrix[:, 2]
        yield_range = (float(np.min(yields)), float(np.max(yields)))
        
        # Check if properly sorted
        properly_sorted = self.validator.validate_yield_ordering(hp_matrix)
        
        # Check conservation
        conservation_valid = self.validator.validate_cumulative_properties(
            hp_matrix, total_production, total_harvest
        )
        
        return {
            'total_cells': total_cells,
            'valid_cells': valid_cells,
            'yield_range': yield_range,
            'properly_sorted': properly_sorted,
            'conservation_valid': conservation_valid
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        validation = analysis['mathematical_validation']
        convergence = analysis['convergence_metrics']
        properties = analysis['envelope_properties']
        
        # Check validation issues
        if not validation['is_valid']:
            recommendations.append("Mathematical validation failed - consider using enforce_convergence method")
        
        # Check coverage
        if convergence['harvest_coverage'] < 0.95:
            recommendations.append("Envelope doesn't reach full harvest area - extend disruption range")
        
        if convergence['endpoint_error'] > 0.05:
            recommendations.append("Large endpoint error - verify total production calculation")
        
        # Check envelope properties
        if not properties['monotonic']:
            recommendations.append("Non-monotonic envelope detected - check data sorting")
        
        if not properties['upper_dominance']:
            recommendations.append("Upper bound dominance violated - check calculation algorithm")
        
        if properties['width_reduction'] < 0.5:
            recommendations.append("Poor envelope convergence - bounds should narrow significantly")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Envelope appears mathematically sound")
        
        return recommendations
    
    def _plot_envelope_with_convergence(self, ax: Axes, 
                                      lower_harvest: np.ndarray, lower_production: np.ndarray,
                                      upper_harvest: np.ndarray, upper_production: np.ndarray,
                                      total_production: float, total_harvest: float) -> None:
        """Plot envelope with convergence point highlighted."""
        ax.fill_between(lower_harvest, lower_production, upper_production, 
                       alpha=0.3, color='blue', label='Envelope')
        ax.plot(lower_harvest, lower_production, 'b-', linewidth=2, label='Lower Bound')
        ax.plot(upper_harvest, upper_production, 'r-', linewidth=2, label='Upper Bound')
        
        # Mark convergence point
        ax.plot(total_harvest, total_production, 'go', markersize=10, 
               label=f'Expected Convergence\n({total_harvest:.0f}, {total_production:.2e})')
        
        # Mark actual endpoint
        if len(lower_harvest) > 0:
            ax.plot(lower_harvest[-1], lower_production[-1], 'ro', markersize=8,
                   label=f'Actual Endpoint\n({lower_harvest[-1]:.0f}, {lower_production[-1]:.2e})')
        
        ax.set_xlabel('Harvest Area (km²)')
        ax.set_ylabel('Production (kcal)')
        ax.set_title('H-P Envelope with Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_envelope_width(self, ax: Axes,
                           lower_harvest: np.ndarray, lower_production: np.ndarray,
                           upper_harvest: np.ndarray, upper_production: np.ndarray) -> None:
        """Plot envelope width analysis."""
        production_width = upper_production - lower_production
        
        ax.plot(lower_harvest, production_width, 'g-', linewidth=2)
        ax.set_xlabel('Harvest Area (km²)')
        ax.set_ylabel('Envelope Width (kcal)')
        ax.set_title('Envelope Width vs Harvest Area')
        ax.grid(True, alpha=0.3)
        
        # Add width reduction annotation
        if len(production_width) > 1:
            initial_width = production_width[0]
            final_width = production_width[-1]
            reduction = (initial_width - final_width) / initial_width * 100
            ax.text(0.05, 0.95, f'Width Reduction: {reduction:.1f}%',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    def _plot_convergence_detail(self, ax: Axes,
                               lower_harvest: np.ndarray, lower_production: np.ndarray,
                               upper_harvest: np.ndarray, upper_production: np.ndarray,
                               total_production: float, total_harvest: float) -> None:
        """Plot detailed view of convergence behavior with bounds approaching each other."""
        # Focus on the last 20% of the envelope
        n_points = len(lower_harvest)
        start_idx = max(0, int(0.8 * n_points))
        
        harvest_detail = lower_harvest[start_idx:]
        lower_detail = lower_production[start_idx:]
        upper_detail = upper_production[start_idx:]
        
        # Plot bounds with different styles to show convergence
        ax.plot(harvest_detail, lower_detail, 'b-', linewidth=3, label='Lower Bound', alpha=0.8)
        ax.plot(harvest_detail, upper_detail, 'r-', linewidth=3, label='Upper Bound', alpha=0.8)
        
        # Fill area between bounds to show convergence
        ax.fill_between(harvest_detail, lower_detail, upper_detail, 
                       alpha=0.2, color='purple', label='Convergence Zone')
        
        # Mark expected convergence point
        ax.plot(total_harvest, total_production, 'go', markersize=12, 
               markeredgecolor='darkgreen', markeredgewidth=2,
               label=f'Expected Convergence\n({total_harvest:.0f}, {total_production:.2e})')
        
        # Mark actual endpoints
        if len(harvest_detail) > 0:
            ax.plot(harvest_detail[-1], lower_detail[-1], 'bo', markersize=10,
                   markeredgecolor='darkblue', markeredgewidth=2,
                   label=f'Actual Lower End\n({harvest_detail[-1]:.0f}, {lower_detail[-1]:.2e})')
            ax.plot(harvest_detail[-1], upper_detail[-1], 'ro', markersize=10,
                   markeredgecolor='darkred', markeredgewidth=2,
                   label=f'Actual Upper End\n({harvest_detail[-1]:.0f}, {upper_detail[-1]:.2e})')
        
        # Add convergence statistics as text annotations
        if len(harvest_detail) > 1:
            convergence_distance = abs(upper_detail[-1] - lower_detail[-1])
            width_reduction = (upper_detail[0] - lower_detail[0] - convergence_distance) / (upper_detail[0] - lower_detail[0]) * 100
            
            stats_text = f'Convergence Statistics:\n'
            stats_text += f'Final Gap: {convergence_distance:.2e} kcal\n'
            stats_text += f'Width Reduction: {width_reduction:.1f}%\n'
            stats_text += f'Points Analyzed: {len(harvest_detail)}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Harvest Area (km²)')
        ax.set_ylabel('Production (kcal)')
        ax.set_title('Convergence Detail - Bounds Approaching Each Other')
        ax.legend(loc='center right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_validation_summary(self, ax: Axes, validation_result: ValidationResult) -> None:
        """Plot validation summary with visual indicators for mathematical property validation."""
        ax.axis('off')
        
        # Create a visual validation dashboard
        properties = validation_result.properties
        n_props = len(properties)
        
        if n_props == 0:
            ax.text(0.5, 0.5, 'No validation properties available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Create a grid layout for property indicators
        cols = 2
        rows = (n_props + 1) // cols
        
        y_positions = np.linspace(0.9, 0.1, rows)
        x_positions = [0.25, 0.75]
        
        prop_items = list(properties.items())
        
        for i, (prop_name, prop_value) in enumerate(prop_items):
            row = i // cols
            col = i % cols
            
            x = x_positions[col]
            y = y_positions[row] if row < len(y_positions) else 0.1
            
            # Choose color and symbol based on validation result
            if prop_value:
                color = 'lightgreen'
                symbol = '✓'
                text_color = 'darkgreen'
            else:
                color = 'lightcoral'
                symbol = '✗'
                text_color = 'darkred'
            
            # Format property name
            formatted_name = prop_name.replace('_', ' ').title()
            
            # Create colored box with symbol
            bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor=text_color)
            ax.text(x, y, f'{symbol} {formatted_name}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color=text_color,
                   bbox=bbox, weight='bold')
        
        # Add overall status at the top
        overall_status = "PASSED" if validation_result.is_valid else "FAILED"
        overall_color = 'darkgreen' if validation_result.is_valid else 'darkred'
        overall_bg = 'lightgreen' if validation_result.is_valid else 'lightcoral'
        
        ax.text(0.5, 0.95, f'Mathematical Validation: {overall_status}', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=14, weight='bold', color=overall_color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor=overall_bg, alpha=0.8))
        
        # Add statistics if available
        if validation_result.statistics:
            stats = validation_result.statistics
            stats_text = "Key Statistics:\n"
            
            if 'max_harvest_coverage' in stats:
                stats_text += f"Harvest Coverage: {stats['max_harvest_coverage']:.1%}\n"
            if 'final_production_width' in stats:
                stats_text += f"Final Width: {stats['final_production_width']:.2e}\n"
            if 'width_reduction_ratio' in stats:
                stats_text += f"Width Reduction: {stats['width_reduction_ratio']:.1%}\n"
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                   verticalalignment='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        # Add errors and warnings count
        error_count = len(validation_result.errors)
        warning_count = len(validation_result.warnings)
        
        if error_count > 0 or warning_count > 0:
            issue_text = f"Issues: {error_count} errors, {warning_count} warnings"
            issue_color = 'lightcoral' if error_count > 0 else 'lightyellow'
            
            ax.text(0.98, 0.02, issue_text, transform=ax.transAxes,
                   verticalalignment='bottom', horizontalalignment='right', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=issue_color, alpha=0.8))
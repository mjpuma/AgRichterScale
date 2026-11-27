"""AgRichter scale and envelope plotting functionality."""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.config import Config
from .publication import PublicationFormatter


class AgRichterPlotter:
    """Creates AgRichter scale visualizations."""
    
    def __init__(self, config: Config):
        """Initialize the AgRichter plotter.
        
        Args:
            config: Configuration object with crop and output settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_scale_plot(self, 
                         scale_data: Optional[Dict[str, np.ndarray]] = None,
                         historical_events: Optional[pd.DataFrame] = None,
                         title: Optional[str] = None,
                         output_path: Optional[Path] = None,
                         dpi: int = 300,
                         figsize: Tuple[float, float] = (10, 8),
                         use_publication_style: bool = True,
                         save_formats: Optional[list] = None) -> plt.Figure:
        """Create AgRichter scale plot showing magnitude vs production loss.
        
        Args:
            scale_data: Dictionary with 'magnitudes' and 'production_kcal' arrays
            historical_events: DataFrame with historical event data
            title: Optional title for the plot
            output_path: Optional path to save the figure
            dpi: Resolution for output image
            figsize: Figure size in inches
            use_publication_style: Whether to apply publication formatting
            save_formats: List of formats to save (if output_path provided)
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating AgRichter scale visualization...")
        
        # Use publication formatter if requested
        formatter = PublicationFormatter(self.config) if use_publication_style else None
        
        if formatter:
            formatter.apply_publication_style()
            fig = formatter.create_figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Set up the scale if data provided
        if scale_data is not None:
            magnitudes = scale_data.get('magnitudes', np.array([]))
            production_kcal = scale_data.get('production_kcal', np.array([]))
            
            if len(magnitudes) > 0 and len(production_kcal) > 0:
                # Plot the scale line
                ax.plot(magnitudes, np.log10(production_kcal), 
                       'b-', linewidth=2, alpha=0.7, label='AgRichter Scale')
        
        # Plot historical events if provided
        if historical_events is not None and len(historical_events) > 0:
            self._plot_historical_events(ax, historical_events)
        
        # Set up axes
        self._setup_scale_axes(ax)
        
        # Set title
        if title is None:
            crop_type = self.config.crop_type.title()
            title = f'AgRichter Scale - {crop_type}'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend if we have data
        if scale_data is not None or (historical_events is not None and len(historical_events) > 0):
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            if formatter and save_formats:
                # Use publication formatter for multiple formats
                formatter.save_figure(fig, output_path, save_formats, dpi=dpi)
            else:
                # Standard single format save
                self.logger.info(f"Saving AgRichter scale plot to {output_path}")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
        
        # Restore original style if formatter was used
        if formatter:
            formatter.restore_original_style()
        
        self.logger.info("AgRichter scale visualization created successfully")
        return fig
    
    def _plot_historical_events(self, ax: plt.Axes, events_df: pd.DataFrame):
        """Plot historical events on the AgRichter scale.
        
        Args:
            ax: Matplotlib axes object
            events_df: DataFrame with event data including magnitude and production loss
        """
        from ..core.constants import EVENT_COLORS, EVENT_MARKERS
        
        # Get crop-specific colors
        crop_colors = EVENT_COLORS.get(self.config.crop_type, {})
        
        # Group events by color/severity
        color_groups = {}
        for _, event in events_df.iterrows():
            event_name = event.get('event_name', 'Unknown')
            color = crop_colors.get(event_name, 'gray')
            
            if color not in color_groups:
                color_groups[color] = {'events': [], 'magnitudes': [], 'losses': []}
            
            color_groups[color]['events'].append(event_name)
            color_groups[color]['magnitudes'].append(event.get('magnitude', 0))
            color_groups[color]['losses'].append(event.get('production_loss_kcal', 1))
        
        # Plot each color group
        for color, group_data in color_groups.items():
            if len(group_data['magnitudes']) > 0:
                marker = EVENT_MARKERS.get(color, 'o')
                
                # Convert losses to log10
                log_losses = np.log10(np.array(group_data['losses']))
                
                # Plot points
                ax.scatter(group_data['magnitudes'], log_losses,
                          c=color, marker=marker, s=60, alpha=0.8,
                          edgecolors='black', linewidths=0.5,
                          label=f'{color.title()} events')
                
                # Add event labels for significant events
                for i, (mag, loss, name) in enumerate(zip(group_data['magnitudes'], 
                                                         log_losses, 
                                                         group_data['events'])):
                    if mag > 2.0:  # Only label larger events
                        ax.annotate(name, (mag, loss), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7)
    
    def _setup_scale_axes(self, ax: plt.Axes):
        """Set up axes for AgRichter scale plot.
        
        Args:
            ax: Matplotlib axes object
        """
        # Set labels
        ax.set_xlabel('Magnitude (log₁₀ disrupted area km²)', fontsize=12)
        ax.set_ylabel('Production Loss (log₁₀ kcal)', fontsize=12)
        
        # Get crop-specific ranges from constants
        from ..core.constants import PRODUCTION_RANGES
        
        crop_type = self.config.crop_type
        if crop_type in PRODUCTION_RANGES:
            y_min, y_max = PRODUCTION_RANGES[crop_type]
            ax.set_ylim(y_min, y_max)
        else:
            # Default range
            ax.set_ylim(10, 16)
        
        # Set x-axis range (magnitude typically 0-6)
        ax.set_xlim(0, 6)
        
        # Add threshold lines
        self._add_threshold_lines(ax)
    
    def _add_threshold_lines(self, ax: plt.Axes):
        """Add threshold lines (T1-T4) to the plot.
        
        Args:
            ax: Matplotlib axes object
        """
        thresholds = self.config.get_thresholds()
        
        # Threshold colors
        threshold_colors = {
            'T1': 'green',
            'T2': 'yellow', 
            'T3': 'orange',
            'T4': 'red'
        }
        
        # Add horizontal lines for thresholds
        for level, threshold_kcal in thresholds.items():
            if threshold_kcal > 0:
                log_threshold = np.log10(threshold_kcal)
                color = threshold_colors.get(level, 'gray')
                
                ax.axhline(y=log_threshold, color=color, linestyle='--', 
                          alpha=0.7, linewidth=1.5, 
                          label=f'{level} ({threshold_kcal:.1e} kcal)')
    
    def create_richter_scale_data(self, 
                                 min_magnitude: float = 0.0,
                                 max_magnitude: float = 6.0,
                                 n_points: int = 100) -> Dict[str, np.ndarray]:
        """Create theoretical AgRichter scale data.
        
        Args:
            min_magnitude: Minimum magnitude value
            max_magnitude: Maximum magnitude value  
            n_points: Number of points to generate
            
        Returns:
            Dictionary with 'magnitudes' and 'production_kcal' arrays
        """
        from ..analysis.agrichter import AgRichterAnalyzer
        
        analyzer = AgRichterAnalyzer(self.config)
        
        # Generate magnitude range
        magnitudes = np.linspace(min_magnitude, max_magnitude, n_points)
        
        # Convert magnitudes to areas (10^magnitude km²)
        areas_km2 = 10 ** magnitudes
        
        # Use analyzer to create scale data
        scale_data = analyzer.create_richter_scale_data(min_magnitude, max_magnitude, n_points)
        
        return scale_data


class EnvelopePlotter:
    """Creates H-P envelope visualizations."""
    
    def __init__(self, config: Config):
        """Initialize the envelope plotter.
        
        Args:
            config: Configuration object with crop and output settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_envelope_plot(self,
                           envelope_data: Optional[Dict[str, np.ndarray]] = None,
                           historical_events: Optional[pd.DataFrame] = None,
                           title: Optional[str] = None,
                           output_path: Optional[Path] = None,
                           dpi: int = 300,
                           figsize: Tuple[float, float] = (12, 8),
                           use_publication_style: bool = True,
                           save_formats: Optional[list] = None) -> plt.Figure:
        """Create H-P envelope plot with filled area and threshold lines.
        
        Args:
            envelope_data: Dictionary with envelope boundary data
            historical_events: DataFrame with historical event data
            title: Optional title for the plot
            output_path: Optional path to save the figure
            dpi: Resolution for output image
            figsize: Figure size in inches
            use_publication_style: Whether to apply publication formatting
            save_formats: List of formats to save (if output_path provided)
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating H-P envelope visualization...")
        
        # Use publication formatter if requested
        formatter = PublicationFormatter(self.config) if use_publication_style else None
        
        if formatter:
            formatter.apply_publication_style()
            fig = formatter.create_figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot envelope if data provided
        if envelope_data is not None:
            self._plot_envelope_area(ax, envelope_data)
        
        # Add threshold lines (scaled to envelope if data provided)
        self._add_threshold_lines(ax, envelope_data)
        
        # Plot historical events if provided
        if historical_events is not None and len(historical_events) > 0:
            self._plot_events_on_envelope(ax, historical_events)
        
        # Set up axes
        self._setup_envelope_axes(ax)
        
        # Set title
        if title is None:
            crop_type = self.config.crop_type.title()
            title = f'H-P Envelope - {crop_type}'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            if formatter and save_formats:
                # Use publication formatter for multiple formats
                formatter.save_figure(fig, output_path, save_formats, dpi=dpi)
            else:
                # Standard single format save
                self.logger.info(f"Saving H-P envelope plot to {output_path}")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
        
        # Restore original style if formatter was used
        if formatter:
            formatter.restore_original_style()
        
        self.logger.info("H-P envelope visualization created successfully")
        return fig
    
    def _plot_envelope_area(self, ax: plt.Axes, envelope_data: Dict[str, np.ndarray]):
        """Plot the filled envelope area using MATLAB-exact fill algorithm.
        
        MATLAB algorithm:
        X=log10([DisturbMatrix_lower(1,1),DisturbMatrix_upper(:,1)', flipud(DisturbMatrix_lower(:,1))']);
        Y=log10(loss_factor*[DisturbMatrix_lower(1,2),DisturbMatrix_upper(:,2)', flipud(DisturbMatrix_lower(:,2))']);
        fill(X,Y,rgb('LightBlue'), 'FaceAlpha', 0.6,'EdgeColor','none');
        
        Args:
            ax: Matplotlib axes object
            envelope_data: Dictionary with envelope boundary data
        """
        # Extract boundary data
        disruption_areas = envelope_data.get('disruption_areas', np.array([]))
        lower_harvest = envelope_data.get('lower_bound_harvest', np.array([]))
        lower_production = envelope_data.get('lower_bound_production', np.array([]))
        upper_harvest = envelope_data.get('upper_bound_harvest', np.array([]))
        upper_production = envelope_data.get('upper_bound_production', np.array([]))
        
        if len(disruption_areas) == 0:
            self.logger.warning("No envelope data to plot")
            return
        
        # Handle NaN and Inf values properly
        lower_harvest = self._clean_array(lower_harvest)
        lower_production = self._clean_array(lower_production)
        upper_harvest = self._clean_array(upper_harvest)
        upper_production = self._clean_array(upper_production)
        
        # Filter out invalid values (NaN, Inf, zero, negative)
        valid_mask = (
            np.isfinite(lower_harvest) & np.isfinite(lower_production) &
            np.isfinite(upper_harvest) & np.isfinite(upper_production) &
            (lower_harvest > 0) & (lower_production > 0) &
            (upper_harvest > 0) & (upper_production > 0)
        )
        
        if not np.any(valid_mask):
            self.logger.warning("No valid envelope data points for plotting")
            return
        
        # Get valid data
        lower_harvest_valid = lower_harvest[valid_mask]
        lower_production_valid = lower_production[valid_mask]
        upper_harvest_valid = upper_harvest[valid_mask]
        upper_production_valid = upper_production[valid_mask]
        
        if len(lower_harvest_valid) == 0:
            self.logger.warning("No valid envelope points after filtering")
            return
        
        # MATLAB-exact fill algorithm using concatenated boundary arrays
        # X=log10([DisturbMatrix_lower(1,1),DisturbMatrix_upper(:,1)', flipud(DisturbMatrix_lower(:,1))']);
        # Y=log10(loss_factor*[DisturbMatrix_lower(1,2),DisturbMatrix_upper(:,2)', flipud(DisturbMatrix_lower(:,2))']);
        
        # Create closed polygon coordinates
        X_polygon = np.concatenate([
            [lower_harvest_valid[0]],           # First point of lower bound
            upper_harvest_valid,                # All upper bound points (left to right)
            np.flipud(lower_harvest_valid)      # All lower bound points (right to left)
        ])
        
        Y_polygon = np.concatenate([
            [lower_production_valid[0]],        # First point of lower bound
            upper_production_valid,             # All upper bound points (left to right)
            np.flipud(lower_production_valid)   # All lower bound points (right to left)
        ])
        
        # Convert to log10 for plotting (MATLAB algorithm)
        X_polygon_log = np.log10(X_polygon)
        Y_polygon_log = np.log10(Y_polygon)
        
        # Handle any remaining NaN/Inf values in log conversion
        valid_log_mask = np.isfinite(X_polygon_log) & np.isfinite(Y_polygon_log)
        if not np.any(valid_log_mask):
            self.logger.warning("No valid points after log10 conversion")
            return
        
        X_polygon_log = X_polygon_log[valid_log_mask]
        Y_polygon_log = Y_polygon_log[valid_log_mask]
        
        # Create filled envelope area with MATLAB-exact styling
        # fill(X,Y,rgb('LightBlue'), 'FaceAlpha', 0.6,'EdgeColor','none');
        light_blue_color = [0.8, 0.9, 1.0]  # Light blue RGB values
        ax.fill(X_polygon_log, Y_polygon_log, 
               color=light_blue_color, alpha=0.6, edgecolor='none',
               label='H-P Envelope', zorder=1)
        
        # Plot boundary lines for clarity
        log_lower_production = np.log10(lower_production_valid)
        log_upper_production = np.log10(upper_production_valid)
        log_lower_harvest = np.log10(lower_harvest_valid)
        log_upper_harvest = np.log10(upper_harvest_valid)
        
        ax.plot(log_lower_harvest, log_lower_production, 'b-', linewidth=2, 
               alpha=0.8, label='Lower Bound (Least Productive)', zorder=2)
        ax.plot(log_upper_harvest, log_upper_production, 'r-', linewidth=2,
               alpha=0.8, label='Upper Bound (Most Productive)', zorder=2)
        
        self.logger.info(f"Plotted MATLAB-exact envelope with {len(lower_harvest_valid)} boundary points")
    
    def _clean_array(self, arr: np.ndarray) -> np.ndarray:
        """Clean array by handling NaN and Inf values.
        
        Args:
            arr: Input array
            
        Returns:
            Cleaned array with NaN and Inf values handled
        """
        # Replace Inf and -Inf with NaN
        arr_clean = np.where(np.isinf(arr), np.nan, arr)
        
        # For production values, replace NaN with small positive value to avoid log(0)
        # For harvest values, replace NaN with 0
        if np.any(np.isnan(arr_clean)):
            # Assume this is production data if values are large, harvest data if small
            if np.nanmax(arr_clean) > 1e6:  # Production data (kcal)
                arr_clean = np.where(np.isnan(arr_clean), 1.0, arr_clean)
            else:  # Harvest data (km²)
                arr_clean = np.where(np.isnan(arr_clean), 0.0, arr_clean)
        
        return arr_clean
    
    def _add_threshold_lines(self, ax: plt.Axes, envelope_data: Optional[Dict[str, np.ndarray]] = None):
        """Add threshold lines (T1-T4) to the envelope plot.
        
        Args:
            ax: Matplotlib axes object
            envelope_data: Optional envelope data to scale thresholds appropriately
        """
        thresholds = self.config.get_thresholds()
        
        # Scale thresholds if envelope data is provided
        if envelope_data is not None:
            thresholds = self._scale_thresholds_to_envelope(thresholds, envelope_data)
        
        # Threshold colors
        threshold_colors = {
            'T1': 'green',
            'T2': 'yellow',
            'T3': 'orange', 
            'T4': 'red'
        }
        
        # Add horizontal lines for thresholds
        for level, threshold_kcal in thresholds.items():
            if threshold_kcal > 0:
                log_threshold = np.log10(threshold_kcal)
                color = threshold_colors.get(level, 'gray')
                
                ax.axhline(y=log_threshold, color=color, linestyle='--',
                          alpha=0.7, linewidth=2,
                          label=f'{level} Threshold', zorder=3)
    
    def _scale_thresholds_to_envelope(self, thresholds: Dict[str, float], 
                                    envelope_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Scale global thresholds to match the envelope data extent.
        
        The AgRichter thresholds are designed for global production scales,
        but envelope calculations often use data samples. This method scales
        thresholds proportionally to the envelope's maximum production.
        
        Args:
            thresholds: Original global thresholds
            envelope_data: Envelope data with production bounds
            
        Returns:
            Scaled thresholds appropriate for the envelope data extent
        """
        # Get maximum production from envelope
        upper_production = envelope_data.get('upper_bound_production', np.array([]))
        
        if len(upper_production) == 0:
            self.logger.warning("No envelope production data for threshold scaling")
            return thresholds
        
        max_envelope_production = upper_production.max()
        
        # Global production reference (from constants analysis)
        # This represents the total global grain production: ~1.04e16 kcal
        global_production_reference = 1.04e16
        
        # Calculate scaling factor based on envelope extent
        scaling_factor = max_envelope_production / global_production_reference
        
        # Apply scaling to thresholds
        scaled_thresholds = {}
        for level, threshold_kcal in thresholds.items():
            if threshold_kcal > 0:
                scaled_threshold = threshold_kcal * scaling_factor
                scaled_thresholds[level] = scaled_threshold
                
                self.logger.debug(f"Scaled {level}: {threshold_kcal:.2e} → {scaled_threshold:.2e} kcal "
                                f"(factor: {scaling_factor:.3f})")
            else:
                scaled_thresholds[level] = threshold_kcal
        
        self.logger.info(f"Scaled thresholds by factor {scaling_factor:.3f} to match envelope extent")
        
        return scaled_thresholds
    
    def _plot_events_on_envelope(self, ax: plt.Axes, events_df: pd.DataFrame):
        """Plot historical events on the H-P envelope.
        
        Args:
            ax: Matplotlib axes object
            events_df: DataFrame with event data
        """
        from ..core.constants import EVENT_COLORS, EVENT_MARKERS
        
        # Get crop-specific colors
        crop_colors = EVENT_COLORS.get(self.config.crop_type, {})
        
        # Plot each event
        for _, event in events_df.iterrows():
            event_name = event.get('event_name', 'Unknown')
            harvest_area = event.get('harvest_area_km2', 0)
            production_loss = event.get('production_loss_kcal', 1)
            
            if harvest_area > 0 and production_loss > 0:
                color = crop_colors.get(event_name, 'gray')
                marker = EVENT_MARKERS.get(color, 'o')
                
                # Convert to log10
                log_production_loss = np.log10(production_loss)
                
                # Plot event point
                ax.scatter(harvest_area, log_production_loss,
                          c=color, marker=marker, s=80, alpha=0.9,
                          edgecolors='black', linewidths=1,
                          zorder=4)
                
                # Add event label
                ax.annotate(event_name, (harvest_area, log_production_loss),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8, zorder=5)
    
    def _setup_envelope_axes(self, ax: plt.Axes):
        """Set up axes for H-P envelope plot.
        
        Args:
            ax: Matplotlib axes object
        """
        # Set labels
        ax.set_xlabel('Harvest Area (km²)', fontsize=12)
        ax.set_ylabel('Production (log₁₀ kcal)', fontsize=12)
        
        # Set axis scales
        ax.set_xscale('log')  # Log scale for harvest area
        
        # Get crop-specific ranges
        from ..core.constants import PRODUCTION_RANGES
        
        crop_type = self.config.crop_type
        if crop_type in PRODUCTION_RANGES:
            y_min, y_max = PRODUCTION_RANGES[crop_type]
            ax.set_ylim(y_min, y_max)
        else:
            # Default range
            ax.set_ylim(10, 16)
        
        # Set reasonable x-axis limits (will be adjusted based on data)
        ax.set_xlim(1, 1e7)  # 1 km² to 10M km²
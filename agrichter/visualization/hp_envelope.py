"""MATLAB-exact H-P Envelope visualization with convergence analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class HPEnvelopeVisualizer:
    """Visualizer for MATLAB-exact H-P Envelope plots with convergence analysis."""
    
    def __init__(self, config):
        """
        Initialize H-P Envelope visualizer.
        
        Args:
            config: Configuration object with crop parameters and thresholds
        """
        self.config = config
        self.crop_type = config.crop_type
        
        # Get IPC colors for threshold lines
        self.ipc_colors = config.get_ipc_colors()
        
        # Set up figure parameters
        self.figure_params = {
            'figsize': (10, 8),
            'dpi': 300,
            'facecolor': 'white'
        }
        
        # Initialize convergence analysis components
        self._convergence_validator = None
        self._envelope_diagnostics = None
        
        # Define unified event type colors and markers (Publication Consistent)
        self.event_type_styles = {
            'Climate & Weather': {'color': '#1f77b4', 'marker': 'o', 'label': 'Climate & Weather'},  # Blue
            'Conflict & Policy': {'color': '#d62728', 'marker': 's', 'label': 'Conflict & Policy'},  # Red
            'Pest & Disease':    {'color': '#2ca02c', 'marker': 'D', 'label': 'Pest & Disease'},     # Green
            'Geophysical':       {'color': '#ff7f0e', 'marker': '^', 'label': 'Geophysical'},        # Orange
            'Compound/Other':    {'color': '#7f7f7f', 'marker': '*', 'label': 'Compound/Other'}      # Gray
        }

        # Override table for specific event labels (Nature Publication Refinement)
        # Format: {crop_type: {event_id: (mag_label, production_kcal_label)}}
        # These coordinates define the STARTING position for adjust_text.
        self.label_overrides = {
            'allgrain': {
                'PotatoFamine': (0.3, 10**14.7),
                'DustBowl': (0.5, 10**14.3),
                'NorthKorea1990s': (1.8, 10**12.5),
                'Bangladesh': (0.5, 10**13.0),
                'Syria': (5.0, 10**12.8),
                'SahelDrought2010': (6.5, 10**12.7),
                'SovietFamine1921': (4.5, 10**14.8),
                'ChineseFamine1960': (5.5, 10**15.0),
                'Drought18761878': (6.8, 10**15.3),
                'ENSO2015_2016': (6.8, 10**15.3)
            },
            'wheat': {
                'GreatFamine': (0.5, 10**14.3),
                'DustBowl': (2.8, 10**14.0),
                'Ethiopia': (1.8, 10**13.3),
                'SahelDrought2010': (3.2, 10**12.8),
                'Syria': (4.2, 10**13.7),
                'MillenniumDrought': (4.5, 10**14.3),
                'SovietFamine1921': (5.8, 10**14.3),
                'Drought18761878': (6.5, 10**14.7)
            },
            'maize': {
                'GreatFamine': (2.0, 10**13.3),
                'SovietFamine1921': (4.2, 10**13.0),
                'SahelDrought2010': (5.0, 10**14.5),
                'DustBowl': (4.5, 10**14.3),
                'ChineseFamine1960': (5.8, 10**15.0),
                'ENSO2015_2016': (5.5, 10**14.7)
            },
            'rice': {
                'Solomon': (1.2, 10**12.0),
                'EastTimor': (2.5, 10**11.7),
                'Bangladesh': (3.2, 10**13.7),
                'Liberia': (4.0, 10**12.7),
                'SierraLeone': (5.2, 10**13.3),
                'ChineseFamine1960': (5.5, 10**14.5),
                'ENSO2015_2016': (6.2, 10**15.0),
                'Drought18761878': (6.5, 10**15.3),
                'Laos': (6.0, 10**13.0)
            }
        }
    
    def _consolidate_event_type(self, event_type_raw: str) -> str:
        """Consolidate detailed event types into publication categories."""
        if pd.isna(event_type_raw):
            return 'Compound/Other'
            
        et = str(event_type_raw).lower()
        
        if any(x in et for x in ['climate', 'drought', 'flood', 'weather', 'rain', 'dry', 'wet']):
            return 'Climate & Weather'
        elif any(x in et for x in ['conflict', 'war', 'policy', 'political', 'civil', 'unrest']):
            return 'Conflict & Policy'
        elif any(x in et for x in ['disease', 'pest', 'locust', 'blight', 'fungus']):
            return 'Pest & Disease'
        elif any(x in et for x in ['volcanic', 'earthquake', 'seismic', 'natural disaster', 'geophysical']):
            return 'Geophysical'
        else:
            return 'Compound/Other'

    def create_hp_envelope_plot(self, envelope_data: Dict, events_data: pd.DataFrame,
                               save_path: Optional[Union[str, Path]] = None,
                               show_convergence: bool = True,
                               show_events: bool = True,
                               show_labels: bool = True,
                               show_axis_labels: bool = True,
                               show_legend: bool = True,
                               is_combined: bool = False,
                               total_production: Optional[float] = None,
                               total_harvest: Optional[float] = None,
                               ax: Optional[plt.Axes] = None,
                               title: Optional[str] = None) -> plt.Figure:
        """
        Create MATLAB-exact H-P Envelope visualization with convergence analysis.
        
        Args:
            envelope_data: Dictionary with envelope bounds
            events_data: DataFrame with event data
            save_path: Optional path to save the figure
            show_convergence: Whether to highlight convergence point
            show_events: Whether to plot historical events
            show_labels: Whether to show event labels
            show_axis_labels: Whether to show individual axis labels
            show_legend: Whether to show the legend
            is_combined: If True, adjust for 4-panel combined figure
            total_production: Total production for convergence validation
            total_harvest: Total harvest area for convergence validation
            ax: Optional matplotlib axes to plot on
            title: Optional custom title for the plot
        """
        # Convert harvest area from hectares to km² if needed
        events_data = self._prepare_events_data(events_data)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 10) if not is_combined else (10, 8), 
                                  dpi=300,
                                  facecolor='white')
        else:
            fig = ax.get_figure()
        
        # Convert harvest area to magnitude: M_D = log10(A_H)
        # Check if envelope has separate X-axes (MATLAB-exact approach)
        if 'lower_bound_harvest' in envelope_data and 'upper_bound_harvest' in envelope_data:
            # Use separate X-axes for each bound
            lower_harvest = np.array(envelope_data['lower_bound_harvest'])
            upper_harvest = np.array(envelope_data['upper_bound_harvest'])
            lower_bound = np.array(envelope_data['lower_bound_production'])
            upper_bound = np.array(envelope_data['upper_bound_production'])
            
            # For plotting, use lower bound X-axis as primary
            disrupted_areas = lower_harvest
            magnitude = np.log10(disrupted_areas)
            
            # Store separate X-axes for validation
            self._lower_harvest = lower_harvest
            self._upper_harvest = upper_harvest
        else:
            # Legacy: single X-axis for both bounds
            disrupted_areas = np.array(envelope_data['disrupted_areas'])
            upper_bound = np.array(envelope_data['upper_bound'])
            lower_bound = np.array(envelope_data['lower_bound'])
            magnitude = np.log10(disrupted_areas)
            self._lower_harvest = disrupted_areas
            self._upper_harvest = disrupted_areas
        
        # Set axis limits - extended to capture smaller events
        ax.set_xlim([1, 7])  # Extended lower bound from 2 to 1
        ax.set_ylim([1e8, 1.62e16])  # Extended lower bound from 1e10 to 1e8
        
        # Plot gray envelope fill between upper and lower bounds
        self._plot_envelope_fill(ax, magnitude, upper_bound, lower_bound)
        
        # Plot upper bound (black) and lower bound (blue) boundary lines
        self._plot_boundary_lines(ax, magnitude, upper_bound, lower_bound)
        
        # Plot threshold lines
        self._plot_agriPhase_thresholds(ax)
        
        # Plot events if requested
        if show_events:
            # CRITICAL: Validate events are within envelope before plotting
            events_data = self._validate_events_within_envelope(
                events_data, magnitude, upper_bound, lower_bound
            )
            
            # Plot historical events as markers with labels
            self._plot_historical_events(ax, events_data, show_labels=show_labels)
        
        # Always highlight convergence point as it is part of the envelope definition
        if show_convergence and total_production is not None and total_harvest is not None:
             self._highlight_convergence_point(ax, envelope_data, total_production, total_harvest)
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Set even powers only for y-axis (10^8, 10^10, 10^12, 10^14, 10^16)
        ax.yaxis.set_major_locator(LogLocator(base=100.0, numticks=5))
        
        # Set axis labels with proper subscript formatting
        if show_axis_labels:
            ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', 
                         fontsize=14, fontweight='bold')
            ax.set_ylabel('Production Loss (kcal)', 
                         fontsize=14, fontweight='bold')
        
        # Tick parameters
        ax.tick_params(axis='both', labelsize=12)
        
        # Set title
        if title:
            plot_title = title
        else:
            crop_name = 'All Grains' if self.crop_type == 'allgrain' else self.crop_type.title()
            plot_title = crop_name
        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            
            # Separate envelope/bounds from event types
            envelope_items = [(h, l) for h, l in zip(handles, labels) 
                             if 'Envelope' in l or 'Bound' in l or 'Convergence' in l]
            event_items = [(h, l) for h, l in zip(handles, labels) 
                          if l not in [item[1] for item in envelope_items]]
            
            # Combine in logical order (envelope first, then event types)
            ordered_handles = [h for h, l in envelope_items] + [h for h, l in event_items]
            ordered_labels = [l for h, l in envelope_items] + [l for h, l in event_items]
            
            ax.legend(ordered_handles, ordered_labels, loc='lower right', 
                      fontsize=10, markerscale=1.5, framealpha=0.9)
        
        # Adjust layout only if we created the figure
        if save_path or ax is None:
            plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            self._save_figure(fig, save_path)
        
        logger.info(f"Created H-P Envelope plot for {self.crop_type}")
        return fig
    
    def _plot_envelope_fill(self, ax: plt.Axes, magnitude: np.ndarray, 
                           upper_bound: np.ndarray, lower_bound: np.ndarray) -> None:
        """Plot gray envelope fill between upper and lower bounds using separate X-axes."""
        # Use separate X-axes for upper and lower bounds (MATLAB-exact approach)
        lower_magnitude = np.log10(self._lower_harvest)
        upper_magnitude = np.log10(self._upper_harvest)
        
        # Create polygon for fill (MATLAB approach)
        # Concatenate: lower bound forward + upper bound backward
        X_env = np.concatenate([lower_magnitude, np.flip(upper_magnitude)])
        Y_env = np.concatenate([lower_bound, np.flip(upper_bound)])
        
        # Fill the polygon
        # REMOVED label to reduce legend clutter
        ax.fill(X_env, Y_env, color='gray', alpha=0.3) #, label='H-P Envelope')
    
    def _plot_boundary_lines(self, ax: plt.Axes, magnitude: np.ndarray,
                            upper_bound: np.ndarray, lower_bound: np.ndarray) -> None:
        """Plot upper bound (black) and lower bound (blue) boundary lines using separate X-axes."""
        # Use separate X-axes for each bound (MATLAB-exact approach)
        lower_magnitude = np.log10(self._lower_harvest)
        upper_magnitude = np.log10(self._upper_harvest)
        
        # Plot upper bound (black line) with its own X-axis
        ax.plot(upper_magnitude, upper_bound, 'k-', linewidth=2, 
               label='Upper Bound', alpha=0.8)
        
        # Plot lower bound (blue line) with its own X-axis
        ax.plot(lower_magnitude, lower_bound, 'b-', linewidth=2, 
               label='Lower Bound', alpha=0.8)
    
    def _format_kcal(self, value: float) -> str:
        """Format kcal values with appropriate units (P, T, G)."""
        if value >= 1e15:
            return f"{value/1e15:.1f}P kcal"
        elif value >= 1e12:
            return f"{value/1e12:.1f}T kcal"
        elif value >= 1e9:
            return f"{value/1e9:.1f}G kcal"
        else:
            return f"{value:.1e} kcal"

    def _plot_agriPhase_thresholds(self, ax: plt.Axes) -> None:
        """Plot AgriPhase/Supply threshold lines as horizontal dashed lines."""
        # Get thresholds from config (values in kcal)
        thresholds = self.config.get_thresholds()
        colors = self.config.get_ipc_colors()
        
        # Get current y-limits to ensure we only plot visible thresholds
        ylim = ax.get_ylim()
        
        # Plot horizontal threshold lines
        for name, value_kcal in thresholds.items():
            # SKIP "Total Stocks" for publication clarity (User requested removal)
            if name == 'Total Stocks':
                continue
            
            # Determine color
            color = colors.get(name, '#000000')
            
            # Format label
            formatted_value = self._format_kcal(value_kcal)
            label = f"{name} (~{formatted_value})"
            
            # Only plot if within reasonable range (or just let matplotlib handle clipping)
            # Adding a label at the right edge is helpful
            ax.axhline(y=value_kcal, color=color, linestyle='--', 
                      linewidth=2, alpha=0.8, label=label)
            
            # Add text annotation inside the panel if within view
            if ylim[0] <= value_kcal <= ylim[1]:
                xlim = ax.get_xlim()
                # Position labels nearly touching the right edge, aligned right
                x_pos = xlim[1] - 0.01 * (xlim[1] - xlim[0])
                
                # Vertical alignment: 3 Months above, 1 Month below with extra spacing
                y_pos = value_kcal
                va = 'center'
                if '3 Months' in name:
                    va = 'bottom' 
                    y_pos *= 1.15 # Move 15% up from the line
                elif '1 Month' in name:
                    va = 'top'    
                    y_pos /= 1.15 # Move 15% down from the line
                
                ax.text(x_pos, y_pos, f"{name}", 
                       color=color, fontsize=8, va=va, ha='right', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6, edgecolor='none'))
    
    def _prepare_events_data(self, events_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare events data for plotting.
        
        Converts harvest area from hectares to km² if needed and filters out
        events with zero or invalid harvest area.
        
        Args:
            events_data: DataFrame with event data
        
        Returns:
            Prepared DataFrame with harvest_area_km2 column
        """
        events_data = events_data.copy()
        
        # Convert harvest area from hectares to km² if needed
        if 'harvest_area_loss_ha' in events_data.columns and 'harvest_area_km2' not in events_data.columns:
            # 1 hectare = 0.01 km²
            events_data['harvest_area_km2'] = events_data['harvest_area_loss_ha'] * 0.01
            logger.info("Converted harvest area from hectares to km²")
        
        # Filter out events with zero or invalid harvest area
        valid_mask = (events_data['harvest_area_km2'] > 0) & np.isfinite(events_data['harvest_area_km2'])
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            logger.warning(f"Filtered out {invalid_count} events with zero or invalid harvest area")
            events_data = events_data[valid_mask]
        
        # Filter out events with zero or invalid production loss
        valid_mask = (events_data['production_loss_kcal'] > 0) & np.isfinite(events_data['production_loss_kcal'])
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            logger.warning(f"Filtered out {invalid_count} events with zero or invalid production loss")
            events_data = events_data[valid_mask]
        
        return events_data
    
    def _classify_event_severity(self, production_loss: float) -> Tuple[int, str, str, str]:
        """
        Classify event severity based on AgriPhase thresholds.
        
        Args:
            production_loss: Production loss in kcal
        
        Returns:
            Tuple of (phase_number, phase_name, color, marker)
        """
        thresholds = self.config.get_thresholds()
        ipc_colors = self.ipc_colors
        
        # Define marker shapes for each phase
        phase_markers = {
            1: 'o',  # Circle - Phase 1 (Minimal)
            2: 's',  # Square - Phase 2 (Stressed)
            3: '^',  # Triangle up - Phase 3 (Crisis)
            4: 'D',  # Diamond - Phase 4 (Emergency)
            5: '*',  # Star - Phase 5 (Famine)
        }
        
        # Classify based on thresholds
        # T1 = Phase 2, T2 = Phase 3, T3 = Phase 4, T4 = Phase 5
        if production_loss < thresholds.get('T1', float('inf')):
            phase = 1
            phase_name = 'Phase 1 (Minimal)'
        elif production_loss < thresholds.get('T2', float('inf')):
            phase = 2
            phase_name = 'Phase 2 (Stressed)'
        elif production_loss < thresholds.get('T3', float('inf')):
            phase = 3
            phase_name = 'Phase 3 (Crisis)'
        elif production_loss < thresholds.get('T4', float('inf')):
            phase = 4
            phase_name = 'Phase 4 (Emergency)'
        else:
            phase = 5
            phase_name = 'Phase 5 (Famine)'
        
        color = ipc_colors.get(phase, '#808080')
        marker = phase_markers.get(phase, 'o')
        
        return phase, phase_name, color, marker
    
    def _validate_events_within_envelope(self, events_data: pd.DataFrame,
                                         magnitude: np.ndarray,
                                         upper_bound: np.ndarray,
                                         lower_bound: np.ndarray) -> pd.DataFrame:
        """
        Validate that all events lie within the H-P envelope.
        
        By construction, all real events MUST be within the envelope bounds.
        If events are outside, it indicates a bug in either:
        - Envelope calculation
        - Event calculation
        - Units mismatch
        
        Args:
            events_data: DataFrame with event data
            magnitude: Envelope magnitude values (log10 of harvest area)
            upper_bound: Envelope upper bound production values
            lower_bound: Envelope lower bound production values
        
        Returns:
            Validated events DataFrame
        """
        if events_data.empty:
            return events_data
        
        events_data = events_data.copy()
        events_data['magnitude'] = np.log10(events_data['harvest_area_km2'])
        
        # CRITICAL FIX: Use separate X-axes for upper and lower bounds
        # The MATLAB-exact envelope has different harvest area sequences for each bound
        
        # Check each event
        events_outside = []
        for idx, event in events_data.iterrows():
            event_harvest = event['harvest_area_km2']
            event_prod = event['production_loss_kcal']
            event_mag = event['magnitude']
            
            # Interpolate each bound using its own X-axis
            # Lower bound: interpolate using lower_harvest X-axis
            lower = np.interp(event_harvest, self._lower_harvest, lower_bound,
                            left=lower_bound[0], right=lower_bound[-1])
            # Upper bound: interpolate using upper_harvest X-axis
            upper = np.interp(event_harvest, self._upper_harvest, upper_bound,
                            left=upper_bound[0], right=upper_bound[-1])
            
            # Diagnostic logging for first few events
            if len(events_outside) < 3:
                logger.info(f"Event {event['event_name']}: A_H={event_harvest:.2f} km², P={event_prod:.2e} kcal")
                logger.info(f"  Envelope bounds (interpolated): lower={lower:.2e}, upper={upper:.2e} kcal")
            
            # Check if event is within bounds (with tolerance for downsampling artifacts)
            tolerance = 0.05  # 5% tolerance due to downsampling
            if not (lower * (1 - tolerance) <= event_prod <= upper * (1 + tolerance)):
                events_outside.append({
                    'name': event['event_name'],
                    'magnitude': event_mag,
                    'production': event_prod,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'below_lower': event_prod < lower,
                    'above_upper': event_prod > upper
                })
        
        # Report validation results
        if events_outside:
            logger.error(f"CRITICAL: {len(events_outside)} events are OUTSIDE the H-P envelope!")
            for event in events_outside:
                if event['above_upper']:
                    logger.error(f"  {event['name']}: ABOVE envelope")
                    logger.error(f"    Magnitude: {event['magnitude']:.2f}")
                    logger.error(f"    Production: {event['production']:.2e} kcal")
                    logger.error(f"    Upper bound: {event['upper_bound']:.2e} kcal")
                    logger.error(f"    Excess: {(event['production']/event['upper_bound'] - 1)*100:.1f}%")
                else:
                    logger.error(f"  {event['name']}: BELOW envelope")
                    logger.error(f"    Magnitude: {event['magnitude']:.2f}")
                    logger.error(f"    Production: {event['production']:.2e} kcal")
                    logger.error(f"    Lower bound: {event['lower_bound']:.2e} kcal")
            
            logger.error("This indicates a bug in envelope or event calculation!")
        else:
            logger.info(f"✓ All {len(events_data)} events are within the H-P envelope")
        
        return events_data
    
    def _plot_historical_events(self, ax: plt.Axes, events_data: pd.DataFrame, show_labels: bool = True) -> None:
        """Plot historical events with severity-based colors and markers."""
        if events_data.empty:
            logger.warning("No events data provided for plotting")
            return
        
        # Priority events for labeling (same as Fig 1)
        priority_map = {
            'allgrain': ['DustBowl', 'ChineseFamine1960', 'Drought18761878', 'ENSO2015_2016', 
                         'SovietFamine1921', 'Ethiopia', 'Bangladesh', 'SahelDrought2010',
                         'MillenniumDrought', 'Syria', 'PotatoFamine', 'NorthKorea1990s'],
            'wheat': ['DustBowl', 'Drought18761878', 'GreatFamine', 'SahelDrought2010',
                      'SovietFamine1921', 'Ethiopia', 'Syria', 'MillenniumDrought'],
            'maize': ['DustBowl', 'ChineseFamine1960', 'GreatFamine', 'ENSO2015_2016',
                      'Haiti', 'SovietFamine1921', 'Ethiopia', 'SahelDrought2010'],
            'rice': ['Drought18761878', 'ChineseFamine1960', 'Bangladesh', 'ENSO2015_2016',
                     'Solomon', 'EastTimor', 'SierraLeone', 'Liberia', 'Laos']
        }
        priority_ids = priority_map.get(self.crop_type, [])

        # Calculate magnitude for events
        events_data = events_data.copy()
        events_data['magnitude'] = np.log10(events_data['harvest_area_km2'])
        
        # Consolidate event types
        if 'event_type' in events_data.columns:
            events_data['consolidated_type'] = events_data['event_type'].apply(self._consolidate_event_type)
        else:
            events_data['consolidated_type'] = 'Unknown'
        
        # Plot events grouped by consolidated event type for proper legend
        for event_type, style in self.event_type_styles.items():
            type_data = events_data[events_data['consolidated_type'] == event_type]
            if not type_data.empty:
                # Plot this event type group
                ax.scatter(type_data['magnitude'], type_data['production_loss_kcal'],
                               c=style['color'], s=70, alpha=0.8, edgecolors='black', linewidth=1,
                               marker=style['marker'], label=style['label'], zorder=5)
        
        # Plot unknown types if any
        unknown_mask = ~events_data['consolidated_type'].isin(self.event_type_styles.keys())
        unknown_data = events_data[unknown_mask]
        if not unknown_data.empty:
             ax.scatter(unknown_data['magnitude'], unknown_data['production_loss_kcal'],
                       c='gray', s=70, alpha=0.8, edgecolors='black', linewidth=1,
                       marker='o', label='Other', zorder=5)
        
        # Handle labels for priority events only if requested
        if show_labels:
            try:
                from adjustText import adjust_text
                texts = []
                targets_x = []
                targets_y = []
                for idx, row in events_data.iterrows():
                    event_id = row['event_name']
                    if event_id in priority_ids:
                        if event_id == 'ChineseFamine1960':
                            display_name = "Great Chinese Famine" if self.crop_type == 'allgrain' else "China 1959"
                        elif event_id == 'GreatFamine' and self.crop_type == 'maize':
                            display_name = "Great European\nFamine 1315"
                        else:
                            display_name = self.config.get_event_label(event_id)
                        
                        if "El Niño 2015" in display_name:
                            display_name = "El Niño 2015"

                        # Apply Directional Bias based on Magnitude
                        mag = row['magnitude']
                        
                        # Apply manual overrides if present (Nature Refinement)
                        overrides = self.label_overrides.get(self.crop_type, {})
                        if event_id in overrides:
                            text_x, text_y = overrides[event_id]
                        else:
                            # Small M -> UL, Large M -> LR
                            if mag < 3.5:
                                x_off, y_off = -0.2, 0.2
                            elif mag > 5.5:
                                x_off, y_off = 0.2, -0.2
                            else:
                                x_off, y_off = 0, 0.1
                            text_x, text_y = mag + x_off, row['production_loss_kcal'] * (10**y_off)

                        # Determine alignment based on displacement to point
                        ha = 'left' if text_x > row['magnitude'] else 'right'
                        va = 'bottom' if text_y > row['production_loss_kcal'] else 'top'

                        ann = ax.annotate(display_name,
                                         xy=(row['magnitude'], row['production_loss_kcal']),
                                         xytext=(text_x, text_y),
                                         fontsize=9, color='#2C5F8D',
                                         ha=ha, va=va, fontweight='normal',
                                         arrowprops=dict(arrowstyle='->', color='#2C5F8D', lw=1.0, alpha=0.8,
                                                       shrinkA=0, shrinkB=3))
                        texts.append(ann)
                        targets_x.append(row['magnitude'])
                        targets_y.append(row['production_loss_kcal'])
                
                if texts:
                    # Use targets_x and targets_y to ensure labels avoid data points.
                    # Annotations automatically update arrows when text is moved.
                    adjust_text(texts, x=targets_x, y=targets_y, ax=ax,
                               expand_points=(1.5, 1.5),
                               expand_text=(1.2, 1.2),
                               force_points=(0.2, 0.4),
                               force_text=(0.2, 0.4))
            except ImportError:
                # Fallback
                for idx, row in events_data.iterrows():
                    event_id = row['event_name']
                    if event_id in priority_ids:
                        display_name = self.config.get_event_label(event_id)
                        ax.annotate(display_name, 
                                   xy=(row['magnitude'], row['production_loss_kcal']),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=9, color='#2C5F8D', fontweight='normal')
    
    def _highlight_convergence_point(self, ax: plt.Axes, envelope_data: Dict,
                                   total_production: float, total_harvest: float) -> None:
        """
        Highlight the convergence point on the envelope plot.
        
        Args:
            ax: Matplotlib axes to plot on
            envelope_data: Envelope data dictionary
            total_production: Expected total production at convergence
            total_harvest: Expected total harvest area at convergence
        """
        # Get envelope bounds
        if 'lower_bound_harvest' in envelope_data and 'upper_bound_harvest' in envelope_data:
            lower_harvest = np.array(envelope_data['lower_bound_harvest'])
            upper_harvest = np.array(envelope_data['upper_bound_harvest'])
            lower_production = np.array(envelope_data['lower_bound_production'])
            upper_production = np.array(envelope_data['upper_bound_production'])
        else:
            # Legacy format
            disrupted_areas = np.array(envelope_data['disrupted_areas'])
            lower_harvest = upper_harvest = disrupted_areas
            lower_production = np.array(envelope_data['lower_bound'])
            upper_production = np.array(envelope_data['upper_bound'])
        
        # Convert to log scale for plotting
        total_harvest_log = np.log10(total_harvest)
        
        # Mark expected convergence point
        ax.plot(total_harvest_log, total_production, 'go', markersize=10, 
               markeredgecolor='darkgreen', markeredgewidth=1.5,
               label='Convergence', zorder=10)
        
        # REMOVED markers and labels for actual endpoints for publication clarity
        # These are usually very close to the convergence point anyway.
    
    def _add_convergence_diagnostics(self, ax: plt.Axes, envelope_data: Dict,
                                   total_production: float, total_harvest: float) -> None:
        """
        Add convergence diagnostic information to the plot.
        
        Args:
            ax: Matplotlib axes to plot on
            envelope_data: Envelope data dictionary
            total_production: Expected total production at convergence
            total_harvest: Expected total harvest area at convergence
        """
        try:
            # Import convergence validator if not already done
            if self._convergence_validator is None:
                from ..analysis.convergence_validator import ConvergenceValidator
                self._convergence_validator = ConvergenceValidator()
            
            # Perform convergence validation
            validation_result = self._convergence_validator.validate_mathematical_properties(
                envelope_data, total_production, total_harvest
            )
            
            # Create validation status text
            status_text = "Mathematical Validation:\n"
            properties = validation_result.properties
            
            # Use symbols for visual clarity
            for prop_name, prop_value in properties.items():
                symbol = "✓" if prop_value else "✗"
                formatted_name = prop_name.replace('_', ' ').title()
                status_text += f"{symbol} {formatted_name}\n"
            
            # Add overall status
            overall_status = "PASSED" if validation_result.is_valid else "FAILED"
            status_color = 'lightgreen' if validation_result.is_valid else 'lightcoral'
            status_text += f"\nOverall: {overall_status}"
            
            # Add validation status box
            ax.text(0.98, 0.98, status_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.8))
            
            # Add envelope width visualization near convergence
            self._add_envelope_width_indicator(ax, envelope_data)
            
        except ImportError:
            logger.warning("Convergence validator not available - skipping diagnostics")
        except Exception as e:
            logger.warning(f"Failed to add convergence diagnostics: {str(e)}")
    
    def _add_envelope_width_indicator(self, ax: plt.Axes, envelope_data: Dict) -> None:
        """
        Add visual indicator of envelope width near convergence.
        
        Args:
            ax: Matplotlib axes to plot on
            envelope_data: Envelope data dictionary
        """
        # Get envelope bounds
        if 'lower_bound_harvest' in envelope_data and 'upper_bound_harvest' in envelope_data:
            lower_harvest = np.array(envelope_data['lower_bound_harvest'])
            lower_production = np.array(envelope_data['lower_bound_production'])
            upper_production = np.array(envelope_data['upper_bound_production'])
        else:
            # Legacy format
            lower_harvest = np.array(envelope_data['disrupted_areas'])
            lower_production = np.array(envelope_data['lower_bound'])
            upper_production = np.array(envelope_data['upper_bound'])
        
        if len(lower_harvest) < 2:
            return
        
        # Calculate envelope width
        envelope_width = upper_production - lower_production
        
        # Focus on the final 20% of the envelope to show convergence behavior
        n_points = len(lower_harvest)
        start_idx = max(0, int(0.8 * n_points))
        
        final_harvest = lower_harvest[start_idx:]
        final_width = envelope_width[start_idx:]
        
        if len(final_harvest) > 1:
            # Convert to log scale
            final_harvest_log = np.log10(final_harvest)
            
            # Create a secondary y-axis for width visualization
            ax2 = ax.twinx()
            ax2.plot(final_harvest_log, final_width, 'purple', linewidth=2, alpha=0.7,
                    linestyle='--', label='Envelope Width')
            ax2.set_ylabel('Envelope Width (kcal)', color='purple', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='purple')
            
            # Calculate width reduction
            if len(final_width) > 1:
                initial_width = final_width[0]
                final_width_val = final_width[-1]
                width_reduction = (initial_width - final_width_val) / initial_width * 100
                
                # Add width reduction annotation
                width_text = f'Width Reduction\n(final 20%): {width_reduction:.1f}%'
                ax.text(0.02, 0.02, width_text, transform=ax.transAxes,
                       verticalalignment='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.8))
    
    def create_convergence_analysis_plot(self, envelope_data: Dict,
                                       total_production: float, total_harvest: float,
                                       save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a detailed convergence analysis plot with multiple subplots.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production for convergence validation
            total_harvest: Total harvest area for convergence validation
            save_path: Optional path to save the figure
        
        Returns:
            matplotlib Figure object with convergence analysis
        """
        try:
            # Import diagnostics if not already done
            if self._envelope_diagnostics is None:
                from ..analysis.envelope_diagnostics import EnvelopeDiagnostics
                self._envelope_diagnostics = EnvelopeDiagnostics()
            
            # Create the convergence analysis plot
            fig = self._envelope_diagnostics.plot_convergence_analysis(
                envelope_data, total_production, total_harvest, save_path
            )
            
            # Add crop type to the title
            fig.suptitle(f'H-P Envelope Convergence Analysis - {self.crop_type.title()}', 
                        fontsize=16, fontweight='bold')
            
            logger.info(f"Created convergence analysis plot for {self.crop_type}")
            return fig
            
        except ImportError:
            logger.error("Envelope diagnostics not available - cannot create convergence analysis plot")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Convergence analysis not available\n(envelope_diagnostics module not found)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Convergence Analysis - {self.crop_type.title()}')
            return fig
        except Exception as e:
            logger.error(f"Failed to create convergence analysis plot: {str(e)}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'Error creating convergence analysis:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Convergence Analysis Error - {self.crop_type.title()}')
            return fig
    
    def create_bounds_convergence_plot(self, envelope_data: Dict,
                                     total_production: float, total_harvest: float,
                                     save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a specialized plot showing how bounds approach each other.
        
        Args:
            envelope_data: Envelope data dictionary
            total_production: Total production for convergence validation
            total_harvest: Total harvest area for convergence validation
            save_path: Optional path to save the figure
        
        Returns:
            matplotlib Figure object with bounds convergence analysis
        """
        try:
            # Import diagnostics if not already done
            if self._envelope_diagnostics is None:
                from ..analysis.envelope_diagnostics import EnvelopeDiagnostics
                self._envelope_diagnostics = EnvelopeDiagnostics()
            
            # Create the bounds convergence plot
            fig = self._envelope_diagnostics.create_bounds_convergence_plot(
                envelope_data, total_production, total_harvest, save_path
            )
            
            # Add crop type to the title
            fig.suptitle(f'Envelope Bounds Convergence - {self.crop_type.title()}', 
                        fontsize=16, fontweight='bold')
            
            logger.info(f"Created bounds convergence plot for {self.crop_type}")
            return fig
            
        except ImportError:
            logger.error("Envelope diagnostics not available - cannot create bounds convergence plot")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Bounds convergence analysis not available\n(envelope_diagnostics module not found)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Bounds Convergence Analysis - {self.crop_type.title()}')
            return fig
        except Exception as e:
            logger.error(f"Failed to create bounds convergence plot: {str(e)}")
            # Create error plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error creating bounds convergence plot:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Bounds Convergence Error - {self.crop_type.title()}')
            return fig
    
    def _save_figure(self, fig: plt.Figure, save_path: Union[str, Path]) -> None:
        """Save figure in multiple formats."""
        save_path = Path(save_path)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        formats = ['png', 'svg', 'eps']
        for fmt in formats:
            output_path = save_path.with_suffix(f'.{fmt}')
            fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved H-P Envelope plot: {output_path}")


def create_sample_envelope_data(crop_type: str) -> Dict:
    """Create sample envelope data for testing."""
    # Create sample disrupted areas (km²)
    disrupted_areas = np.logspace(2, 6, 50)  # 100 to 1,000,000 km²
    
    # Create sample upper and lower bounds (kcal)
    # Upper bound: higher production loss (more productive areas disrupted first)
    upper_bound = 1e12 * disrupted_areas**1.2
    
    # Lower bound: lower production loss (less productive areas disrupted first)  
    lower_bound = 5e11 * disrupted_areas**1.1
    
    # Add some realistic scaling based on crop type
    if crop_type == 'allgrain':
        upper_bound *= 2.0
        lower_bound *= 1.5
    elif crop_type == 'wheat':
        upper_bound *= 1.2
        lower_bound *= 1.0
    elif crop_type == 'rice':
        upper_bound *= 1.0
        lower_bound *= 0.8
    
    # Ensure bounds don't exceed realistic limits
    upper_bound = np.minimum(upper_bound, 1e16)
    lower_bound = np.minimum(lower_bound, upper_bound * 0.8)
    
    return {
        'disrupted_areas': disrupted_areas,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound
    }


def create_sample_events_data(crop_type: str) -> pd.DataFrame:
    """Create sample events data for testing."""
    # Sample historical events with approximate values
    sample_events = {
        'wheat': [
            {'event_name': 'Dust Bowl', 'harvest_area_km2': 100000, 'production_loss_kcal': 5e14},
            {'event_name': 'Soviet Famine 1921', 'harvest_area_km2': 200000, 'production_loss_kcal': 8e14},
            {'event_name': 'Great Famine', 'harvest_area_km2': 50000, 'production_loss_kcal': 3e14}
        ],
        'rice': [
            {'event_name': 'Bangladesh 1974', 'harvest_area_km2': 30000, 'production_loss_kcal': 2e14},
            {'event_name': 'Chinese Famine 1960', 'harvest_area_km2': 80000, 'production_loss_kcal': 4e14},
            {'event_name': 'Drought 1876-1878', 'harvest_area_km2': 150000, 'production_loss_kcal': 6e14}
        ],
        'allgrain': [
            {'event_name': 'Dust Bowl', 'harvest_area_km2': 300000, 'production_loss_kcal': 1.2e15},
            {'event_name': 'Chinese Famine 1960', 'harvest_area_km2': 500000, 'production_loss_kcal': 2e15},
            {'event_name': 'Drought 1876-1878', 'harvest_area_km2': 800000, 'production_loss_kcal': 3.5e15}
        ]
    }
    
    events = sample_events.get(crop_type, sample_events['wheat'])
    return pd.DataFrame(events)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from agrichter.core.config import Config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test for wheat
    config = Config('wheat', use_dynamic_thresholds=True)
    visualizer = HPEnvelopeVisualizer(config)
    
    # Create sample data
    envelope_data = create_sample_envelope_data('wheat')
    events_data = create_sample_events_data('wheat')
    
    # Create plot with convergence analysis
    total_production = 1e15  # Example total production
    total_harvest = 1e6      # Example total harvest area
    
    fig = visualizer.create_hp_envelope_plot(
        envelope_data, events_data, 'test_hp_envelope_wheat.png',
        show_convergence=True, total_production=total_production, total_harvest=total_harvest
    )
    
    # Create detailed convergence analysis plot
    fig_convergence = visualizer.create_convergence_analysis_plot(
        envelope_data, total_production, total_harvest, 'test_convergence_analysis_wheat.png'
    )
    
    # Create bounds convergence plot
    fig_bounds = visualizer.create_bounds_convergence_plot(
        envelope_data, total_production, total_harvest, 'test_bounds_convergence_wheat.png'
    )
    
    print("H-P Envelope visualization with convergence analysis test completed!")
    plt.show()
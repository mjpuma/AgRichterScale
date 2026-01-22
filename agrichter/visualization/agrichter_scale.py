"""AgRichter Scale visualization - analogous to earthquake Richter scale.

The AgRichter Scale presents agricultural disruption magnitude similar to how
the Richter scale presents earthquake magnitude:
- X-axis: Magnitude (M_D = log₁₀(A_H)) - increases to the right
- Y-axis: Harvest Area Disrupted (km²) - on logarithmic scale, increases upward

This orientation matches the familiar Richter scale convention where magnitude
increases horizontally and physical impact increases vertically.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class AgRichterScaleVisualizer:
    """Visualizer for AgRichter Scale plots (analogous to earthquake Richter scale)."""
    
    def __init__(self, config, use_event_types: bool = True):
        """
        Initialize AgRichter Scale visualizer.
        
        Args:
            config: Configuration object with crop parameters and thresholds
            use_event_types: If True, color events by type from food_disruptions.csv.
                           If False, use IPC phase colors (backward compatible)
        """
        self.config = config
        self.crop_type = config.crop_type
        self.use_event_types = use_event_types
        
        # Get IPC colors for threshold lines (backward compatibility)
        self.ipc_colors = config.get_ipc_colors()
        
        # Load event metadata if using event types
        self.event_metadata = None
        if use_event_types:
            self.event_metadata = self._load_event_metadata()
        
        # Set up figure parameters
        self.figure_params = {
            'figsize': (10, 8),
            'dpi': 300,
            'facecolor': 'white'
        }
        
        # Define unified event type colors and markers (Publication Consistent)
        self.event_type_styles = {
            'Climate & Weather': {'color': '#1f77b4', 'marker': 'o', 'label': 'Climate & Weather'},  # Blue
            'Conflict & Policy': {'color': '#d62728', 'marker': 's', 'label': 'Conflict & Policy'},  # Red
            'Pest & Disease':    {'color': '#2ca02c', 'marker': 'D', 'label': 'Pest & Disease'},     # Green
            'Geophysical':       {'color': '#ff7f0e', 'marker': '^', 'label': 'Geophysical'},        # Orange
            'Compound/Other':    {'color': '#7f7f7f', 'marker': '*', 'label': 'Compound/Other'}      # Gray
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
    
    def _load_event_metadata(self) -> Optional[pd.DataFrame]:
        """Load event metadata from food_disruptions.csv.
        
        Returns:
            DataFrame with event metadata, or None if file not found
        """
        try:
            metadata_file = self.config.root_dir / 'ancillary' / 'food_disruptions.csv'
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                logger.info(f"Loaded event metadata for {len(df)} events")
                return df
            else:
                logger.warning(f"Event metadata file not found: {metadata_file}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load event metadata: {e}")
            return None
    
    def create_agrichter_scale_plot(self, events_data: pd.DataFrame, 
                                    save_path: Optional[Union[str, Path]] = None,
                                    ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create AgRichter Scale visualization (analogous to earthquake Richter scale).
        
        Plots Magnitude (M_D) on X-axis and Harvest Area Disrupted on Y-axis,
        matching the familiar Richter scale convention where magnitude increases
        to the right and physical impact increases upward.
        
        Args:
            events_data: DataFrame with columns: event_name, harvest_area_loss_ha, production_loss_kcal
                        (harvest_area_km2 will be calculated from harvest_area_loss_ha)
            save_path: Optional path to save the figure
            ax: Optional matplotlib axes to plot on. If None, creates a new figure.
        
        Returns:
            matplotlib Figure object
        """
        # Convert harvest area from hectares to km² if needed
        events_data = self._prepare_events_data(events_data)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_params['figsize'], 
                                  dpi=self.figure_params['dpi'],
                                  facecolor=self.figure_params['facecolor'])
        else:
            fig = ax.get_figure()
        
        # Calculate magnitude: M_D = log10(A_H)
        events_data = events_data.copy()
        events_data['magnitude'] = np.log10(events_data['harvest_area_km2'])
        
        # Set axis limits based on crop type
        xlim, ylim = self._get_axis_limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Plot the magnitude-area relationship line (A_H = 10^M_D)
        self._plot_magnitude_area_relationship(ax, xlim)
        
        # REMOVED for publication: Thresholds (Production-based) are arbitrary on Area-based plot
        # Plot Threshold lines (from Config, converted to Area)
        # self._plot_agriPhase_thresholds_richter_style(ax, xlim)
        
        # Plot historical events as circles with labels
        self._plot_historical_events_richter_style(ax, events_data)
        
        # Set logarithmic y-scale (harvest area)
        ax.set_yscale('log')
        
        # Set axis labels - Richter scale style
        ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Harvest Area Disrupted (km²)', fontsize=13, fontweight='bold')
        
        # Set title
        title = f'AgRichter Scale - {self.crop_type.title()}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend in bottom right quadrant
        # Get handles and labels, organize by type
        handles, labels = ax.get_legend_handles_labels()
        
        # Separate theoretical line from thresholds from event types
        theoretical_items = [(h, l) for h, l in zip(handles, labels) 
                            if 'Theoretical' in l or 'uniform' in l or 'Ending Stocks' in l]
        threshold_items = [(h, l) for h, l in zip(handles, labels) 
                          if ('Supply' in l or 'Stocks' in l or 'Phase' in l) and 'Ending Stocks' not in l]
        event_items = [(h, l) for h, l in zip(handles, labels) 
                      if l not in [item[1] for item in theoretical_items + threshold_items]]
        
        # Combine in order: theoretical, thresholds, event types
        ordered_handles = [h for h, l in theoretical_items] + [h for h, l in threshold_items] + [h for h, l in event_items]
        ordered_labels = [l for h, l in theoretical_items] + [l for h, l in threshold_items] + [l for h, l in event_items]
        
        # Create legend in bottom right with appropriate columns
        if len(ordered_labels) > 6:
            ax.legend(ordered_handles, ordered_labels, loc='lower right', fontsize=9, ncol=2)
        else:
            ax.legend(ordered_handles, ordered_labels, loc='lower right', fontsize=10)
        
        # Adjust layout only if we created the figure
        if save_path or ax is None:
            plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            self._save_figure(fig, save_path)
        
        logger.info(f"Created AgRichter Scale plot for {self.crop_type}")
        return fig
    
    def _get_axis_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get appropriate axis limits for the crop type (Richter scale style)."""
        # X-axis: Magnitude limits (M_D = log10 of harvest area in km²)
        # Extended lower bound to include smaller events (Magnitude 0 = 1 km²)
        magnitude_limits = {
            'allgrain': (0, 7.5),
            'wheat': (0, 7),
            'rice': (0, 6.5),
            'maize': (0, 7)
        }
        
        # Y-axis: Harvest area limits (km²) - will be displayed on log scale
        # Extended lower bound to 1 km² (10^0)
        harvest_area_limits = {
            'allgrain': (1e0, 10**7.5), 
            'wheat': (1e0, 1e7),
            'rice': (1e0, 10**6.5),
            'maize': (1e0, 1e7)
        }
        
        xlim = magnitude_limits.get(self.crop_type, (0, 7))
        ylim = harvest_area_limits.get(self.crop_type, (1e0, 1e7))
        
        return xlim, ylim
    
    def _plot_magnitude_area_relationship(self, ax: plt.Axes, xlim: Tuple[float, float]) -> None:
        """
        Plot the fundamental magnitude-area relationship line.
        
        This shows the direct relationship: A_H = 10^(M_D)
        Since M_D = log₁₀(A_H), this appears as a diagonal line on the log-scale plot.
        This is analogous to the reference line on earthquake Richter scale plots.
        """
        # Create magnitude range
        magnitudes = np.linspace(xlim[0], xlim[1], 100)
        
        # Calculate corresponding harvest areas: A_H = 10^M_D
        harvest_areas = 10 ** magnitudes
        
        # Plot the relationship line
        ax.plot(magnitudes, harvest_areas, 
               color='black', linestyle='-', linewidth=2.5, 
               alpha=0.8, zorder=2,
               label='Magnitude-Area Relationship (A_H = 10^M_D)')
    
    def _format_area(self, value: float) -> str:
        """Format area values with appropriate units (M, k)."""
        if value >= 1e6:
            return f"{value/1e6:.1f}M km²"
        elif value >= 1e3:
            return f"{value/1e3:.1f}k km²"
        else:
            return f"{value:.0f} km²"

    def _plot_agriPhase_thresholds_richter_style(self, ax: plt.Axes, xlim: Tuple[float, float]) -> None:
        """Plot AgriPhase/Supply threshold lines with colors."""
        # Get thresholds from config (values in kcal)
        thresholds = self.config.get_thresholds()
        colors = self.config.get_ipc_colors()
        
        # Calculate yield (kcal/km²) to convert production thresholds to harvest area
        # Use the same assumptions as _plot_theoretical_line for consistency
        if self.crop_type == 'allgrain':
            base_production = 2.5e15  # kcal
            area_scaling = 1e6  # km²
        elif self.crop_type == 'wheat':
            base_production = 8e14
            area_scaling = 2.2e5
        elif self.crop_type == 'rice':
            base_production = 6e14
            area_scaling = 1.6e5
        elif self.crop_type == 'maize':
            base_production = 1.2e15
            area_scaling = 1.9e5
        else:
            base_production = 8e14
            area_scaling = 2e5
        
        yield_density = base_production / area_scaling  # kcal/km²
        
        # Plot horizontal threshold lines
        for name, value_kcal in thresholds.items():
            # SKIP "Total Stocks" for publication clarity if it's very close to "3 Months"
            if name == 'Total Stocks':
                if '3 Months' in thresholds:
                    dist = abs(np.log10(value_kcal) - np.log10(thresholds['3 Months']))
                    if dist < 0.1:
                        logger.info(f"Skipping '{name}' threshold line: too close to '3 Months' ({dist:.2f} log units)")
                        continue

            # Convert kcal threshold to km² equivalent
            area_km2 = value_kcal / yield_density
            
            # Determine color and label
            color = colors.get(name, '#000000')
            
            # Format label with area
            formatted_area = self._format_area(area_km2)
            label = f"{name} (~{formatted_area})"
            
            # Only plot if within reasonable range
            if area_km2 > 10**xlim[0] and area_km2 < 10**xlim[1]:
                ax.axhline(y=area_km2, color=color, linestyle='--', 
                          linewidth=2, alpha=0.8, label=label)
                
                # Add text annotation on the right side
                ax.text(xlim[1], area_km2, f" {name}", 
                       color=color, fontsize=8, va='center', ha='left', fontweight='bold')

    def _plot_agriPhase_thresholds(self, ax: plt.Axes, xlim: Tuple[float, float]) -> None:
        """
        Alias for _plot_agriPhase_thresholds_richter_style.
        Maintained for backward compatibility if needed.
        """
        self._plot_agriPhase_thresholds_richter_style(ax, xlim)
    
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
        
        # Handle empty DataFrame
        if events_data.empty:
            if 'harvest_area_km2' not in events_data.columns:
                events_data['harvest_area_km2'] = []
            return events_data
        
        # Convert harvest area from hectares to km² if needed
        if 'harvest_area_loss_ha' in events_data.columns and 'harvest_area_km2' not in events_data.columns:
            # 1 hectare = 0.01 km²
            events_data['harvest_area_km2'] = events_data['harvest_area_loss_ha'] * 0.01
            logger.info("Converted harvest area from hectares to km²")
        
        # Filter out events with zero or invalid harvest area
        if len(events_data) > 0:
            valid_mask = (events_data['harvest_area_km2'] > 0) & np.isfinite(events_data['harvest_area_km2'])
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                logger.warning(f"Filtered out {invalid_count} events with zero or invalid harvest area")
                events_data = events_data[valid_mask]
        
        # Filter out events with zero or invalid production loss
        if len(events_data) > 0:
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
    
    def _plot_historical_events_richter_style(self, ax: plt.Axes, events_data: pd.DataFrame) -> None:
        """
        Plot historical events (Richter scale style).
        
        X-axis: Magnitude (M_D)
        Y-axis: Harvest Area (km²)
        
        If use_event_types=True, colors events by type from food_disruptions.csv
        Otherwise uses red for all events (backward compatible)
        """
        if events_data.empty:
            logger.warning("No events data to plot")
            return
        
        # Filter out events with zero or NaN harvest area
        valid_events = events_data[
            (events_data['harvest_area_km2'] > 0) & 
            (events_data['harvest_area_km2'].notna()) &
            (events_data['magnitude'].notna())
        ].copy()
        
        if valid_events.empty:
            logger.warning("No valid events with non-zero harvest area")
            return
        
        logger.info(f"Plotting {len(valid_events)} historical events (Richter style)")

        events = valid_events
        
        # Consolidate event types
        if self.use_event_types and 'event_type' in events.columns:
            events['consolidated_type'] = events['event_type'].apply(self._consolidate_event_type)
        else:
            events['consolidated_type'] = 'Unknown'
        
        # Plot events grouped by consolidated event type for proper legend
        plotted_types = set()
        for event_type, style in self.event_type_styles.items():
            type_data = events[events['consolidated_type'] == event_type]
            if not type_data.empty:
                # Plot this event type group
                ax.scatter(type_data['magnitude'], type_data['harvest_area_km2'],
                           c=style['color'], s=100, alpha=0.8, edgecolors='black', linewidth=1,
                           marker=style['marker'], label=style['label'], zorder=5)
                plotted_types.add(event_type)
        
        # Plot unknown types if any
        unknown_mask = ~events['consolidated_type'].isin(self.event_type_styles.keys())
        unknown_data = events[unknown_mask]
        if not unknown_data.empty:
             ax.scatter(unknown_data['magnitude'], unknown_data['harvest_area_km2'],
                       c='gray', s=100, alpha=0.8, edgecolors='black', linewidth=1,
                       marker='o', label='Other', zorder=5)
        
        # Try to use adjustText for better label placement
        try:
            from adjustText import adjust_text
            
            # Create text annotations
            texts = []
            for idx, row in events.iterrows():
                # Use event type color for text label
                etype = row.get('consolidated_type', 'Unknown')
                if etype in self.event_type_styles:
                    text_color = self.event_type_styles[etype]['color']
                else:
                    text_color = 'black'
                
                # Get publication-ready label
                display_name = self.config.get_event_label(row['event_name'])
                    
                text = ax.text(row['magnitude'], row['harvest_area_km2'], 
                             f"  {display_name}", fontsize=8,
                                 ha='left', va='center', color=text_color,
                                 fontweight='bold')
                texts.append(text)
            
            # Adjust text positions to avoid overlap
            if texts:
                adjust_text(texts, ax=ax,
                          expand_points=(1.8, 1.8),
                          expand_text=(1.5, 1.5),
                          force_points=(0.8, 0.8),
                          force_text=(0.8, 0.8),
                          lim=1000,
                          arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.8))
        except ImportError:
            # Fallback if adjustText not available
            for idx, row in events.iterrows():
                etype = row.get('consolidated_type', 'Unknown')
                if etype in self.event_type_styles:
                    text_color = self.event_type_styles[etype]['color']
                else:
                    text_color = 'black'

                # Get publication-ready label
                display_name = self.config.get_event_label(row['event_name'])

                # Position label slightly offset from point
                ax.annotate(display_name, 
                           xy=(row['magnitude'], row['harvest_area_km2']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, ha='left', color=text_color,
                              fontweight='bold',
                              arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.6))
    
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
            logger.info(f"Saved AgRichter Scale plot: {output_path}")


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
    visualizer = AgRichterScaleVisualizer(config)
    
    # Create sample events data
    events_data = create_sample_events_data('wheat')
    
    # Create plot
    fig = visualizer.create_agrichter_scale_plot(events_data, 'test_agrichter_scale_wheat.png')
    
    print("AgRichter Scale visualization test completed!")
    plt.show()
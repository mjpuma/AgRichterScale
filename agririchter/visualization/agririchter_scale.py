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


class AgriRichterScaleVisualizer:
    """Visualizer for AgRichter Scale plots (analogous to earthquake Richter scale)."""
    
    def __init__(self, config):
        """
        Initialize AgRichter Scale visualizer.
        
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
    
    def create_agririchter_scale_plot(self, events_data: pd.DataFrame, 
                                    save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create AgRichter Scale visualization (analogous to earthquake Richter scale).
        
        Plots Magnitude (M_D) on X-axis and Harvest Area Disrupted on Y-axis,
        matching the familiar Richter scale convention where magnitude increases
        to the right and physical impact increases upward.
        
        Args:
            events_data: DataFrame with columns: event_name, harvest_area_loss_ha, production_loss_kcal
                        (harvest_area_km2 will be calculated from harvest_area_loss_ha)
            save_path: Optional path to save the figure
        
        Returns:
            matplotlib Figure object
        """
        # Convert harvest area from hectares to km² if needed
        events_data = self._prepare_events_data(events_data)
        
        fig, ax = plt.subplots(figsize=self.figure_params['figsize'], 
                              dpi=self.figure_params['dpi'],
                              facecolor=self.figure_params['facecolor'])
        
        # Calculate magnitude: M_D = log10(A_H)
        events_data = events_data.copy()
        events_data['magnitude'] = np.log10(events_data['harvest_area_km2'])
        
        # Set axis limits based on crop type
        xlim, ylim = self._get_axis_limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Plot the magnitude-area relationship line (A_H = 10^M_D)
        self._plot_magnitude_area_relationship(ax, xlim)
        
        # Plot AgriPhase threshold lines (horizontal lines at different harvest areas)
        self._plot_agriPhase_thresholds_richter_style(ax, xlim)
        
        # Plot historical events as circles with labels
        self._plot_historical_events_richter_style(ax, events_data)
        
        # Set logarithmic y-scale (harvest area)
        ax.set_yscale('log')
        
        # Set axis labels - Richter scale style
        ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H)$)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Harvest Area Disrupted (km²)', fontsize=13, fontweight='bold')
        
        # Set title
        title = f'AgRichter Scale - {self.crop_type.title()}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend with better organization
        # Get handles and labels, organize by type
        handles, labels = ax.get_legend_handles_labels()
        
        # Separate theoretical line from thresholds from events
        theoretical_items = [(h, l) for h, l in zip(handles, labels) 
                            if 'Theoretical' in l or 'uniform' in l]
        threshold_items = [(h, l) for h, l in zip(handles, labels) 
                          if 'Phase' in l and '(' in l and ('Stressed' in l or 'Crisis' in l or 'Emergency' in l or 'Famine' in l)]
        event_items = [(h, l) for h, l in zip(handles, labels) 
                      if 'Phase' in l and '(' in l and 'Minimal' in l]
        
        # Combine in order: theoretical, thresholds, events
        ordered_handles = [h for h, l in theoretical_items] + [h for h, l in threshold_items] + [h for h, l in event_items]
        ordered_labels = [l for h, l in theoretical_items] + [l for h, l in threshold_items] + [l for h, l in event_items]
        
        # Create legend with two columns if many items
        if len(ordered_labels) > 6:
            ax.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=9, ncol=2)
        else:
            ax.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            self._save_figure(fig, save_path)
        
        logger.info(f"Created AgriRichter Scale plot for {self.crop_type}")
        return fig
    
    def _get_axis_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get appropriate axis limits for the crop type (Richter scale style)."""
        # X-axis: Magnitude limits (M_D = log10 of harvest area in km²)
        magnitude_limits = {
            'allgrain': (2, 7),
            'wheat': (2, 6.5),
            'rice': (2, 6),
            'maize': (2, 6.5)
        }
        
        # Y-axis: Harvest area limits (km²) - will be displayed on log scale
        harvest_area_limits = {
            'allgrain': (1e2, 1e7),  # 100 km² to 10 million km²
            'wheat': (1e2, 1e7),
            'rice': (1e2, 1e6),
            'maize': (1e2, 1e7)
        }
        
        xlim = magnitude_limits.get(self.crop_type, (2, 7))
        ylim = harvest_area_limits.get(self.crop_type, (1e2, 1e7))
        
        return xlim, ylim
    
    def _plot_theoretical_line(self, ax: plt.Axes, xlim: Tuple[float, float]) -> None:
        """Plot theoretical production loss line using uniform production assumption."""
        # Create magnitude range
        magnitude_range = np.linspace(xlim[0], xlim[1], 100)
        
        # Calculate theoretical production loss
        # Assumption: uniform production density across disrupted area
        # Production loss ∝ (disrupted area)^1 for uniform distribution
        
        # Use crop-specific parameters for scaling
        if self.crop_type == 'allgrain':
            # Combined global production estimate
            base_production = 2.5e15  # kcal (approximate global grain production)
            area_scaling = 1e6  # km² (approximate global grain area)
        elif self.crop_type == 'wheat':
            base_production = 8e14  # kcal
            area_scaling = 2.2e5  # km²
        elif self.crop_type == 'rice':
            base_production = 6e14  # kcal
            area_scaling = 1.6e5  # km²
        elif self.crop_type == 'maize':
            base_production = 1.2e15  # kcal
            area_scaling = 1.9e5  # km²
        else:
            base_production = 8e14
            area_scaling = 2e5
        
        # Calculate production density (kcal/km²)
        production_density = base_production / area_scaling
        
        # Theoretical production loss = production_density * disrupted_area
        disrupted_area = 10**magnitude_range  # Convert from log10
        theoretical_loss = production_density * disrupted_area
        
        # Plot theoretical line
        ax.plot(magnitude_range, theoretical_loss, 'k--', linewidth=2, 
                label='Theoretical (uniform production)', alpha=0.7)
    
    def _plot_agriPhase_thresholds(self, ax: plt.Axes, xlim: Tuple[float, float]) -> None:
        """Plot AgriPhase threshold lines with IPC colors."""
        # Get thresholds from config
        thresholds = self.config.get_thresholds()
        
        # IPC Phase mapping
        phase_mapping = {'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}
        phase_names = {'T1': 'Phase 2 (Stressed)', 'T2': 'Phase 3 (Crisis)', 
                      'T3': 'Phase 4 (Emergency)', 'T4': 'Phase 5 (Famine)'}
        
        # Plot horizontal threshold lines
        for threshold_name, threshold_value in thresholds.items():
            if threshold_name in phase_mapping:
                phase_num = phase_mapping[threshold_name]
                color = self.ipc_colors.get(phase_num, '#000000')
                phase_name = phase_names.get(threshold_name, threshold_name)
                
                # Plot horizontal line across magnitude range
                ax.axhline(y=threshold_value, color=color, linestyle='--', 
                          linewidth=2, alpha=0.8, label=phase_name)
    
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
    
    def _plot_historical_events(self, ax: plt.Axes, events_data: pd.DataFrame) -> None:
        """Plot historical events with severity-based colors and markers using adjustText for non-overlapping placement."""
        if events_data.empty:
            logger.warning("No events data provided for plotting")
            return
        
        # Classify events by severity
        events_data['phase'] = events_data['production_loss_kcal'].apply(
            lambda x: self._classify_event_severity(x)[0]
        )
        events_data['phase_name'] = events_data['production_loss_kcal'].apply(
            lambda x: self._classify_event_severity(x)[1]
        )
        events_data['color'] = events_data['production_loss_kcal'].apply(
            lambda x: self._classify_event_severity(x)[2]
        )
        events_data['marker'] = events_data['production_loss_kcal'].apply(
            lambda x: self._classify_event_severity(x)[3]
        )
        
        # Plot events grouped by severity phase for proper legend
        plotted_phases = set()
        for phase in sorted(events_data['phase'].unique()):
            phase_data = events_data[events_data['phase'] == phase]
            if not phase_data.empty:
                phase_name = phase_data.iloc[0]['phase_name']
                color = phase_data.iloc[0]['color']
                marker = phase_data.iloc[0]['marker']
                
                # Plot this severity group
                scatter = ax.scatter(phase_data['magnitude'], phase_data['production_loss_kcal'],
                               c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1,
                               marker=marker, label=phase_name, zorder=5)
                plotted_phases.add(phase)
        
        # Try to use adjustText for better label placement
        try:
            from adjustText import adjust_text
            
            # Create text annotations
            texts = []
            for idx, row in events_data.iterrows():
                text = ax.text(row['magnitude'], row['production_loss_kcal'], 
                             row['event_name'],
                             fontsize=8, ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     alpha=0.7, edgecolor='red', linewidth=0.5))
                texts.append(text)
            
            # Adjust text positions to avoid overlaps
            adjust_text(texts, ax=ax,
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.6, lw=0.5),
                       expand_points=(1.5, 1.5),
                       force_points=(0.5, 0.5))
            
            logger.info(f"Plotted {len(texts)} events with adjustText label placement")
            
        except ImportError:
            logger.warning("adjustText not available, using simple label placement")
            # Fallback to simple label placement
            for idx, row in events_data.iterrows():
                # Position label slightly offset from point
                x_offset = 0.05
                y_offset = row['production_loss_kcal'] * 1.2
                
                ax.annotate(row['event_name'], 
                           xy=(row['magnitude'], row['production_loss_kcal']),
                           xytext=(row['magnitude'] + x_offset, y_offset),
                           fontsize=8, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                         color='red', alpha=0.6))
    
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
    
    def _plot_agriPhase_thresholds_richter_style(self, ax: plt.Axes, xlim: Tuple[float, float]) -> None:
        """
        Plot AgriPhase threshold lines (Richter scale style).
        
        Horizontal lines at different harvest area levels, colored by IPC phase.
        """
        # Get thresholds from config (these are production thresholds)
        thresholds = self.config.thresholds
        
        # For Richter style, we need to convert production thresholds to harvest area thresholds
        # This is approximate - we'll use typical production per unit area
        # For now, plot reference lines at round harvest area values
        
        # Define harvest area thresholds (km²) for different severity levels
        area_thresholds = {
            'Minimal': 1e3,      # 1,000 km²
            'Stressed': 1e4,     # 10,000 km²
            'Crisis': 1e5,       # 100,000 km²
            'Emergency': 1e6,    # 1,000,000 km²
            'Famine': 1e7        # 10,000,000 km²
        }
        
        # Plot horizontal lines
        for phase_name, area_threshold in area_thresholds.items():
            color = self.ipc_colors.get(phase_name, 'gray')
            ax.axhline(y=area_threshold, color=color, linestyle='--',
                      linewidth=2, alpha=0.6, label=f'{phase_name} (~{area_threshold/1e3:.0f}k km²)')
    
    def _plot_historical_events_richter_style(self, ax: plt.Axes, events_data: pd.DataFrame) -> None:
        """
        Plot historical events (Richter scale style).
        
        X-axis: Magnitude (M_D)
        Y-axis: Harvest Area (km²)
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
        
        # Plot events as red circles
        ax.scatter(valid_events['magnitude'], valid_events['harvest_area_km2'],
                  s=100, c='red', alpha=0.7, edgecolors='darkred', linewidths=1.5,
                  zorder=5, label='Historical Events')
        
        # Add event labels with smart placement
        try:
            from adjustText import adjust_text
            texts = []
            for _, event in valid_events.iterrows():
                if event['harvest_area_km2'] > 0:
                    text = ax.text(event['magnitude'], event['harvest_area_km2'],
                                 f"  {event['event_name']}", fontsize=8,
                                 ha='left', va='center', color='darkred',
                                 fontweight='bold')
                    texts.append(text)
            
            # Adjust text positions to avoid overlap with improved parameters
            if texts:
                adjust_text(texts, ax=ax,
                          expand_points=(1.5, 1.5),    # More space around points
                          expand_text=(1.3, 1.3),      # More space around text
                          force_points=(0.5, 0.5),     # Stronger repulsion from points
                          force_text=(0.5, 0.5),       # Stronger repulsion between labels
                          lim=500,                     # More iterations for better placement
                          arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.6))
        except ImportError:
            # Fallback if adjustText not available
            for _, event in valid_events.iterrows():
                if event['harvest_area_km2'] > 0:
                    ax.annotate(event['event_name'],
                              xy=(event['magnitude'], event['harvest_area_km2']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, ha='left', color='darkred',
                              fontweight='bold',
                              arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.6))
    
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
            logger.info(f"Saved AgriRichter Scale plot: {output_path}")


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
    
    from agririchter.core.config import Config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test for wheat
    config = Config('wheat', use_dynamic_thresholds=True)
    visualizer = AgriRichterScaleVisualizer(config)
    
    # Create sample events data
    events_data = create_sample_events_data('wheat')
    
    # Create plot
    fig = visualizer.create_agririchter_scale_plot(events_data, 'test_agririchter_scale_wheat.png')
    
    print("AgriRichter Scale visualization test completed!")
    plt.show()
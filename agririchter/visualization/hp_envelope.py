"""MATLAB-exact H-P Envelope visualization."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class HPEnvelopeVisualizer:
    """Visualizer for MATLAB-exact H-P Envelope plots."""
    
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
    
    def create_hp_envelope_plot(self, envelope_data: Dict, events_data: pd.DataFrame,
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create MATLAB-exact H-P Envelope visualization.
        
        Args:
            envelope_data: Dictionary with 'disrupted_areas', 'upper_bound', 'lower_bound'
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
        
        # Convert harvest area to magnitude: M_D = log10(A_H)
        disrupted_areas = np.array(envelope_data['disrupted_areas'])
        upper_bound = np.array(envelope_data['upper_bound'])
        lower_bound = np.array(envelope_data['lower_bound'])
        
        # Calculate magnitude for x-axis
        magnitude = np.log10(disrupted_areas)
        
        # Set axis limits (MATLAB exact)
        ax.set_xlim([2, 7])
        ax.set_ylim([1e10, 1.62e16])
        
        # Plot gray envelope fill between upper and lower bounds
        self._plot_envelope_fill(ax, magnitude, upper_bound, lower_bound)
        
        # Plot upper bound (black) and lower bound (blue) boundary lines
        self._plot_boundary_lines(ax, magnitude, upper_bound, lower_bound)
        
        # Plot AgriPhase threshold lines as horizontal dashed lines
        self._plot_agriPhase_thresholds(ax)
        
        # Plot historical events as red circles with labels
        self._plot_historical_events(ax, events_data)
        
        # Set logarithmic y-scale
        ax.set_yscale('log')
        
        # Set axis labels
        ax.set_xlabel('Magnitude M_D = log₁₀(A_H) [km²]', fontsize=12)
        ax.set_ylabel('Production Loss [kcal]', fontsize=12)
        
        # Set title
        title = f'H-P Envelope - {self.crop_type.title()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend with better organization
        # Get handles and labels, organize by type
        handles, labels = ax.get_legend_handles_labels()
        
        # Separate envelope/bounds from thresholds from events
        envelope_items = [(h, l) for h, l in zip(handles, labels) 
                         if 'Envelope' in l or 'Bound' in l]
        threshold_items = [(h, l) for h, l in zip(handles, labels) 
                          if 'Phase' in l and '(' in l and 'Stressed' in l or 'Crisis' in l or 'Emergency' in l or 'Famine' in l]
        event_items = [(h, l) for h, l in zip(handles, labels) 
                      if 'Phase' in l and '(' in l and 'Minimal' in l]
        
        # Combine in order: envelope, thresholds, events
        ordered_handles = [h for h, l in envelope_items] + [h for h, l in threshold_items] + [h for h, l in event_items]
        ordered_labels = [l for h, l in envelope_items] + [l for h, l in threshold_items] + [l for h, l in event_items]
        
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
        
        logger.info(f"Created H-P Envelope plot for {self.crop_type}")
        return fig
    
    def _plot_envelope_fill(self, ax: plt.Axes, magnitude: np.ndarray, 
                           upper_bound: np.ndarray, lower_bound: np.ndarray) -> None:
        """Plot gray envelope fill between upper and lower bounds."""
        # Remove NaN and Inf values
        valid_mask = (np.isfinite(magnitude) & np.isfinite(upper_bound) & 
                     np.isfinite(lower_bound) & (upper_bound > 0) & (lower_bound > 0))
        
        if not np.any(valid_mask):
            logger.warning("No valid data points for envelope fill")
            return
        
        magnitude_clean = magnitude[valid_mask]
        upper_clean = upper_bound[valid_mask]
        lower_clean = lower_bound[valid_mask]
        
        # Sort by magnitude for proper filling
        sort_idx = np.argsort(magnitude_clean)
        magnitude_sorted = magnitude_clean[sort_idx]
        upper_sorted = upper_clean[sort_idx]
        lower_sorted = lower_clean[sort_idx]
        
        # Fill between upper and lower bounds with gray color
        ax.fill_between(magnitude_sorted, lower_sorted, upper_sorted,
                       color='gray', alpha=0.3, label='H-P Envelope')
    
    def _plot_boundary_lines(self, ax: plt.Axes, magnitude: np.ndarray,
                            upper_bound: np.ndarray, lower_bound: np.ndarray) -> None:
        """Plot upper bound (black) and lower bound (blue) boundary lines."""
        # Remove NaN and Inf values
        valid_mask = (np.isfinite(magnitude) & np.isfinite(upper_bound) & 
                     np.isfinite(lower_bound) & (upper_bound > 0) & (lower_bound > 0))
        
        if not np.any(valid_mask):
            logger.warning("No valid data points for boundary lines")
            return
        
        magnitude_clean = magnitude[valid_mask]
        upper_clean = upper_bound[valid_mask]
        lower_clean = lower_bound[valid_mask]
        
        # Sort by magnitude
        sort_idx = np.argsort(magnitude_clean)
        magnitude_sorted = magnitude_clean[sort_idx]
        upper_sorted = upper_clean[sort_idx]
        lower_sorted = lower_clean[sort_idx]
        
        # Plot upper bound (black line)
        ax.plot(magnitude_sorted, upper_sorted, 'k-', linewidth=2, 
               label='Upper Bound', alpha=0.8)
        
        # Plot lower bound (blue line)
        ax.plot(magnitude_sorted, lower_sorted, 'b-', linewidth=2, 
               label='Lower Bound', alpha=0.8)
    
    def _plot_agriPhase_thresholds(self, ax: plt.Axes) -> None:
        """Plot AgriPhase threshold lines as horizontal dashed lines."""
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
                
                # Plot horizontal dashed line across magnitude range
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
    
    def _plot_historical_events(self, ax: plt.Axes, events_data: pd.DataFrame) -> None:
        """Plot historical events with severity-based colors and markers using adjustText for non-overlapping placement."""
        if events_data.empty:
            logger.warning("No events data provided for plotting")
            return
        
        # Calculate magnitude for events
        events_data = events_data.copy()
        events_data['magnitude'] = np.log10(events_data['harvest_area_km2'])
        
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
    
    from agririchter.core.config import Config
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test for wheat
    config = Config('wheat', use_dynamic_thresholds=True)
    visualizer = HPEnvelopeVisualizer(config)
    
    # Create sample data
    envelope_data = create_sample_envelope_data('wheat')
    events_data = create_sample_events_data('wheat')
    
    # Create plot
    fig = visualizer.create_hp_envelope_plot(envelope_data, events_data, 'test_hp_envelope_wheat.png')
    
    print("H-P Envelope visualization test completed!")
    plt.show()
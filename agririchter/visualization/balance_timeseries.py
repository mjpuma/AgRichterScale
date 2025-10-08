"""Balance Time Series visualization for USDA PSD data."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BalanceTimeSeriesVisualizer:
    """Visualizer for Balance Time Series plots using USDA PSD data."""
    
    def __init__(self, config):
        """
        Initialize Balance Time Series visualizer.
        
        Args:
            config: Configuration object with USDA data access
        """
        self.config = config
        self.crop_type = config.crop_type
        
        # Set up figure parameters
        self.figure_params = {
            'figsize': (12, 10),
            'dpi': 300,
            'facecolor': 'white'
        }
    
    def create_balance_timeseries_plot(self, year_range: Optional[Tuple[int, int]] = None,
                                     save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create Balance Time Series visualization with 2x2 subplot layout.
        
        Args:
            year_range: Optional year range for data (default: 1990-2020)
            save_path: Optional path to save the figure
        
        Returns:
            matplotlib Figure object
        """
        # Get USDA data
        year_range = year_range or (1990, 2020)
        usda_data = self.config.get_usda_data(year_range)
        
        if not usda_data:
            raise ValueError("No USDA data available for time series visualization")
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=self.figure_params['figsize'], 
                                dpi=self.figure_params['dpi'],
                                facecolor=self.figure_params['facecolor'])
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Plot each metric in a separate subplot
        self._plot_sur_timeseries(axes_flat[0], usda_data)
        self._plot_stocks_timeseries(axes_flat[1], usda_data)
        self._plot_production_timeseries(axes_flat[2], usda_data)
        self._plot_consumption_timeseries(axes_flat[3], usda_data)
        
        # Set overall title
        fig.suptitle(f'Balance Time Series - {self.crop_type.title()}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if path provided
        if save_path:
            self._save_figure(fig, save_path)
        
        logger.info(f"Created Balance Time Series plot for {self.crop_type}")
        return fig
    
    def _plot_sur_timeseries(self, ax: plt.Axes, usda_data: Dict) -> None:
        """Plot Stock-to-Use Ratio (SUR) time series."""
        # Calculate SUR for each crop in the data
        all_sur_data = []
        
        for crop_name, crop_data in usda_data.items():
            if not crop_data.empty:
                # Calculate SUR = Ending Stocks / Consumption
                sur = crop_data['EndingStocks'] / crop_data['Consumption']
                sur_df = pd.DataFrame({
                    'Year': crop_data['Year'],
                    'SUR': sur,
                    'Crop': crop_name
                })
                all_sur_data.append(sur_df)
        
        if not all_sur_data:
            ax.text(0.5, 0.5, 'No SUR data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Stock-to-Use Ratio (SUR)')
            return
        
        # Combine all SUR data
        combined_sur = pd.concat(all_sur_data, ignore_index=True)
        
        # Plot time series for each crop
        for crop_name in combined_sur['Crop'].unique():
            crop_sur = combined_sur[combined_sur['Crop'] == crop_name]
            ax.plot(crop_sur['Year'], crop_sur['SUR'], 'o-', label=crop_name, linewidth=2, markersize=4)
        
        # Calculate and plot median and percentiles
        yearly_stats = combined_sur.groupby('Year')['SUR'].agg(['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
        yearly_stats.columns = ['median', 'p25', 'p75']
        
        ax.plot(yearly_stats.index, yearly_stats['median'], 'k--', linewidth=2, alpha=0.7, label='Median')
        ax.fill_between(yearly_stats.index, yearly_stats['p25'], yearly_stats['p75'], 
                       alpha=0.2, color='gray', label='25-75th percentile')
        
        # Formatting
        ax.set_title('Stock-to-Use Ratio (SUR)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('SUR')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_stocks_timeseries(self, ax: plt.Axes, usda_data: Dict) -> None:
        """Plot Ending Stocks time series."""
        for crop_name, crop_data in usda_data.items():
            if not crop_data.empty:
                ax.plot(crop_data['Year'], crop_data['EndingStocks'], 'o-', 
                       label=crop_name, linewidth=2, markersize=4)
        
        # Calculate combined statistics if multiple crops
        if len(usda_data) > 1:
            all_stocks = []
            for crop_data in usda_data.values():
                if not crop_data.empty:
                    all_stocks.append(crop_data[['Year', 'EndingStocks']])
            
            if all_stocks:
                combined_stocks = pd.concat(all_stocks, ignore_index=True)
                yearly_median = combined_stocks.groupby('Year')['EndingStocks'].median()
                ax.plot(yearly_median.index, yearly_median.values, 'k--', 
                       linewidth=2, alpha=0.7, label='Median')
        
        # Formatting
        ax.set_title('Ending Stocks', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Stocks (1000 MT)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_production_timeseries(self, ax: plt.Axes, usda_data: Dict) -> None:
        """Plot Production time series."""
        for crop_name, crop_data in usda_data.items():
            if not crop_data.empty:
                ax.plot(crop_data['Year'], crop_data['Production'], 'o-', 
                       label=crop_name, linewidth=2, markersize=4)
        
        # Calculate combined statistics if multiple crops
        if len(usda_data) > 1:
            all_production = []
            for crop_data in usda_data.values():
                if not crop_data.empty:
                    all_production.append(crop_data[['Year', 'Production']])
            
            if all_production:
                combined_production = pd.concat(all_production, ignore_index=True)
                yearly_median = combined_production.groupby('Year')['Production'].median()
                ax.plot(yearly_median.index, yearly_median.values, 'k--', 
                       linewidth=2, alpha=0.7, label='Median')
        
        # Formatting
        ax.set_title('Production', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Production (1000 MT)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_consumption_timeseries(self, ax: plt.Axes, usda_data: Dict) -> None:
        """Plot Consumption time series."""
        for crop_name, crop_data in usda_data.items():
            if not crop_data.empty:
                ax.plot(crop_data['Year'], crop_data['Consumption'], 'o-', 
                       label=crop_name, linewidth=2, markersize=4)
        
        # Calculate combined statistics if multiple crops
        if len(usda_data) > 1:
            all_consumption = []
            for crop_data in usda_data.values():
                if not crop_data.empty:
                    all_consumption.append(crop_data[['Year', 'Consumption']])
            
            if all_consumption:
                combined_consumption = pd.concat(all_consumption, ignore_index=True)
                yearly_median = combined_consumption.groupby('Year')['Consumption'].median()
                ax.plot(yearly_median.index, yearly_median.values, 'k--', 
                       linewidth=2, alpha=0.7, label='Median')
        
        # Formatting
        ax.set_title('Consumption', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Consumption (1000 MT)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _save_figure(self, fig: plt.Figure, save_path: Union[str, Path]) -> None:
        """Save figure as PNG with crop-specific naming."""
        save_path = Path(save_path)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use PNG format as specified in requirements
        if save_path.suffix.lower() != '.png':
            save_path = save_path.with_suffix('.png')
        
        fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        logger.info(f"Saved Balance Time Series plot: {save_path}")


def create_balance_timeseries_for_crop(crop_type: str, year_range: Tuple[int, int] = (1990, 2020),
                                     output_dir: Union[str, Path] = "outputs") -> plt.Figure:
    """
    Create balance time series plot for a specific crop type.
    
    Args:
        crop_type: Crop type ('wheat', 'rice', 'maize', 'allgrain')
        year_range: Year range for data
        output_dir: Output directory for saving plots
    
    Returns:
        matplotlib Figure object
    """
    from ..core.config import Config
    
    # Initialize config with USDA data
    config = Config(crop_type, use_dynamic_thresholds=True, usda_year_range=year_range)
    
    # Create visualizer
    visualizer = BalanceTimeSeriesVisualizer(config)
    
    # Create output path
    output_path = Path(output_dir) / f'BalanceTimeSeries_{crop_type.title()}.png'
    
    # Create plot
    fig = visualizer.create_balance_timeseries_plot(year_range, output_path)
    
    return fig


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
    config = Config('wheat', use_dynamic_thresholds=True, usda_year_range=(1990, 2020))
    visualizer = BalanceTimeSeriesVisualizer(config)
    
    # Create plot
    fig = visualizer.create_balance_timeseries_plot(save_path='test_balance_timeseries_wheat.png')
    
    print("Balance Time Series visualization test completed!")
    plt.show()
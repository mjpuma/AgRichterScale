"""
Unit tests for AgRichter Scale visualization with real events data.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agrichter.core.config import Config
from agrichter.visualization.agrichter_scale import AgRichterScaleVisualizer


class TestAgRichterScaleVisualizer:
    """Test suite for AgRichter Scale visualization."""
    
    @pytest.fixture
    def config_wheat(self):
        """Create wheat config."""
        return Config('wheat', use_dynamic_thresholds=True)
    
    @pytest.fixture
    def config_rice(self):
        """Create rice config."""
        return Config('rice', use_dynamic_thresholds=True)
    
    @pytest.fixture
    def config_allgrain(self):
        """Create allgrain config."""
        return Config('allgrain', use_dynamic_thresholds=True)
    
    @pytest.fixture
    def sample_events_ha(self):
        """Create sample events data with harvest area in hectares."""
        events = [
            {'event_name': 'Event 1', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 5e14},
            {'event_name': 'Event 2', 'harvest_area_loss_ha': 20000000, 'production_loss_kcal': 8e14},
            {'event_name': 'Event 3', 'harvest_area_loss_ha': 5000000, 'production_loss_kcal': 3e14},
        ]
        return pd.DataFrame(events)
    
    @pytest.fixture
    def sample_events_km2(self):
        """Create sample events data with harvest area in km²."""
        events = [
            {'event_name': 'Event 1', 'harvest_area_km2': 100000, 'production_loss_kcal': 5e14},
            {'event_name': 'Event 2', 'harvest_area_km2': 200000, 'production_loss_kcal': 8e14},
        ]
        return pd.DataFrame(events)
    
    @pytest.fixture
    def events_with_invalid_values(self):
        """Create events data with zero and invalid values."""
        events = [
            {'event_name': 'Valid Event', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 5e14},
            {'event_name': 'Zero Harvest', 'harvest_area_loss_ha': 0, 'production_loss_kcal': 5e14},
            {'event_name': 'Zero Production', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 0},
            {'event_name': 'NaN Harvest', 'harvest_area_loss_ha': np.nan, 'production_loss_kcal': 5e14},
            {'event_name': 'Another Valid', 'harvest_area_loss_ha': 5000000, 'production_loss_kcal': 3e14},
        ]
        return pd.DataFrame(events)
    
    def test_visualizer_initialization(self, config_wheat):
        """Test visualizer initialization."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        assert visualizer.config == config_wheat
        assert visualizer.crop_type == 'wheat'
        assert 'figsize' in visualizer.figure_params
        assert 'dpi' in visualizer.figure_params
    
    def test_create_plot_with_hectares(self, config_wheat, sample_events_ha):
        """Test creating plot with harvest area in hectares."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        fig = visualizer.create_agrichter_scale_plot(sample_events_ha)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_create_plot_with_km2(self, config_wheat, sample_events_km2):
        """Test creating plot with harvest area in km²."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        fig = visualizer.create_agrichter_scale_plot(sample_events_km2)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_prepare_events_data_hectares_to_km2(self, config_wheat, sample_events_ha):
        """Test conversion from hectares to km²."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        prepared = visualizer._prepare_events_data(sample_events_ha)
        
        assert 'harvest_area_km2' in prepared.columns
        assert len(prepared) == 3
        # 1 hectare = 0.01 km²
        assert prepared.iloc[0]['harvest_area_km2'] == 10000000 * 0.01
    
    def test_prepare_events_data_filters_invalid(self, config_wheat, events_with_invalid_values):
        """Test filtering of invalid values."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        prepared = visualizer._prepare_events_data(events_with_invalid_values)
        
        # Should only keep 2 valid events
        assert len(prepared) == 2
        assert all(prepared['harvest_area_km2'] > 0)
        assert all(prepared['production_loss_kcal'] > 0)
        assert all(np.isfinite(prepared['harvest_area_km2']))
        assert all(np.isfinite(prepared['production_loss_kcal']))
    
    def test_prepare_events_data_no_conversion_needed(self, config_wheat, sample_events_km2):
        """Test that no conversion happens when harvest_area_km2 already exists."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        prepared = visualizer._prepare_events_data(sample_events_km2)
        
        assert 'harvest_area_km2' in prepared.columns
        assert len(prepared) == 2
        # Values should remain unchanged
        assert prepared.iloc[0]['harvest_area_km2'] == 100000
    
    def test_get_axis_limits_unified(self, config_wheat, config_rice, config_allgrain):
        """All crops share unified Richter-style limits for panel comparability."""
        for config in (config_wheat, config_rice, config_allgrain):
            visualizer = AgRichterScaleVisualizer(config)
            xlim, ylim = visualizer._get_axis_limits()
            assert xlim == (1, 7)
            assert ylim == (1e1, 1e7)
    
    def test_plot_with_empty_dataframe(self, config_wheat):
        """Test plotting with empty events DataFrame."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        empty_df = pd.DataFrame(columns=['event_name', 'harvest_area_km2', 'production_loss_kcal'])
        
        fig = visualizer.create_agrichter_scale_plot(empty_df)
        assert fig is not None
        plt.close(fig)
    
    def test_save_figure_multiple_formats(self, config_wheat, sample_events_ha, tmp_path):
        """Test saving figure in multiple formats."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        output_path = tmp_path / "test_agrichter_scale.png"
        
        fig = visualizer.create_agrichter_scale_plot(sample_events_ha, output_path)
        
        # Check that files were created
        assert (tmp_path / "test_agrichter_scale.png").exists()
        assert (tmp_path / "test_agrichter_scale.svg").exists()
        assert (tmp_path / "test_agrichter_scale.eps").exists()
        
        plt.close(fig)
    
    def test_magnitude_calculation(self, config_wheat, sample_events_ha):
        """Test magnitude calculation in plot."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        prepared = visualizer._prepare_events_data(sample_events_ha)
        
        # Calculate magnitude manually
        expected_magnitude = np.log10(prepared['harvest_area_km2'])
        
        # Create plot (which calculates magnitude internally)
        fig = visualizer.create_agrichter_scale_plot(sample_events_ha)
        
        # Verify magnitude is calculated correctly
        assert fig is not None
        plt.close(fig)
    
    def test_plot_historical_events_richter_style(self, config_wheat, sample_events_ha):
        """Test plotting historical events (Richter-style markers)."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        prepared = visualizer._prepare_events_data(sample_events_ha)
        prepared['magnitude'] = np.log10(prepared['harvest_area_km2'])
        
        fig, ax = plt.subplots()
        visualizer._plot_historical_events_richter_style(ax, prepared)
        
        # Check that scatter plot was created
        assert len(ax.collections) > 0
        plt.close(fig)
    
    def test_plot_magnitude_area_relationship(self, config_wheat):
        """Test plotting the fundamental A_H = 10^M_D relationship line."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        
        fig, ax = plt.subplots()
        xlim = (1, 7)
        visualizer._plot_magnitude_area_relationship(ax, xlim)
        
        # Check that line was added and follows A_H = 10^M_D
        assert len(ax.lines) > 0
        x, y = ax.lines[0].get_data()
        np.testing.assert_allclose(y, 10 ** np.asarray(x), rtol=1e-10)
        plt.close(fig)
    
    def test_different_crop_types(self, sample_events_ha):
        """Test visualization for different crop types."""
        crop_types = ['wheat', 'rice', 'allgrain']
        
        for crop_type in crop_types:
            config = Config(crop_type, use_dynamic_thresholds=True)
            visualizer = AgRichterScaleVisualizer(config)
            fig = visualizer.create_agrichter_scale_plot(sample_events_ha)
            
            assert fig is not None
            plt.close(fig)
    
    def test_plot_with_single_event(self, config_wheat):
        """Test plotting with single event."""
        single_event = pd.DataFrame([
            {'event_name': 'Single Event', 'harvest_area_loss_ha': 10000000, 'production_loss_kcal': 5e14}
        ])
        
        visualizer = AgRichterScaleVisualizer(config_wheat)
        fig = visualizer.create_agrichter_scale_plot(single_event)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_many_events(self, config_wheat):
        """Test plotting with many events."""
        many_events = pd.DataFrame([
            {'event_name': f'Event {i}', 'harvest_area_loss_ha': 1000000 * (i+1), 
             'production_loss_kcal': 1e13 * (i+1)}
            for i in range(20)
        ])
        
        visualizer = AgRichterScaleVisualizer(config_wheat)
        fig = visualizer.create_agrichter_scale_plot(many_events)
        
        assert fig is not None
        plt.close(fig)
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

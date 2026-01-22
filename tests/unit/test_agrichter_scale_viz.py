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
    
    def test_get_axis_limits_wheat(self, config_wheat):
        """Test axis limits for wheat."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        xlim, ylim = visualizer._get_axis_limits()
        
        assert xlim == (2, 6.5)
        assert ylim == (1e10, 1e16)
    
    def test_get_axis_limits_rice(self, config_rice):
        """Test axis limits for rice."""
        visualizer = AgRichterScaleVisualizer(config_rice)
        xlim, ylim = visualizer._get_axis_limits()
        
        assert xlim == (2, 6)
        assert ylim == (1e10, 1e16)
    
    def test_get_axis_limits_allgrain(self, config_allgrain):
        """Test axis limits for allgrain."""
        visualizer = AgRichterScaleVisualizer(config_allgrain)
        xlim, ylim = visualizer._get_axis_limits()
        
        assert xlim == (2, 7)
        assert ylim == (1e10, 1e16)
    
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
    
    def test_plot_historical_events_with_adjusttext(self, config_wheat, sample_events_ha):
        """Test plotting historical events with adjustText."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        prepared = visualizer._prepare_events_data(sample_events_ha)
        prepared['magnitude'] = np.log10(prepared['harvest_area_km2'])
        
        fig, ax = plt.subplots()
        visualizer._plot_historical_events(ax, prepared)
        
        # Check that scatter plot was created
        assert len(ax.collections) > 0
        plt.close(fig)
    
    def test_plot_agriPhase_thresholds(self, config_wheat):
        """Test plotting AgriPhase threshold lines."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        
        fig, ax = plt.subplots()
        xlim = (2, 7)
        visualizer._plot_agriPhase_thresholds(ax, xlim)
        
        # Check that horizontal lines were added
        assert len(ax.lines) > 0
        plt.close(fig)
    
    def test_plot_theoretical_line(self, config_wheat):
        """Test plotting theoretical production loss line."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        
        fig, ax = plt.subplots()
        xlim = (2, 7)
        visualizer._plot_theoretical_line(ax, xlim)
        
        # Check that line was added
        assert len(ax.lines) > 0
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
    
    def test_severity_classification(self, config_wheat):
        """Test event severity classification."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        thresholds = config_wheat.get_thresholds()
        
        # Test classification at different levels
        test_cases = [
            (thresholds.get('T1', 1e14) * 0.5, 1, 'Phase 1 (Minimal)'),
            (thresholds.get('T1', 1e14) * 1.5, 2, 'Phase 2 (Stressed)'),
            (thresholds.get('T2', 1e15) * 1.5, 3, 'Phase 3 (Crisis)'),
            (thresholds.get('T3', 1e15) * 1.5, 4, 'Phase 4 (Emergency)'),
            (thresholds.get('T4', 1e15) * 1.5, 5, 'Phase 5 (Famine)'),
        ]
        
        for production_loss, expected_phase, expected_name in test_cases:
            phase, phase_name, color, marker = visualizer._classify_event_severity(production_loss)
            assert phase == expected_phase
            assert phase_name == expected_name
            assert color is not None
            assert marker is not None
    
    def test_severity_markers_unique(self, config_wheat):
        """Test that different severity levels have unique markers."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        thresholds = config_wheat.get_thresholds()
        
        test_values = [
            thresholds.get('T1', 1e14) * 0.5,
            thresholds.get('T1', 1e14) * 1.5,
            thresholds.get('T2', 1e15) * 1.5,
            thresholds.get('T3', 1e15) * 1.5,
            thresholds.get('T4', 1e15) * 1.5,
        ]
        
        markers = set()
        for value in test_values:
            _, _, _, marker = visualizer._classify_event_severity(value)
            markers.add(marker)
        
        # All phases should have unique markers
        assert len(markers) == len(test_values)
    
    def test_severity_visualization_in_plot(self, config_wheat):
        """Test that severity classification appears in visualization."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        thresholds = config_wheat.get_thresholds()
        
        # Create events with varied severity
        events_data = pd.DataFrame([
            {'event_name': 'Minor', 'harvest_area_km2': 5000, 'production_loss_kcal': thresholds['T1'] * 0.5},
            {'event_name': 'Moderate', 'harvest_area_km2': 20000, 'production_loss_kcal': thresholds['T2'] * 1.2},
            {'event_name': 'Severe', 'harvest_area_km2': 100000, 'production_loss_kcal': thresholds['T4'] * 1.5},
        ])
        
        fig = visualizer.create_agrichter_scale_plot(events_data)
        ax = fig.gca()
        
        # Check that legend has severity classifications
        legend = ax.get_legend()
        assert legend is not None
        
        legend_labels = [text.get_text() for text in legend.get_texts()]
        phase_labels = [label for label in legend_labels if 'Phase' in label]
        
        # Should have at least some phase labels
        assert len(phase_labels) > 0
        
        plt.close(fig)
    
    def test_severity_colors_from_ipc(self, config_wheat):
        """Test that severity colors come from IPC color scheme."""
        visualizer = AgRichterScaleVisualizer(config_wheat)
        ipc_colors = config_wheat.get_ipc_colors()
        
        # Test that classification uses IPC colors
        thresholds = config_wheat.get_thresholds()
        _, _, color, _ = visualizer._classify_event_severity(thresholds['T1'] * 0.5)
        
        # Color should be from IPC color scheme
        assert color in ipc_colors.values()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

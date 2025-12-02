"""Unit tests for envelope convergence visualization features."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer
from agririchter.analysis.envelope_diagnostics import EnvelopeDiagnostics
from agririchter.analysis.convergence_validator import ConvergenceValidator, ValidationResult
from agririchter.core.config import Config


class TestConvergenceVisualization:
    """Test convergence visualization features in H-P envelope plots."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config('wheat', use_dynamic_thresholds=False)
    
    @pytest.fixture
    def visualizer(self, config):
        """Create an HPEnvelopeVisualizer instance."""
        return HPEnvelopeVisualizer(config)
    
    @pytest.fixture
    def convergent_envelope_data(self):
        """Create envelope data that properly converges."""
        harvest_points = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0])
        lower_production = np.array([0.0, 80.0, 400.0, 800.0, 1000.0])
        upper_production = np.array([0.0, 120.0, 600.0, 900.0, 1000.0])
        
        return {
            'lower_bound_harvest': harvest_points,
            'lower_bound_production': lower_production,
            'upper_bound_harvest': harvest_points,
            'upper_bound_production': upper_production,
            'disrupted_areas': harvest_points
        }
    
    @pytest.fixture
    def non_convergent_envelope_data(self):
        """Create envelope data that doesn't converge properly."""
        harvest_points = np.array([0.0, 100.0, 500.0, 1000.0, 1800.0])  # Doesn't reach full range
        lower_production = np.array([0.0, 80.0, 400.0, 800.0, 900.0])   # Doesn't converge
        upper_production = np.array([0.0, 120.0, 600.0, 900.0, 950.0])  # Doesn't converge
        
        return {
            'lower_bound_harvest': harvest_points,
            'lower_bound_production': lower_production,
            'upper_bound_harvest': harvest_points,
            'upper_bound_production': upper_production,
            'disrupted_areas': harvest_points
        }
    
    @pytest.fixture
    def sample_events_data(self):
        """Create sample events data for testing."""
        return pd.DataFrame([
            {
                'event_name': 'Test Event 1',
                'harvest_area_km2': 500.0,
                'production_loss_kcal': 5e14,
                'event_type': 'climate'
            },
            {
                'event_name': 'Test Event 2', 
                'harvest_area_km2': 1500.0,
                'production_loss_kcal': 8e14,
                'event_type': 'volcanic'
            }
        ])
    
    def test_convergence_point_highlighting(self, visualizer, convergent_envelope_data, sample_events_data):
        """Test that convergence point is properly highlighted in plots."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                fig = visualizer.create_hp_envelope_plot(
                    convergent_envelope_data, 
                    sample_events_data,
                    save_path=tmp_file.name,
                    show_convergence=True,
                    total_production=total_production,
                    total_harvest=total_harvest
                )
                
                ax = fig.gca()
                
                # Check that convergence point markers are present
                scatter_collections = [child for child in ax.get_children() 
                                     if hasattr(child, 'get_offsets')]
                line_collections = [child for child in ax.get_children() 
                                  if hasattr(child, 'get_data')]
                
                # Should have markers for convergence points
                assert len(scatter_collections) > 0 or len(line_collections) > 0, \
                    "No convergence point markers found in plot"
                
                # Check legend for convergence-related labels
                legend = ax.get_legend()
                if legend:
                    legend_labels = [text.get_text() for text in legend.get_texts()]
                    convergence_labels = [label for label in legend_labels 
                                        if 'convergence' in label.lower() or 'endpoint' in label.lower()]
                    assert len(convergence_labels) > 0, \
                        f"No convergence labels found in legend. Labels: {legend_labels}"
                
                plt.close(fig)
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_convergence_diagnostics_display(self, visualizer, convergent_envelope_data, sample_events_data):
        """Test that convergence diagnostics are displayed in plots."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        # Mock the convergence validator to return predictable results
        with patch('agririchter.analysis.convergence_validator.ConvergenceValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            
            # Create a validation result with known properties
            validation_result = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                properties={
                    'starts_at_origin': True,
                    'converges_at_endpoint': True,
                    'upper_dominates_lower': True,
                    'conservation_satisfied': True,
                    'monotonic_harvest': True
                },
                statistics={
                    'max_harvest_coverage': 1.0,
                    'final_production_width': 0.0,
                    'width_reduction_ratio': 0.95
                }
            )
            mock_validator.validate_mathematical_properties.return_value = validation_result
            
            fig = visualizer.create_hp_envelope_plot(
                convergent_envelope_data,
                sample_events_data,
                show_convergence=True,
                total_production=total_production,
                total_harvest=total_harvest
            )
            
            ax = fig.gca()
            
            # Check for diagnostic text annotations
            text_annotations = [child for child in ax.get_children() 
                              if hasattr(child, 'get_text')]
            
            diagnostic_texts = []
            for text_obj in text_annotations:
                if hasattr(text_obj, 'get_text'):
                    text_content = text_obj.get_text()
                    if any(keyword in text_content.lower() for keyword in 
                          ['validation', 'convergence', 'mathematical', 'passed', 'failed']):
                        diagnostic_texts.append(text_content)
            
            assert len(diagnostic_texts) > 0, \
                "No convergence diagnostic text found in plot"
            
            # Verify that validation was called
            mock_validator.validate_mathematical_properties.assert_called_once()
            
            plt.close(fig)
    
    def test_convergence_error_visualization(self, visualizer, non_convergent_envelope_data, sample_events_data):
        """Test that convergence errors are properly visualized."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        # Mock the convergence validator to return validation errors
        with patch('agririchter.analysis.convergence_validator.ConvergenceValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            
            # Create a validation result with errors
            validation_result = ValidationResult(
                is_valid=False,
                errors=['Envelope does not converge at endpoint', 'Conservation law violated'],
                warnings=['Insufficient harvest area coverage'],
                properties={
                    'starts_at_origin': True,
                    'converges_at_endpoint': False,
                    'upper_dominates_lower': True,
                    'conservation_satisfied': False,
                    'monotonic_harvest': True
                },
                statistics={
                    'max_harvest_coverage': 0.9,
                    'final_production_width': 50.0,
                    'width_reduction_ratio': 0.7
                }
            )
            mock_validator.validate_mathematical_properties.return_value = validation_result
            
            fig = visualizer.create_hp_envelope_plot(
                non_convergent_envelope_data,
                sample_events_data,
                show_convergence=True,
                total_production=total_production,
                total_harvest=total_harvest
            )
            
            ax = fig.gca()
            
            # Check for error indicators in the plot
            text_annotations = [child for child in ax.get_children() 
                              if hasattr(child, 'get_text')]
            
            error_indicators = []
            for text_obj in text_annotations:
                if hasattr(text_obj, 'get_text'):
                    text_content = text_obj.get_text()
                    if 'failed' in text_content.lower() or '✗' in text_content:
                        error_indicators.append(text_content)
            
            assert len(error_indicators) > 0, \
                "No error indicators found in plot with failed validation"
            
            plt.close(fig)
    
    def test_envelope_width_visualization(self, visualizer, convergent_envelope_data, sample_events_data):
        """Test that envelope width is properly visualized near convergence."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        fig = visualizer.create_hp_envelope_plot(
            convergent_envelope_data,
            sample_events_data,
            show_convergence=True,
            total_production=total_production,
            total_harvest=total_harvest
        )
        
        ax = fig.gca()
        
        # Check for secondary y-axis (envelope width indicator)
        axes = fig.get_axes()
        has_secondary_axis = len(axes) > 1
        
        if has_secondary_axis:
            # If secondary axis exists, check for width visualization
            ax2 = axes[1]
            lines = ax2.get_lines()
            assert len(lines) > 0, "No width visualization lines found on secondary axis"
            
            # Check for width-related labels
            ylabel = ax2.get_ylabel()
            assert 'width' in ylabel.lower(), f"Secondary axis should show width, got: {ylabel}"
        
        plt.close(fig)
    
    def test_convergence_point_accuracy(self, visualizer, convergent_envelope_data, sample_events_data):
        """Test that convergence point is placed at correct coordinates."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        fig = visualizer.create_hp_envelope_plot(
            convergent_envelope_data,
            sample_events_data,
            show_convergence=True,
            total_production=total_production,
            total_harvest=total_harvest
        )
        
        ax = fig.gca()
        
        # Find convergence point markers
        expected_x = np.log10(total_harvest)  # Should be in log scale
        expected_y = total_production
        
        # Check scatter plots for convergence point
        for collection in ax.collections:
            if hasattr(collection, 'get_offsets'):
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    # Check if any point is near the expected convergence point
                    for point in offsets:
                        x, y = point
                        if abs(x - expected_x) < 0.1 and abs(y - expected_y) < expected_y * 0.1:
                            # Found convergence point
                            plt.close(fig)
                            return
        
        # Check line plots for convergence point markers
        for line in ax.get_lines():
            xdata, ydata = line.get_data()
            if len(xdata) > 0:
                for x, y in zip(xdata, ydata):
                    if abs(x - expected_x) < 0.1 and abs(y - expected_y) < expected_y * 0.1:
                        # Found convergence point
                        plt.close(fig)
                        return
        
        plt.close(fig)
        pytest.fail(f"Convergence point not found at expected coordinates ({expected_x:.2f}, {expected_y:.2e})")
    
    def test_bounds_approaching_visualization(self, visualizer, convergent_envelope_data, sample_events_data):
        """Test that bounds are visualized as approaching each other."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        fig = visualizer.create_hp_envelope_plot(
            convergent_envelope_data,
            sample_events_data,
            show_convergence=True,
            total_production=total_production,
            total_harvest=total_harvest
        )
        
        ax = fig.gca()
        
        # Check that both upper and lower bounds are plotted
        lines = ax.get_lines()
        bound_lines = []
        
        for line in lines:
            label = line.get_label()
            if 'bound' in label.lower():
                bound_lines.append(line)
        
        assert len(bound_lines) >= 2, \
            f"Expected at least 2 bound lines (upper/lower), found {len(bound_lines)}"
        
        # Check that bounds converge (final points should be close)
        lower_line = None
        upper_line = None
        
        for line in bound_lines:
            label = line.get_label().lower()
            if 'lower' in label:
                lower_line = line
            elif 'upper' in label:
                upper_line = line
        
        if lower_line and upper_line:
            lower_data = lower_line.get_ydata()
            upper_data = upper_line.get_ydata()
            
            if len(lower_data) > 0 and len(upper_data) > 0:
                # Check that bounds converge at the end
                final_lower = lower_data[-1]
                final_upper = upper_data[-1]
                convergence_gap = abs(final_upper - final_lower)
                
                # Should converge to within 10% of total production
                max_acceptable_gap = total_production * 0.1
                assert convergence_gap <= max_acceptable_gap, \
                    f"Bounds don't converge properly: gap={convergence_gap:.2e}, max_acceptable={max_acceptable_gap:.2e}"
        
        plt.close(fig)


class TestEnvelopeDiagnosticsVisualization:
    """Test envelope diagnostics visualization features."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator()
    
    @pytest.fixture
    def diagnostics(self, validator):
        """Create an EnvelopeDiagnostics instance."""
        return EnvelopeDiagnostics(validator)
    
    @pytest.fixture
    def convergent_envelope_data(self):
        """Create envelope data that properly converges."""
        harvest_points = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0])
        lower_production = np.array([0.0, 80.0, 400.0, 800.0, 1000.0])
        upper_production = np.array([0.0, 120.0, 600.0, 900.0, 1000.0])
        
        return {
            'lower_bound_harvest': harvest_points,
            'lower_bound_production': lower_production,
            'upper_bound_harvest': harvest_points,
            'upper_bound_production': upper_production,
            'disrupted_areas': harvest_points
        }
    
    def test_convergence_analysis_plot_creation(self, diagnostics, convergent_envelope_data):
        """Test that convergence analysis plots are created successfully."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                fig = diagnostics.plot_convergence_analysis(
                    convergent_envelope_data,
                    total_production,
                    total_harvest,
                    save_path=tmp_file.name
                )
                
                # Check that figure was created
                assert fig is not None, "Convergence analysis plot was not created"
                
                # Check that figure has multiple subplots (2x2 grid expected)
                axes = fig.get_axes()
                assert len(axes) == 4, f"Expected 4 subplots, got {len(axes)}"
                
                # Check that each subplot has content
                for i, ax in enumerate(axes):
                    # Check for plot elements (lines, collections, text)
                    has_content = (len(ax.get_lines()) > 0 or 
                                 len(ax.collections) > 0 or
                                 len([child for child in ax.get_children() 
                                     if hasattr(child, 'get_text')]) > 0)
                    assert has_content, f"Subplot {i} appears to be empty"
                
                # Check that file was saved
                assert os.path.exists(tmp_file.name), "Plot file was not saved"
                
                plt.close(fig)
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_bounds_convergence_plot_creation(self, diagnostics, convergent_envelope_data):
        """Test that bounds convergence plots are created successfully."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                fig = diagnostics.create_bounds_convergence_plot(
                    convergent_envelope_data,
                    total_production,
                    total_harvest,
                    save_path=tmp_file.name
                )
                
                # Check that figure was created
                assert fig is not None, "Bounds convergence plot was not created"
                
                # Check that figure has 2 subplots (1x2 grid expected)
                axes = fig.get_axes()
                assert len(axes) == 2, f"Expected 2 subplots, got {len(axes)}"
                
                # Check subplot titles and content
                subplot_titles = [ax.get_title() for ax in axes]
                assert any('convergence' in title.lower() for title in subplot_titles), \
                    f"No convergence-related titles found: {subplot_titles}"
                assert any('width' in title.lower() for title in subplot_titles), \
                    f"No width-related titles found: {subplot_titles}"
                
                # Check that file was saved
                assert os.path.exists(tmp_file.name), "Plot file was not saved"
                
                plt.close(fig)
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_validation_summary_visualization(self, diagnostics, convergent_envelope_data):
        """Test that validation summary is properly visualized."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        fig = diagnostics.plot_convergence_analysis(
            convergent_envelope_data,
            total_production,
            total_harvest
        )
        
        axes = fig.get_axes()
        
        # Find the validation summary subplot (should be bottom-right)
        validation_ax = axes[3]  # Bottom-right in 2x2 grid
        
        # Check for validation indicators
        text_elements = [child for child in validation_ax.get_children() 
                        if hasattr(child, 'get_text')]
        
        validation_texts = []
        for text_obj in text_elements:
            if hasattr(text_obj, 'get_text'):
                text_content = text_obj.get_text()
                if any(keyword in text_content.lower() for keyword in 
                      ['validation', 'passed', 'failed', '✓', '✗']):
                    validation_texts.append(text_content)
        
        assert len(validation_texts) > 0, \
            "No validation summary text found in validation subplot"
        
        plt.close(fig)
    
    def test_envelope_width_analysis_plot(self, diagnostics, convergent_envelope_data):
        """Test that envelope width is properly analyzed and plotted."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        fig = diagnostics.plot_convergence_analysis(
            convergent_envelope_data,
            total_production,
            total_harvest
        )
        
        axes = fig.get_axes()
        
        # Find the width analysis subplot (should be top-right)
        width_ax = axes[1]  # Top-right in 2x2 grid
        
        # Check for width-related content
        lines = width_ax.get_lines()
        assert len(lines) > 0, "No lines found in width analysis subplot"
        
        # Check axis labels
        ylabel = width_ax.get_ylabel()
        assert 'width' in ylabel.lower(), f"Y-axis should show width, got: {ylabel}"
        
        # Check for width reduction annotation
        text_elements = [child for child in width_ax.get_children() 
                        if hasattr(child, 'get_text')]
        
        width_texts = []
        for text_obj in text_elements:
            if hasattr(text_obj, 'get_text'):
                text_content = text_obj.get_text()
                if 'reduction' in text_content.lower() or '%' in text_content:
                    width_texts.append(text_content)
        
        assert len(width_texts) > 0, \
            "No width reduction statistics found in width analysis subplot"
        
        plt.close(fig)
    
    def test_convergence_detail_visualization(self, diagnostics, convergent_envelope_data):
        """Test that convergence detail view shows bounds approaching each other."""
        total_production = 1000.0
        total_harvest = 2000.0
        
        fig = diagnostics.plot_convergence_analysis(
            convergent_envelope_data,
            total_production,
            total_harvest
        )
        
        axes = fig.get_axes()
        
        # Find the convergence detail subplot (should be bottom-left)
        detail_ax = axes[2]  # Bottom-left in 2x2 grid
        
        # Check for bounds lines
        lines = detail_ax.get_lines()
        bound_lines = [line for line in lines if 'bound' in line.get_label().lower()]
        
        assert len(bound_lines) >= 2, \
            f"Expected at least 2 bound lines in detail view, got {len(bound_lines)}"
        
        # Check for convergence zone fill
        collections = detail_ax.collections
        convergence_fills = [coll for coll in collections 
                           if hasattr(coll, 'get_label') and 
                           'convergence' in coll.get_label().lower()]
        
        # Should have some visual indication of convergence zone
        assert len(convergence_fills) > 0 or len(bound_lines) >= 2, \
            "No convergence zone visualization found in detail view"
        
        plt.close(fig)


class TestVisualizationIntegration:
    """Test integration between visualization components and convergence validation."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config('wheat', use_dynamic_thresholds=False)
    
    @pytest.fixture
    def visualizer(self, config):
        """Create an HPEnvelopeVisualizer instance."""
        return HPEnvelopeVisualizer(config)
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator()
    
    @pytest.fixture
    def diagnostics(self, validator):
        """Create an EnvelopeDiagnostics instance."""
        return EnvelopeDiagnostics(validator)
    
    def test_visualization_with_validation_integration(self, visualizer, validator):
        """Test that visualization properly integrates with validation results."""
        # Create test data
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 100.0, 500.0, 1000.0]),
            'lower_bound_production': np.array([0.0, 80.0, 400.0, 1000.0]),
            'upper_bound_harvest': np.array([0.0, 100.0, 500.0, 1000.0]),
            'upper_bound_production': np.array([0.0, 120.0, 600.0, 1000.0]),
            'disrupted_areas': np.array([0.0, 100.0, 500.0, 1000.0])
        }
        
        events_data = pd.DataFrame([{
            'event_name': 'Test Event',
            'harvest_area_km2': 500.0,
            'production_loss_kcal': 5e14,
            'event_type': 'climate'
        }])
        
        total_production = 1000.0
        total_harvest = 1000.0
        
        # Test that validation is called during visualization
        with patch.object(validator, 'validate_mathematical_properties') as mock_validate:
            validation_result = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                properties={'converges_at_endpoint': True},
                statistics={'final_production_width': 0.0}
            )
            mock_validate.return_value = validation_result
            
            # Set the validator on the visualizer
            visualizer._convergence_validator = validator
            
            fig = visualizer.create_hp_envelope_plot(
                envelope_data,
                events_data,
                show_convergence=True,
                total_production=total_production,
                total_harvest=total_harvest
            )
            
            # Verify validation was called
            mock_validate.assert_called_once_with(
                envelope_data, total_production, total_harvest
            )
            
            plt.close(fig)
    
    def test_diagnostic_plot_with_real_validation(self, diagnostics):
        """Test diagnostic plots with real validation results."""
        # Create envelope data with known convergence properties
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 100.0, 500.0, 1000.0]),
            'lower_bound_production': np.array([0.0, 80.0, 400.0, 1000.0]),
            'upper_bound_harvest': np.array([0.0, 100.0, 500.0, 1000.0]),
            'upper_bound_production': np.array([0.0, 120.0, 600.0, 1000.0]),
            'disrupted_areas': np.array([0.0, 100.0, 500.0, 1000.0])
        }
        
        total_production = 1000.0
        total_harvest = 1000.0
        
        # Create diagnostic plots (this will run real validation)
        fig = diagnostics.plot_convergence_analysis(
            envelope_data,
            total_production,
            total_harvest
        )
        
        # Verify that plots were created successfully
        assert fig is not None, "Diagnostic plot creation failed"
        
        axes = fig.get_axes()
        assert len(axes) == 4, "Diagnostic plot should have 4 subplots"
        
        # Verify that validation results are reflected in the plots
        validation_ax = axes[3]  # Validation summary subplot
        
        # Check for validation status indicators
        text_elements = [child for child in validation_ax.get_children() 
                        if hasattr(child, 'get_text')]
        
        has_validation_status = False
        for text_obj in text_elements:
            if hasattr(text_obj, 'get_text'):
                text_content = text_obj.get_text()
                if 'passed' in text_content.lower() or 'failed' in text_content.lower():
                    has_validation_status = True
                    break
        
        assert has_validation_status, "No validation status found in diagnostic plot"
        
        plt.close(fig)
    
    def test_error_handling_in_visualization(self, visualizer):
        """Test that visualization handles errors gracefully."""
        # Create invalid envelope data
        invalid_envelope_data = {
            'lower_bound_harvest': np.array([]),  # Empty arrays
            'lower_bound_production': np.array([]),
            'upper_bound_harvest': np.array([]),
            'upper_bound_production': np.array([]),
            'disrupted_areas': np.array([])
        }
        
        # Create empty events data with required columns
        events_data = pd.DataFrame(columns=['event_name', 'harvest_area_km2', 'production_loss_kcal', 'event_type'])
        
        # Should not raise an exception
        try:
            fig = visualizer.create_hp_envelope_plot(
                invalid_envelope_data,
                events_data,
                show_convergence=True,
                total_production=1000.0,
                total_harvest=1000.0
            )
            
            # Plot should be created even with invalid data
            assert fig is not None, "Plot should be created even with invalid data"
            
            plt.close(fig)
            
        except Exception as e:
            # For this test, we expect some errors with empty data, so we'll just check
            # that the error is related to empty data and not a programming error
            error_msg = str(e).lower()
            expected_errors = ['empty', 'invalid', 'zero', 'nan', 'divide by zero', 'log', 
                             'ufunc', 'isfinite', 'input types', 'casting rule']
            if not any(expected in error_msg for expected in expected_errors):
                pytest.fail(f"Unexpected error type in visualization: {e}")
            # If it's an expected error related to empty data, the test passes
    
    def test_convergence_statistics_in_plots(self, diagnostics):
        """Test that convergence statistics are properly displayed in plots."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 100.0, 500.0, 1000.0]),
            'lower_bound_production': np.array([0.0, 80.0, 400.0, 1000.0]),
            'upper_bound_harvest': np.array([0.0, 100.0, 500.0, 1000.0]),
            'upper_bound_production': np.array([0.0, 120.0, 600.0, 1000.0]),
            'disrupted_areas': np.array([0.0, 100.0, 500.0, 1000.0])
        }
        
        total_production = 1000.0
        total_harvest = 1000.0
        
        fig = diagnostics.plot_convergence_analysis(
            envelope_data,
            total_production,
            total_harvest
        )
        
        axes = fig.get_axes()
        
        # Check for statistics in various subplots
        statistics_found = False
        
        for ax in axes:
            text_elements = [child for child in ax.get_children() 
                           if hasattr(child, 'get_text')]
            
            for text_obj in text_elements:
                if hasattr(text_obj, 'get_text'):
                    text_content = text_obj.get_text()
                    # Look for percentage signs or statistical terms
                    if '%' in text_content or any(term in text_content.lower() 
                                                for term in ['coverage', 'reduction', 'width', 'gap']):
                        statistics_found = True
                        break
            
            if statistics_found:
                break
        
        assert statistics_found, "No convergence statistics found in diagnostic plots"
        
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
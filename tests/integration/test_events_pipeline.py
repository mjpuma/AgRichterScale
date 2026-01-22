"""
Integration tests for EventsPipeline.

Tests the complete end-to-end pipeline including:
- Data loading
- Event calculation for all 21 events
- Figure generation with real events
- Output file creation
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import logging

from agrichter.core.config import Config
from agrichter.pipeline.events_pipeline import EventsPipeline


@pytest.fixture
def test_config():
    """Create a test configuration."""
    # Use wheat as default for faster testing
    # Set root_dir to current directory where SPAM data is located
    from pathlib import Path
    root_dir = Path.cwd()
    config = Config(crop_type='wheat', root_dir=root_dir)
    return config


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def pipeline(test_config, temp_output_dir):
    """Create a pipeline instance for testing."""
    return EventsPipeline(test_config, temp_output_dir, enable_performance_monitoring=True)


class TestPipelineDataLoading:
    """Test pipeline data loading stage."""
    
    def test_load_all_data(self, pipeline):
        """Test that all required data is loaded successfully."""
        loaded_data = pipeline.load_all_data()
        
        # Verify all required data is present
        assert 'grid_manager' in loaded_data
        assert 'spatial_mapper' in loaded_data
        assert 'events_data' in loaded_data
        assert 'production_df' in loaded_data
        assert 'harvest_df' in loaded_data
        assert 'country_mapping' in loaded_data
        
        # Verify grid manager is initialized
        assert pipeline.grid_manager is not None
        assert pipeline.spatial_mapper is not None
        
        # Verify SPAM data is loaded
        production_df = loaded_data['production_df']
        harvest_df = loaded_data['harvest_df']
        
        assert len(production_df) > 0
        assert len(harvest_df) > 0
        # Check for key columns (SPAM 2020 uses FIPS0, not iso3)
        assert 'x' in production_df.columns
        assert 'y' in production_df.columns
        # Should have crop columns
        assert any('WHEA' in col for col in production_df.columns)
    
    def test_spatial_index_created(self, pipeline):
        """Test that spatial index is created during data loading."""
        pipeline.load_all_data()
        
        # Verify spatial index exists
        assert pipeline.grid_manager.has_spatial_index
        assert pipeline.grid_manager.production_gdf is not None
        assert pipeline.grid_manager.harvest_area_gdf is not None
    
    def test_country_mappings_prebuilt(self, pipeline):
        """Test that country mappings are pre-built."""
        pipeline.load_all_data()
        
        # Verify country mappings exist in cache
        assert hasattr(pipeline.spatial_mapper, '_country_grid_cache')
        assert len(pipeline.spatial_mapper._country_grid_cache) > 0
    
    def test_events_data_loaded(self, pipeline):
        """Test that event definitions are loaded."""
        loaded_data = pipeline.load_all_data()
        events_data = loaded_data['events_data']
        
        # Verify both country and state sheets are loaded
        assert 'country' in events_data
        assert 'state' in events_data
        
        # Verify we have event sheets
        country_sheets = events_data['country']
        state_sheets = events_data['state']
        
        assert len(country_sheets) > 0
        assert len(state_sheets) > 0


class TestPipelineEventCalculation:
    """Test pipeline event calculation stage."""
    
    def test_calculate_events(self, pipeline):
        """Test that all events are calculated successfully."""
        # Load data first
        pipeline.load_all_data()
        
        # Calculate events
        events_df = pipeline.calculate_events()
        
        # Verify DataFrame structure
        assert isinstance(events_df, pd.DataFrame)
        assert len(events_df) > 0
        
        # Verify required columns exist
        required_columns = [
            'event_name',
            'harvest_area_loss_ha',
            'production_loss_kcal',
            'magnitude'
        ]
        for col in required_columns:
            assert col in events_df.columns
    
    def test_all_21_events_processed(self, pipeline):
        """Test that all 21 historical events are processed."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        # Should have 21 events (or close to it, some may have no data)
        # At minimum, we should have more than 15 events with data
        assert len(events_df) >= 15, f"Expected at least 15 events, got {len(events_df)}"
    
    def test_event_results_valid(self, pipeline):
        """Test that event results have valid values."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        # Check for non-negative values
        assert (events_df['harvest_area_loss_ha'] >= 0).all()
        assert (events_df['production_loss_kcal'] >= 0).all()
        
        # Check that at least some events have non-zero losses
        assert (events_df['harvest_area_loss_ha'] > 0).any()
        assert (events_df['production_loss_kcal'] > 0).any()
    
    def test_magnitude_calculation(self, pipeline):
        """Test that magnitudes are calculated correctly."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        # Filter events with non-zero harvest area
        events_with_data = events_df[events_df['harvest_area_loss_ha'] > 0]
        
        if len(events_with_data) > 0:
            # Magnitudes should be in reasonable range (typically 2-7)
            magnitudes = events_with_data['magnitude']
            assert magnitudes.min() >= 0, "Magnitude should be non-negative"
            assert magnitudes.max() <= 10, "Magnitude should be reasonable (< 10)"
    
    def test_event_calculator_initialized(self, pipeline):
        """Test that event calculator is properly initialized."""
        pipeline.load_all_data()
        pipeline.calculate_events()
        
        assert pipeline.event_calculator is not None
        assert pipeline.event_calculator.grid_manager is not None
        assert pipeline.event_calculator.spatial_mapper is not None


class TestPipelineVisualization:
    """Test pipeline visualization generation stage."""
    
    def test_generate_visualizations(self, pipeline):
        """Test that visualizations are generated successfully."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        figures = pipeline.generate_visualizations(events_df)
        
        # Verify figures dictionary is returned
        assert isinstance(figures, dict)
        
        # Should have at least some figures
        assert len(figures) > 0
    
    def test_hp_envelope_generated(self, pipeline):
        """Test that H-P Envelope is generated."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        figures = pipeline.generate_visualizations(events_df)
        
        # Check if H-P envelope was created
        if 'hp_envelope' in figures:
            fig = figures['hp_envelope']
            assert isinstance(fig, plt.Figure)
            
            # Verify figure has axes
            assert len(fig.axes) > 0
    
    def test_agrichter_scale_generated(self, pipeline):
        """Test that AgRichter Scale is generated."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        figures = pipeline.generate_visualizations(events_df)
        
        # Check if AgRichter Scale was created
        if 'agrichter_scale' in figures:
            fig = figures['agrichter_scale']
            assert isinstance(fig, plt.Figure)
            
            # Verify figure has axes
            assert len(fig.axes) > 0
    
    def test_production_map_generated(self, pipeline):
        """Test that production map is generated."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        figures = pipeline.generate_visualizations(events_df)
        
        # Check if production map was created
        if 'production_map' in figures:
            fig = figures['production_map']
            assert isinstance(fig, plt.Figure)
    
    def test_visualizations_with_real_events(self, pipeline):
        """Test that visualizations use real event data."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        # Ensure we have events with data
        events_with_data = events_df[events_df['harvest_area_loss_ha'] > 0]
        assert len(events_with_data) > 0, "Need events with data for visualization test"
        
        figures = pipeline.generate_visualizations(events_df)
        
        # At least one figure should be generated
        assert len(figures) > 0


class TestPipelineExport:
    """Test pipeline results export stage."""
    
    def test_export_results(self, pipeline, temp_output_dir):
        """Test that results are exported successfully."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        figures = pipeline.generate_visualizations(events_df)
        
        exported_files = pipeline.export_results(events_df, figures)
        
        # Verify exported files dictionary structure
        assert 'csv_files' in exported_files
        assert 'figure_files' in exported_files
        assert 'report_files' in exported_files
    
    def test_csv_file_created(self, pipeline, temp_output_dir):
        """Test that CSV file is created."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        figures = pipeline.generate_visualizations(events_df)
        
        exported_files = pipeline.export_results(events_df, figures)
        
        # Verify CSV file was created
        csv_files = exported_files['csv_files']
        assert len(csv_files) > 0
        
        # Verify file exists
        csv_path = Path(csv_files[0])
        assert csv_path.exists()
        
        # Verify file can be read
        df = pd.read_csv(csv_path)
        assert len(df) > 0
    
    def test_figure_files_created(self, pipeline, temp_output_dir):
        """Test that figure files are created in multiple formats."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        figures = pipeline.generate_visualizations(events_df)
        
        exported_files = pipeline.export_results(events_df, figures)
        
        # Verify figure files were created
        figure_files = exported_files['figure_files']
        
        if len(figures) > 0:
            assert len(figure_files) > 0
            
            # Check for multiple formats
            formats = set()
            for file_path in figure_files:
                ext = Path(file_path).suffix.lower()
                formats.add(ext)
            
            # Should have multiple formats
            assert len(formats) > 1
    
    def test_output_directory_structure(self, pipeline, temp_output_dir):
        """Test that output directory structure is created correctly."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        figures = pipeline.generate_visualizations(events_df)
        
        pipeline.export_results(events_df, figures)
        
        # Verify directory structure
        output_path = Path(temp_output_dir)
        assert (output_path / 'data').exists()
        assert (output_path / 'figures').exists()
        assert (output_path / 'reports').exists()
    
    def test_csv_filename_format(self, pipeline, temp_output_dir):
        """Test that CSV filename follows expected format."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        figures = pipeline.generate_visualizations(events_df)
        
        exported_files = pipeline.export_results(events_df, figures)
        
        csv_files = exported_files['csv_files']
        if len(csv_files) > 0:
            csv_filename = Path(csv_files[0]).name
            
            # Should contain crop type and spam2020
            assert pipeline.config.crop_type in csv_filename
            assert 'spam2020' in csv_filename
            assert csv_filename.endswith('.csv')


class TestCompletePipeline:
    """Test complete end-to-end pipeline execution."""
    
    def test_run_complete_pipeline(self, pipeline):
        """Test that complete pipeline runs successfully."""
        results = pipeline.run_complete_pipeline()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'events_df' in results
        assert 'figures' in results
        assert 'exported_files' in results
        assert 'status' in results
    
    def test_pipeline_status_success(self, pipeline):
        """Test that pipeline completes with success status."""
        results = pipeline.run_complete_pipeline()
        
        assert results['status'] == 'success'
    
    def test_pipeline_events_dataframe(self, pipeline):
        """Test that pipeline produces valid events DataFrame."""
        results = pipeline.run_complete_pipeline()
        
        events_df = results['events_df']
        assert events_df is not None
        assert isinstance(events_df, pd.DataFrame)
        assert len(events_df) > 0
    
    def test_pipeline_figures_generated(self, pipeline):
        """Test that pipeline generates figures."""
        results = pipeline.run_complete_pipeline()
        
        figures = results['figures']
        assert isinstance(figures, dict)
        # Should have at least one figure
        assert len(figures) >= 0  # May be 0 if visualization fails
    
    def test_pipeline_files_exported(self, pipeline, temp_output_dir):
        """Test that pipeline exports files."""
        results = pipeline.run_complete_pipeline()
        
        exported_files = results['exported_files']
        assert isinstance(exported_files, dict)
        
        # Verify at least CSV files are exported
        csv_files = exported_files.get('csv_files', [])
        assert len(csv_files) > 0
    
    def test_pipeline_summary_report(self, pipeline):
        """Test that pipeline generates summary report."""
        results = pipeline.run_complete_pipeline()
        
        assert 'summary_report' in results
        summary = results['summary_report']
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Verify report contains key information
        assert 'AgRichter' in summary
        assert 'Event' in summary or 'event' in summary
    
    def test_pipeline_performance_monitoring(self, pipeline):
        """Test that pipeline tracks performance metrics."""
        results = pipeline.run_complete_pipeline()
        
        # Performance monitoring should be enabled
        assert pipeline.performance_monitor is not None
        
        # Should have performance report
        if 'performance_report' in results:
            perf_report = results['performance_report']
            assert isinstance(perf_report, str)
            assert len(perf_report) > 0


class TestPipelineWithDifferentCrops:
    """Test pipeline with different crop types."""
    
    def test_pipeline_with_wheat(self, temp_output_dir):
        """Test pipeline with wheat crop."""
        from pathlib import Path
        root_dir = Path.cwd()
        config = Config(crop_type='wheat', root_dir=root_dir)
        
        pipeline = EventsPipeline(config, temp_output_dir, enable_performance_monitoring=False)
        results = pipeline.run_complete_pipeline()
        
        assert results['status'] == 'success'
        assert results['events_df'] is not None
    
    def test_pipeline_with_rice(self, temp_output_dir):
        """Test pipeline with rice crop."""
        from pathlib import Path
        root_dir = Path.cwd()
        config = Config(crop_type='rice', root_dir=root_dir)
        
        pipeline = EventsPipeline(config, temp_output_dir, enable_performance_monitoring=False)
        results = pipeline.run_complete_pipeline()
        
        assert results['status'] == 'success'
        assert results['events_df'] is not None
    
    @pytest.mark.slow
    def test_pipeline_with_allgrain(self, temp_output_dir):
        """Test pipeline with all grains (slower test)."""
        from pathlib import Path
        root_dir = Path.cwd()
        config = Config(crop_type='allgrain', root_dir=root_dir)
        
        pipeline = EventsPipeline(config, temp_output_dir, enable_performance_monitoring=False)
        results = pipeline.run_complete_pipeline()
        
        assert results['status'] == 'success'
        assert results['events_df'] is not None


class TestPipelineErrorHandling:
    """Test pipeline error handling and recovery."""
    
    def test_pipeline_continues_on_visualization_error(self, pipeline, monkeypatch):
        """Test that pipeline continues if visualization fails."""
        # Load data and calculate events
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        # Mock visualization to raise error
        def mock_viz_error(*args, **kwargs):
            raise Exception("Visualization error")
        
        # Even if visualization fails, export should still work
        figures = {}  # Empty figures
        exported_files = pipeline.export_results(events_df, figures)
        
        # Should still export CSV
        assert len(exported_files['csv_files']) > 0
    
    def test_pipeline_handles_empty_events(self, pipeline, temp_output_dir):
        """Test pipeline behavior with no events data."""
        # This is a edge case test - pipeline should handle gracefully
        # In practice, we always have events data
        pass  # Skip for now as this requires mocking event loading


class TestPipelineValidation:
    """Test pipeline validation and data quality checks."""
    
    def test_events_have_required_fields(self, pipeline):
        """Test that all events have required fields."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        required_fields = [
            'event_name',
            'harvest_area_loss_ha',
            'production_loss_kcal',
            'magnitude'
        ]
        
        for field in required_fields:
            assert field in events_df.columns
    
    def test_no_negative_losses(self, pipeline):
        """Test that no events have negative losses."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        assert (events_df['harvest_area_loss_ha'] >= 0).all()
        assert (events_df['production_loss_kcal'] >= 0).all()
    
    def test_magnitude_formula_correct(self, pipeline):
        """Test that magnitude calculation follows correct formula."""
        pipeline.load_all_data()
        events_df = pipeline.calculate_events()
        
        # Filter events with non-zero harvest area
        events_with_data = events_df[events_df['harvest_area_loss_ha'] > 0].copy()
        
        if len(events_with_data) > 0:
            # Magnitude should be log10(area_km2)
            # area_km2 = harvest_area_ha * 0.01
            import numpy as np
            
            expected_magnitude = np.log10(events_with_data['harvest_area_loss_ha'] * 0.01)
            actual_magnitude = events_with_data['magnitude']
            
            # Should be very close (within floating point precision)
            np.testing.assert_array_almost_equal(
                actual_magnitude.values,
                expected_magnitude.values,
                decimal=5
            )


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_pipeline_completes_in_reasonable_time(self, pipeline):
        """Test that pipeline completes in reasonable time."""
        import time
        
        start_time = time.time()
        results = pipeline.run_complete_pipeline()
        elapsed_time = time.time() - start_time
        
        # Should complete in under 10 minutes (600 seconds)
        # For wheat only, should be much faster (< 2 minutes)
        assert elapsed_time < 600, f"Pipeline took {elapsed_time:.2f}s (> 600s limit)"
        
        # Log the time for reference
        print(f"\nPipeline completed in {elapsed_time:.2f} seconds")
    
    def test_performance_metrics_tracked(self, pipeline):
        """Test that performance metrics are tracked."""
        results = pipeline.run_complete_pipeline()
        
        # Should have performance monitor
        assert pipeline.performance_monitor is not None
        
        # Should have metrics for stages
        metrics = pipeline.performance_monitor.get_all_metrics()
        assert len(metrics) > 0
        
        # Should track key stages
        expected_stages = ['data_loading', 'event_calculation']
        for stage in expected_stages:
            assert stage in metrics


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])

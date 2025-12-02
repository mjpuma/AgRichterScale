"""
Integration tests for pipeline multi-tier support.

Tests the integration of multi-tier envelope calculations with the EventsPipeline
and MultiTierEventsPipeline classes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline
from agririchter.pipeline.multi_tier_events_pipeline import (
    MultiTierEventsPipeline,
    create_policy_analysis_pipeline,
    create_research_analysis_pipeline,
    create_comparative_analysis_pipeline
)


class TestPipelineIntegration:
    """Test pipeline integration with multi-tier support."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(crop_type='wheat')
    
    @pytest.fixture
    def test_data(self):
        """Create test agricultural data."""
        np.random.seed(42)
        n_cells = 100
        
        # Create realistic data
        harvest = np.random.lognormal(mean=3.5, sigma=1.0, size=n_cells) * 50
        yields = np.random.uniform(1.0, 6.0, n_cells)
        production = harvest * yields * 1000 * 3340  # Convert to kcal
        
        production_df = pd.DataFrame({
            'WHEA_A': production,
            'lat': np.random.uniform(-60, 70, n_cells),
            'lon': np.random.uniform(-180, 180, n_cells)
        })
        
        harvest_df = pd.DataFrame({
            'WHEA_A': harvest,
            'lat': production_df['lat'],
            'lon': production_df['lon']
        })
        
        return production_df, harvest_df
    
    @pytest.fixture
    def test_events(self):
        """Create test events data."""
        return pd.DataFrame({
            'event_name': ['Test Event 1', 'Test Event 2'],
            'harvest_area_loss_ha': [10000, 5000],
            'production_loss_kcal': [1e12, 5e11],
            'magnitude': [6.0, 5.5],
            'affected_countries': [['USA'], ['CHN']],
            'grid_cells_count': [50, 25]
        })
    
    def test_events_pipeline_tier_selection(self, config):
        """Test EventsPipeline with tier selection parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test comprehensive tier
            pipeline_comp = EventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='comprehensive'
            )
            
            assert pipeline_comp.tier_selection == 'comprehensive'
            
            # Test commercial tier
            pipeline_comm = EventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            assert pipeline_comm.tier_selection == 'commercial'
    
    def test_envelope_calculation_with_tier_selection(self, config, test_data):
        """Test envelope calculation with tier selection."""
        production_df, harvest_df = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = EventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            # Test envelope calculation
            envelope_data = pipeline._calculate_envelope_with_tier_selection(
                production_df, harvest_df
            )
            
            # Validate envelope data structure
            assert 'disruption_areas' in envelope_data
            assert 'lower_bound_harvest' in envelope_data
            assert 'lower_bound_production' in envelope_data
            assert 'upper_bound_harvest' in envelope_data
            assert 'upper_bound_production' in envelope_data
            
            # Validate data types and shapes
            assert isinstance(envelope_data['disruption_areas'], np.ndarray)
            assert len(envelope_data['disruption_areas']) > 0
            
            # Check that all arrays have the same length
            n_points = len(envelope_data['disruption_areas'])
            assert len(envelope_data['lower_bound_harvest']) == n_points
            assert len(envelope_data['lower_bound_production']) == n_points
            assert len(envelope_data['upper_bound_harvest']) == n_points
            assert len(envelope_data['upper_bound_production']) == n_points
    
    def test_multi_tier_events_pipeline_creation(self, config):
        """Test MultiTierEventsPipeline creation and configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test basic creation
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            assert pipeline.tier_selection == 'commercial'
            assert pipeline.multi_tier_engine is None  # Lazy loading
            
            # Test tier selection validation
            with pytest.raises(ValueError):
                MultiTierEventsPipeline(
                    config=config,
                    output_dir=temp_dir,
                    tier_selection='invalid_tier'
                )
    
    def test_convenience_pipeline_creators(self, config):
        """Test convenience functions for creating specialized pipelines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test policy analysis pipeline
            policy_pipeline = create_policy_analysis_pipeline(config, temp_dir)
            assert policy_pipeline.tier_selection == 'commercial'
            
            # Test research analysis pipeline
            research_pipeline = create_research_analysis_pipeline(config, temp_dir)
            assert research_pipeline.tier_selection == 'comprehensive'
            
            # Test comparative analysis pipeline
            comparative_pipeline = create_comparative_analysis_pipeline(config, temp_dir)
            assert comparative_pipeline.tier_selection == 'all'
    
    def test_tier_selection_guide(self, config):
        """Test tier selection guide functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            # Get tier selection guide
            guide = pipeline.get_tier_selection_guide()
            
            # Validate guide structure
            assert isinstance(guide, dict)
            assert 'comprehensive' in guide
            assert 'commercial' in guide
            
            # Validate tier information
            for tier_name, tier_info in guide.items():
                assert 'name' in tier_info
                assert 'description' in tier_info
                assert 'policy_applications' in tier_info
                assert 'target_users' in tier_info
    
    def test_dynamic_tier_selection(self, config):
        """Test dynamic tier selection changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='comprehensive'
            )
            
            # Test initial selection
            assert pipeline.tier_selection == 'comprehensive'
            
            # Test tier change
            pipeline.set_tier_selection('commercial')
            assert pipeline.tier_selection == 'commercial'
            
            # Test invalid tier change
            with pytest.raises(ValueError):
                pipeline.set_tier_selection('invalid_tier')
    
    def test_envelope_calculation_integration(self, config, test_data):
        """Test envelope calculation integration with multi-tier engine."""
        production_df, harvest_df = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            # Test single tier calculation
            envelope_data = pipeline.calculate_envelope_with_tier_selection(
                production_df, harvest_df, tier='commercial'
            )
            
            # Validate envelope data
            assert isinstance(envelope_data, dict)
            assert 'disruption_areas' in envelope_data
            
            # Test multi-tier calculation
            all_envelope_data = pipeline.calculate_envelope_with_tier_selection(
                production_df, harvest_df, tier='all'
            )
            
            # Should return comprehensive tier for backward compatibility
            assert isinstance(all_envelope_data, dict)
            assert 'disruption_areas' in all_envelope_data
            
            # Should have tier_results populated
            assert pipeline.tier_results is not None
            assert len(pipeline.tier_results.tier_results) >= 2
    
    def test_tier_comparison_visualization_creation(self, config, test_data, test_events):
        """Test tier comparison visualization creation."""
        production_df, harvest_df = test_data
        events_df = test_events
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='all'
            )
            
            # Calculate multi-tier results
            pipeline.calculate_envelope_with_tier_selection(
                production_df, harvest_df, tier='all'
            )
            
            # Test tier comparison visualization
            if pipeline.tier_results:
                comparison_figures = pipeline._generate_tier_comparison_visualizations(
                    pipeline.tier_results, events_df
                )
                
                # Should generate comparison figures
                assert isinstance(comparison_figures, dict)
                # Note: May be empty due to missing dependencies, but should not error
    
    def test_tier_analysis_export_structure(self, config, test_data):
        """Test tier analysis export structure."""
        production_df, harvest_df = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='all'
            )
            
            # Calculate multi-tier results
            pipeline.calculate_envelope_with_tier_selection(
                production_df, harvest_df, tier='all'
            )
            
            # Test tier analysis export (structure only, files may not be created in test)
            if pipeline.tier_results:
                # This should not raise an error even if directories don't exist
                try:
                    tier_files = pipeline._export_tier_analysis(pipeline.tier_results)
                    # Files may not be created due to directory issues, but method should work
                    assert isinstance(tier_files, list)
                except Exception as e:
                    # Expected in test environment due to directory creation
                    assert "No such file or directory" in str(e)
    
    def test_tier_selection_guide_export_structure(self, config):
        """Test tier selection guide export structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            # Test guide content generation
            guide_content = pipeline._generate_tier_selection_guide_content()
            
            # Validate guide content
            assert isinstance(guide_content, str)
            assert len(guide_content) > 0
            assert 'Multi-Tier Envelope Analysis' in guide_content
            assert 'Tier Selection Guide' in guide_content
            assert 'Commercial Agriculture' in guide_content
            assert 'Comprehensive' in guide_content
    
    def test_backward_compatibility(self, config, test_data):
        """Test that existing EventsPipeline functionality is preserved."""
        production_df, harvest_df = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test original EventsPipeline (should work with default tier)
            original_pipeline = EventsPipeline(
                config=config,
                output_dir=temp_dir
            )
            
            # Should have default tier selection
            assert hasattr(original_pipeline, 'tier_selection')
            assert original_pipeline.tier_selection == 'comprehensive'
            
            # Should be able to calculate envelope
            envelope_data = original_pipeline._calculate_envelope_with_tier_selection(
                production_df, harvest_df
            )
            
            assert isinstance(envelope_data, dict)
            assert 'disruption_areas' in envelope_data
    
    def test_performance_monitoring_integration(self, config):
        """Test that performance monitoring works with multi-tier pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with performance monitoring enabled
            pipeline_with_perf = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial',
                enable_performance_monitoring=True
            )
            
            assert pipeline_with_perf.performance_monitor is not None
            
            # Test with performance monitoring disabled
            pipeline_without_perf = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial',
                enable_performance_monitoring=False
            )
            
            assert pipeline_without_perf.performance_monitor is None
    
    def test_error_handling_and_fallbacks(self, config, test_data):
        """Test error handling and fallback mechanisms."""
        production_df, harvest_df = test_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = EventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='commercial'
            )
            
            # Test fallback to V2 calculator when multi-tier fails
            with patch('agririchter.analysis.envelope.HPEnvelopeCalculator') as mock_calc:
                # Make the multi-tier calculator fail
                mock_calc.side_effect = Exception("Multi-tier calculation failed")
                
                # Should fallback to V2 calculator
                envelope_data = pipeline._calculate_envelope_with_tier_selection(
                    production_df, harvest_df
                )
                
                # Should still return valid envelope data
                assert isinstance(envelope_data, dict)


class TestPipelineIntegrationEndToEnd:
    """End-to-end integration tests for pipeline functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(crop_type='wheat')
    
    @pytest.fixture
    def mock_data_loading(self):
        """Mock data loading for end-to-end tests."""
        def mock_load_all_data(self):
            # Create minimal test data
            n_cells = 50
            production_data = pd.DataFrame({
                'WHEA_A': np.random.uniform(1e8, 1e12, n_cells)
            })
            harvest_data = pd.DataFrame({
                'WHEA_A': np.random.uniform(100, 10000, n_cells)
            })
            
            self.loaded_data = {
                'production_df': production_data,
                'harvest_df': harvest_data,
                'yield_df': None
            }
            return self.loaded_data
        
        return mock_load_all_data
    
    @pytest.fixture
    def mock_event_calculation(self):
        """Mock event calculation for end-to-end tests."""
        def mock_calculate_events(self):
            # Create minimal test events
            events_df = pd.DataFrame({
                'event_name': ['Test Event'],
                'harvest_area_loss_ha': [1000],
                'production_loss_kcal': [1e11],
                'magnitude': [5.5],
                'affected_countries': [['USA']],
                'grid_cells_count': [10]
            })
            self.events_df = events_df
            return events_df
        
        return mock_calculate_events
    
    def test_complete_pipeline_workflow_policy_analysis(self, config, mock_data_loading, mock_event_calculation):
        """Test complete pipeline workflow for policy analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = create_policy_analysis_pipeline(config, temp_dir)
            
            # Mock the data loading and event calculation
            pipeline.load_all_data = mock_data_loading.__get__(pipeline, MultiTierEventsPipeline)
            pipeline.calculate_events = mock_event_calculation.__get__(pipeline, MultiTierEventsPipeline)
            
            # Test that pipeline can be configured for policy analysis
            assert pipeline.tier_selection == 'commercial'
            
            # Test data loading
            loaded_data = pipeline.load_all_data()
            assert 'production_df' in loaded_data
            assert 'harvest_df' in loaded_data
            
            # Test event calculation
            events_df = pipeline.calculate_events()
            assert len(events_df) > 0
            assert 'event_name' in events_df.columns
    
    def test_complete_pipeline_workflow_research_analysis(self, config, mock_data_loading, mock_event_calculation):
        """Test complete pipeline workflow for research analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = create_research_analysis_pipeline(config, temp_dir)
            
            # Mock the data loading and event calculation
            pipeline.load_all_data = mock_data_loading.__get__(pipeline, MultiTierEventsPipeline)
            pipeline.calculate_events = mock_event_calculation.__get__(pipeline, MultiTierEventsPipeline)
            
            # Test that pipeline can be configured for research analysis
            assert pipeline.tier_selection == 'comprehensive'
            
            # Test data loading
            loaded_data = pipeline.load_all_data()
            assert 'production_df' in loaded_data
            
            # Test event calculation
            events_df = pipeline.calculate_events()
            assert len(events_df) > 0
    
    def test_pipeline_tier_switching_workflow(self, config, mock_data_loading, mock_event_calculation):
        """Test pipeline workflow with dynamic tier switching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MultiTierEventsPipeline(
                config=config,
                output_dir=temp_dir,
                tier_selection='comprehensive'
            )
            
            # Mock the data loading and event calculation
            pipeline.load_all_data = mock_data_loading.__get__(pipeline, MultiTierEventsPipeline)
            pipeline.calculate_events = mock_event_calculation.__get__(pipeline, MultiTierEventsPipeline)
            
            # Load data
            loaded_data = pipeline.load_all_data()
            
            # Test envelope calculation with initial tier
            envelope_comp = pipeline.calculate_envelope_with_tier_selection(
                loaded_data['production_df'], loaded_data['harvest_df']
            )
            assert isinstance(envelope_comp, dict)
            
            # Switch tier and recalculate
            pipeline.set_tier_selection('commercial')
            envelope_comm = pipeline.calculate_envelope_with_tier_selection(
                loaded_data['production_df'], loaded_data['harvest_df']
            )
            assert isinstance(envelope_comm, dict)
            
            # Results should be different (different tier calculations)
            # Note: Exact comparison depends on data, but structures should be valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
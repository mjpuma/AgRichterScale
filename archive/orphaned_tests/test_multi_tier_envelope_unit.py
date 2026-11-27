"""
Unit tests for MultiTierEnvelopeEngine components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from agrichter.core.config import Config
from agrichter.analysis.multi_tier_envelope import (
    MultiTierEnvelopeEngine, 
    MultiTierResults, 
    TierConfiguration,
    TIER_CONFIGURATIONS
)
from agrichter.validation.spam_data_filter import SPAMDataFilter


class TestMultiTierEnvelopeEngine:
    """Unit tests for MultiTierEnvelopeEngine."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return Config(crop_type='wheat')
    
    @pytest.fixture
    def spam_filter(self):
        """Mock SPAM filter."""
        mock_filter = Mock(spec=SPAMDataFilter)
        mock_filter.filter_crop_data.return_value = (
            np.array([True] * 100),  # filter_mask
            {'retention_rate': 85.0, 'cells_filtered': 15}  # filter_stats
        )
        return mock_filter
    
    @pytest.fixture
    def engine(self, config, spam_filter):
        """MultiTierEnvelopeEngine instance."""
        return MultiTierEnvelopeEngine(config, spam_filter)
    
    @pytest.fixture
    def sample_data(self):
        """Sample production and harvest data."""
        n_cells = 100
        production_data = pd.DataFrame({
            'WHEA_A': np.random.exponential(1000, n_cells),
            'x': np.random.uniform(-180, 180, n_cells),
            'y': np.random.uniform(-90, 90, n_cells)
        })
        harvest_data = pd.DataFrame({
            'WHEA_A': np.random.exponential(100, n_cells),
            'x': production_data['x'],
            'y': production_data['y']
        })
        return production_data, harvest_data
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.crop_type == 'wheat'
        assert engine.spam_filter is not None
        assert engine.base_calculator is not None
        assert engine.convergence_validator is not None
        assert len(engine.tier_configs) >= 2
        assert 'comprehensive' in engine.tier_configs
        assert 'commercial' in engine.tier_configs
    
    def test_tier_configurations(self):
        """Test tier configuration structure."""
        assert len(TIER_CONFIGURATIONS) >= 2
        
        for tier_name, config in TIER_CONFIGURATIONS.items():
            assert isinstance(config, TierConfiguration)
            assert config.name
            assert config.description
            assert 0 <= config.yield_percentile_min <= 100
            assert 0 <= config.yield_percentile_max <= 100
            assert config.yield_percentile_min <= config.yield_percentile_max
            assert isinstance(config.policy_applications, list)
            assert isinstance(config.target_users, list)
    
    def test_get_tier_info(self, engine):
        """Test tier information retrieval."""
        tier_info = engine.get_tier_info()
        
        assert isinstance(tier_info, dict)
        assert len(tier_info) >= 2
        
        for tier_name, info in tier_info.items():
            assert 'name' in info
            assert 'description' in info
            assert 'yield_percentile_range' in info
            assert 'policy_applications' in info
            assert 'target_users' in info
            assert 'expected_width_reduction' in info
    
    def test_get_crop_columns(self, engine, sample_data):
        """Test crop column identification."""
        production_df, _ = sample_data
        
        crop_columns = engine._get_crop_columns(production_df)
        
        assert isinstance(crop_columns, list)
        assert len(crop_columns) > 0
        assert 'WHEA_A' in crop_columns
        
        # Test with different crop types
        engine.config.crop_type = 'maize'
        production_df['MAIZ_A'] = production_df['WHEA_A']
        
        maize_columns = engine._get_crop_columns(production_df)
        assert 'MAIZ_A' in maize_columns
        assert 'WHEA_A' not in maize_columns
    
    def test_aggregate_crop_data(self, engine, sample_data):
        """Test crop data aggregation."""
        production_df, harvest_df = sample_data
        crop_columns = ['WHEA_A']
        
        total_production, total_harvest = engine._aggregate_crop_data(
            production_df, harvest_df, crop_columns
        )
        
        assert isinstance(total_production, np.ndarray)
        assert isinstance(total_harvest, np.ndarray)
        assert len(total_production) == len(production_df)
        assert len(total_harvest) == len(harvest_df)
        assert np.all(total_production >= 0)
        assert np.all(total_harvest >= 0)
    
    def test_prepare_base_data(self, engine, sample_data):
        """Test base data preparation."""
        production_df, harvest_df = sample_data
        
        base_data = engine._prepare_base_data(production_df, harvest_df)
        
        assert isinstance(base_data, dict)
        assert 'production' in base_data
        assert 'harvest' in base_data
        assert 'yields' in base_data
        assert 'filter_stats' in base_data
        
        # Check data consistency
        assert len(base_data['production']) == len(base_data['harvest'])
        assert len(base_data['production']) == len(base_data['yields'])
        
        # Check yields calculation
        expected_yields = np.divide(
            base_data['production'],
            base_data['harvest'],
            out=np.zeros_like(base_data['production']),
            where=(base_data['harvest'] > 0)
        )
        np.testing.assert_array_equal(base_data['yields'], expected_yields)
    
    def test_apply_tier_filtering(self, engine):
        """Test tier-specific filtering."""
        # Create test data with known yield distribution
        n_cells = 1000
        yields = np.random.exponential(2.0, n_cells)
        base_data = {
            'production': yields * 100,  # production = yield * area
            'harvest': np.full(n_cells, 100),  # constant area
            'yields': yields
        }
        
        # Test comprehensive tier (should include all data)
        comprehensive_config = TIER_CONFIGURATIONS['comprehensive']
        comprehensive_data = engine._apply_tier_filtering(base_data, comprehensive_config)
        
        assert len(comprehensive_data['production']) == n_cells
        
        # Test commercial tier (should exclude bottom 20%)
        commercial_config = TIER_CONFIGURATIONS['commercial']
        commercial_data = engine._apply_tier_filtering(base_data, commercial_config)
        
        # Should have fewer cells than comprehensive
        assert len(commercial_data['production']) < len(comprehensive_data['production'])
        
        # Should exclude low-yield cells
        min_yield_commercial = np.min(commercial_data['yields'])
        percentile_20 = np.percentile(yields, 20)
        assert min_yield_commercial >= percentile_20 * 0.99  # Allow small numerical error
    
    def test_calculate_width_analysis(self, engine):
        """Test width reduction calculation."""
        # Create mock envelope data
        from agrichter.analysis.envelope import EnvelopeData
        
        # Comprehensive envelope (wider)
        comprehensive_envelope = EnvelopeData(
            disruption_areas=np.array([100, 200, 300]),
            lower_bound_production=np.array([1000, 2000, 3000]),
            upper_bound_production=np.array([1500, 2500, 3500]),
            lower_bound_harvest=np.array([100, 200, 300]),
            upper_bound_harvest=np.array([100, 200, 300]),
            convergence_point=(300, 3500),
            convergence_validated=True,
            mathematical_properties={'monotonicity': True, 'dominance': True},
            convergence_statistics={'convergence_error': 0.01},
            crop_type='wheat_comprehensive'
        )
        
        # Commercial envelope (narrower)
        commercial_envelope = EnvelopeData(
            disruption_areas=np.array([100, 200, 300]),
            lower_bound_production=np.array([1100, 2100, 3100]),
            upper_bound_production=np.array([1400, 2400, 3400]),
            lower_bound_harvest=np.array([100, 200, 300]),
            upper_bound_harvest=np.array([100, 200, 300]),
            convergence_point=(300, 3400),
            convergence_validated=True,
            mathematical_properties={'monotonicity': True, 'dominance': True},
            convergence_statistics={'convergence_error': 0.01},
            crop_type='wheat_commercial'
        )
        
        tier_results = {
            'comprehensive': comprehensive_envelope,
            'commercial': commercial_envelope
        }
        
        width_analysis = engine._calculate_width_analysis(tier_results)
        
        assert 'comprehensive_width' in width_analysis
        assert 'commercial_width' in width_analysis
        assert 'commercial_width_reduction_pct' in width_analysis
        
        # Commercial should show width reduction
        reduction_pct = width_analysis['commercial_width_reduction_pct']
        assert reduction_pct > 0, "Commercial tier should show width reduction"
        assert reduction_pct < 100, "Width reduction should be less than 100%"
    
    def test_calculate_representative_width(self, engine):
        """Test representative width calculation."""
        from agrichter.analysis.envelope import EnvelopeData
        
        envelope_data = EnvelopeData(
            disruption_areas=np.array([100, 200, 300]),
            lower_bound_production=np.array([1000, 2000, 3000]),
            upper_bound_production=np.array([1500, 2500, 3500]),
            lower_bound_harvest=np.array([100, 200, 300]),
            upper_bound_harvest=np.array([100, 200, 300]),
            convergence_point=(300, 3500),
            convergence_validated=True,
            mathematical_properties={'monotonicity': True, 'dominance': True},
            convergence_statistics={'convergence_error': 0.01},
            crop_type='test'
        )
        
        width = engine._calculate_representative_width(envelope_data)
        
        expected_widths = envelope_data.upper_bound_production - envelope_data.lower_bound_production
        expected_width = np.median(expected_widths)
        
        assert width == expected_width
    
    def test_calculate_base_statistics(self, engine):
        """Test base statistics calculation."""
        base_data = {
            'production': np.array([1000, 2000, 3000]),
            'harvest': np.array([100, 200, 300]),
            'yields': np.array([10, 10, 10]),
            'filter_stats': {'retention_rate': 85.0}
        }
        
        stats = engine._calculate_base_statistics(base_data)
        
        assert 'total_cells' in stats
        assert 'total_production' in stats
        assert 'total_harvest' in stats
        assert 'mean_yield' in stats
        assert 'median_yield' in stats
        assert 'yield_std' in stats
        assert 'spam_filter_stats' in stats
        
        assert stats['total_cells'] == 3
        assert stats['total_production'] == 6000
        assert stats['total_harvest'] == 600
        assert stats['mean_yield'] == 10
    
    def test_multi_tier_results_class(self):
        """Test MultiTierResults class functionality."""
        from agrichter.analysis.envelope import EnvelopeData
        
        # Create test envelope data
        envelope_data = EnvelopeData(
            disruption_areas=np.array([100, 200]),
            lower_bound_production=np.array([1000, 2000]),
            upper_bound_production=np.array([1500, 2500]),
            lower_bound_harvest=np.array([100, 200]),
            upper_bound_harvest=np.array([100, 200]),
            convergence_point=(200, 2500),
            convergence_validated=True,
            mathematical_properties={'monotonicity': True, 'dominance': True},
            convergence_statistics={'convergence_error': 0.01},
            crop_type='test'
        )
        
        tier_results = {'comprehensive': envelope_data}
        width_analysis = {'comprehensive_width': 500}
        base_statistics = {'total_cells': 100}
        
        results = MultiTierResults(
            tier_results=tier_results,
            width_analysis=width_analysis,
            base_statistics=base_statistics,
            crop_type='wheat',
            calculation_metadata={'timestamp': '2024-01-01'}
        )
        
        # Test methods
        assert results.get_tier_envelope('comprehensive') == envelope_data
        assert results.get_tier_envelope('nonexistent') is None
        
        assert results.get_width_reduction('comprehensive') == 0.0
        assert results.get_width_reduction('nonexistent') is None
        
        summary = results.get_summary_statistics()
        assert summary['crop_type'] == 'wheat'
        assert 'comprehensive_tier' in summary
    
    def test_error_handling(self, engine):
        """Test error handling for invalid inputs."""
        # Test with empty DataFrames
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            engine._get_crop_columns(empty_df)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'invalid_col': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            engine._get_crop_columns(invalid_df)
        
        # Test with invalid crop type that results in no matching columns
        engine.config.crop_type = 'wheat'
        production_df = pd.DataFrame({'RICE_A': [1, 2, 3]})  # Rice data but wheat config
        
        with pytest.raises(ValueError):
            engine._get_crop_columns(production_df)


class TestTierConfiguration:
    """Unit tests for TierConfiguration class."""
    
    def test_tier_configuration_creation(self):
        """Test TierConfiguration creation and validation."""
        config = TierConfiguration(
            name='Test Tier',
            description='Test description',
            yield_percentile_min=10,
            yield_percentile_max=90,
            policy_applications=['test'],
            target_users=['researchers'],
            expected_width_reduction='10-20%'
        )
        
        assert config.name == 'Test Tier'
        assert config.description == 'Test description'
        assert config.yield_percentile_min == 10
        assert config.yield_percentile_max == 90
        assert config.policy_applications == ['test']
        assert config.target_users == ['researchers']
    
    def test_tier_configuration_validation(self):
        """Test tier configuration validation."""
        # Valid configuration should work
        valid_config = TierConfiguration(
            name='Valid',
            description='Valid config',
            yield_percentile_min=0,
            yield_percentile_max=100,
            policy_applications=[],
            target_users=[],
            expected_width_reduction='0%'
        )
        
        assert valid_config.yield_percentile_min <= valid_config.yield_percentile_max
        
        # Test edge cases
        edge_config = TierConfiguration(
            name='Edge',
            description='Edge case',
            yield_percentile_min=50,
            yield_percentile_max=50,  # Same min and max
            policy_applications=[],
            target_users=[],
            expected_width_reduction='0%'
        )
        
        assert edge_config.yield_percentile_min == edge_config.yield_percentile_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
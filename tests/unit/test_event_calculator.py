"""Unit tests for EventCalculator."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper
from agrichter.analysis.event_calculator import EventCalculator


@pytest.fixture
def config():
    """Create test configuration."""
    # Use current directory as root
    root_dir = Path.cwd()
    return Config(crop_type='wheat', root_dir=root_dir, spam_version='2020')


@pytest.fixture
def grid_manager(config):
    """Create and initialize GridDataManager."""
    manager = GridDataManager(config)
    
    # Check if SPAM data files exist
    file_paths = config.get_file_paths()
    if not file_paths['production'].exists():
        pytest.skip(f"SPAM production file not found: {file_paths['production']}")
    
    manager.load_spam_data()
    return manager


@pytest.fixture
def spatial_mapper(config, grid_manager):
    """Create and initialize SpatialMapper."""
    mapper = SpatialMapper(config, grid_manager)
    
    # Check if country codes file exists
    file_paths = config.get_file_paths()
    if not file_paths['country_codes'].exists():
        pytest.skip(f"Country codes file not found: {file_paths['country_codes']}")
    
    mapper.load_country_codes_mapping()
    return mapper


@pytest.fixture
def event_calculator(config, grid_manager, spatial_mapper):
    """Create EventCalculator instance."""
    return EventCalculator(config, grid_manager, spatial_mapper)


class TestEventCalculatorInit:
    """Test EventCalculator initialization."""
    
    def test_init(self, event_calculator, config):
        """Test basic initialization."""
        assert event_calculator.config == config
        assert event_calculator.grid_manager is not None
        assert event_calculator.spatial_mapper is not None
        assert isinstance(event_calculator.event_results, dict)
        assert len(event_calculator.event_results) == 0
    
    def test_repr(self, event_calculator):
        """Test string representation."""
        repr_str = repr(event_calculator)
        assert 'EventCalculator' in repr_str
        assert 'wheat' in repr_str
        assert 'events_calculated=0' in repr_str


class TestMagnitudeCalculation:
    """Test magnitude calculation."""
    
    def test_magnitude_calculation_normal(self, event_calculator):
        """Test magnitude calculation with normal values."""
        # 1,000,000 ha = 10,000 km² → log10(10,000) = 4.0
        magnitude = event_calculator.calculate_magnitude(1_000_000)
        assert magnitude == pytest.approx(4.0, rel=1e-6)
    
    def test_magnitude_calculation_small(self, event_calculator):
        """Test magnitude calculation with small values."""
        # 10,000 ha = 100 km² → log10(100) = 2.0
        magnitude = event_calculator.calculate_magnitude(10_000)
        assert magnitude == pytest.approx(2.0, rel=1e-6)
    
    def test_magnitude_calculation_large(self, event_calculator):
        """Test magnitude calculation with large values."""
        # 100,000,000 ha = 1,000,000 km² → log10(1,000,000) = 6.0
        magnitude = event_calculator.calculate_magnitude(100_000_000)
        assert magnitude == pytest.approx(6.0, rel=1e-6)
    
    def test_magnitude_calculation_zero(self, event_calculator):
        """Test magnitude calculation with zero harvest area."""
        magnitude = event_calculator.calculate_magnitude(0)
        assert np.isnan(magnitude)
    
    def test_magnitude_calculation_negative(self, event_calculator):
        """Test magnitude calculation with negative harvest area."""
        magnitude = event_calculator.calculate_magnitude(-1000)
        assert np.isnan(magnitude)


class TestCountryLevelLoss:
    """Test country-level loss calculation."""
    
    def test_country_level_loss_usa(self, event_calculator):
        """Test country-level loss calculation for USA."""
        # USA GDAM code is typically 231
        harvest_loss, production_loss, grid_cells = event_calculator.calculate_country_level_loss(231)
        
        # USA should have significant wheat production
        assert harvest_loss > 0, "USA should have harvest area data"
        assert production_loss > 0, "USA should have production data"
        assert grid_cells > 0, "USA should have grid cells"
        
        # Sanity checks
        assert harvest_loss < 1e9, "Harvest loss should be reasonable"
        assert production_loss < 1e18, "Production loss should be reasonable"
    
    def test_country_level_loss_invalid_code(self, event_calculator):
        """Test country-level loss with invalid country code."""
        harvest_loss, production_loss, grid_cells = event_calculator.calculate_country_level_loss(99999)
        
        assert harvest_loss == 0
        assert production_loss == 0
        assert grid_cells == 0
    
    def test_country_level_loss_china(self, event_calculator):
        """Test country-level loss calculation for China."""
        # China GDAM code is typically 45
        harvest_loss, production_loss, grid_cells = event_calculator.calculate_country_level_loss(45)
        
        # China should have significant wheat production
        assert harvest_loss > 0, "China should have harvest area data"
        assert production_loss > 0, "China should have production data"
        assert grid_cells > 0, "China should have grid cells"


class TestStateLevelLoss:
    """Test state-level loss calculation."""
    
    def test_state_level_loss_with_states(self, event_calculator):
        """Test state-level loss calculation with state codes."""
        # USA with some state codes (example)
        state_codes = [1, 2, 3]  # Example state codes
        harvest_loss, production_loss, grid_cells = event_calculator.calculate_state_level_loss(
            231, state_codes
        )
        
        # Should return some data (may be zero if state codes don't match)
        assert isinstance(harvest_loss, float)
        assert isinstance(production_loss, float)
        assert isinstance(grid_cells, int)
    
    def test_state_level_loss_empty_states(self, event_calculator):
        """Test state-level loss with empty state codes."""
        # Should fall back to country-level
        harvest_loss, production_loss, grid_cells = event_calculator.calculate_state_level_loss(
            231, []
        )
        
        # Should return country-level data for USA
        assert harvest_loss > 0
        assert production_loss > 0
        assert grid_cells > 0


class TestSingleEventCalculation:
    """Test single event calculation."""
    
    def test_single_event_country_level(self, event_calculator):
        """Test single event calculation at country level."""
        event_data = {
            'country_codes': [231],  # USA
            'state_flags': [0],  # Country-level
            'state_codes': []
        }
        
        result = event_calculator.calculate_single_event('TestEvent', event_data)
        
        assert 'harvest_area_loss_ha' in result
        assert 'production_loss_kcal' in result
        assert 'affected_countries' in result
        assert 'affected_states' in result
        assert 'grid_cells_count' in result
        
        assert result['harvest_area_loss_ha'] > 0
        assert result['production_loss_kcal'] > 0
        assert result['affected_countries'] >= 1
        assert result['grid_cells_count'] > 0
    
    def test_single_event_multiple_countries(self, event_calculator):
        """Test single event with multiple countries."""
        event_data = {
            'country_codes': [231, 45],  # USA and China
            'state_flags': [0, 0],  # Both country-level
            'state_codes': []
        }
        
        result = event_calculator.calculate_single_event('MultiCountryEvent', event_data)
        
        assert result['harvest_area_loss_ha'] > 0
        assert result['production_loss_kcal'] > 0
        assert result['affected_countries'] >= 1
    
    def test_single_event_no_countries(self, event_calculator):
        """Test single event with no country codes."""
        event_data = {
            'country_codes': [],
            'state_flags': [],
            'state_codes': []
        }
        
        result = event_calculator.calculate_single_event('EmptyEvent', event_data)
        
        assert result['harvest_area_loss_ha'] == 0
        assert result['production_loss_kcal'] == 0
        assert result['affected_countries'] == 0


class TestBatchEventProcessing:
    """Test batch event processing."""
    
    def test_calculate_all_events_small_batch(self, event_calculator):
        """Test batch processing with small number of events."""
        events_definitions = {
            'Event1': {
                'country_codes': [231],  # USA
                'state_flags': [0],
                'state_codes': []
            },
            'Event2': {
                'country_codes': [45],  # China
                'state_flags': [0],
                'state_codes': []
            }
        }
        
        results_df = event_calculator.calculate_all_events(events_definitions)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2
        assert 'event_name' in results_df.columns
        assert 'harvest_area_loss_ha' in results_df.columns
        assert 'production_loss_kcal' in results_df.columns
        assert 'magnitude' in results_df.columns
        
        # Check that both events have data
        assert (results_df['harvest_area_loss_ha'] > 0).sum() >= 1
    
    def test_calculate_all_events_with_errors(self, event_calculator):
        """Test batch processing handles errors gracefully."""
        events_definitions = {
            'ValidEvent': {
                'country_codes': [231],
                'state_flags': [0],
                'state_codes': []
            },
            'InvalidEvent': {
                'country_codes': [99999],  # Invalid code
                'state_flags': [0],
                'state_codes': []
            }
        }
        
        results_df = event_calculator.calculate_all_events(events_definitions)
        
        # Should complete despite error
        assert len(results_df) == 2
        
        # Valid event should have data
        valid_row = results_df[results_df['event_name'] == 'ValidEvent'].iloc[0]
        assert valid_row['harvest_area_loss_ha'] > 0
        
        # Invalid event should have zero data
        invalid_row = results_df[results_df['event_name'] == 'InvalidEvent'].iloc[0]
        assert invalid_row['harvest_area_loss_ha'] == 0


class TestEventResultsValidation:
    """Test event results validation."""
    
    def test_validate_event_results_valid(self, event_calculator):
        """Test validation with valid results."""
        # Create sample results
        results_df = pd.DataFrame([
            {
                'event_name': 'Event1',
                'harvest_area_loss_ha': 1_000_000,
                'production_loss_kcal': 1e15,
                'magnitude': 4.0,
                'affected_countries': 1,
                'affected_states': 0,
                'grid_cells_count': 100
            },
            {
                'event_name': 'Event2',
                'harvest_area_loss_ha': 500_000,
                'production_loss_kcal': 5e14,
                'magnitude': 3.7,
                'affected_countries': 1,
                'affected_states': 0,
                'grid_cells_count': 50
            }
        ])
        
        validation_results = event_calculator.validate_event_results(results_df)
        
        assert 'valid' in validation_results
        assert 'errors' in validation_results
        assert 'warnings' in validation_results
        assert 'metrics' in validation_results
        assert 'suspicious_events' in validation_results
        
        # Should have metrics
        assert 'global_production_kcal' in validation_results['metrics']
        assert 'total_events' in validation_results['metrics']
        assert validation_results['metrics']['total_events'] == 2
    
    def test_validate_event_results_with_zeros(self, event_calculator):
        """Test validation with zero losses."""
        results_df = pd.DataFrame([
            {
                'event_name': 'ZeroEvent',
                'harvest_area_loss_ha': 0,
                'production_loss_kcal': 0,
                'magnitude': np.nan,
                'affected_countries': 0,
                'affected_states': 0,
                'grid_cells_count': 0
            }
        ])
        
        validation_results = event_calculator.validate_event_results(results_df)
        
        # Should have warnings about zero losses
        assert len(validation_results['warnings']) > 0
        assert any('zero losses' in w.lower() for w in validation_results['warnings'])
    
    def test_generate_validation_report(self, event_calculator):
        """Test validation report generation."""
        results_df = pd.DataFrame([
            {
                'event_name': 'Event1',
                'harvest_area_loss_ha': 1_000_000,
                'production_loss_kcal': 1e15,
                'magnitude': 4.0,
                'affected_countries': 1,
                'affected_states': 0,
                'grid_cells_count': 100
            }
        ])
        
        report = event_calculator.generate_validation_report(results_df)
        
        assert isinstance(report, str)
        assert 'VALIDATION REPORT' in report
        assert 'Event1' in report or 'PASSED' in report
        assert 'SUMMARY METRICS' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

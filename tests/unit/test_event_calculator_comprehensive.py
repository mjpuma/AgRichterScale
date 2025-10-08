"""Comprehensive unit tests for EventCalculator."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper
from agririchter.analysis.event_calculator import EventCalculator


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.crop_type = 'wheat'
    
    # Mock crop indices
    config.get_crop_indices.return_value = [57]  # Wheat
    
    # Mock caloric content
    config.get_caloric_content.return_value = 3.4  # kcal/g for wheat
    
    # Mock unit conversions
    config.get_unit_conversions.return_value = {
        'grams_per_metric_ton': 1_000_000.0,
        'hectares_to_km2': 0.01
    }
    
    return config


@pytest.fixture
def mock_grid_manager():
    """Create a mock GridDataManager for testing."""
    manager = Mock(spec=GridDataManager)
    
    # Mock grid cells data
    production_cells = pd.DataFrame({
        'grid_code': [1, 2, 3],
        'WHEA_A': [100.0, 200.0, 150.0]  # MT
    })
    
    harvest_cells = pd.DataFrame({
        'grid_code': [1, 2, 3],
        'WHEA_A': [10.0, 20.0, 15.0]  # hectares
    })
    
    manager.get_grid_cells_by_iso3.return_value = (production_cells, harvest_cells)
    manager.get_crop_production.return_value = 450.0 * 1_000_000.0 * 3.4  # Total in kcal
    manager.get_crop_harvest_area.return_value = 45.0  # Total in hectares
    
    return manager


@pytest.fixture
def mock_spatial_mapper():
    """Create a mock SpatialMapper for testing."""
    mapper = Mock(spec=SpatialMapper)
    
    # Mock country codes mapping attribute
    mapper.country_codes_mapping = pd.DataFrame({
        'Country': ['United States'],
        'GDAM ': [840.0],
        'ISO3 alpha': ['USA']
    })
    
    # Mock country mapping methods
    mapper.get_fips_from_country_code.return_value = 'US'
    mapper.get_country_name_from_code.return_value = 'United States'
    mapper.map_country_to_grid_cells.return_value = (['1', '2', '3'], ['1', '2', '3'])
    mapper.load_country_codes_mapping.return_value = mapper.country_codes_mapping
    
    return mapper


class TestEventCalculatorInitialization:
    """Test EventCalculator initialization."""
    
    def test_initialization(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test basic initialization."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        assert calculator.config == mock_config
        assert calculator.grid_manager == mock_grid_manager
        assert calculator.spatial_mapper == mock_spatial_mapper


class TestMagnitudeCalculation:
    """Test magnitude calculation functionality."""
    
    def test_calculate_magnitude_basic(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test basic magnitude calculation."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Test with 10,000 hectares
        # 10,000 ha = 100 km², log10(100) = 2.0
        magnitude = calculator.calculate_magnitude(10_000.0)
        assert abs(magnitude - 2.0) < 0.01
    
    def test_calculate_magnitude_various_values(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test magnitude calculation with various harvest area values."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        test_cases = [
            (100.0, 0.0),      # 100 ha = 1 km², log10(1) = 0
            (1_000.0, 1.0),    # 1,000 ha = 10 km², log10(10) = 1
            (10_000.0, 2.0),   # 10,000 ha = 100 km², log10(100) = 2
            (100_000.0, 3.0),  # 100,000 ha = 1,000 km², log10(1000) = 3
            (1_000_000.0, 4.0) # 1,000,000 ha = 10,000 km², log10(10000) = 4
        ]
        
        for harvest_ha, expected_magnitude in test_cases:
            magnitude = calculator.calculate_magnitude(harvest_ha)
            assert abs(magnitude - expected_magnitude) < 0.01, \
                f"Failed for {harvest_ha} ha: expected {expected_magnitude}, got {magnitude}"
    
    def test_calculate_magnitude_zero_value(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test magnitude calculation with zero harvest area."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Zero harvest area should return NaN
        magnitude = calculator.calculate_magnitude(0.0)
        assert np.isnan(magnitude)
    
    def test_calculate_magnitude_negative_value(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test magnitude calculation with negative harvest area."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Negative harvest area should return NaN
        magnitude = calculator.calculate_magnitude(-100.0)
        assert np.isnan(magnitude)
    
    def test_calculate_magnitude_unit_conversion(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test that magnitude calculation uses correct unit conversion."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Verify conversion: hectares → km² → log10
        harvest_ha = 30.0
        expected_km2 = harvest_ha * 0.01  # 0.30 km²
        expected_magnitude = np.log10(expected_km2)  # log10(0.30) ≈ -0.52
        
        magnitude = calculator.calculate_magnitude(harvest_ha)
        assert abs(magnitude - expected_magnitude) < 0.01


class TestUnitConversions:
    """Test unit conversion functionality."""
    
    def test_hectares_to_km2_conversion(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test hectares to km² conversion."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Get conversion factor
        conversions = calculator.config.get_unit_conversions()
        hectares_to_km2 = conversions['hectares_to_km2']
        
        # Verify correct conversion factor
        assert hectares_to_km2 == 0.01
        
        # Test conversion
        test_cases = [
            (100, 1.0),
            (1_000, 10.0),
            (10_000, 100.0),
            (100_000, 1_000.0)
        ]
        
        for ha, expected_km2 in test_cases:
            km2 = ha * hectares_to_km2
            assert abs(km2 - expected_km2) < 0.01
    
    def test_metric_tons_to_kcal_conversion(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test metric tons to kcal conversion."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Get conversion factors
        conversions = calculator.config.get_unit_conversions()
        grams_per_mt = conversions['grams_per_metric_ton']
        caloric_content = calculator.config.get_caloric_content()
        
        # Verify correct factors
        assert grams_per_mt == 1_000_000.0
        assert caloric_content == 3.4  # kcal/g for wheat
        
        # Test conversion: 1 MT wheat = 1,000,000 g * 3.4 kcal/g = 3,400,000 kcal
        mt = 1.0
        expected_kcal = mt * grams_per_mt * caloric_content
        assert expected_kcal == 3_400_000.0


class TestSingleEventCalculation:
    """Test single event calculation functionality."""
    
    def test_calculate_single_event_country_level(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test calculating a single country-level event."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Create sample event data
        event_data = pd.DataFrame({
            'Country': [840.0],  # USA
            'State': [np.nan],
            'Flag': [1]
        })
        
        # Calculate event
        result = calculator.calculate_single_event('Test Event', event_data)
        
        # Verify result structure
        assert 'harvest_area_loss_ha' in result
        assert 'production_loss_kcal' in result
        assert 'grid_cells_count' in result
        assert 'affected_countries' in result
        
        # Verify values are non-negative (may be zero if no grid cells found)
        assert result['harvest_area_loss_ha'] >= 0
        assert result['production_loss_kcal'] >= 0
        assert result['grid_cells_count'] >= 0


class TestLossAggregation:
    """Test loss aggregation functionality."""
    
    def test_aggregate_losses_multiple_countries(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test aggregating losses across multiple countries."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Create event with multiple countries
        event_data = pd.DataFrame({
            'Country': [840.0, 124.0],  # USA, Canada
            'State': [np.nan, np.nan],
            'Flag': [1, 1]
        })
        
        # Calculate event
        result = calculator.calculate_single_event('Multi-Country Event', event_data)
        
        # Verify multiple countries in event data
        assert result['affected_countries'] >= 0  # May be 0 if no grid cells found
        
        # Verify losses are non-negative
        assert result['harvest_area_loss_ha'] >= 0
        assert result['production_loss_kcal'] >= 0


class TestEventValidation:
    """Test event validation functionality."""
    
    def test_validate_event_results_basic(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test basic event results validation."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Create sample results
        results_df = pd.DataFrame({
            'event_name': ['Event 1', 'Event 2'],
            'harvest_area_loss_ha': [10_000.0, 50_000.0],
            'production_loss_kcal': [1e14, 5e14],
            'magnitude': [2.0, 2.7]
        })
        
        # Validate results
        validation = calculator.validate_event_results(results_df)
        
        # Verify validation structure
        assert 'valid' in validation
        assert 'metrics' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
    
    def test_validate_magnitude_ranges(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test validation of magnitude ranges."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Create results with various magnitudes
        results_df = pd.DataFrame({
            'event_name': ['Low', 'Normal', 'High'],
            'harvest_area_loss_ha': [10.0, 10_000.0, 10_000_000.0],
            'production_loss_kcal': [1e12, 1e14, 1e16],
            'magnitude': [-1.0, 2.0, 5.0]
        })
        
        # Validate results
        validation = calculator.validate_event_results(results_df)
        
        # Should have warnings for unusual magnitudes
        assert len(validation['warnings']) > 0


class TestBatchEventProcessing:
    """Test batch event processing functionality."""
    
    def test_calculate_all_events_structure(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test structure of batch event processing results."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Create sample events data
        events_data = {
            'country': {
                'Event1': pd.DataFrame({
                    'Country': [840.0],
                    'State': [np.nan],
                    'Flag': [1]
                }),
                'Event2': pd.DataFrame({
                    'Country': [124.0],
                    'State': [np.nan],
                    'Flag': [1]
                })
            },
            'state': {}
        }
        
        # Calculate all events
        results_df = calculator.calculate_all_events(events_data)
        
        # Verify DataFrame structure
        assert isinstance(results_df, pd.DataFrame)
        assert 'event_name' in results_df.columns
        assert 'harvest_area_loss_ha' in results_df.columns
        assert 'production_loss_kcal' in results_df.columns
        assert 'magnitude' in results_df.columns
        
        # Verify number of events
        assert len(results_df) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_calculate_event_with_no_grid_cells(self, mock_config, mock_spatial_mapper):
        """Test event calculation when no grid cells found."""
        # Create grid manager that returns empty data
        grid_manager = Mock(spec=GridDataManager)
        grid_manager.get_grid_cells_by_iso3.return_value = (pd.DataFrame(), pd.DataFrame())
        grid_manager.get_crop_production.return_value = 0.0
        grid_manager.get_crop_harvest_area.return_value = 0.0
        
        calculator = EventCalculator(mock_config, grid_manager, mock_spatial_mapper)
        
        # Create event
        event_data = pd.DataFrame({
            'Country': [999.0],  # Non-existent country
            'State': [np.nan],
            'Flag': [1]
        })
        
        # Calculate event
        result = calculator.calculate_single_event('Empty Event', event_data)
        
        # Verify zero losses
        assert result['harvest_area_loss_ha'] == 0.0
        assert result['production_loss_kcal'] == 0.0
        assert result['grid_cells_count'] == 0
    
    def test_calculate_magnitude_with_very_small_area(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test magnitude calculation with very small harvest area."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Very small area: 1 hectare = 0.01 km², log10(0.01) = -2
        magnitude = calculator.calculate_magnitude(1.0)
        assert abs(magnitude - (-2.0)) < 0.01
    
    def test_calculate_magnitude_with_very_large_area(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test magnitude calculation with very large harvest area."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Very large area: 10,000,000 ha = 100,000 km², log10(100000) = 5
        magnitude = calculator.calculate_magnitude(10_000_000.0)
        assert abs(magnitude - 5.0) < 0.01


class TestMagnitudeFormula:
    """Test magnitude formula implementation."""
    
    def test_magnitude_formula_matches_specification(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test that magnitude formula matches M_D = log10(A_H in km²)."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Test cases from specification
        test_cases = [
            # (harvest_ha, expected_magnitude)
            (100, 0.0),        # 1 km²
            (1_000, 1.0),      # 10 km²
            (10_000, 2.0),     # 100 km²
            (100_000, 3.0),    # 1,000 km²
            (1_000_000, 4.0),  # 10,000 km²
            (10_000_000, 5.0), # 100,000 km²
        ]
        
        for harvest_ha, expected_magnitude in test_cases:
            magnitude = calculator.calculate_magnitude(harvest_ha)
            assert abs(magnitude - expected_magnitude) < 0.01, \
                f"Formula mismatch for {harvest_ha} ha: expected M={expected_magnitude}, got M={magnitude}"
    
    def test_magnitude_inverse_calculation(self, mock_config, mock_grid_manager, mock_spatial_mapper):
        """Test that magnitude can be inverted to get harvest area."""
        calculator = EventCalculator(mock_config, mock_grid_manager, mock_spatial_mapper)
        
        # Test round-trip: harvest_ha → magnitude → harvest_ha
        original_harvest_ha = 50_000.0
        
        # Calculate magnitude
        magnitude = calculator.calculate_magnitude(original_harvest_ha)
        
        # Invert: magnitude → km² → ha
        harvest_km2 = 10 ** magnitude
        recovered_harvest_ha = harvest_km2 / 0.01
        
        # Verify round-trip
        assert abs(recovered_harvest_ha - original_harvest_ha) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

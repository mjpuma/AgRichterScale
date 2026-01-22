"""Unit tests for ConvergenceValidator mathematical validation framework."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import logging

from agrichter.analysis.convergence_validator import (
    ConvergenceValidator,
    ValidationResult,
    MathematicalProperties,
    ConvergenceError,
    MathematicalPropertyError,
    ConservationError
)


class TestMathematicalProperties:
    """Test MathematicalProperties dataclass."""
    
    def test_is_mathematically_valid_all_true(self):
        """Test that all properties true returns valid."""
        props = MathematicalProperties(
            starts_at_origin=True,
            converges_at_endpoint=True,
            upper_dominates_lower=True,
            conservation_satisfied=True,
            monotonic_harvest=True
        )
        assert props.is_mathematically_valid() is True
    
    def test_is_mathematically_valid_one_false(self):
        """Test that any property false returns invalid."""
        props = MathematicalProperties(
            starts_at_origin=False,
            converges_at_endpoint=True,
            upper_dominates_lower=True,
            conservation_satisfied=True,
            monotonic_harvest=True
        )
        assert props.is_mathematically_valid() is False
    
    def test_is_mathematically_valid_all_false(self):
        """Test that all properties false returns invalid."""
        props = MathematicalProperties(
            starts_at_origin=False,
            converges_at_endpoint=False,
            upper_dominates_lower=False,
            conservation_satisfied=False,
            monotonic_harvest=False
        )
        assert props.is_mathematically_valid() is False


class TestConvergenceValidator:
    """Test ConvergenceValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    @pytest.fixture
    def valid_envelope_data(self):
        """Create valid envelope data for testing."""
        # Create envelope that starts at origin and converges properly
        harvest_points = np.array([0.0, 100.0, 500.0, 1000.0])
        lower_production = np.array([0.0, 80.0, 400.0, 1000.0])
        upper_production = np.array([0.0, 120.0, 600.0, 1000.0])
        disruption_areas = harvest_points.copy()
        
        return {
            'lower_bound_harvest': harvest_points,
            'lower_bound_production': lower_production,
            'upper_bound_harvest': harvest_points,
            'upper_bound_production': upper_production,
            'disruption_areas': disruption_areas
        }
    
    @pytest.fixture
    def invalid_envelope_data(self):
        """Create invalid envelope data for testing."""
        # Create envelope with convergence issues
        harvest_points = np.array([10.0, 100.0, 500.0, 900.0])  # Doesn't start at origin
        lower_production = np.array([5.0, 80.0, 400.0, 800.0])  # Doesn't converge
        upper_production = np.array([15.0, 70.0, 350.0, 750.0])  # Upper < lower at some points
        disruption_areas = harvest_points.copy()
        
        return {
            'lower_bound_harvest': harvest_points,
            'lower_bound_production': lower_production,
            'upper_bound_harvest': harvest_points,
            'upper_bound_production': upper_production,
            'disruption_areas': disruption_areas
        }
    
    def test_init_default_tolerance(self):
        """Test validator initialization with default tolerance."""
        validator = ConvergenceValidator()
        assert validator.tolerance == 1e-6
        assert validator.logger.name == 'agrichter.convergence_validator'
    
    def test_init_custom_tolerance(self):
        """Test validator initialization with custom tolerance."""
        validator = ConvergenceValidator(tolerance=1e-4)
        assert validator.tolerance == 1e-4
    
    def test_validate_mathematical_properties_valid_envelope(self, validator, valid_envelope_data):
        """Test validation of mathematically correct envelope."""
        total_production = 1000.0
        total_harvest = 1000.0
        
        result = validator.validate_mathematical_properties(
            valid_envelope_data, total_production, total_harvest
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert 'starts_at_origin' in result.properties
        assert 'converges_at_endpoint' in result.properties
        assert 'upper_dominates_lower' in result.properties
        assert 'conservation_satisfied' in result.properties
        assert 'monotonic_harvest' in result.properties
        assert len(result.statistics) > 0
    
    def test_validate_mathematical_properties_invalid_envelope(self, validator, invalid_envelope_data):
        """Test validation of mathematically incorrect envelope."""
        total_production = 1000.0
        total_harvest = 1000.0
        
        result = validator.validate_mathematical_properties(
            invalid_envelope_data, total_production, total_harvest
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('origin' in error.lower() for error in result.errors)
    
    def test_validate_mathematical_properties_empty_data(self, validator):
        """Test validation with empty envelope data."""
        empty_data = {
            'lower_bound_harvest': np.array([]),
            'lower_bound_production': np.array([]),
            'upper_bound_harvest': np.array([]),
            'upper_bound_production': np.array([]),
            'disruption_areas': np.array([])
        }
        
        result = validator.validate_mathematical_properties(empty_data, 1000.0, 1000.0)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_mathematical_properties_exception_handling(self, validator):
        """Test validation handles exceptions gracefully."""
        # Invalid data that will cause exceptions
        invalid_data = {
            'lower_bound_harvest': "not_an_array",
            'lower_bound_production': np.array([1, 2, 3]),
            'upper_bound_harvest': np.array([1, 2]),
            'upper_bound_production': np.array([1, 2, 3, 4]),
            'disruption_areas': np.array([1])
        }
        
        result = validator.validate_mathematical_properties(invalid_data, 1000.0, 1000.0)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('exception' in error.lower() for error in result.errors)


class TestEndpointConvergence:
    """Test endpoint convergence validation methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_check_endpoint_convergence_perfect(self, validator):
        """Test convergence check with perfect convergence."""
        lower_bound = np.array([0.0, 250.0, 500.0, 1000.0])
        upper_bound = np.array([0.0, 750.0, 500.0, 1000.0])
        harvest_areas = np.array([0.0, 250.0, 500.0, 1000.0])
        total_production = 1000.0
        total_harvest = 1000.0
        
        result = validator.check_endpoint_convergence(
            lower_bound, upper_bound, harvest_areas, total_production, total_harvest
        )
        
        assert result == True
    
    def test_check_endpoint_convergence_within_tolerance(self, validator):
        """Test convergence check within tolerance."""
        lower_bound = np.array([0.0, 250.0, 500.0, 999.999])  # Very close to 1000
        upper_bound = np.array([0.0, 750.0, 500.0, 1000.001])  # Very close to 1000
        harvest_areas = np.array([0.0, 250.0, 500.0, 1000.0])
        total_production = 1000.0
        total_harvest = 1000.0
        
        result = validator.check_endpoint_convergence(
            lower_bound, upper_bound, harvest_areas, total_production, total_harvest
        )
        
        assert result == True
    
    def test_check_endpoint_convergence_fails(self, validator):
        """Test convergence check failure."""
        lower_bound = np.array([0.0, 250.0, 500.0, 800.0])  # Doesn't converge
        upper_bound = np.array([0.0, 750.0, 500.0, 900.0])  # Doesn't converge
        harvest_areas = np.array([0.0, 250.0, 500.0, 1000.0])
        total_production = 1000.0
        total_harvest = 1000.0
        
        result = validator.check_endpoint_convergence(
            lower_bound, upper_bound, harvest_areas, total_production, total_harvest
        )
        
        assert result == False
    
    def test_check_endpoint_convergence_empty_arrays(self, validator):
        """Test convergence check with empty arrays."""
        result = validator.check_endpoint_convergence(
            np.array([]), np.array([]), np.array([]), 1000.0, 1000.0
        )
        
        assert result == False
    
    def test_check_endpoint_convergence_insufficient_coverage(self, validator):
        """Test convergence check with insufficient harvest coverage."""
        lower_bound = np.array([0.0, 250.0, 500.0])
        upper_bound = np.array([0.0, 750.0, 500.0])
        harvest_areas = np.array([0.0, 250.0, 500.0])  # Only 50% coverage
        total_production = 1000.0
        total_harvest = 1000.0
        
        with patch.object(validator.logger, 'warning') as mock_warning:
            result = validator.check_endpoint_convergence(
                lower_bound, upper_bound, harvest_areas, total_production, total_harvest
            )
            mock_warning.assert_called()


class TestConvergenceEnforcement:
    """Test convergence enforcement methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_enforce_convergence_missing_point(self, validator):
        """Test convergence enforcement when convergence point is missing."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 250.0, 500.0]),
            'lower_bound_production': np.array([0.0, 200.0, 400.0]),
            'upper_bound_harvest': np.array([0.0, 250.0, 500.0]),
            'upper_bound_production': np.array([0.0, 300.0, 600.0]),
            'disruption_areas': np.array([0.0, 250.0, 500.0])
        }
        total_production = 1000.0
        total_harvest = 1000.0
        
        with patch.object(validator, 'validate_mathematical_properties') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True, errors=[], warnings=[], properties={}, statistics={}
            )
            
            corrected_data = validator.enforce_convergence(
                envelope_data, total_production, total_harvest
            )
        
        # Check that convergence point was added
        assert len(corrected_data['lower_bound_harvest']) == 4
        assert corrected_data['lower_bound_harvest'][-1] == total_harvest
        assert corrected_data['lower_bound_production'][-1] == total_production
        assert corrected_data['upper_bound_harvest'][-1] == total_harvest
        assert corrected_data['upper_bound_production'][-1] == total_production
    
    def test_enforce_convergence_correct_existing_point(self, validator):
        """Test convergence enforcement when correcting existing endpoint."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 250.0, 500.0, 1000.0]),
            'lower_bound_production': np.array([0.0, 200.0, 400.0, 900.0]),  # Wrong value
            'upper_bound_harvest': np.array([0.0, 250.0, 500.0, 1000.0]),
            'upper_bound_production': np.array([0.0, 300.0, 600.0, 950.0]),  # Wrong value
            'disruption_areas': np.array([0.0, 250.0, 500.0, 1000.0])
        }
        total_production = 1000.0
        total_harvest = 1000.0
        
        with patch.object(validator, 'validate_mathematical_properties') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True, errors=[], warnings=[], properties={}, statistics={}
            )
            
            corrected_data = validator.enforce_convergence(
                envelope_data, total_production, total_harvest
            )
        
        # Check that endpoint was corrected
        assert len(corrected_data['lower_bound_harvest']) == 4  # Same length
        assert corrected_data['lower_bound_harvest'][-1] == total_harvest
        assert corrected_data['lower_bound_production'][-1] == total_production
        assert corrected_data['upper_bound_harvest'][-1] == total_harvest
        assert corrected_data['upper_bound_production'][-1] == total_production


class TestConservationLaws:
    """Test conservation law validation methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_check_conservation_laws_satisfied(self, validator):
        """Test conservation law validation when satisfied."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 250.0, 500.0, 1000.0]),
            'lower_bound_production': np.array([0.0, 200.0, 400.0, 1000.0]),
            'upper_bound_harvest': np.array([0.0, 250.0, 500.0, 1000.0]),
            'upper_bound_production': np.array([0.0, 300.0, 600.0, 1000.0]),
            'disruption_areas': np.array([0.0, 250.0, 500.0, 1000.0])
        }
        total_production = 1000.0
        total_harvest = 1000.0
        
        result = validator._check_conservation_laws(envelope_data, total_production, total_harvest)
        
        assert result == True
    
    def test_check_conservation_laws_violated(self, validator):
        """Test conservation law validation when violated."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 250.0, 500.0, 1000.0]),
            'lower_bound_production': np.array([0.0, 200.0, 400.0, 800.0]),  # Wrong total
            'upper_bound_harvest': np.array([0.0, 250.0, 500.0, 1000.0]),
            'upper_bound_production': np.array([0.0, 300.0, 600.0, 900.0]),  # Wrong total
            'disruption_areas': np.array([0.0, 250.0, 500.0, 1000.0])
        }
        total_production = 1000.0
        total_harvest = 1000.0
        
        with patch.object(validator.logger, 'warning') as mock_warning:
            result = validator._check_conservation_laws(envelope_data, total_production, total_harvest)
            mock_warning.assert_called()
        
        assert result == False
    
    def test_check_conservation_laws_incomplete_envelope(self, validator):
        """Test conservation law validation with incomplete envelope."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 250.0, 500.0]),  # Only 50% coverage
            'lower_bound_production': np.array([0.0, 200.0, 400.0]),
            'upper_bound_harvest': np.array([0.0, 250.0, 500.0]),
            'upper_bound_production': np.array([0.0, 300.0, 600.0]),
            'disruption_areas': np.array([0.0, 250.0, 500.0])
        }
        total_production = 1000.0
        total_harvest = 1000.0
        
        with patch.object(validator.logger, 'warning') as mock_warning:
            result = validator._check_conservation_laws(envelope_data, total_production, total_harvest)
            mock_warning.assert_called()
        
        # Should return True for incomplete envelopes (don't fail validation)
        assert result == True


class TestMathematicalPropertyChecks:
    """Test individual mathematical property check methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_check_origin_start_valid(self, validator):
        """Test origin start check with valid data."""
        lower_harvest = np.array([0.0, 100.0, 200.0])
        lower_production = np.array([0.0, 50.0, 100.0])
        upper_harvest = np.array([0.0, 100.0, 200.0])
        upper_production = np.array([0.0, 150.0, 300.0])
        
        result = validator._check_origin_start(
            lower_harvest, lower_production, upper_harvest, upper_production
        )
        
        assert result == True
    
    def test_check_origin_start_invalid(self, validator):
        """Test origin start check with invalid data."""
        lower_harvest = np.array([10.0, 100.0, 200.0])  # Doesn't start at 0
        lower_production = np.array([5.0, 50.0, 100.0])  # Doesn't start at 0
        upper_harvest = np.array([0.0, 100.0, 200.0])
        upper_production = np.array([0.0, 150.0, 300.0])
        
        result = validator._check_origin_start(
            lower_harvest, lower_production, upper_harvest, upper_production
        )
        
        assert result == False
    
    def test_check_origin_start_empty_arrays(self, validator):
        """Test origin start check with empty arrays."""
        result = validator._check_origin_start(
            np.array([]), np.array([]), np.array([]), np.array([])
        )
        
        assert result == False
    
    def test_check_upper_dominance_valid(self, validator):
        """Test upper bound dominance check with valid data."""
        lower_production = np.array([0.0, 50.0, 100.0, 200.0])
        upper_production = np.array([0.0, 150.0, 300.0, 400.0])
        
        result = validator._check_upper_dominance(lower_production, upper_production)
        
        assert result == True
    
    def test_check_upper_dominance_invalid(self, validator):
        """Test upper bound dominance check with invalid data."""
        lower_production = np.array([0.0, 150.0, 300.0, 400.0])
        upper_production = np.array([0.0, 50.0, 100.0, 200.0])  # Upper < lower
        
        with patch.object(validator.logger, 'warning') as mock_warning:
            result = validator._check_upper_dominance(lower_production, upper_production)
            mock_warning.assert_called()
        
        assert result == False
    
    def test_check_upper_dominance_mismatched_lengths(self, validator):
        """Test upper bound dominance check with mismatched array lengths."""
        lower_production = np.array([0.0, 50.0, 100.0])
        upper_production = np.array([0.0, 150.0])  # Different length
        
        result = validator._check_upper_dominance(lower_production, upper_production)
        
        assert result == False
    
    def test_check_monotonic_harvest_valid(self, validator):
        """Test monotonic harvest check with valid data."""
        lower_harvest = np.array([0.0, 100.0, 200.0, 300.0])
        upper_harvest = np.array([0.0, 100.0, 200.0, 300.0])
        
        result = validator._check_monotonic_harvest(lower_harvest, upper_harvest)
        
        assert result == True
    
    def test_check_monotonic_harvest_invalid(self, validator):
        """Test monotonic harvest check with invalid data."""
        lower_harvest = np.array([0.0, 200.0, 100.0, 300.0])  # Not monotonic
        upper_harvest = np.array([0.0, 100.0, 200.0, 300.0])
        
        result = validator._check_monotonic_harvest(lower_harvest, upper_harvest)
        
        assert result == False


class TestYieldOrderingValidation:
    """Test yield ordering validation methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_validate_yield_ordering_valid(self, validator):
        """Test yield ordering validation with properly sorted data."""
        # H-P matrix: [harvest_area, production, yield]
        hp_matrix = np.array([
            [100.0, 50.0, 0.5],   # Low yield
            [200.0, 150.0, 0.75], # Medium yield
            [150.0, 150.0, 1.0],  # High yield
            [100.0, 200.0, 2.0]   # Very high yield
        ])
        
        result = validator.validate_yield_ordering(hp_matrix)
        
        assert result == True
    
    def test_validate_yield_ordering_invalid(self, validator):
        """Test yield ordering validation with improperly sorted data."""
        # H-P matrix: [harvest_area, production, yield] - not sorted by yield
        hp_matrix = np.array([
            [100.0, 200.0, 2.0],  # High yield first
            [200.0, 150.0, 0.75], # Lower yield second
            [150.0, 150.0, 1.0],  # Medium yield third
            [100.0, 50.0, 0.5]    # Lowest yield last
        ])
        
        with patch.object(validator.logger, 'error') as mock_error:
            result = validator.validate_yield_ordering(hp_matrix)
            mock_error.assert_called()
        
        assert result == False
    
    def test_validate_yield_ordering_single_cell(self, validator):
        """Test yield ordering validation with single cell."""
        hp_matrix = np.array([[100.0, 50.0, 0.5]])
        
        result = validator.validate_yield_ordering(hp_matrix)
        
        assert result == True
    
    def test_validate_yield_ordering_empty(self, validator):
        """Test yield ordering validation with empty matrix."""
        hp_matrix = np.array([]).reshape(0, 3)
        
        result = validator.validate_yield_ordering(hp_matrix)
        
        assert result == True


class TestCumulativeProperties:
    """Test cumulative properties validation methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_validate_cumulative_properties_valid(self, validator):
        """Test cumulative properties validation with valid data."""
        hp_matrix = np.array([
            [100.0, 50.0, 0.5],
            [200.0, 150.0, 0.75],
            [150.0, 150.0, 1.0],
            [100.0, 200.0, 2.0]
        ])
        total_production = 550.0  # Sum of production column
        total_harvest = 550.0     # Sum of harvest column
        
        result = validator.validate_cumulative_properties(
            hp_matrix, total_production, total_harvest
        )
        
        assert result == True
    
    def test_validate_cumulative_properties_invalid(self, validator):
        """Test cumulative properties validation with invalid totals."""
        hp_matrix = np.array([
            [100.0, 50.0, 0.5],
            [200.0, 150.0, 0.75],
            [150.0, 150.0, 1.0],
            [100.0, 200.0, 2.0]
        ])
        total_production = 1000.0  # Wrong total
        total_harvest = 1000.0     # Wrong total
        
        with patch.object(validator.logger, 'error') as mock_error:
            result = validator.validate_cumulative_properties(
                hp_matrix, total_production, total_harvest
            )
            mock_error.assert_called()
        
        assert result == False
    
    def test_validate_cumulative_properties_empty(self, validator):
        """Test cumulative properties validation with empty matrix."""
        hp_matrix = np.array([]).reshape(0, 3)
        
        result = validator.validate_cumulative_properties(hp_matrix, 1000.0, 1000.0)
        
        assert result == False


class TestEnvelopeContinuity:
    """Test envelope continuity validation methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_check_envelope_continuity_valid(self, validator):
        """Test envelope continuity check with continuous data."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 100.0, 200.0, 300.0]),
            'lower_bound_production': np.array([0.0, 50.0, 100.0, 150.0]),
            'upper_bound_harvest': np.array([0.0, 100.0, 200.0, 300.0]),
            'upper_bound_production': np.array([0.0, 150.0, 300.0, 450.0])
        }
        
        result = validator.check_envelope_continuity(envelope_data)
        
        assert result == True
    
    def test_check_envelope_continuity_discontinuous(self, validator):
        """Test envelope continuity check with discontinuous data."""
        envelope_data = {
            'lower_bound_harvest': np.array([0.0, 10.0, 2000.0, 2100.0]),  # Large jump
            'lower_bound_production': np.array([0.0, 5.0, 1000.0, 1050.0]),  # Large jump
            'upper_bound_harvest': np.array([0.0, 10.0, 2000.0, 2100.0]),
            'upper_bound_production': np.array([0.0, 15.0, 3000.0, 3150.0])
        }
        
        with patch.object(validator.logger, 'warning') as mock_warning:
            result = validator.check_envelope_continuity(envelope_data)
            mock_warning.assert_called()
        
        assert result == False
    
    def test_check_envelope_continuity_single_point(self, validator):
        """Test envelope continuity check with single point."""
        envelope_data = {
            'lower_bound_harvest': np.array([100.0]),
            'lower_bound_production': np.array([50.0]),
            'upper_bound_harvest': np.array([100.0]),
            'upper_bound_production': np.array([150.0])
        }
        
        result = validator.check_envelope_continuity(envelope_data)
        
        assert result == True


class TestConvergenceStatistics:
    """Test convergence statistics calculation methods."""
    
    @pytest.fixture
    def validator(self):
        """Create a ConvergenceValidator instance."""
        return ConvergenceValidator(tolerance=1e-6)
    
    def test_calculate_convergence_statistics_valid(self, validator):
        """Test convergence statistics calculation with valid data."""
        lower_harvest = np.array([0.0, 250.0, 500.0, 1000.0])
        lower_production = np.array([0.0, 200.0, 400.0, 1000.0])
        upper_harvest = np.array([0.0, 250.0, 500.0, 1000.0])
        upper_production = np.array([0.0, 800.0, 600.0, 1000.0])
        total_production = 1000.0
        total_harvest = 1000.0
        
        stats = validator._calculate_convergence_statistics(
            lower_harvest, lower_production, upper_harvest, upper_production,
            total_production, total_harvest
        )
        
        assert isinstance(stats, dict)
        assert 'max_harvest_coverage' in stats
        assert 'max_production_coverage' in stats
        assert 'avg_production_width' in stats
        assert 'final_production_width' in stats
        assert 'width_reduction_ratio' in stats
        assert 'envelope_points' in stats
        
        assert stats['max_harvest_coverage'] == 1.0
        assert stats['max_production_coverage'] == 1.0
        assert stats['envelope_points'] == 4
        assert stats['final_production_width'] == 0.0  # Converges
    
    def test_calculate_convergence_statistics_empty(self, validator):
        """Test convergence statistics calculation with empty data."""
        stats = validator._calculate_convergence_statistics(
            np.array([]), np.array([]), np.array([]), np.array([]),
            1000.0, 1000.0
        )
        
        assert stats == {}


class TestValidationExceptions:
    """Test validation exception classes."""
    
    def test_convergence_error(self):
        """Test ConvergenceError exception."""
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Test convergence error")
    
    def test_mathematical_property_error(self):
        """Test MathematicalPropertyError exception."""
        with pytest.raises(MathematicalPropertyError):
            raise MathematicalPropertyError("Test mathematical property error")
    
    def test_conservation_error(self):
        """Test ConservationError exception."""
        with pytest.raises(ConservationError):
            raise ConservationError("Test conservation error")


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and attributes."""
        result = ValidationResult(
            is_valid=True,
            errors=["error1", "error2"],
            warnings=["warning1"],
            properties={"prop1": True, "prop2": False},
            statistics={"stat1": 1.0, "stat2": 2.0}
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert len(result.properties) == 2
        assert len(result.statistics) == 2
        assert result.properties["prop1"] is True
        assert result.statistics["stat1"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
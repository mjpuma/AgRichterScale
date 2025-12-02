"""
Comprehensive tests for envelope builder - Task 8.

This module implements all test requirements from Task 8:
- 8.1: Synthetic data tests with known properties
- 8.2: Unit tests for validation logic
- 8.3: Unit tests for discrete envelope building
- 8.4: Unit tests for interpolation
- 8.5: Integration tests with real data (placeholder)

All tests follow the TEST-FIRST approach and verify mathematical properties.
"""

import numpy as np
import pytest
from agririchter.analysis.envelope_builder import (
    build_envelope,
    _validate_and_prepare_inputs,
    _build_discrete_sequence,
    _create_query_grid,
    _interpolate_envelopes
)


# ============================================================================
# TASK 8.1: Synthetic data tests (MUST PASS BEFORE REAL DATA)
# ============================================================================

class TestSyntheticData:
    """Test envelope builder with synthetic data with known properties."""
    
    def test_synthetic_uniform_yield_1000_cells(self):
        """
        Test with 1000 cells with uniform yield distribution.
        
        Properties to verify:
        - Cumsum totals match sum of inputs
        - Monotonicity of cumulative sequences
        - Dominance property (upper >= lower) at all points
        - Interpolation preserves monotonicity
        """
        # Create 1000 cells with uniform yield distribution
        np.random.seed(42)
        n_cells = 1000
        
        # Uniform yield between 1 and 10 mt/ha
        Y_mt_per_ha = np.random.uniform(1.0, 10.0, n_cells)
        
        # Uniform harvest area between 10 and 100 ha
        H_ha = np.random.uniform(10.0, 100.0, n_cells)
        
        # Production = Yield * Harvest
        P_mt = Y_mt_per_ha * H_ha
        
        # Known totals
        expected_total_P = np.sum(P_mt)
        expected_total_H_ha = np.sum(H_ha)
        expected_total_H_km2 = expected_total_H_ha * 0.01
        
        # Build envelope without interpolation
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=False)
        
        # Test 1: Cumsum totals match sum of inputs
        lower = result['lower']
        upper = result['upper']
        
        assert np.isclose(lower['cum_H_km2'][-1], expected_total_H_km2, rtol=1e-10), \
            "Lower envelope: cumulative harvest doesn't match total"
        assert np.isclose(lower['cum_P_mt'][-1], expected_total_P, rtol=1e-10), \
            "Lower envelope: cumulative production doesn't match total"
        assert np.isclose(upper['cum_H_km2'][-1], expected_total_H_km2, rtol=1e-10), \
            "Upper envelope: cumulative harvest doesn't match total"
        assert np.isclose(upper['cum_P_mt'][-1], expected_total_P, rtol=1e-10), \
            "Upper envelope: cumulative production doesn't match total"
        
        # Test 2: Monotonicity of cumulative sequences
        assert np.all(np.diff(lower['cum_H_km2']) > 0), \
            "Lower envelope: cumulative harvest not strictly increasing"
        assert np.all(np.diff(lower['cum_P_mt']) >= 0), \
            "Lower envelope: cumulative production not non-decreasing"
        assert np.all(np.diff(upper['cum_H_km2']) > 0), \
            "Upper envelope: cumulative harvest not strictly increasing"
        assert np.all(np.diff(upper['cum_P_mt']) >= 0), \
            "Upper envelope: cumulative production not non-decreasing"
        
        # Test 3: Yield ordering
        assert np.all(np.diff(lower['Y_sorted']) >= 0), \
            "Lower envelope: yields not sorted ascending"
        assert np.all(np.diff(upper['Y_sorted']) <= 0), \
            "Upper envelope: yields not sorted descending"
        
        # Test 4: Dominance property at all discrete points
        # Sample at lower envelope's harvest levels
        for i in range(0, len(lower['cum_H_km2']), 100):  # Sample every 100th point
            H_target = lower['cum_H_km2'][i]
            P_low = lower['cum_P_mt'][i]
            
            # Find corresponding point in upper envelope
            k_up = np.searchsorted(upper['cum_H_km2'], H_target)
            if k_up >= len(upper['cum_P_mt']):
                k_up = len(upper['cum_P_mt']) - 1
            P_up = upper['cum_P_mt'][k_up]
            
            assert P_up >= P_low - 1e-6, \
                f"Dominance violated at H={H_target}: P_up={P_up} < P_low={P_low}"
        
        # Test 5: Interpolation preserves monotonicity
        result_interp = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=True, n_points=200)
        interp = result_interp['interpolated']
        
        assert np.all(np.diff(interp['Hq_km2']) > 0), \
            "Query grid not strictly increasing"
        assert np.all(np.diff(interp['P_low_interp']) >= 0), \
            "Interpolated lower envelope not non-decreasing"
        assert np.all(np.diff(interp['P_up_interp']) >= 0), \
            "Interpolated upper envelope not non-decreasing"
        
        # Test 6: Dominance after interpolation
        assert np.all(interp['P_up_interp'] >= interp['P_low_interp']), \
            "Dominance violated after interpolation"

    
    def test_synthetic_normal_yield_distribution(self):
        """Test with 1000 cells with normal yield distribution."""
        np.random.seed(123)
        n_cells = 1000
        
        # Normal yield distribution: mean=5, std=2, clipped to positive
        Y_mt_per_ha = np.abs(np.random.normal(5.0, 2.0, n_cells))
        Y_mt_per_ha = np.maximum(Y_mt_per_ha, 0.1)  # Ensure positive
        
        # Uniform harvest area
        H_ha = np.random.uniform(50.0, 150.0, n_cells)
        
        # Production
        P_mt = Y_mt_per_ha * H_ha
        
        # Build envelope
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=True)
        
        # Verify all mathematical properties
        lower = result['lower']
        upper = result['upper']
        
        # Conservation
        total_P = np.sum(P_mt)
        total_H_km2 = np.sum(H_ha) * 0.01
        assert np.isclose(lower['cum_P_mt'][-1], total_P, rtol=1e-10)
        assert np.isclose(upper['cum_P_mt'][-1], total_P, rtol=1e-10)
        assert np.isclose(lower['cum_H_km2'][-1], total_H_km2, rtol=1e-10)
        assert np.isclose(upper['cum_H_km2'][-1], total_H_km2, rtol=1e-10)
        
        # Monotonicity
        assert np.all(np.diff(lower['cum_P_mt']) >= 0)
        assert np.all(np.diff(upper['cum_P_mt']) >= 0)
        
        # Dominance
        interp = result['interpolated']
        assert np.all(interp['P_up_interp'] >= interp['P_low_interp'])
    
    def test_synthetic_extreme_yield_range(self):
        """Test with extreme yield range (3+ orders of magnitude)."""
        np.random.seed(456)
        n_cells = 1000
        
        # Create yield with guaranteed extreme range by using log-uniform distribution
        # This ensures we span exactly 4 orders of magnitude
        log_Y = np.random.uniform(np.log10(0.01), np.log10(100.0), n_cells)
        Y_mt_per_ha = 10 ** log_Y
        
        # Uniform harvest area
        H_ha = np.random.uniform(10.0, 100.0, n_cells)
        
        # Production
        P_mt = Y_mt_per_ha * H_ha
        
        # Build envelope with interpolation (should use log-spacing)
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=True, n_points=300)
        
        # Verify properties hold even with extreme range
        lower = result['lower']
        upper = result['upper']
        interp = result['interpolated']
        
        # Conservation
        assert np.isclose(lower['cum_P_mt'][-1], np.sum(P_mt), rtol=1e-10)
        assert np.isclose(upper['cum_P_mt'][-1], np.sum(P_mt), rtol=1e-10)
        
        # Monotonicity
        assert np.all(np.diff(interp['P_low_interp']) >= 0)
        assert np.all(np.diff(interp['P_up_interp']) >= 0)
        
        # Dominance
        assert np.all(interp['P_up_interp'] >= interp['P_low_interp'])
        
        # Check that we have extreme yield range (span >= 1000)
        span = Y_mt_per_ha.max() / Y_mt_per_ha.min()
        assert span >= 1000, f"Test should have extreme yield range (got {span:.1f}x)"

    
    def test_synthetic_known_bounds(self):
        """Test with synthetic data where min/max are known."""
        # Create data with known min and max yield cells
        n_cells = 1000
        
        # Most cells have yield around 5 mt/ha
        Y_mt_per_ha = np.full(n_cells, 5.0)
        
        # One cell with minimum yield
        Y_mt_per_ha[0] = 0.1
        
        # One cell with maximum yield
        Y_mt_per_ha[-1] = 50.0
        
        # All cells have same harvest area
        H_ha = np.full(n_cells, 100.0)
        
        # Production
        P_mt = Y_mt_per_ha * H_ha
        
        # Build envelope
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=False)
        
        lower = result['lower']
        upper = result['upper']
        
        # Lower envelope should start with minimum yield cell
        assert lower['Y_sorted'][0] == 0.1, "Lower envelope should start with min yield"
        
        # Upper envelope should start with maximum yield cell
        assert upper['Y_sorted'][0] == 50.0, "Upper envelope should start with max yield"
        
        # Lower envelope should end with maximum yield cell
        assert lower['Y_sorted'][-1] == 50.0, "Lower envelope should end with max yield"
        
        # Upper envelope should end with minimum yield cell
        assert upper['Y_sorted'][-1] == 0.1, "Upper envelope should end with min yield"


# ============================================================================
# TASK 8.2: Unit tests for validation logic
# ============================================================================

class TestValidationLogic:
    """Test input validation and unit conversion."""
    
    def test_nan_handling(self):
        """Test that NaN values are properly dropped."""
        P_mt = np.array([100.0, np.nan, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, np.nan, 100.0])
        Y_mt_per_ha = np.array([1.0, 2.0, 2.0, np.nan])
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should only have 1 valid cell (index 0)
        assert result['n_valid'] == 1
        assert result['dropped_counts']['nan_P'] == 1
        assert result['dropped_counts']['nan_H'] == 1
        assert result['dropped_counts']['nan_Y'] == 1
        
        # Valid data should be from index 0
        assert result['P_mt'][0] == 100.0
        assert result['H_ha'][0] == 100.0
        assert result['Y_mt_per_ha'][0] == 1.0
    
    def test_zero_handling(self):
        """Test that zero and negative values are properly dropped."""
        P_mt = np.array([100.0, 0.0, -50.0, 200.0])
        H_ha = np.array([100.0, 100.0, 100.0, 0.0])
        Y_mt_per_ha = None
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should have 1 valid cell (index 0)
        assert result['n_valid'] == 1
        assert result['dropped_counts']['zero_or_negative_P'] >= 1
        assert result['dropped_counts']['zero_or_negative_H'] >= 1

    
    def test_unit_conversion_single_conversion(self):
        """Test that harvest area is converted exactly once."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 200.0, 300.0])
        Y_mt_per_ha = None
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Check conversion: H_km2 = H_ha * 0.01
        expected_H_km2 = np.sum(H_ha) * 0.01
        assert np.isclose(result['total_H_km2'], expected_H_km2, rtol=1e-10)
        
        # Check that production is unchanged
        assert np.isclose(result['total_P_mt'], np.sum(P_mt), rtol=1e-10)
    
    def test_unit_conversion_conservation(self):
        """Test that totals are conserved after conversion."""
        np.random.seed(789)
        P_mt = np.random.uniform(100, 500, 100)
        H_ha = np.random.uniform(50, 150, 100)
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, None, tol=0.05)
        
        # Totals should be conserved
        assert np.isclose(result['total_P_mt'], np.sum(P_mt), rtol=1e-10)
        assert np.isclose(result['total_H_ha'], np.sum(H_ha), rtol=1e-10)
        assert np.isclose(result['total_H_km2'], np.sum(H_ha) * 0.01, rtol=1e-10)
    
    def test_yield_consistency_check_pass(self):
        """Test yield consistency check when P = Y * H."""
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        P_mt = Y_mt_per_ha * H_ha  # Exact consistency
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should pass consistency check
        assert result['yield_validation']['mismatch_count'] == 0
        assert result['yield_validation']['computed'] == False
        assert result['yield_validation'].get('recomputed', False) == False
    
    def test_yield_consistency_check_fail(self):
        """Test yield consistency check when P != Y * H."""
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P_mt = Y_mt_per_ha * H_ha
        
        # Introduce large mismatch in 2 cells (40% of cells)
        P_mt[0] = 200.0  # Should be 100
        P_mt[1] = 400.0  # Should be 200
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should detect mismatches and recompute
        assert result['yield_validation']['mismatch_pct'] > 1.0
        assert result['yield_validation']['recomputed'] == True
        
        # Recomputed yield should match P/H
        expected_Y = P_mt / H_ha
        np.testing.assert_array_almost_equal(result['Y_mt_per_ha'], expected_Y)
    
    def test_yield_computed_when_not_provided(self):
        """Test that yield is computed when not provided."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, None, tol=0.05)
        
        # Yield should be computed
        assert result['yield_validation']['computed'] == True
        
        # Computed yield should be P/H
        expected_Y = P_mt / H_ha
        np.testing.assert_array_almost_equal(result['Y_mt_per_ha'], expected_Y)
    
    def test_low_sample_warning(self):
        """Test that low sample warning is triggered for < 100 cells."""
        P_mt = np.random.uniform(100, 500, 50)
        H_ha = np.random.uniform(50, 150, 50)
        
        result = _validate_and_prepare_inputs(P_mt, H_ha, None, tol=0.05)
        
        # Should have fewer than 100 valid cells
        assert result['n_valid'] < 100


# ============================================================================
# TASK 8.3: Unit tests for discrete envelope building
# ============================================================================

class TestDiscreteEnvelopeBuilding:
    """Test discrete envelope building with known data."""

    
    def test_sorting_ascending_for_lower(self):
        """Test that lower envelope sorts by yield ascending."""
        # Create data with known yield order
        Y_mt_per_ha = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        H_km2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        P_mt = Y_mt_per_ha * H_km2 * 100  # Convert to mt (H is in km2)
        
        result = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        
        # Yields should be sorted ascending
        expected_Y_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result['Y_sorted'], expected_Y_sorted)
        
        # Check that sort order is correct
        assert np.all(np.diff(result['Y_sorted']) >= 0)
    
    def test_sorting_descending_for_upper(self):
        """Test that upper envelope sorts by yield descending."""
        # Create data with known yield order
        Y_mt_per_ha = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        H_km2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        P_mt = Y_mt_per_ha * H_km2 * 100
        
        result = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=False)
        
        # Yields should be sorted descending
        expected_Y_sorted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        np.testing.assert_array_equal(result['Y_sorted'], expected_Y_sorted)
        
        # Check that sort order is correct
        assert np.all(np.diff(result['Y_sorted']) <= 0)
    
    def test_cumsum_hand_calculated(self):
        """Test cumulative sum with hand-calculated example."""
        # Simple example: 3 cells
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0])
        H_km2 = np.array([10.0, 20.0, 30.0])
        P_mt = np.array([100.0, 200.0, 300.0])
        
        # Lower envelope (ascending): cells in order [0, 1, 2]
        result_low = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        
        # Hand-calculated cumulative sums
        expected_cum_H = np.array([10.0, 30.0, 60.0])
        expected_cum_P = np.array([100.0, 300.0, 600.0])
        
        np.testing.assert_array_almost_equal(result_low['cum_H_km2'], expected_cum_H)
        np.testing.assert_array_almost_equal(result_low['cum_P_mt'], expected_cum_P)
        
        # Upper envelope (descending): cells in order [2, 1, 0]
        result_up = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=False)
        
        # Hand-calculated cumulative sums (reverse order)
        expected_cum_H_up = np.array([30.0, 50.0, 60.0])
        expected_cum_P_up = np.array([300.0, 500.0, 600.0])
        
        np.testing.assert_array_almost_equal(result_up['cum_H_km2'], expected_cum_H_up)
        np.testing.assert_array_almost_equal(result_up['cum_P_mt'], expected_cum_P_up)
    
    def test_cumsum_final_equals_total(self):
        """Test that cumsum[-1] equals sum(inputs)."""
        np.random.seed(999)
        Y_mt_per_ha = np.random.uniform(1, 10, 100)
        H_km2 = np.random.uniform(1, 10, 100)
        P_mt = np.random.uniform(100, 1000, 100)
        
        # Build both envelopes
        result_low = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        result_up = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=False)
        
        # Calculate expected totals
        total_H = np.sum(H_km2)
        total_P = np.sum(P_mt)
        
        # Check lower envelope
        assert np.isclose(result_low['cum_H_km2'][-1], total_H, rtol=1e-10)
        assert np.isclose(result_low['cum_P_mt'][-1], total_P, rtol=1e-10)
        
        # Check upper envelope
        assert np.isclose(result_up['cum_H_km2'][-1], total_H, rtol=1e-10)
        assert np.isclose(result_up['cum_P_mt'][-1], total_P, rtol=1e-10)
    
    def test_monotonicity_of_cumsum(self):
        """Test that cumulative sums are monotonic."""
        np.random.seed(111)
        Y_mt_per_ha = np.random.uniform(1, 10, 50)
        H_km2 = np.random.uniform(1, 10, 50)
        P_mt = np.random.uniform(100, 1000, 50)
        
        result = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        
        # Cumulative harvest should be strictly increasing
        assert np.all(np.diff(result['cum_H_km2']) > 0)
        
        # Cumulative production should be non-decreasing
        assert np.all(np.diff(result['cum_P_mt']) >= 0)


# ============================================================================
# TASK 8.4: Unit tests for interpolation (CRITICAL)
# ============================================================================

class TestInterpolation:
    """Test interpolation preserves mathematical properties."""

    
    def test_np_interp_preserves_monotonicity(self):
        """Test that np.interp preserves monotonicity on monotonic input."""
        # Create monotonic input
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        # Create query points
        xq = np.linspace(1.0, 5.0, 50)
        
        # Interpolate
        yq = np.interp(xq, x, y)
        
        # Check monotonicity preserved
        assert np.all(np.diff(yq) >= 0), "np.interp should preserve monotonicity"
    
    def test_no_extrapolation_in_query_grid(self):
        """Test that query grid doesn't extrapolate beyond data range."""
        # Create discrete sequences
        cum_H_low = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cum_H_up = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        # Create query grid
        Hq = _create_query_grid(cum_H_low, cum_H_up, n_points=100)
        
        # Check no extrapolation
        H_min = min(cum_H_low[0], cum_H_up[0])
        H_max = max(cum_H_low[-1], cum_H_up[-1])
        
        assert Hq[0] >= H_min - 1e-10, "Query grid extrapolates below minimum"
        assert Hq[-1] <= H_max + 1e-10, "Query grid extrapolates above maximum"
    
    def test_query_grid_strictly_increasing(self):
        """Test that query grid is strictly increasing."""
        cum_H_low = np.linspace(1.0, 100.0, 50)
        cum_H_up = np.linspace(1.0, 100.0, 50)
        
        Hq = _create_query_grid(cum_H_low, cum_H_up, n_points=200)
        
        # Check strictly increasing
        assert np.all(np.diff(Hq) > 0), "Query grid not strictly increasing"
    
    def test_interpolation_preserves_monotonicity(self):
        """Test that interpolation preserves monotonicity of envelopes."""
        # Create monotonic discrete sequences
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H_km2 = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        P_mt = Y_mt_per_ha * H_km2 * 10
        
        lower = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        upper = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=False)
        
        # Interpolate
        result = _interpolate_envelopes(lower, upper, n_points=100)
        
        # Check monotonicity preserved
        assert np.all(np.diff(result['P_low_interp']) >= 0), \
            "Interpolation broke monotonicity of lower envelope"
        assert np.all(np.diff(result['P_up_interp']) >= 0), \
            "Interpolation broke monotonicity of upper envelope"
    
    def test_dominance_after_interpolation(self):
        """Test that upper >= lower after interpolation."""
        # Create data where dominance should hold
        np.random.seed(222)
        Y_mt_per_ha = np.random.uniform(1, 10, 100)
        H_km2 = np.random.uniform(5, 15, 100)
        P_mt = Y_mt_per_ha * H_km2 * 10
        
        lower = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        upper = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=False)
        
        # Interpolate
        result = _interpolate_envelopes(lower, upper, n_points=200)
        
        # Check dominance
        assert np.all(result['P_up_interp'] >= result['P_low_interp']), \
            "Dominance violated after interpolation"
    
    def test_interpolation_with_log_spacing(self):
        """Test interpolation with log-spaced query grid (extreme range)."""
        # Create data with extreme range (4 orders of magnitude)
        Y_mt_per_ha = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        H_km2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        P_mt = Y_mt_per_ha * H_km2 * 100
        
        lower = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=True)
        upper = _build_discrete_sequence(Y_mt_per_ha, H_km2, P_mt, ascending=False)
        
        # Interpolate (should use log-spacing internally)
        result = _interpolate_envelopes(lower, upper, n_points=300)
        
        # Check properties still hold
        assert np.all(np.diff(result['Hq_km2']) > 0)
        assert np.all(np.diff(result['P_low_interp']) >= 0)
        assert np.all(np.diff(result['P_up_interp']) >= 0)
        assert np.all(result['P_up_interp'] >= result['P_low_interp'])


# ============================================================================
# TASK 8.5: Integration tests with real data (placeholder)
# ============================================================================

class TestRealDataIntegration:
    """Integration tests with real agricultural data."""
    
    @pytest.mark.skip(reason="Requires real data files - implement after unit tests pass")
    def test_wheat_dataset(self):
        """Test with real wheat dataset."""
        # TODO: Load real wheat data
        # TODO: Build envelope
        # TODO: Verify outputs match expected format
        # TODO: Check all QA assertions pass
        pass
    
    @pytest.mark.skip(reason="Requires real data files - implement after unit tests pass")
    def test_rice_dataset(self):
        """Test with real rice dataset."""
        pass
    
    @pytest.mark.skip(reason="Requires real data files - implement after unit tests pass")
    def test_allgrain_dataset(self):
        """Test with real allgrain dataset."""
        pass
    
    @pytest.mark.skip(reason="Requires MATLAB reference - implement after unit tests pass")
    def test_compare_with_matlab_implementation(self):
        """Compare outputs with current MATLAB-exact implementation."""
        pass

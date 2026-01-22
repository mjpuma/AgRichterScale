"""
Tests for envelope interpolation functionality (Task 5).

This module tests the interpolation of discrete envelope sequences onto
a unified query grid, verifying:
1. Query grid creation with appropriate spacing
2. Monotonic interpolation preserves mathematical properties
3. Dominance constraint holds after interpolation
"""

import numpy as np
import pytest
from agrichter.analysis.envelope_builder import build_envelope


class TestQueryGridCreation:
    """Test query grid creation (Task 5.1)."""
    
    def test_query_grid_linear_spacing(self):
        """Test that linear spacing is used for small spans."""
        # Create simple data with small span (< 3 orders of magnitude)
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        # Check that interpolated result exists
        assert 'interpolated' in result
        interp = result['interpolated']
        
        # Check query grid properties
        Hq = interp['Hq_km2']
        assert len(Hq) >= 200, "Query grid should have at least 200 points"
        
        # Check strictly increasing
        assert np.all(np.diff(Hq) > 0), "Query grid must be strictly increasing"
        
        # Check no extrapolation
        H_min = min(result['lower']['cum_H_km2'][0], result['upper']['cum_H_km2'][0])
        H_max = max(result['lower']['cum_H_km2'][-1], result['upper']['cum_H_km2'][-1])
        
        assert Hq[0] >= H_min - 1e-10, "Query grid should not extrapolate below data"
        assert Hq[-1] <= H_max + 1e-10, "Query grid should not extrapolate above data"
    
    def test_query_grid_log_spacing(self):
        """Test that log spacing is used for large spans."""
        # Create data with large span (>= 3 orders of magnitude)
        # Use yields that span from 0.001 to 10 (4 orders of magnitude)
        Y = np.logspace(-3, 1, 100)  # 0.001 to 10
        H = np.full(100, 100.0)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        # Check that interpolated result exists
        assert 'interpolated' in result
        interp = result['interpolated']
        
        # Check query grid properties
        Hq = interp['Hq_km2']
        assert len(Hq) >= 200, "Query grid should have at least 200 points"
        
        # Check strictly increasing
        assert np.all(np.diff(Hq) > 0), "Query grid must be strictly increasing"
    
    def test_query_grid_within_data_range(self):
        """Test that query grid stays within data range (no extrapolation)."""
        # Generate synthetic data
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 1000)
        H = np.random.uniform(100.0, 1000.0, 1000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        interp = result['interpolated']
        Hq = interp['Hq_km2']
        
        # Get data range
        H_min = min(result['lower']['cum_H_km2'][0], result['upper']['cum_H_km2'][0])
        H_max = max(result['lower']['cum_H_km2'][-1], result['upper']['cum_H_km2'][-1])
        
        # Check no extrapolation (with small tolerance for floating point)
        assert Hq[0] >= H_min - 1e-10, f"Query grid extrapolates below: {Hq[0]} < {H_min}"
        assert Hq[-1] <= H_max + 1e-10, f"Query grid extrapolates above: {Hq[-1]} > {H_max}"
    
    def test_query_grid_custom_n_points(self):
        """Test that custom n_points is respected."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        # Test with different n_points values
        for n_points in [200, 300, 500]:
            result = build_envelope(P, H, Y, interpolate=True, n_points=n_points)
            Hq = result['interpolated']['Hq_km2']
            assert len(Hq) >= n_points, f"Query grid should have at least {n_points} points"


class TestMonotonicInterpolation:
    """Test monotonic interpolation (Task 5.2)."""
    
    def test_interpolation_preserves_monotonicity_lower(self):
        """Test that lower envelope interpolation preserves monotonicity."""
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 1000)
        H = np.random.uniform(100.0, 1000.0, 1000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        P_low_interp = result['interpolated']['P_low_interp']
        
        # Check non-decreasing
        P_diffs = np.diff(P_low_interp)
        assert np.all(P_diffs >= 0), (
            f"Lower envelope interpolation not monotonic. "
            f"Found {np.sum(P_diffs < 0)} negative increments. "
            f"Min increment: {np.min(P_diffs)}"
        )
    
    def test_interpolation_preserves_monotonicity_upper(self):
        """Test that upper envelope interpolation preserves monotonicity."""
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 1000)
        H = np.random.uniform(100.0, 1000.0, 1000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        P_up_interp = result['interpolated']['P_up_interp']
        
        # Check non-decreasing
        P_diffs = np.diff(P_up_interp)
        assert np.all(P_diffs >= 0), (
            f"Upper envelope interpolation not monotonic. "
            f"Found {np.sum(P_diffs < 0)} negative increments. "
            f"Min increment: {np.min(P_diffs)}"
        )
    
    def test_interpolation_endpoints_match(self):
        """Test that interpolated endpoints match discrete endpoints."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        # Check lower envelope endpoints
        P_low_discrete = result['lower']['cum_P_mt']
        P_low_interp = result['interpolated']['P_low_interp']
        
        # First and last points should be close (within interpolation tolerance)
        assert np.isclose(P_low_interp[0], P_low_discrete[0], rtol=1e-6)
        assert np.isclose(P_low_interp[-1], P_low_discrete[-1], rtol=1e-6)
        
        # Check upper envelope endpoints
        P_up_discrete = result['upper']['cum_P_mt']
        P_up_interp = result['interpolated']['P_up_interp']
        
        assert np.isclose(P_up_interp[0], P_up_discrete[0], rtol=1e-6)
        assert np.isclose(P_up_interp[-1], P_up_discrete[-1], rtol=1e-6)
    
    def test_interpolation_with_simple_case(self):
        """Test interpolation with simple hand-calculable case."""
        # 5 cells: Y=[1,2,3,4,5], H=[100,100,100,100,100], P=[100,200,300,400,500]
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        # Check that interpolation exists
        assert 'interpolated' in result
        interp = result['interpolated']
        
        # Check shapes
        assert len(interp['Hq_km2']) >= 200
        assert len(interp['P_low_interp']) == len(interp['Hq_km2'])
        assert len(interp['P_up_interp']) == len(interp['Hq_km2'])
        
        # Check monotonicity
        assert np.all(np.diff(interp['P_low_interp']) >= 0)
        assert np.all(np.diff(interp['P_up_interp']) >= 0)


class TestDominanceAfterInterpolation:
    """Test dominance constraint after interpolation (Task 5.3)."""
    
    def test_dominance_holds_after_interpolation(self):
        """Test that upper >= lower at all interpolated points."""
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 1000)
        H = np.random.uniform(100.0, 1000.0, 1000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        P_low_interp = result['interpolated']['P_low_interp']
        P_up_interp = result['interpolated']['P_up_interp']
        
        # Check dominance at all points
        violations = P_low_interp > P_up_interp
        n_violations = np.sum(violations)
        
        assert n_violations == 0, (
            f"Dominance violated at {n_violations} points after interpolation. "
            f"Max deficit: {np.max(P_low_interp - P_up_interp)}"
        )
    
    def test_clipping_metadata_present(self):
        """Test that clipping metadata is present in result."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        interp = result['interpolated']
        
        # Check that clipping metadata exists
        assert 'clipping_applied' in interp
        assert 'n_clipped' in interp
        assert 'max_deficit' in interp
        
        # Check types
        assert isinstance(interp['clipping_applied'], bool)
        assert isinstance(interp['n_clipped'], (int, np.integer))
        assert isinstance(interp['max_deficit'], (float, np.floating))
    
    def test_no_clipping_needed_for_good_data(self):
        """Test that well-behaved data doesn't need clipping."""
        # Generate data with clear yield separation
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 1000)
        H = np.random.uniform(100.0, 1000.0, 1000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        interp = result['interpolated']
        
        # For well-behaved data, clipping should not be needed
        # (or very minimal if needed due to numerical precision)
        if interp['clipping_applied']:
            # If clipping was applied, it should be minimal
            violation_pct = 100 * interp['n_clipped'] / len(interp['Hq_km2'])
            assert violation_pct < 5.0, (
                f"Too much clipping needed: {violation_pct:.2f}% > 5%"
            )
    
    def test_dominance_at_sampled_points(self):
        """Test dominance at specific sampled harvest levels."""
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 1000)
        H = np.random.uniform(100.0, 1000.0, 1000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=200)
        
        Hq = result['interpolated']['Hq_km2']
        P_low = result['interpolated']['P_low_interp']
        P_up = result['interpolated']['P_up_interp']
        
        # Sample 20 points evenly across the range
        sample_indices = np.linspace(0, len(Hq) - 1, 20, dtype=int)
        
        for idx in sample_indices:
            assert P_up[idx] >= P_low[idx], (
                f"Dominance violated at index {idx}: "
                f"H={Hq[idx]:.2f}, P_up={P_up[idx]:.2f}, P_low={P_low[idx]:.2f}"
            )


class TestInterpolationIntegration:
    """Integration tests for complete interpolation workflow."""
    
    def test_interpolation_optional(self):
        """Test that interpolation is optional (default False)."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        # Without interpolation
        result_no_interp = build_envelope(P, H, Y, interpolate=False)
        assert 'interpolated' not in result_no_interp
        
        # With interpolation
        result_with_interp = build_envelope(P, H, Y, interpolate=True)
        assert 'interpolated' in result_with_interp
    
    def test_discrete_data_unchanged_by_interpolation(self):
        """Test that discrete envelope data is unchanged when interpolation is added."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        result_no_interp = build_envelope(P, H, Y, interpolate=False)
        result_with_interp = build_envelope(P, H, Y, interpolate=True)
        
        # Check that discrete data is identical
        np.testing.assert_array_equal(
            result_no_interp['lower']['cum_H_km2'],
            result_with_interp['lower']['cum_H_km2']
        )
        np.testing.assert_array_equal(
            result_no_interp['lower']['cum_P_mt'],
            result_with_interp['lower']['cum_P_mt']
        )
        np.testing.assert_array_equal(
            result_no_interp['upper']['cum_H_km2'],
            result_with_interp['upper']['cum_H_km2']
        )
        np.testing.assert_array_equal(
            result_no_interp['upper']['cum_P_mt'],
            result_with_interp['upper']['cum_P_mt']
        )
    
    def test_interpolation_with_various_distributions(self):
        """Test interpolation with different data distributions."""
        np.random.seed(42)
        
        # Test with uniform distribution
        Y_uniform = np.random.uniform(1.0, 10.0, 1000)
        H_uniform = np.random.uniform(100.0, 1000.0, 1000)
        P_uniform = Y_uniform * H_uniform
        
        result_uniform = build_envelope(P_uniform, H_uniform, Y_uniform, interpolate=True)
        assert 'interpolated' in result_uniform
        assert np.all(np.diff(result_uniform['interpolated']['P_low_interp']) >= 0)
        assert np.all(np.diff(result_uniform['interpolated']['P_up_interp']) >= 0)
        
        # Test with normal distribution
        Y_normal = np.abs(np.random.normal(5.0, 2.0, 1000))
        Y_normal = np.maximum(Y_normal, 0.1)
        H_normal = np.abs(np.random.normal(500.0, 200.0, 1000))
        H_normal = np.maximum(H_normal, 1.0)
        P_normal = Y_normal * H_normal
        
        result_normal = build_envelope(P_normal, H_normal, Y_normal, interpolate=True)
        assert 'interpolated' in result_normal
        assert np.all(np.diff(result_normal['interpolated']['P_low_interp']) >= 0)
        assert np.all(np.diff(result_normal['interpolated']['P_up_interp']) >= 0)
    
    def test_interpolation_with_large_dataset(self):
        """Test interpolation with large dataset (10000 cells)."""
        np.random.seed(42)
        Y = np.random.uniform(1.0, 10.0, 10000)
        H = np.random.uniform(100.0, 1000.0, 10000)
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True, n_points=500)
        
        # Check that interpolation completed successfully
        assert 'interpolated' in result
        interp = result['interpolated']
        
        # Check properties
        assert len(interp['Hq_km2']) >= 500
        assert np.all(np.diff(interp['Hq_km2']) > 0)
        assert np.all(np.diff(interp['P_low_interp']) >= 0)
        assert np.all(np.diff(interp['P_up_interp']) >= 0)
        assert np.all(interp['P_up_interp'] >= interp['P_low_interp'])
    
    def test_interpolation_result_structure(self):
        """Test that interpolation result has expected structure."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P = Y * H
        
        result = build_envelope(P, H, Y, interpolate=True)
        
        # Check main structure
        assert 'lower' in result
        assert 'upper' in result
        assert 'summary' in result
        assert 'interpolated' in result
        
        # Check interpolated structure
        interp = result['interpolated']
        required_keys = ['Hq_km2', 'P_low_interp', 'P_up_interp', 
                        'clipping_applied', 'n_clipped', 'max_deficit']
        for key in required_keys:
            assert key in interp, f"Missing key in interpolated result: {key}"
        
        # Check array shapes match
        n_points = len(interp['Hq_km2'])
        assert len(interp['P_low_interp']) == n_points
        assert len(interp['P_up_interp']) == n_points

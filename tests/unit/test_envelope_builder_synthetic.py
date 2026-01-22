"""
Synthetic data generation and mathematical property tests for envelope builder.

This module provides utilities to generate controlled test data with known properties
and tests to verify mathematical invariants of the envelope building process.
"""

import numpy as np
import pytest
from typing import Dict, Tuple


class SyntheticDataGenerator:
    """Generate synthetic agricultural data with known properties for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
    
    def generate_uniform_yield(
        self, 
        n_cells: int = 1000,
        yield_min: float = 1.0,
        yield_max: float = 10.0,
        area_min: float = 100.0,
        area_max: float = 1000.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic data with uniform yield distribution.
        
        Parameters
        ----------
        n_cells : int
            Number of grid cells to generate
        yield_min : float
            Minimum yield (mt/ha)
        yield_max : float
            Maximum yield (mt/ha)
        area_min : float
            Minimum harvest area per cell (ha)
        area_max : float
            Maximum harvest area per cell (ha)
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'Y_mt_per_ha': yield array (mt/ha)
            - 'H_ha': harvest area array (ha)
            - 'P_mt': production array (mt)
            - 'known_totals': dict with sum(P_mt), sum(H_ha)
            - 'known_bounds': dict with min/max yield values
        """
        # Generate yields uniformly
        Y_mt_per_ha = self.rng.uniform(yield_min, yield_max, n_cells)
        
        # Generate harvest areas uniformly
        H_ha = self.rng.uniform(area_min, area_max, n_cells)
        
        # Calculate production: P = Y * H
        P_mt = Y_mt_per_ha * H_ha
        
        # Calculate known properties
        known_totals = {
            'sum_P_mt': np.sum(P_mt),
            'sum_H_ha': np.sum(H_ha),
            'sum_H_km2': np.sum(H_ha) * 0.01
        }
        
        known_bounds = {
            'min_yield': np.min(Y_mt_per_ha),
            'max_yield': np.max(Y_mt_per_ha),
            'min_yield_idx': np.argmin(Y_mt_per_ha),
            'max_yield_idx': np.argmax(Y_mt_per_ha)
        }
        
        return {
            'Y_mt_per_ha': Y_mt_per_ha,
            'H_ha': H_ha,
            'P_mt': P_mt,
            'known_totals': known_totals,
            'known_bounds': known_bounds,
            'n_cells': n_cells
        }
    
    def generate_normal_yield(
        self,
        n_cells: int = 1000,
        yield_mean: float = 5.0,
        yield_std: float = 2.0,
        area_mean: float = 500.0,
        area_std: float = 200.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic data with normal (Gaussian) yield distribution.
        
        Parameters
        ----------
        n_cells : int
            Number of grid cells to generate
        yield_mean : float
            Mean yield (mt/ha)
        yield_std : float
            Standard deviation of yield (mt/ha)
        area_mean : float
            Mean harvest area per cell (ha)
        area_std : float
            Standard deviation of harvest area (ha)
            
        Returns
        -------
        dict
            Dictionary with same structure as generate_uniform_yield
        """
        # Generate yields from normal distribution, ensure positive
        Y_mt_per_ha = np.abs(self.rng.normal(yield_mean, yield_std, n_cells))
        Y_mt_per_ha = np.maximum(Y_mt_per_ha, 0.1)  # Ensure minimum positive value
        
        # Generate harvest areas from normal distribution, ensure positive
        H_ha = np.abs(self.rng.normal(area_mean, area_std, n_cells))
        H_ha = np.maximum(H_ha, 1.0)  # Ensure minimum positive value
        
        # Calculate production: P = Y * H
        P_mt = Y_mt_per_ha * H_ha
        
        # Calculate known properties
        known_totals = {
            'sum_P_mt': np.sum(P_mt),
            'sum_H_ha': np.sum(H_ha),
            'sum_H_km2': np.sum(H_ha) * 0.01
        }
        
        known_bounds = {
            'min_yield': np.min(Y_mt_per_ha),
            'max_yield': np.max(Y_mt_per_ha),
            'min_yield_idx': np.argmin(Y_mt_per_ha),
            'max_yield_idx': np.argmax(Y_mt_per_ha)
        }
        
        return {
            'Y_mt_per_ha': Y_mt_per_ha,
            'H_ha': H_ha,
            'P_mt': P_mt,
            'known_totals': known_totals,
            'known_bounds': known_bounds,
            'n_cells': n_cells
        }
    
    def generate_bimodal_yield(
        self,
        n_cells: int = 1000,
        low_yield_mean: float = 2.0,
        high_yield_mean: float = 8.0,
        yield_std: float = 0.5,
        area_mean: float = 500.0,
        area_std: float = 100.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic data with bimodal yield distribution.
        
        Useful for testing edge cases where yields cluster in two groups.
        
        Parameters
        ----------
        n_cells : int
            Number of grid cells to generate
        low_yield_mean : float
            Mean of low-yield cluster (mt/ha)
        high_yield_mean : float
            Mean of high-yield cluster (mt/ha)
        yield_std : float
            Standard deviation within each cluster (mt/ha)
        area_mean : float
            Mean harvest area per cell (ha)
        area_std : float
            Standard deviation of harvest area (ha)
            
        Returns
        -------
        dict
            Dictionary with same structure as generate_uniform_yield
        """
        # Split cells into two groups
        n_low = n_cells // 2
        n_high = n_cells - n_low
        
        # Generate low-yield cluster
        Y_low = np.abs(self.rng.normal(low_yield_mean, yield_std, n_low))
        Y_low = np.maximum(Y_low, 0.1)
        
        # Generate high-yield cluster
        Y_high = np.abs(self.rng.normal(high_yield_mean, yield_std, n_high))
        Y_high = np.maximum(Y_high, 0.1)
        
        # Combine and shuffle
        Y_mt_per_ha = np.concatenate([Y_low, Y_high])
        shuffle_idx = self.rng.permutation(n_cells)
        Y_mt_per_ha = Y_mt_per_ha[shuffle_idx]
        
        # Generate harvest areas
        H_ha = np.abs(self.rng.normal(area_mean, area_std, n_cells))
        H_ha = np.maximum(H_ha, 1.0)
        
        # Calculate production
        P_mt = Y_mt_per_ha * H_ha
        
        # Calculate known properties
        known_totals = {
            'sum_P_mt': np.sum(P_mt),
            'sum_H_ha': np.sum(H_ha),
            'sum_H_km2': np.sum(H_ha) * 0.01
        }
        
        known_bounds = {
            'min_yield': np.min(Y_mt_per_ha),
            'max_yield': np.max(Y_mt_per_ha),
            'min_yield_idx': np.argmin(Y_mt_per_ha),
            'max_yield_idx': np.argmax(Y_mt_per_ha)
        }
        
        return {
            'Y_mt_per_ha': Y_mt_per_ha,
            'H_ha': H_ha,
            'P_mt': P_mt,
            'known_totals': known_totals,
            'known_bounds': known_bounds,
            'n_cells': n_cells
        }
    
    def generate_simple_test_case(self) -> Dict[str, np.ndarray]:
        """
        Generate a very simple test case with hand-calculable values.
        
        Returns
        -------
        dict
            Dictionary with 5 cells with simple, round numbers
        """
        # 5 cells with simple values
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P_mt = Y_mt_per_ha * H_ha  # [100, 200, 300, 400, 500]
        
        known_totals = {
            'sum_P_mt': 1500.0,
            'sum_H_ha': 500.0,
            'sum_H_km2': 5.0
        }
        
        known_bounds = {
            'min_yield': 1.0,
            'max_yield': 5.0,
            'min_yield_idx': 0,
            'max_yield_idx': 4
        }
        
        return {
            'Y_mt_per_ha': Y_mt_per_ha,
            'H_ha': H_ha,
            'P_mt': P_mt,
            'known_totals': known_totals,
            'known_bounds': known_bounds,
            'n_cells': 5
        }


# Fixtures for pytest
@pytest.fixture
def synthetic_generator():
    """Provide a synthetic data generator with fixed seed."""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def uniform_data_1000(synthetic_generator):
    """Generate 1000 cells with uniform yield distribution."""
    return synthetic_generator.generate_uniform_yield(n_cells=1000)


@pytest.fixture
def normal_data_1000(synthetic_generator):
    """Generate 1000 cells with normal yield distribution."""
    return synthetic_generator.generate_normal_yield(n_cells=1000)


@pytest.fixture
def bimodal_data_1000(synthetic_generator):
    """Generate 1000 cells with bimodal yield distribution."""
    return synthetic_generator.generate_bimodal_yield(n_cells=1000)


@pytest.fixture
def simple_test_case(synthetic_generator):
    """Generate simple 5-cell test case with hand-calculable values."""
    return synthetic_generator.generate_simple_test_case()


# Test that synthetic data has expected properties
class TestSyntheticDataGeneration:
    """Test that synthetic data generators produce valid data."""
    
    def test_uniform_data_shape(self, uniform_data_1000):
        """Test that uniform data has correct shape."""
        assert uniform_data_1000['Y_mt_per_ha'].shape == (1000,)
        assert uniform_data_1000['H_ha'].shape == (1000,)
        assert uniform_data_1000['P_mt'].shape == (1000,)
    
    def test_uniform_data_positive(self, uniform_data_1000):
        """Test that uniform data has all positive values."""
        assert np.all(uniform_data_1000['Y_mt_per_ha'] > 0)
        assert np.all(uniform_data_1000['H_ha'] > 0)
        assert np.all(uniform_data_1000['P_mt'] > 0)
    
    def test_uniform_data_consistency(self, uniform_data_1000):
        """Test that P = Y * H for uniform data."""
        Y = uniform_data_1000['Y_mt_per_ha']
        H = uniform_data_1000['H_ha']
        P = uniform_data_1000['P_mt']
        P_expected = Y * H
        np.testing.assert_allclose(P, P_expected, rtol=1e-10)
    
    def test_uniform_data_totals(self, uniform_data_1000):
        """Test that known totals match actual sums."""
        P_sum = np.sum(uniform_data_1000['P_mt'])
        H_sum = np.sum(uniform_data_1000['H_ha'])
        
        assert np.isclose(P_sum, uniform_data_1000['known_totals']['sum_P_mt'])
        assert np.isclose(H_sum, uniform_data_1000['known_totals']['sum_H_ha'])
        assert np.isclose(H_sum * 0.01, uniform_data_1000['known_totals']['sum_H_km2'])
    
    def test_uniform_data_bounds(self, uniform_data_1000):
        """Test that known bounds match actual min/max."""
        Y = uniform_data_1000['Y_mt_per_ha']
        bounds = uniform_data_1000['known_bounds']
        
        assert np.isclose(np.min(Y), bounds['min_yield'])
        assert np.isclose(np.max(Y), bounds['max_yield'])
        assert Y[bounds['min_yield_idx']] == bounds['min_yield']
        assert Y[bounds['max_yield_idx']] == bounds['max_yield']
    
    def test_normal_data_shape(self, normal_data_1000):
        """Test that normal data has correct shape."""
        assert normal_data_1000['Y_mt_per_ha'].shape == (1000,)
        assert normal_data_1000['H_ha'].shape == (1000,)
        assert normal_data_1000['P_mt'].shape == (1000,)
    
    def test_normal_data_positive(self, normal_data_1000):
        """Test that normal data has all positive values."""
        assert np.all(normal_data_1000['Y_mt_per_ha'] > 0)
        assert np.all(normal_data_1000['H_ha'] > 0)
        assert np.all(normal_data_1000['P_mt'] > 0)
    
    def test_bimodal_data_shape(self, bimodal_data_1000):
        """Test that bimodal data has correct shape."""
        assert bimodal_data_1000['Y_mt_per_ha'].shape == (1000,)
        assert bimodal_data_1000['H_ha'].shape == (1000,)
        assert bimodal_data_1000['P_mt'].shape == (1000,)
    
    def test_simple_case_values(self, simple_test_case):
        """Test that simple case has expected hand-calculated values."""
        assert simple_test_case['n_cells'] == 5
        assert simple_test_case['known_totals']['sum_P_mt'] == 1500.0
        assert simple_test_case['known_totals']['sum_H_ha'] == 500.0
        assert simple_test_case['known_totals']['sum_H_km2'] == 5.0
        
        # Check individual values
        np.testing.assert_array_equal(
            simple_test_case['Y_mt_per_ha'], 
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        np.testing.assert_array_equal(
            simple_test_case['P_mt'], 
            [100.0, 200.0, 300.0, 400.0, 500.0]
        )



# Helper functions for envelope building (minimal implementation for testing)
def build_discrete_envelope(Y_mt_per_ha: np.ndarray, H_ha: np.ndarray, P_mt: np.ndarray) -> Dict:
    """
    Build discrete envelope sequences (no interpolation).
    
    This is a minimal implementation for testing mathematical properties.
    The full implementation will be in agrichter/analysis/envelope_builder.py
    
    Parameters
    ----------
    Y_mt_per_ha : np.ndarray
        Yield per hectare (mt/ha)
    H_ha : np.ndarray
        Harvest area (ha)
    P_mt : np.ndarray
        Production (mt)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'lower': dict with 'cum_H_km2', 'cum_P_mt', 'Y_sorted'
        - 'upper': dict with 'cum_H_km2', 'cum_P_mt', 'Y_sorted'
    """
    # Convert H to km2
    H_km2 = H_ha * 0.01
    
    # Lower envelope: sort by yield ascending (least productive first)
    idx_low = np.argsort(Y_mt_per_ha)
    Y_low = Y_mt_per_ha[idx_low]
    H_low_km2 = H_km2[idx_low]
    P_low_mt = P_mt[idx_low]
    
    # Cumulative sums for lower envelope
    cum_H_low_km2 = np.cumsum(H_low_km2)
    cum_P_low_mt = np.cumsum(P_low_mt)
    
    # Upper envelope: sort by yield descending (most productive first)
    idx_up = np.argsort(Y_mt_per_ha)[::-1]
    Y_up = Y_mt_per_ha[idx_up]
    H_up_km2 = H_km2[idx_up]
    P_up_mt = P_mt[idx_up]
    
    # Cumulative sums for upper envelope
    cum_H_up_km2 = np.cumsum(H_up_km2)
    cum_P_up_mt = np.cumsum(P_up_mt)
    
    return {
        'lower': {
            'cum_H_km2': cum_H_low_km2,
            'cum_P_mt': cum_P_low_mt,
            'Y_sorted': Y_low
        },
        'upper': {
            'cum_H_km2': cum_H_up_km2,
            'cum_P_mt': cum_P_up_mt,
            'Y_sorted': Y_up
        }
    }


class TestMathematicalProperties:
    """
    Test mathematical properties that MUST hold for envelope building.
    
    These tests verify fundamental invariants:
    1. Cumulative sums are monotonic non-decreasing
    2. Upper envelope >= lower envelope at all cumulative harvest levels
    3. Total production matches sum(P_i)
    4. Yield ordering is correct (ascending/descending)
    
    These properties MUST pass on synthetic data before proceeding to real data.
    """
    
    def test_cumsum_monotonic_lower(self, uniform_data_1000):
        """Test that lower envelope cumulative sums are monotonic non-decreasing."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_H = envelope['lower']['cum_H_km2']
        cum_P = envelope['lower']['cum_P_mt']
        
        # Check monotonicity: diff should be >= 0
        assert np.all(np.diff(cum_H) > 0), "Cumulative harvest must be strictly increasing"
        assert np.all(np.diff(cum_P) >= 0), "Cumulative production must be non-decreasing"
    
    def test_cumsum_monotonic_upper(self, uniform_data_1000):
        """Test that upper envelope cumulative sums are monotonic non-decreasing."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_H = envelope['upper']['cum_H_km2']
        cum_P = envelope['upper']['cum_P_mt']
        
        # Check monotonicity: diff should be >= 0
        assert np.all(np.diff(cum_H) > 0), "Cumulative harvest must be strictly increasing"
        assert np.all(np.diff(cum_P) >= 0), "Cumulative production must be non-decreasing"
    
    def test_total_production_conservation_lower(self, uniform_data_1000):
        """Test that total production is conserved in lower envelope."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_P_final = envelope['lower']['cum_P_mt'][-1]
        expected_total = uniform_data_1000['known_totals']['sum_P_mt']
        
        np.testing.assert_allclose(
            cum_P_final, 
            expected_total, 
            rtol=1e-10,
            err_msg="Final cumulative production must equal sum(P_i)"
        )
    
    def test_total_production_conservation_upper(self, uniform_data_1000):
        """Test that total production is conserved in upper envelope."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_P_final = envelope['upper']['cum_P_mt'][-1]
        expected_total = uniform_data_1000['known_totals']['sum_P_mt']
        
        np.testing.assert_allclose(
            cum_P_final, 
            expected_total, 
            rtol=1e-10,
            err_msg="Final cumulative production must equal sum(P_i)"
        )
    
    def test_total_harvest_conservation_lower(self, uniform_data_1000):
        """Test that total harvest area is conserved in lower envelope."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_H_final = envelope['lower']['cum_H_km2'][-1]
        expected_total = uniform_data_1000['known_totals']['sum_H_km2']
        
        np.testing.assert_allclose(
            cum_H_final, 
            expected_total, 
            rtol=1e-10,
            err_msg="Final cumulative harvest must equal sum(H_i) * 0.01"
        )
    
    def test_total_harvest_conservation_upper(self, uniform_data_1000):
        """Test that total harvest area is conserved in upper envelope."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_H_final = envelope['upper']['cum_H_km2'][-1]
        expected_total = uniform_data_1000['known_totals']['sum_H_km2']
        
        np.testing.assert_allclose(
            cum_H_final, 
            expected_total, 
            rtol=1e-10,
            err_msg="Final cumulative harvest must equal sum(H_i) * 0.01"
        )
    
    def test_yield_ordering_lower_ascending(self, uniform_data_1000):
        """Test that lower envelope yields are sorted ascending."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        Y_sorted = envelope['lower']['Y_sorted']
        
        # Check that yields are in ascending order
        assert np.all(np.diff(Y_sorted) >= 0), "Lower envelope yields must be sorted ascending"
    
    def test_yield_ordering_upper_descending(self, uniform_data_1000):
        """Test that upper envelope yields are sorted descending."""
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        Y_sorted = envelope['upper']['Y_sorted']
        
        # Check that yields are in descending order
        assert np.all(np.diff(Y_sorted) <= 0), "Upper envelope yields must be sorted descending"
    
    def test_dominance_property_discrete(self, uniform_data_1000):
        """
        Test that upper envelope >= lower envelope at all cumulative harvest levels.
        
        This is the fundamental property that makes envelopes meaningful:
        at any given cumulative harvest level, the upper bound (best case) 
        should produce at least as much as the lower bound (worst case).
        """
        envelope = build_discrete_envelope(
            uniform_data_1000['Y_mt_per_ha'],
            uniform_data_1000['H_ha'],
            uniform_data_1000['P_mt']
        )
        
        cum_H_low = envelope['lower']['cum_H_km2']
        cum_P_low = envelope['lower']['cum_P_mt']
        cum_H_up = envelope['upper']['cum_H_km2']
        cum_P_up = envelope['upper']['cum_P_mt']
        
        # Test dominance at 10 evenly-spaced harvest levels
        H_min = min(cum_H_low[0], cum_H_up[0])
        H_max = max(cum_H_low[-1], cum_H_up[-1])
        H_test = np.linspace(H_min, H_max, 10)
        
        for H_target in H_test:
            # Find smallest k where cum_H >= H_target
            idx_low = np.searchsorted(cum_H_low, H_target, side='left')
            idx_up = np.searchsorted(cum_H_up, H_target, side='left')
            
            # Handle edge cases
            if idx_low >= len(cum_P_low):
                idx_low = len(cum_P_low) - 1
            if idx_up >= len(cum_P_up):
                idx_up = len(cum_P_up) - 1
            
            P_low_at_H = cum_P_low[idx_low]
            P_up_at_H = cum_P_up[idx_up]
            
            assert P_up_at_H >= P_low_at_H, (
                f"Dominance violated at H={H_target:.2f}: "
                f"upper={P_up_at_H:.2f} < lower={P_low_at_H:.2f}"
            )
    
    def test_simple_case_hand_calculation(self, simple_test_case):
        """
        Test envelope building with simple hand-calculable case.
        
        5 cells: Y=[1,2,3,4,5], H=[100,100,100,100,100], P=[100,200,300,400,500]
        
        Lower envelope (ascending yield):
        - cum_H_km2 = [1, 2, 3, 4, 5]
        - cum_P_mt = [100, 300, 600, 1000, 1500]
        
        Upper envelope (descending yield):
        - cum_H_km2 = [1, 2, 3, 4, 5]
        - cum_P_mt = [500, 900, 1200, 1400, 1500]
        """
        envelope = build_discrete_envelope(
            simple_test_case['Y_mt_per_ha'],
            simple_test_case['H_ha'],
            simple_test_case['P_mt']
        )
        
        # Check lower envelope
        expected_cum_H = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_cum_P_low = np.array([100.0, 300.0, 600.0, 1000.0, 1500.0])
        
        np.testing.assert_allclose(
            envelope['lower']['cum_H_km2'], 
            expected_cum_H,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['lower']['cum_P_mt'], 
            expected_cum_P_low,
            rtol=1e-10
        )
        
        # Check upper envelope
        expected_cum_P_up = np.array([500.0, 900.0, 1200.0, 1400.0, 1500.0])
        
        np.testing.assert_allclose(
            envelope['upper']['cum_H_km2'], 
            expected_cum_H,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['upper']['cum_P_mt'], 
            expected_cum_P_up,
            rtol=1e-10
        )
    
    def test_properties_with_normal_distribution(self, normal_data_1000):
        """Test that all mathematical properties hold with normal distribution."""
        envelope = build_discrete_envelope(
            normal_data_1000['Y_mt_per_ha'],
            normal_data_1000['H_ha'],
            normal_data_1000['P_mt']
        )
        
        # Test monotonicity
        assert np.all(np.diff(envelope['lower']['cum_H_km2']) > 0)
        assert np.all(np.diff(envelope['lower']['cum_P_mt']) >= 0)
        assert np.all(np.diff(envelope['upper']['cum_H_km2']) > 0)
        assert np.all(np.diff(envelope['upper']['cum_P_mt']) >= 0)
        
        # Test conservation
        expected_total_P = normal_data_1000['known_totals']['sum_P_mt']
        expected_total_H = normal_data_1000['known_totals']['sum_H_km2']
        
        np.testing.assert_allclose(
            envelope['lower']['cum_P_mt'][-1], 
            expected_total_P, 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['upper']['cum_P_mt'][-1], 
            expected_total_P, 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['lower']['cum_H_km2'][-1], 
            expected_total_H, 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['upper']['cum_H_km2'][-1], 
            expected_total_H, 
            rtol=1e-10
        )
        
        # Test yield ordering
        assert np.all(np.diff(envelope['lower']['Y_sorted']) >= 0)
        assert np.all(np.diff(envelope['upper']['Y_sorted']) <= 0)
    
    def test_properties_with_bimodal_distribution(self, bimodal_data_1000):
        """Test that all mathematical properties hold with bimodal distribution."""
        envelope = build_discrete_envelope(
            bimodal_data_1000['Y_mt_per_ha'],
            bimodal_data_1000['H_ha'],
            bimodal_data_1000['P_mt']
        )
        
        # Test monotonicity
        assert np.all(np.diff(envelope['lower']['cum_H_km2']) > 0)
        assert np.all(np.diff(envelope['lower']['cum_P_mt']) >= 0)
        assert np.all(np.diff(envelope['upper']['cum_H_km2']) > 0)
        assert np.all(np.diff(envelope['upper']['cum_P_mt']) >= 0)
        
        # Test conservation
        expected_total_P = bimodal_data_1000['known_totals']['sum_P_mt']
        expected_total_H = bimodal_data_1000['known_totals']['sum_H_km2']
        
        np.testing.assert_allclose(
            envelope['lower']['cum_P_mt'][-1], 
            expected_total_P, 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['upper']['cum_P_mt'][-1], 
            expected_total_P, 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['lower']['cum_H_km2'][-1], 
            expected_total_H, 
            rtol=1e-10
        )
        np.testing.assert_allclose(
            envelope['upper']['cum_H_km2'][-1], 
            expected_total_H, 
            rtol=1e-10
        )
        
        # Test yield ordering
        assert np.all(np.diff(envelope['lower']['Y_sorted']) >= 0)
        assert np.all(np.diff(envelope['upper']['Y_sorted']) <= 0)

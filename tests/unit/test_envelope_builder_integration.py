"""
Integration tests for the envelope_builder module.

Tests the actual build_envelope() function with synthetic data.
"""

import numpy as np
import pytest
from agrichter.analysis.envelope_builder import build_envelope


class TestEnvelopeBuilderIntegration:
    """Test the build_envelope function with synthetic data."""
    
    def test_simple_case(self):
        """Test with simple hand-calculable case."""
        # 5 cells: Y=[1,2,3,4,5], H=[100,100,100,100,100], P=[100,200,300,400,500]
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        P_mt = Y_mt_per_ha * H_ha
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        # Check structure
        assert 'lower' in result
        assert 'upper' in result
        assert 'summary' in result
        
        # Check lower envelope
        expected_cum_H = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_cum_P_low = np.array([100.0, 300.0, 600.0, 1000.0, 1500.0])
        
        np.testing.assert_allclose(
            result['lower']['cum_H_km2'],
            expected_cum_H,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            result['lower']['cum_P_mt'],
            expected_cum_P_low,
            rtol=1e-10
        )
        
        # Check upper envelope
        expected_cum_P_up = np.array([500.0, 900.0, 1200.0, 1400.0, 1500.0])
        
        np.testing.assert_allclose(
            result['upper']['cum_H_km2'],
            expected_cum_H,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            result['upper']['cum_P_mt'],
            expected_cum_P_up,
            rtol=1e-10
        )
        
        # Check summary
        assert result['summary']['n_valid'] == 5
        assert result['summary']['n_dropped'] == 0
        assert np.isclose(result['summary']['totals']['P_mt'], 1500.0)
        assert np.isclose(result['summary']['totals']['H_km2'], 5.0)
    
    def test_without_yield_provided(self):
        """Test that yield is computed correctly when not provided."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        # Should compute Y = P / H = [1, 2, 3, 4, 5]
        assert result['summary']['yield_validation']['computed'] is True
        assert result['summary']['n_valid'] == 5
        
        # Check that envelopes are correct
        expected_cum_P_low = np.array([100.0, 300.0, 600.0, 1000.0, 1500.0])
        np.testing.assert_allclose(
            result['lower']['cum_P_mt'],
            expected_cum_P_low,
            rtol=1e-10
        )
    
    def test_with_nans_and_zeros(self):
        """Test that NaNs and zeros are properly dropped."""
        P_mt = np.array([100.0, np.nan, 300.0, 0.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        # Should drop 2 cells (NaN and zero)
        assert result['summary']['n_valid'] == 3
        assert result['summary']['n_dropped'] == 2
        assert result['summary']['dropped_counts']['nan_P'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_P'] == 1
        
        # Remaining cells: indices 0, 2, 4 with P=[100, 300, 500]
        # Lower: sorted by Y=[1,3,5] -> P=[100, 300, 500]
        # Upper: sorted by Y=[5,3,1] -> P=[500, 300, 100]
        expected_cum_P_low = np.array([100.0, 400.0, 900.0])
        expected_cum_P_up = np.array([500.0, 800.0, 900.0])
        
        np.testing.assert_allclose(
            result['lower']['cum_P_mt'],
            expected_cum_P_low,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            result['upper']['cum_P_mt'],
            expected_cum_P_up,
            rtol=1e-10
        )
    
    def test_qa_results_all_pass(self):
        """Test that QA results show all properties passing."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        qa = result['summary']['qa_results']
        
        # All properties should pass (use == not 'is' for numpy bool compatibility)
        assert qa['lower_H_monotonic'] == True
        assert qa['lower_P_monotonic'] == True
        assert qa['lower_Y_ascending'] == True
        assert qa['upper_H_monotonic'] == True
        assert qa['upper_P_monotonic'] == True
        assert qa['upper_Y_descending'] == True
        assert qa['lower_H_conserved'] == True
        assert qa['lower_P_conserved'] == True
        assert qa['upper_H_conserved'] == True
        assert qa['upper_P_conserved'] == True
        assert qa['dominance_holds'] == True
        assert qa['dominance_violations'] == 0
    
    def test_yield_consistency_validation(self):
        """Test yield consistency checking when yield is provided."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Provide correct yield
        Y_mt_per_ha = P_mt / H_ha
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        # Should pass consistency check
        assert result['summary']['yield_validation']['computed'] is False
        assert result['summary']['yield_validation']['mismatch_count'] == 0
    
    def test_yield_consistency_mismatch(self):
        """Test yield consistency checking with mismatched data."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Provide incorrect yield (off by 10%)
        Y_mt_per_ha = (P_mt / H_ha) * 1.1
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should detect mismatches
        assert result['summary']['yield_validation']['computed'] is False
        assert result['summary']['yield_validation']['mismatch_count'] == 5
        assert result['summary']['yield_validation']['mismatch_pct'] == 100.0
    
    def test_large_dataset(self):
        """Test with larger dataset (1000 cells)."""
        np.random.seed(42)
        n_cells = 1000
        
        Y_mt_per_ha = np.random.uniform(1.0, 10.0, n_cells)
        H_ha = np.random.uniform(100.0, 1000.0, n_cells)
        P_mt = Y_mt_per_ha * H_ha
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        # Check basic properties
        assert result['summary']['n_valid'] == n_cells
        assert result['summary']['n_dropped'] == 0
        
        # Check conservation
        expected_total_P = np.sum(P_mt)
        expected_total_H_km2 = np.sum(H_ha) * 0.01
        
        assert np.isclose(
            result['summary']['totals']['P_mt'],
            expected_total_P,
            rtol=1e-10
        )
        assert np.isclose(
            result['summary']['totals']['H_km2'],
            expected_total_H_km2,
            rtol=1e-10
        )
        
        # Check QA
        assert result['summary']['qa_results']['dominance_holds'] is True
    
    def test_deterministic_output(self):
        """Test that same inputs produce same outputs."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        result1 = build_envelope(P_mt, H_ha)
        result2 = build_envelope(P_mt, H_ha)
        
        # Should be identical
        np.testing.assert_array_equal(
            result1['lower']['cum_P_mt'],
            result2['lower']['cum_P_mt']
        )
        np.testing.assert_array_equal(
            result1['upper']['cum_P_mt'],
            result2['upper']['cum_P_mt']
        )

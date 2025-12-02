"""
Unit tests for envelope builder input validation and unit conversion.

This module tests the validation logic in envelope_builder.py to ensure:
- Task 3.1: Input validation (NaN, zero, negative handling)
- Task 3.2: Unit conversion (single conversion, conservation)
- Task 3.3: Yield consistency checking
"""

import numpy as np
import pytest
from agririchter.analysis.envelope_builder import build_envelope


class TestInputValidation:
    """Test input validation logic (Task 3.1)."""
    
    def test_valid_inputs_no_yield(self):
        """Test that valid inputs without yield work correctly."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 3
        assert result['summary']['n_dropped'] == 0
        assert result['summary']['yield_validation']['computed'] is True
    
    def test_valid_inputs_with_yield(self):
        """Test that valid inputs with yield work correctly."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        assert result['summary']['n_valid'] == 3
        assert result['summary']['n_dropped'] == 0
        assert result['summary']['yield_validation']['computed'] is False
    
    def test_drop_nan_production(self):
        """Test that NaN production values are dropped."""
        P_mt = np.array([100.0, np.nan, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['nan_P'] == 1
    
    def test_drop_nan_harvest(self):
        """Test that NaN harvest area values are dropped."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, np.nan, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['nan_H'] == 1
    
    def test_drop_nan_yield(self):
        """Test that NaN yield values are dropped when yield is provided."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, np.nan, 3.0])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['nan_Y'] == 1
    
    def test_drop_zero_production(self):
        """Test that zero production values are dropped."""
        P_mt = np.array([100.0, 0.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_P'] == 1
    
    def test_drop_negative_production(self):
        """Test that negative production values are dropped."""
        P_mt = np.array([100.0, -50.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_P'] == 1
    
    def test_drop_zero_harvest(self):
        """Test that zero harvest area values are dropped."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 0.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_H'] == 1
    
    def test_drop_negative_harvest(self):
        """Test that negative harvest area values are dropped."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, -50.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_H'] == 1
    
    def test_drop_zero_yield(self):
        """Test that zero yield values are dropped when yield is provided."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, 0.0, 3.0])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_Y'] == 1
    
    def test_drop_negative_yield(self):
        """Test that negative yield values are dropped when yield is provided."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, -2.0, 3.0])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_Y'] == 1
    
    def test_drop_multiple_reasons(self):
        """Test that cells are dropped for multiple reasons and counted correctly."""
        P_mt = np.array([100.0, np.nan, 0.0, -50.0, 500.0])
        H_ha = np.array([100.0, 100.0, 0.0, 100.0, np.nan])
        
        result = build_envelope(P_mt, H_ha)
        
        # Only first cell is valid
        assert result['summary']['n_valid'] == 1
        assert result['summary']['n_dropped'] == 4
        
        # Check individual drop reasons
        assert result['summary']['dropped_counts']['nan_P'] == 1
        assert result['summary']['dropped_counts']['nan_H'] == 1
        assert result['summary']['dropped_counts']['zero_or_negative_P'] >= 1
        assert result['summary']['dropped_counts']['zero_or_negative_H'] >= 1
    
    def test_n_valid_reported(self):
        """Test that N_valid is correctly reported (Task 3.1 requirement)."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        # Verify N_valid is reported and equals expected count
        assert 'n_valid' in result['summary']
        assert result['summary']['n_valid'] == 5
        assert result['summary']['n_valid'] == len(P_mt)
    
    def test_array_length_mismatch_P_H(self):
        """Test that mismatched array lengths raise ValueError."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0])  # Wrong length
        
        with pytest.raises(ValueError, match="must have same length"):
            build_envelope(P_mt, H_ha)
    
    def test_array_length_mismatch_Y(self):
        """Test that mismatched yield array length raises ValueError."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, 2.0])  # Wrong length
        
        with pytest.raises(ValueError, match="must have same length"):
            build_envelope(P_mt, H_ha, Y_mt_per_ha)


class TestUnitConversion:
    """Test unit conversion logic (Task 3.2)."""
    
    def test_harvest_area_conversion(self):
        """Test that H_ha is converted to H_km2 exactly once."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 200.0, 300.0])
        
        result = build_envelope(P_mt, H_ha)
        
        # Check that totals are correct
        expected_H_ha = 600.0
        expected_H_km2 = 6.0
        
        assert result['summary']['totals']['H_ha'] == expected_H_ha
        assert result['summary']['totals']['H_km2'] == expected_H_km2
        assert result['summary']['totals']['H_km2'] == expected_H_ha * 0.01
    
    def test_production_unchanged(self):
        """Test that P_mt is not modified during conversion."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        # Check that total production is conserved
        expected_P_mt = 600.0
        assert result['summary']['totals']['P_mt'] == expected_P_mt
    
    def test_yield_computed_when_not_provided(self):
        """Test that yield is computed as P/H when not provided."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        # Yield should be computed
        assert result['summary']['yield_validation']['computed'] is True
        
        # Check yield stats
        expected_yields = P_mt / H_ha  # [1.0, 2.0, 3.0]
        assert result['summary']['yield_stats']['min'] == 1.0
        assert result['summary']['yield_stats']['median'] == 2.0
        assert result['summary']['yield_stats']['max'] == 3.0
    
    def test_conservation_of_totals(self):
        """Test that sum(H_km2) = sum(H_ha) * 0.01 exactly (Task 3.2 requirement)."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([123.45, 234.56, 345.67, 456.78, 567.89])
        
        result = build_envelope(P_mt, H_ha)
        
        # Verify exact conversion
        sum_H_ha = np.sum(H_ha)
        sum_H_km2 = sum_H_ha * 0.01
        
        assert result['summary']['totals']['H_ha'] == sum_H_ha
        np.testing.assert_allclose(
            result['summary']['totals']['H_km2'],
            sum_H_km2,
            rtol=1e-10
        )
    
    def test_production_conservation(self):
        """Test that sum(P_mt) is unchanged after conversion (Task 3.2 requirement)."""
        P_mt = np.array([123.45, 234.56, 345.67, 456.78, 567.89])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        sum_P_before = np.sum(P_mt)
        
        result = build_envelope(P_mt, H_ha)
        
        # Verify production is conserved
        np.testing.assert_allclose(
            result['summary']['totals']['P_mt'],
            sum_P_before,
            rtol=1e-10
        )
    
    def test_cumulative_sequences_use_km2(self):
        """Test that cumulative sequences use km² for harvest area."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        # Check that cumulative harvest is in km²
        cum_H_lower = result['lower']['cum_H_km2']
        cum_H_upper = result['upper']['cum_H_km2']
        
        # Final cumulative should equal total in km²
        expected_total_km2 = 3.0  # 300 ha = 3 km²
        
        np.testing.assert_allclose(cum_H_lower[-1], expected_total_km2, rtol=1e-10)
        np.testing.assert_allclose(cum_H_upper[-1], expected_total_km2, rtol=1e-10)


class TestYieldConsistency:
    """Test yield consistency validation (Task 3.3)."""
    
    def test_consistent_yield_no_warning(self):
        """Test that consistent yield passes validation without warnings."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.0, 2.0, 3.0])  # Exactly P/H
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
        
        # Should have no mismatches
        assert result['summary']['yield_validation']['mismatch_count'] == 0
        assert result['summary']['yield_validation']['mismatch_pct'] == 0.0
    
    def test_small_inconsistency_within_tolerance(self):
        """Test that small inconsistencies within tolerance are accepted."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        # Yield slightly off but within 5% tolerance
        Y_mt_per_ha = np.array([1.02, 2.03, 3.01])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should have no or very few mismatches
        assert result['summary']['yield_validation']['mismatch_pct'] < 1.0
    
    def test_large_inconsistency_flagged(self):
        """Test that large inconsistencies are flagged as MISMATCH."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Yield significantly off for 2 cells (40%)
        Y_mt_per_ha = np.array([1.0, 5.0, 3.0, 10.0, 5.0])
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should flag mismatches
        assert result['summary']['yield_validation']['mismatch_count'] > 0
        assert result['summary']['yield_validation']['mismatch_pct'] > 1.0
    
    def test_mismatch_per_cell_not_aggregate(self):
        """Test that consistency is checked per-cell, not on aggregates (Task 3.3 guard)."""
        # Create data where aggregate P = aggregate Y*H but individual cells don't match
        P_mt = np.array([100.0, 300.0])  # Total = 400
        H_ha = np.array([100.0, 100.0])  # Total = 200
        Y_mt_per_ha = np.array([3.0, 1.0])  # Total Y*H = 3*100 + 1*100 = 400
        
        # Aggregate: sum(P) = 400, sum(Y*H) = 400 (matches!)
        # But per-cell: P[0]=100 vs Y[0]*H[0]=300 (mismatch!)
        #               P[1]=300 vs Y[1]*H[1]=100 (mismatch!)
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Should detect per-cell mismatches
        assert result['summary']['yield_validation']['mismatch_count'] == 2
        assert result['summary']['yield_validation']['mismatch_pct'] == 100.0
    
    def test_tolerance_parameter(self):
        """Test that tolerance parameter controls mismatch detection."""
        P_mt = np.array([100.0, 200.0, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        Y_mt_per_ha = np.array([1.08, 2.08, 3.08])  # 8 mt absolute difference
        
        # With 5% tolerance:
        # P=100: |100-108| = 8 > 0.05*100 = 5 → MISMATCH
        # P=200: |200-208| = 8 > 0.05*200 = 10 → NO (8 < 10)
        # P=300: |300-308| = 8 > 0.05*300 = 15 → NO (8 < 15)
        result_strict = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        assert result_strict['summary']['yield_validation']['mismatch_count'] == 1
        
        # With 10% tolerance, all should pass
        result_loose = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.10)
        assert result_loose['summary']['yield_validation']['mismatch_count'] == 0
    
    def test_mismatch_report_includes_count(self):
        """Test that mismatch report includes cell counts (Task 3.3 requirement)."""
        P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Check tolerance: |P - Y*H| > tol * max(1, P)
        # P=100, Y=1.0: |100-100| = 0 → OK
        # P=200, Y=5.0: |200-500| = 300 > 0.05*200 = 10 → MISMATCH
        # P=300, Y=3.0: |300-300| = 0 → OK
        # P=400, Y=4.0: |400-400| = 0 → OK
        # P=500, Y=5.0: |500-500| = 0 → OK
        Y_mt_per_ha = np.array([1.0, 5.0, 3.0, 4.0, 5.0])  # 1 mismatch
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha, tol=0.05)
        
        # Check that mismatch count is reported
        assert 'mismatch_count' in result['summary']['yield_validation']
        assert 'mismatch_pct' in result['summary']['yield_validation']
        assert result['summary']['yield_validation']['mismatch_count'] == 1
        assert result['summary']['yield_validation']['mismatch_pct'] == 20.0


class TestEdgeCases:
    """Test edge cases for validation logic."""
    
    def test_all_cells_dropped(self):
        """Test behavior when all cells are dropped."""
        P_mt = np.array([0.0, -100.0, np.nan])
        H_ha = np.array([0.0, 100.0, np.nan])
        
        # This should fail because no valid cells remain
        # The function will try to build envelope with empty arrays
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            build_envelope(P_mt, H_ha)
    
    def test_single_valid_cell(self):
        """Test that single valid cell works."""
        P_mt = np.array([100.0])
        H_ha = np.array([100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 1
        assert result['summary']['totals']['P_mt'] == 100.0
        assert result['summary']['totals']['H_km2'] == 1.0
    
    def test_very_small_values(self):
        """Test that very small but positive values are accepted."""
        P_mt = np.array([0.001, 0.002, 0.003])
        H_ha = np.array([0.1, 0.1, 0.1])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 3
        assert result['summary']['n_dropped'] == 0
    
    def test_very_large_values(self):
        """Test that very large values are handled correctly."""
        P_mt = np.array([1e6, 2e6, 3e6])
        H_ha = np.array([1e5, 1e5, 1e5])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 3
        assert result['summary']['totals']['P_mt'] == 6e6
    
    def test_inf_values_dropped(self):
        """Test that infinite values are dropped."""
        P_mt = np.array([100.0, np.inf, 300.0])
        H_ha = np.array([100.0, 100.0, 100.0])
        
        result = build_envelope(P_mt, H_ha)
        
        assert result['summary']['n_valid'] == 2
        assert result['summary']['n_dropped'] == 1
        assert result['summary']['dropped_counts']['nan_P'] == 1

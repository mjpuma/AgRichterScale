"""
Unit tests for envelope builder summary and QA reporting (Task 6).

This module tests:
- Task 6.1: Calculate summary statistics from discrete data
- Task 6.2: Create QA validation report with mathematical assertions
"""

import numpy as np
import pytest
from agririchter.analysis.envelope_builder import build_envelope


class TestSummaryStatistics:
    """Test Task 6.1: Calculate summary statistics from discrete data."""
    
    def test_basic_summary_statistics(self):
        """Test that summary includes all required statistics."""
        # Create simple test data
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check cell counts
        assert summary['n_total'] == 5
        assert summary['n_valid'] == 5
        assert summary['n_dropped'] == 0
        
        # Check totals
        assert summary['totals']['P_mt'] == 1500.0
        assert summary['totals']['H_ha'] == 500.0
        assert summary['totals']['H_km2'] == 5.0
        
        # Check yield statistics
        assert 'yield_stats' in summary
        assert 'min' in summary['yield_stats']
        assert 'median' in summary['yield_stats']
        assert 'max' in summary['yield_stats']
        assert 'mean' in summary['yield_stats']
        assert 'std' in summary['yield_stats']
        
        # Verify yield values
        assert summary['yield_stats']['min'] == 1.0  # 100/100
        assert summary['yield_stats']['max'] == 5.0  # 500/100
        assert summary['yield_stats']['median'] == 3.0  # 300/100
    
    def test_unit_conversion_check(self):
        """Test that unit conversion check is performed correctly."""
        P_mt = np.array([100, 200, 300])
        H_ha = np.array([50, 100, 150])
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check conservation checks exist
        assert 'conservation_checks' in summary
        assert 'unit_conversion' in summary['conservation_checks']
        
        # Verify unit conversion
        unit_check = summary['conservation_checks']['unit_conversion']
        assert unit_check['passed'] == True
        assert unit_check['total_H_ha'] == 300.0
        assert unit_check['total_H_km2'] == 3.0
        assert unit_check['expected_H_km2'] == 3.0
        assert unit_check['diff'] < 1e-10
    
    def test_cumsum_conservation_checks(self):
        """Test that cumulative sum conservation is verified."""
        P_mt = np.array([100, 200, 300, 400])
        H_ha = np.array([100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check lower envelope conservation
        assert 'lower_H_conservation' in summary['conservation_checks']
        assert 'lower_P_conservation' in summary['conservation_checks']
        
        lower_H = summary['conservation_checks']['lower_H_conservation']
        assert lower_H['passed'] == True
        assert lower_H['cum_H_final'] == summary['totals']['H_km2']
        
        lower_P = summary['conservation_checks']['lower_P_conservation']
        assert lower_P['passed'] == True
        assert lower_P['cum_P_final'] == summary['totals']['P_mt']
        
        # Check upper envelope conservation
        assert 'upper_H_conservation' in summary['conservation_checks']
        assert 'upper_P_conservation' in summary['conservation_checks']
        
        upper_H = summary['conservation_checks']['upper_H_conservation']
        assert upper_H['passed'] == True
        
        upper_P = summary['conservation_checks']['upper_P_conservation']
        assert upper_P['passed'] == True
    
    def test_dropped_cells_reporting(self):
        """Test that dropped cells are correctly reported."""
        # Create data with NaNs and zeros
        P_mt = np.array([100, np.nan, 0, 200, -10])
        H_ha = np.array([100, 100, 100, 0, 100])
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check counts
        assert summary['n_total'] == 5
        assert summary['n_valid'] == 1  # Only first cell is valid
        assert summary['n_dropped'] == 4
        
        # Check dropped counts
        dropped = summary['dropped_counts']
        assert dropped['nan_P'] == 1
        assert dropped['zero_or_negative_P'] >= 1
        assert dropped['zero_or_negative_H'] >= 1


class TestQAValidationReport:
    """Test Task 6.2: Create QA validation report with mathematical assertions."""
    
    def test_discrete_qa_results_included(self):
        """Test that discrete QA results are included in summary."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check discrete QA results exist
        assert 'discrete_qa_results' in summary
        qa = summary['discrete_qa_results']
        
        # Verify all expected properties are checked
        assert 'lower_H_monotonic' in qa
        assert 'lower_P_monotonic' in qa
        assert 'lower_Y_ascending' in qa
        assert 'upper_H_monotonic' in qa
        assert 'upper_P_monotonic' in qa
        assert 'upper_Y_descending' in qa
        assert 'dominance_holds' in qa
        
        # All should pass for valid data
        assert qa['lower_H_monotonic'] == True
        assert qa['lower_P_monotonic'] == True
        assert qa['lower_Y_ascending'] == True
        assert qa['upper_H_monotonic'] == True
        assert qa['upper_P_monotonic'] == True
        assert qa['upper_Y_descending'] == True
        assert qa['dominance_holds'] == True
    
    def test_interpolation_qa_when_interpolated(self):
        """Test that interpolation QA is performed when interpolate=True."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha, interpolate=True)
        summary = result['summary']
        
        # Check interpolation QA exists
        assert 'interpolation_qa' in summary
        assert summary['interpolation_qa'] is not None
        
        qa = summary['interpolation_qa']
        
        # Verify all interpolation assertions are checked
        assert 'Hq_strictly_increasing' in qa
        assert 'P_low_monotonic' in qa
        assert 'P_up_monotonic' in qa
        assert 'dominance_after_clipping' in qa
        
        # All should pass
        assert qa['Hq_strictly_increasing']['passed'] == True
        assert qa['P_low_monotonic']['passed'] == True
        assert qa['P_up_monotonic']['passed'] == True
        assert qa['dominance_after_clipping']['passed'] == True
    
    def test_no_interpolation_qa_when_not_interpolated(self):
        """Test that interpolation QA is None when interpolate=False."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha, interpolate=False)
        summary = result['summary']
        
        # Interpolation QA should be None
        assert summary['interpolation_qa'] is None
    
    def test_low_sample_warning(self):
        """Test that LOW_SAMPLE warning is generated when n_valid < 100."""
        # Create data with only 50 valid cells
        P_mt = np.random.uniform(100, 500, 50)
        H_ha = np.random.uniform(50, 150, 50)
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check for LOW_SAMPLE warning
        assert 'warnings' in summary
        warnings = summary['warnings']
        
        low_sample_warnings = [w for w in warnings if w['type'] == 'LOW_SAMPLE']
        assert len(low_sample_warnings) == 1
        assert low_sample_warnings[0]['severity'] == 'warning'
        assert '50' in low_sample_warnings[0]['message']
    
    def test_no_low_sample_warning_when_sufficient(self):
        """Test that no LOW_SAMPLE warning when n_valid >= 100."""
        # Create data with 100 valid cells
        P_mt = np.random.uniform(100, 500, 100)
        H_ha = np.random.uniform(50, 150, 100)
        
        result = build_envelope(P_mt, H_ha)
        summary = result['summary']
        
        # Check no LOW_SAMPLE warning
        warnings = summary['warnings']
        low_sample_warnings = [w for w in warnings if w['type'] == 'LOW_SAMPLE']
        assert len(low_sample_warnings) == 0
    
    def test_clipping_warning_recorded(self):
        """Test that clipping warnings are recorded when clipping occurs."""
        # Create data that might need clipping (though unlikely with proper sorting)
        # We'll use a large dataset to increase chances
        np.random.seed(42)
        P_mt = np.random.uniform(100, 1000, 1000)
        H_ha = np.random.uniform(50, 200, 1000)
        
        result = build_envelope(P_mt, H_ha, interpolate=True)
        summary = result['summary']
        
        # Check if clipping was applied
        if result['interpolated'].get('clipping_applied', False):
            warnings = summary['warnings']
            clipping_warnings = [w for w in warnings if w['type'] == 'CLIPPING_APPLIED']
            assert len(clipping_warnings) == 1
            assert 'n_clipped' in clipping_warnings[0]
            assert 'max_deficit' in clipping_warnings[0]
    
    def test_all_checks_passed_flag(self):
        """Test that all_checks_passed flag is set correctly."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha, interpolate=True)
        summary = result['summary']
        
        # Check overall status flag
        assert 'all_checks_passed' in summary
        assert summary['all_checks_passed'] is True
    
    def test_qa_violation_details(self):
        """Test that QA violations include detailed information."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha, interpolate=True)
        summary = result['summary']
        
        # Check that violation details include counts
        qa = summary['interpolation_qa']
        
        assert 'n_violations' in qa['Hq_strictly_increasing']
        assert 'n_violations' in qa['P_low_monotonic']
        assert 'n_violations' in qa['P_up_monotonic']
        assert 'n_violations' in qa['dominance_after_clipping']
        
        # For passing tests, violations should be 0
        assert qa['Hq_strictly_increasing']['n_violations'] == 0
        assert qa['P_low_monotonic']['n_violations'] == 0
        assert qa['P_up_monotonic']['n_violations'] == 0
        assert qa['dominance_after_clipping']['n_violations'] == 0


class TestSummaryIntegration:
    """Integration tests for complete summary generation."""
    
    def test_complete_summary_structure(self):
        """Test that summary has complete expected structure."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha, interpolate=True)
        summary = result['summary']
        
        # Check all required top-level keys
        required_keys = [
            'n_total', 'n_valid', 'n_dropped', 'dropped_counts',
            'yield_validation', 'totals', 'yield_stats',
            'conservation_checks', 'discrete_qa_results',
            'interpolation_qa', 'warnings', 'all_checks_passed'
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing required key: {key}"
    
    def test_summary_with_provided_yield(self):
        """Test summary generation when yield is provided."""
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        Y_mt_per_ha = P_mt / H_ha  # Consistent yield
        
        result = build_envelope(P_mt, H_ha, Y_mt_per_ha=Y_mt_per_ha)
        summary = result['summary']
        
        # Check yield validation
        assert summary['yield_validation']['computed'] is False
        assert summary['yield_validation']['mismatch_count'] == 0
    
    def test_summary_with_large_dataset(self):
        """Test summary generation with realistic large dataset."""
        np.random.seed(42)
        n_cells = 1000
        
        P_mt = np.random.uniform(100, 1000, n_cells)
        H_ha = np.random.uniform(50, 200, n_cells)
        
        result = build_envelope(P_mt, H_ha, interpolate=True, n_points=300)
        summary = result['summary']
        
        # Verify all checks pass
        assert summary['n_valid'] == n_cells
        assert summary['all_checks_passed'] == True
        
        # Verify conservation
        assert summary['conservation_checks']['unit_conversion']['passed'] == True
        assert summary['conservation_checks']['lower_H_conservation']['passed'] == True
        assert summary['conservation_checks']['lower_P_conservation']['passed'] == True
        assert summary['conservation_checks']['upper_H_conservation']['passed'] == True
        assert summary['conservation_checks']['upper_P_conservation']['passed'] == True
    
    def test_summary_json_serializable(self):
        """Test that summary can be serialized to JSON."""
        import json
        
        P_mt = np.array([100, 200, 300, 400, 500])
        H_ha = np.array([100, 100, 100, 100, 100])
        
        result = build_envelope(P_mt, H_ha, interpolate=True)
        summary = result['summary']
        
        # Remove non-serializable numpy arrays if any
        summary_copy = summary.copy()
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(summary_copy, indent=2)
            assert len(json_str) > 0
        except TypeError as e:
            pytest.fail(f"Summary not JSON serializable: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

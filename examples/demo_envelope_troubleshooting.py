#!/usr/bin/env python3
"""
Troubleshooting examples for the envelope builder.

This script demonstrates common issues and how to diagnose them.
"""

import numpy as np
import warnings
from agririchter.analysis.envelope_builder import build_envelope


def example_low_sample_warning():
    """Example: LOW_SAMPLE warning when fewer than 100 valid cells."""
    print("=" * 80)
    print("EXAMPLE 1: LOW_SAMPLE Warning")
    print("=" * 80)
    
    # Create small dataset
    n_cells = 45
    P_mt = np.random.uniform(100, 1000, n_cells)
    H_ha = np.random.uniform(100, 500, n_cells)
    
    print(f"\nInput: {n_cells} cells (< 100 threshold)")
    print("Expected: LOW_SAMPLE warning")
    
    result = build_envelope(P_mt, H_ha)
    summary = result['summary']
    
    print(f"\nResult:")
    print(f"  Valid cells: {summary['n_valid']}")
    print(f"  Warnings: {summary.get('warnings', [])}")
    
    if summary['n_valid'] < 100:
        print("\n✓ LOW_SAMPLE warning issued as expected")
        print("  Impact: Envelope may not be smooth, statistical properties less reliable")
        print("  Action: Accept lower quality results or aggregate more data")
    
    print("\n" + "=" * 80)


def example_yield_mismatch():
    """Example: MISMATCH warning when provided yield is inconsistent with P and H."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Yield Mismatch")
    print("=" * 80)
    
    # Create data with inconsistent yield
    n_cells = 200
    P_mt = np.random.uniform(100, 1000, n_cells)
    H_ha = np.random.uniform(100, 500, n_cells)
    
    # Compute correct yield
    Y_correct = P_mt / H_ha
    
    # Introduce errors in 20% of cells
    Y_provided = Y_correct.copy()
    error_mask = np.random.rand(n_cells) < 0.20
    Y_provided[error_mask] *= np.random.uniform(1.1, 1.5, np.sum(error_mask))
    
    print(f"\nInput: {n_cells} cells")
    print(f"  Yield errors introduced in {np.sum(error_mask)} cells ({100*np.sum(error_mask)/n_cells:.1f}%)")
    print("Expected: MISMATCH warning and yield recomputation")
    
    result = build_envelope(P_mt, H_ha, Y_mt_per_ha=Y_provided, tol=0.05)
    summary = result['summary']
    
    yield_val = summary['yield_validation']
    print(f"\nResult:")
    print(f"  Mismatch count: {yield_val['mismatch_count']}")
    print(f"  Mismatch pct: {yield_val['mismatch_pct']:.2f}%")
    print(f"  Recomputed: {yield_val.get('recomputed', False)}")
    
    if yield_val.get('recomputed', False):
        print("\n✓ Yield recomputed from P/H (P is authoritative)")
        print("  Impact: Original yield data discarded, results still valid")
        print("  Action: Use consistent data sources or don't provide yield")
    
    print("\n" + "=" * 80)


def example_dropped_cells():
    """Example: Cells dropped due to invalid data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Dropped Cells")
    print("=" * 80)
    
    # Create data with various invalid values
    n_cells = 500
    P_mt = np.random.uniform(100, 1000, n_cells)
    H_ha = np.random.uniform(100, 500, n_cells)
    
    # Introduce invalid values
    P_mt[10:15] = np.nan  # NaN production
    H_ha[20:23] = 0  # Zero harvest area
    P_mt[30:32] = -100  # Negative production
    H_ha[40:44] = np.nan  # NaN harvest area
    
    n_invalid = 5 + 3 + 2 + 4  # Total invalid cells
    
    print(f"\nInput: {n_cells} cells")
    print(f"  Invalid cells introduced: {n_invalid}")
    print(f"    - NaN production: 5")
    print(f"    - Zero harvest area: 3")
    print(f"    - Negative production: 2")
    print(f"    - NaN harvest area: 4")
    print("Expected: Cells dropped with detailed logging")
    
    result = build_envelope(P_mt, H_ha)
    summary = result['summary']
    
    print(f"\nResult:")
    print(f"  Total cells: {summary['n_total']}")
    print(f"  Valid cells: {summary['n_valid']}")
    print(f"  Dropped cells: {summary['n_dropped']}")
    print(f"\n  Dropped by reason:")
    for reason, count in summary['dropped_counts'].items():
        if count > 0:
            print(f"    - {reason}: {count}")
    
    print("\n✓ Invalid cells identified and dropped")
    print("  Impact: Only valid cells used in envelope calculation")
    print("  Action: Check data quality, investigate why cells are invalid")
    
    print("\n" + "=" * 80)


def example_interpolation_clipping():
    """Example: Clipping applied after interpolation (rare but possible)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Interpolation Clipping")
    print("=" * 80)
    
    # Create data that might cause small interpolation artifacts
    # This is rare with np.interp but can happen with numerical precision
    n_cells = 100
    np.random.seed(789)
    
    # Create yields with some very close values (can cause precision issues)
    Y_mt_per_ha = np.random.uniform(2.0, 8.0, n_cells)
    H_ha = np.random.uniform(100, 1000, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    print(f"\nInput: {n_cells} cells")
    print("Expected: Possible small clipping due to numerical precision")
    
    result = build_envelope(P_mt, H_ha, interpolate=True, n_points=200)
    
    if 'interpolated' in result:
        interp = result['interpolated']
        
        print(f"\nResult:")
        print(f"  Clipping applied: {interp.get('clipping_applied', False)}")
        
        if interp.get('clipping_applied', False):
            n_clipped = interp['n_clipped']
            max_deficit = interp['max_deficit']
            pct_clipped = 100 * n_clipped / len(interp['Hq_km2'])
            
            print(f"  Points clipped: {n_clipped}/{len(interp['Hq_km2'])} ({pct_clipped:.2f}%)")
            print(f"  Max deficit corrected: {max_deficit:.2e} mt")
            
            if pct_clipped < 5.0:
                print("\n✓ Small amount of clipping (<5%) is acceptable")
                print("  Impact: Numerical precision artifact, corrected automatically")
                print("  Action: None required, results are valid")
            else:
                print("\n✗ Excessive clipping (>5%) indicates a problem")
                print("  Impact: Discrete envelopes may not satisfy dominance")
                print("  Action: Check discrete envelope properties, investigate data quality")
        else:
            print("\n✓ No clipping needed - interpolation preserved dominance")
    
    print("\n" + "=" * 80)


def example_debugging_dominance_failure():
    """Example: How to debug if dominance fails (should never happen)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Debugging Dominance Failure (Hypothetical)")
    print("=" * 80)
    
    print("\nThis example shows how to debug if dominance fails.")
    print("Note: This should NEVER happen with correct implementation!")
    
    # Create normal data (dominance will hold)
    n_cells = 200
    np.random.seed(999)
    Y_mt_per_ha = np.random.uniform(1, 10, n_cells)
    H_ha = np.random.uniform(100, 500, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    print(f"\nInput: {n_cells} cells with normal data")
    
    result = build_envelope(P_mt, H_ha, interpolate=False)
    lower = result['lower']
    upper = result['upper']
    summary = result['summary']
    
    print("\nDebugging steps:")
    print("-" * 80)
    
    # Step 1: Check yield sorting
    print("\n1. Verify yield sorting:")
    Y_low_ascending = np.all(np.diff(lower['Y_sorted']) >= 0)
    Y_up_descending = np.all(np.diff(upper['Y_sorted']) <= 0)
    print(f"   Lower yields ascending:  {Y_low_ascending} {'✓' if Y_low_ascending else '✗'}")
    print(f"   Upper yields descending: {Y_up_descending} {'✓' if Y_up_descending else '✗'}")
    
    if not Y_low_ascending or not Y_up_descending:
        print("   → Problem: Yields not sorted correctly!")
        print("   → Check: Sorting implementation, NaN/Inf values")
    
    # Step 2: Check for invalid values
    print("\n2. Check for invalid values:")
    print(f"   NaN in yields: {np.sum(~np.isfinite(Y_mt_per_ha))}")
    print(f"   Negative yields: {np.sum(Y_mt_per_ha < 0)}")
    print(f"   Zero yields: {np.sum(Y_mt_per_ha == 0)}")
    print(f"   NaN in production: {np.sum(~np.isfinite(P_mt))}")
    print(f"   Negative production: {np.sum(P_mt < 0)}")
    
    # Step 3: Check dominance at specific points
    print("\n3. Check dominance at test points:")
    test_indices = np.linspace(0, len(lower['cum_H_km2'])-1, 5, dtype=int)
    all_ok = True
    for i in test_indices:
        H_target = lower['cum_H_km2'][i]
        P_low = lower['cum_P_mt'][i]
        
        k_up = np.searchsorted(upper['cum_H_km2'], H_target)
        if k_up >= len(upper['cum_P_mt']):
            k_up = len(upper['cum_P_mt']) - 1
        P_up = upper['cum_P_mt'][k_up]
        
        dominance_ok = P_up >= P_low
        all_ok = all_ok and dominance_ok
        status = '✓' if dominance_ok else '✗'
        
        print(f"   H={H_target:8.2f} km²: P_low={P_low:10.2f} mt, P_up={P_up:10.2f} mt {status}")
        
        if not dominance_ok:
            deficit = P_low - P_up
            print(f"      → VIOLATION: deficit = {deficit:.2f} mt")
    
    # Step 4: Check conservation
    print("\n4. Check conservation:")
    expected_P = np.sum(P_mt)
    expected_H = np.sum(H_ha) * 0.01
    print(f"   Lower final P: {lower['cum_P_mt'][-1]:.2f} mt (expected {expected_P:.2f})")
    print(f"   Upper final P: {upper['cum_P_mt'][-1]:.2f} mt (expected {expected_P:.2f})")
    print(f"   Lower final H: {lower['cum_H_km2'][-1]:.2f} km² (expected {expected_H:.2f})")
    print(f"   Upper final H: {upper['cum_H_km2'][-1]:.2f} km² (expected {expected_H:.2f})")
    
    print("\n" + "-" * 80)
    if all_ok:
        print("✓ All checks passed - dominance holds as expected")
    else:
        print("✗ Dominance violations found - investigate causes above")
    
    print("\n" + "=" * 80)


def example_qa_report():
    """Example: Understanding the QA report."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Understanding the QA Report")
    print("=" * 80)
    
    # Create normal data
    n_cells = 500
    np.random.seed(111)
    Y_mt_per_ha = np.random.uniform(2, 8, n_cells)
    H_ha = np.random.uniform(100, 1000, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    print(f"\nInput: {n_cells} cells")
    
    result = build_envelope(P_mt, H_ha, interpolate=True)
    summary = result['summary']
    qa = summary['qa_results']
    
    print("\n" + "-" * 80)
    print("QA Report Interpretation:")
    print("-" * 80)
    
    print("\nMathematical Properties:")
    print(f"  {'✓' if qa['lower_H_monotonic'] else '✗'} lower_H_monotonic: Cumulative harvest strictly increasing")
    print(f"  {'✓' if qa['lower_P_monotonic'] else '✗'} lower_P_monotonic: Cumulative production non-decreasing")
    print(f"  {'✓' if qa['lower_Y_ascending'] else '✗'} lower_Y_ascending: Yields sorted ascending (worst first)")
    print(f"  {'✓' if qa['upper_H_monotonic'] else '✗'} upper_H_monotonic: Cumulative harvest strictly increasing")
    print(f"  {'✓' if qa['upper_P_monotonic'] else '✗'} upper_P_monotonic: Cumulative production non-decreasing")
    print(f"  {'✓' if qa['upper_Y_descending'] else '✗'} upper_Y_descending: Yields sorted descending (best first)")
    
    print("\nConservation:")
    print(f"  {'✓' if qa['lower_H_conserved'] else '✗'} lower_H_conserved: Total harvest area preserved")
    print(f"  {'✓' if qa['lower_P_conserved'] else '✗'} lower_P_conserved: Total production preserved")
    print(f"  {'✓' if qa['upper_H_conserved'] else '✗'} upper_H_conserved: Total harvest area preserved")
    print(f"  {'✓' if qa['upper_P_conserved'] else '✗'} upper_P_conserved: Total production preserved")
    
    print("\nDominance:")
    print(f"  {'✓' if qa['dominance_holds'] else '✗'} dominance_holds: Upper >= lower at all test points")
    if 'dominance_test_points' in qa:
        print(f"     Tested at {qa['dominance_test_points']} harvest levels")
    if 'dominance_violations' in qa and qa['dominance_violations'] > 0:
        print(f"     ✗ {qa['dominance_violations']} violations found!")
    
    print("\nSummary Statistics:")
    print(f"  Valid cells: {summary['n_valid']}")
    print(f"  Total production: {summary['total_P_mt']:.2f} mt")
    print(f"  Total harvest: {summary['total_H_km2']:.2f} km²")
    print(f"  Yield range: [{summary['yield_min']:.2f}, {summary['yield_max']:.2f}] mt/ha")
    print(f"  Yield median: {summary['yield_median']:.2f} mt/ha")
    
    all_passed = all([
        qa['lower_H_monotonic'],
        qa['lower_P_monotonic'],
        qa['lower_Y_ascending'],
        qa['upper_H_monotonic'],
        qa['upper_P_monotonic'],
        qa['upper_Y_descending'],
        qa['lower_H_conserved'],
        qa['lower_P_conserved'],
        qa['upper_H_conserved'],
        qa['upper_P_conserved'],
        qa['dominance_holds']
    ])
    
    print("\n" + "-" * 80)
    if all_passed:
        print("✓ All QA checks passed - envelope is mathematically valid")
    else:
        print("✗ Some QA checks failed - investigate issues")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("ENVELOPE BUILDER TROUBLESHOOTING EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate common issues and how to diagnose them.")
    print("See docs/ENVELOPE_BUILDER_MATHEMATICAL_GUIDE.md for detailed troubleshooting.")
    print("\n")
    
    # Run all examples
    example_low_sample_warning()
    example_yield_mismatch()
    example_dropped_cells()
    example_interpolation_clipping()
    example_debugging_dominance_failure()
    example_qa_report()
    
    print("\n" + "=" * 80)
    print("TROUBLESHOOTING EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor more information:")
    print("  - Mathematical guide: docs/ENVELOPE_BUILDER_MATHEMATICAL_GUIDE.md")
    print("  - API reference: docs/API_REFERENCE.md")
    print("  - User guide: docs/USER_GUIDE.md")
    print("=" * 80)

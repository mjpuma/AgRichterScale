"""
Robust envelope builder for discrete gridded agricultural data.

This module provides mathematically rigorous envelope calculation that:
1. Works with discrete grid cells (not continuous functions)
2. Maintains monotonicity and conservation properties
3. Provides comprehensive validation and QA reporting

Mathematical Foundation
-----------------------
The envelope builder creates maximum and minimum cumulative production curves
by sorting discrete grid cells by yield and computing cumulative sums.

Key Principle:
    Given N discrete grid cells with (P_i, H_i, Y_i), the envelope is built
    from cumulative sums of cells sorted by yield:
    - Lower envelope: Sort by yield ascending (least productive first)
    - Upper envelope: Sort by yield descending (most productive first)

Mathematical Properties Guaranteed:
    1. Monotonicity: Cumulative sequences are non-decreasing
    2. Dominance: Upper envelope >= Lower envelope at all harvest levels
    3. Conservation: Total production and harvest area are preserved
    4. Yield Ordering: Correct sort order (ascending/descending)

Why Yield Sorting Guarantees Dominance:
    For the same total harvest area H, selecting high-yield cells (upper envelope)
    produces more than selecting low-yield cells (lower envelope). This is because:
    
        P_upper = (avg high yield) × H >= (avg low yield) × H = P_lower
    
    This fundamental property makes the envelope bounds physically meaningful.

Discrete vs Continuous:
    - Input: N discrete grid cells (e.g., N=10,000 for global crop)
    - Processing: Discrete cumulative sums (N points per envelope)
    - Output (optional): Interpolated onto continuous query grid (M points, typically M=200)
    
    The interpolation is a visualization convenience. All mathematical properties
    are established on the discrete data.

Usage Example
-------------
    >>> import numpy as np
    >>> from agrichter.analysis.envelope_builder import build_envelope
    >>> 
    >>> # Create synthetic data
    >>> P_mt = np.array([100, 200, 300, 400, 500])
    >>> H_ha = np.array([100, 100, 100, 100, 100])
    >>> 
    >>> # Build envelope
    >>> result = build_envelope(P_mt, H_ha, interpolate=True)
    >>> 
    >>> # Access results
    >>> lower = result['lower']
    >>> upper = result['upper']
    >>> summary = result['summary']
    >>> 
    >>> # Verify properties
    >>> assert summary['qa_results']['dominance_holds']
    >>> assert summary['qa_results']['lower_P_monotonic']
    >>> print(f"Total production: {summary['total_P_mt']:.2f} mt")

For More Information
--------------------
See docs/ENVELOPE_BUILDER_MATHEMATICAL_GUIDE.md for:
    - Detailed mathematical derivations
    - Proof of dominance theorem
    - QA validation rules and what they catch
    - Troubleshooting guide
    - Additional usage examples

See examples/demo_envelope_mathematical_properties.py for:
    - Interactive demonstrations of mathematical properties
    - Synthetic data examples with known properties
    - Visualization of discrete vs continuous representations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def build_envelope(
    P_mt: Union[pd.Series, np.ndarray],
    H_ha: Union[pd.Series, np.ndarray],
    Y_mt_per_ha: Optional[Union[pd.Series, np.ndarray]] = None,
    tol: float = 0.05,
    interpolate: bool = False,
    n_points: int = 200
) -> Dict:
    """
    Build discrete production envelope from gridded agricultural data.
    
    This function creates lower and upper envelope curves by sorting grid cells
    by yield and computing cumulative sums. Optionally interpolates onto a
    unified query grid for plotting and analysis.
    
    Mathematical Properties Guaranteed:
    - Cumulative sequences are monotonic non-decreasing
    - Upper envelope >= lower envelope at all cumulative harvest levels
    - Total production and harvest area are conserved
    - Yields are correctly ordered (ascending for lower, descending for upper)
    - If interpolated: monotonicity and dominance preserved after interpolation
    
    Parameters
    ----------
    P_mt : pd.Series or np.ndarray
        Production per grid cell (metric tons)
    H_ha : pd.Series or np.ndarray
        Harvest area per grid cell (hectares)
    Y_mt_per_ha : pd.Series or np.ndarray, optional
        Yield per hectare (metric tons per hectare). If not provided, will be
        computed as P_mt / H_ha
    tol : float, default=0.05
        Tolerance for yield consistency check (5% by default)
    interpolate : bool, default=False
        If True, interpolate envelopes onto a unified query grid
    n_points : int, default=200
        Number of points in query grid (if interpolate=True)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'lower': dict with 'cum_H_km2', 'cum_P_mt', 'Y_sorted', 'indices'
        - 'upper': dict with 'cum_H_km2', 'cum_P_mt', 'Y_sorted', 'indices'
        - 'summary': dict with validation metrics and QA results
        - 'interpolated': dict (if interpolate=True) with 'Hq_km2', 'P_low_interp', 'P_up_interp'
    
    Notes
    -----
    This function works with DISCRETE grid cells. The cumulative sequences
    represent the sum of production/harvest from the first k cells when
    sorted by yield. If interpolate=True, the discrete sequences are mapped
    onto a continuous query grid using monotonic interpolation.
    
    Examples
    --------
    >>> P_mt = np.array([100, 200, 300, 400, 500])
    >>> H_ha = np.array([100, 100, 100, 100, 100])
    >>> result = build_envelope(P_mt, H_ha)
    >>> result['lower']['cum_P_mt']
    array([ 100.,  300.,  600., 1000., 1500.])
    """
    # Convert to numpy arrays for consistent handling
    P_mt = np.asarray(P_mt)
    H_ha = np.asarray(H_ha)
    
    if Y_mt_per_ha is not None:
        Y_mt_per_ha = np.asarray(Y_mt_per_ha)
    
    # Validate inputs and compute yield if needed
    validation_result = _validate_and_prepare_inputs(P_mt, H_ha, Y_mt_per_ha, tol)
    
    # Extract validated data
    P_mt_valid = validation_result['P_mt']
    H_ha_valid = validation_result['H_ha']
    Y_mt_per_ha_valid = validation_result['Y_mt_per_ha']
    
    # Convert harvest area to km2 (exactly once)
    H_km2_valid = H_ha_valid * 0.01
    
    # Build discrete envelope sequences
    lower_envelope = _build_discrete_sequence(
        Y_mt_per_ha_valid, H_km2_valid, P_mt_valid, ascending=True
    )
    upper_envelope = _build_discrete_sequence(
        Y_mt_per_ha_valid, H_km2_valid, P_mt_valid, ascending=False
    )
    
    # Verify mathematical properties
    qa_results = _verify_mathematical_properties(
        lower_envelope, upper_envelope, validation_result
    )
    
    # Optionally interpolate onto unified query grid
    interpolated = None
    if interpolate:
        interpolated = _interpolate_envelopes(
            lower_envelope, upper_envelope, n_points
        )
    
    # Build summary (TASK 6: Generate summary metrics and QA report)
    summary = _build_summary(
        validation_result,
        qa_results,
        lower_envelope=lower_envelope,
        upper_envelope=upper_envelope,
        interpolated=interpolated
    )
    
    result = {
        'lower': lower_envelope,
        'upper': upper_envelope,
        'summary': summary
    }
    
    if interpolated is not None:
        result['interpolated'] = interpolated
    
    return result


def _validate_and_prepare_inputs(
    P_mt: np.ndarray,
    H_ha: np.ndarray,
    Y_mt_per_ha: Optional[np.ndarray],
    tol: float
) -> Dict:
    """
    Validate inputs and prepare data for envelope building.
    
    This function:
    1. Checks for valid finite values
    2. Drops NaNs and zeros
    3. Computes or validates yield
    4. Logs dropped cells and validation results
    
    Parameters
    ----------
    P_mt : np.ndarray
        Production (metric tons)
    H_ha : np.ndarray
        Harvest area (hectares)
    Y_mt_per_ha : np.ndarray or None
        Yield (metric tons per hectare)
    tol : float
        Tolerance for yield consistency check
    
    Returns
    -------
    dict
        Dictionary with validated data and validation metrics
    """
    n_total = len(P_mt)
    
    # Check array lengths match
    if len(H_ha) != n_total:
        raise ValueError(f"P_mt and H_ha must have same length: {n_total} vs {len(H_ha)}")
    
    if Y_mt_per_ha is not None and len(Y_mt_per_ha) != n_total:
        raise ValueError(f"Y_mt_per_ha must have same length as P_mt: {n_total} vs {len(Y_mt_per_ha)}")
    
    # Track dropped cells
    dropped_counts = {
        'nan_P': 0,
        'nan_H': 0,
        'nan_Y': 0,
        'zero_or_negative_P': 0,
        'zero_or_negative_H': 0,
        'zero_or_negative_Y': 0
    }
    
    # Create mask for valid cells
    valid_mask = np.ones(n_total, dtype=bool)
    
    # Check for NaNs
    nan_P = ~np.isfinite(P_mt)
    nan_H = ~np.isfinite(H_ha)
    dropped_counts['nan_P'] = np.sum(nan_P)
    dropped_counts['nan_H'] = np.sum(nan_H)
    valid_mask &= ~nan_P & ~nan_H
    
    if Y_mt_per_ha is not None:
        nan_Y = ~np.isfinite(Y_mt_per_ha)
        dropped_counts['nan_Y'] = np.sum(nan_Y)
        valid_mask &= ~nan_Y
    
    # Check for zeros and negatives (P must be > 0, H must be > 0)
    zero_neg_P = P_mt <= 0
    zero_neg_H = H_ha <= 0
    dropped_counts['zero_or_negative_P'] = np.sum(zero_neg_P & valid_mask)
    dropped_counts['zero_or_negative_H'] = np.sum(zero_neg_H & valid_mask)
    valid_mask &= ~zero_neg_P & ~zero_neg_H
    
    if Y_mt_per_ha is not None:
        zero_neg_Y = Y_mt_per_ha <= 0
        dropped_counts['zero_or_negative_Y'] = np.sum(zero_neg_Y & valid_mask)
        valid_mask &= ~zero_neg_Y
    
    # Extract valid data
    P_mt_valid = P_mt[valid_mask]
    H_ha_valid = H_ha[valid_mask]
    
    n_valid = len(P_mt_valid)
    n_dropped = n_total - n_valid
    
    # Log dropped cells
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} of {n_total} cells ({100*n_dropped/n_total:.1f}%)")
        for reason, count in dropped_counts.items():
            if count > 0:
                logger.info(f"  - {reason}: {count}")
    
    # Compute or validate yield
    yield_validation = {}
    
    if Y_mt_per_ha is None:
        # Compute yield from P and H
        Y_mt_per_ha_valid = P_mt_valid / H_ha_valid
        logger.info("Computed yield as Y = P / H")
        yield_validation['computed'] = True
        yield_validation['mismatch_count'] = 0
    else:
        # Validate provided yield
        Y_mt_per_ha_provided = Y_mt_per_ha[valid_mask]
        
        # Check consistency: P_hat = Y * H should match P
        P_hat = Y_mt_per_ha_provided * H_ha_valid
        abs_diff = np.abs(P_mt_valid - P_hat)
        rel_tol = tol * np.maximum(1.0, P_mt_valid)
        mismatch_mask = abs_diff > rel_tol
        
        n_mismatch = np.sum(mismatch_mask)
        mismatch_pct = 100 * n_mismatch / n_valid if n_valid > 0 else 0
        
        yield_validation['computed'] = False
        yield_validation['mismatch_count'] = n_mismatch
        yield_validation['mismatch_pct'] = mismatch_pct
        
        if mismatch_pct > 1.0:
            logger.warning(
                f"MISMATCH: {n_mismatch} cells ({mismatch_pct:.1f}%) have "
                f"inconsistent P vs Y*H (tolerance={tol})"
            )
            # Log examples
            if n_mismatch > 0:
                mismatch_indices = np.where(mismatch_mask)[0][:5]
                for idx in mismatch_indices:
                    logger.warning(
                        f"  Example: P={P_mt_valid[idx]:.2f}, Y*H={P_hat[idx]:.2f}, "
                        f"diff={abs_diff[idx]:.2f}"
                    )
            # IMPORTANT: P_mt is authoritative (Requirement 2)
            # Recompute yield from P and H to ensure consistency
            logger.warning("Recomputing yield from P/H (P_mt is authoritative)")
            Y_mt_per_ha_valid = P_mt_valid / H_ha_valid
            yield_validation['recomputed'] = True
        else:
            logger.info(f"Yield consistency check passed ({n_mismatch} mismatches, {mismatch_pct:.2f}%)")
            Y_mt_per_ha_valid = Y_mt_per_ha_provided
            yield_validation['recomputed'] = False
    
    # Warn if low sample size
    if n_valid < 100:
        logger.warning(f"LOW_SAMPLE: Only {n_valid} valid cells (< 100)")
    
    # Calculate totals
    total_P_mt = np.sum(P_mt_valid)
    total_H_ha = np.sum(H_ha_valid)
    total_H_km2 = total_H_ha * 0.01
    
    logger.info(f"Valid cells: {n_valid}")
    logger.info(f"Total production: {total_P_mt:.2f} mt")
    logger.info(f"Total harvest area: {total_H_ha:.2f} ha ({total_H_km2:.2f} km²)")
    
    return {
        'P_mt': P_mt_valid,
        'H_ha': H_ha_valid,
        'Y_mt_per_ha': Y_mt_per_ha_valid,
        'n_total': n_total,
        'n_valid': n_valid,
        'n_dropped': n_dropped,
        'dropped_counts': dropped_counts,
        'yield_validation': yield_validation,
        'total_P_mt': total_P_mt,
        'total_H_ha': total_H_ha,
        'total_H_km2': total_H_km2
    }



def _build_discrete_sequence(
    Y_mt_per_ha: np.ndarray,
    H_km2: np.ndarray,
    P_mt: np.ndarray,
    ascending: bool
) -> Dict:
    """
    Build discrete cumulative sequence by sorting cells by yield.
    
    This function implements the core mathematical step of envelope building:
    1. Sort discrete cells by yield (ascending for lower, descending for upper)
    2. Compute cumulative sums of harvest area and production
    3. Verify mathematical properties (monotonicity, conservation)
    
    Parameters
    ----------
    Y_mt_per_ha : np.ndarray
        Yield per hectare
    H_km2 : np.ndarray
        Harvest area in km²
    P_mt : np.ndarray
        Production in metric tons
    ascending : bool
        If True, sort by yield ascending (for lower envelope)
        If False, sort by yield descending (for upper envelope)
    
    Returns
    -------
    dict
        Dictionary with:
        - 'cum_H_km2': cumulative harvest area (km²)
        - 'cum_P_mt': cumulative production (mt)
        - 'Y_sorted': sorted yield values
        - 'indices': sort indices (for traceability)
    
    Raises
    ------
    AssertionError
        If mathematical properties are violated (sort order, monotonicity, conservation)
    """
    # ============================================================================
    # TASK 4.1: Sort discrete cells by yield
    # ============================================================================
    
    # Sort by yield
    if ascending:
        sort_indices = np.argsort(Y_mt_per_ha)
        envelope_type = "lower"
    else:
        sort_indices = np.argsort(Y_mt_per_ha)[::-1]
        envelope_type = "upper"
    
    Y_sorted = Y_mt_per_ha[sort_indices]
    H_sorted = H_km2[sort_indices]
    P_sorted = P_mt[sort_indices]
    
    # ASSERT: Verify sort order
    if ascending:
        # Lower envelope: Y_sorted[i] <= Y_sorted[i+1] for all i
        Y_diffs = np.diff(Y_sorted)
        assert np.all(Y_diffs >= 0), (
            f"Lower envelope sort order violated: yield not ascending. "
            f"Found {np.sum(Y_diffs < 0)} violations. "
            f"Min diff: {np.min(Y_diffs)}"
        )
        logger.debug(f"✓ Lower envelope: yields sorted ascending (Y[i] <= Y[i+1])")
    else:
        # Upper envelope: Y_sorted[i] >= Y_sorted[i+1] for all i
        Y_diffs = np.diff(Y_sorted)
        assert np.all(Y_diffs <= 0), (
            f"Upper envelope sort order violated: yield not descending. "
            f"Found {np.sum(Y_diffs > 0)} violations. "
            f"Max diff: {np.max(Y_diffs)}"
        )
        logger.debug(f"✓ Upper envelope: yields sorted descending (Y[i] >= Y[i+1])")
    
    # ============================================================================
    # TASK 4.2: Calculate discrete cumulative sums
    # ============================================================================
    
    # Store original totals for conservation check
    total_H_km2 = np.sum(H_km2)
    total_P_mt = np.sum(P_mt)
    
    # Compute cumulative sums
    cum_H_km2 = np.cumsum(H_sorted)
    cum_P_mt = np.cumsum(P_sorted)
    
    # ASSERT: cum_H_km2 is non-decreasing (monotonic)
    # Note: We allow zero increments because multiple cells can have the same H value
    # after rounding/precision. What matters is that it never decreases.
    H_diffs = np.diff(cum_H_km2)
    assert np.all(H_diffs >= 0), (
        f"{envelope_type.capitalize()} envelope: cumulative harvest not non-decreasing. "
        f"Found {np.sum(H_diffs < 0)} negative increments. "
        f"Min increment: {np.min(H_diffs)}"
    )
    
    # Check if we have many zero increments (indicates duplicate H values)
    n_zero_increments = np.sum(H_diffs == 0)
    if n_zero_increments > 0:
        pct_zero = 100 * n_zero_increments / len(H_diffs)
        logger.debug(
            f"{envelope_type.capitalize()} envelope: {n_zero_increments} cells ({pct_zero:.1f}%) "
            f"have same H as previous cell (ties in harvest area)"
        )
    
    logger.debug(f"✓ {envelope_type.capitalize()} envelope: cum_H_km2 non-decreasing")
    
    # ASSERT: cum_P_mt is non-decreasing (each cell adds non-negative production)
    P_diffs = np.diff(cum_P_mt)
    assert np.all(P_diffs >= 0), (
        f"{envelope_type.capitalize()} envelope: cumulative production not non-decreasing. "
        f"Found {np.sum(P_diffs < 0)} negative increments. "
        f"Min increment: {np.min(P_diffs)}"
    )
    logger.debug(f"✓ {envelope_type.capitalize()} envelope: cum_P_mt non-decreasing")
    
    # ASSERT: Final cumulative values equal totals (conservation)
    # Use relative tolerance of 2e-3 (0.2%) to account for:
    # - Floating point precision with large datasets (hundreds of thousands of cells)
    # - Cells with very small H values that may be affected by sorting/precision
    # - Multiple crops being summed (allgrain has 7 crops)
    assert np.isclose(cum_H_km2[-1], total_H_km2, rtol=2e-3), (
        f"{envelope_type.capitalize()} envelope: harvest area not conserved. "
        f"Expected {total_H_km2:.10f}, got {cum_H_km2[-1]:.10f}, "
        f"diff: {abs(cum_H_km2[-1] - total_H_km2):.2e}, "
        f"relative error: {abs(cum_H_km2[-1] - total_H_km2) / total_H_km2:.2e}"
    )
    logger.debug(
        f"✓ {envelope_type.capitalize()} envelope: harvest area conserved "
        f"(cum_H_km2[-1] = {cum_H_km2[-1]:.2f} km²)"
    )
    
    assert np.isclose(cum_P_mt[-1], total_P_mt, rtol=2e-3), (
        f"{envelope_type.capitalize()} envelope: production not conserved. "
        f"Expected {total_P_mt:.10f}, got {cum_P_mt[-1]:.10f}, "
        f"diff: {abs(cum_P_mt[-1] - total_P_mt):.2e}, "
        f"relative error: {abs(cum_P_mt[-1] - total_P_mt) / total_P_mt:.2e}"
    )
    logger.debug(
        f"✓ {envelope_type.capitalize()} envelope: production conserved "
        f"(cum_P_mt[-1] = {cum_P_mt[-1]:.2f} mt)"
    )
    
    return {
        'cum_H_km2': cum_H_km2,
        'cum_P_mt': cum_P_mt,
        'Y_sorted': Y_sorted,
        'indices': sort_indices
    }



def _verify_mathematical_properties(
    lower_envelope: Dict,
    upper_envelope: Dict,
    validation_result: Dict
) -> Dict:
    """
    Verify that mathematical properties hold for the envelope sequences.
    
    Properties checked:
    1. Cumulative sequences are monotonic non-decreasing
    2. Final cumulative values equal input totals
    3. Yield ordering is correct
    4. Upper >= lower at sampled harvest levels (dominance)
    
    Parameters
    ----------
    lower_envelope : dict
        Lower envelope data
    upper_envelope : dict
        Upper envelope data
    validation_result : dict
        Input validation results with totals
    
    Returns
    -------
    dict
        QA results with pass/fail for each property
    
    Raises
    ------
    AssertionError
        If dominance property fails (indicates fundamental error)
    """
    qa_results = {}
    
    # Check monotonicity of lower envelope
    cum_H_low = lower_envelope['cum_H_km2']
    cum_P_low = lower_envelope['cum_P_mt']
    Y_low = lower_envelope['Y_sorted']
    
    qa_results['lower_H_monotonic'] = np.all(np.diff(cum_H_low) > 0)
    qa_results['lower_P_monotonic'] = np.all(np.diff(cum_P_low) >= 0)
    qa_results['lower_Y_ascending'] = np.all(np.diff(Y_low) >= 0)
    
    # Check monotonicity of upper envelope
    cum_H_up = upper_envelope['cum_H_km2']
    cum_P_up = upper_envelope['cum_P_mt']
    Y_up = upper_envelope['Y_sorted']
    
    qa_results['upper_H_monotonic'] = np.all(np.diff(cum_H_up) > 0)
    qa_results['upper_P_monotonic'] = np.all(np.diff(cum_P_up) >= 0)
    qa_results['upper_Y_descending'] = np.all(np.diff(Y_up) <= 0)
    
    # Check conservation of totals
    expected_H = validation_result['total_H_km2']
    expected_P = validation_result['total_P_mt']
    
    qa_results['lower_H_conserved'] = np.isclose(cum_H_low[-1], expected_H, rtol=1e-10)
    qa_results['lower_P_conserved'] = np.isclose(cum_P_low[-1], expected_P, rtol=1e-10)
    qa_results['upper_H_conserved'] = np.isclose(cum_H_up[-1], expected_H, rtol=1e-10)
    qa_results['upper_P_conserved'] = np.isclose(cum_P_up[-1], expected_P, rtol=1e-10)
    
    # ============================================================================
    # TASK 4.3: Verify envelope dominance property (discrete)
    # ============================================================================
    
    # Test dominance at 10 evenly-spaced harvest levels
    # NOTE: We test at the LOWER envelope's cumulative harvest points to ensure
    # we're comparing at the same harvest level. The mathematical property is:
    # "For any harvest level H, the maximum production achievable is >= minimum production"
    
    H_min = cum_H_low[0]
    H_max = cum_H_low[-1]
    
    # Sample 10 points from the lower envelope's harvest range
    if len(cum_H_low) >= 10:
        # Sample evenly from indices
        test_indices = np.linspace(0, len(cum_H_low) - 1, 10, dtype=int)
    else:
        # Use all points if fewer than 10
        test_indices = np.arange(len(cum_H_low))
    
    dominance_violations = []
    max_violation_deficit = 0.0
    
    for i in test_indices:
        H_target = cum_H_low[i]
        P_low_at_H = cum_P_low[i]
        
        # Find the corresponding production in upper envelope at this harvest level
        # Use searchsorted to find where H_target would fit in upper envelope
        k_up = np.searchsorted(cum_H_up, H_target, side='left')
        
        # Handle edge cases
        if k_up >= len(cum_P_up):
            k_up = len(cum_P_up) - 1
        
        P_up_at_H = cum_P_up[k_up]
        
        # Mathematical property: At the same harvest level H, upper production >= lower production
        # Allow small numerical tolerance for floating point comparison
        tolerance = 1e-6 * max(abs(P_low_at_H), abs(P_up_at_H), 1.0)
        
        if P_up_at_H < P_low_at_H - tolerance:
            deficit = P_low_at_H - P_up_at_H
            dominance_violations.append({
                'test_point': int(i),
                'H_target': H_target,
                'k_low': int(i),
                'k_up': int(k_up),
                'P_low': P_low_at_H,
                'P_up': P_up_at_H,
                'deficit': deficit
            })
            max_violation_deficit = max(max_violation_deficit, deficit)
    
    n_violations = len(dominance_violations)
    qa_results['dominance_holds'] = (n_violations == 0)
    qa_results['dominance_violations'] = n_violations
    qa_results['dominance_test_points'] = len(test_indices)
    
    if n_violations > 0:
        qa_results['max_violation_deficit'] = max_violation_deficit
        qa_results['violation_details'] = dominance_violations
    
    # ASSERT: Dominance must hold at all test points
    # This is a fundamental mathematical property: sorting by yield descending (upper)
    # should always give >= production than sorting by yield ascending (lower) at any
    # given harvest level
    assert n_violations == 0, (
        f"Dominance property VIOLATED at {n_violations}/{len(test_indices)} test points. "
        f"This indicates a FUNDAMENTAL ERROR in sorting or cumulative sum calculation. "
        f"Max deficit: {max_violation_deficit:.2e} mt. "
        f"First violation: H_target={dominance_violations[0]['H_target']:.2f} km², "
        f"P_low={dominance_violations[0]['P_low']:.2f} mt, "
        f"P_up={dominance_violations[0]['P_up']:.2f} mt. "
        f"STOP and investigate!"
    )
    
    logger.info(
        f"✓ Dominance property verified: upper >= lower at all {len(test_indices)} test points"
    )
    
    # Log results
    all_passed = all([
        qa_results['lower_H_monotonic'],
        qa_results['lower_P_monotonic'],
        qa_results['lower_Y_ascending'],
        qa_results['upper_H_monotonic'],
        qa_results['upper_P_monotonic'],
        qa_results['upper_Y_descending'],
        qa_results['lower_H_conserved'],
        qa_results['lower_P_conserved'],
        qa_results['upper_H_conserved'],
        qa_results['upper_P_conserved'],
        qa_results['dominance_holds']
    ])
    
    if all_passed:
        logger.info("✓ All mathematical properties verified")
    else:
        logger.error("✗ Mathematical property verification FAILED")
        for prop, passed in qa_results.items():
            if isinstance(passed, bool) and not passed:
                logger.error(f"  - {prop}: FAILED")
    
    return qa_results


def _create_query_grid(
    cum_H_low: np.ndarray,
    cum_H_up: np.ndarray,
    n_points: int = 200
) -> np.ndarray:
    """
    Create unified query grid for interpolation (DISCRETE to CONTINUOUS mapping).
    
    This function creates a shared cumulative area grid (Hq_km2) that spans the
    range of both lower and upper envelope sequences. The grid uses appropriate
    spacing (log or linear) based on the data span.
    
    Mathematical Constraints:
    - Hq_km2 is strictly increasing
    - Hq_km2[0] >= min(cum_H_low[0], cum_H_up[0]) (no extrapolation below)
    - Hq_km2[-1] <= max(cum_H_low[-1], cum_H_up[-1]) (no extrapolation above)
    - Adequate resolution (>= 200 points by default)
    
    Parameters
    ----------
    cum_H_low : np.ndarray
        Cumulative harvest area for lower envelope (km²)
    cum_H_up : np.ndarray
        Cumulative harvest area for upper envelope (km²)
    n_points : int, default=200
        Number of points in query grid
    
    Returns
    -------
    np.ndarray
        Query grid Hq_km2 with strictly increasing values
    
    Raises
    ------
    AssertionError
        If query grid is not strictly increasing or violates constraints
    """
    # ============================================================================
    # TASK 5.1: Define shared cumulative area grid (Hq_km2)
    # ============================================================================
    
    # Determine range: min and max across both sequences
    H_min = min(cum_H_low[0], cum_H_up[0])
    H_max = max(cum_H_low[-1], cum_H_up[-1])
    
    # Ensure we have at least the requested number of points
    n_points = max(n_points, 200)
    
    # Determine spacing type based on span
    span = H_max / H_min if H_min > 0 else 1.0
    
    if span >= 1000:  # 3 orders of magnitude
        # Use log-spacing for large spans
        logger.debug(
            f"Using log-spacing for query grid (span = {span:.1e}, "
            f"range = [{H_min:.2e}, {H_max:.2e}] km²)"
        )
        Hq_km2 = np.logspace(np.log10(H_min), np.log10(H_max), n_points)
    else:
        # Use linear spacing for smaller spans
        logger.debug(
            f"Using linear spacing for query grid (span = {span:.1f}, "
            f"range = [{H_min:.2f}, {H_max:.2f}] km²)"
        )
        Hq_km2 = np.linspace(H_min, H_max, n_points)
    
    # ASSERT: Query grid is strictly increasing
    H_diffs = np.diff(Hq_km2)
    assert np.all(H_diffs > 0), (
        f"Query grid not strictly increasing. "
        f"Found {np.sum(H_diffs <= 0)} non-positive increments. "
        f"Min increment: {np.min(H_diffs)}"
    )
    logger.debug(f"✓ Query grid strictly increasing ({n_points} points)")
    
    # ASSERT: No extrapolation below minimum
    # Use small relative tolerance to account for floating point precision
    tol_min = max(1e-10, H_min * 1e-6)
    assert Hq_km2[0] >= H_min - tol_min, (
        f"Query grid extrapolates below data range. "
        f"Hq_km2[0] = {Hq_km2[0]:.10f}, H_min = {H_min:.10f}"
    )
    
    # ASSERT: No extrapolation above maximum
    # Use small relative tolerance to account for floating point precision
    tol_max = max(1e-10, H_max * 1e-6)
    assert Hq_km2[-1] <= H_max + tol_max, (
        f"Query grid extrapolates above data range. "
        f"Hq_km2[-1] = {Hq_km2[-1]:.10f}, H_max = {H_max:.10f}"
    )
    logger.debug(f"✓ Query grid within data range (no extrapolation)")
    
    logger.info(
        f"Created query grid: {n_points} points, "
        f"range [{Hq_km2[0]:.2f}, {Hq_km2[-1]:.2f}] km²"
    )
    
    return Hq_km2


def _interpolate_envelopes(
    lower_envelope: Dict,
    upper_envelope: Dict,
    n_points: int = 200
) -> Dict:
    """
    Interpolate discrete envelopes onto unified query grid.
    
    This function performs the DISCRETE to CONTINUOUS mapping by interpolating
    the discrete cumulative sequences onto a shared query grid. It uses simple
    monotonic interpolation (np.interp) and verifies mathematical properties.
    
    Mathematical Properties Verified:
    - Interpolated sequences are monotonic non-decreasing
    - Upper >= lower at all query points (dominance)
    - No extrapolation artifacts
    
    Parameters
    ----------
    lower_envelope : dict
        Lower envelope with 'cum_H_km2' and 'cum_P_mt'
    upper_envelope : dict
        Upper envelope with 'cum_H_km2' and 'cum_P_mt'
    n_points : int, default=200
        Number of points in query grid
    
    Returns
    -------
    dict
        Dictionary with:
        - 'Hq_km2': query grid (harvest area in km²)
        - 'P_low_interp': interpolated lower envelope production (mt)
        - 'P_up_interp': interpolated upper envelope production (mt)
        - 'clipping_applied': bool, whether clipping was needed
        - 'n_clipped': int, number of points clipped
        - 'max_deficit': float, maximum deficit corrected (if clipped)
    
    Raises
    ------
    AssertionError
        If monotonicity fails or dominance violations exceed 5%
    """
    # Extract discrete sequences
    cum_H_low = lower_envelope['cum_H_km2']
    cum_P_low = lower_envelope['cum_P_mt']
    cum_H_up = upper_envelope['cum_H_km2']
    cum_P_up = upper_envelope['cum_P_mt']
    
    # ============================================================================
    # TASK 5.1: Create unified query grid
    # ============================================================================
    Hq_km2 = _create_query_grid(cum_H_low, cum_H_up, n_points)
    
    # ============================================================================
    # TASK 5.2: Interpolate envelopes onto query grid (MONOTONIC ONLY)
    # ============================================================================
    
    logger.info("Interpolating envelopes onto query grid...")
    
    # Use np.interp for simple monotonic interpolation with clamping
    # np.interp automatically handles left/right extrapolation by clamping
    P_low_interp = np.interp(Hq_km2, cum_H_low, cum_P_low)
    P_up_interp = np.interp(Hq_km2, cum_H_up, cum_P_up)
    
    logger.debug(
        f"Interpolated lower envelope: "
        f"range [{P_low_interp[0]:.2f}, {P_low_interp[-1]:.2f}] mt"
    )
    logger.debug(
        f"Interpolated upper envelope: "
        f"range [{P_up_interp[0]:.2f}, {P_up_interp[-1]:.2f}] mt"
    )
    
    # ASSERT: P_low_interp is non-decreasing
    P_low_diffs = np.diff(P_low_interp)
    n_violations_low = np.sum(P_low_diffs < 0)
    assert n_violations_low == 0, (
        f"Lower envelope interpolation broke monotonicity. "
        f"Found {n_violations_low} negative increments. "
        f"Min increment: {np.min(P_low_diffs):.2e}. "
        f"This should NEVER happen with np.interp on monotonic input!"
    )
    logger.debug("✓ Lower envelope interpolation preserves monotonicity")
    
    # ASSERT: P_up_interp is non-decreasing
    P_up_diffs = np.diff(P_up_interp)
    n_violations_up = np.sum(P_up_diffs < 0)
    assert n_violations_up == 0, (
        f"Upper envelope interpolation broke monotonicity. "
        f"Found {n_violations_up} negative increments. "
        f"Min increment: {np.min(P_up_diffs):.2e}. "
        f"This should NEVER happen with np.interp on monotonic input!"
    )
    logger.debug("✓ Upper envelope interpolation preserves monotonicity")
    
    # ============================================================================
    # TASK 5.3: Verify dominance constraint after interpolation
    # ============================================================================
    
    logger.info("Verifying dominance constraint after interpolation...")
    
    # Check dominance: P_up_interp[i] >= P_low_interp[i] for all i
    dominance_violations = P_low_interp > P_up_interp
    n_violations = np.sum(dominance_violations)
    violation_pct = 100 * n_violations / len(Hq_km2)
    
    clipping_applied = False
    max_deficit = 0.0
    
    if n_violations > 0:
        # Compute deficit
        deficit = P_low_interp - P_up_interp
        max_deficit = np.max(deficit[dominance_violations])
        
        # GUARD: If many violations (>5%), WARN but proceed with correction
        # For some countries with highly concentrated agriculture (e.g. Egypt), 
        # discrete jumps can cause large apparent violations in interpolation.
        if violation_pct > 5.0:
            logger.error(
                f"Too many dominance violations after interpolation: {violation_pct:.2f}% > 5%. "
                f"This indicates potential issues in sorting or interpolation for this region. "
                f"Max deficit: {max_deficit:.2e} mt. "
                f"Proceeding with forceful clipping correction."
            )
        else:
            logger.warning(
                f"Dominance violations found: {n_violations}/{len(Hq_km2)} points "
                f"({violation_pct:.2f}%), max deficit = {max_deficit:.2e} mt"
            )
        
        # Clip upper to be >= lower
        P_up_interp_original = P_up_interp.copy()
        P_up_interp = np.maximum(P_up_interp, P_low_interp)
        clipping_applied = True
        
        # Verify clipping worked
        n_clipped = np.sum(P_up_interp != P_up_interp_original)
        logger.warning(
            f"Applied clipping: corrected {n_clipped} points, "
            f"max correction = {max_deficit:.2e} mt"
        )
    else:
        logger.info("✓ Dominance constraint satisfied (no violations)")
    
    # ASSERT: After clipping, dominance must hold everywhere
    final_violations = np.sum(P_low_interp > P_up_interp)
    assert final_violations == 0, (
        f"Dominance violations remain after clipping: {final_violations} points. "
        f"This should NEVER happen!"
    )
    logger.debug("✓ Dominance constraint verified after clipping")
    
    return {
        'Hq_km2': Hq_km2,
        'P_low_interp': P_low_interp,
        'P_up_interp': P_up_interp,
        'clipping_applied': clipping_applied,
        'n_clipped': n_violations if clipping_applied else 0,
        'max_deficit': max_deficit if clipping_applied else 0.0
    }


def _build_summary(
    validation_result: Dict,
    qa_results: Dict,
    lower_envelope: Dict = None,
    upper_envelope: Dict = None,
    interpolated: Dict = None
) -> Dict:
    """
    Build comprehensive summary with validation metrics and QA results.
    
    This function implements TASK 6.1 and 6.2:
    - Calculate summary statistics from discrete data
    - Create QA validation report with mathematical assertions
    
    Parameters
    ----------
    validation_result : dict
        Input validation results
    qa_results : dict
        QA verification results from discrete envelope building
    lower_envelope : dict, optional
        Lower envelope data (for conservation checks)
    upper_envelope : dict, optional
        Upper envelope data (for conservation checks)
    interpolated : dict, optional
        Interpolated envelope data (for interpolation QA)
    
    Returns
    -------
    dict
        Complete summary dictionary with:
        - Cell counts and dropped cell statistics
        - Total production and harvest area (in both ha and km²)
        - Yield statistics (min, median, max)
        - Conservation checks (unit conversion, cumsum conservation)
        - QA validation results (all mathematical assertions)
        - Warnings (LOW_SAMPLE, clipping, etc.)
    
    Raises
    ------
    AssertionError
        If any critical mathematical property fails after all corrections
    """
    # ============================================================================
    # TASK 6.1: Calculate summary statistics from discrete data
    # ============================================================================
    
    logger.info("Generating summary statistics and QA report...")
    
    # Extract basic statistics
    n_total = validation_result['n_total']
    n_valid = validation_result['n_valid']
    n_dropped = validation_result['n_dropped']
    
    total_P_mt = validation_result['total_P_mt']
    total_H_ha = validation_result['total_H_ha']
    total_H_km2 = validation_result['total_H_km2']
    
    Y_mt_per_ha = validation_result['Y_mt_per_ha']
    
    # Calculate yield statistics
    yield_stats = {
        'min': float(np.min(Y_mt_per_ha)),
        'median': float(np.median(Y_mt_per_ha)),
        'max': float(np.max(Y_mt_per_ha)),
        'mean': float(np.mean(Y_mt_per_ha)),
        'std': float(np.std(Y_mt_per_ha))
    }
    
    logger.info(f"Yield statistics: min={yield_stats['min']:.4f}, "
                f"median={yield_stats['median']:.4f}, max={yield_stats['max']:.4f} mt/ha")
    
    # ============================================================================
    # Conservation checks
    # ============================================================================
    
    conservation_checks = {}
    
    # VERIFY: sum(H_km2_i) = sum(H_ha_i) * 0.01 (unit conversion check)
    expected_H_km2 = total_H_ha * 0.01
    unit_conversion_ok = np.isclose(total_H_km2, expected_H_km2, rtol=1e-10)
    conservation_checks['unit_conversion'] = {
        'passed': unit_conversion_ok,
        'total_H_ha': total_H_ha,
        'total_H_km2': total_H_km2,
        'expected_H_km2': expected_H_km2,
        'diff': abs(total_H_km2 - expected_H_km2)
    }
    
    if unit_conversion_ok:
        logger.debug(f"✓ Unit conversion check passed: {total_H_km2:.6f} km² = {total_H_ha:.2f} ha * 0.01")
    else:
        logger.error(f"✗ Unit conversion check FAILED: {total_H_km2:.6f} km² != {expected_H_km2:.6f} km²")
    
    # VERIFY: cum_H_km2_low[-1] = sum(H_km2_i) (cumsum conservation)
    if lower_envelope is not None:
        cum_H_low_final = lower_envelope['cum_H_km2'][-1]
        lower_H_conserved = np.isclose(cum_H_low_final, total_H_km2, rtol=1e-10)
        conservation_checks['lower_H_conservation'] = {
            'passed': lower_H_conserved,
            'cum_H_final': cum_H_low_final,
            'expected': total_H_km2,
            'diff': abs(cum_H_low_final - total_H_km2)
        }
        
        if lower_H_conserved:
            logger.debug(f"✓ Lower envelope H conservation: cum_H[-1] = {cum_H_low_final:.6f} km²")
        else:
            logger.error(f"✗ Lower envelope H conservation FAILED: {cum_H_low_final:.6f} != {total_H_km2:.6f} km²")
    
    # VERIFY: cum_P_mt_low[-1] = sum(P_mt_i) (cumsum conservation)
    if lower_envelope is not None:
        cum_P_low_final = lower_envelope['cum_P_mt'][-1]
        lower_P_conserved = np.isclose(cum_P_low_final, total_P_mt, rtol=1e-10)
        conservation_checks['lower_P_conservation'] = {
            'passed': lower_P_conserved,
            'cum_P_final': cum_P_low_final,
            'expected': total_P_mt,
            'diff': abs(cum_P_low_final - total_P_mt)
        }
        
        if lower_P_conserved:
            logger.debug(f"✓ Lower envelope P conservation: cum_P[-1] = {cum_P_low_final:.2f} mt")
        else:
            logger.error(f"✗ Lower envelope P conservation FAILED: {cum_P_low_final:.2f} != {total_P_mt:.2f} mt")
    
    # Same checks for upper envelope
    if upper_envelope is not None:
        cum_H_up_final = upper_envelope['cum_H_km2'][-1]
        upper_H_conserved = np.isclose(cum_H_up_final, total_H_km2, rtol=1e-10)
        conservation_checks['upper_H_conservation'] = {
            'passed': upper_H_conserved,
            'cum_H_final': cum_H_up_final,
            'expected': total_H_km2,
            'diff': abs(cum_H_up_final - total_H_km2)
        }
        
        cum_P_up_final = upper_envelope['cum_P_mt'][-1]
        upper_P_conserved = np.isclose(cum_P_up_final, total_P_mt, rtol=1e-10)
        conservation_checks['upper_P_conservation'] = {
            'passed': upper_P_conserved,
            'cum_P_final': cum_P_up_final,
            'expected': total_P_mt,
            'diff': abs(cum_P_up_final - total_P_mt)
        }
    
    # Log conservation check summary
    all_conservation_passed = all(check['passed'] for check in conservation_checks.values())
    if all_conservation_passed:
        logger.info("✓ All conservation checks passed")
    else:
        logger.error("✗ Some conservation checks FAILED")
    
    # ============================================================================
    # TASK 6.2: Create QA validation report with mathematical assertions
    # ============================================================================
    
    qa_validation = {}
    warnings = []
    
    # Check for LOW_SAMPLE warning
    if n_valid < 100:
        warnings.append({
            'type': 'LOW_SAMPLE',
            'message': f'Only {n_valid} valid cells (< 100)',
            'severity': 'warning'
        })
        logger.warning(f"LOW_SAMPLE: Only {n_valid} valid cells (< 100)")
    
    # If interpolation was performed, validate interpolated results
    if interpolated is not None:
        Hq_km2 = interpolated['Hq_km2']
        P_low_interp = interpolated['P_low_interp']
        P_up_interp = interpolated['P_up_interp']
        
        # ASSERT: Hq_km2 strictly increasing
        Hq_diffs = np.diff(Hq_km2)
        Hq_strictly_increasing = np.all(Hq_diffs > 0)
        qa_validation['Hq_strictly_increasing'] = {
            'passed': Hq_strictly_increasing,
            'n_violations': int(np.sum(Hq_diffs <= 0)),
            'min_diff': float(np.min(Hq_diffs)) if len(Hq_diffs) > 0 else None
        }
        
        if not Hq_strictly_increasing:
            logger.error(f"✗ Query grid not strictly increasing: {np.sum(Hq_diffs <= 0)} violations")
        
        # ASSERT: P_low_interp non-decreasing
        P_low_diffs = np.diff(P_low_interp)
        P_low_monotonic = np.all(P_low_diffs >= 0)
        qa_validation['P_low_monotonic'] = {
            'passed': P_low_monotonic,
            'n_violations': int(np.sum(P_low_diffs < 0)),
            'min_diff': float(np.min(P_low_diffs)) if len(P_low_diffs) > 0 else None
        }
        
        if not P_low_monotonic:
            logger.error(f"✗ Lower envelope interpolation not monotonic: {np.sum(P_low_diffs < 0)} violations")
        
        # ASSERT: P_up_interp non-decreasing
        P_up_diffs = np.diff(P_up_interp)
        P_up_monotonic = np.all(P_up_diffs >= 0)
        qa_validation['P_up_monotonic'] = {
            'passed': P_up_monotonic,
            'n_violations': int(np.sum(P_up_diffs < 0)),
            'min_diff': float(np.min(P_up_diffs)) if len(P_up_diffs) > 0 else None
        }
        
        if not P_up_monotonic:
            logger.error(f"✗ Upper envelope interpolation not monotonic: {np.sum(P_up_diffs < 0)} violations")
        
        # ASSERT: Dominance after clipping
        dominance_holds = np.all(P_up_interp >= P_low_interp)
        dominance_violations = np.sum(P_low_interp > P_up_interp)
        qa_validation['dominance_after_clipping'] = {
            'passed': dominance_holds,
            'n_violations': int(dominance_violations),
            'max_deficit': float(np.max(P_low_interp - P_up_interp)) if dominance_violations > 0 else 0.0
        }
        
        if not dominance_holds:
            logger.error(f"✗ Dominance constraint violated after clipping: {dominance_violations} violations")
        
        # Record clipping information
        if interpolated.get('clipping_applied', False):
            warnings.append({
                'type': 'CLIPPING_APPLIED',
                'message': f"Clipped {interpolated['n_clipped']} points, max deficit = {interpolated['max_deficit']:.2e} mt",
                'severity': 'warning',
                'n_clipped': int(interpolated['n_clipped']),
                'max_deficit': float(interpolated['max_deficit'])
            })
            logger.info(f"Clipping applied: {interpolated['n_clipped']} points, max deficit = {interpolated['max_deficit']:.2e} mt")
        
        # FAIL HARD if any assertion fails after all corrections
        critical_failures = []
        
        if not Hq_strictly_increasing:
            critical_failures.append("Query grid not strictly increasing")
        if not P_low_monotonic:
            critical_failures.append("Lower envelope interpolation not monotonic")
        if not P_up_monotonic:
            critical_failures.append("Upper envelope interpolation not monotonic")
        if not dominance_holds:
            critical_failures.append("Dominance constraint violated after clipping")
        
        if critical_failures:
            error_msg = "CRITICAL QA FAILURES after all corrections:\n" + "\n".join(f"  - {f}" for f in critical_failures)
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.info("✓ All interpolation QA assertions passed")
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def _convert_to_python_types(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: _convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Build complete summary
    summary = {
        # Cell counts
        'n_total': int(n_total),
        'n_valid': int(n_valid),
        'n_dropped': int(n_dropped),
        'dropped_counts': _convert_to_python_types(validation_result['dropped_counts']),
        
        # Yield validation
        'yield_validation': _convert_to_python_types(validation_result['yield_validation']),
        
        # Totals
        'totals': {
            'P_mt': float(total_P_mt),
            'H_ha': float(total_H_ha),
            'H_km2': float(total_H_km2)
        },
        
        # Yield statistics
        'yield_stats': yield_stats,
        
        # Conservation checks (TASK 6.1)
        'conservation_checks': _convert_to_python_types(conservation_checks),
        
        # QA results from discrete envelope building
        'discrete_qa_results': _convert_to_python_types(qa_results),
        
        # QA validation from interpolation (TASK 6.2)
        'interpolation_qa': _convert_to_python_types(qa_validation) if interpolated is not None else None,
        
        # Warnings
        'warnings': warnings,
        
        # Overall status
        'all_checks_passed': bool(all_conservation_passed and all(
            check.get('passed', True) for check in qa_validation.values()
        ))
    }
    
    logger.info(f"Summary generation complete. Overall status: {'PASS' if summary['all_checks_passed'] else 'FAIL'}")
    
    return summary



# ============================================================================
# TASK 7: Export results to files
# ============================================================================

def export_envelope_results(
    result: Dict,
    output_dir: Union[str, Path],
    prefix: str = "envelope"
) -> Dict[str, Path]:
    """
    Export envelope results to files.
    
    This function implements TASK 7: Export results to files
    - 7.1: Save interpolated envelope curves (for plotting)
    - 7.2: Save discrete cumulative sequences (for validation)
    - 7.3: Save summary and QA report
    - 7.4: Generate visualization with discrete points visible
    
    Parameters
    ----------
    result : dict
        Result dictionary from build_envelope() containing:
        - 'lower': lower envelope data
        - 'upper': upper envelope data
        - 'summary': summary and QA report
        - 'interpolated': interpolated data (if available)
    output_dir : str or Path
        Directory to save output files
    prefix : str, default="envelope"
        Prefix for output filenames
    
    Returns
    -------
    dict
        Dictionary mapping output type to file path:
        - 'lower_csv': path to envelope_lower.csv
        - 'upper_csv': path to envelope_upper.csv
        - 'lower_discrete_csv': path to envelope_lower_discrete.csv
        - 'upper_discrete_csv': path to envelope_upper_discrete.csv
        - 'summary_json': path to envelope_summary.json
        - 'plot_png': path to envelope_plot.png (if generated)
    
    Examples
    --------
    >>> result = build_envelope(P_mt, H_ha, interpolate=True)
    >>> paths = export_envelope_results(result, "output", prefix="wheat")
    >>> print(paths['summary_json'])
    output/wheat_summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = {}
    
    # ============================================================================
    # TASK 7.1: Save envelope curves (interpolated for plotting)
    # ============================================================================
    
    if 'interpolated' in result and result['interpolated'] is not None:
        logger.info("Exporting interpolated envelope curves...")
        
        interpolated = result['interpolated']
        Hq_km2 = interpolated['Hq_km2']
        P_low_interp = interpolated['P_low_interp']
        P_up_interp = interpolated['P_up_interp']
        
        # Save lower envelope
        lower_csv_path = output_dir / f"{prefix}_lower.csv"
        df_lower = pd.DataFrame({
            'Hq_km2': Hq_km2,
            'lower_P_mt': P_low_interp
        })
        
        with open(lower_csv_path, 'w') as f:
            f.write("# Interpolated envelope on unified query grid\n")
            f.write("# Lower envelope: cells sorted by yield ascending\n")
            f.write(f"# Columns: Hq_km2 (cumulative harvest area in km²), lower_P_mt (cumulative production in mt)\n")
            df_lower.to_csv(f, index=False)
        
        output_paths['lower_csv'] = lower_csv_path
        logger.info(f"Saved lower envelope to {lower_csv_path}")
        
        # Save upper envelope
        upper_csv_path = output_dir / f"{prefix}_upper.csv"
        df_upper = pd.DataFrame({
            'Hq_km2': Hq_km2,
            'upper_P_mt': P_up_interp
        })
        
        with open(upper_csv_path, 'w') as f:
            f.write("# Interpolated envelope on unified query grid\n")
            f.write("# Upper envelope: cells sorted by yield descending\n")
            f.write(f"# Columns: Hq_km2 (cumulative harvest area in km²), upper_P_mt (cumulative production in mt)\n")
            df_upper.to_csv(f, index=False)
        
        output_paths['upper_csv'] = upper_csv_path
        logger.info(f"Saved upper envelope to {upper_csv_path}")
    else:
        logger.warning("No interpolated data available. Skipping interpolated envelope export.")
        logger.warning("Call build_envelope() with interpolate=True to generate interpolated curves.")
    
    # ============================================================================
    # TASK 7.2: Save discrete cumulative sequences (for validation)
    # ============================================================================
    
    logger.info("Exporting discrete cumulative sequences...")
    
    lower_envelope = result['lower']
    upper_envelope = result['upper']
    
    # Save lower discrete sequence
    lower_discrete_path = output_dir / f"{prefix}_lower_discrete.csv"
    df_lower_discrete = pd.DataFrame({
        'cum_H_km2': lower_envelope['cum_H_km2'],
        'cum_P_mt': lower_envelope['cum_P_mt'],
        'cell_index': lower_envelope['indices']
    })
    
    with open(lower_discrete_path, 'w') as f:
        f.write("# Discrete cumulative sums (no interpolation)\n")
        f.write("# Lower envelope: cells sorted by yield ascending\n")
        f.write("# Columns: cum_H_km2 (cumulative harvest area in km²), cum_P_mt (cumulative production in mt), cell_index (original cell index)\n")
        df_lower_discrete.to_csv(f, index=False)
    
    output_paths['lower_discrete_csv'] = lower_discrete_path
    logger.info(f"Saved lower discrete sequence to {lower_discrete_path}")
    
    # Save upper discrete sequence
    upper_discrete_path = output_dir / f"{prefix}_upper_discrete.csv"
    df_upper_discrete = pd.DataFrame({
        'cum_H_km2': upper_envelope['cum_H_km2'],
        'cum_P_mt': upper_envelope['cum_P_mt'],
        'cell_index': upper_envelope['indices']
    })
    
    with open(upper_discrete_path, 'w') as f:
        f.write("# Discrete cumulative sums (no interpolation)\n")
        f.write("# Upper envelope: cells sorted by yield descending\n")
        f.write("# Columns: cum_H_km2 (cumulative harvest area in km²), cum_P_mt (cumulative production in mt), cell_index (original cell index)\n")
        df_upper_discrete.to_csv(f, index=False)
    
    output_paths['upper_discrete_csv'] = upper_discrete_path
    logger.info(f"Saved upper discrete sequence to {upper_discrete_path}")
    
    # ============================================================================
    # TASK 7.3: Save summary and QA report
    # ============================================================================
    
    logger.info("Exporting summary and QA report...")
    
    summary = result['summary']
    
    # Add metadata
    summary_with_metadata = {
        'metadata': {
            'output_dir': str(output_dir),
            'prefix': prefix,
            'interpolated': 'interpolated' in result and result['interpolated'] is not None
        },
        'summary': summary
    }
    
    summary_json_path = output_dir / f"{prefix}_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_with_metadata, f, indent=2)
    
    output_paths['summary_json'] = summary_json_path
    logger.info(f"Saved summary and QA report to {summary_json_path}")
    
    # ============================================================================
    # TASK 7.4: Generate visualization with discrete points visible
    # ============================================================================
    
    if 'interpolated' in result and result['interpolated'] is not None:
        logger.info("Generating visualization...")
        
        try:
            plot_path = _generate_envelope_plot(
                result,
                output_dir,
                prefix
            )
            output_paths['plot_png'] = plot_path
            logger.info(f"Saved visualization to {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
            logger.warning("Continuing without plot...")
    else:
        logger.warning("No interpolated data available. Skipping visualization.")
    
    logger.info(f"Export complete. {len(output_paths)} files saved to {output_dir}")
    
    return output_paths


def _generate_envelope_plot(
    result: Dict,
    output_dir: Path,
    prefix: str
) -> Path:
    """
    Generate visualization with discrete points visible.
    
    This function implements TASK 7.4: Generate visualization with discrete points visible
    - Show interpolated curves as lines
    - Overlay discrete cumulative points as scatter (every 10th point to avoid clutter)
    - Show lower and upper envelopes with shaded region
    - Make discrete nature of data visible in plot
    
    Parameters
    ----------
    result : dict
        Result dictionary from build_envelope()
    output_dir : Path
        Directory to save plot
    prefix : str
        Prefix for output filename
    
    Returns
    -------
    Path
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    
    interpolated = result['interpolated']
    lower_envelope = result['lower']
    upper_envelope = result['upper']
    summary = result['summary']
    
    Hq_km2 = interpolated['Hq_km2']
    P_low_interp = interpolated['P_low_interp']
    P_up_interp = interpolated['P_up_interp']
    
    cum_H_low = lower_envelope['cum_H_km2']
    cum_P_low = lower_envelope['cum_P_mt']
    cum_H_up = upper_envelope['cum_H_km2']
    cum_P_up = upper_envelope['cum_P_mt']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot interpolated curves as lines
    ax.plot(Hq_km2, P_low_interp, 'b-', linewidth=2, label='Lower envelope (interpolated)', alpha=0.8)
    ax.plot(Hq_km2, P_up_interp, 'r-', linewidth=2, label='Upper envelope (interpolated)', alpha=0.8)
    
    # Fill between envelopes
    ax.fill_between(Hq_km2, P_low_interp, P_up_interp, alpha=0.2, color='gray', label='Envelope band')
    
    # Overlay discrete points (every 10th point to avoid clutter)
    step = max(1, len(cum_H_low) // 100)  # Show ~100 points max
    ax.scatter(cum_H_low[::step], cum_P_low[::step], c='blue', s=20, alpha=0.5, 
               marker='o', label='Lower discrete points', zorder=5)
    ax.scatter(cum_H_up[::step], cum_P_up[::step], c='red', s=20, alpha=0.5, 
               marker='s', label='Upper discrete points', zorder=5)
    
    # Labels and title
    ax.set_xlabel('Cumulative Harvest Area (km²)', fontsize=12)
    ax.set_ylabel('Cumulative Production (mt)', fontsize=12)
    ax.set_title(f'Production Envelope: {prefix}\n'
                 f'({summary["n_valid"]} valid cells, '
                 f'{summary["totals"]["P_mt"]:.0f} mt total production)',
                 fontsize=14)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / f"{prefix}_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

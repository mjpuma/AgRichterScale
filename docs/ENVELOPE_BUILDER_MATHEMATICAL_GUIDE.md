# Envelope Builder: Mathematical Guide

## Overview

This guide provides a comprehensive mathematical explanation of the robust envelope builder for discrete gridded agricultural data. The envelope builder creates maximum and minimum cumulative production curves that bound the range of possible production outcomes under different disruption scenarios.

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Discrete vs Continuous Nature](#discrete-vs-continuous-nature)
3. [Core Mathematical Properties](#core-mathematical-properties)
4. [Why Yield Sorting Guarantees Dominance](#why-yield-sorting-guarantees-dominance)
5. [Algorithm Steps](#algorithm-steps)
6. [QA Validation Rules](#qa-validation-rules)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Usage Examples](#usage-examples)

---

## Mathematical Foundation

### Problem Statement

Given N discrete grid cells, each with:
- **P_i**: Production (metric tons)
- **H_i**: Harvest area (hectares)
- **Y_i**: Yield (metric tons per hectare), where Y_i = P_i / H_i

**Goal**: Construct two cumulative production curves:
- **Lower envelope**: Minimum cumulative production as a function of cumulative harvest area
- **Upper envelope**: Maximum cumulative production as a function of cumulative harvest area

### Key Principle

The envelope is built from cumulative sums of discrete cells sorted by yield:
- **Lower envelope**: Sort cells by yield ascending (least productive first)
- **Upper envelope**: Sort cells by yield descending (most productive first)


### Mathematical Invariants

The following properties MUST hold throughout the calculation:

1. **Monotonicity**: Cumulative sequences are monotonic non-decreasing by construction
2. **Dominance**: Upper envelope ≥ Lower envelope at all cumulative harvest levels
3. **Conservation**: Total production and harvest area are preserved
4. **No Interpolation Artifacts**: Interpolation must not violate monotonicity

---

## Discrete vs Continuous Nature

### Understanding the Data

**CRITICAL**: We work with **discrete grid cells**, not continuous functions.

- Each grid cell i represents a finite spatial unit with specific (P_i, H_i, Y_i)
- We have exactly N cells (e.g., N = 10,000 for a global crop dataset)
- The cumulative sequences represent sums over the first k cells when sorted by yield

### Discrete Cumulative Sequences

For the lower envelope (sorted by yield ascending):

```
cum_H_km2[k] = Σ(i=0 to k) H_i     (sum of harvest area for first k cells)
cum_P_mt[k]  = Σ(i=0 to k) P_i     (sum of production for first k cells)
```

These are **discrete sequences** with exactly N points.

### Continuous Interpolation (Optional)

For plotting and analysis, we may interpolate onto a unified query grid:
- Create a shared grid Hq_km2 with M points (typically M = 200)
- Interpolate: P_low_interp = interp(Hq_km2, cum_H_km2_low, cum_P_mt_low)
- This maps discrete data to a continuous representation

**Important**: The interpolation is a visualization/analysis convenience. The mathematical properties are established on the discrete data.

---

## Core Mathematical Properties

### Property 1: Monotonicity

**Definition**: A sequence is monotonic non-decreasing if x[i] ≤ x[i+1] for all i.

**Why it matters**: Cumulative sums are inherently monotonic because we're adding non-negative quantities.

**Verification**:
```python
# For cumulative harvest area (strictly increasing)
assert np.all(np.diff(cum_H_km2) > 0)

# For cumulative production (non-decreasing)
assert np.all(np.diff(cum_P_mt) >= 0)
```

**What it catches**: 
- Negative values in input data
- Sorting errors
- Numerical precision issues


### Property 2: Conservation

**Definition**: The final cumulative values must equal the sum of all inputs.

**Mathematical statement**:
```
cum_H_km2[-1] = Σ(i=0 to N-1) H_i
cum_P_mt[-1]  = Σ(i=0 to N-1) P_i
```

**Why it matters**: Ensures no data is lost or duplicated during sorting and accumulation.

**Verification**:
```python
assert np.isclose(cum_H_km2[-1], np.sum(H_km2), rtol=1e-10)
assert np.isclose(cum_P_mt[-1], np.sum(P_mt), rtol=1e-10)
```

**What it catches**:
- Missing or duplicated cells during sorting
- Unit conversion errors
- Numerical precision loss

### Property 3: Dominance

**Definition**: At any cumulative harvest level H, the upper envelope production must be ≥ lower envelope production.

**Mathematical statement**: For any harvest level H_target:
```
P_upper(H_target) ≥ P_lower(H_target)
```

**Why it matters**: This is the fundamental property that makes the envelope meaningful. The upper bound must always be above the lower bound.

**Verification**: Test at multiple harvest levels:
```python
for H_target in test_points:
    k_low = find_index_where(cum_H_low >= H_target)
    k_up = find_index_where(cum_H_up >= H_target)
    assert cum_P_up[k_up] >= cum_P_low[k_low]
```

**What it catches**:
- Incorrect yield sorting
- Fundamental algorithmic errors
- Data inconsistencies

### Property 4: Yield Ordering

**Definition**: After sorting, yields must be in the correct order.

**Mathematical statement**:
- Lower envelope: Y_sorted[i] ≤ Y_sorted[i+1] for all i (ascending)
- Upper envelope: Y_sorted[i] ≥ Y_sorted[i+1] for all i (descending)

**Why it matters**: Correct yield ordering is the foundation for the dominance property.

**Verification**:
```python
# Lower envelope
assert np.all(np.diff(Y_sorted_low) >= 0)

# Upper envelope
assert np.all(np.diff(Y_sorted_up) <= 0)
```

**What it catches**:
- Sorting implementation errors
- NaN or infinite yield values
- Data corruption

---

## Why Yield Sorting Guarantees Dominance

This is the key mathematical insight that makes the envelope algorithm work.


### Theorem: Yield Sorting Guarantees Dominance

**Statement**: If we sort cells by yield descending for the upper envelope and ascending for the lower envelope, then at any cumulative harvest level H, the upper envelope production is ≥ lower envelope production.

**Proof**:

Consider any harvest level H_target. We want to show:
```
P_upper(H_target) ≥ P_lower(H_target)
```

**Step 1**: Define the problem
- We have N cells with yields Y_1, Y_2, ..., Y_N
- Each cell i has harvest area H_i and production P_i = Y_i × H_i
- We want to select a subset S of cells such that Σ(i∈S) H_i ≈ H_target

**Step 2**: Lower envelope construction
- Sort cells by yield ascending: Y_1 ≤ Y_2 ≤ ... ≤ Y_N
- Select cells greedily from lowest yield: S_low = {1, 2, ..., k_low}
- Where k_low is chosen such that Σ(i=1 to k_low) H_i ≈ H_target
- Production: P_lower = Σ(i=1 to k_low) P_i = Σ(i=1 to k_low) Y_i × H_i

**Step 3**: Upper envelope construction
- Sort cells by yield descending: Y_1 ≥ Y_2 ≥ ... ≥ Y_N
- Select cells greedily from highest yield: S_up = {1, 2, ..., k_up}
- Where k_up is chosen such that Σ(i=1 to k_up) H_i ≈ H_target
- Production: P_upper = Σ(i=1 to k_up) P_i = Σ(i=1 to k_up) Y_i × H_i

**Step 4**: Compare productions
- For the same total harvest area H_target, we're comparing:
  - P_lower: sum of (low yields × areas)
  - P_upper: sum of (high yields × areas)

**Step 5**: Key insight
- Since yields are sorted, the upper envelope uses cells with Y_i ≥ median(Y)
- The lower envelope uses cells with Y_i ≤ median(Y)
- For the same total area, higher yields produce more production
- Therefore: P_upper ≥ P_lower

**Formal argument**:
Let Y_avg_low = (Σ Y_i × H_i) / (Σ H_i) for cells in S_low
Let Y_avg_up = (Σ Y_i × H_i) / (Σ H_i) for cells in S_up

By construction: Y_avg_up ≥ Y_avg_low (we selected high-yield cells for upper)

Since both have approximately the same total harvest area H_target:
```
P_upper = Y_avg_up × H_target ≥ Y_avg_low × H_target = P_lower
```

**Q.E.D.**

### Intuitive Explanation

Think of it this way:
- **Lower envelope**: "What's the minimum production if I'm forced to use the worst land first?"
- **Upper envelope**: "What's the maximum production if I can use the best land first?"

Since "best land" (high yield) produces more than "worst land" (low yield) for the same area, the upper envelope is always ≥ lower envelope.

### When Dominance Can Fail

Dominance should **never** fail on discrete data if the algorithm is correct. If it does fail, it indicates:

1. **Sorting error**: Yields not sorted correctly
2. **Data corruption**: NaN, Inf, or negative values in input
3. **Numerical precision**: Extreme values causing floating-point errors
4. **Implementation bug**: Error in cumulative sum calculation

**Action**: If dominance fails, STOP immediately and investigate. This is a critical error.

---

## Algorithm Steps

### Step 1: Input Validation

**Purpose**: Ensure data quality and consistency

**Operations**:
1. Check for finite values: P_mt ≥ 0, H_ha > 0, Y_mt_per_ha > 0
2. Drop NaNs, zeros, and negatives
3. Count and log dropped cells by reason
4. Compute yield if not provided: Y = P / H
5. Verify yield consistency if provided: |P - Y×H| < tolerance

**Mathematical checks**:
- N_valid = count of valid cells
- Total_P = Σ P_i (before and after validation)
- Total_H = Σ H_i (before and after validation)


### Step 2: Unit Conversion

**Purpose**: Convert units exactly once with explicit tracking

**Operations**:
1. Convert H_ha to H_km2 = H_ha × 0.01
2. Keep P_mt as-is (metric tons)
3. Verify: Σ H_km2 = Σ H_ha × 0.01 exactly

**Critical rule**: Convert units ONCE and track explicitly. Never reconvert.

**Why it matters**: Multiple conversions can introduce numerical errors and unit confusion.

### Step 3: Sort Cells by Yield

**Purpose**: Order cells for envelope construction

**Operations**:
1. **Lower envelope**: Sort by yield ascending (Y_1 ≤ Y_2 ≤ ... ≤ Y_N)
2. **Upper envelope**: Sort by yield descending (Y_1 ≥ Y_2 ≥ ... ≥ Y_N)

**Verification**:
```python
# Lower envelope
sort_indices_low = np.argsort(Y)
Y_sorted_low = Y[sort_indices_low]
assert np.all(np.diff(Y_sorted_low) >= 0)

# Upper envelope
sort_indices_up = np.argsort(Y)[::-1]
Y_sorted_up = Y[sort_indices_up]
assert np.all(np.diff(Y_sorted_up) <= 0)
```

### Step 4: Compute Cumulative Sums

**Purpose**: Build discrete envelope sequences

**Operations**:
1. Sort H_km2 and P_mt according to yield order
2. Compute cumulative sums:
   ```python
   cum_H_km2 = np.cumsum(H_sorted)
   cum_P_mt = np.cumsum(P_sorted)
   ```

**Verification**:
```python
# Monotonicity
assert np.all(np.diff(cum_H_km2) > 0)  # Strictly increasing
assert np.all(np.diff(cum_P_mt) >= 0)  # Non-decreasing

# Conservation
assert np.isclose(cum_H_km2[-1], np.sum(H_km2))
assert np.isclose(cum_P_mt[-1], np.sum(P_mt))
```

### Step 5: Verify Dominance (Discrete)

**Purpose**: Ensure upper ≥ lower at sampled harvest levels

**Operations**:
1. Sample 10 harvest levels from lower envelope
2. For each H_target, find corresponding production in both envelopes
3. Verify: P_upper(H_target) ≥ P_lower(H_target)

**Implementation**:
```python
test_indices = np.linspace(0, len(cum_H_low)-1, 10, dtype=int)
for i in test_indices:
    H_target = cum_H_low[i]
    P_low = cum_P_low[i]
    
    k_up = np.searchsorted(cum_H_up, H_target)
    P_up = cum_P_up[k_up]
    
    assert P_up >= P_low, "Dominance violated!"
```

### Step 6: Interpolate (Optional)

**Purpose**: Map discrete sequences to continuous query grid for plotting

**Operations**:
1. Create query grid Hq_km2 with ≥200 points
2. Use log-spacing if span ≥ 3 orders of magnitude, else linear
3. Interpolate using np.interp (monotonic, with clamping)
4. Verify monotonicity after interpolation
5. Verify dominance after interpolation
6. Clip if needed: P_up_interp = max(P_up_interp, P_low_interp)

**Verification**:
```python
# Monotonicity
assert np.all(np.diff(P_low_interp) >= 0)
assert np.all(np.diff(P_up_interp) >= 0)

# Dominance
violations = P_low_interp > P_up_interp
if np.any(violations):
    # Clip to enforce dominance
    P_up_interp = np.maximum(P_up_interp, P_low_interp)
    
# Final check
assert np.all(P_up_interp >= P_low_interp)
```


### Step 7: Generate Summary and QA Report

**Purpose**: Document results and validate mathematical properties

**Operations**:
1. Calculate summary statistics (N_valid, Total_H, Total_P, yield stats)
2. Record all QA checks (pass/fail)
3. Log warnings (LOW_SAMPLE, MISMATCH, clipping)
4. Export to JSON for downstream analysis

---

## QA Validation Rules

### What Each Rule Checks

| Rule | What It Checks | What It Catches | Action if Failed |
|------|----------------|-----------------|------------------|
| **lower_H_monotonic** | cum_H_low strictly increasing | Negative areas, sorting errors | STOP - critical error |
| **lower_P_monotonic** | cum_P_low non-decreasing | Negative production, sorting errors | STOP - critical error |
| **lower_Y_ascending** | Yields sorted ascending | Sorting implementation bug | STOP - critical error |
| **upper_H_monotonic** | cum_H_up strictly increasing | Negative areas, sorting errors | STOP - critical error |
| **upper_P_monotonic** | cum_P_up non-decreasing | Negative production, sorting errors | STOP - critical error |
| **upper_Y_descending** | Yields sorted descending | Sorting implementation bug | STOP - critical error |
| **lower_H_conserved** | Total harvest area preserved | Data loss during sorting | STOP - critical error |
| **lower_P_conserved** | Total production preserved | Data loss during sorting | STOP - critical error |
| **upper_H_conserved** | Total harvest area preserved | Data loss during sorting | STOP - critical error |
| **upper_P_conserved** | Total production preserved | Data loss during sorting | STOP - critical error |
| **dominance_holds** | Upper ≥ lower at all test points | Fundamental algorithm error | STOP - critical error |
| **Hq_strictly_increasing** | Query grid monotonic | Grid generation error | STOP - critical error |
| **P_low_interp_monotonic** | Interpolation preserves monotonicity | Interpolation artifact | STOP - critical error |
| **P_up_interp_monotonic** | Interpolation preserves monotonicity | Interpolation artifact | STOP - critical error |
| **dominance_after_interp** | Upper ≥ lower after interpolation | Interpolation artifact | Clip and warn if <5% |

### Warning Levels

**CRITICAL (STOP)**: Mathematical property violated
- Indicates fundamental error in algorithm or data
- Cannot proceed - results would be invalid
- Examples: dominance fails, monotonicity breaks, conservation violated

**WARNING (Continue with caution)**:
- Data quality issue but algorithm can proceed
- Results may be less reliable
- Examples: LOW_SAMPLE (<100 cells), MISMATCH (yield inconsistency), clipping (>0% but <5%)

**INFO (Normal operation)**:
- Expected behavior, no issues
- Examples: dropped NaN cells, computed yield from P/H

### Critical Guards Summary

**STOP and investigate if:**

1. **Dominance property fails on discrete data** (before interpolation)
   - This should NEVER happen with correct yield sorting
   - Indicates fundamental error in sorting or cumulative sum calculation
   
2. **Cumulative sums don't equal input totals**
   - Indicates data loss or duplication during sorting
   - Check for NaN handling, array indexing errors
   
3. **Monotonicity breaks after interpolation**
   - Should never happen with np.interp on monotonic input
   - Indicates interpolation implementation error
   
4. **More than 5% of interpolated points need clipping**
   - Small amount of clipping (<5%) is acceptable due to numerical precision
   - Large amount indicates fundamental problem with discrete envelopes
   
5. **Any mathematical assertion fails**
   - All assertions are critical mathematical properties
   - Failure indicates algorithm or data error

---

## Troubleshooting Guide

### Problem: Dominance Fails on Discrete Data

**Symptoms**:
```
AssertionError: Dominance property VIOLATED at X/10 test points
```

**Possible causes**:
1. Yields not sorted correctly
2. NaN or Inf values in yield data
3. Negative production or harvest area values
4. Numerical precision issues with extreme values

**Debugging steps**:
```python
# Check yield sorting
print("Lower envelope yields (should be ascending):")
print(Y_sorted_low[:10], "...", Y_sorted_low[-10:])
print("Diffs:", np.diff(Y_sorted_low)[:10])

print("Upper envelope yields (should be descending):")
print(Y_sorted_up[:10], "...", Y_sorted_up[-10:])
print("Diffs:", np.diff(Y_sorted_up)[:10])

# Check for invalid values
print("NaN in yields:", np.sum(~np.isfinite(Y)))
print("Negative yields:", np.sum(Y < 0))
print("Zero yields:", np.sum(Y == 0))

# Check dominance at specific point
i = 5  # Test point where dominance failed
H_target = cum_H_low[i]
P_low = cum_P_low[i]
k_up = np.searchsorted(cum_H_up, H_target)
P_up = cum_P_up[k_up]
print(f"At H={H_target:.2f}: P_low={P_low:.2f}, P_up={P_up:.2f}, deficit={P_low-P_up:.2f}")
```

**Solutions**:
- Verify input data quality (no NaN, Inf, negatives)
- Check sorting implementation
- Verify yield calculation (Y = P / H)
- Check for numerical precision issues (very large or very small values)


### Problem: Monotonicity Breaks After Interpolation

**Symptoms**:
```
AssertionError: Lower envelope interpolation broke monotonicity
```

**Possible causes**:
1. Discrete envelope not monotonic (should be caught earlier)
2. Interpolation implementation error
3. Query grid not strictly increasing

**Debugging steps**:
```python
# Check discrete envelope monotonicity
print("Discrete lower envelope monotonic:", np.all(np.diff(cum_P_low) >= 0))
print("Discrete upper envelope monotonic:", np.all(np.diff(cum_P_up) >= 0))

# Check query grid
print("Query grid monotonic:", np.all(np.diff(Hq_km2) > 0))
print("Query grid range:", Hq_km2[0], "to", Hq_km2[-1])
print("Data range:", cum_H_low[0], "to", cum_H_low[-1])

# Check interpolation
P_low_interp = np.interp(Hq_km2, cum_H_low, cum_P_low)
print("Interpolated monotonic:", np.all(np.diff(P_low_interp) >= 0))
print("Negative diffs:", np.sum(np.diff(P_low_interp) < 0))
```

**Solutions**:
- Verify discrete envelopes are monotonic before interpolation
- Check query grid is strictly increasing
- Verify no extrapolation (query grid within data range)
- Use np.interp (simplest, most reliable)

### Problem: Conservation Violated

**Symptoms**:
```
AssertionError: Lower envelope: production not conserved
```

**Possible causes**:
1. Data lost during sorting
2. Array indexing error
3. Numerical precision loss

**Debugging steps**:
```python
# Check totals before and after sorting
print("Original total P:", np.sum(P_mt))
print("Original total H:", np.sum(H_km2))

print("Sorted total P:", np.sum(P_sorted))
print("Sorted total H:", np.sum(H_sorted))

print("Cumsum final P:", cum_P_mt[-1])
print("Cumsum final H:", cum_H_km2[-1])

# Check for data loss
print("Number of cells:", len(P_mt), "->", len(P_sorted), "->", len(cum_P_mt))
```

**Solutions**:
- Verify no cells are dropped during sorting
- Check array lengths match at each step
- Use high precision (float64) for cumulative sums
- Verify sort indices are valid

### Problem: Excessive Clipping (>5%)

**Symptoms**:
```
WARNING: Dominance violations found: 150/200 points (75.0%)
```

**Possible causes**:
1. Discrete envelopes don't satisfy dominance (fundamental error)
2. Interpolation artifacts (unlikely with np.interp)
3. Query grid issues

**Debugging steps**:
```python
# Check discrete dominance first
test_indices = np.linspace(0, len(cum_H_low)-1, 10, dtype=int)
for i in test_indices:
    H_target = cum_H_low[i]
    P_low = cum_P_low[i]
    k_up = np.searchsorted(cum_H_up, H_target)
    P_up = cum_P_up[k_up]
    print(f"H={H_target:.2f}: P_low={P_low:.2f}, P_up={P_up:.2f}, OK={P_up >= P_low}")

# If discrete dominance holds, check interpolation
violations = P_low_interp > P_up_interp
print("Violation indices:", np.where(violations)[0])
print("Violation H values:", Hq_km2[violations])
```

**Solutions**:
- If discrete dominance fails: Fix sorting or data quality
- If discrete dominance holds but interpolation fails: Check query grid and interpolation method
- Small clipping (<5%) is acceptable and will be applied automatically

### Problem: Low Sample Size

**Symptoms**:
```
WARNING: LOW_SAMPLE: Only 45 valid cells (< 100)
```

**Possible causes**:
1. Small dataset (e.g., single country)
2. Many cells dropped due to invalid data
3. Aggressive filtering

**Impact**:
- Envelope may not be smooth
- Statistical properties less reliable
- Interpolation may be less accurate

**Solutions**:
- Check why cells were dropped (see dropped_counts in summary)
- Relax validation criteria if appropriate
- Accept lower quality results for small datasets
- Consider aggregating data from multiple sources

### Problem: Yield Mismatch

**Symptoms**:
```
WARNING: MISMATCH: 150 cells (15.0%) have inconsistent P vs Y*H
```

**Possible causes**:
1. Yield data from different source than P and H
2. Rounding errors in original data
3. Different calculation methods

**Impact**:
- Yield will be recomputed from P/H (P is authoritative)
- Original yield data is discarded
- Results should still be valid

**Solutions**:
- Use consistent data sources
- If mismatch is expected, don't provide Y (let it be computed)
- Check tolerance parameter (default 5%)

---

## Usage Examples

### Example 1: Basic Usage with Synthetic Data

```python
import numpy as np
from agririchter.analysis.envelope_builder import build_envelope

# Create synthetic data: 1000 cells with known properties
np.random.seed(42)
n_cells = 1000

# Generate yields from normal distribution
Y_mt_per_ha = np.random.normal(loc=5.0, scale=1.5, size=n_cells)
Y_mt_per_ha = np.maximum(Y_mt_per_ha, 0.1)  # Ensure positive

# Generate harvest areas (uniform)
H_ha = np.random.uniform(low=100, high=1000, size=n_cells)

# Compute production
P_mt = Y_mt_per_ha * H_ha

# Build envelope
result = build_envelope(P_mt, H_ha, interpolate=True)

# Access results
lower = result['lower']
upper = result['upper']
summary = result['summary']
interp = result['interpolated']

print(f"Valid cells: {summary['n_valid']}")
print(f"Total production: {summary['total_P_mt']:.2f} mt")
print(f"Total harvest: {summary['total_H_km2']:.2f} km²")
print(f"Yield range: {summary['yield_min']:.2f} to {summary['yield_max']:.2f} mt/ha")

# Verify mathematical properties
assert summary['qa_results']['dominance_holds']
assert summary['qa_results']['lower_P_monotonic']
assert summary['qa_results']['upper_P_monotonic']
print("✓ All mathematical properties verified")
```


### Example 2: Demonstrating Dominance Property

```python
import numpy as np
import matplotlib.pyplot as plt
from agririchter.analysis.envelope_builder import build_envelope

# Create simple example with 5 cells
P_mt = np.array([100, 200, 300, 400, 500])
H_ha = np.array([100, 100, 100, 100, 100])
Y_mt_per_ha = P_mt / H_ha  # [1, 2, 3, 4, 5] mt/ha

result = build_envelope(P_mt, H_ha, interpolate=True)

# Show discrete sequences
lower = result['lower']
upper = result['upper']

print("Lower envelope (sorted by yield ascending):")
print(f"  Yields: {lower['Y_sorted']}")
print(f"  Cum H:  {lower['cum_H_km2']}")
print(f"  Cum P:  {lower['cum_P_mt']}")

print("\nUpper envelope (sorted by yield descending):")
print(f"  Yields: {upper['Y_sorted']}")
print(f"  Cum H:  {upper['cum_H_km2']}")
print(f"  Cum P:  {upper['cum_P_mt']}")

# Verify dominance at each cumulative harvest level
print("\nDominance verification:")
for i in range(len(lower['cum_H_km2'])):
    H = lower['cum_H_km2'][i]
    P_low = lower['cum_P_mt'][i]
    
    # Find corresponding upper envelope production
    k_up = np.searchsorted(upper['cum_H_km2'], H)
    if k_up >= len(upper['cum_P_mt']):
        k_up = len(upper['cum_P_mt']) - 1
    P_up = upper['cum_P_mt'][k_up]
    
    print(f"  H={H:.2f} km²: P_low={P_low:.0f} mt, P_up={P_up:.0f} mt, "
          f"dominance={'✓' if P_up >= P_low else '✗'}")

# Plot
interp = result['interpolated']
plt.figure(figsize=(10, 6))
plt.fill_between(interp['Hq_km2'], 
                 interp['P_low_interp'], 
                 interp['P_up_interp'],
                 alpha=0.3, label='Envelope band')
plt.plot(interp['Hq_km2'], interp['P_low_interp'], 'b-', label='Lower envelope')
plt.plot(interp['Hq_km2'], interp['P_up_interp'], 'r-', label='Upper envelope')
plt.scatter(lower['cum_H_km2'], lower['cum_P_mt'], c='blue', marker='o', s=50, 
            label='Lower (discrete)', zorder=5)
plt.scatter(upper['cum_H_km2'], upper['cum_P_mt'], c='red', marker='s', s=50,
            label='Upper (discrete)', zorder=5)
plt.xlabel('Cumulative Harvest Area (km²)')
plt.ylabel('Cumulative Production (mt)')
plt.title('Production Envelope: Discrete and Interpolated')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('envelope_example.png', dpi=150)
print("\n✓ Plot saved to envelope_example.png")
```

### Example 3: Handling Real Data with Validation

```python
import pandas as pd
from agririchter.analysis.envelope_builder import build_envelope

# Load real data (example: wheat production)
data = pd.read_csv('wheat_production.csv')

# Extract columns
P_mt = data['production_mt'].values
H_ha = data['harvest_area_ha'].values
Y_mt_per_ha = data['yield_mt_per_ha'].values  # Optional

# Build envelope with validation
result = build_envelope(
    P_mt, 
    H_ha, 
    Y_mt_per_ha=Y_mt_per_ha,
    tol=0.05,  # 5% tolerance for yield consistency
    interpolate=True,
    n_points=200
)

# Check summary
summary = result['summary']
print(f"Dataset: {summary['n_total']} cells")
print(f"Valid: {summary['n_valid']} cells ({100*summary['n_valid']/summary['n_total']:.1f}%)")
print(f"Dropped: {summary['n_dropped']} cells")

# Check dropped cells by reason
for reason, count in summary['dropped_counts'].items():
    if count > 0:
        print(f"  - {reason}: {count}")

# Check yield validation
yield_val = summary['yield_validation']
if not yield_val['computed']:
    print(f"\nYield consistency: {yield_val['mismatch_count']} mismatches "
          f"({yield_val['mismatch_pct']:.2f}%)")
    if yield_val.get('recomputed', False):
        print("  ⚠ Yield recomputed from P/H (P is authoritative)")

# Check QA results
qa = summary['qa_results']
print("\nQA Results:")
for check, passed in qa.items():
    if isinstance(passed, bool):
        status = '✓' if passed else '✗'
        print(f"  {status} {check}")

# Check warnings
if 'warnings' in summary:
    print("\nWarnings:")
    for warning in summary['warnings']:
        print(f"  ⚠ {warning}")

# Export results
import json
with open('envelope_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n✓ Summary exported to envelope_summary.json")
```

### Example 4: Comparing Discrete vs Interpolated

```python
import numpy as np
import matplotlib.pyplot as plt
from agririchter.analysis.envelope_builder import build_envelope

# Generate data
np.random.seed(42)
n_cells = 100
Y = np.random.uniform(1, 10, n_cells)
H_ha = np.random.uniform(100, 1000, n_cells)
P_mt = Y * H_ha

# Build envelope with and without interpolation
result_discrete = build_envelope(P_mt, H_ha, interpolate=False)
result_interp = build_envelope(P_mt, H_ha, interpolate=True, n_points=200)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Discrete only
ax = axes[0]
lower = result_discrete['lower']
upper = result_discrete['upper']
ax.plot(lower['cum_H_km2'], lower['cum_P_mt'], 'b.-', label='Lower (discrete)', markersize=3)
ax.plot(upper['cum_H_km2'], upper['cum_P_mt'], 'r.-', label='Upper (discrete)', markersize=3)
ax.set_xlabel('Cumulative Harvest Area (km²)')
ax.set_ylabel('Cumulative Production (mt)')
ax.set_title('Discrete Envelope (100 cells)')
ax.legend()
ax.grid(True, alpha=0.3)

# Interpolated with discrete overlay
ax = axes[1]
interp = result_interp['interpolated']
ax.fill_between(interp['Hq_km2'], 
                interp['P_low_interp'], 
                interp['P_up_interp'],
                alpha=0.2, color='gray', label='Envelope band')
ax.plot(interp['Hq_km2'], interp['P_low_interp'], 'b-', label='Lower (interpolated)', linewidth=2)
ax.plot(interp['Hq_km2'], interp['P_up_interp'], 'r-', label='Upper (interpolated)', linewidth=2)
# Overlay discrete points (every 5th to avoid clutter)
ax.scatter(lower['cum_H_km2'][::5], lower['cum_P_mt'][::5], 
          c='blue', marker='o', s=20, alpha=0.5, label='Lower (discrete)')
ax.scatter(upper['cum_H_km2'][::5], upper['cum_P_mt'][::5],
          c='red', marker='s', s=20, alpha=0.5, label='Upper (discrete)')
ax.set_xlabel('Cumulative Harvest Area (km²)')
ax.set_ylabel('Cumulative Production (mt)')
ax.set_title('Interpolated Envelope (200 points) with Discrete Overlay')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('discrete_vs_interpolated.png', dpi=150)
print("✓ Comparison plot saved to discrete_vs_interpolated.png")
```

---

## Mathematical Notation Reference

| Symbol | Meaning | Units |
|--------|---------|-------|
| N | Number of grid cells | - |
| i | Cell index (0 to N-1) | - |
| P_i | Production of cell i | metric tons (mt) |
| H_i | Harvest area of cell i | hectares (ha) or km² |
| Y_i | Yield of cell i | mt/ha |
| k | Cumulative index (0 to N-1) | - |
| cum_H[k] | Cumulative harvest area up to cell k | km² |
| cum_P[k] | Cumulative production up to cell k | mt |
| H_target | Target harvest level for dominance test | km² |
| Hq_km2 | Query grid for interpolation | km² |
| M | Number of points in query grid | - |
| P_low_interp | Interpolated lower envelope | mt |
| P_up_interp | Interpolated upper envelope | mt |

---

## References

### Related Documentation

- [API Reference](API_REFERENCE.md) - Function signatures and parameters
- [User Guide](USER_GUIDE.md) - Practical usage examples
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Data Requirements](DATA_REQUIREMENTS.md) - Input data specifications

### Mathematical Background

- **Cumulative Distribution Functions**: The envelope curves are related to empirical CDFs of the yield distribution
- **Order Statistics**: Sorting by yield is equivalent to using order statistics
- **Monotonic Interpolation**: Ensures physical constraints are preserved during continuous mapping

### Implementation Notes

- Uses NumPy for efficient array operations
- Employs `np.interp` for simple, reliable monotonic interpolation
- All assertions use `assert` statements for fail-fast behavior
- Logging uses Python's `logging` module for structured output

---

## Appendix: Test Progression

The implementation follows a strict test-first approach:

### Level 1: Synthetic Data Tests (MUST PASS FIRST)
- Create 1000 cells with known properties
- Test cumsum totals match sum of inputs
- Test monotonicity of cumulative sequences
- Test dominance property at all points
- Test interpolation preserves monotonicity

### Level 2: Unit Tests (MUST PASS SECOND)
- Test input validation (NaN, zero, negative handling)
- Test unit conversion (single conversion, conservation)
- Test yield consistency checking
- Test sorting by yield with known data
- Test cumulative sum calculation with hand-calculated examples
- Test interpolation with monotonic input

### Level 3: Integration Tests (MUST PASS THIRD)
- Test with wheat, rice, allgrain datasets
- Verify outputs match expected format
- Check all QA assertions pass
- Compare with MATLAB reference implementation

### Level 4: Parallel Validation (MUST BE SIMILAR)
- Run both old and new implementations side-by-side
- Compare outputs on test data
- Verify new implementation has better validation

### Level 5: Production Deployment (ONLY AFTER ALL TESTS PASS)
- Replace old implementation
- Monitor for issues
- Maintain backward compatibility

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-09  
**Author**: AgRichter Development Team

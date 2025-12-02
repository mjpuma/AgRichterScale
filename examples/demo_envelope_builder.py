"""
Demonstration of the envelope_builder module.

This script shows how to use the build_envelope function with synthetic data.
"""

import numpy as np
from pathlib import Path
from agririchter.analysis.envelope_builder import build_envelope, export_envelope_results


def demo_simple_case():
    """Demonstrate with a simple hand-calculable case."""
    print("=" * 70)
    print("DEMO: Simple Case (5 cells)")
    print("=" * 70)
    
    # Create simple data: 5 cells with yields 1-5 mt/ha
    Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    P_mt = Y_mt_per_ha * H_ha  # [100, 200, 300, 400, 500]
    
    print("\nInput data:")
    print(f"  Yields (mt/ha): {Y_mt_per_ha}")
    print(f"  Harvest areas (ha): {H_ha}")
    print(f"  Production (mt): {P_mt}")
    print(f"  Total production: {np.sum(P_mt)} mt")
    print(f"  Total harvest: {np.sum(H_ha)} ha = {np.sum(H_ha)*0.01} km²")
    
    # Build envelope
    result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
    
    print("\nLower envelope (worst case - low yields first):")
    print(f"  Cumulative harvest (km²): {result['lower']['cum_H_km2']}")
    print(f"  Cumulative production (mt): {result['lower']['cum_P_mt']}")
    print(f"  Sorted yields: {result['lower']['Y_sorted']}")
    
    print("\nUpper envelope (best case - high yields first):")
    print(f"  Cumulative harvest (km²): {result['upper']['cum_H_km2']}")
    print(f"  Cumulative production (mt): {result['upper']['cum_P_mt']}")
    print(f"  Sorted yields: {result['upper']['Y_sorted']}")
    
    print("\nSummary:")
    print(f"  Valid cells: {result['summary']['n_valid']}")
    print(f"  Dropped cells: {result['summary']['n_dropped']}")
    print(f"  Total P: {result['summary']['totals']['P_mt']:.2f} mt")
    print(f"  Total H: {result['summary']['totals']['H_km2']:.2f} km²")
    print(f"  Yield range: {result['summary']['yield_stats']['min']:.2f} - "
          f"{result['summary']['yield_stats']['max']:.2f} mt/ha")
    
    print("\nQA Results:")
    qa = result['summary']['discrete_qa_results']
    all_passed = all([
        qa['lower_H_monotonic'], qa['lower_P_monotonic'],
        qa['upper_H_monotonic'], qa['upper_P_monotonic'],
        qa['dominance_holds']
    ])
    print(f"  All checks passed: {all_passed}")
    print(f"  Dominance violations: {qa['dominance_violations']}")


def demo_without_yield():
    """Demonstrate computing yield from P and H."""
    print("\n" + "=" * 70)
    print("DEMO: Computing Yield from Production and Harvest Area")
    print("=" * 70)
    
    # Provide only P and H, let the function compute Y
    P_mt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    
    print("\nInput data (no yield provided):")
    print(f"  Production (mt): {P_mt}")
    print(f"  Harvest areas (ha): {H_ha}")
    
    result = build_envelope(P_mt, H_ha)
    
    print("\nComputed yields (mt/ha):")
    print(f"  Lower envelope: {result['lower']['Y_sorted']}")
    print(f"  Upper envelope: {result['upper']['Y_sorted']}")
    
    print(f"\nYield was computed: {result['summary']['yield_validation']['computed']}")


def demo_with_invalid_data():
    """Demonstrate handling of NaNs and zeros."""
    print("\n" + "=" * 70)
    print("DEMO: Handling Invalid Data (NaNs and zeros)")
    print("=" * 70)
    
    # Include some invalid data
    P_mt = np.array([100.0, np.nan, 300.0, 0.0, 500.0, -10.0])
    H_ha = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    Y_mt_per_ha = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    print("\nInput data (with invalid values):")
    print(f"  Production (mt): {P_mt}")
    print(f"  Harvest areas (ha): {H_ha}")
    print(f"  Yields (mt/ha): {Y_mt_per_ha}")
    
    result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
    
    print("\nValidation results:")
    print(f"  Total cells: {result['summary']['n_total']}")
    print(f"  Valid cells: {result['summary']['n_valid']}")
    print(f"  Dropped cells: {result['summary']['n_dropped']}")
    print("\nDropped by reason:")
    for reason, count in result['summary']['dropped_counts'].items():
        if count > 0:
            print(f"    {reason}: {count}")
    
    print(f"\nRemaining valid production: {result['summary']['totals']['P_mt']:.2f} mt")


def demo_large_dataset():
    """Demonstrate with a larger synthetic dataset."""
    print("\n" + "=" * 70)
    print("DEMO: Large Dataset (1000 cells)")
    print("=" * 70)
    
    np.random.seed(42)
    n_cells = 1000
    
    # Generate random data
    Y_mt_per_ha = np.random.uniform(1.0, 10.0, n_cells)
    H_ha = np.random.uniform(100.0, 1000.0, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    print(f"\nGenerated {n_cells} cells with random yields and harvest areas")
    print(f"  Yield range: {np.min(Y_mt_per_ha):.2f} - {np.max(Y_mt_per_ha):.2f} mt/ha")
    print(f"  Harvest range: {np.min(H_ha):.2f} - {np.max(H_ha):.2f} ha")
    print(f"  Total production: {np.sum(P_mt):.2f} mt")
    
    result = build_envelope(P_mt, H_ha, Y_mt_per_ha)
    
    print("\nEnvelope statistics:")
    print(f"  Lower envelope points: {len(result['lower']['cum_P_mt'])}")
    print(f"  Upper envelope points: {len(result['upper']['cum_P_mt'])}")
    print(f"  Final cumulative P (lower): {result['lower']['cum_P_mt'][-1]:.2f} mt")
    print(f"  Final cumulative P (upper): {result['upper']['cum_P_mt'][-1]:.2f} mt")
    
    # Check dominance at a few points
    print("\nDominance check (upper >= lower at sample points):")
    for i in [0, 249, 499, 749, 999]:
        H_target = result['lower']['cum_H_km2'][i]
        P_low = result['lower']['cum_P_mt'][i]
        
        # Find corresponding upper envelope point
        idx_up = np.searchsorted(result['upper']['cum_H_km2'], H_target)
        if idx_up >= len(result['upper']['cum_P_mt']):
            idx_up = len(result['upper']['cum_P_mt']) - 1
        P_up = result['upper']['cum_P_mt'][idx_up]
        
        print(f"  At H={H_target:.2f} km²: lower={P_low:.2f} mt, upper={P_up:.2f} mt, "
              f"diff={P_up-P_low:.2f} mt")


def demo_interpolation():
    """Demonstrate interpolation onto unified query grid."""
    print("\n" + "=" * 70)
    print("DEMO: Interpolation onto Unified Query Grid")
    print("=" * 70)
    
    np.random.seed(42)
    n_cells = 1000
    
    # Generate random data
    Y_mt_per_ha = np.random.uniform(1.0, 10.0, n_cells)
    H_ha = np.random.uniform(100.0, 1000.0, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    print(f"\nGenerated {n_cells} cells")
    print(f"  Total production: {np.sum(P_mt):.2f} mt")
    print(f"  Total harvest: {np.sum(H_ha):.2f} ha = {np.sum(H_ha)*0.01:.2f} km²")
    
    # Build envelope WITH interpolation
    result = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=True, n_points=300)
    
    print("\nDiscrete envelope (original data):")
    print(f"  Lower envelope: {len(result['lower']['cum_P_mt'])} discrete points")
    print(f"  Upper envelope: {len(result['upper']['cum_P_mt'])} discrete points")
    
    print("\nInterpolated envelope (unified query grid):")
    interp = result['interpolated']
    print(f"  Query grid points: {len(interp['Hq_km2'])}")
    print(f"  Query grid range: [{interp['Hq_km2'][0]:.2f}, {interp['Hq_km2'][-1]:.2f}] km²")
    print(f"  Lower envelope range: [{interp['P_low_interp'][0]:.2f}, {interp['P_low_interp'][-1]:.2f}] mt")
    print(f"  Upper envelope range: [{interp['P_up_interp'][0]:.2f}, {interp['P_up_interp'][-1]:.2f}] mt")
    
    print("\nInterpolation quality:")
    print(f"  Clipping applied: {interp['clipping_applied']}")
    if interp['clipping_applied']:
        print(f"  Points clipped: {interp['n_clipped']}")
        print(f"  Max deficit corrected: {interp['max_deficit']:.2e} mt")
    
    # Verify monotonicity
    P_low_diffs = np.diff(interp['P_low_interp'])
    P_up_diffs = np.diff(interp['P_up_interp'])
    print(f"  Lower envelope monotonic: {np.all(P_low_diffs >= 0)}")
    print(f"  Upper envelope monotonic: {np.all(P_up_diffs >= 0)}")
    
    # Verify dominance
    dominance_holds = np.all(interp['P_up_interp'] >= interp['P_low_interp'])
    print(f"  Dominance constraint satisfied: {dominance_holds}")
    
    # Sample a few points
    print("\nSample interpolated values:")
    sample_indices = [0, 74, 149, 224, 299]
    for idx in sample_indices:
        H = interp['Hq_km2'][idx]
        P_low = interp['P_low_interp'][idx]
        P_up = interp['P_up_interp'][idx]
        print(f"  H={H:8.2f} km²: P_low={P_low:10.2f} mt, P_up={P_up:10.2f} mt, "
              f"spread={P_up-P_low:8.2f} mt")


def demo_interpolation_spacing():
    """Demonstrate log vs linear spacing in query grid."""
    print("\n" + "=" * 70)
    print("DEMO: Query Grid Spacing (Linear vs Log)")
    print("=" * 70)
    
    # Case 1: Small span (linear spacing)
    print("\nCase 1: Small span (< 3 orders of magnitude) -> Linear spacing")
    Y_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    H_small = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    P_small = Y_small * H_small
    
    result_small = build_envelope(P_small, H_small, Y_small, interpolate=True, n_points=200)
    Hq_small = result_small['interpolated']['Hq_km2']
    span_small = Hq_small[-1] / Hq_small[0]
    
    print(f"  Query grid span: {span_small:.2f}x")
    print(f"  Query grid range: [{Hq_small[0]:.4f}, {Hq_small[-1]:.4f}] km²")
    print(f"  Spacing type: Linear (uniform increments)")
    
    # Case 2: Large span (log spacing)
    print("\nCase 2: Large span (>= 3 orders of magnitude) -> Log spacing")
    Y_large = np.logspace(-3, 1, 100)  # 0.001 to 10 (4 orders of magnitude)
    H_large = np.full(100, 100.0)
    P_large = Y_large * H_large
    
    result_large = build_envelope(P_large, H_large, Y_large, interpolate=True, n_points=200)
    Hq_large = result_large['interpolated']['Hq_km2']
    span_large = Hq_large[-1] / Hq_large[0]
    
    print(f"  Query grid span: {span_large:.2e}x")
    print(f"  Query grid range: [{Hq_large[0]:.4e}, {Hq_large[-1]:.4e}] km²")
    print(f"  Spacing type: Logarithmic (exponential increments)")
    
    # Show first few increments
    print(f"\n  First 5 increments (linear case): {np.diff(Hq_small[:6])}")
    print(f"  First 5 increments (log case): {np.diff(Hq_large[:6])}")


def demo_export():
    """Demonstrate exporting envelope results to files."""
    print("\n" + "=" * 70)
    print("DEMO: Exporting Envelope Results to Files")
    print("=" * 70)
    
    np.random.seed(42)
    n_cells = 500
    
    # Generate random data
    Y_mt_per_ha = np.random.uniform(1.0, 10.0, n_cells)
    H_ha = np.random.uniform(100.0, 1000.0, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    print(f"\nGenerated {n_cells} cells")
    print(f"  Total production: {np.sum(P_mt):.2f} mt")
    print(f"  Total harvest: {np.sum(H_ha):.2f} ha")
    
    # Build envelope with interpolation
    print("\nBuilding envelope with interpolation...")
    result = build_envelope(P_mt, H_ha, Y_mt_per_ha, interpolate=True, n_points=200)
    
    # Export results
    print("\nExporting results to files...")
    output_dir = Path("demo_output")
    output_paths = export_envelope_results(
        result=result,
        output_dir=output_dir,
        prefix="demo_envelope"
    )
    
    print(f"\nExported {len(output_paths)} files to {output_dir}:")
    for file_type, file_path in output_paths.items():
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  ✓ {file_type:20s}: {file_path.name:35s} ({file_size:6.1f} KB)")
    
    print("\nFile descriptions:")
    print("  - lower_csv/upper_csv: Interpolated curves for plotting (200 points)")
    print("  - lower_discrete_csv/upper_discrete_csv: Discrete cumulative sums (500 points)")
    print("  - summary_json: Complete QA report with all validation results")
    print("  - plot_png: Visualization showing both discrete and interpolated data")
    
    print(f"\nYou can now:")
    print(f"  - Load CSV files into Excel/Python for analysis")
    print(f"  - View plot: open {output_paths['plot_png']}")
    print(f"  - Inspect QA report: {output_paths['summary_json']}")


if __name__ == "__main__":
    demo_simple_case()
    demo_without_yield()
    demo_with_invalid_data()
    demo_large_dataset()
    demo_interpolation()
    demo_interpolation_spacing()
    demo_export()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)

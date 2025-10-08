#!/usr/bin/env python3
"""
Test script for MATLAB validation functionality.
Creates mock MATLAB data to test the comparison pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from validate_matlab_comparison import MATLABValidator


def create_mock_matlab_data():
    """Create mock MATLAB reference data for testing."""
    
    # Event names (21 events)
    event_names = [
        'GreatFamine', 'Laki1783', 'NoSummer', 'Drought18761878',
        'SovietFamine1921', 'ChineseFamine1960', 'DustBowl', 'SahelDrought2010',
        'MillenniumDrought', 'NorthKorea1990s', 'Solomon', 'Vanuatu',
        'EastTimor', 'Haiti', 'SierraLeone', 'Liberia',
        'Yemen', 'Ethiopia', 'Laos', 'Bangladesh', 'Syria'
    ]
    
    # Create mock data with realistic values
    np.random.seed(42)
    
    mock_data = {
        'event_name': event_names,
        'harvest_area_loss_ha': np.random.uniform(1e5, 1e7, 21),
        'production_loss_kcal': np.random.uniform(1e12, 1e15, 21),
    }
    
    # Calculate magnitudes
    mock_data['magnitude'] = np.log10(mock_data['harvest_area_loss_ha'] * 0.01)
    
    df = pd.DataFrame(mock_data)
    
    return df


def create_mock_python_data(matlab_df, add_differences=True):
    """
    Create mock Python data based on MATLAB data.
    
    Args:
        matlab_df: MATLAB reference DataFrame
        add_differences: If True, add small differences to simulate real comparison
    """
    python_df = matlab_df.copy()
    
    if add_differences:
        # Add small random differences (within 5%)
        python_df['harvest_area_loss_ha'] *= np.random.uniform(0.97, 1.03, len(python_df))
        python_df['production_loss_kcal'] *= np.random.uniform(0.97, 1.03, len(python_df))
        
        # Recalculate magnitudes
        python_df['magnitude'] = np.log10(python_df['harvest_area_loss_ha'] * 0.01)
        
        # Add one event with larger difference to test threshold detection
        python_df.loc[5, 'harvest_area_loss_ha'] *= 1.10  # 10% difference
        python_df.loc[5, 'magnitude'] = np.log10(python_df.loc[5, 'harvest_area_loss_ha'] * 0.01)
    
    return python_df


def test_validation_workflow():
    """Test the complete validation workflow."""
    
    print("=" * 80)
    print("TESTING MATLAB VALIDATION WORKFLOW")
    print("=" * 80)
    
    # Create test directories
    test_matlab_dir = Path('test_matlab_outputs')
    test_python_dir = Path('test_python_outputs')
    test_comparison_dir = Path('test_comparison_reports')
    
    test_matlab_dir.mkdir(exist_ok=True)
    test_python_dir.mkdir(exist_ok=True)
    test_comparison_dir.mkdir(exist_ok=True)
    
    # Create mock data for wheat
    print("\n1. Creating mock MATLAB reference data...")
    matlab_wheat = create_mock_matlab_data()
    matlab_file = test_matlab_dir / 'matlab_events_wheat_spam2010.csv'
    matlab_wheat.to_csv(matlab_file, index=False)
    print(f"   Created: {matlab_file}")
    
    # Create mock Python data
    print("\n2. Creating mock Python results...")
    python_wheat = create_mock_python_data(matlab_wheat, add_differences=True)
    
    # Create Python output directory structure
    wheat_dir = test_python_dir / 'wheat'
    wheat_dir.mkdir(exist_ok=True)
    python_file = wheat_dir / 'python_events_wheat_spam2020.csv'
    python_wheat.to_csv(python_file, index=False)
    print(f"   Created: {python_file}")
    
    # Initialize validator
    print("\n3. Initializing validator...")
    validator = MATLABValidator(
        matlab_output_dir=str(test_matlab_dir),
        python_output_dir=str(test_python_dir),
        comparison_output_dir=str(test_comparison_dir)
    )
    
    # Load data
    print("\n4. Loading MATLAB and Python results...")
    validator.matlab_results = {'wheat': matlab_wheat}
    validator.python_results = {'wheat': python_wheat}
    
    # Compare results
    print("\n5. Comparing results...")
    stats = validator.compare_results('wheat')
    
    if stats:
        print(f"   ✓ Comparison complete")
        print(f"   - Mean harvest area difference: {stats['harvest_area_mean_diff_pct']:.2f}%")
        print(f"   - Mean production difference: {stats['production_mean_diff_pct']:.2f}%")
        print(f"   - Events exceeding threshold: {stats['harvest_area_issues_count']}")
    
    # Investigate differences
    print("\n6. Investigating differences...")
    findings = validator.investigate_differences(stats)
    
    if findings:
        print(f"   ✓ Investigation complete")
        print(f"   - Systematic differences: {len(findings['systematic_differences'])}")
        print(f"   - Event-specific issues: {len(findings['event_specific_issues'])}")
        print(f"   - Recommendations: {len(findings['recommendations'])}")
    
    # Create visualizations
    print("\n7. Creating comparison visualizations...")
    validator.create_comparison_visualizations(stats)
    print(f"   ✓ Visualization saved")
    
    # Generate report
    print("\n8. Generating comparison report...")
    report_path = validator.generate_comparison_report(
        {'wheat': stats},
        {'wheat': findings}
    )
    print(f"   ✓ Report saved: {report_path}")
    
    # Evaluate thresholds
    print("\n9. Evaluating validation thresholds...")
    validator.update_validation_thresholds({'wheat': stats})
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nTest outputs saved to:")
    print(f"  - MATLAB data: {test_matlab_dir}")
    print(f"  - Python data: {test_python_dir}")
    print(f"  - Comparison reports: {test_comparison_dir}")
    print("\nYou can review the generated report and visualizations.")
    
    return True


if __name__ == '__main__':
    try:
        success = test_validation_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

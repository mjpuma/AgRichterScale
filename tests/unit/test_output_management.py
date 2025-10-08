#!/usr/bin/env python3
"""Test script for output management system."""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the agririchter package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agririchter.output.manager import OutputManager
from agririchter.output.organizer import FileOrganizer
from agririchter.output.exporter import DataExporter, FigureExporter

def test_file_organizer():
    """Test file organizer functionality."""
    print("Testing File Organizer...")
    
    try:
        # Create organizer
        organizer = FileOrganizer("test_outputs/organizer_test")
        print("  ✓ File organizer created")
        
        # Test path generation
        figure_path = organizer.get_figure_path('wheat', 'agririchter_scale', 'png')
        print(f"  ✓ Figure path: {figure_path}")
        
        data_path = organizer.get_data_path('wheat', 'event_losses', 'csv')
        print(f"  ✓ Data path: {data_path}")
        
        report_path = organizer.get_report_path('wheat', 'analysis')
        print(f"  ✓ Report path: {report_path}")
        
        # Test directory info
        info = organizer.get_directory_info()
        print(f"  ✓ Directory info retrieved: {len(info)} directories")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing file organizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_exporter():
    """Test data exporter functionality."""
    print("\nTesting Data Exporter...")
    
    try:
        # Create organizer and exporter
        organizer = FileOrganizer("test_outputs/exporter_test")
        exporter = DataExporter(organizer)
        print("  ✓ Data exporter created")
        
        # Test event losses export
        sample_events = pd.DataFrame({
            'event_name': ['Test Event 1', 'Test Event 2', 'Test Event 3'],
            'harvest_area_km2': [100000, 200000, 150000],
            'production_loss_kcal': [1e14, 2e14, 1.5e14]
        })
        
        csv_path = exporter.export_event_losses(sample_events, 'wheat', 'csv')
        print(f"  ✓ Event losses exported to CSV: {csv_path}")
        
        # Test envelope data export
        sample_envelope = {
            'disrupted_areas': np.array([100, 1000, 10000, 100000]),
            'upper_bound': np.array([1e12, 1e13, 1e14, 1e15]),
            'lower_bound': np.array([5e11, 5e12, 5e13, 5e14])
        }
        
        envelope_path = exporter.export_envelope_data(sample_envelope, 'wheat', 'csv')
        print(f"  ✓ Envelope data exported: {envelope_path}")
        
        # Test threshold data export
        thresholds = {'T1': 1e14, 'T2': 5e14, 'T3': 1.5e15, 'T4': 3e15}
        sur_thresholds = {2: 0.25, 3: 0.20, 4: 0.15, 5: 0.10}
        
        threshold_path = exporter.export_threshold_data(thresholds, sur_thresholds, 'wheat', 'csv')
        print(f"  ✓ Threshold data exported: {threshold_path}")
        
        # Test USDA data export
        sample_usda = {
            'wheat': pd.DataFrame({
                'Year': [2018, 2019, 2020],
                'Production': [731000, 765000, 760000],
                'Consumption': [740000, 750000, 755000],
                'EndingStocks': [180000, 195000, 200000],
                'BeginningStocks': [189000, 180000, 195000]
            })
        }
        
        usda_path = exporter.export_usda_data(sample_usda, 'wheat', 'csv')
        print(f"  ✓ USDA data exported: {usda_path}")
        
        # Verify files exist
        for path in [csv_path, envelope_path, threshold_path, usda_path]:
            if path.exists():
                print(f"    ✓ File exists: {path.name}")
            else:
                print(f"    ✗ File missing: {path.name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing data exporter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_figure_exporter():
    """Test figure exporter functionality."""
    print("\nTesting Figure Exporter...")
    
    try:
        # Create organizer and exporter
        organizer = FileOrganizer("test_outputs/figure_test")
        exporter = FigureExporter(organizer)
        print("  ✓ Figure exporter created")
        
        # Create sample figure
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Test Figure')
        ax.grid(True, alpha=0.3)
        
        # Export figure
        exported_paths = exporter.export_figure(fig, 'wheat', 'agririchter_scale', ['png', 'svg'])
        print(f"  ✓ Figure exported to {len(exported_paths)} formats")
        
        # Verify files exist
        for path in exported_paths:
            if path.exists():
                print(f"    ✓ File exists: {path.name}")
            else:
                print(f"    ✗ File missing: {path.name}")
                return False
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing figure exporter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_manager():
    """Test complete output manager functionality."""
    print("\nTesting Output Manager...")
    
    try:
        # Create output manager
        manager = OutputManager("test_outputs/manager_test")
        print("  ✓ Output manager created")
        
        # Create comprehensive sample analysis data
        sample_analysis = {
            'events_data': pd.DataFrame({
                'event_name': ['Dust Bowl', 'Soviet Famine 1921', 'Great Famine'],
                'harvest_area_km2': [100000, 200000, 50000],
                'production_loss_kcal': [5e14, 8e14, 3e14]
            }),
            'envelope_data': {
                'disrupted_areas': np.logspace(2, 6, 20),  # 100 to 1M km²
                'upper_bound': np.logspace(12, 16, 20),    # 1e12 to 1e16 kcal
                'lower_bound': np.logspace(11, 15, 20)     # 1e11 to 1e15 kcal
            },
            'thresholds': {
                'T1': 1.00e14,
                'T2': 5.75e14,
                'T3': 1.86e15,
                'T4': 3.00e15
            },
            'sur_thresholds': {
                2: 0.259,
                3: 0.240,
                4: 0.230,
                5: 0.220
            },
            'usda_data': {
                'wheat': pd.DataFrame({
                    'Year': list(range(2015, 2021)),
                    'Production': [735000, 762000, 760000, 731000, 765000, 760000],
                    'Consumption': [740000, 745000, 750000, 740000, 750000, 755000],
                    'EndingStocks': [180000, 197000, 207000, 198000, 213000, 218000],
                    'BeginningStocks': [189000, 180000, 197000, 207000, 198000, 213000]
                })
            }
        }
        
        # Create sample figures
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.scatter([3, 4, 5], [1e14, 5e14, 2e15], c='red', s=100)
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Production Loss (kcal)')
        ax1.set_title('AgriRichter Scale - Wheat')
        ax1.set_yscale('log')
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        x = np.linspace(2, 6, 50)
        y_upper = 1e12 * (10**x)**1.2
        y_lower = 5e11 * (10**x)**1.1
        ax2.fill_between(x, y_lower, y_upper, alpha=0.3, color='gray')
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Production Loss (kcal)')
        ax2.set_title('H-P Envelope - Wheat')
        ax2.set_yscale('log')
        
        sample_analysis['figures'] = {
            'agririchter_scale': fig1,
            'hp_envelope': fig2
        }
        
        # Export complete analysis
        exported_paths = manager.export_complete_analysis(sample_analysis, 'wheat')
        print(f"  ✓ Complete analysis exported")
        print(f"    Data files: {len(exported_paths['data'])}")
        print(f"    Figure files: {len(exported_paths['figures'])}")
        print(f"    Report files: {len(exported_paths['reports'])}")
        
        # Get output summary
        summary = manager.get_output_summary()
        print(f"  ✓ Output summary generated")
        print(f"    Total files: {summary['total_files']}")
        print(f"    Total size: {summary['total_size_mb']:.2f} MB")
        
        # Verify key files exist
        key_files_exist = True
        for file_list in exported_paths.values():
            for file_path in file_list:
                # Handle both single paths and lists of paths
                if isinstance(file_path, list):
                    for path in file_path:
                        if not path.exists():
                            print(f"    ✗ Missing file: {path}")
                            key_files_exist = False
                else:
                    if not file_path.exists():
                        print(f"    ✗ Missing file: {file_path}")
                        key_files_exist = False
        
        if key_files_exist:
            print("  ✓ All exported files exist")
        
        plt.close('all')
        return key_files_exist
        
    except Exception as e:
        print(f"  ✗ Error testing output manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_organization():
    """Test file organization from existing directory."""
    print("\nTesting File Organization...")
    
    try:
        # Create some test files to organize
        test_source_dir = Path("test_outputs/source_files")
        test_source_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample files
        sample_files = [
            "wheat_production_map.png",
            "rice_agririchter_scale.svg",
            "event_losses_wheat.csv",
            "envelope_data_allgrain.xlsx",
            "analysis_report.txt"
        ]
        
        for filename in sample_files:
            file_path = test_source_dir / filename
            file_path.write_text(f"Sample content for {filename}")
        
        print(f"  ✓ Created {len(sample_files)} test files")
        
        # Organize files
        organizer = FileOrganizer("test_outputs/organized")
        organized_files = organizer.organize_existing_files(test_source_dir)
        
        print(f"  ✓ Organized files:")
        for file_type, file_list in organized_files.items():
            print(f"    {file_type}: {len(file_list)} files")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing file organization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Output Management System Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_file_organizer()
    success &= test_data_exporter()
    success &= test_figure_exporter()
    success &= test_output_manager()
    success &= test_file_organization()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
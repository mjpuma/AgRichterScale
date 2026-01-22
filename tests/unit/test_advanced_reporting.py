#!/usr/bin/env python3
"""Test script for advanced reporting system."""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add the agrichter package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agrichter.output.reporter import AnalysisReporter
from agrichter.output.organizer import FileOrganizer
from agrichter.output.manager import OutputManager

def test_analysis_reporter():
    """Test analysis reporter functionality."""
    print("Testing Analysis Reporter...")
    
    try:
        # Create organizer and reporter
        organizer = FileOrganizer("test_outputs/reporter_test")
        reporter = AnalysisReporter(organizer)
        print("  ✓ Analysis reporter created")
        
        # Create comprehensive sample analysis data
        sample_analysis = {
            'events_data': pd.DataFrame({
                'event_name': ['Dust Bowl', 'Soviet Famine 1921', 'Great Famine', 'Chinese Famine 1960', 'Bangladesh 1974'],
                'harvest_area_km2': [100000, 200000, 50000, 300000, 80000],
                'production_loss_kcal': [5e14, 8e14, 3e14, 1.2e15, 4e14]
            }),
            'envelope_data': {
                'disrupted_areas': np.logspace(2, 6, 25),
                'upper_bound': np.logspace(12, 16, 25),
                'lower_bound': np.logspace(11, 15, 25)
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
        
        # Generate comprehensive report
        comprehensive_path = reporter.generate_comprehensive_report(sample_analysis, 'wheat')
        print(f"  ✓ Comprehensive report generated: {comprehensive_path.name}")
        
        # Generate summary report
        summary_path = reporter.generate_summary_report(sample_analysis, 'wheat')
        print(f"  ✓ Summary report generated: {summary_path.name}")
        
        # Generate validation report
        validation_path = reporter.generate_validation_report(sample_analysis, 'wheat')
        print(f"  ✓ Validation report generated: {validation_path.name}")
        
        # Verify files exist and have content
        for path in [comprehensive_path, summary_path, validation_path]:
            if path.exists():
                file_size = path.stat().st_size
                print(f"    ✓ {path.name}: {file_size} bytes")
                
                # Check that file has substantial content
                if file_size > 1000:  # At least 1KB
                    print(f"      ✓ Substantial content")
                else:
                    print(f"      ⚠ Small file size")
            else:
                print(f"    ✗ Missing file: {path.name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing analysis reporter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_report_content_quality():
    """Test the quality and completeness of report content."""
    print("\nTesting Report Content Quality...")
    
    try:
        # Create organizer and reporter
        organizer = FileOrganizer("test_outputs/content_test")
        reporter = AnalysisReporter(organizer)
        
        # Create rich sample data
        sample_analysis = {
            'events_data': pd.DataFrame({
                'event_name': ['Event A', 'Event B', 'Event C', 'Event D'],
                'harvest_area_km2': [50000, 150000, 300000, 75000],
                'production_loss_kcal': [2e14, 8e14, 1.5e15, 3e14]
            }),
            'thresholds': {
                'T1': 1e14,
                'T2': 5e14,
                'T3': 1.5e15,
                'T4': 3e15
            },
            'sur_thresholds': {
                2: 0.25,
                3: 0.20,
                4: 0.15,
                5: 0.10
            },
            'usda_data': {
                'wheat': pd.DataFrame({
                    'Year': [2018, 2019, 2020],
                    'Production': [731000, 765000, 760000],
                    'Consumption': [740000, 750000, 755000],
                    'EndingStocks': [180000, 195000, 200000],
                    'BeginningStocks': [189000, 180000, 195000]
                })
            }
        }
        
        # Generate comprehensive report
        report_path = reporter.generate_comprehensive_report(sample_analysis, 'wheat')
        
        # Read and analyze report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            'EXECUTIVE SUMMARY',
            'METHODOLOGY',
            'THRESHOLD ANALYSIS',
            'HISTORICAL EVENTS ANALYSIS',
            'USDA PSD DATA ANALYSIS',
            'STATISTICAL SUMMARY',
            'DATA QUALITY ASSESSMENT',
            'CONCLUSIONS AND RECOMMENDATIONS'
        ]
        
        sections_found = 0
        for section in required_sections:
            if section in content:
                sections_found += 1
                print(f"    ✓ Found section: {section}")
            else:
                print(f"    ✗ Missing section: {section}")
        
        coverage = (sections_found / len(required_sections)) * 100
        print(f"  ✓ Section coverage: {coverage:.1f}% ({sections_found}/{len(required_sections)})")
        
        # Check for statistical content
        statistical_terms = ['Mean:', 'Median:', 'Standard deviation:', 'Range:', 'correlation']
        stats_found = sum(1 for term in statistical_terms if term in content)
        print(f"  ✓ Statistical content: {stats_found}/{len(statistical_terms)} terms found")
        
        # Check report length (should be substantial)
        word_count = len(content.split())
        print(f"  ✓ Report length: {word_count} words")
        
        if word_count > 500:
            print("    ✓ Substantial report content")
        else:
            print("    ⚠ Report seems short")
        
        return coverage >= 80 and word_count > 500
        
    except Exception as e:
        print(f"  ✗ Error testing report content quality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_output_manager():
    """Test enhanced output manager with advanced reporting."""
    print("\nTesting Enhanced Output Manager...")
    
    try:
        # Create enhanced output manager
        manager = OutputManager("test_outputs/enhanced_manager_test")
        print("  ✓ Enhanced output manager created")
        
        # Create comprehensive sample analysis data
        sample_analysis = {
            'events_data': pd.DataFrame({
                'event_name': ['Historical Event 1', 'Historical Event 2', 'Historical Event 3'],
                'harvest_area_km2': [120000, 180000, 250000],
                'production_loss_kcal': [6e14, 9e14, 1.3e15]
            }),
            'envelope_data': {
                'disrupted_areas': np.logspace(2, 5, 15),
                'upper_bound': np.logspace(12, 15, 15),
                'lower_bound': np.logspace(11, 14, 15)
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
                    'Year': list(range(2016, 2021)),
                    'Production': [728000, 760000, 731000, 765000, 760000],
                    'Consumption': [735000, 745000, 740000, 750000, 755000],
                    'EndingStocks': [175000, 190000, 181000, 196000, 201000],
                    'BeginningStocks': [182000, 175000, 190000, 181000, 196000]
                })
            }
        }
        
        # Export complete analysis with enhanced reporting
        exported_paths = manager.export_complete_analysis(sample_analysis, 'wheat')
        print(f"  ✓ Complete analysis exported with enhanced reporting")
        
        # Check that we now have multiple reports
        report_count = len(exported_paths['reports'])
        print(f"  ✓ Generated {report_count} reports")
        
        if report_count >= 3:
            print("    ✓ Multiple report types generated (comprehensive, summary, validation)")
        else:
            print(f"    ⚠ Expected at least 3 reports, got {report_count}")
        
        # Verify all report files exist
        all_reports_exist = True
        for report_path in exported_paths['reports']:
            if report_path.exists():
                file_size = report_path.stat().st_size
                print(f"    ✓ {report_path.name}: {file_size} bytes")
            else:
                print(f"    ✗ Missing report: {report_path.name}")
                all_reports_exist = False
        
        return all_reports_exist and report_count >= 3
        
    except Exception as e:
        print(f"  ✗ Error testing enhanced output manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_report_metadata_tracking():
    """Test metadata and provenance tracking in reports."""
    print("\nTesting Report Metadata Tracking...")
    
    try:
        # Create organizer and reporter
        organizer = FileOrganizer("test_outputs/metadata_test")
        reporter = AnalysisReporter(organizer)
        
        # Create sample data with metadata
        sample_analysis = {
            'events_data': pd.DataFrame({
                'event_name': ['Test Event'],
                'harvest_area_km2': [100000],
                'production_loss_kcal': [5e14]
            }),
            'thresholds': {
                'T1': 1e14,
                'T2': 5e14
            }
        }
        
        # Generate report
        report_path = reporter.generate_comprehensive_report(sample_analysis, 'wheat')
        
        # Read report and check for metadata
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Check for timestamp
        if 'Generated:' in content:
            print("  ✓ Timestamp included in report")
        else:
            print("  ✗ Missing timestamp in report")
            return False
        
        # Check for version information
        if 'Analysis Version:' in content:
            print("  ✓ Version information included")
        else:
            print("  ⚠ Version information not found")
        
        # Check for data quality assessment
        if 'DATA QUALITY ASSESSMENT' in content:
            print("  ✓ Data quality assessment included")
        else:
            print("  ✗ Missing data quality assessment")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing metadata tracking: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Advanced Reporting System Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_analysis_reporter()
    success &= test_report_content_quality()
    success &= test_enhanced_output_manager()
    success &= test_report_metadata_tracking()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    sys.exit(0 if success else 1)
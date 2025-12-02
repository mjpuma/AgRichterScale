#!/usr/bin/env python3
"""
Validate alignment between food_disruptions.csv and event Excel files.

Checks:
1. All events in Excel files have entries in food_disruptions.csv
2. All events in food_disruptions.csv exist in Excel files
3. Event names match exactly
4. Reports any mismatches or missing data
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("EVENT DATA VALIDATION")
    print("=" * 80)
    
    # Load food disruptions metadata
    disruptions_file = Path('ancillary/food_disruptions.csv')
    if not disruptions_file.exists():
        print(f"ERROR: {disruptions_file} not found!")
        sys.exit(1)
    
    disruptions_df = pd.read_csv(disruptions_file)
    disruption_events = set(disruptions_df['event_name'].values)
    
    print(f"\n✓ Loaded food_disruptions.csv: {len(disruption_events)} events")
    print(f"  Events: {sorted(disruption_events)}")
    
    # Load Excel event files
    country_file = Path('ancillary/DisruptionCountry.xls')
    state_file = Path('ancillary/DisruptionStateProvince.xls')
    
    if not country_file.exists():
        print(f"ERROR: {country_file} not found!")
        sys.exit(1)
    if not state_file.exists():
        print(f"ERROR: {state_file} not found!")
        sys.exit(1)
    
    country_sheets = pd.read_excel(country_file, sheet_name=None, engine='xlrd')
    state_sheets = pd.read_excel(state_file, sheet_name=None, engine='xlrd')
    
    country_events = set(country_sheets.keys())
    state_events = set(state_sheets.keys())
    excel_events = country_events | state_events
    
    print(f"\n✓ Loaded DisruptionCountry.xls: {len(country_events)} events")
    print(f"  Events: {sorted(country_events)}")
    
    print(f"\n✓ Loaded DisruptionStateProvince.xls: {len(state_events)} events")
    print(f"  Events: {sorted(state_events)}")
    
    print(f"\n✓ Total unique events in Excel files: {len(excel_events)}")
    
    # Check alignment
    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK")
    print("=" * 80)
    
    # Events in Excel but not in CSV
    missing_in_csv = excel_events - disruption_events
    if missing_in_csv:
        print(f"\n⚠ WARNING: {len(missing_in_csv)} events in Excel files but NOT in food_disruptions.csv:")
        for event in sorted(missing_in_csv):
            print(f"  - {event}")
    else:
        print("\n✓ All Excel events have entries in food_disruptions.csv")
    
    # Events in CSV but not in Excel
    missing_in_excel = disruption_events - excel_events
    if missing_in_excel:
        print(f"\n⚠ WARNING: {len(missing_in_excel)} events in food_disruptions.csv but NOT in Excel files:")
        for event in sorted(missing_in_excel):
            print(f"  - {event}")
    else:
        print("\n✓ All food_disruptions.csv events exist in Excel files")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not missing_in_csv and not missing_in_excel:
        print("\n✓✓✓ PERFECT ALIGNMENT! All events match between files.")
        
        # Show event type distribution
        print("\n" + "=" * 80)
        print("EVENT TYPE DISTRIBUTION")
        print("=" * 80)
        type_counts = disruptions_df['event_type'].value_counts()
        for event_type, count in type_counts.items():
            print(f"  {event_type}: {count} events")
        
        # Show primary cause distribution
        print("\n" + "=" * 80)
        print("PRIMARY CAUSE DISTRIBUTION")
        print("=" * 80)
        cause_counts = disruptions_df['primary_cause'].value_counts()
        for cause, count in cause_counts.items():
            print(f"  {cause}: {count} events")
        
        return 0
    else:
        print(f"\n⚠ MISALIGNMENT DETECTED:")
        print(f"  - Events missing from CSV: {len(missing_in_csv)}")
        print(f"  - Events missing from Excel: {len(missing_in_excel)}")
        print(f"\nPlease fix alignment before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

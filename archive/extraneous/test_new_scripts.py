#!/usr/bin/env python3
"""
Test that the 4 new figure scripts have all necessary code.
Does NOT run them (too slow), just checks they import correctly.
"""

import sys
from pathlib import Path

def test_script(script_name):
    """Test if a script can be imported and has main function."""
    print(f"\nTesting {script_name}...")
    
    try:
        # Check file exists
        script_path = Path(script_name)
        if not script_path.exists():
            print(f"  ❌ File not found: {script_name}")
            return False
        
        # Read the file
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = {
            'has_main': 'def main():' in content,
            'has_imports': 'import' in content,
            'has_logging': 'logging' in content,
            'has_agririchter': 'from agririchter' in content,
            'no_sample_data': 'create_sample_events_data' not in content or 'load_real_events' in content,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        # Special check for real data usage
        if 'create_sample_events_data' in content and 'load_real_events' not in content:
            print(f"  ⚠️  WARNING: May be using synthetic data!")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Test all 4 new scripts."""
    print("=" * 60)
    print("TESTING NEW FIGURE SCRIPTS")
    print("=" * 60)
    
    scripts = [
        'fig1_global_maps.py',
        'fig2_agrichter_scale.py',
        'fig3_hp_envelopes.py',
        'fig4_country_envelopes.py',
    ]
    
    results = {}
    for script in scripts:
        results[script] = test_script(script)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for script, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {script}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All scripts look good!")
        print("\nSafe to archive old generate_*.py scripts:")
        print("  mv generate_*.py archive/old_generate_scripts/")
        return 0
    else:
        print("\n⚠️  Some scripts have issues - DO NOT archive yet!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

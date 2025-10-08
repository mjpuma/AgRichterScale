#!/bin/bash
# Repository Cleanup Script
# Organizes development files and removes temporary outputs

echo "=== AgRichter Scale Repository Cleanup ==="
echo ""

# Create directories
echo "Creating organization directories..."
mkdir -p docs/development
mkdir -p examples
mkdir -p archive

# Move task implementation summaries
echo "Moving task summaries to docs/development/..."
mv TASK_*.md docs/development/ 2>/dev/null
mv FINAL_COMPREHENSIVE_TEST_RESULTS.md docs/development/ 2>/dev/null
mv NEWER_COUNTRIES_STATE_FILTERING_VERIFICATION.md docs/development/ 2>/dev/null
mv HP_ENVELOPE_AXIS_VERIFICATION.md docs/development/ 2>/dev/null
mv COUNTRY_STATE_FILTERING_FIX.md docs/development/ 2>/dev/null
mv SPATIAL_MAPPER_VERIFICATION.md docs/development/ 2>/dev/null
mv STATE_LEVEL_VERIFICATION_GUIDE.md docs/development/ 2>/dev/null
mv SPAM_2020_STRUCTURE.md docs/development/ 2>/dev/null

# Move demo scripts
echo "Moving demo scripts to examples/..."
mv demo_*.py examples/ 2>/dev/null
mv demo_*.sh examples/ 2>/dev/null

# Move test/verification scripts
echo "Moving test scripts to archive/..."
mv test_newer_countries_states.py archive/ 2>/dev/null
mv test_severity_classification.py archive/ 2>/dev/null
mv test_state_filtering.py archive/ 2>/dev/null
mv verify_*.py archive/ 2>/dev/null
mv debug_*.py archive/ 2>/dev/null

# Remove temporary output files
echo "Removing temporary output files..."
rm -f test_*.png test_*.eps test_*.svg test_*.jpg 2>/dev/null
rm -f test_agririchter_scale_*.* 2>/dev/null
rm -f test_hp_envelope_*.* 2>/dev/null
rm -f test_severity_*.* 2>/dev/null

# Remove old Python files
echo "Moving old Python files to archive/..."
mv AgriRichterv2.py archive/ 2>/dev/null
mv BalanceAnalysis.py archive/ 2>/dev/null
mv figplot_StocksSampler.py archive/ 2>/dev/null
mv Stocks_IC.py archive/ 2>/dev/null
mv stocks_ton_kcalconvert.py archive/ 2>/dev/null
mv StocksSampler.py archive/ 2>/dev/null

# Remove temporary text files
echo "Removing temporary documentation..."
rm -f MIGRATION_SUMMARY.txt 2>/dev/null

# Clean up test output directories
echo "Cleaning test output directories..."
rm -rf test_matlab_outputs test_python_outputs test_comparison_reports 2>/dev/null

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Summary:"
echo "  - Task summaries moved to: docs/development/"
echo "  - Demo scripts moved to: examples/"
echo "  - Test/old files moved to: archive/"
echo "  - Temporary outputs removed"
echo ""
echo "Review the changes with: git status"

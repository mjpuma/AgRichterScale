#!/bin/bash
# Demo script showing how to use the AgriRichter analysis pipeline

echo "=========================================="
echo "AgriRichter Pipeline Demo"
echo "=========================================="
echo ""

# Example 1: Run analysis for wheat
echo "Example 1: Run analysis for wheat"
echo "Command: python scripts/run_agririchter_analysis.py --crop wheat"
echo ""

# Example 2: Run analysis for all crops
echo "Example 2: Run analysis for all crops"
echo "Command: python scripts/run_agririchter_analysis.py --all"
echo ""

# Example 3: Run with custom output directory
echo "Example 3: Run with custom output directory"
echo "Command: python scripts/run_agririchter_analysis.py --crop rice --output my_results"
echo ""

# Example 4: Run with debug logging
echo "Example 4: Run with debug logging and log file"
echo "Command: python scripts/run_agririchter_analysis.py --crop allgrain --log-level DEBUG --log-file analysis.log"
echo ""

# Example 5: Run with SPAM 2010 data
echo "Example 5: Run with SPAM 2010 data"
echo "Command: python scripts/run_agririchter_analysis.py --crop wheat --spam-version 2010"
echo ""

# Example 6: Run with static thresholds
echo "Example 6: Run with static thresholds"
echo "Command: python scripts/run_agririchter_analysis.py --crop wheat --use-static-thresholds"
echo ""

echo "=========================================="
echo "To see all available options:"
echo "python scripts/run_agririchter_analysis.py --help"
echo "=========================================="

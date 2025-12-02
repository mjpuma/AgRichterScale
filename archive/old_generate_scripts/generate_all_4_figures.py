#!/usr/bin/env python3
"""
Master script to generate all 4 publication figures.

Runs each figure generation script in sequence.
"""

import logging
import sys
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('master')


def run_script(script_name: str) -> bool:
    """Run a figure generation script."""
    logger.info("=" * 80)
    logger.info(f"Running {script_name}...")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            ['python3', script_name],
            capture_output=False,
            text=True,
            check=True
        )
        logger.info(f"✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ {script_name} failed: {e}")
        return False


def main():
    """Generate all 4 figures."""
    logger.info("🌾 AGRICHTER PUBLICATION FIGURES GENERATOR")
    logger.info("Generating all 4 publication figures using real SPAM data")
    logger.info("=" * 80)
    
    scripts = [
        'fig1_global_maps.py',
        'fig2_agrichter_scale.py',
        'fig3_hp_envelopes.py',
        'fig4_country_envelopes.py'
    ]
    
    results = {}
    for script in scripts:
        if not Path(script).exists():
            logger.error(f"❌ Script not found: {script}")
            results[script] = False
            continue
        
        results[script] = run_script(script)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 GENERATION SUMMARY")
    logger.info("=" * 80)
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{script}: {status}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("")
        logger.info("🎉 ALL 4 FIGURES GENERATED SUCCESSFULLY!")
        logger.info("✅ Figure 1: results/figure1_global_maps.png")
        logger.info("✅ Figure 2: results/figure2_agrichter_scale.png")
        logger.info("✅ Figure 3: results/figure3_hp_envelopes.png")
        logger.info("✅ Figure 4: results/figure4_country_envelopes.png")
        return 0
    else:
        logger.info("")
        logger.info("⚠️  Some figures failed to generate")
        return 1


if __name__ == "__main__":
    sys.exit(main())

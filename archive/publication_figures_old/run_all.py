#!/usr/bin/env python3
"""
Run all 4 publication figure generation scripts.
"""

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('run_all')

def run_script(script_name):
    """Run a figure generation script."""
    logger.info(f"=" * 60)
    logger.info(f"Running {script_name}...")
    logger.info(f"=" * 60)
    
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
    """Run all 4 figure scripts."""
    logger.info("🌾 GENERATING ALL 4 PUBLICATION FIGURES")
    logger.info("=" * 60)
    
    scripts = [
        'fig1_global_maps.py',
        'fig2_agrichter_scale.py',
        'fig3_hp_envelopes.py',
        'fig4_country_envelopes.py'
    ]
    
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 SUMMARY")
    logger.info("=" * 60)
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{script}: {status}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("")
        logger.info("🎉 ALL 4 FIGURES GENERATED SUCCESSFULLY!")
        return 0
    else:
        logger.info("")
        logger.info("⚠️  Some figures failed to generate")
        return 1

if __name__ == "__main__":
    sys.exit(main())

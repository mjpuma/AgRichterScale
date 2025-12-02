#!/usr/bin/env python3
"""
Generate All Publication Figures

Runs all 4 figure generation scripts in sequence to create the complete
set of publication-ready figures for the AgRichter methodology paper.
"""

import logging
import sys
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('all_figures')


def run_figure_script(script_name: str) -> bool:
    """Run a figure generation script and return success status."""
    logger.info(f"🎯 Running {script_name}...")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        # Log the output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        logger.info(f"✅ {script_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {script_name} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ {script_name} failed with error: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("🌾 AGRICHTER PUBLICATION FIGURES GENERATOR")
    logger.info("Generating all 4 publication-ready figures")
    logger.info("=" * 60)
    
    # List of figure scripts to run
    figure_scripts = [
        'generate_figure1.py',  # Global Maps (8-panel)
        'generate_figure2.py',  # AgRichter Scale (4-panel)
        'generate_figure3.py',  # H-P Envelopes (4-panel)
        'generate_figure4.py'   # Country H-P Envelopes
    ]
    
    success_count = 0
    
    for script in figure_scripts:
        if Path(script).exists():
            if run_figure_script(script):
                success_count += 1
            logger.info("")  # Add spacing between scripts
        else:
            logger.error(f"❌ Script not found: {script}")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"📊 SUMMARY: {success_count}/{len(figure_scripts)} figures generated successfully")
    
    if success_count == len(figure_scripts):
        logger.info("🎉 ALL FIGURES GENERATED SUCCESSFULLY!")
        logger.info("")
        logger.info("✅ Publication-ready figures:")
        logger.info("  - Figure 1: Global Maps (results/figure1_global_maps.png)")
        logger.info("  - Figure 2: AgRichter Scale (results/figure2_agrichter_scale.png)")
        logger.info("  - Figure 3: H-P Envelopes (results/figure3_hp_envelopes.png)")
        logger.info("  - Figure 4: Country Envelopes (results/figure4_country_envelopes.png)")
        logger.info("")
        logger.info("🔬 Features:")
        logger.info("  ✅ Real SPAM 2020 data with coordinate alignment fixes")
        logger.info("  ✅ Historical events with proper labels (no old thresholds)")
        logger.info("  ✅ Professional fonts and formatting for Science/Nature Food")
        logger.info("  ✅ Avoids very small values in H-P envelope figures")
        logger.info("  ✅ 300 DPI publication quality")
        
        return 0
    else:
        logger.warning(f"⚠️ Only {success_count}/{len(figure_scripts)} figures completed")
        logger.info("Check the logs above for specific errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
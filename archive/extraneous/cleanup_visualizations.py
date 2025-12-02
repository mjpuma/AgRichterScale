#!/usr/bin/env python3
"""
Cleanup Visualizations Folder

The results/visualizations folder contains 58MB of intermediate development files
from building the comprehensive figure system. This script helps clean it up.
"""

import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cleanup')


def analyze_visualizations_folder():
    """Analyze what's in the visualizations folder."""
    viz_dir = Path('results/visualizations')
    
    if not viz_dir.exists():
        logger.info("No visualizations folder found")
        return
    
    logger.info("📊 VISUALIZATIONS FOLDER ANALYSIS")
    logger.info("=" * 50)
    
    # Get folder size
    total_size = sum(f.stat().st_size for f in viz_dir.rglob('*') if f.is_file())
    logger.info(f"Total size: {total_size / (1024*1024):.1f} MB")
    
    # Count files by type
    png_files = list(viz_dir.rglob('*.png'))
    md_files = list(viz_dir.rglob('*.md'))
    other_files = [f for f in viz_dir.rglob('*') if f.is_file() and f.suffix not in ['.png', '.md']]
    
    logger.info(f"PNG files: {len(png_files)}")
    logger.info(f"Markdown files: {len(md_files)}")
    logger.info(f"Other files: {len(other_files)}")
    
    # List subdirectories
    subdirs = [d for d in viz_dir.iterdir() if d.is_dir()]
    logger.info(f"Subdirectories: {len(subdirs)}")
    
    for subdir in sorted(subdirs):
        file_count = len(list(subdir.rglob('*')))
        logger.info(f"  - {subdir.name}: {file_count} files")
    
    return total_size, len(png_files), len(subdirs)


def cleanup_visualizations(keep_final=True):
    """Clean up visualizations folder, optionally keeping final versions."""
    viz_dir = Path('results/visualizations')
    
    if not viz_dir.exists():
        logger.info("No visualizations folder to clean")
        return
    
    logger.info("🧹 CLEANING VISUALIZATIONS FOLDER")
    logger.info("=" * 50)
    
    if keep_final:
        # Keep only essential directories
        keep_dirs = [
            'corrected_final',
            'enhanced_agrichter', 
            'final_publication'
        ]
        
        # Keep essential markdown files
        keep_files = [
            'COMPREHENSIVE_FIGURE_SUMMARY.md',
            'CORRECTED_EVENT_LABELS_SUMMARY.md',
            'ENHANCED_EVENT_LABELS_SUMMARY.md'
        ]
        
        removed_count = 0
        
        for item in viz_dir.iterdir():
            if item.is_dir() and item.name not in keep_dirs:
                logger.info(f"Removing directory: {item.name}")
                shutil.rmtree(item)
                removed_count += 1
            elif item.is_file() and item.name not in keep_files:
                logger.info(f"Removing file: {item.name}")
                item.unlink()
                removed_count += 1
        
        logger.info(f"✅ Removed {removed_count} items, kept essential directories")
        
    else:
        # Remove entire visualizations folder
        logger.info("Removing entire visualizations folder...")
        shutil.rmtree(viz_dir)
        logger.info("✅ Visualizations folder removed")


def main():
    """Main function."""
    logger.info("🔍 VISUALIZATIONS FOLDER CLEANUP TOOL")
    logger.info("=" * 50)
    
    # Analyze current state
    analyze_visualizations_folder()
    
    logger.info("")
    logger.info("💡 RECOMMENDATIONS:")
    logger.info("The visualizations folder contains intermediate development files.")
    logger.info("You can safely clean it up since the main figures are in results/")
    logger.info("")
    logger.info("Options:")
    logger.info("1. Keep essential directories only (recommended)")
    logger.info("2. Remove entire visualizations folder")
    logger.info("3. Keep everything as-is")
    logger.info("")
    logger.info("To clean up, run:")
    logger.info("  python cleanup_visualizations.py --clean")
    logger.info("  python cleanup_visualizations.py --clean-all")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clean':
            cleanup_visualizations(keep_final=True)
        elif sys.argv[1] == '--clean-all':
            cleanup_visualizations(keep_final=False)
        else:
            main()
    else:
        main()
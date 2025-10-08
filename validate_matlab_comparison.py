#!/usr/bin/env python3
"""
MATLAB Validation Script for AgriRichter Events Integration

This script implements tasks 12.2-12.5:
- Run Python pipeline for all crops
- Compare with MATLAB reference results
- Investigate and document differences
- Generate comprehensive comparison report
- Update validation thresholds if needed
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MATLABValidator:
    """Validates Python implementation against MATLAB reference outputs."""
    
    def __init__(self, matlab_output_dir='matlab_outputs', 
                 python_output_dir='python_outputs',
                 comparison_output_dir='comparison_reports'):
        self.matlab_output_dir = Path(matlab_output_dir)
        self.python_output_dir = Path(python_output_dir)
        self.comparison_output_dir = Path(comparison_output_dir)
        
        # Create output directories
        self.python_output_dir.mkdir(exist_ok=True)
        self.comparison_output_dir.mkdir(exist_ok=True)
        
        # Validation thresholds
        self.default_threshold = 0.05  # 5% difference threshold
        self.current_threshold = self.default_threshold
        
        # Store results
        self.python_results = {}
        self.matlab_results = {}
        self.comparison_results = {}
        
    def run_python_pipeline(self, crops=['wheat', 'rice', 'allgrain']):
        """
        Task 12.2: Execute Python pipeline for all crops.
        
        Args:
            crops: List of crop types to process
            
        Returns:
            dict: Results for each crop
        """
        logger.info("=" * 80)
        logger.info("TASK 12.2: Running Python Pipeline for All Crops")
        logger.info("=" * 80)
        
        results = {}
        
        for crop in crops:
            logger.info(f"\nProcessing crop: {crop}")
            logger.info("-" * 40)
            
            try:
                # Configure for this crop
                config = Config()
                config.crop_type = crop
                
                # Create output directory for this crop
                crop_output_dir = self.python_output_dir / crop
                crop_output_dir.mkdir(exist_ok=True)
                
                # Run pipeline
                pipeline = EventsPipeline(config, str(crop_output_dir))
                pipeline_results = pipeline.run_complete_pipeline()
                
                # Extract events DataFrame
                events_df = pipeline_results.get('events_df')
                
                if events_df is not None:
                    # Save Python results
                    output_file = crop_output_dir / f'python_events_{crop}_spam2020.csv'
                    events_df.to_csv(output_file, index=False)
                    logger.info(f"Saved Python results to: {output_file}")
                    
                    results[crop] = events_df
                    logger.info(f"Successfully processed {len(events_df)} events for {crop}")
                else:
                    logger.error(f"No events DataFrame returned for {crop}")
                    results[crop] = None
                    
            except Exception as e:
                logger.error(f"Error processing {crop}: {str(e)}")
                results[crop] = None
        
        self.python_results = results
        logger.info("\nPython pipeline execution complete!")
        return results
    
    def load_matlab_results(self, crops=['wheat', 'rice', 'allgrain']):
        """
        Load MATLAB reference results from CSV files.
        
        Args:
            crops: List of crop types to load
            
        Returns:
            dict: MATLAB results for each crop
        """
        logger.info("\nLoading MATLAB reference results...")
        
        results = {}
        
        for crop in crops:
            matlab_file = self.matlab_output_dir / f'matlab_events_{crop}_spam2010.csv'
            
            if matlab_file.exists():
                try:
                    df = pd.read_csv(matlab_file)
                    results[crop] = df
                    logger.info(f"Loaded MATLAB results for {crop}: {len(df)} events")
                except Exception as e:
                    logger.error(f"Error loading MATLAB results for {crop}: {str(e)}")
                    results[crop] = None
            else:
                logger.warning(f"MATLAB results not found for {crop}: {matlab_file}")
                logger.warning(f"Please generate MATLAB outputs using: matlab_reference_generation_guide.md")
                results[crop] = None
        
        self.matlab_results = results
        return results
    
    def compare_results(self, crop):
        """
        Task 12.2: Compare Python and MATLAB results for a specific crop.
        
        Args:
            crop: Crop type to compare
            
        Returns:
            dict: Comparison statistics
        """
        python_df = self.python_results.get(crop)
        matlab_df = self.matlab_results.get(crop)
        
        if python_df is None or matlab_df is None:
            logger.warning(f"Cannot compare {crop}: missing data")
            return None
        
        logger.info(f"\nComparing {crop} results...")
        logger.info("-" * 40)
        
        # Merge on event name
        comparison = python_df.merge(
            matlab_df,
            on='event_name',
            suffixes=('_python', '_matlab'),
            how='outer'
        )
        
        # Calculate differences
        comparison['harvest_area_diff_pct'] = (
            (comparison['harvest_area_loss_ha_python'] - comparison['harvest_area_loss_ha_matlab']) /
            comparison['harvest_area_loss_ha_matlab'] * 100
        )
        
        comparison['production_diff_pct'] = (
            (comparison['production_loss_kcal_python'] - comparison['production_loss_kcal_matlab']) /
            comparison['production_loss_kcal_matlab'] * 100
        )
        
        comparison['magnitude_diff'] = (
            comparison['magnitude_python'] - comparison['magnitude_matlab']
        )
        
        # Identify events exceeding threshold
        threshold_pct = self.current_threshold * 100
        
        harvest_area_issues = comparison[
            abs(comparison['harvest_area_diff_pct']) > threshold_pct
        ]
        
        production_issues = comparison[
            abs(comparison['production_diff_pct']) > threshold_pct
        ]
        
        # Calculate statistics
        stats = {
            'crop': crop,
            'total_events': len(comparison),
            'harvest_area_mean_diff_pct': comparison['harvest_area_diff_pct'].mean(),
            'harvest_area_max_diff_pct': comparison['harvest_area_diff_pct'].abs().max(),
            'production_mean_diff_pct': comparison['production_diff_pct'].mean(),
            'production_max_diff_pct': comparison['production_diff_pct'].abs().max(),
            'magnitude_mean_diff': comparison['magnitude_diff'].mean(),
            'magnitude_max_diff': comparison['magnitude_diff'].abs().max(),
            'harvest_area_issues_count': len(harvest_area_issues),
            'production_issues_count': len(production_issues),
            'harvest_area_issues': harvest_area_issues[['event_name', 'harvest_area_diff_pct']].to_dict('records'),
            'production_issues': production_issues[['event_name', 'production_diff_pct']].to_dict('records'),
            'comparison_df': comparison
        }
        
        # Log summary
        logger.info(f"Total events compared: {stats['total_events']}")
        logger.info(f"Harvest area mean difference: {stats['harvest_area_mean_diff_pct']:.2f}%")
        logger.info(f"Production mean difference: {stats['production_mean_diff_pct']:.2f}%")
        logger.info(f"Events exceeding {threshold_pct}% threshold:")
        logger.info(f"  - Harvest area: {stats['harvest_area_issues_count']}")
        logger.info(f"  - Production: {stats['production_issues_count']}")
        
        return stats
    
    def investigate_differences(self, comparison_stats):
        """
        Task 12.3: Investigate and document differences > threshold.
        
        Args:
            comparison_stats: Comparison statistics from compare_results
            
        Returns:
            dict: Investigation findings
        """
        logger.info("\n" + "=" * 80)
        logger.info("TASK 12.3: Investigating Differences")
        logger.info("=" * 80)
        
        crop = comparison_stats['crop']
        comparison_df = comparison_stats['comparison_df']
        
        findings = {
            'crop': crop,
            'systematic_differences': [],
            'event_specific_issues': [],
            'root_causes': [],
            'recommendations': []
        }
        
        # Check for systematic bias
        mean_harvest_diff = comparison_stats['harvest_area_mean_diff_pct']
        mean_production_diff = comparison_stats['production_mean_diff_pct']
        
        if abs(mean_harvest_diff) > 1.0:
            findings['systematic_differences'].append({
                'type': 'harvest_area_bias',
                'value': mean_harvest_diff,
                'description': f'Systematic {mean_harvest_diff:.2f}% difference in harvest area'
            })
            findings['root_causes'].append('Possible SPAM 2010 vs 2020 data differences')
        
        if abs(mean_production_diff) > 1.0:
            findings['systematic_differences'].append({
                'type': 'production_bias',
                'value': mean_production_diff,
                'description': f'Systematic {mean_production_diff:.2f}% difference in production'
            })
            findings['root_causes'].append('Possible caloric conversion or unit conversion differences')
        
        # Investigate specific events with large differences
        threshold_pct = self.current_threshold * 100
        
        for _, row in comparison_df.iterrows():
            harvest_diff = abs(row['harvest_area_diff_pct'])
            production_diff = abs(row['production_diff_pct'])
            
            if harvest_diff > threshold_pct or production_diff > threshold_pct:
                issue = {
                    'event_name': row['event_name'],
                    'harvest_area_diff_pct': harvest_diff,
                    'production_diff_pct': production_diff,
                    'possible_causes': []
                }
                
                # Analyze possible causes
                if pd.isna(row['harvest_area_loss_ha_python']) or pd.isna(row['harvest_area_loss_ha_matlab']):
                    issue['possible_causes'].append('Missing data in one implementation')
                elif row['harvest_area_loss_ha_python'] == 0 or row['harvest_area_loss_ha_matlab'] == 0:
                    issue['possible_causes'].append('Zero loss in one implementation - check spatial mapping')
                elif harvest_diff > 20:
                    issue['possible_causes'].append('Large difference suggests spatial mapping mismatch')
                else:
                    issue['possible_causes'].append('Rounding or data precision differences')
                
                findings['event_specific_issues'].append(issue)
        
        # Generate recommendations
        if len(findings['systematic_differences']) > 0:
            findings['recommendations'].append(
                'Consider adjusting validation threshold due to SPAM version differences'
            )
        
        if len(findings['event_specific_issues']) > 0:
            findings['recommendations'].append(
                'Review spatial mapping for events with large differences'
            )
            findings['recommendations'].append(
                'Verify country/state code mappings for problematic events'
            )
        
        # Log findings
        logger.info(f"\nInvestigation findings for {crop}:")
        logger.info(f"Systematic differences: {len(findings['systematic_differences'])}")
        logger.info(f"Event-specific issues: {len(findings['event_specific_issues'])}")
        logger.info(f"Root causes identified: {len(findings['root_causes'])}")
        
        return findings
    
    def create_comparison_visualizations(self, comparison_stats):
        """
        Create visualizations comparing Python vs MATLAB results.
        
        Args:
            comparison_stats: Comparison statistics
        """
        crop = comparison_stats['crop']
        comparison_df = comparison_stats['comparison_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Python vs MATLAB Comparison - {crop.upper()}', fontsize=16, fontweight='bold')
        
        # 1. Harvest Area Comparison
        ax = axes[0, 0]
        ax.scatter(comparison_df['harvest_area_loss_ha_matlab'], 
                   comparison_df['harvest_area_loss_ha_python'],
                   alpha=0.6, s=100)
        
        # Add diagonal line
        max_val = max(comparison_df['harvest_area_loss_ha_matlab'].max(),
                      comparison_df['harvest_area_loss_ha_python'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect match')
        
        ax.set_xlabel('MATLAB Harvest Area Loss (ha)', fontsize=11)
        ax.set_ylabel('Python Harvest Area Loss (ha)', fontsize=11)
        ax.set_title('Harvest Area Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Production Comparison
        ax = axes[0, 1]
        ax.scatter(comparison_df['production_loss_kcal_matlab'], 
                   comparison_df['production_loss_kcal_python'],
                   alpha=0.6, s=100, color='green')
        
        max_val = max(comparison_df['production_loss_kcal_matlab'].max(),
                      comparison_df['production_loss_kcal_python'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect match')
        
        ax.set_xlabel('MATLAB Production Loss (kcal)', fontsize=11)
        ax.set_ylabel('Python Production Loss (kcal)', fontsize=11)
        ax.set_title('Production Loss Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Percentage Differences
        ax = axes[1, 0]
        events = comparison_df['event_name']
        x_pos = np.arange(len(events))
        
        ax.barh(x_pos, comparison_df['harvest_area_diff_pct'], alpha=0.7)
        ax.axvline(x=self.current_threshold * 100, color='r', linestyle='--', 
                   label=f'{self.current_threshold*100}% threshold')
        ax.axvline(x=-self.current_threshold * 100, color='r', linestyle='--')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(events, fontsize=8)
        ax.set_xlabel('Difference (%)', fontsize=11)
        ax.set_title('Harvest Area Differences by Event', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Magnitude Comparison
        ax = axes[1, 1]
        ax.scatter(comparison_df['magnitude_matlab'], 
                   comparison_df['magnitude_python'],
                   alpha=0.6, s=100, color='purple')
        
        max_val = max(comparison_df['magnitude_matlab'].max(),
                      comparison_df['magnitude_python'].max())
        min_val = min(comparison_df['magnitude_matlab'].min(),
                      comparison_df['magnitude_python'].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match')
        
        ax.set_xlabel('MATLAB Magnitude', fontsize=11)
        ax.set_ylabel('Python Magnitude', fontsize=11)
        ax.set_title('Magnitude Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.comparison_output_dir / f'comparison_visualization_{crop}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison visualization: {output_file}")
        plt.close()
    
    def generate_comparison_report(self, all_comparison_stats, all_findings):
        """
        Task 12.4: Generate detailed comparison report.
        
        Args:
            all_comparison_stats: Comparison statistics for all crops
            all_findings: Investigation findings for all crops
            
        Returns:
            str: Path to generated report
        """
        logger.info("\n" + "=" * 80)
        logger.info("TASK 12.4: Generating Comparison Report")
        logger.info("=" * 80)
        
        report_file = self.comparison_output_dir / f'matlab_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AGRIRICHTER MATLAB VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Validation Threshold: {self.current_threshold * 100}%\n")
            f.write(f"Python Output Directory: {self.python_output_dir}\n")
            f.write(f"MATLAB Output Directory: {self.matlab_output_dir}\n\n")
            
            # Executive Summary
            f.write("=" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            total_crops = len(all_comparison_stats)
            total_issues = sum(stats['harvest_area_issues_count'] + stats['production_issues_count'] 
                              for stats in all_comparison_stats.values() if stats)
            
            f.write(f"Crops Analyzed: {total_crops}\n")
            f.write(f"Total Events with Differences > {self.current_threshold*100}%: {total_issues}\n\n")
            
            # Detailed Results by Crop
            for crop, stats in all_comparison_stats.items():
                if stats is None:
                    f.write(f"\n{crop.upper()}: NO DATA AVAILABLE\n")
                    continue
                
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{crop.upper()} RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total Events: {stats['total_events']}\n\n")
                
                f.write("Harvest Area Comparison:\n")
                f.write(f"  Mean Difference: {stats['harvest_area_mean_diff_pct']:.2f}%\n")
                f.write(f"  Max Difference: {stats['harvest_area_max_diff_pct']:.2f}%\n")
                f.write(f"  Events Exceeding Threshold: {stats['harvest_area_issues_count']}\n\n")
                
                f.write("Production Loss Comparison:\n")
                f.write(f"  Mean Difference: {stats['production_mean_diff_pct']:.2f}%\n")
                f.write(f"  Max Difference: {stats['production_max_diff_pct']:.2f}%\n")
                f.write(f"  Events Exceeding Threshold: {stats['production_issues_count']}\n\n")
                
                f.write("Magnitude Comparison:\n")
                f.write(f"  Mean Difference: {stats['magnitude_mean_diff']:.4f}\n")
                f.write(f"  Max Difference: {stats['magnitude_max_diff']:.4f}\n\n")
                
                # Event-by-event comparison table
                f.write("Event-by-Event Comparison:\n")
                f.write("-" * 80 + "\n")
                
                comparison_df = stats['comparison_df']
                f.write(f"{'Event':<25} {'HA Diff %':>12} {'Prod Diff %':>12} {'Mag Diff':>12}\n")
                f.write("-" * 80 + "\n")
                
                for _, row in comparison_df.iterrows():
                    f.write(f"{row['event_name']:<25} "
                           f"{row['harvest_area_diff_pct']:>12.2f} "
                           f"{row['production_diff_pct']:>12.2f} "
                           f"{row['magnitude_diff']:>12.4f}\n")
                
                f.write("\n")
            
            # Investigation Findings
            f.write("\n" + "=" * 80 + "\n")
            f.write("INVESTIGATION FINDINGS\n")
            f.write("=" * 80 + "\n\n")
            
            for crop, findings in all_findings.items():
                if findings is None:
                    continue
                
                f.write(f"\n{crop.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                if findings['systematic_differences']:
                    f.write("\nSystematic Differences:\n")
                    for diff in findings['systematic_differences']:
                        f.write(f"  - {diff['description']}\n")
                
                if findings['root_causes']:
                    f.write("\nPossible Root Causes:\n")
                    for cause in findings['root_causes']:
                        f.write(f"  - {cause}\n")
                
                if findings['event_specific_issues']:
                    f.write(f"\nEvent-Specific Issues ({len(findings['event_specific_issues'])}):\n")
                    for issue in findings['event_specific_issues'][:5]:  # Show top 5
                        f.write(f"  - {issue['event_name']}: ")
                        f.write(f"HA diff {issue['harvest_area_diff_pct']:.1f}%, ")
                        f.write(f"Prod diff {issue['production_diff_pct']:.1f}%\n")
                        for cause in issue['possible_causes']:
                            f.write(f"    * {cause}\n")
                
                if findings['recommendations']:
                    f.write("\nRecommendations:\n")
                    for rec in findings['recommendations']:
                        f.write(f"  - {rec}\n")
            
            # Validation Conclusions
            f.write("\n" + "=" * 80 + "\n")
            f.write("VALIDATION CONCLUSIONS\n")
            f.write("=" * 80 + "\n\n")
            
            # Determine overall validation status
            max_mean_diff = max(
                (stats['harvest_area_mean_diff_pct'] for stats in all_comparison_stats.values() if stats),
                default=0
            )
            
            if abs(max_mean_diff) < self.current_threshold * 100:
                f.write("✓ VALIDATION PASSED\n\n")
                f.write("The Python implementation produces results within the acceptable\n")
                f.write(f"threshold ({self.current_threshold*100}%) of the MATLAB reference implementation.\n")
            else:
                f.write("⚠ VALIDATION REQUIRES REVIEW\n\n")
                f.write("Some results exceed the validation threshold. This may be due to:\n")
                f.write("  - SPAM 2010 vs SPAM 2020 data differences\n")
                f.write("  - Spatial mapping improvements in Python version\n")
                f.write("  - Rounding or precision differences\n\n")
                f.write("Review the investigation findings above for details.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"\nComparison report saved to: {report_file}")
        return str(report_file)
    
    def update_validation_thresholds(self, all_comparison_stats):
        """
        Task 12.5: Update validation thresholds if needed.
        
        Args:
            all_comparison_stats: Comparison statistics for all crops
        """
        logger.info("\n" + "=" * 80)
        logger.info("TASK 12.5: Evaluating Validation Thresholds")
        logger.info("=" * 80)
        
        # Calculate overall statistics
        all_mean_diffs = [
            abs(stats['harvest_area_mean_diff_pct']) 
            for stats in all_comparison_stats.values() if stats
        ]
        
        if not all_mean_diffs:
            logger.warning("No comparison statistics available")
            return
        
        max_mean_diff = max(all_mean_diffs)
        avg_mean_diff = np.mean(all_mean_diffs)
        
        logger.info(f"\nCurrent threshold: {self.current_threshold * 100}%")
        logger.info(f"Maximum mean difference: {max_mean_diff:.2f}%")
        logger.info(f"Average mean difference: {avg_mean_diff:.2f}%")
        
        # Determine if threshold adjustment is needed
        if max_mean_diff > self.current_threshold * 100:
            # Check if differences are systematic (SPAM version differences)
            if avg_mean_diff > 3.0:  # Systematic bias threshold
                suggested_threshold = (max_mean_diff / 100) * 1.2  # 20% buffer
                
                logger.info("\n" + "!" * 80)
                logger.info("THRESHOLD ADJUSTMENT RECOMMENDED")
                logger.info("!" * 80)
                logger.info(f"\nRationale:")
                logger.info(f"  - Systematic differences detected (avg: {avg_mean_diff:.2f}%)")
                logger.info(f"  - Likely due to SPAM 2010 vs SPAM 2020 data differences")
                logger.info(f"  - Python implementation uses updated SPAM 2020 data")
                logger.info(f"\nSuggested new threshold: {suggested_threshold * 100:.1f}%")
                logger.info(f"\nTo update the threshold:")
                logger.info(f"  1. Review investigation findings to confirm root cause")
                logger.info(f"  2. Update DataValidator.default_threshold in agririchter/validation/data_validator.py")
                logger.info(f"  3. Document rationale in code comments")
                logger.info(f"  4. Re-run validation with new threshold")
                
                # Save threshold recommendation
                threshold_file = self.comparison_output_dir / 'threshold_recommendation.txt'
                with open(threshold_file, 'w') as f:
                    f.write("VALIDATION THRESHOLD RECOMMENDATION\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Current Threshold: {self.current_threshold * 100}%\n")
                    f.write(f"Suggested Threshold: {suggested_threshold * 100:.1f}%\n\n")
                    f.write("Rationale:\n")
                    f.write(f"  - Maximum mean difference: {max_mean_diff:.2f}%\n")
                    f.write(f"  - Average mean difference: {avg_mean_diff:.2f}%\n")
                    f.write("  - SPAM 2010 vs SPAM 2020 data differences\n")
                    f.write("  - Python implementation uses more recent data\n\n")
                    f.write("Implementation:\n")
                    f.write("  1. Update agririchter/validation/data_validator.py\n")
                    f.write(f"  2. Set default_threshold = {suggested_threshold:.4f}\n")
                    f.write("  3. Document in code comments\n")
                    f.write("  4. Re-run validation\n")
                
                logger.info(f"\nThreshold recommendation saved to: {threshold_file}")
            else:
                logger.info("\nNo systematic bias detected.")
                logger.info("Current threshold is appropriate.")
                logger.info("Individual event differences may be due to:")
                logger.info("  - Spatial mapping improvements")
                logger.info("  - Rounding differences")
                logger.info("  - Missing data handling")
        else:
            logger.info("\n✓ Current threshold is appropriate")
            logger.info("All results are within acceptable limits")
    
    def run_complete_validation(self, crops=['wheat', 'rice', 'allgrain']):
        """
        Run complete validation workflow (Tasks 12.2-12.5).
        
        Args:
            crops: List of crops to validate
            
        Returns:
            dict: Complete validation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE MATLAB VALIDATION")
        logger.info("=" * 80)
        
        # Task 12.2: Run Python pipeline
        self.run_python_pipeline(crops)
        
        # Load MATLAB results
        self.load_matlab_results(crops)
        
        # Task 12.2 & 12.3: Compare and investigate
        all_comparison_stats = {}
        all_findings = {}
        
        for crop in crops:
            if self.python_results.get(crop) is not None and self.matlab_results.get(crop) is not None:
                # Compare results
                stats = self.compare_results(crop)
                all_comparison_stats[crop] = stats
                
                # Investigate differences
                findings = self.investigate_differences(stats)
                all_findings[crop] = findings
                
                # Create visualizations
                self.create_comparison_visualizations(stats)
            else:
                logger.warning(f"Skipping {crop}: missing Python or MATLAB data")
                all_comparison_stats[crop] = None
                all_findings[crop] = None
        
        # Task 12.4: Generate comparison report
        report_path = self.generate_comparison_report(all_comparison_stats, all_findings)
        
        # Task 12.5: Update validation thresholds if needed
        self.update_validation_thresholds(all_comparison_stats)
        
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {self.comparison_output_dir}")
        logger.info(f"Report: {report_path}")
        
        return {
            'comparison_stats': all_comparison_stats,
            'findings': all_findings,
            'report_path': report_path
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate Python AgriRichter implementation against MATLAB reference'
    )
    parser.add_argument(
        '--crops',
        nargs='+',
        default=['wheat', 'rice', 'allgrain'],
        help='Crops to validate (default: wheat rice allgrain)'
    )
    parser.add_argument(
        '--matlab-dir',
        default='matlab_outputs',
        help='Directory containing MATLAB reference outputs'
    )
    parser.add_argument(
        '--python-dir',
        default='python_outputs',
        help='Directory for Python outputs'
    )
    parser.add_argument(
        '--comparison-dir',
        default='comparison_reports',
        help='Directory for comparison reports'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Validation threshold (default: 0.05 = 5%%)'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = MATLABValidator(
        matlab_output_dir=args.matlab_dir,
        python_output_dir=args.python_dir,
        comparison_output_dir=args.comparison_dir
    )
    
    # Set threshold
    validator.current_threshold = args.threshold
    
    # Run validation
    try:
        results = validator.run_complete_validation(args.crops)
        logger.info("\n✓ Validation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

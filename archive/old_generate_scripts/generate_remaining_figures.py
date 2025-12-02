#!/usr/bin/env python3
"""
Generate Remaining Publication Figures

Generate the 3 remaining required figures:
- Figure 2: AgRichter Scale for Allgrain, Wheat, Maize, Rice
- Figure 3: H-P Envelopes for Allgrain, Wheat, Maize, Rice  
- Figure 4: Country H-P Envelopes for Allgrain

Uses existing AgRichter framework with real SPAM data and historical events.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('remaining_figures')

# Import AgRichter modules
from agririchter.core.config import Config
from agririchter.analysis.agririchter import AgRichterAnalyzer
from agririchter.analysis.envelope import EnvelopeAnalyzer
from agririchter.visualization.agririchter_scale import AgRichterScaleVisualizer
from agririchter.visualization.hp_envelope import HPEnvelopeVisualizer


def generate_figure2_agrichter_scale():
    """Generate Figure 2: AgRichter Scale for 4 crops."""
    logger.info("🎯 GENERATING FIGURE 2: AgRichter Scale")
    
    crops = ['allgrain', 'wheat', 'maize', 'rice']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, crop in enumerate(crops):
        logger.info(f"  Processing {crop}...")
        
        try:
            # Initialize config and calculator
            config = Config(crop_type=crop, root_dir='.')
            config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
            config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
            
            calculator = AgRichterCalculator(config)
            
            # Calculate AgRichter scale
            results = calculator.calculate_agrichter_scale()
            
            # Create visualizer and plot
            visualizer = AgRichterVisualizer(config)
            visualizer.plot_agrichter_scale(
                results, 
                ax=axes[i], 
                title=f'{crop.title()} AgRichter Scale',
                include_events=True
            )
            
            logger.info(f"  ✅ {crop} AgRichter scale plotted")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to process {crop}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {crop}', ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/figure2_agrichter_scale.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Figure 2 saved: {output_path}")
    return True


def generate_figure3_hp_envelopes():
    """Generate Figure 3: H-P Envelopes for 4 crops."""
    logger.info("🎯 GENERATING FIGURE 3: H-P Envelopes")
    
    crops = ['allgrain', 'wheat', 'maize', 'rice']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, crop in enumerate(crops):
        logger.info(f"  Processing {crop}...")
        
        try:
            # Initialize config and calculator
            config = Config(crop_type=crop, root_dir='.')
            config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
            config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
            
            calculator = EnvelopeCalculator(config)
            
            # Calculate H-P envelope
            envelope_data = calculator.calculate_hp_envelope()
            
            # Create visualizer and plot
            visualizer = EnvelopeVisualizer(config)
            visualizer.plot_hp_envelope(
                envelope_data,
                ax=axes[i],
                title=f'{crop.title()} H-P Envelope',
                include_events=True
            )
            
            logger.info(f"  ✅ {crop} H-P envelope plotted")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to process {crop}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {crop}', ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/figure3_hp_envelopes.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Figure 3 saved: {output_path}")
    return True


def generate_figure4_country_envelopes():
    """Generate Figure 4: Country H-P Envelopes for Allgrain."""
    logger.info("🎯 GENERATING FIGURE 4: Country H-P Envelopes")
    
    # Key countries for analysis
    countries = ['USA', 'CHN', 'IND', 'BRA', 'RUS', 'ARG']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, country in enumerate(countries):
        logger.info(f"  Processing {country}...")
        
        try:
            # Initialize config for allgrain
            config = Config(crop_type='allgrain', root_dir='.')
            config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
            config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
            
            # Set country filter
            config.country_filter = country
            
            calculator = EnvelopeCalculator(config)
            
            # Calculate country-specific H-P envelope
            envelope_data = calculator.calculate_hp_envelope()
            
            # Create visualizer and plot
            visualizer = EnvelopeVisualizer(config)
            visualizer.plot_hp_envelope(
                envelope_data,
                ax=axes[i],
                title=f'{country} Allgrain H-P Envelope',
                include_events=True
            )
            
            logger.info(f"  ✅ {country} H-P envelope plotted")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to process {country}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {country}', ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/figure4_country_envelopes.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Figure 4 saved: {output_path}")
    return True


def main():
    """Main execution function."""
    logger.info("🌾 GENERATING REMAINING PUBLICATION FIGURES")
    logger.info("Target: Figures 2, 3, and 4 with real SPAM data and historical events")
    logger.info("=" * 60)
    
    success_count = 0
    
    try:
        # Generate Figure 2: AgRichter Scale
        if generate_figure2_agrichter_scale():
            success_count += 1
        
        # Generate Figure 3: H-P Envelopes  
        if generate_figure3_hp_envelopes():
            success_count += 1
            
        # Generate Figure 4: Country H-P Envelopes
        if generate_figure4_country_envelopes():
            success_count += 1
        
        logger.info("")
        logger.info(f"🎉 SUCCESS: {success_count}/3 figures generated!")
        
        if success_count == 3:
            logger.info("✅ All required publication figures completed:")
            logger.info("  - Figure 1: Global Maps (already exists)")
            logger.info("  - Figure 2: AgRichter Scale")
            logger.info("  - Figure 3: H-P Envelopes") 
            logger.info("  - Figure 4: Country H-P Envelopes")
            return 0
        else:
            logger.warning(f"⚠️  Only {success_count}/3 figures completed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
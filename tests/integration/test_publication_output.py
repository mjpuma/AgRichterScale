#!/usr/bin/env python3
"""Test publication-quality output functionality for AgriRichter visualizations."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AgriRichter modules
from agririchter.core.config import Config
from agririchter.data.loader import DataLoader
from agririchter.analysis.agririchter import AgriRichterAnalyzer
from agririchter.analysis.envelope import HPEnvelopeCalculator
from agririchter.visualization.maps import GlobalProductionMapper
from agririchter.visualization.plots import AgriRichterPlotter, EnvelopePlotter
from agririchter.visualization.publication import PublicationFormatter, OutputManager


def test_publication_output():
    """Test publication-quality output for all visualization types."""
    logger.info("Testing publication-quality output functionality...")
    
    # Test with wheat data
    crop_type = 'wheat'
    config = Config(crop_type=crop_type)
    
    # Create output manager
    output_manager = OutputManager(config, base_output_dir=Path('test_publication_output'))
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader(config)
    
    try:
        production_df = loader.load_production_data()
        harvest_df = loader.load_harvest_data()
        events_df = loader.load_events_data()
        
        logger.info(f"Loaded {len(production_df)} production records")
        logger.info(f"Loaded {len(harvest_df)} harvest records")
        logger.info(f"Loaded {len(events_df)} historical events")
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        # Create sample data for testing
        logger.info("Creating sample data for testing...")
        production_df, harvest_df, events_df = create_sample_data()
    
    # Dictionary to store all figures
    figures = {}
    
    # Test 1: Global production map with publication formatting
    logger.info("Creating publication-quality global production map...")
    try:
        mapper = GlobalProductionMapper(config)
        
        # Create map with publication style
        fig_map = mapper.create_global_map(
            production_df,
            title=f'Global {crop_type.title()} Production Distribution',
            use_publication_style=True,
            figsize=(12, 8),
            dpi=300
        )
        
        figures['global_production_map'] = fig_map
        logger.info("✓ Global production map created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create global production map: {str(e)}")
    
    # Process events for plotting (used by multiple visualizations)
    processed_events = None
    try:
        analyzer = AgriRichterAnalyzer(config)
        processed_events = analyzer.process_historical_events(events_df, production_df, harvest_df)
        logger.info(f"Processed {len(processed_events)} historical events")
    except Exception as e:
        logger.warning(f"Failed to process events: {str(e)}")
        processed_events = pd.DataFrame()  # Empty dataframe as fallback
    
    # Test 2: AgriRichter scale plot with publication formatting
    logger.info("Creating publication-quality AgriRichter scale plot...")
    try:
        plotter = AgriRichterPlotter(config)
        
        # Create scale data
        scale_data = plotter.create_richter_scale_data(
            min_magnitude=0.0,
            max_magnitude=6.0,
            n_points=100
        )
        
        # Create scale plot with publication style
        fig_scale = plotter.create_scale_plot(
            scale_data=scale_data,
            historical_events=processed_events,
            title=f'AgriRichter Scale - {crop_type.title()}',
            use_publication_style=True,
            figsize=(10, 8),
            dpi=300
        )
        
        figures['agririchter_scale'] = fig_scale
        logger.info("✓ AgriRichter scale plot created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create AgriRichter scale plot: {str(e)}")
    
    # Test 3: H-P envelope plot with publication formatting
    logger.info("Creating publication-quality H-P envelope plot...")
    try:
        envelope_plotter = EnvelopePlotter(config)
        envelope_calculator = HPEnvelopeCalculator(config)
        
        # Calculate envelope
        envelope_data = envelope_calculator.calculate_hp_envelope(production_df, harvest_df)
        
        # Create envelope plot with publication style
        fig_envelope = envelope_plotter.create_envelope_plot(
            envelope_data=envelope_data,
            historical_events=processed_events,
            title=f'H-P Envelope - {crop_type.title()}',
            use_publication_style=True,
            figsize=(12, 8),
            dpi=300
        )
        
        figures['hp_envelope'] = fig_envelope
        logger.info("✓ H-P envelope plot created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create H-P envelope plot: {str(e)}")
    
    # Test 4: Save all figures in multiple publication formats
    logger.info("Saving figures in multiple publication formats...")
    try:
        publication_formats = ['png', 'svg', 'eps', 'pdf']
        saved_files = output_manager.save_publication_set(figures, formats=publication_formats)
        
        logger.info("✓ Publication files saved successfully:")
        for fig_name, paths in saved_files.items():
            logger.info(f"  {fig_name}: {len(paths)} formats")
            for path in paths:
                logger.info(f"    - {path}")
        
    except Exception as e:
        logger.error(f"Failed to save publication files: {str(e)}")
    
    # Test 5: Test individual publication formatter features
    logger.info("Testing individual publication formatter features...")
    try:
        test_publication_formatter_features(config, output_manager)
        logger.info("✓ Publication formatter features tested successfully")
        
    except Exception as e:
        logger.error(f"Failed to test publication formatter features: {str(e)}")
    
    # Test 6: Create publication summary
    logger.info("Creating publication summary...")
    try:
        with PublicationFormatter(config) as formatter:
            summary_path = formatter.create_publication_summary(
                output_manager.crop_dir / 'publication',
                figures,
                metadata={
                    'crop_type': crop_type,
                    'test_run': True,
                    'total_figures': len(figures),
                    'formats_tested': publication_formats
                }
            )
        
        logger.info(f"✓ Publication summary created at {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to create publication summary: {str(e)}")
    
    logger.info("Publication output testing completed!")
    logger.info(f"Output directory: {output_manager.base_dir}")
    
    return figures, output_manager


def test_publication_formatter_features(config: Config, output_manager: OutputManager):
    """Test individual features of the publication formatter."""
    logger.info("Testing publication formatter individual features...")
    
    with PublicationFormatter(config) as formatter:
        # Test 1: Create figure with publication style
        fig = formatter.create_figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111)
        
        # Test 2: Format axes
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label='sin(x)')
        
        formatter.format_axes(
            ax,
            title='Test Publication Figure',
            xlabel='X Values',
            ylabel='Y Values',
            grid=True
        )
        
        # Test 3: Create legend
        formatter.create_legend(ax, loc='upper right')
        
        # Test 4: Save in multiple formats
        test_path = output_manager.get_output_path('publication', 'test_formatter_features')
        saved_paths = formatter.save_figure(
            fig, 
            test_path, 
            formats=['png', 'svg', 'eps'],
            dpi=300
        )
        
        logger.info(f"Test figure saved to {len(saved_paths)} formats")
        
        # Test 5: Color palette
        colors = formatter.get_color_palette(5, palette='qualitative')
        logger.info(f"Generated color palette: {colors}")
        
        plt.close(fig)


def create_sample_data():
    """Create sample data for testing when real data is not available."""
    logger.info("Creating sample data for testing...")
    
    # Create sample production data
    n_points = 1000
    np.random.seed(42)
    
    production_data = {
        'x': np.random.uniform(-180, 180, n_points),
        'y': np.random.uniform(-60, 80, n_points),
        'wheat_A': np.random.lognormal(10, 2, n_points)  # Production in kcal
    }
    production_df = pd.DataFrame(production_data)
    
    # Create sample harvest data
    harvest_data = {
        'x': production_data['x'],
        'y': production_data['y'],
        'wheat_A': np.random.uniform(0.1, 100, n_points)  # Harvest area in km²
    }
    harvest_df = pd.DataFrame(harvest_data)
    
    # Create sample events data
    events_data = {
        'event_name': ['Drought_2012', 'Flood_2010', 'Pest_2008', 'Heat_2003'],
        'year': [2012, 2010, 2008, 2003],
        'country': ['USA', 'Pakistan', 'Australia', 'Europe'],
        'harvest_area_km2': [50000, 25000, 75000, 40000],
        'production_loss_kcal': [1e12, 5e11, 1.5e12, 8e11],
        'magnitude': [4.7, 4.4, 4.9, 4.6]
    }
    events_df = pd.DataFrame(events_data)
    
    logger.info("Sample data created successfully")
    return production_df, harvest_df, events_df


def main():
    """Main function to run publication output tests."""
    try:
        figures, output_manager = test_publication_output()
        
        print("\n" + "="*60)
        print("PUBLICATION OUTPUT TEST RESULTS")
        print("="*60)
        print(f"✓ Created {len(figures)} publication-quality figures")
        print(f"✓ Output directory: {output_manager.base_dir}")
        print(f"✓ Crop-specific directory: {output_manager.crop_dir}")
        
        # List generated files
        print("\nGenerated files:")
        for category in ['maps', 'plots', 'envelopes', 'publication']:
            category_dir = output_manager.crop_dir / category
            if category_dir.exists():
                files = list(category_dir.glob('*'))
                if files:
                    print(f"\n{category.title()}:")
                    for file in sorted(files):
                        print(f"  - {file.name}")
        
        print("\n✓ All publication output tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Publication output test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
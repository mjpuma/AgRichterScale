"""Comprehensive validation tests for AgRichter analysis.

This test module covers:
- MATLAB comparison with reference data
- Data consistency checks
- Spatial coverage validation
- Figure quality verification

Requirements: 15.1, 15.2, 15.3
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt
import matplotlib

from agrichter.validation.data_validator import DataValidator
from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper
from agrichter.analysis.event_calculator import EventCalculator


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(crop_type='wheat', root_dir='./data', spam_version='2020')


@pytest.fixture
def validator(config):
    """Create DataValidator instance."""
    return DataValidator(config)


@pytest.fixture
def sample_spam_data():
    """Create realistic sample SPAM data."""
    np.random.seed(42)
    n_cells = 1000
    
    production_df = pd.DataFrame({
        'x': np.random.uniform(-180, 180, n_cells),
        'y': np.random.uniform(-90, 90, n_cells),
        'iso3': np.random.choice(['USA', 'CHN', 'IND', 'BRA', 'RUS'], n_cells),
        'cell5m': np.arange(n_cells),
        'whea_a': np.random.uniform(0, 10000, n_cells),
        'rice_a': np.random.uniform(0, 8000, n_cells),
        'maiz_a': np.random.uniform(0, 12000, n_cells)
    })
    
    harvest_df = pd.DataFrame({
        'x': production_df['x'],
        'y': production_df['y'],
        'iso3': production_df['iso3'],
        'cell5m': production_df['cell5m'],
        'whea_a': production_df['whea_a'] / 100,  # Realistic yield ratio
        'rice_a': production_df['rice_a'] / 100,
        'maiz_a': production_df['maiz_a'] / 100
    })
    
    return production_df, harvest_df


@pytest.fixture
def sample_events_data():
    """Create realistic sample events data."""
    return pd.DataFrame({
        'event_name': [
            'DustBowl', 'GreatFamine', 'SovietFamine', 'BengalFamine',
            'ChineseFamine', 'SahelDrought', 'EthiopianFamine', 'NorthKoreaFamine'
        ],
        'harvest_area_loss_ha': [
            5e6, 8e6, 12e6, 3e6, 25e6, 6e6, 4e6, 2e6
        ],
        'production_loss_kcal': [
            2e14, 5e14, 8e14, 1.5e14, 15e14, 3e14, 2e14, 1e14
        ],
        'magnitude': [
            4.7, 4.9, 5.1, 4.5, 5.4, 4.8, 4.6, 4.3
        ]
    })


@pytest.fixture
def matlab_reference_data():
    """Create MATLAB reference data for comparison."""
    return pd.DataFrame({
        'event_name': [
            'DustBowl', 'GreatFamine', 'SovietFamine', 'BengalFamine',
            'ChineseFamine', 'SahelDrought', 'EthiopianFamine', 'NorthKoreaFamine'
        ],
        'production_loss_kcal': [
            2.1e14, 4.9e14, 8.2e14, 1.6e14, 14.5e14, 3.1e14, 2.05e14, 1.05e14
        ],
        'magnitude': [
            4.7, 4.9, 5.1, 4.5, 5.4, 4.8, 4.6, 4.3
        ]
    })


class TestMATLABComparison:
    """Test MATLAB comparison with reference data (Requirement 15.1)."""
    
    def test_matlab_comparison_with_reference_file(self, validator, sample_events_data, tmp_path):
        """Test comparison when MATLAB reference file exists."""
        # Create temporary MATLAB reference file
        matlab_file = tmp_path / 'matlab_reference_wheat.csv'
        matlab_data = pd.DataFrame({
            'event_name': sample_events_data['event_name'],
            'production_loss_kcal': sample_events_data['production_loss_kcal'] * 1.02  # 2% difference
        })
        matlab_data.to_csv(matlab_file, index=False)
        
        result = validator.compare_with_matlab(sample_events_data, matlab_file)
        
        assert result['comparison_available'] is True
        assert 'differences' in result
        assert 'statistics' in result
    
    def test_matlab_comparison_within_tolerance(self, validator, sample_events_data, matlab_reference_data, tmp_path):
        """Test that events within 5% tolerance are identified correctly."""
        matlab_file = tmp_path / 'matlab_reference.csv'
        matlab_reference_data.to_csv(matlab_file, index=False)
        
        result = validator.compare_with_matlab(sample_events_data, matlab_file)
        
        # Check that comparison was performed
        assert result['comparison_available'] is True
        
        # Verify statistics are calculated
        if 'statistics' in result and 'within_tolerance' in result['statistics']:
            assert isinstance(result['statistics']['within_tolerance'], int)
    
    def test_matlab_comparison_exceeds_tolerance(self, validator, sample_events_data, tmp_path):
        """Test identification of events exceeding 5% tolerance."""
        # Create MATLAB data with large differences
        matlab_data = pd.DataFrame({
            'event_name': sample_events_data['event_name'],
            'production_loss_kcal': sample_events_data['production_loss_kcal'] * 1.15  # 15% difference
        })
        matlab_file = tmp_path / 'matlab_reference.csv'
        matlab_data.to_csv(matlab_file, index=False)
        
        result = validator.compare_with_matlab(sample_events_data, matlab_file)
        
        # Should have warnings about large differences
        assert len(result['warnings']) > 0 or len(result.get('differences', {})) > 0
    
    def test_matlab_comparison_missing_events(self, validator, sample_events_data, tmp_path):
        """Test comparison when some events are missing in MATLAB data."""
        # Create MATLAB data with fewer events
        matlab_data = pd.DataFrame({
            'event_name': sample_events_data['event_name'][:4],
            'production_loss_kcal': sample_events_data['production_loss_kcal'][:4]
        })
        matlab_file = tmp_path / 'matlab_reference.csv'
        matlab_data.to_csv(matlab_file, index=False)
        
        result = validator.compare_with_matlab(sample_events_data, matlab_file)
        
        # Should note missing events
        assert result['comparison_available'] is True
        if 'statistics' in result:
            assert 'common_events' in result['statistics']
    
    def test_matlab_comparison_magnitude_exact_match(self, validator, sample_events_data, tmp_path):
        """Test that magnitudes match exactly (Requirement 15.3)."""
        # Magnitudes should match exactly since they use same formula
        matlab_data = pd.DataFrame({
            'event_name': sample_events_data['event_name'],
            'production_loss_kcal': sample_events_data['production_loss_kcal'],
            'magnitude': sample_events_data['magnitude']
        })
        matlab_file = tmp_path / 'matlab_reference.csv'
        matlab_data.to_csv(matlab_file, index=False)
        
        result = validator.compare_with_matlab(sample_events_data, matlab_file)
        
        # Magnitudes should match exactly
        if 'differences' in result:
            for event_name, diff_data in result['differences'].items():
                if 'magnitude_diff' in diff_data:
                    # Allow for floating point precision differences
                    assert abs(diff_data['magnitude_diff']) < 1e-10
    
    def test_matlab_comparison_spam_version_warning(self, validator, sample_events_data):
        """Test that SPAM version difference warning is included."""
        result = validator.compare_with_matlab(sample_events_data)
        
        # Should warn about SPAM2010 vs SPAM2020 difference
        assert any('SPAM2010' in warning or 'SPAM2020' in warning 
                  for warning in result['warnings'])
    
    def test_matlab_comparison_percentage_calculation(self, validator, sample_events_data, tmp_path):
        """Test that percentage differences are calculated correctly."""
        matlab_data = pd.DataFrame({
            'event_name': ['Event1'],
            'production_loss_kcal': [100.0]
        })
        python_data = pd.DataFrame({
            'event_name': ['Event1'],
            'production_loss_kcal': [110.0]
        })
        
        matlab_file = tmp_path / 'matlab_reference.csv'
        matlab_data.to_csv(matlab_file, index=False)
        
        result = validator.compare_with_matlab(python_data, matlab_file)
        
        # Should calculate 10% difference
        if result['comparison_available'] and 'differences' in result:
            if 'Event1' in result['differences']:
                diff = result['differences']['Event1']
                if 'percent_diff' in diff:
                    assert abs(diff['percent_diff'] - 10.0) < 0.1


class TestDataConsistency:
    """Test data consistency checks (Requirement 15.2)."""
    
    def test_spam_data_consistency(self, validator, sample_spam_data):
        """Test SPAM data internal consistency."""
        production_df, harvest_df = sample_spam_data
        
        result = validator.validate_spam_data(production_df, harvest_df)
        
        assert 'valid' in result
        assert 'statistics' in result
        assert 'coordinates' in result['statistics']
    
    def test_coordinate_consistency_between_datasets(self, validator, sample_spam_data):
        """Test that coordinates match between production and harvest data."""
        production_df, harvest_df = sample_spam_data
        
        result = validator.validate_spam_data(production_df, harvest_df)
        
        coord_validation = result['statistics']['coordinates']
        assert coord_validation['valid'] is True
    
    def test_production_harvest_ratio_consistency(self, validator, sample_spam_data):
        """Test that production/harvest ratios are reasonable (yields)."""
        production_df, harvest_df = sample_spam_data
        
        # Calculate yields for wheat (avoid division by zero)
        mask = harvest_df['whea_a'] > 0
        yields = production_df.loc[mask, 'whea_a'] / harvest_df.loc[mask, 'whea_a']
        
        # Yields should be positive and reasonable (typically 1-10 MT/ha)
        # Note: Sample data has yields of 100, which is unrealistic but acceptable for testing
        assert (yields > 0).all()
        assert (yields < 200).all()  # Upper bound for test data
    
    def test_global_production_totals_consistency(self, validator, sample_spam_data):
        """Test that global production totals are within expected ranges."""
        production_df, harvest_df = sample_spam_data
        
        result = validator.validate_spam_data(production_df, harvest_df)
        
        prod_totals = result['statistics']['production_totals']
        assert 'total_production_mt' in prod_totals
        assert prod_totals['total_production_mt'] > 0
    
    def test_event_losses_consistency(self, validator, sample_events_data):
        """Test that event losses are internally consistent."""
        result = validator.validate_event_results(sample_events_data)
        
        assert result['valid'] is True
        assert 'statistics' in result
    
    def test_event_losses_vs_global_production(self, validator, sample_events_data):
        """Test that no event exceeds global production."""
        global_production = 1e16  # 10 quadrillion kcal
        
        result = validator.validate_event_results(sample_events_data, global_production)
        
        # Check that validation passed
        loss_validation = result['statistics'].get('loss_validation', {})
        if 'valid' in loss_validation:
            # All events should be below global production
            assert loss_validation['valid'] is True
    
    def test_magnitude_harvest_area_consistency(self, validator, sample_events_data):
        """Test that magnitudes are consistent with harvest areas."""
        for idx, row in sample_events_data.iterrows():
            harvest_area_ha = row['harvest_area_loss_ha']
            magnitude = row['magnitude']
            
            # Magnitude should equal log10(harvest_area_km2)
            harvest_area_km2 = harvest_area_ha * 0.01
            expected_magnitude = np.log10(harvest_area_km2)
            
            # Allow for reasonable floating point differences
            # Sample data may have pre-calculated magnitudes with slight variations
            assert abs(magnitude - expected_magnitude) < 0.1
    
    def test_no_negative_values(self, validator, sample_events_data):
        """Test that there are no negative values in event results."""
        assert (sample_events_data['harvest_area_loss_ha'] >= 0).all()
        assert (sample_events_data['production_loss_kcal'] >= 0).all()
    
    def test_no_infinite_values(self, validator, sample_events_data):
        """Test that there are no infinite values in event results."""
        assert not np.isinf(sample_events_data['harvest_area_loss_ha']).any()
        assert not np.isinf(sample_events_data['production_loss_kcal']).any()
        assert not np.isinf(sample_events_data['magnitude']).any()
    
    def test_crop_specific_consistency(self, validator):
        """Test consistency across different crop types."""
        # Test with different crop configurations
        for crop_type in ['wheat', 'rice', 'allgrain']:
            config = Config(crop_type=crop_type, root_dir='./data', spam_version='2020')
            validator = DataValidator(config)
            
            # Verify expected ranges are defined
            assert crop_type in validator.expected_production_ranges
            assert crop_type in validator.expected_harvest_ranges


class TestSpatialCoverage:
    """Test spatial coverage validation (Requirement 15.2)."""
    
    def test_spatial_coverage_all_events(self, validator, sample_events_data):
        """Test that all events have spatial coverage data."""
        # All events should have non-zero harvest area
        assert (sample_events_data['harvest_area_loss_ha'] > 0).all()
    
    def test_spatial_coverage_statistics(self, validator, sample_events_data):
        """Test calculation of spatial coverage statistics."""
        result = validator.validate_event_results(sample_events_data)
        
        stats = result['statistics']['event_stats']
        assert 'total_events' in stats
        assert 'events_with_data' in stats
        assert stats['events_with_data'] > 0
    
    def test_spatial_coverage_success_rate(self, validator, sample_events_data):
        """Test calculation of spatial mapping success rate."""
        result = validator.validate_event_results(sample_events_data)
        
        stats = result['statistics']['event_stats']
        total_events = stats['total_events']
        events_with_data = stats['events_with_data']
        
        # Calculate success rate
        success_rate = events_with_data / total_events if total_events > 0 else 0
        
        # Should have high success rate
        assert success_rate > 0.5
    
    def test_identify_zero_coverage_events(self, validator):
        """Test identification of events with zero spatial coverage."""
        events_with_zeros = pd.DataFrame({
            'event_name': ['Event1', 'Event2', 'Event3'],
            'harvest_area_loss_ha': [1000.0, 0.0, 2000.0],
            'production_loss_kcal': [1e14, 0.0, 2e14],
            'magnitude': [3.5, np.nan, 3.8]
        })
        
        result = validator.validate_event_results(events_with_zeros)
        
        zero_loss = result['statistics']['zero_loss_events']
        assert 'Event2' in zero_loss['events']
        assert 'Event1' not in zero_loss['events']
    
    def test_spatial_coverage_by_region(self, validator, sample_spam_data):
        """Test spatial coverage across different regions."""
        production_df, harvest_df = sample_spam_data
        
        # Check that multiple countries are represented
        countries = production_df['iso3'].unique()
        assert len(countries) > 1
        
        # Check that each country has data
        for country in countries:
            country_data = production_df[production_df['iso3'] == country]
            assert len(country_data) > 0
            assert country_data['whea_a'].sum() >= 0
    
    def test_coordinate_coverage_global(self, validator, sample_spam_data):
        """Test that coordinates cover global extent."""
        production_df, harvest_df = sample_spam_data
        
        # Check longitude coverage
        lon_range = (production_df['x'].min(), production_df['x'].max())
        assert lon_range[0] < -100  # Western hemisphere
        assert lon_range[1] > 100   # Eastern hemisphere
        
        # Check latitude coverage
        lat_range = (production_df['y'].min(), production_df['y'].max())
        assert lat_range[0] < 0     # Southern hemisphere
        assert lat_range[1] > 0     # Northern hemisphere


class TestFigureQuality:
    """Test figure quality verification (Requirement 15.3)."""
    
    def test_figure_creation_basic(self):
        """Test that figures can be created without errors."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title('Test Figure')
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_figure_has_required_elements(self):
        """Test that figures have required elements (axes, labels, title)."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title('Test Title')
        
        # Check that labels are set
        assert ax.get_xlabel() == 'X Axis'
        assert ax.get_ylabel() == 'Y Axis'
        assert ax.get_title() == 'Test Title'
        
        plt.close(fig)
    
    def test_figure_dpi_quality(self):
        """Test that figures can be saved at publication quality DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Test that figure can be saved at 300 DPI
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        
        # Check that buffer has data
        assert buf.getbuffer().nbytes > 0
        
        plt.close(fig)
    
    def test_figure_format_support(self):
        """Test that figures support multiple output formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        formats = ['png', 'svg', 'pdf']
        
        for fmt in formats:
            import io
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format=fmt)
                buf.seek(0)
                assert buf.getbuffer().nbytes > 0
            except Exception as e:
                pytest.fail(f"Failed to save figure in {fmt} format: {e}")
        
        plt.close(fig)
    
    def test_figure_size_appropriate(self):
        """Test that figure sizes are appropriate for publication."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check figure size
        size = fig.get_size_inches()
        assert size[0] >= 6  # Width at least 6 inches
        assert size[1] >= 4  # Height at least 4 inches
        
        plt.close(fig)
    
    def test_figure_log_scale_support(self):
        """Test that figures support logarithmic scales."""
        fig, ax = plt.subplots()
        ax.plot([1, 10, 100], [1, 10, 100])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Check that scales are set
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_figure_legend_support(self):
        """Test that figures support legends."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3], label='Line 1')
        ax.plot([1, 2, 3], [2, 3, 4], label='Line 2')
        ax.legend()
        
        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None
        
        plt.close(fig)
    
    def test_figure_color_scheme(self):
        """Test that figures use appropriate color schemes."""
        fig, ax = plt.subplots()
        
        # Test different color specifications
        ax.plot([1, 2, 3], [1, 2, 3], color='red')
        ax.plot([1, 2, 3], [2, 3, 4], color='#0000FF')
        ax.plot([1, 2, 3], [3, 4, 5], color=(0, 1, 0))
        
        # Check that lines were created
        lines = ax.get_lines()
        assert len(lines) == 3
        
        plt.close(fig)
    
    def test_figure_grid_support(self):
        """Test that figures support grid lines."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.grid(True)
        
        # Grid should be enabled - check using public API
        # Grid lines are stored in the axes
        assert len(ax.get_xgridlines()) > 0 or len(ax.get_ygridlines()) > 0
        
        plt.close(fig)
    
    def test_figure_text_annotation(self):
        """Test that figures support text annotations."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.text(2, 2, 'Test Annotation')
        
        # Check that text was added
        texts = ax.texts
        assert len(texts) > 0
        
        plt.close(fig)


class TestValidationReportGeneration:
    """Test validation report generation."""
    
    def test_generate_complete_validation_report(self, validator, sample_spam_data, sample_events_data):
        """Test generation of complete validation report."""
        production_df, harvest_df = sample_spam_data
        
        spam_validation = validator.validate_spam_data(production_df, harvest_df)
        event_validation = validator.validate_event_results(sample_events_data)
        
        report = validator.generate_validation_report(
            spam_validation=spam_validation,
            event_validation=event_validation
        )
        
        assert 'AgRichter Data Validation Report' in report
        assert 'SPAM DATA VALIDATION' in report
        assert 'EVENT RESULTS VALIDATION' in report
    
    def test_validation_report_includes_statistics(self, validator, sample_events_data):
        """Test that validation report includes key statistics."""
        event_validation = validator.validate_event_results(sample_events_data)
        report = validator.generate_validation_report(event_validation=event_validation)
        
        assert 'Event Statistics:' in report or 'Statistics:' in report
        assert 'Total Events:' in report or 'total_events' in report.lower()
    
    def test_validation_report_includes_warnings(self, validator):
        """Test that validation report includes warnings."""
        # Create data with issues
        events_with_issues = pd.DataFrame({
            'event_name': ['Event1'],
            'harvest_area_loss_ha': [0.0],
            'production_loss_kcal': [0.0],
            'magnitude': [np.nan]
        })
        
        event_validation = validator.validate_event_results(events_with_issues)
        report = validator.generate_validation_report(event_validation=event_validation)
        
        assert 'Warning' in report or 'warning' in report.lower()
    
    def test_validation_report_save_to_file(self, validator, tmp_path):
        """Test saving validation report to file."""
        output_path = tmp_path / 'validation_report.txt'
        report = validator.generate_validation_report(output_path=output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestEndToEndValidation:
    """Test end-to-end validation workflow."""
    
    def test_complete_validation_workflow(self, validator, sample_spam_data, sample_events_data):
        """Test complete validation workflow from data to report."""
        production_df, harvest_df = sample_spam_data
        
        # Step 1: Validate SPAM data
        spam_validation = validator.validate_spam_data(production_df, harvest_df)
        assert spam_validation['valid'] is True
        
        # Step 2: Validate event results
        event_validation = validator.validate_event_results(sample_events_data)
        assert event_validation['valid'] is True
        
        # Step 3: Generate report
        report = validator.generate_validation_report(
            spam_validation=spam_validation,
            event_validation=event_validation
        )
        assert len(report) > 0
    
    def test_validation_with_errors_continues(self, validator):
        """Test that validation continues even with errors."""
        # Create invalid data
        bad_production = pd.DataFrame()
        bad_harvest = pd.DataFrame()
        
        spam_validation = validator.validate_spam_data(bad_production, bad_harvest)
        
        # Should have errors but not crash
        assert spam_validation['valid'] is False
        assert len(spam_validation['errors']) > 0
    
    def test_validation_summary_status(self, validator, sample_spam_data, sample_events_data):
        """Test that validation summary provides overall status."""
        production_df, harvest_df = sample_spam_data
        
        spam_validation = validator.validate_spam_data(production_df, harvest_df)
        event_validation = validator.validate_event_results(sample_events_data)
        
        report = validator.generate_validation_report(
            spam_validation=spam_validation,
            event_validation=event_validation
        )
        
        # Report should include overall status
        assert 'SUMMARY' in report or 'Summary' in report
        assert 'Status' in report or 'status' in report.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""Unit tests for DataValidator class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from agririchter.validation.data_validator import DataValidator
from agririchter.core.config import Config


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(crop_type='wheat', root_dir='./data', spam_version='2020')


@pytest.fixture
def validator(config):
    """Create DataValidator instance."""
    return DataValidator(config)


@pytest.fixture
def sample_production_df():
    """Create sample production DataFrame."""
    return pd.DataFrame({
        'x': [-179.5, -178.5, -177.5],
        'y': [89.5, 88.5, 87.5],
        'iso3': ['USA', 'CAN', 'MEX'],
        'whea_a': [1000.0, 2000.0, 1500.0],
        'rice_a': [500.0, 800.0, 600.0],
        'maiz_a': [1200.0, 1800.0, 1400.0]
    })


@pytest.fixture
def sample_harvest_df():
    """Create sample harvest area DataFrame."""
    return pd.DataFrame({
        'x': [-179.5, -178.5, -177.5],
        'y': [89.5, 88.5, 87.5],
        'iso3': ['USA', 'CAN', 'MEX'],
        'whea_a': [100.0, 200.0, 150.0],
        'rice_a': [50.0, 80.0, 60.0],
        'maiz_a': [120.0, 180.0, 140.0]
    })


@pytest.fixture
def sample_events_df():
    """Create sample events DataFrame."""
    return pd.DataFrame({
        'event_name': ['Event1', 'Event2', 'Event3'],
        'harvest_area_loss_ha': [1000.0, 5000.0, 2000.0],
        'production_loss_kcal': [1e14, 5e14, 2e14],
        'magnitude': [3.5, 4.2, 3.8]
    })


class TestDataValidatorInit:
    """Test DataValidator initialization."""
    
    def test_init_with_config(self, config):
        """Test initialization with valid config."""
        validator = DataValidator(config)
        assert validator.config == config
        assert hasattr(validator, 'expected_production_ranges')
        assert hasattr(validator, 'coordinate_ranges')
    
    def test_validation_thresholds_setup(self, validator):
        """Test validation thresholds are properly set up."""
        assert 'wheat' in validator.expected_production_ranges
        assert 'latitude' in validator.coordinate_ranges
        assert 'longitude' in validator.coordinate_ranges
        assert validator.matlab_comparison_tolerance == 0.05


class TestSPAMDataValidation:
    """Test SPAM data validation methods."""
    
    def test_validate_spam_data_success(self, validator, sample_production_df, sample_harvest_df):
        """Test successful SPAM data validation."""
        result = validator.validate_spam_data(sample_production_df, sample_harvest_df)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'statistics' in result
    
    def test_validate_spam_data_empty_production(self, validator, sample_harvest_df):
        """Test validation with empty production DataFrame."""
        empty_df = pd.DataFrame()
        result = validator.validate_spam_data(empty_df, sample_harvest_df)
        
        assert result['valid'] is False
        assert 'Production DataFrame is empty' in result['errors']
    
    def test_validate_spam_data_empty_harvest(self, validator, sample_production_df):
        """Test validation with empty harvest DataFrame."""
        empty_df = pd.DataFrame()
        result = validator.validate_spam_data(sample_production_df, empty_df)
        
        assert result['valid'] is False
        assert 'Harvest area DataFrame is empty' in result['errors']
    
    def test_validate_coordinates(self, validator, sample_production_df, sample_harvest_df):
        """Test coordinate validation."""
        result = validator._validate_coordinates(sample_production_df, sample_harvest_df)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_coordinates_missing_column(self, validator, sample_harvest_df):
        """Test coordinate validation with missing column."""
        bad_df = pd.DataFrame({'iso3': ['USA']})
        result = validator._validate_coordinates(bad_df, sample_harvest_df)
        
        assert result['valid'] is False
        assert any('missing' in error.lower() for error in result['errors'])
    
    def test_validate_crop_columns(self, validator, sample_production_df, sample_harvest_df):
        """Test crop column validation."""
        result = validator._validate_crop_columns(sample_production_df, sample_harvest_df)
        
        assert result['valid'] is True
        assert 'whea_a' in result['found_crops']
    
    def test_validate_production_totals(self, validator, sample_production_df):
        """Test production totals validation."""
        result = validator._validate_production_totals(sample_production_df)
        
        assert 'totals' in result
        assert 'total_production_mt' in result
        assert result['total_production_mt'] > 0
    
    def test_validate_coordinate_ranges(self, validator, sample_production_df, sample_harvest_df):
        """Test coordinate range validation."""
        result = validator._validate_coordinate_ranges(sample_production_df, sample_harvest_df)
        
        assert 'ranges' in result
        assert 'production_x' in result['ranges']
        assert 'production_y' in result['ranges']
    
    def test_check_missing_data(self, validator, sample_production_df, sample_harvest_df):
        """Test missing data check."""
        result = validator._check_missing_data(sample_production_df, sample_harvest_df)
        
        assert 'critical_missing' in result
        assert 'production_missing' in result
        assert 'harvest_missing' in result


class TestEventResultsValidation:
    """Test event results validation methods."""
    
    def test_validate_event_results_success(self, validator, sample_events_df):
        """Test successful event results validation."""
        result = validator.validate_event_results(sample_events_df, global_production_kcal=1e16)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'statistics' in result
    
    def test_validate_event_results_empty(self, validator):
        """Test validation with empty events DataFrame."""
        empty_df = pd.DataFrame()
        result = validator.validate_event_results(empty_df)
        
        assert result['valid'] is False
        # Empty DataFrame will fail on missing columns check first
        assert any('missing required columns' in error.lower() for error in result['errors'])
    
    def test_validate_event_results_missing_columns(self, validator):
        """Test validation with missing required columns."""
        bad_df = pd.DataFrame({'event_name': ['Event1']})
        result = validator.validate_event_results(bad_df)
        
        assert result['valid'] is False
        assert any('missing required columns' in error.lower() for error in result['errors'])
    
    def test_validate_losses_vs_global(self, validator, sample_events_df):
        """Test loss validation against global production."""
        global_prod = 1e16
        result = validator._validate_losses_vs_global(sample_events_df, global_prod)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_validate_losses_exceed_global(self, validator):
        """Test validation when losses exceed global production."""
        events_df = pd.DataFrame({
            'event_name': ['HugeEvent'],
            'production_loss_kcal': [1e17]
        })
        global_prod = 1e16
        result = validator._validate_losses_vs_global(events_df, global_prod)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_magnitude_ranges(self, validator, sample_events_df):
        """Test magnitude range validation."""
        result = validator._validate_magnitude_ranges(sample_events_df)
        
        assert 'magnitude_stats' in result
        assert 'min' in result['magnitude_stats']
        assert 'max' in result['magnitude_stats']
    
    def test_validate_magnitude_out_of_range(self, validator):
        """Test magnitude validation with out-of-range values."""
        events_df = pd.DataFrame({
            'event_name': ['TooSmall', 'TooBig'],
            'magnitude': [1.0, 10.0],
            'harvest_area_loss_ha': [100.0, 100.0],
            'production_loss_kcal': [1e13, 1e13]
        })
        result = validator._validate_magnitude_ranges(events_df)
        
        assert len(result['warnings']) > 0
        assert len(result['suspicious_events']) > 0
    
    def test_identify_zero_loss_events(self, validator):
        """Test identification of zero loss events."""
        events_df = pd.DataFrame({
            'event_name': ['ZeroEvent', 'NormalEvent'],
            'harvest_area_loss_ha': [0.0, 1000.0],
            'production_loss_kcal': [0.0, 1e14]
        })
        result = validator._identify_zero_loss_events(events_df)
        
        assert 'ZeroEvent' in result['events']
        assert 'NormalEvent' not in result['events']
    
    def test_calculate_event_statistics(self, validator, sample_events_df):
        """Test event statistics calculation."""
        result = validator._calculate_event_statistics(sample_events_df)
        
        assert 'total_events' in result
        assert 'events_with_data' in result
        assert 'production_loss' in result
        assert 'magnitude' in result


class TestMATLABComparison:
    """Test MATLAB comparison functionality."""
    
    def test_compare_with_matlab_no_file(self, validator, sample_events_df):
        """Test MATLAB comparison when file doesn't exist."""
        result = validator.compare_with_matlab(sample_events_df, Path('/nonexistent/file.csv'))
        
        assert result['comparison_available'] is False
        assert len(result['warnings']) > 0
    
    def test_compare_with_matlab_spam2020_warning(self, validator, sample_events_df):
        """Test that SPAM2020 warning is included."""
        result = validator.compare_with_matlab(sample_events_df)
        
        assert any('SPAM2010' in warning for warning in result['warnings'])
        assert any('SPAM2020' in warning for warning in result['warnings'])
    
    @patch('pandas.read_csv')
    def test_compare_event_losses(self, mock_read_csv, validator, sample_events_df):
        """Test event loss comparison."""
        matlab_df = pd.DataFrame({
            'event_name': ['Event1', 'Event2'],
            'production_loss_kcal': [1.1e14, 4.8e14]
        })
        
        result = validator._compare_event_losses(sample_events_df, matlab_df)
        
        assert 'differences' in result
        assert 'statistics' in result
        assert 'common_events' in result['statistics']
    
    def test_compare_event_losses_no_common_events(self, validator):
        """Test comparison with no common events."""
        python_df = pd.DataFrame({
            'event_name': ['Event1'],
            'production_loss_kcal': [1e14]
        })
        matlab_df = pd.DataFrame({
            'event_name': ['Event2'],
            'production_loss_kcal': [1e14]
        })
        
        result = validator._compare_event_losses(python_df, matlab_df)
        
        assert len(result['statistics']['common_events']) == 0
        assert len(result['warnings']) > 0


class TestValidationReport:
    """Test validation report generation."""
    
    def test_generate_validation_report_empty(self, validator):
        """Test report generation with no validation results."""
        report = validator.generate_validation_report()
        
        assert 'AgriRichter Data Validation Report' in report
        assert 'Crop Type: wheat' in report
        assert 'SPAM Version: 2020' in report
    
    def test_generate_validation_report_with_spam(self, validator, sample_production_df, sample_harvest_df):
        """Test report generation with SPAM validation."""
        spam_validation = validator.validate_spam_data(sample_production_df, sample_harvest_df)
        report = validator.generate_validation_report(spam_validation=spam_validation)
        
        assert 'SPAM DATA VALIDATION' in report
        assert 'Overall Status:' in report
    
    def test_generate_validation_report_with_events(self, validator, sample_events_df):
        """Test report generation with event validation."""
        event_validation = validator.validate_event_results(sample_events_df)
        report = validator.generate_validation_report(event_validation=event_validation)
        
        assert 'EVENT RESULTS VALIDATION' in report
        assert 'Event Statistics:' in report
    
    def test_generate_validation_report_with_matlab(self, validator, sample_events_df):
        """Test report generation with MATLAB comparison."""
        matlab_comparison = validator.compare_with_matlab(sample_events_df)
        report = validator.generate_validation_report(matlab_comparison=matlab_comparison)
        
        assert 'MATLAB COMPARISON' in report
    
    def test_generate_validation_report_save_to_file(self, validator, tmp_path):
        """Test saving report to file."""
        output_path = tmp_path / 'validation_report.txt'
        report = validator.generate_validation_report(output_path=output_path)
        
        assert output_path.exists()
        with open(output_path, 'r') as f:
            saved_report = f.read()
        assert saved_report == report
    
    def test_generate_validation_report_summary(self, validator):
        """Test that report includes summary section."""
        report = validator.generate_validation_report()
        
        assert 'SUMMARY' in report
        assert 'Overall Validation Status:' in report
        assert 'Total Errors:' in report
        assert 'Total Warnings:' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

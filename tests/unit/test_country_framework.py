"""
Unit tests for the Country Framework module.

Tests the scalable framework for adding new countries to the multi-tier envelope system,
including template management, validation procedures, and configuration creation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from agririchter.core.config import Config
from agririchter.data.country_framework import CountryFramework, CountryTemplate
from agririchter.data.country_boundary_manager import CountryConfiguration, CountryBoundaryManager


class TestCountryTemplate(unittest.TestCase):
    """Test CountryTemplate dataclass."""
    
    def test_template_creation(self):
        """Test basic template creation."""
        template = CountryTemplate(
            country_code='TEST',
            country_name='Test Country',
            fips_code='TS',
            iso3_code='TST',
            agricultural_focus='food_security',
            priority_crops=['wheat', 'maize']
        )
        
        self.assertEqual(template.country_code, 'TEST')
        self.assertEqual(template.country_name, 'Test Country')
        self.assertEqual(template.fips_code, 'TS')
        self.assertEqual(template.agricultural_focus, 'food_security')
        self.assertEqual(template.priority_crops, ['wheat', 'maize'])
        self.assertIsNotNone(template.created_date)
        self.assertEqual(template.template_version, '1.0')
    
    def test_template_with_optional_fields(self):
        """Test template creation with optional fields."""
        template = CountryTemplate(
            country_code='TEST',
            country_name='Test Country',
            fips_code='TS',
            iso3_code='TST',
            agricultural_focus='export_capacity',
            priority_crops=['wheat', 'maize', 'rice'],
            regional_subdivisions=['north', 'south'],
            policy_scenarios=['drought', 'trade'],
            continent='Test Continent',
            agricultural_area_km2=100000,
            min_cells_required=500,
            min_crop_coverage_percent=2.0
        )
        
        self.assertEqual(template.regional_subdivisions, ['north', 'south'])
        self.assertEqual(template.policy_scenarios, ['drought', 'trade'])
        self.assertEqual(template.continent, 'Test Continent')
        self.assertEqual(template.agricultural_area_km2, 100000)
        self.assertEqual(template.min_cells_required, 500)
        self.assertEqual(template.min_crop_coverage_percent, 2.0)


class TestCountryFramework(unittest.TestCase):
    """Test CountryFramework class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock(spec=Config)
        self.config.output_dir = self.temp_dir
        
        self.boundary_manager = Mock(spec=CountryBoundaryManager)
        self.framework = CountryFramework(self.config, self.boundary_manager)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.framework.config, self.config)
        self.assertEqual(self.framework.boundary_manager, self.boundary_manager)
        self.assertTrue(self.framework.templates_dir.exists())
        self.assertGreater(len(self.framework.COUNTRY_TEMPLATES), 0)
    
    def test_get_available_templates(self):
        """Test getting available templates."""
        available = self.framework.get_available_templates()
        
        self.assertIsInstance(available, list)
        self.assertIn('BRA', available)
        self.assertIn('IND', available)
        self.assertIn('RUS', available)
        self.assertGreater(len(available), 5)
    
    def test_get_country_template(self):
        """Test getting specific country templates."""
        # Test existing template
        template = self.framework.get_country_template('BRA')
        self.assertIsInstance(template, CountryTemplate)
        self.assertEqual(template.country_code, 'BRA')
        self.assertEqual(template.country_name, 'Brazil')
        self.assertEqual(template.fips_code, 'BR')
        
        # Test case insensitive
        template_lower = self.framework.get_country_template('bra')
        self.assertEqual(template_lower.country_code, 'BRA')
        
        # Test non-existent template
        template_none = self.framework.get_country_template('XXX')
        self.assertIsNone(template_none)
    
    def test_create_country_configuration(self):
        """Test creating country configuration from template."""
        template = self.framework.get_country_template('BRA')
        
        # Test without validation
        config, validation = self.framework.create_country_configuration(
            template, validate_data=False
        )
        
        self.assertIsInstance(config, CountryConfiguration)
        self.assertEqual(config.country_code, 'BRA')
        self.assertEqual(config.country_name, 'Brazil')
        self.assertEqual(config.fips_code, 'BR')
        self.assertEqual(config.agricultural_focus, 'export_capacity')
        self.assertEqual(validation['template_used'], 'BRA')
    
    def test_create_custom_template(self):
        """Test creating custom templates."""
        custom_template = self.framework.create_custom_template(
            country_code='NGA',
            country_name='Nigeria',
            fips_code='NI',
            iso3_code='NGA',
            agricultural_focus='food_security',
            priority_crops=['wheat', 'maize', 'rice']
        )
        
        self.assertEqual(custom_template.country_code, 'NGA')
        self.assertEqual(custom_template.country_name, 'Nigeria')
        self.assertEqual(custom_template.fips_code, 'NI')
        self.assertEqual(custom_template.iso3_code, 'NGA')
        self.assertEqual(custom_template.agricultural_focus, 'food_security')
        self.assertEqual(custom_template.priority_crops, ['wheat', 'maize', 'rice'])
        self.assertFalse(custom_template.validated)
    
    def test_save_and_load_template(self):
        """Test saving and loading templates."""
        # Create custom template
        template = self.framework.create_custom_template(
            country_code='TEST',
            country_name='Test Country',
            fips_code='TS',
            iso3_code='TST'
        )
        
        # Save template
        saved_path = self.framework.save_template(template)
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.name, 'test_template.yaml')
        
        # Load template
        loaded_template = self.framework.load_template(saved_path)
        self.assertEqual(loaded_template.country_code, 'TEST')
        self.assertEqual(loaded_template.country_name, 'Test Country')
        self.assertEqual(loaded_template.fips_code, 'TS')
    
    def test_validation_methods_structure(self):
        """Test that validation methods have correct structure."""
        template = self.framework.get_country_template('BRA')
        
        # Test validation method signatures exist
        self.assertTrue(hasattr(self.framework, 'validate_country_template'))
        self.assertTrue(hasattr(self.framework, '_check_spam_data_availability'))
        self.assertTrue(hasattr(self.framework, '_check_data_coverage'))
        self.assertTrue(hasattr(self.framework, '_check_geographic_extent'))
        self.assertTrue(hasattr(self.framework, '_check_crop_coverage'))
        self.assertTrue(hasattr(self.framework, '_check_configuration_consistency'))
        self.assertTrue(hasattr(self.framework, '_generate_recommendations'))
    
    def test_configuration_consistency_check(self):
        """Test configuration consistency validation."""
        template = self.framework.get_country_template('BRA')
        
        # This should pass for valid template
        consistency_result = self.framework._check_configuration_consistency(template)
        
        self.assertIsInstance(consistency_result, dict)
        self.assertIn('passed', consistency_result)
        self.assertIn('checks', consistency_result)
        self.assertIn('message', consistency_result)
        
        # Test with invalid template
        invalid_template = CountryTemplate(
            country_code='INVALID',
            country_name='Invalid Country',
            fips_code='IV',
            iso3_code='INV',
            agricultural_focus='invalid_focus',  # Invalid
            priority_crops=['invalid_crop'],     # Invalid
            min_cells_required=-100,             # Invalid
            min_crop_coverage_percent=200        # Invalid
        )
        
        consistency_result = self.framework._check_configuration_consistency(invalid_template)
        self.assertFalse(consistency_result['passed'])
    
    def test_batch_validate_templates_structure(self):
        """Test batch validation structure."""
        # Test with subset of countries
        selected_countries = ['BRA', 'IND']
        
        # Mock the validation method to avoid SPAM data dependency
        with patch.object(self.framework, 'validate_country_template') as mock_validate:
            mock_validate.return_value = {
                'overall_status': 'passed',
                'checks': {},
                'errors': []
            }
            
            batch_results = self.framework.batch_validate_templates(selected_countries)
            
            self.assertIsInstance(batch_results, dict)
            self.assertEqual(len(batch_results), 2)
            self.assertIn('BRA', batch_results)
            self.assertIn('IND', batch_results)
            
            # Verify validation was called for each country
            self.assertEqual(mock_validate.call_count, 2)


class TestCountryFrameworkValidation(unittest.TestCase):
    """Test validation functionality with mocked data."""
    
    def setUp(self):
        """Set up test fixtures with mocked boundary manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock(spec=Config)
        self.config.output_dir = self.temp_dir
        
        # Create mock boundary manager with realistic responses
        self.boundary_manager = Mock(spec=CountryBoundaryManager)
        self.framework = CountryFramework(self.config, self.boundary_manager)
        
        # Mock successful data responses
        self.mock_production_df = pd.DataFrame({
            'x': np.random.uniform(-60, -40, 1000),
            'y': np.random.uniform(-30, 5, 1000),
            'wheat_a': np.random.uniform(0, 1000, 1000),
            'maize_a': np.random.uniform(0, 2000, 1000),
            'rice_a': np.random.uniform(0, 500, 1000)
        })
        
        self.mock_harvest_df = pd.DataFrame({
            'x': np.random.uniform(-60, -40, 1000),
            'y': np.random.uniform(-30, 5, 1000),
            'wheat_a': np.random.uniform(0, 100, 1000),
            'maize_a': np.random.uniform(0, 200, 1000),
            'rice_a': np.random.uniform(0, 50, 1000)
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_spam_data_availability_check_success(self):
        """Test successful SPAM data availability check."""
        template = self.framework.get_country_template('BRA')
        
        # Mock successful data retrieval
        self.boundary_manager.get_country_data.return_value = (
            self.mock_production_df, self.mock_harvest_df
        )
        
        result = self.framework._check_spam_data_availability(template)
        
        self.assertTrue(result['passed'])
        self.assertEqual(result['production_cells'], 1000)
        self.assertEqual(result['harvest_area_cells'], 1000)
        self.assertEqual(result['fips_code_found'], 'BR')
        self.assertIn('SPAM data found', result['message'])
    
    def test_spam_data_availability_check_failure(self):
        """Test failed SPAM data availability check."""
        template = self.framework.get_country_template('BRA')
        
        # Mock failed data retrieval
        self.boundary_manager.get_country_data.side_effect = ValueError("No data found")
        
        result = self.framework._check_spam_data_availability(template)
        
        self.assertFalse(result['passed'])
        self.assertIn('error', result)
        self.assertEqual(result['fips_code_tested'], 'BR')
        self.assertIn('No SPAM data found', result['message'])
    
    def test_data_coverage_check_success(self):
        """Test successful data coverage check."""
        template = self.framework.get_country_template('BRA')
        
        # Mock successful coverage validation
        self.boundary_manager.validate_country_data_coverage.return_value = {
            'meets_minimum': True,
            'production_cells': 2500,
            'warnings': []
        }
        
        result = self.framework._check_data_coverage(template)
        
        self.assertTrue(result['passed'])
        self.assertEqual(result['cells_found'], 2500)
        self.assertEqual(result['cells_required'], template.min_cells_required)
        self.assertGreater(result['coverage_ratio'], 1.0)
    
    def test_data_coverage_check_failure(self):
        """Test failed data coverage check."""
        template = self.framework.get_country_template('BRA')
        
        # Mock insufficient coverage
        self.boundary_manager.validate_country_data_coverage.return_value = {
            'meets_minimum': False,
            'production_cells': 500,
            'warnings': ['Insufficient coverage']
        }
        
        result = self.framework._check_data_coverage(template)
        
        self.assertFalse(result['passed'])
        self.assertEqual(result['cells_found'], 500)
        self.assertLess(result['coverage_ratio'], 1.0)
    
    def test_geographic_extent_check(self):
        """Test geographic extent validation."""
        template = self.framework.get_country_template('BRA')
        
        # Mock geographic statistics
        self.boundary_manager.get_country_statistics.return_value = {
            'geographic_extent': {
                'lat_range': 35.0,
                'lon_range': 20.0,
                'lat_min': -30.0,
                'lat_max': 5.0,
                'lon_min': -60.0,
                'lon_max': -40.0
            }
        }
        
        result = self.framework._check_geographic_extent(template)
        
        self.assertTrue(result['passed'])
        self.assertEqual(result['lat_range'], 35.0)
        self.assertEqual(result['lon_range'], 20.0)
        self.assertGreater(result['actual_extent_km2'], 0)
    
    def test_crop_coverage_check(self):
        """Test crop coverage validation."""
        template = self.framework.get_country_template('BRA')
        
        # Mock crop coverage data
        self.boundary_manager.validate_country_data_coverage.return_value = {
            'crop_coverage': {
                'maize': {'coverage_percent': 75.0, 'cells_with_data': 750},
                'wheat': {'coverage_percent': 30.0, 'cells_with_data': 300},
                'rice': {'coverage_percent': 15.0, 'cells_with_data': 150}
            }
        }
        
        result = self.framework._check_crop_coverage(template)
        
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['crop_results']), 3)
        
        # Check individual crop results
        for crop in template.priority_crops:
            self.assertIn(crop, result['crop_results'])
            crop_result = result['crop_results'][crop]
            self.assertIn('coverage_percent', crop_result)
            self.assertIn('meets_minimum', crop_result)
    
    def test_full_validation_success(self):
        """Test complete validation with successful results."""
        template = self.framework.get_country_template('BRA')
        
        # Mock all validation components to succeed
        self.boundary_manager.get_country_data.return_value = (
            self.mock_production_df, self.mock_harvest_df
        )
        
        self.boundary_manager.validate_country_data_coverage.return_value = {
            'meets_minimum': True,
            'production_cells': 2500,
            'crop_coverage': {
                'maize': {'coverage_percent': 75.0, 'cells_with_data': 750},
                'wheat': {'coverage_percent': 30.0, 'cells_with_data': 300},
                'rice': {'coverage_percent': 15.0, 'cells_with_data': 150}
            }
        }
        
        self.boundary_manager.get_country_statistics.return_value = {
            'geographic_extent': {
                'lat_range': 35.0,
                'lon_range': 20.0
            }
        }
        
        validation_results = self.framework.validate_country_template(template)
        
        self.assertEqual(validation_results['overall_status'], 'passed')
        self.assertEqual(validation_results['country_code'], 'BRA')
        self.assertIn('checks', validation_results)
        self.assertIn('recommendations', validation_results)
        
        # All checks should pass
        for check_name, check_result in validation_results['checks'].items():
            self.assertTrue(check_result.get('passed', False), 
                          f"Check {check_name} should pass")


class TestCountryFrameworkIntegration(unittest.TestCase):
    """Test integration aspects of the country framework."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock(spec=Config)
        self.config.output_dir = self.temp_dir
        
        self.boundary_manager = Mock(spec=CountryBoundaryManager)
        self.framework = CountryFramework(self.config, self.boundary_manager)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_template_to_configuration_conversion(self):
        """Test conversion from template to configuration."""
        template = self.framework.get_country_template('IND')
        
        config, validation = self.framework.create_country_configuration(
            template, validate_data=False
        )
        
        # Verify all template fields are properly converted
        self.assertEqual(config.country_code, template.country_code)
        self.assertEqual(config.country_name, template.country_name)
        self.assertEqual(config.fips_code, template.fips_code)
        self.assertEqual(config.iso3_code, template.iso3_code)
        self.assertEqual(config.agricultural_focus, template.agricultural_focus)
        self.assertEqual(config.priority_crops, template.priority_crops)
        self.assertEqual(config.regional_subdivisions, template.regional_subdivisions)
        self.assertEqual(config.policy_scenarios, template.policy_scenarios)
    
    def test_framework_repr(self):
        """Test string representation of framework."""
        repr_str = repr(self.framework)
        
        self.assertIn('CountryFramework', repr_str)
        self.assertIn('templates=', repr_str)
        self.assertIn('templates_dir=', repr_str)


if __name__ == '__main__':
    unittest.main()
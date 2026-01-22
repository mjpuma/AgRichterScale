"""Unit tests for SpatialMapper class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.spatial_mapper import SpatialMapper


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(crop_type='wheat', root_dir='.', spam_version='2020')


@pytest.fixture
def grid_manager(config):
    """Create and load GridDataManager."""
    manager = GridDataManager(config)
    # Load data if files exist
    try:
        manager.load_spam_data()
    except FileNotFoundError:
        pytest.skip("SPAM data files not found")
    return manager


@pytest.fixture
def spatial_mapper(config, grid_manager):
    """Create SpatialMapper instance."""
    return SpatialMapper(config, grid_manager)


class TestSpatialMapperInit:
    """Test SpatialMapper initialization."""
    
    def test_init(self, config, grid_manager):
        """Test basic initialization."""
        mapper = SpatialMapper(config, grid_manager)
        
        assert mapper.config == config
        assert mapper.grid_manager == grid_manager
        assert mapper.country_codes_mapping is None
        assert mapper.boundary_data_loaded is False
        assert len(mapper._cache) == 0
    
    def test_repr(self, spatial_mapper):
        """Test string representation."""
        repr_str = repr(spatial_mapper)
        assert 'SpatialMapper' in repr_str
        assert 'mapping=' in repr_str
        assert 'boundaries=' in repr_str


class TestCountryCodeMapping:
    """Test country code mapping functionality."""
    
    def test_load_country_codes_mapping(self, spatial_mapper):
        """Test loading country codes mapping."""
        df = spatial_mapper.load_country_codes_mapping()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'Country' in df.columns
        assert 'GDAM ' in df.columns
        assert 'ISO3 alpha' in df.columns
        
        # Verify caching
        df2 = spatial_mapper.load_country_codes_mapping()
        assert df is df2  # Should return same object
    
    def test_get_iso3_from_country_code(self, spatial_mapper):
        """Test ISO3 code lookup."""
        spatial_mapper.load_country_codes_mapping()
        
        # Test known country codes (these should exist in the mapping)
        # USA (GDAM code varies, but ISO3 should be USA)
        # We'll test with a few common countries
        
        # Get first valid GDAM code from the mapping
        df = spatial_mapper.country_codes_mapping
        valid_row = df[df['GDAM '].notna() & df['ISO3 alpha'].notna()].iloc[0]
        
        gdam_code = valid_row['GDAM ']
        expected_iso3 = valid_row['ISO3 alpha']
        
        iso3 = spatial_mapper.get_iso3_from_country_code(gdam_code, 'GDAM ')
        assert iso3 == expected_iso3
    
    def test_get_iso3_invalid_code(self, spatial_mapper):
        """Test ISO3 lookup with invalid code."""
        spatial_mapper.load_country_codes_mapping()
        
        # Test with non-existent code
        iso3 = spatial_mapper.get_iso3_from_country_code(99999, 'GDAM ')
        assert iso3 is None
    
    def test_get_iso3_nan_code(self, spatial_mapper):
        """Test ISO3 lookup with NaN code."""
        spatial_mapper.load_country_codes_mapping()
        
        iso3 = spatial_mapper.get_iso3_from_country_code(np.nan, 'GDAM ')
        assert iso3 is None
    
    def test_get_country_name_from_code(self, spatial_mapper):
        """Test country name lookup."""
        spatial_mapper.load_country_codes_mapping()
        
        # Get first valid entry
        df = spatial_mapper.country_codes_mapping
        valid_row = df[df['GDAM '].notna()].iloc[0]
        
        gdam_code = valid_row['GDAM ']
        expected_name = valid_row['Country']
        
        name = spatial_mapper.get_country_name_from_code(gdam_code, 'GDAM ')
        assert name == expected_name
    
    def test_get_all_code_systems(self, spatial_mapper):
        """Test getting all code systems."""
        spatial_mapper.load_country_codes_mapping()
        
        code_systems = spatial_mapper.get_all_code_systems()
        
        assert isinstance(code_systems, list)
        assert len(code_systems) > 0
        assert 'GDAM ' in code_systems
        assert 'ISO3 alpha' in code_systems
        assert 'Country' not in code_systems  # Should be excluded


class TestCountryMapping:
    """Test country-level grid cell mapping."""
    
    def test_map_country_to_grid_cells(self, spatial_mapper):
        """Test mapping country to grid cells."""
        spatial_mapper.load_country_codes_mapping()
        
        # Get a valid country code with known ISO3
        df = spatial_mapper.country_codes_mapping
        # Try to find USA (common test case)
        usa_row = df[df['ISO3 alpha'] == 'USA']
        
        if len(usa_row) > 0:
            gdam_code = usa_row.iloc[0]['GDAM ']
            
            prod_ids, harv_ids = spatial_mapper.map_country_to_grid_cells(gdam_code, 'GDAM ')
            
            # USA should have grid cells
            assert isinstance(prod_ids, list)
            assert isinstance(harv_ids, list)
            # Note: May be empty if SPAM data doesn't have USA data
        else:
            pytest.skip("USA not found in country codes mapping")
    
    def test_map_country_invalid_code(self, spatial_mapper):
        """Test mapping with invalid country code."""
        spatial_mapper.load_country_codes_mapping()
        
        prod_ids, harv_ids = spatial_mapper.map_country_to_grid_cells(99999, 'GDAM ')
        
        assert prod_ids == []
        assert harv_ids == []
    
    def test_get_country_grid_cells_dataframe(self, spatial_mapper):
        """Test getting country grid cells as DataFrames."""
        spatial_mapper.load_country_codes_mapping()
        
        # Get a valid country
        df = spatial_mapper.country_codes_mapping
        valid_row = df[df['GDAM '].notna() & df['ISO3 alpha'].notna()].iloc[0]
        gdam_code = valid_row['GDAM ']
        
        prod_df, harv_df = spatial_mapper.get_country_grid_cells_dataframe(gdam_code, 'GDAM ')
        
        assert isinstance(prod_df, pd.DataFrame)
        assert isinstance(harv_df, pd.DataFrame)
    
    def test_map_country_caching(self, spatial_mapper):
        """Test that country mapping results are cached."""
        spatial_mapper.load_country_codes_mapping()
        
        # Get a valid country
        df = spatial_mapper.country_codes_mapping
        valid_row = df[df['GDAM '].notna()].iloc[0]
        gdam_code = valid_row['GDAM ']
        
        # First call
        prod_ids1, harv_ids1 = spatial_mapper.map_country_to_grid_cells(gdam_code, 'GDAM ')
        
        # Second call should use cache
        prod_ids2, harv_ids2 = spatial_mapper.map_country_to_grid_cells(gdam_code, 'GDAM ')
        
        assert prod_ids1 == prod_ids2
        assert harv_ids1 == harv_ids2
        assert len(spatial_mapper._cache) > 0


class TestStateMapping:
    """Test state/province-level mapping."""
    
    def test_map_state_to_grid_cells(self, spatial_mapper):
        """Test mapping states to grid cells."""
        spatial_mapper.load_country_codes_mapping()
        
        # Get a valid country
        df = spatial_mapper.country_codes_mapping
        valid_row = df[df['GDAM '].notna() & df['ISO3 alpha'].notna()].iloc[0]
        gdam_code = valid_row['GDAM ']
        
        # Test with dummy state codes
        state_codes = [1.0, 2.0]
        
        prod_ids, harv_ids = spatial_mapper.map_state_to_grid_cells(
            gdam_code, state_codes, 'GDAM '
        )
        
        assert isinstance(prod_ids, list)
        assert isinstance(harv_ids, list)
    
    def test_get_state_grid_cells_dataframe(self, spatial_mapper):
        """Test getting state grid cells as DataFrames."""
        spatial_mapper.load_country_codes_mapping()
        
        # Get a valid country
        df = spatial_mapper.country_codes_mapping
        valid_row = df[df['GDAM '].notna()].iloc[0]
        gdam_code = valid_row['GDAM ']
        
        state_codes = [1.0]
        
        prod_df, harv_df = spatial_mapper.get_state_grid_cells_dataframe(
            gdam_code, state_codes, 'GDAM '
        )
        
        assert isinstance(prod_df, pd.DataFrame)
        assert isinstance(harv_df, pd.DataFrame)


class TestBoundaryData:
    """Test boundary data loading (optional)."""
    
    def test_load_boundary_data_no_files(self, spatial_mapper):
        """Test loading boundary data when files don't exist."""
        # Should not raise error, just log warnings
        spatial_mapper.load_boundary_data(
            country_shapefile=Path('nonexistent.shp'),
            state_shapefile=Path('nonexistent.shp')
        )
        
        assert spatial_mapper.country_boundaries is None
        assert spatial_mapper.state_boundaries is None
        assert spatial_mapper.boundary_data_loaded is False
    
    def test_load_boundary_data_none(self, spatial_mapper):
        """Test loading boundary data with None paths."""
        spatial_mapper.load_boundary_data()
        
        assert spatial_mapper.country_boundaries is None
        assert spatial_mapper.state_boundaries is None


class TestValidation:
    """Test spatial mapping validation."""
    
    def test_validate_spatial_mapping_empty(self, spatial_mapper):
        """Test validation with empty mappings."""
        event_mappings = {}
        
        results = spatial_mapper.validate_spatial_mapping(event_mappings)
        
        assert results['total_events'] == 0
        assert results['events_with_cells'] == 0
        assert results['events_without_cells'] == 0
        assert results['success_rate_percent'] == 0.0
    
    def test_validate_spatial_mapping_with_data(self, spatial_mapper):
        """Test validation with sample mappings."""
        event_mappings = {
            'Event1': (['cell1', 'cell2'], ['cell1', 'cell2']),
            'Event2': (['cell3'], ['cell3']),
            'Event3': ([], []),  # No cells
        }
        
        results = spatial_mapper.validate_spatial_mapping(event_mappings)
        
        assert results['total_events'] == 3
        assert results['events_with_cells'] == 2
        assert results['events_without_cells'] == 1
        assert results['success_rate_percent'] == pytest.approx(66.67, rel=0.1)
        assert 'Event3' in results['events_with_zero_cells']
        assert results['total_production_cells'] == 3
        assert results['total_harvest_cells'] == 3
    
    def test_generate_spatial_mapping_report(self, spatial_mapper):
        """Test report generation."""
        event_mappings = {
            'Event1': (['cell1', 'cell2'], ['cell1', 'cell2']),
            'Event2': ([], []),
        }
        
        report = spatial_mapper.generate_spatial_mapping_report(event_mappings)
        
        assert isinstance(report, str)
        assert 'SPATIAL MAPPING QUALITY REPORT' in report
        assert 'Total Events: 2' in report
        assert 'Event2' in report  # Should list event with zero cells
    
    def test_get_mapping_statistics(self, spatial_mapper):
        """Test getting mapping statistics."""
        stats = spatial_mapper.get_mapping_statistics()
        
        assert isinstance(stats, dict)
        assert 'country_codes_loaded' in stats
        assert 'boundary_data_loaded' in stats
        assert 'cache_size' in stats
        
        # Load country codes and check stats update
        spatial_mapper.load_country_codes_mapping()
        stats = spatial_mapper.get_mapping_statistics()
        
        assert stats['country_codes_loaded'] is True
        assert 'total_countries' in stats
        assert stats['total_countries'] > 0


class TestCaching:
    """Test caching functionality."""
    
    def test_clear_cache(self, spatial_mapper):
        """Test cache clearing."""
        # Add something to cache
        spatial_mapper._cache['test'] = 'value'
        assert len(spatial_mapper._cache) > 0
        
        spatial_mapper.clear_cache()
        assert len(spatial_mapper._cache) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

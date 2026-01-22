"""Unit tests for GridDataManager."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager


class TestGridDataManager:
    """Test suite for GridDataManager class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        # Use wheat as test crop
        return Config(crop_type='wheat', root_dir='.', spam_version='2020')
    
    @pytest.fixture
    def grid_manager(self, config):
        """Create GridDataManager instance."""
        return GridDataManager(config)
    
    def test_initialization(self, grid_manager, config):
        """Test GridDataManager initialization."""
        assert grid_manager.config == config
        assert not grid_manager.is_loaded()
        assert not grid_manager.has_spatial_index()
        assert grid_manager.production_df is None
        assert grid_manager.harvest_area_df is None
    
    def test_load_spam_data(self, grid_manager):
        """Test SPAM data loading."""
        # Check if SPAM files exist before testing
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        if not file_paths['harvest_area'].exists():
            pytest.skip("SPAM harvest area file not found")
        
        # Load data
        prod_df, harvest_df = grid_manager.load_spam_data()
        
        # Verify data loaded
        assert grid_manager.is_loaded()
        assert prod_df is not None
        assert harvest_df is not None
        assert len(prod_df) > 0
        assert len(harvest_df) > 0
        
        # Verify required columns exist
        required_cols = ['x', 'y', 'FIPS0', 'grid_code']
        for col in required_cols:
            assert col in prod_df.columns
            assert col in harvest_df.columns
        
        # Verify coordinate ranges
        assert prod_df['x'].min() >= -180
        assert prod_df['x'].max() <= 180
        assert prod_df['y'].min() >= -90
        assert prod_df['y'].max() <= 90
    
    def test_load_spam_data_caching(self, grid_manager):
        """Test that loading data twice returns cached data."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data first time
        prod_df1, harvest_df1 = grid_manager.load_spam_data()
        
        # Load data second time (should be cached)
        prod_df2, harvest_df2 = grid_manager.load_spam_data()
        
        # Should return same objects
        assert prod_df1 is prod_df2
        assert harvest_df1 is harvest_df2
    
    def test_create_spatial_index(self, grid_manager):
        """Test spatial index creation."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data first
        grid_manager.load_spam_data()
        
        # Create spatial index
        grid_manager.create_spatial_index()
        
        # Verify spatial index created
        assert grid_manager.has_spatial_index()
        assert grid_manager.production_gdf is not None
        assert grid_manager.harvest_area_gdf is not None
        
        # Verify GeoDataFrame has geometry column
        assert 'geometry' in grid_manager.production_gdf.columns
        assert 'geometry' in grid_manager.harvest_area_gdf.columns
    
    def test_get_grid_cells_by_iso3(self, grid_manager):
        """Test filtering grid cells by ISO3 code."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data
        grid_manager.load_spam_data()
        
        # Get available ISO3 codes
        iso3_codes = grid_manager.get_available_iso3_codes()
        assert len(iso3_codes) > 0
        
        # Test with first available country
        test_iso3 = iso3_codes[0]
        prod_cells, harvest_cells = grid_manager.get_grid_cells_by_iso3(test_iso3)
        
        # Verify results
        assert len(prod_cells) > 0
        assert len(harvest_cells) > 0
        assert len(prod_cells) == len(harvest_cells)
        
        # Verify all cells have correct ISO3
        assert (prod_cells['FIPS0'] == test_iso3).all()
        assert (harvest_cells['FIPS0'] == test_iso3).all()
    
    def test_get_grid_cells_by_iso3_caching(self, grid_manager):
        """Test that ISO3 queries are cached."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data
        grid_manager.load_spam_data()
        
        # Get ISO3 codes
        iso3_codes = grid_manager.get_available_iso3_codes()
        test_iso3 = iso3_codes[0]
        
        # Query twice
        prod_cells1, harvest_cells1 = grid_manager.get_grid_cells_by_iso3(test_iso3)
        prod_cells2, harvest_cells2 = grid_manager.get_grid_cells_by_iso3(test_iso3)
        
        # Should return same objects (from cache)
        assert prod_cells1 is prod_cells2
        assert harvest_cells1 is harvest_cells2
    
    def test_get_grid_cells_by_coordinates(self, grid_manager):
        """Test filtering grid cells by bounding box."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data and create spatial index
        grid_manager.load_spam_data()
        grid_manager.create_spatial_index()
        
        # Test with a small bounding box (e.g., around Europe)
        bounds = (0, 40, 20, 60)  # min_x, min_y, max_x, max_y
        prod_cells, harvest_cells = grid_manager.get_grid_cells_by_coordinates(bounds)
        
        # Verify results are within bounds
        assert (prod_cells['x'] >= bounds[0]).all()
        assert (prod_cells['x'] <= bounds[2]).all()
        assert (prod_cells['y'] >= bounds[1]).all()
        assert (prod_cells['y'] <= bounds[3]).all()
    
    def test_get_crop_production(self, grid_manager):
        """Test crop production aggregation."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data
        grid_manager.load_spam_data()
        
        # Get grid cells for a country
        iso3_codes = grid_manager.get_available_iso3_codes()
        test_iso3 = iso3_codes[0]
        prod_cells, _ = grid_manager.get_grid_cells_by_iso3(test_iso3)
        
        # Get crop indices for wheat
        crop_indices = grid_manager.config.get_crop_indices()
        
        # Calculate production in kcal
        production_kcal = grid_manager.get_crop_production(
            prod_cells, crop_indices, convert_to_kcal=True
        )
        
        # Verify result
        assert production_kcal >= 0
        assert isinstance(production_kcal, float)
        
        # Calculate production in MT
        production_mt = grid_manager.get_crop_production(
            prod_cells, crop_indices, convert_to_kcal=False
        )
        
        # Verify result
        assert production_mt >= 0
        assert isinstance(production_mt, float)
        
        # kcal should be much larger than MT
        if production_mt > 0:
            assert production_kcal > production_mt
    
    def test_get_crop_harvest_area(self, grid_manager):
        """Test crop harvest area aggregation."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data
        grid_manager.load_spam_data()
        
        # Get grid cells for a country
        iso3_codes = grid_manager.get_available_iso3_codes()
        test_iso3 = iso3_codes[0]
        _, harvest_cells = grid_manager.get_grid_cells_by_iso3(test_iso3)
        
        # Get crop indices
        crop_indices = grid_manager.config.get_crop_indices()
        
        # Calculate harvest area
        harvest_area_ha = grid_manager.get_crop_harvest_area(harvest_cells, crop_indices)
        
        # Verify result
        assert harvest_area_ha >= 0
        assert isinstance(harvest_area_ha, float)
    
    def test_validate_grid_data(self, grid_manager):
        """Test grid data validation."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data
        grid_manager.load_spam_data()
        
        # Validate data
        validation_results = grid_manager.validate_grid_data()
        
        # Verify validation results structure
        assert 'valid' in validation_results
        assert 'errors' in validation_results
        assert 'warnings' in validation_results
        assert 'metrics' in validation_results
        
        # Check metrics
        metrics = validation_results['metrics']
        assert 'x_range' in metrics
        assert 'y_range' in metrics
        assert 'grid_cell_count' in metrics
        assert 'country_count' in metrics
        
        # Verify coordinate ranges are valid
        x_range = metrics['x_range']
        y_range = metrics['y_range']
        assert -180 <= x_range[0] <= x_range[1] <= 180
        assert -90 <= y_range[0] <= y_range[1] <= 90
    
    def test_generate_validation_report(self, grid_manager):
        """Test validation report generation."""
        file_paths = grid_manager.config.get_file_paths()
        if not file_paths['production'].exists():
            pytest.skip("SPAM production file not found")
        
        # Load data
        grid_manager.load_spam_data()
        
        # Generate report
        report = grid_manager.generate_validation_report()
        
        # Verify report is a string
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Verify report contains key sections
        assert "VALIDATION REPORT" in report
        assert "METRICS" in report
        assert "Grid Cells:" in report
    
    def test_repr(self, grid_manager):
        """Test string representation."""
        repr_str = repr(grid_manager)
        assert "GridDataManager" in repr_str
        assert "not loaded" in repr_str
        
        # Load data and check repr changes
        file_paths = grid_manager.config.get_file_paths()
        if file_paths['production'].exists():
            grid_manager.load_spam_data()
            repr_str = repr(grid_manager)
            assert "loaded" in repr_str
            assert "grid_cells=" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

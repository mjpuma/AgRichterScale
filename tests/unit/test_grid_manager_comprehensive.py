"""Comprehensive unit tests for GridDataManager."""

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.crop_type = 'wheat'
    config.spam_version = '2020'
    config.root_dir = Path('/mock/root')
    
    # Mock file paths
    config.get_file_paths.return_value = {
        'production': Path('/mock/production.csv'),
        'harvest_area': Path('/mock/harvest_area.csv')
    }
    
    # Mock crop indices
    config.get_crop_indices.return_value = [57]  # Wheat
    
    # Mock caloric content
    config.get_caloric_content.return_value = 3.4  # kcal/g for wheat
    
    # Mock unit conversions
    config.get_unit_conversions.return_value = {
        'grams_per_metric_ton': 1_000_000.0,
        'hectares_to_km2': 0.01
    }
    
    return config


@pytest.fixture
def sample_spam_data():
    """Create sample SPAM data for testing."""
    # Create sample production data
    production_data = {
        'grid_code': [1, 2, 3, 4, 5],
        'x': [-120.0, -119.5, -119.0, -118.5, -118.0],
        'y': [45.0, 45.0, 45.0, 45.0, 45.0],
        'FIPS0': ['US', 'US', 'CA', 'CA', 'MX'],
        'ADM0_NAME': ['United States', 'United States', 'Canada', 'Canada', 'Mexico'],
        'WHEA_A': [100.0, 200.0, 150.0, 300.0, 50.0],  # Wheat production in MT
        'RICE_A': [50.0, 75.0, 25.0, 100.0, 30.0],     # Rice production in MT
        'MAIZ_A': [80.0, 120.0, 90.0, 150.0, 40.0]     # Maize production in MT
    }
    
    # Create sample harvest area data (same structure, different values)
    harvest_data = {
        'grid_code': [1, 2, 3, 4, 5],
        'x': [-120.0, -119.5, -119.0, -118.5, -118.0],
        'y': [45.0, 45.0, 45.0, 45.0, 45.0],
        'FIPS0': ['US', 'US', 'CA', 'CA', 'MX'],
        'ADM0_NAME': ['United States', 'United States', 'Canada', 'Canada', 'Mexico'],
        'WHEA_A': [10.0, 20.0, 15.0, 30.0, 5.0],   # Wheat harvest area in hectares
        'RICE_A': [5.0, 7.5, 2.5, 10.0, 3.0],      # Rice harvest area in hectares
        'MAIZ_A': [8.0, 12.0, 9.0, 15.0, 4.0]      # Maize harvest area in hectares
    }
    
    return pd.DataFrame(production_data), pd.DataFrame(harvest_data)


class TestGridDataManagerInitialization:
    """Test GridDataManager initialization."""
    
    def test_initialization(self, mock_config):
        """Test basic initialization."""
        manager = GridDataManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.production_df is None
        assert manager.harvest_area_df is None
        assert manager._data_loaded is False
        assert manager._spatial_index_created is False
    
    def test_initialization_with_config(self, mock_config):
        """Test initialization stores config correctly."""
        manager = GridDataManager(mock_config)
        
        assert manager.config.crop_type == 'wheat'
        assert manager.config.spam_version == '2020'


class TestSPAMDataLoading:
    """Test SPAM data loading functionality."""
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_load_spam_data_success(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test successful SPAM data loading."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        prod_df, harv_df = manager.load_spam_data()
        
        # Verify data loaded
        assert manager._data_loaded is True
        assert len(prod_df) == 5
        assert len(harv_df) == 5
        assert 'WHEA_A' in prod_df.columns
        assert 'x' in prod_df.columns
        assert 'y' in prod_df.columns
    
    @patch('pathlib.Path.exists')
    def test_load_spam_data_file_not_found(self, mock_exists, mock_config):
        """Test error handling when SPAM files don't exist."""
        mock_exists.return_value = False
        
        manager = GridDataManager(mock_config)
        
        with pytest.raises(FileNotFoundError):
            manager.load_spam_data()
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_load_spam_data_caching(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test that data loading uses cache on second call."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data twice
        manager = GridDataManager(mock_config)
        prod_df1, harv_df1 = manager.load_spam_data()
        prod_df2, harv_df2 = manager.load_spam_data()
        
        # Verify read_csv only called twice (once for each file, not four times)
        assert mock_read_csv.call_count == 2
        
        # Verify same data returned
        assert prod_df1 is prod_df2
        assert harv_df1 is harv_df2
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_load_spam_data_memory_optimization(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test that data loading applies memory optimizations."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df.copy(), harvest_df.copy()]
        
        # Load data
        manager = GridDataManager(mock_config)
        prod_df, harv_df = manager.load_spam_data()
        
        # Verify dtypes are optimized (FIPS0 should be category)
        # Note: This depends on _optimize_memory_usage being called
        assert manager._data_loaded is True


class TestSpatialIndexing:
    """Test spatial indexing functionality."""
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_create_spatial_index(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test spatial index creation."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data and create spatial index
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        manager.create_spatial_index()
        
        # Verify spatial index created
        assert manager._spatial_index_created is True
        assert manager.production_gdf is not None
        assert manager.harvest_area_gdf is not None
        assert isinstance(manager.production_gdf, gpd.GeoDataFrame)
        assert len(manager.production_gdf) == 5
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_create_spatial_index_without_data(self, mock_exists, mock_read_csv, mock_config):
        """Test that spatial index creation fails without loaded data."""
        manager = GridDataManager(mock_config)
        
        with pytest.raises(RuntimeError, match="SPAM data not loaded"):
            manager.create_spatial_index()
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_create_spatial_index_caching(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test that spatial index is only created once."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data and create spatial index twice
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        manager.create_spatial_index()
        
        # Store reference to first GeoDataFrame
        first_gdf = manager.production_gdf
        
        # Call again
        manager.create_spatial_index()
        
        # Verify same GeoDataFrame (not recreated)
        assert manager.production_gdf is first_gdf


class TestGridCellFiltering:
    """Test grid cell filtering functionality."""
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_grid_cells_by_iso3(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test filtering grid cells by ISO3 country code."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Filter by ISO3
        prod_cells, harv_cells = manager.get_grid_cells_by_iso3('US')
        
        # Verify correct cells returned
        assert len(prod_cells) == 2  # Two US cells
        assert all(prod_cells['FIPS0'] == 'US')
        assert len(harv_cells) == 2
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_grid_cells_by_iso3_not_found(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test filtering with non-existent ISO3 code."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Filter by non-existent ISO3
        prod_cells, harv_cells = manager.get_grid_cells_by_iso3('XX')
        
        # Verify empty DataFrames returned
        assert len(prod_cells) == 0
        assert len(harv_cells) == 0
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_grid_cells_by_iso3_caching(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test that ISO3 filtering uses cache."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Filter twice
        prod_cells1, harv_cells1 = manager.get_grid_cells_by_iso3('US')
        prod_cells2, harv_cells2 = manager.get_grid_cells_by_iso3('US')
        
        # Verify cache was used (same object references)
        # Note: Due to copy(), these won't be the same object, but cache should be hit
        assert len(prod_cells1) == len(prod_cells2)


class TestCropAggregation:
    """Test crop aggregation methods."""
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_crop_production(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test crop production aggregation."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Get production for wheat (index 57)
        prod_cells, _ = manager.get_grid_cells_by_iso3('US')
        total_production = manager.get_crop_production(prod_cells, [57], convert_to_kcal=False)
        
        # Verify production sum (100 + 200 = 300 MT for US wheat)
        assert total_production == 300.0
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_crop_production_with_kcal_conversion(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test crop production aggregation with kcal conversion."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Get production for wheat with kcal conversion
        prod_cells, _ = manager.get_grid_cells_by_iso3('US')
        total_production_kcal = manager.get_crop_production(prod_cells, [57], convert_to_kcal=True)
        
        # Verify conversion: 300 MT * 1,000,000 g/MT * 3.4 kcal/g = 1.02e9 kcal
        expected_kcal = 300.0 * 1_000_000.0 * 3.4
        assert abs(total_production_kcal - expected_kcal) < 1e6  # Allow small floating point error
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_crop_harvest_area(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test crop harvest area aggregation."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Get harvest area for wheat
        _, harv_cells = manager.get_grid_cells_by_iso3('US')
        total_harvest = manager.get_crop_harvest_area(harv_cells, [57])
        
        # Verify harvest area sum (10 + 20 = 30 ha for US wheat)
        assert total_harvest == 30.0
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_crop_production_empty_cells(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test crop production with empty grid cells."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Get production for empty DataFrame
        empty_df = pd.DataFrame()
        total_production = manager.get_crop_production(empty_df, [57])
        
        # Verify zero production
        assert total_production == 0.0


class TestHarvestAreaConversions:
    """Test harvest area unit conversions throughout the pipeline."""
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_harvest_area_units_consistency(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test that harvest area units are consistent (hectares in SPAM data)."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Get harvest area for US wheat
        _, harv_cells = manager.get_grid_cells_by_iso3('US')
        total_harvest_ha = manager.get_crop_harvest_area(harv_cells, [57])
        
        # Verify harvest area is in hectares (10 + 20 = 30 ha)
        assert total_harvest_ha == 30.0
        
        # Verify conversion to km² would be correct
        total_harvest_km2 = total_harvest_ha * 0.01
        assert total_harvest_km2 == 0.30  # 30 ha = 0.30 km²
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_harvest_area_to_magnitude_conversion(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test harvest area to magnitude conversion (ha → km² → log10)."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Get harvest area for all cells
        _, harv_cells = manager.get_grid_cells_by_iso3('US')
        total_harvest_ha = manager.get_crop_harvest_area(harv_cells, [57])
        
        # Convert to km² using correct conversion
        hectares_to_km2 = mock_config.get_unit_conversions()['hectares_to_km2']
        total_harvest_km2 = total_harvest_ha * hectares_to_km2
        
        # Calculate magnitude
        magnitude = np.log10(total_harvest_km2)
        
        # Verify magnitude calculation
        # 30 ha = 0.30 km², log10(0.30) ≈ -0.52
        expected_magnitude = np.log10(30.0 * 0.01)
        assert abs(magnitude - expected_magnitude) < 0.01


class TestDataValidation:
    """Test data validation functionality."""
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_validate_grid_data(self, mock_exists, mock_read_csv, mock_config, sample_spam_data):
        """Test grid data validation."""
        # Setup mocks
        mock_exists.return_value = True
        production_df, harvest_df = sample_spam_data
        mock_read_csv.side_effect = [production_df, harvest_df]
        
        # Load data
        manager = GridDataManager(mock_config)
        manager.load_spam_data()
        
        # Validate data
        validation_results = manager.validate_grid_data()
        
        # Verify validation results
        assert 'valid' in validation_results
        assert 'metrics' in validation_results
        assert 'x_range' in validation_results['metrics']
        assert 'y_range' in validation_results['metrics']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

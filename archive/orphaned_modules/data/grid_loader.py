"""Grid structure loader for SPAM data visualization."""

import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from ..core.constants import GRID_PARAMS


class GridLoader:
    """Loads and manages SPAM grid structure from CELL5M.asc file."""
    
    def __init__(self, grid_file_path: str = "ancillary/CELL5M.asc"):
        """Initialize grid loader.
        
        Args:
            grid_file_path: Path to CELL5M.asc file
        """
        self.grid_file_path = Path(grid_file_path)
        self.logger = logging.getLogger(__name__)
        self._grid_data = None
        self._cell_to_coords = None
        
    def load_grid_structure(self) -> np.ndarray:
        """Load the grid structure from CELL5M.asc file.
        
        Returns:
            2D numpy array with cell IDs
        """
        if self._grid_data is not None:
            return self._grid_data
            
        self.logger.info(f"Loading grid structure from {self.grid_file_path}")
        
        if not self.grid_file_path.exists():
            raise FileNotFoundError(f"Grid file not found: {self.grid_file_path}")
        
        # Read the grid file
        # The file contains space-separated cell IDs in a grid format
        try:
            # Read as text and parse manually for better control
            with open(self.grid_file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse grid data
            grid_rows = []
            for line_num, line in enumerate(lines):
                if line_num % 1000 == 0:  # Progress indicator
                    self.logger.info(f"Reading grid line {line_num + 1}/{len(lines)}")
                
                # Split line into cell IDs
                cell_ids = line.strip().split()
                if cell_ids:  # Skip empty lines
                    # Convert to integers, handle missing values
                    row_data = []
                    for cell_id in cell_ids:
                        try:
                            row_data.append(int(cell_id))
                        except ValueError:
                            row_data.append(0)  # Use 0 for missing/invalid cells
                    grid_rows.append(row_data)
            
            # Convert to numpy array
            self._grid_data = np.array(grid_rows, dtype=np.int32)
            
            self.logger.info(f"Grid loaded: {self._grid_data.shape} cells")
            self.logger.info(f"Cell ID range: {self._grid_data.min()} to {self._grid_data.max()}")
            
            return self._grid_data
            
        except Exception as e:
            self.logger.error(f"Failed to load grid structure: {str(e)}")
            raise
    
    def create_cell_coordinate_mapping(self) -> Dict[int, Tuple[float, float]]:
        """Create mapping from cell IDs to geographic coordinates.
        
        Returns:
            Dictionary mapping cell_id -> (longitude, latitude)
        """
        if self._cell_to_coords is not None:
            return self._cell_to_coords
        
        # Load grid if not already loaded
        grid_data = self.load_grid_structure()
        
        self.logger.info("Creating cell ID to coordinate mapping...")
        
        # Get grid parameters
        ncols = int(GRID_PARAMS['ncols'])
        nrows = int(GRID_PARAMS['nrows'])
        xllcorner = GRID_PARAMS['xllcorner']
        yllcorner = GRID_PARAMS['yllcorner']
        cellsize = GRID_PARAMS['cellsize']
        
        # Create coordinate arrays
        lon_1d = np.linspace(xllcorner + cellsize/2, 
                            xllcorner + (ncols-1)*cellsize + cellsize/2, 
                            ncols)
        lat_1d = np.linspace(yllcorner + cellsize/2,
                            yllcorner + (nrows-1)*cellsize + cellsize/2,
                            nrows)
        
        # Create mapping
        self._cell_to_coords = {}
        
        for row_idx in range(grid_data.shape[0]):
            for col_idx in range(grid_data.shape[1]):
                cell_id = grid_data[row_idx, col_idx]
                
                if cell_id > 0:  # Skip invalid cells
                    # Calculate coordinates for this grid position
                    if col_idx < len(lon_1d) and row_idx < len(lat_1d):
                        lon = lon_1d[col_idx]
                        lat = lat_1d[row_idx]
                        self._cell_to_coords[cell_id] = (lon, lat)
        
        self.logger.info(f"Created coordinate mapping for {len(self._cell_to_coords)} cells")
        
        return self._cell_to_coords
    
    def map_production_to_grid(self, production_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map production data to proper grid using cell IDs.
        
        Args:
            production_df: DataFrame with production data and grid_code column
            
        Returns:
            Tuple of (longitude_grid, latitude_grid, production_grid)
        """
        self.logger.info("Mapping production data to grid using cell IDs...")
        
        # Get cell coordinate mapping
        cell_coords = self.create_cell_coordinate_mapping()
        
        # Load grid structure
        grid_data = self.load_grid_structure()
        
        # Get grid parameters
        ncols = int(GRID_PARAMS['ncols'])
        nrows = int(GRID_PARAMS['nrows'])
        xllcorner = GRID_PARAMS['xllcorner']
        yllcorner = GRID_PARAMS['yllcorner']
        cellsize = GRID_PARAMS['cellsize']
        
        # Create coordinate grids
        lon_1d = np.linspace(xllcorner + cellsize/2, 
                            xllcorner + (ncols-1)*cellsize + cellsize/2, 
                            ncols)
        lat_1d = np.linspace(yllcorner + cellsize/2,
                            yllcorner + (nrows-1)*cellsize + cellsize/2,
                            nrows)
        
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
        
        # Initialize production grid
        production_grid = np.zeros(grid_data.shape, dtype=np.float64)
        
        # Get crop columns
        crop_columns = [col for col in production_df.columns if col.endswith('_A')]
        
        if not crop_columns:
            self.logger.warning("No crop columns found in production data")
            return lon_grid, lat_grid, production_grid
        
        # Calculate total production per cell
        total_production = production_df[crop_columns].sum(axis=1)
        
        # Map production values using grid_code
        mapped_cells = 0
        for idx, row in production_df.iterrows():
            grid_code = row.get('grid_code')
            
            if pd.notna(grid_code) and grid_code in cell_coords:
                prod_value = total_production.iloc[idx] if idx < len(total_production) else 0
                
                if prod_value > 0:
                    # Find grid position for this cell ID
                    cell_positions = np.where(grid_data == grid_code)
                    
                    if len(cell_positions[0]) > 0:
                        # Use first occurrence if multiple (shouldn't happen)
                        row_idx = cell_positions[0][0]
                        col_idx = cell_positions[1][0]
                        
                        # Store log10 of production for visualization
                        production_grid[row_idx, col_idx] = np.log10(prod_value)
                        mapped_cells += 1
        
        self.logger.info(f"Mapped {mapped_cells} cells with production data to grid")
        
        # Count non-zero cells
        non_zero_cells = (production_grid > 0).sum()
        self.logger.info(f"Grid cells with production data: {non_zero_cells}")
        
        if non_zero_cells > 0:
            self.logger.info(f"Production range: {production_grid[production_grid > 0].min():.2f} - {production_grid.max():.2f} (log10 kcal)")
        
        return lon_grid, lat_grid, production_grid
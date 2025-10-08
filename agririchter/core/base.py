"""Base interfaces and abstract classes for AgriRichter framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseDataLoader(ABC):
    """Abstract base class for data loading components."""
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data."""
        pass


class BaseProcessor(ABC):
    """Abstract base class for data processing components."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process input data."""
        pass


class BaseAnalyzer(ABC):
    """Abstract base class for analysis components."""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis on input data."""
        pass


class BaseVisualizer(ABC):
    """Abstract base class for visualization components."""
    
    @abstractmethod
    def create_figure(self, data: Any) -> plt.Figure:
        """Create a matplotlib figure."""
        pass
    
    @abstractmethod
    def save_figure(self, fig: plt.Figure, filename: str, **kwargs) -> None:
        """Save figure to file."""
        pass


class BaseValidator(ABC):
    """Abstract base class for data validation components."""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data and return success status and error messages."""
        pass


class BaseOutputManager(ABC):
    """Abstract base class for output management components."""
    
    @abstractmethod
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save analysis results to output directory."""
        pass
    
    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate summary report."""
        pass
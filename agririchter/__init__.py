"""
AgriRichter: A Python framework for analyzing agricultural production disruptions.

This package provides tools for creating Richter-scale measurements of agricultural
disruptions using SPAM 2020 data and historical event analysis.
"""

__version__ = "1.0.0"
__author__ = "AgriRichter Development Team"

from .core.config import Config
from .analysis.agririchter import AgriRichterAnalyzer

__all__ = ["Config", "AgriRichterAnalyzer"]
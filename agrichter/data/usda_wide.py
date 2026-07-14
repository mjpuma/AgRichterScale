"""Shared loader for wide-format USDA PSD grain files (years as columns).

Used by the Figure 4 risk scripts and the methodology/sensitivity supplements.
Values are in the native USDA PSD unit (1000 MT).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FILE_MAP = {
    "production": "grains_world_usdapsd_production_jan212026.csv",
    "consumption": "grains_world_usdapsd_consumption_jan212026.csv",
    "endingstocks": "grains_world_usdapsd_endingstocks_jan212026.csv",
}


def load_usda_wide(data_type: str = "production", data_dir: str = "USDAdata") -> pd.DataFrame:
    """
    Load a wide-format USDA PSD file and melt it to long format.

    Args:
        data_type: One of 'production', 'consumption', 'endingstocks'
        data_dir: Directory containing the USDA CSV files

    Returns:
        Long-format DataFrame with columns:
        Commodity, Attribute, Country, Unit Description, Year_Range, Value, Year
        (Value in 1000 MT)
    """
    data_path = Path(data_dir)
    file_path = data_path / FILE_MAP[data_type]
    if not file_path.exists():
        # Fall back to any matching vintage of the same data type
        candidates = sorted(data_path.glob(f"grains_world_usdapsd_{data_type}_*.csv"))
        if candidates:
            file_path = candidates[-1]
            logger.warning(f"Preferred file missing; falling back to {file_path}")
        else:
            raise FileNotFoundError(f"Missing required USDA file: {file_path}")

    logger.info(f"Loading {data_type} data from {file_path}")
    df = pd.read_csv(file_path)

    # Year columns look like "1960/1961"
    year_cols = [c for c in df.columns if '/' in c and len(c) == 9]

    df_long = df.melt(
        id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'],
        value_vars=year_cols,
        var_name='Year_Range',
        value_name='Value',
    )
    df_long['Year'] = df_long['Year_Range'].str[:4].astype(int)

    if df_long['Value'].dtype == object:
        df_long['Value'] = (
            df_long['Value'].str.replace(',', '').str.replace('"', '').astype(float)
        )

    return df_long

#!/usr/bin/env python3
"""
USDA PSD Data Refresh Script (V3 - corrected World aggregation)

Downloads the latest PSD data and computes World totals by summing ALL
country rows for each commodity/attribute/year.

WHY NO EXCLUSIONS: USDA structures psd_alldata so that reporting entities
are mutually exclusive in time — e.g. 'EU-15' rows run 1960-1998 and are
replaced by 'European Union' from 1999; 'Former Czechoslovakia' runs
1960-1991 and is replaced by successor states from 1992. Individual EU
members (France, Germany, ...) never appear as separate rows, so excluding
the EU row (as a previous version did to "avoid double-counting") silently
dropped ~10% of world production. The plain sum over all rows matches the
official USDA world totals within +/-0.5% for every benchmark year checked
(see BENCHMARKS below, from USDA World Agricultural Production circulars).

Pass --local to rebuild the derived CSVs from an existing
USDAdata/psd_alldata.csv without re-downloading.
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USDA_URL = "https://apps.fas.usda.gov/psdonline/downloads/psd_alldata_csv.zip"
DATA_DIR = Path("USDAdata")
TARGET_COMMODITIES = [
    'Barley', 'Corn', 'Millet', 'Mixed Grain', 'Oats', 
    'Rice, Milled', 'Rye', 'Sorghum', 'Wheat'
]

# Official world production (1000 MT) from USDA World Agricultural
# Production circulars, used to validate the all-rows World sum.
BENCHMARKS = {
    ('Wheat', 'Production', 1990): 588_000,
    ('Wheat', 'Production', 2000): 583_000,
    ('Wheat', 'Production', 2010): 652_000,
    ('Wheat', 'Production', 2020): 774_500,
    ('Corn', 'Production', 2020): 1_129_000,
    ('Rice, Milled', 'Production', 2020): 509_000,
}
BENCHMARK_RTOL = 0.01

REFRESH_DATE = datetime.now().strftime("%b%d%Y").lower()

def download_and_extract():
    logger.info(f"Downloading latest USDA PSD data...")
    response = requests.get(USDA_URL)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
        z.extract(csv_filename, path=DATA_DIR)
        return DATA_DIR / csv_filename

def get_world_aggregate(df):
    """Sum all reporting entities to get the World aggregate.

    USDA reporting entities are mutually exclusive in time (EU-15 rows end
    in 1998 exactly when the European Union rows begin; former-state rows
    end when successor-state rows begin), so a plain sum over all rows is
    the correct world total. See module docstring.
    """
    world = df.groupby(['Commodity_Description', 'Attribute_Description', 'Market_Year', 'Unit_Description']).agg({
        'Value': 'sum'
    }).reset_index()
    
    world['Country_Name'] = 'World'
    return world


def validate_world_aggregate(world_df):
    """Check the computed World sums against official USDA circular values."""
    failures = []
    for (commodity, attribute, year), official in BENCHMARKS.items():
        row = world_df[
            (world_df['Commodity_Description'] == commodity)
            & (world_df['Attribute_Description'] == attribute)
            & (world_df['Market_Year'] == year)
        ]
        if row.empty:
            failures.append(f"{commodity}/{attribute}/{year}: missing")
            continue
        value = float(row['Value'].iloc[0])
        rel = abs(value - official) / official
        status = "OK" if rel <= BENCHMARK_RTOL else "FAIL"
        logger.info(f"Benchmark {commodity} {attribute} {year}: "
                    f"computed={value:,.0f} official~{official:,} ({rel:.2%}) {status}")
        if rel > BENCHMARK_RTOL:
            failures.append(f"{commodity}/{attribute}/{year}: {value:,.0f} vs {official:,}")
    if failures:
        raise ValueError(f"World aggregation failed benchmarks: {failures}")

def transform_to_wide(world_df, attribute_name, output_filename):
    logger.info(f"Generating wide format for {attribute_name}...")
    
    subset = world_df[
        (world_df['Attribute_Description'] == attribute_name) &
        (world_df['Commodity_Description'].isin(TARGET_COMMODITIES))
    ].copy()
    
    def format_year(y):
        return f"{y}/{y+1}"
    subset['Year_Range'] = subset['Market_Year'].apply(format_year)
    
    wide = subset.pivot(
        index=['Commodity_Description', 'Attribute_Description', 'Country_Name', 'Unit_Description'],
        columns='Year_Range',
        values='Value'
    ).reset_index()
    
    wide = wide.rename(columns={
        'Commodity_Description': 'Commodity',
        'Attribute_Description': 'Attribute',
        'Country_Name': 'Country',
        'Unit_Description': 'Unit Description'
    })
    
    year_cols = [c for c in wide.columns if '/' in c]
    allgrain = wide[year_cols].sum().to_frame().T
    allgrain['Commodity'] = 'Allgrain'
    allgrain['Attribute'] = attribute_name
    allgrain['Country'] = 'World'
    allgrain['Unit Description'] = wide['Unit Description'].iloc[0]
    
    final_df = pd.concat([wide, allgrain], ignore_index=True)
    cols = ['Commodity', 'Attribute', 'Country'] + year_cols + ['Unit Description']
    final_df[cols].to_csv(DATA_DIR / output_filename, index=False)

def generate_individual_crop_files(world_df):
    crop_map = {'Wheat': 'wheat', 'Rice, Milled': 'rice', 'Corn': 'maize'}
    for usda_name, file_name in crop_map.items():
        subset = world_df[world_df['Commodity_Description'] == usda_name].copy()
        pivoted = subset.pivot(index='Market_Year', columns='Attribute_Description', values='Value')
        
        rename_dict = {
            'Beginning Stocks': 'Beginning Stocks TY',
            'Domestic Consumption': 'Domestic Consumption TY',
            'Ending Stocks': 'Ending Stocks TY',
            'Production': 'Production TY'
        }
        available_cols = [c for c in rename_dict.keys() if c in pivoted.columns]
        pivoted = pivoted[available_cols].rename(columns=rename_dict)
        
        output_path = DATA_DIR / f"usda_psd_1961to2025_{file_name}.csv"
        with open(output_path, 'w') as f:
            f.write(f"Commodity,{file_name},{file_name},{file_name},{file_name}\n")
            f.write(f"Attribute,{','.join(pivoted.columns)}\n")
            f.write(f"Country,World,World,World,World\n")
            pivoted.to_csv(f, header=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true',
                        help='Rebuild derived CSVs from existing USDAdata/psd_alldata.csv')
    args = parser.parse_args()

    if args.local:
        csv_path = DATA_DIR / 'psd_alldata.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found; run without --local to download")
        logger.info(f"Using existing {csv_path}")
    else:
        csv_path = download_and_extract()
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['Commodity_Description'].isin(TARGET_COMMODITIES)]
    
    logger.info("Computing World aggregates from country-level data...")
    world_df = get_world_aggregate(df)
    validate_world_aggregate(world_df)
    
    transform_to_wide(world_df, 'Production', f"grains_world_usdapsd_production_{REFRESH_DATE}.csv")
    transform_to_wide(world_df, 'Domestic Consumption', f"grains_world_usdapsd_consumption_{REFRESH_DATE}.csv")
    transform_to_wide(world_df, 'Ending Stocks', f"grains_world_usdapsd_endingstocks_{REFRESH_DATE}.csv")
    
    generate_individual_crop_files(world_df)
    
    with open(DATA_DIR / "data_provenance.txt", "w") as f:
        f.write(f"Refresh Date: {datetime.now().isoformat()}\n")
        f.write("Method: Summed all USDA reporting entities per commodity/attribute/year\n")
        f.write("        (entities are mutually exclusive in time; validated against\n")
        f.write("        official WAP circular world totals to within 1%)\n")
        f.write(f"Latest Year: {world_df['Market_Year'].max()}\n")

    logger.info("Data refresh complete!")

if __name__ == "__main__":
    main()

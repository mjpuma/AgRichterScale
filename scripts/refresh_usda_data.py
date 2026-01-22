#!/usr/bin/env python3
"""
USDA PSD Data Refresh Script (V2 - Aggregate Calculation)

This script downloads the latest PSD data and COMPUTES the World Total 
by summing individual countries, ensuring 150% accuracy by avoiding 
double-counting of regional aggregates.
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
# Regional aggregates to EXCLUDE from the World Sum
EXCLUDE_REGIONS = [
    'EE', 'E2', 'E4', 'E5', 'EU-15', 'European Union', 
    'Former Soviet Union', 'Former Czechoslovakia', 'Former Yugoslavia',
    'Former Serbia and Montenegro', 'Other Western Europe'
]

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
    """Filter out regions and sum countries to get a clean World aggregate."""
    # Exclude regions by name and code
    clean_df = df[
        (~df['Country_Name'].isin(EXCLUDE_REGIONS)) & 
        (~df['Country_Code'].isin(EXCLUDE_REGIONS))
    ].copy()
    
    # Group by Commodity, Attribute, Year and sum
    world = clean_df.groupby(['Commodity_Description', 'Attribute_Description', 'Market_Year', 'Unit_Description']).agg({
        'Value': 'sum'
    }).reset_index()
    
    world['Country_Name'] = 'World'
    return world

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
    csv_path = download_and_extract()
    df = pd.read_csv(csv_path)
    
    logger.info("Computing World aggregates from country-level data...")
    world_df = get_world_aggregate(df)
    
    transform_to_wide(world_df, 'Production', f"grains_world_usdapsd_production_{REFRESH_DATE}.csv")
    transform_to_wide(world_df, 'Domestic Consumption', f"grains_world_usdapsd_consumption_{REFRESH_DATE}.csv")
    transform_to_wide(world_df, 'Ending Stocks', f"grains_world_usdapsd_endingstocks_{REFRESH_DATE}.csv")
    
    generate_individual_crop_files(world_df)
    
    with open(DATA_DIR / "data_provenance.txt", "w") as f:
        f.write(f"Refresh Date: {datetime.now().isoformat()}\n")
        f.write(f"Method: Summed individual countries (excluding regions {EXCLUDE_REGIONS})\n")
        f.write(f"Latest Year: {world_df['Market_Year'].max()}\n")

    logger.info("Data refresh complete!")

if __name__ == "__main__":
    main()

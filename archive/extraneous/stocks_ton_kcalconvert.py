#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 01:38:33 2023
converts 1000 mt to kcal for USDA PSD data
"""

import pandas as pd

# Define the path
path = "/Users/mjp38/GitHub/AgriRichterScale/"

# Create a dictionary with the conversion factors
conversion_factors = {
    'Barley': 3.32 * 1000000,
    'Corn': 3.56 * 1000000,
    'Millet': 3.40 * 1000000,
    'Mixed Grain': 3.40 * 1000000,
    'Oats': 3.85 * 1000000,
    'Rice, Milled': 3.60 * 1000000,
    'Rye': 3.19 * 1000000,
    'Sorghum': 3.43 * 1000000,
    'Wheat': 3.34 * 1000000,
}

# Function to convert data from metric tonnes to kcal
def convert_to_kcal(df):
    # List of columns with the data
    data_columns = df.columns[3:-1]

    # Replace commas and convert data to float
    for column in data_columns:
        if df[column].dtype == 'object':  # If the data is of string type
            df[column] = df[column].str.replace(',', '').astype(float)

    # Convert data from metric tonnes to kcal
    for commodity, conversion_factor in conversion_factors.items():
        df.loc[df['Commodity'] == commodity, data_columns] *= conversion_factor

    # Add a row with the sum of all commodities for each year
    sum_row = pd.DataFrame(df[data_columns].sum()).transpose()
    sum_row['Commodity'] = 'Allgrain'
    sum_row['Attribute'] = 'Ending Stocks'
    sum_row['Country'] = 'World'
    sum_row['Unit Description'] = '(kcal)'
    df = pd.concat([df, sum_row], ignore_index=True)

    # Update the unit description
    df.loc[df['Commodity'].isin(conversion_factors.keys()), 'Unit Description'] = '(kcal)'

    return df

# Function to convert data from 1000 metric tonnes to metric tonnes
def convert_to_metric_tonnes(df):
    # Replace commas and convert data to float
    for column in data_columns:
        if df[column].dtype == 'object':  # If the data is of string type
            df[column] = df[column].str.replace(',', '').astype(float)

    # Convert data from 1000 metric tonnes to metric tonnes
    df[data_columns] *= 1000

    return df

# Load data from csv files
stocks_df = pd.read_csv(f"{path}grains_world_usdapsd_endingstocks_jul142023.csv")
production_df = pd.read_csv(f"{path}grains_world_usdapsd_production_jul142023.csv") 
consumption_df = pd.read_csv(f"{path}grains_world_usdapsd_consumption_jul142023.csv")

# List of columns with the data
data_columns = stocks_df.columns[3:-1]

# Convert 1000 mt to mt
stocks_df = convert_to_metric_tonnes(stocks_df)
production_df = convert_to_metric_tonnes(production_df)
consumption_df = convert_to_metric_tonnes(consumption_df)

# Convert mt to kcal
stocks_df = convert_to_kcal(stocks_df)
production_df = convert_to_kcal(production_df)
consumption_df = convert_to_kcal(consumption_df)

# Save the updated datasets to new CSV files
stocks_df.to_csv(f"{path}grains_world_usdapsd_endingstocks_jul142023_kcal.csv", index=False)
production_df.to_csv(f"{path}grains_world_usdapsd_production_jul142023_kcal.csv", index=False)
consumption_df.to_csv(f"{path}grains_world_usdapsd_consumption_jul142023_kcal.csv", index=False)

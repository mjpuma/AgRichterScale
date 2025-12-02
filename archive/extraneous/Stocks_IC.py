#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initial Conditions Generator for Stocks (in metric tons)

Instructions:
-------------
1. Ensure the required data file 'grains_world_usdapsd_endingstocks_jul142023.csv' is in the specified path.
2. Adjust the 'path' variable to point to the directory where the data file is located.
3. Choose the method for generating initial conditions:
    - For bootstrapping approach: set `USE_EVT` to False.
    - For the Extreme Value Theory (EVT) approach: set `USE_EVT` to True.
4. Run the script. The generated initial conditions will be saved to a CSV file named 'stocks_initial_conditions.csv'.
5. A histogram of the generated initial conditions will also be displayed for visualization.

Dependencies:
-------------
- pandas
- numpy
- matplotlib

Author: Michael Puma
Date: September 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the data
path = "/Users/mjp38/GitHub/AgriRichterScale/"  # Modify this to the appropriate path on your system

# Setting the random seed for reproducibility
np.random.seed(42)

# Define the number of samples
bootstrap_samples = 100

# Define the upper and lower percentiles for EVT
upper_percentile = 0.95
lower_percentile = 0.05

# Load data from csv files
stocks_df = pd.read_csv(f"{path}/grains_world_usdapsd_endingstocks_jul142023_kcal.csv")

# Extract the rows corresponding to the desired commodity
selected_commodity = "Wheat"
commodity_stocks = stocks_df[stocks_df["Commodity"] == selected_commodity]

# Drop non-numeric columns
commodity_stocks = commodity_stocks.drop(columns=["Commodity", "Attribute", "Unit Description"])

## Remove rows with non-numeric entries in the years' columns
#commodity_stocks = commodity_stocks[commodity_stocks.applymap(np.isreal).all(1)]

## Convert the DataFrame to float type
commodity_stocks = commodity_stocks.astype(float)

# Convert the DataFrame to a Series
stocks_series = commodity_stocks.stack().reset_index(drop=True)

# Handle NaN values
if stocks_series.isna().any():
    print("Warning: There are NaN values in the data. Handling them.")
    stocks_series = stocks_series.dropna()

USE_EVT = False  # Change this to True if you want to use the EVT approach

if USE_EVT:
    # EVT approach
    upper_threshold = stocks_series.quantile(upper_percentile)
    lower_threshold = stocks_series.quantile(lower_percentile)
    
    # Filter the stocks to only those that are beyond these thresholds
    extreme_values = stocks_series[(stocks_series >= upper_threshold) | (stocks_series <= lower_threshold)]
    
    # Check if there are enough extreme values to sample from
    if extreme_values.shape[0] >= bootstrap_samples:
        # Sample from these extreme values
        bootstrap_IC = extreme_values.sample(n=bootstrap_samples, replace=True).reset_index(drop=True)
    else:
        print(f"Warning: Not enough extreme values ({extreme_values.shape[0]}) for sampling. Using original series.")
        bootstrap_IC = stocks_series.sample(n=bootstrap_samples, replace=True).reset_index(drop=True)
else:
    # Bootstrapping approach
    bootstrap_IC = stocks_series.sample(n=bootstrap_samples, replace=True).reset_index(drop=True)


# Convert bootstrap_IC to a DataFrame for saving to a CSV file
bootstrap_IC_df = pd.DataFrame(bootstrap_IC, columns=["Initial Stocks (1000 MT)"])

# Save the DataFrame to a CSV file
bootstrap_IC_df.to_csv(f"{path}stocks_initial_conditions.csv", index=False)

# Plotting the histogram of the initial conditions for visualization
plt.hist(bootstrap_IC_df["Initial Stocks (1000 MT)"], bins=30, color='blue', edgecolor='black')
plt.title("Histogram of Sampled Initial Conditions")
plt.xlabel("Initial Stocks (1000 MT)")
plt.ylabel("Frequency")
plt.show()
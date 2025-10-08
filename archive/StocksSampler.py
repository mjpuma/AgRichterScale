#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks Sampler

Description:
    This script is designed to compute and visualize the AgriRichter scale for a given crop type. 
    The AgriRichter scale provides a quantitative measure of the impact of production losses, 
    allowing for comparisons across different events and crop types.

Inputs:
    1. Path to the data directory containing CSV files.
    2. CSV files containing global production, consumption, and ending stocks data for the crop type.
    3. Crop type (e.g., 'Wheat', 'Corn').

Outputs:
    1. Plots visualizing the AgriRichter scale, production losses, and their magnitudes.
    2. CSV file (`threshold_values_{crop_type}.csv`) containing threshold values computed for the given crop type.
    3. Visualization saved in SVG and PNG formats: `histogram_comparison_{crop_type}.png`.

Dependencies:
    - Requires pandas, numpy, matplotlib, and scikit-learn libraries.

Usage:
    Before running the script, ensure all the required input CSV files are available in the specified directory.
    Adjust any parameters or configurations as needed before executing.

Author: Michael J Puma
Date: Sept 2023
Version: 1.1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --------------------------- Utility Functions ---------------------------

def simplify_commodity_name(name):
    """Simplify commodity name by extracting the first part before a comma."""
    return name.split(',')[0].strip()

def construct_filename(prefix, scenario_name, parameter, crop_type, decimal_places=3, is_scientific=False):
    """
    Constructs a standardized filename with given parameters.
    
    Parameters:
        prefix (str): The prefix of the filename.
        scenario_name (str): The name of the scenario.
        parameter (float): The numerical parameter to include.
        crop_type (str): The crop type.
        decimal_places (int): Number of decimal places for formatting.
        is_scientific (bool): Whether to format the number in scientific notation.
    
    Returns:
        str: The constructed filename.
    """
    if is_scientific:
        formatted_param = f"{parameter:.2e}".replace('.', 'p').replace('-', 'm')
    else:
        formatted_param = f"{parameter:.{decimal_places}f}".replace('.', 'p')
    filename = f"{prefix}_{scenario_name}_{formatted_param}_{crop_type}.csv"
    return filename

def verify_file_exists(file_path):
    """Verifies that a file exists at the given path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    else:
        print(f"Verified existence of file: {file_path}")

def verify_units(value, expected_unit, actual_unit):
    """
    Verifies that a value is in the expected unit.
    
    Parameters:
        value (float): The numerical value.
        expected_unit (str): The unit that the value should be in.
        actual_unit (str): The unit that the value is currently in.
    
    Raises:
        ValueError: If the actual unit does not match the expected unit.
    """
    if actual_unit.lower() != expected_unit.lower():
        raise ValueError(f"Unit mismatch: Expected {expected_unit}, but got {actual_unit}")
    else:
        print(f"Unit verification passed: {expected_unit}")

# --------------------------- Main Script ---------------------------

def main():
    # Define the path
    path = "/Users/mjp38/Dropbox (Personal)/GitHub/AgriRichterScale/"
    
    # Define the crop type
    crop_type = "Rice"  # Tested options: "Wheat", "Rice", "Allgrain"
    crop_type = crop_type.capitalize()  # Ensure consistent capitalization
    
    # Specify the range of years (for the USDA PSD data)
    start_year = '2001/2002'  # '1975/1976'
    end_year = '2021/2022'
    
    # Extract the start and end years
    start = int(start_year.split('/')[0])
    end = int(end_year.split('/')[1])
    
    # Generate the range of years based on start_year and end_year
    years = [str(i) + "/" + str(i+1) for i in range(start, end)]
    
    # Function to filter dataframe based on crop type
    def filter_dataframe(df, crop_type):
        # Special handling for rice
        if crop_type.lower() == "rice":
            crop_search = "Rice, Milled"
        else:
            crop_search = crop_type
        
        # Filter the dataframe for rows where 'Commodity' contains crop_type
        # Handle non-string entries by converting to string and setting na=False
        df_filtered = df[df['Commodity'].astype(str).str.contains(crop_search, case=False, na=False)]
        return df_filtered
    
    # Load data from csv files
    stocks_csv_path = os.path.join(path, "USDAdata", "grains_world_usdapsd_endingstocks_jul142023.csv")
    stocks_df = pd.read_csv(stocks_csv_path)
    
    # Convert 'Commodity' column to string and handle missing values
    stocks_df['Commodity'] = stocks_df['Commodity'].astype(str).fillna('')
    
    # Simplify commodity names
    stocks_df['Commodity'] = stocks_df['Commodity'].apply(simplify_commodity_name)
    
    # Transpose the dataframes and select the range of years
    # Ensure that all year columns are strings matching the 'years' list
    stocks_df[years] = stocks_df[years].replace(',', '', regex=True).astype(float)
    stocks = stocks_df.set_index('Commodity')[years]
    
    # Filter the dataframe for the selected crop type
    df_filtered = filter_dataframe(stocks_df, crop_type)
    
    if df_filtered.empty:
        raise ValueError(f"No data found for crop type: {crop_type}")
    
    # Melt the filtered dataframe
    df_melted = df_filtered.melt(id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'],
                                 var_name='Year', value_name='Value')
    
    # Filter years and ensure conversion to numeric
    df_melted = df_melted[df_melted['Year'].isin(years)]
    df_melted['Value'] = pd.to_numeric(df_melted['Value'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Drop rows with NaN in 'Value'
    df_melted = df_melted.dropna(subset=['Value'])
    
    # Create the stocks_series
    stocks_series = df_melted['Value'].reset_index(drop=True)
    
    # Create a year array that matches the length of stocks_series
    years_per_crop = len(df_filtered)
    years_series = np.tile(np.arange(start, end), years_per_crop)
    
    # Check the length
    if len(years_series) != len(stocks_series):
        raise ValueError(f"Length mismatch: years_series ({len(years_series)}) and stocks_series ({len(stocks_series)})")
    
    # Detrend the data
    model = LinearRegression().fit(years_series.reshape(-1, 1), stocks_series)
    trend = model.predict(years_series.reshape(-1, 1))
    detrended_stocks = stocks_series - trend
    
    # Setting the random seed for reproducibility
    np.random.seed(42)
    
    # Define the number of samples
    bootstrap_samples = 100
    
    # Define whether to use EVT
    USE_EVT = False
    
    # Apply either EVT or Bootstrapping approach on detrended data
    if USE_EVT:
        # EVT approach
        upper_percentile = 0.95
        lower_percentile = 0.05
        upper_threshold = np.quantile(detrended_stocks, upper_percentile)
        lower_threshold = np.quantile(detrended_stocks, lower_percentile)
        extreme_values = detrended_stocks[(detrended_stocks >= upper_threshold) | (detrended_stocks <= lower_threshold)]
        bootstrap_IC = np.random.choice(extreme_values, size=bootstrap_samples, replace=True)
    else:
        # Bootstrapping approach
        bootstrap_IC = np.random.choice(detrended_stocks, size=bootstrap_samples, replace=True)
    
    # Reintroduce the trend to the bootstrapped samples
    # Generate a series of evenly spaced years over the range
    full_years_series = np.linspace(start, end, bootstrap_samples)
    
    # Recalculate the trend for the new years
    full_trend = model.predict(full_years_series.reshape(-1, 1))
    
    # Reintroduce the trend to the bootstrapped samples
    reintroduced_trend_samples = bootstrap_IC + full_trend
    
    # Convert the reintroduced trend samples to a DataFrame
    bootstrap_IC_df = pd.DataFrame(reintroduced_trend_samples, columns=['Stocks (1000 metric tons)'])
    
    # Reset the index of the DataFrame
    bootstrap_IC_df = bootstrap_IC_df.reset_index(drop=True)
    
    # Save the DataFrame to a CSV file with crop name appended
    bootstrap_IC_filename = f"bootstrap_initial_conditions_{crop_type}.csv"
    bootstrap_IC_path = os.path.join(path, bootstrap_IC_filename)
    bootstrap_IC_df.to_csv(bootstrap_IC_path, index=False)
    print(f"Saved bootstrap initial conditions to {bootstrap_IC_filename}")
    
    # Verify that the bootstrap file exists
    verify_file_exists(bootstrap_IC_path)
    
    # Plotting normalized histograms
    plt.figure(figsize=(10, 6))
    
    # Normalized histogram for the original stocks series
    plt.hist(stocks_series, bins=30, alpha=0.5, label='Original Stocks', density=True, color='blue')
    
    # Normalized histogram for the reintroduced trend samples
    plt.hist(reintroduced_trend_samples, bins=30, alpha=0.5, label=f'Sampled Values (n={len(reintroduced_trend_samples)})', density=True, color='orange')
    
    plt.title(f'Normalized Distribution of Original Stocks vs. Sampled Values for {crop_type}')
    plt.xlabel('Stocks (1000 mt)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot with crop name appended
    histogram_filename = f"histogram_comparison_{crop_type}.png"
    histogram_path = os.path.join(path, histogram_filename)
    plt.savefig(histogram_path, dpi=300)
    print(f"Saved histogram comparison plot to {histogram_filename}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()

# --------------------------- End of Script ---------------------------



"""
Sampling Approaches in the Code:
--------------------------------
1. Bootstrapping:
    - The code can use the non-parametric bootstrap approach.
    - This method involves resampling the original data with replacement to create many "pseudo-samples".
    - The statistic of interest (e.g., mean, median, etc.) is then computed on each of these pseudo-samples.
    - Limitation: Bootstrapping cannot generate more extreme values than what's in the original dataset.

2. Extreme Value Theory (EVT):
    - The code can also use the EVT approach which is geared towards understanding extreme values.
    - Specifically, the generalized Pareto distribution (GPD) is used to model exceedances over a threshold.
    - For this implementation, the 5th and 95th percentiles are chosen as lower and upper thresholds respectively.
    - We then sample from the fitted GPDs to generate extreme values.

Sampling Methods Overview:
--------------------------
1. Non-parametric Bootstrap: Resamples directly from the observed data. Requires no assumptions about the population distribution.
2. Parametric Bootstrap: Fits a statistical model to the data and samples from this model. Assumes a specific distribution for the data.
3. Smoothed Bootstrap: Adds random noise to each resampled observation. Useful for discrete data that's believed to be from a continuous phenomenon.
4. Block Bootstrap: Designed for time series data. Resamples "blocks" of consecutive data points to maintain temporal structure.
5. Spatial and Clustered Bootstraps: Variants for spatial data or data in naturally occurring clusters.
6. Bias-Corrected Bootstrap: Adjusts for bias in the bootstrap distribution.
7. Bootstrap-t: Adjusts for potential skewness in the bootstrap distribution.

Pitfalls and Considerations:
----------------------------
- Bootstrapping:
    - Cannot generate more extreme values than what's in the original dataset.
    - Reliability of extreme estimates from bootstrapping is limited by the most extreme values in the data.
- EVT:
    - Selecting an appropriate threshold is critical.
    - Not suitable for modeling the entire distribution, only the tails.
    - Requires a sufficient number of extreme observations for reliable parameter estimation.
"""


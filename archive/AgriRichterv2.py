#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgriRichter Scale Analysis Script

Description:
    This script is designed to compute and visualize the AgriRichter scale for a given crop type. 
    The AgriRichter scale provides a quantitative measure of the impact of production losses, 
    allowing for comparisons across different events and crop types.

Inputs:
    1. Path to the data directory containing CSV files.
    2. CSV files containing global production, consumption, and ending stocks data for the crop type.
    3. Crop type (e.g., 'wheat', 'maize').

Outputs:
    1. Plots visualizing the AgriRichter scale, production losses, and their magnitudes.
    2. CSV file (`threshold_values_{crop_type}.csv`) containing threshold values computed for the given crop type.
    3. Visualization saved in SVG and PNG formats: `Lost_Production_vs_Magnitude_{crop_type}_all_labels_shaded`.

Dependencies:
    - Requires pandas, numpy, and matplotlib libraries.

Usage:
    Before running the script, ensure all the required input CSV files are available in the specified directory.
    Adjust any parameters or configurations as needed before executing.

Author:Michael J Puma
Date: Sept 2023
Version:1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto

def simplify_commodity_name(name):
    return name.split(',')[0].strip()

# Define the path
path = "/Users/mjp38/GitHub/AgriRichterScale/"

# Define the crop type
crop_type = "Corn"  # Tested options: "Wheat", "Rice", "Allgrain" 

# Specify the range of years (for the USDA PSD data)
#start_year = '1961/1962'
#end_year = '1971/1972'
start_year = '2001/2002'
end_year = '2021/2022'

# Extract the start and end years
start = int(start_year.split('/')[0])
end = int(end_year.split('/')[1])

# Generate the range of years based on start_year and end_year
years = [str(i)+"/"+str(i+1) for i in range(start, end)]

# Define a dictionary for calories_cropprod
calories_cropprod_dict = {
    'Allgrain': 3.45,  # kcal/g
    'Barley': 3.32,  # kcal/g
    'Corn': 3.56,  # kcal/g
    'Millet': 3.40,  # kcal/g
    'Mixed Grain': 3.40,  # kcal/g
    'Oats': 3.85,  # kcal/g
    'Rice': 3.60,  # kcal/g
    'Rye': 3.19,  # kcal/g
    'Sorghum': 3.43,  # kcal/g
    'Wheat': 3.34,  # kcal/g
}

# Function to filter dataframe based on crop type
def filter_dataframe(df, crop_type):
    # Special handling for rice
    if crop_type.lower() == "rice":
        crop_type = "Rice, Milled"
    
    # Filter the dataframe for rows where 'Commodity' contains crop_type
    df_filtered = df[df['Commodity'].str.contains(crop_type, case=False)]
    
    return df_filtered

# Load data from csv files
stocks_df = pd.read_csv(f"{path}/USDAdata/grains_world_usdapsd_endingstocks_jul142023_kcal.csv")
production_df = pd.read_csv(f"{path}/USDAdata/grains_world_usdapsd_production_jul142023_kcal.csv")
consumption_df = pd.read_csv(f"{path}/USDAdata/grains_world_usdapsd_consumption_jul142023_kcal.csv")

# Prepare dataframes for average computation
dfs = [stocks_df, production_df, consumption_df]
df_names = ['stocks', 'production', 'consumption']

# Compute the averages for 2018 to 2022 for each dataframe

for df, name in zip(dfs, df_names):
    df_commodity = filter_dataframe(df, crop_type)
    df_commodity = df_commodity.melt(id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'], var_name='Year', value_name='Value')
    if df_commodity['Value'].dtype == object:
        df_commodity['Value'] = df_commodity['Value'].str.replace(',', '').astype(float)
    df_commodity['Year'] = df_commodity['Year'].str[:4].astype(int)
    df_commodity_grouped = df_commodity.groupby('Year').mean(numeric_only=True)


def compute_prod_range(df, commodity, min_adjustment_factor=0.9, max_adjustment_factor=1.1):
    """
    Compute the production range for a given commodity.
    
    Parameters:
    - df: DataFrame containing the production data.
    - commodity: Name of the commodity to filter.
    - min_adjustment_factor: Factor to adjust the minimum production value (default is 0.9, i.e., 90% of the actual minimum).
    - max_adjustment_factor: Factor to adjust the maximum production value (default is 1.1, i.e., 110% of the actual maximum).
    
    Returns:
    - prod_range: Array containing the production range values.
    """
    # Melt the dataframe to a long format
    df_commodity = df[df['Commodity'].str.contains(commodity)]
    df_commodity = df_commodity.melt(id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'], var_name='Year', value_name='Value')

    # Handle the data type of 'Value' column
    if df_commodity['Value'].dtype == object:
        df_commodity['Value'] = df_commodity['Value'].str.replace(',', '').astype(float)

    # Get the production values
    production_values = df_commodity['Value']

    # Compute the production range with adjustments
    min_prod = production_values.min() * min_adjustment_factor
    max_prod = production_values.max() * max_adjustment_factor
    prod_range = np.logspace(np.log10(min_prod), np.log10(max_prod), 1000)

    return prod_range


def calculate_percentile(df, percentile=15):
    """Calculate the given percentile for each commodity."""
    simplified_index = [name.split(',')[0] for name in df.index]
    percentile_values = pd.Series(
        df.apply(lambda x: np.percentile(x, percentile), axis=1).values,
        index=simplified_index
    )
    return percentile_values

def calculate_average(df):
    """Calculate the mean for each commodity."""
    return df.mean(axis=1)

def calculate_thresholds(crop_type):
    """
    Calculate the thresholds for the shaded regions.
    Threshold 1: Average loss needed to drop down to the 15th percentile.
    Threshold 2: Complete loss equivalent to the 15th percentile of stocks.
    Threshold 3: Complete loss of the average stock level.
    Threshold 4: Average of stocks plus average annual production.
    """   
    # Calculate the median (50th percentile) and other percentiles
    SUR_15percentile = calculate_percentile(stocks_to_consumption_ratio, percentile=15)
    SUR_median = calculate_percentile(stocks_to_consumption_ratio, percentile=50)

    stocks_median = calculate_percentile(stocks, percentile=50)
    production_median = calculate_percentile(production, percentile=50)
    consumption_median = calculate_percentile(consumption, percentile=50)

    # Calculate the mean for each 
    #stocks_mean = calculate_average(stocks)
    #production_mean = calculate_average(production)
    #consumption_mean = calculate_average(consumption)


    # Get the values for the given crop type 
    stocks_median_crop = stocks_median[crop_type]
    production_median_crop = production_median[crop_type]
    consumption_median_crop = consumption_median[crop_type]

    # Compute the thresholds for the shaded regions
    alphaC = 0.5
    threshold_1 = stocks_median_crop - SUR_15percentile[crop_type] * consumption_median_crop
    threshold_2 = SUR_15percentile[crop_type] * consumption_median_crop
    threshold_3 = SUR_median[crop_type] * consumption_median_crop + alphaC * production_median_crop
    threshold_4 = SUR_median[crop_type] * consumption_median_crop + production_median_crop

    return [threshold_1, threshold_2, threshold_3, threshold_4]


# Simplify names in the Commodity column
consumption_df['Commodity'] = consumption_df['Commodity'].apply(simplify_commodity_name)
production_df['Commodity'] = production_df['Commodity'].apply(simplify_commodity_name)
stocks_df['Commodity'] = stocks_df['Commodity'].apply(simplify_commodity_name)

# Transpose the dataframes and select the range of years
consumption = consumption_df.set_index('Commodity')[years].astype(float)
production = production_df.set_index('Commodity')[years].astype(float)
stocks = stocks_df.set_index('Commodity')[years].astype(float)

# Calculate the stocks-to-consumption ratio
stocks_to_consumption_ratio = stocks / consumption

# Calculate the thresholds for the shaded regions for a given crop type
thresholds = calculate_thresholds(crop_type)

# Extract single values from the Series
threshold_1_value = thresholds[0]
threshold_2_value = thresholds[1]
threshold_3_value = thresholds[2]
threshold_4_value = thresholds[3]


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot Stocks to Consumption Ratio
ax = axes[0, 0]
ax.set_facecolor('white')
ax.plot(stocks_to_consumption_ratio.loc[crop_type], label=crop_type)
ax.axhline(stocks_to_consumption_ratio.loc[crop_type].quantile(0.50), color='gray', linestyle='--', linewidth=0.7)
ax.axhline(stocks_to_consumption_ratio.loc[crop_type].quantile(0.15), color='red', linestyle='--', linewidth=0.7)
ax.set_title("Stocks to Use Ratio (SUR) - " + crop_type)
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Plot Stocks
ax = axes[0, 1]
ax.set_facecolor('white')
ax.plot(stocks.loc[crop_type], label=crop_type)
ax.axhline(stocks.loc[crop_type].quantile(0.50), color='gray', linestyle='--', linewidth=0.7)
ax.set_title("Stocks - " + crop_type)
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Plot Production
ax = axes[1, 0]
ax.set_facecolor('white')
ax.plot(production.loc[crop_type], label=crop_type)
ax.axhline(production.loc[crop_type].quantile(0.50), color='gray', linestyle='--', linewidth=0.7)
ax.set_title("Production - " + crop_type)
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Plot Consumption
ax = axes[1, 1]
ax.set_facecolor('white')
ax.plot(consumption.loc[crop_type], label=crop_type)
ax.axhline(consumption.loc[crop_type].quantile(0.50), color='gray', linestyle='--', linewidth=0.7)
ax.set_title("Consumption - " + crop_type)
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# Save the plot as PNG
plt.savefig(f'{path}media/BalanceTimeSeries_{crop_type}.png',facecolor=fig.get_facecolor())
plt.show()


# Get the calories_cropprod for the given crop type
calories_cropprod = calories_cropprod_dict[crop_type]
grams_per_metric_ton = 1e6  # 1 metric ton = 1 million grams
constant = grams_per_metric_ton * calories_cropprod

# Load lost production by event estimated using SPAM 2010v2
df = pd.read_csv(f'{path}inputs/LostProd_Events_spam2010V2r0_{crop_type.lower()}.csv')
total_prod_2010 = df.iloc[-1]['Production Lost (kcal)']

# Drop the last row
df = df[:-1]

# Compute the scale coefficient for a specific commodity
scaling_constant = production.loc[crop_type] / total_prod_2010


df['Scaled Production Lost (kcal)'] = df['Production Lost (kcal)']

# Replace zeros and negative values with a small positive value
df['Scaled Production Lost (kcal)'] = df['Scaled Production Lost (kcal)'].apply(lambda x: max(x, 1e-10))

# Now compute the magnitude
df['Magnitude'] = np.log10(df['Scaled Production Lost (kcal)'] / constant)


# Replace -np.inf with np.nan
df['Magnitude'] = df['Magnitude'].replace(-np.inf, np.nan)

# Exclude rows with NaN Magnitude when plotting
df = df.dropna(subset=['Magnitude'])

# Get the production range for the given crop type with 90% min and 110% max adjustment factors
prod_range = compute_prod_range(production_df, crop_type, 1e-6, 1.5)

# Compute the X values for the line
X = np.log10(prod_range / constant)

# Compute the Y values for the line
Y = prod_range

# Get the colormap
#cmap = cm.get_cmap('YlOrRd')
## Get five different colors from the colormap
#colors = [cmap(i) for i in np.linspace(0.5, 1, 5)]

colors = [plt.get_cmap('Greens')(0.4), 
          plt.get_cmap('hot')(0.8), 
          plt.get_cmap('Oranges')(0.5), 
          plt.get_cmap('Reds')(0.8), 
          plt.get_cmap('Reds')(0.3)]


# Plotting
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor('none')

# Define the scatter plot first
plt.scatter(df['Magnitude'], df['Production Lost (kcal)'])

# Now plot the line
plt.plot(X, Y, 'k-')

# Explicitly set the x-axis limits
# Get the limits
# Compute data range for x-axis
x_min, x_max = plt.gca().get_xlim()
x_min = df['Magnitude'].min()-.1
#x_max = df['Magnitude'].max()
y_min, y_max = plt.gca().get_ylim()


# Plot horizontal lines at the threshold values
plt.axhline(y=threshold_1_value, color='white', linestyle='--')
plt.axhline(y=threshold_2_value, color='white', linestyle='--')
plt.axhline(y=threshold_3_value, color='white', linestyle='--')
plt.axhline(y=threshold_4_value, color='white', linestyle='--')

# Label all points on the plot
for i in range(df.shape[0]):
    plt.text(df['Magnitude'].iat[i], df['Production Lost (kcal)'].iat[i], df['Event'].iat[i])

# Add labels for the thresholds
plt.text(df['Magnitude'].min(), threshold_1_value, r'$T_1 = S_{\mathrm{median}} - \mathrm{SUR}_{15\%} \times C_{\mathrm{median}}$', va='top', ha='left')
plt.text(df['Magnitude'].min(), threshold_2_value, r'$T_2 = \mathrm{SUR}_{15\%} \times C_{\mathrm{median}}$', va='top', ha='left')
plt.text(df['Magnitude'].min(), threshold_3_value, r'$T_3 = (\mathrm{SUR}_{\mathrm{median}} \times C_{\mathrm{median}}) + \alpha*P_{\mathrm{median}}$', va='top', ha='left')
plt.text(df['Magnitude'].min(), threshold_4_value, r'$T_4 = (\mathrm{SUR}_{\mathrm{median}} \times C_{\mathrm{median}}) + P_{\mathrm{median}}$', va='top', ha='left')


# Add the shaded regions
# Define initial threshold as the first value from thresholds list
prev_threshold = 0#threshold_1_value

# Now we iterate over the thresholds list starting from the second element
for threshold, color in zip(thresholds, colors):
    plt.fill_betweenx([prev_threshold, threshold], x_min, x_max, color=color, alpha=.7)
    prev_threshold = threshold

# Add additional shading for threshold_4 to y_max
#plt.fill_betweenx([threshold_4, y_max], x_min, x_max, color=colors[-1], alpha=1)

plt.xlabel('Magnitude')
plt.ylabel('Commodity Losses (kcal)')
plt.title(f'AgriRichter Scale for {crop_type.capitalize()}')

plt.xlim(x_min, x_max)
plt.ylim(y_min, threshold_4_value)  # Here we set the maximum y value to threshold_4

# Turn off autoscaling for x-axis
plt.gca().autoscale(enable=False, axis='x')


# Save the plot 
plt.savefig(f'{path}media/Lost_Production_vs_Magnitude_{crop_type}_all_labels_shaded.svg',transparent=True)
plt.savefig(f'{path}media/Lost_Production_vs_Magnitude_{crop_type}_all_labels_shaded.png',transparent=True)

plt.show()

# Save threshold values to file
threshold_data = {
    'Description': ['T1', 'T2', 'T3', 'T4'],
    'Value': [threshold_1_value, threshold_2_value, threshold_3_value, threshold_4_value],
    'Units': ['kcal', 'kcal', 'kcal', 'kcal']
}

threshold_df = pd.DataFrame(threshold_data)


# Stocks sampling for TWIST IC
stocks_series = stocks.stack().reset_index(drop=True)

# Setting the random seed for reproducibility
np.random.seed(42)

# Define the number of samples
bootstrap_samples = 100

# Define the upper and lower percentiles for EVT
upper_percentile = 0.95
lower_percentile = 0.05

# Extract stocks from the dataframe
stocks_series = stocks.stack().reset_index(drop=True)

USE_EVT = True  # Change this to True if you want to use the EVT approach

if USE_EVT:
    # EVT approach
    upper_threshold = stocks_series.quantile(upper_percentile)
    lower_threshold = stocks_series.quantile(lower_percentile)
    
    # Filter the stocks to only those that are beyond these thresholds
    extreme_values = stocks_series[(stocks_series >= upper_threshold) | (stocks_series <= lower_threshold)]
    
    # Sample from these extreme values
    bootstrap_IC = extreme_values.sample(n=bootstrap_samples, replace=True).reset_index(drop=True)
else:
    # Bootstrapping approach
    bootstrap_IC = stocks_series.sample(n=bootstrap_samples, replace=True).reset_index(drop=True)


# Convert the Series to a DataFrame with a column name
bootstrap_IC_df = bootstrap_IC.to_frame(name='Stocks (kcal)')

# Save the DataFrame to a CSV file
bootstrap_IC_df.to_csv(f"{path}bootstrap_initial_conditions.csv", index=False)


# Plotting
plt.figure(figsize=(10, 6))
plt.hist(stocks_series, bins=30, alpha=0.5, label='Original Stocks')
plt.hist(bootstrap_IC, bins=30, alpha=0.5, label=f'Sampled Values (n={len(bootstrap_IC)})', color='orange')
plt.title('Distribution of Original Stocks vs. Sampled Values')
plt.xlabel('Stocks (kcal)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot
plt.savefig(f"{path}histogram_comparison.png", dpi=300)

# Display the plot
plt.show()


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


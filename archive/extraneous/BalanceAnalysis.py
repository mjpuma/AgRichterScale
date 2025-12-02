import pandas as pd
import matplotlib.pyplot as plt

# **1. Summary:**
# This script processes and visualizes data related to the world's grain supply. It takes data on grain stocks, production, harvest area, and consumption from CSV files and generates plots of these quantities over time for various grain commodities. It then calculates and plots the ratio of stocks to consumption for each commodity. Lastly, the script computes the 15th percentile of the stocks-to-consumption ratio for each commodity and saves this information to a CSV file.

# **2. Input:**
# - The script requires four CSV files as input, each containing data on a different aspect of the world's grain supply. The filenames are hardcoded into the script and are as follows:
#     1. `grains_world_usdapsd_endingstocks_jul142023_kcal.csv`: Data on grain stocks
#     2. `grains_world_usdapsd_production_jul142023_kcal.csv`: Data on grain production
#     3. `grains_world_usdapsd_harvestarea_jul142023.csv`: Data on the area harvested
#     4. `grains_world_usdapsd_consumption_jul142023_kcal.csv`: Data on grain consumption
# - The path to the directory containing these files is also hardcoded into the script and is set to `"/Users/mjp38/GitHub/AgriRichterScale/"`.

# **3. Output:**
# - The script generates a series of plots for each grain commodity. These plots are saved as PNG files in the `media/` subdirectory of the path specified above. The filename of each plot is determined by the commodity and the type of plot. For instance, the mean time series plot for wheat is saved as `wheat_mean_timeseries.png` and the stocks-to-consumption ratio plot for corn is saved as `corn_stocks_to_consumption_ratio.png`.
# - In addition to the plots, the script also generates a CSV file containing the 15th percentile of the stocks-to-consumption ratio for each commodity. This file is saved in the directory specified by the path above and is named `percentile_15_values.csv`.

# Define the path
path = "/Users/mjp38/GitHub/AgriRichterScale/"

# Load data from csv files
stocks_df = pd.read_csv(f"{path}grains_world_usdapsd_endingstocks_jul142023_kcal.csv")
production_df = pd.read_csv(f"{path}grains_world_usdapsd_production_jul142023_kcal.csv")
area_df = pd.read_csv(f"{path}grains_world_usdapsd_harvestarea_jul142023.csv")
consumption_df = pd.read_csv(f"{path}grains_world_usdapsd_consumption_jul142023_kcal.csv")

# Prepare dataframes list and titles
dfs = [stocks_df, production_df, area_df, consumption_df]
titles = ['Ending Stocks', 'Production', 'Harvest Area', 'Consumption']

commodities = ['Allgrain', 'Barley', 'Corn', 'Millet', 'Mixed Grain', 'Oats', 'Rice, Milled', 'Rye', 'Sorghum', 'Wheat']

# Initialize an empty dictionary to store the 15th percentile values for each commodity
percentile_15_values = {}

# Plot data for each commodity
for commodity in commodities:
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    plt.suptitle(commodity, fontsize=16)

    # Separate data for each commodity and plot
    for i, df in enumerate(dfs):
        ax = plt.subplot(2, 2, i+1)
        df_commodity = df[df['Commodity'] == commodity]
        df_commodity = df_commodity.melt(id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'], var_name='Year', value_name='Value')
        
        # Handling the data type of 'Value' column
        if df_commodity['Value'].dtype == object:
            df_commodity['Value'] = df_commodity['Value'].str.replace(',', '').astype(float)
        
        df_commodity['Value'] = pd.to_numeric(df_commodity['Value'], errors='coerce')
        df_commodity['Year'] = df_commodity['Year'].str[:4].astype(int)
        df_commodity_grouped = df_commodity.groupby('Year').mean(numeric_only=True)
        ax.plot(df_commodity_grouped.index, df_commodity_grouped['Value'])
        average = df_commodity_grouped.loc[2009:2011].mean()
        ax.axhline(y=average['Value'], color='r', linestyle='--', label='2009-2011 average')
        ax.legend()
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # adjust subplot layout to accommodate the title
    plt.savefig(f'{path}media/{commodity}_mean_timeseries.png')
    
    plt.show()
    
    plt.close(fig)  # Close the figure to free up memory

# Plot of stocks to consumption ratio for each commodity
for commodity in commodities:
    fig, ax = plt.subplots(figsize=(15,10))
    plt.title(f'{commodity} - Stocks to Consumption Ratio')

    stocks_df_commodity = stocks_df[stocks_df['Commodity'] == commodity]
    stocks_df_commodity = stocks_df_commodity.melt(id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'], var_name='Year', value_name='Value')
    if stocks_df_commodity['Value'].dtype == object:
        stocks_df_commodity['Value'] = stocks_df_commodity['Value'].str.replace(',', '').astype(float)
    stocks_df_commodity['Year'] = stocks_df_commodity['Year'].str[:4].astype(int)

    consumption_df_commodity = consumption_df[consumption_df['Commodity'] == commodity]
    consumption_df_commodity = consumption_df_commodity.melt(id_vars=['Commodity', 'Attribute', 'Country', 'Unit Description'], var_name='Year', value_name='Value')
    if consumption_df_commodity['Value'].dtype == object:
        consumption_df_commodity['Value'] = consumption_df_commodity['Value'].str.replace(',', '').astype(float)
    consumption_df_commodity['Year'] = consumption_df_commodity['Year'].str[:4].astype(int)

    stocks_to_consumption_commodity = pd.merge(stocks_df_commodity, consumption_df_commodity, on=['Year', 'Commodity', 'Country'])
    stocks_to_consumption_commodity['ratio'] = stocks_to_consumption_commodity['Value_x'] / stocks_to_consumption_commodity['Value_y']

    # Compute the 20th percentile value
    percentile_15 = stocks_to_consumption_commodity['ratio'].quantile(0.15)
    percentile_15_values[commodity] = percentile_15

    ax.plot(stocks_to_consumption_commodity['Year'], stocks_to_consumption_commodity['ratio'])
    ax.axhline(y=percentile_15, color='r', linestyle='--', label='15th percentile')
    ax.legend()
    plt.savefig(f'{path}media/{commodity}_stocks_to_consumption_ratio.png')
    plt.close(fig)  # Close the figure to free up memory

# Create a dataframe and save the 20th percentile values to a CSV file
percentile_15_df = pd.DataFrame.from_dict(percentile_15_values, orient='index', columns=['15th Percentile'])
percentile_15_df.to_csv(f'{path}percentile_15_values.csv')


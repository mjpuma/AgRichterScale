import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    
    # Define the crop types
    crop_types = ["Wheat", "Corn"]  # Add more crops if needed
    
    # Specify the range of years (for the USDA PSD data)
    start_year = '2001/2002'  # '1975/1976'
    end_year = '2021/2022'
    
    # Extract the start and end years
    start = int(start_year.split('/')[0])
    end = int(end_year.split('/')[1])
    
    # Generate the range of years based on start_year and end_year
    years = [f"{i}/{i+1}" for i in range(start, end)]
    
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
    
    # Function to extract original reserves for a given crop from USDA PSD data
    def get_original_reserves(crop_type, data_path):
        # Load the USDA PSD reserves data
        stocks_csv_path = os.path.join(data_path, "USDAdata", "grains_world_usdapsd_endingstocks_jul142023.csv")
        stocks_df = pd.read_csv(stocks_csv_path)
        
        # Convert 'Commodity' column to string and handle missing values
        stocks_df['Commodity'] = stocks_df['Commodity'].astype(str).fillna('')
        
        # Simplify commodity names
        stocks_df['Commodity'] = stocks_df['Commodity'].apply(simplify_commodity_name)
        
        # Filter dataframe for the given crop type
        df_filtered = filter_dataframe(stocks_df, crop_type)
        
        if df_filtered.empty:
            raise ValueError(f"No data found for crop type: {crop_type}")
        
        # Extract reserves for the specified years
        df_filtered[years] = df_filtered[years].replace(',', '', regex=True).astype(float)
        reserves = df_filtered.set_index('Commodity')[years]
        
        # Sum reserves across all countries to obtain global reserves for each year
        global_reserves = reserves.sum()
        
        # Convert to a NumPy array
        reserves_series = global_reserves.values
        
        return reserves_series
    
    # Initialize dictionaries to store data
    original_reserves_dict = {}
    bootstrap_reserves_dict = {}
    
    for crop_type in crop_types:
        # Get original reserves
        original_reserves = get_original_reserves(crop_type, path)
        original_reserves_dict[crop_type] = original_reserves
        
        # Construct bootstrap filename
        bootstrap_IC_filename = f"bootstrap_initial_conditions_{crop_type}.csv"
        bootstrap_IC_path = os.path.join(path, bootstrap_IC_filename)
        
        # Verify that the bootstrap file exists
        verify_file_exists(bootstrap_IC_path)
        
        # Load bootstrap-sampled reserves
        bootstrap_reserves = pd.read_csv(bootstrap_IC_path)['Stocks (1000 metric tons)']
        bootstrap_reserves_dict[crop_type] = bootstrap_reserves.values  # Convert to NumPy array for consistency
    
    # Create subplots for each crop
    num_crops = len(crop_types)
    fig, axes = plt.subplots(nrows=num_crops, ncols=1, figsize=(10, 6 * num_crops))
    
    if num_crops == 1:
        axes = [axes]  # Make it iterable
    
    for ax, crop_type in zip(axes, crop_types):
        original_data = original_reserves_dict[crop_type]
        sampled_data = bootstrap_reserves_dict[crop_type]
        
        # Plot normalized histogram for the original reserves
        ax.hist(original_data, bins=30, alpha=0.5, label='Original Stocks', density=True, color='blue')
        
        # Plot normalized histogram for the bootstrap-sampled reserves
        ax.hist(sampled_data, bins=30, alpha=0.5, label=f'Sampled Values (n={len(sampled_data)})', density=True, color='orange')
        
        # Set titles and labels
        ax.set_title(f'Normalized Distribution of Original Stocks vs. Sampled Values for {crop_type}')
        ax.set_xlabel('Stocks (1000 mt)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    histogram_filename = f"Figure2_Histogram_Comparison_Original_vs_Sampled.png"
    histogram_path = os.path.join(path, histogram_filename)
    plt.savefig(histogram_path, dpi=300)
    print(f"Saved Figure 2 to {histogram_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()

# --------------------------- End of Script ---------------------------
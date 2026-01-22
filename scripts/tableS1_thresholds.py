#!/usr/bin/env python3
"""
Generate Supplementary Table S1: National Food Security Thresholds.

This script calculates the thresholds for the countries featured in the main paper
using the absolute latest USDA (Jan 2026) and SPAM 2020 data.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.usda import USDADataLoader, AgriPhaseThresholdCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of countries and their FIPS0 codes (as used in SPAM 2020)
COUNTRY_MAP = {
    'USA': 'US',
    'China': 'CH',
    'India': 'IN',
    'Brazil': 'BR',
    'France': 'FR',
    'Egypt': 'EG',
    'Australia': 'AS',
    'Argentina': 'AR'
}

def format_sci(val):
    """Format value in scientific notation for LaTeX: 1.23 \times 10^{14}"""
    if val == 0: return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    coeff = val / (10**exponent)
    return f"${coeff:.2f} \\times 10^{{{exponent}}}$"

def generate_table_s1():
    config = Config(crop_type='allgrain', root_dir='.')
    
    # 1. Load Global Thresholds from refreshed USDA data
    usda_loader = USDADataLoader()
    threshold_calc = AgriPhaseThresholdCalculator(usda_loader)
    
    # Use 2019-2021 range to align with SPAM 2020 base
    global_mt_thresholds = threshold_calc.calculate_consumption_thresholds('allgrain', year_range=(2019, 2021))
    
    # Get caloric conversion factor for USDA data (which is in 1000 MT)
    caloric_content = config.get_caloric_content()
    kcal_per_tmt = 1_000_000_000.0 * caloric_content 
    
    # Calculate Global Production for scaling (USDA 2020 total)
    prod_file = "USDAdata/grains_world_usdapsd_production_jan212026.csv"
    global_prod_df = pd.read_csv(prod_file)
    # The sum across years for "Allgrain" in the wide format
    global_total_prod_tmt = global_prod_df[global_prod_df['Commodity'] == 'Allgrain']['2020/2021'].iloc[0]
    
    # 2. Initialize SPAM Data Manager
    grid_manager = GridDataManager(config)
    grid_manager.load_spam_data()
    
    # 3. Build the table rows
    rows = []
    crop_indices = config.get_crop_indices()
    
    for display_name, iso3 in COUNTRY_MAP.items():
        logger.info(f"Processing {display_name} ({iso3})...")
        
        try:
            # Get grid cells for country
            p_cells, h_cells = grid_manager.get_grid_cells_by_iso3(iso3)
            
            if p_cells.empty:
                logger.warning(f"No grid cells found for {iso3}")
                continue
                
            # Get production (kcal) and harvest area (ha -> km2)
            country_prod_kcal = grid_manager.get_crop_production(p_cells, crop_indices, convert_to_kcal=True)
            country_harv_ha = grid_manager.get_crop_harvest_area(h_cells, crop_indices)
            country_harv_km2 = country_harv_ha / 100.0
            
            # For scaling, we need country production in same units as USDA (TMT)
            country_prod_mt = grid_manager.get_crop_production(p_cells, crop_indices, convert_to_kcal=False)
            country_prod_tmt = country_prod_mt / 1000.0
            
            # Scaling Factor: Ratio of country production to global production
            scaling_factor = country_prod_tmt / global_total_prod_tmt
            
            # Apply scaling to global thresholds
            t1_kcal = global_mt_thresholds['1 Month'] * kcal_per_tmt * scaling_factor
            t3_kcal = global_mt_thresholds['3 Months'] * kcal_per_tmt * scaling_factor
            ts_kcal = global_mt_thresholds['Total Stocks'] * kcal_per_tmt * scaling_factor
            
            rows.append({
                'Country': display_name,
                'Annual Production': format_sci(country_prod_kcal),
                'Harvest Area': f"{country_harv_km2:,.0f}",
                '1-Month Supply': format_sci(t1_kcal),
                '3-Month Supply': format_sci(t3_kcal),
                'Total Stocks': format_sci(ts_kcal)
            })
            
        except Exception as e:
            logger.error(f"Failed to process {display_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to process {country}: {e}")

    # 4. Generate LaTeX
    latex = "\\begin{table}[H]\n    \\centering\n"
    latex += "    \\caption{\\textbf{National Food Security Thresholds.} \n"
    latex += "    Consumption-based thresholds used in Figure~4 of the main text. Thresholds are calculated by scaling global USDA stocks-to-use ratios by each country's annual caloric production (derived from SPAM 2020 data). The ``Total Stocks'' column represents the theoretical systemic failure point if all reserves were fully depleted. Values assume proportional consumption relative to production; actual vulnerability depends on trade balances, storage infrastructure, and emergency reserves. Countries with production exceeding domestic consumption (e.g., Australia, Argentina) are net exporters; disruptions affect global markets rather than solely domestic supply.}\n"
    latex += "    \\label{tab:S1_country_stocks}\n    \\vspace{0.3cm}\n"
    latex += "    \\begin{tabular}{lccccc}\n        \\toprule\n"
    latex += "        \\textbf{Country} & \\textbf{Annual} & \\textbf{Harvest} & \\textbf{1-Month} & \\textbf{3-Month} & \\textbf{Total} \\\\\n"
    latex += "         & \\textbf{Production} & \\textbf{Area} & \\textbf{Supply} & \\textbf{Supply} & \\textbf{Stocks} \\\\\n"
    latex += "         & (kcal) & (km$^2$) & (kcal) & (kcal) & (kcal) \\\\\n        \\midrule\n"
    
    for r in rows:
        latex += f"        {r['Country']} & {r['Annual Production']} & {r['Harvest Area']} & {r['1-Month Supply']} & {r['3-Month Supply']} & {r['Total Stocks']} \\\\\n"
        
    latex += "        \\bottomrule\n    \\end{tabular}\n\\end{table}"
    
    print("\n--- GENERATED LATEX TABLE S1 ---\n")
    print(latex)
    
    # Save to a file as well
    with open("results/tableS1_supplementary.tex", "w") as f:
        f.write(latex)
    logger.info("Saved Table S1 to results/tableS1_supplementary.tex")

if __name__ == "__main__":
    generate_table_s1()

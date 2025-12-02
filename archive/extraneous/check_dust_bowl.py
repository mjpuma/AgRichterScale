import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dust_bowl():
    try:
        country_file = 'ancillary/DisruptionCountry.xls'
        logger.info(f"Loading {country_file}...")
        
        # Read all sheets
        xls = pd.ExcelFile(country_file)
        logger.info(f"Sheet names: {xls.sheet_names}")
        
        if 'DustBowl' in xls.sheet_names:
            df = pd.read_excel(country_file, sheet_name='DustBowl')
            print("\nDustBowl Sheet Content:")
            print(df.to_string())
            
            # Check for USA (code 240)
            usa_rows = df[df.iloc[:, 1] == 240] # Assuming 2nd column is Country Code
            if not usa_rows.empty:
                print("\nFound USA rows:")
                print(usa_rows.to_string())
            else:
                print("\nNo rows found with Country Code 240")
        else:
            print("DustBowl sheet not found!")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    check_dust_bowl()


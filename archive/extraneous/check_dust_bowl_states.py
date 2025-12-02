import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dust_bowl_states():
    try:
        state_file = 'ancillary/DisruptionStateProvince.xls'
        logger.info(f"Loading {state_file}...")
        
        if 'DustBowl' in pd.ExcelFile(state_file).sheet_names:
            df = pd.read_excel(state_file, sheet_name='DustBowl')
            print("\nDustBowl State Sheet Content:")
            print(df.to_string())
        else:
            print("DustBowl sheet not found in state file!")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    check_dust_bowl_states()


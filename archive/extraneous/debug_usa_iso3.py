# Debug ISO3 lookup for USA
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path('.').absolute()))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager

config = Config(crop_type='allgrain', root_dir='.')
config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')

grid_manager = GridDataManager(config)
prod_df, _ = grid_manager.load_spam_data()

print("Checking available ISO3 codes in SPAM data:")
print("Columns:", prod_df.columns)
if 'iso3' in prod_df.columns:
    print("iso3 present")
elif 'FIPS0' in prod_df.columns:
    print("FIPS0 present - this is often used as ISO3 proxy in SPAM")
    usa_rows = prod_df[prod_df['FIPS0'] == 'US']
    if not usa_rows.empty:
        print(f"Found {len(usa_rows)} rows with FIPS0='US'")
        print("Sample ADM1 names for US:")
        print(usa_rows['ADM1_NAME'].dropna().unique()[:10])
    else:
        print("No rows with FIPS0='US'")
        # Check what code USA uses
        usa_name_rows = prod_df[prod_df['ADM0_NAME'] == 'United States of America']
        if not usa_name_rows.empty:
            print(f"Found rows for 'United States of America'. FIPS0 code is: {usa_name_rows['FIPS0'].unique()}")
else:
    print("Neither iso3 nor FIPS0 found")


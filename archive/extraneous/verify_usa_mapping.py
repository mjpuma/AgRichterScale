import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path('.').absolute()))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper

def verify_usa_mapping():
    print("1. Initializing components...")
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    config.data_files['country_codes'] = Path('ancillary/CountryCode_Convert.xls')

    grid_manager = GridDataManager(config)
    mapper = SpatialMapper(config, grid_manager)
    
    print("2. Loading Grid Data...")
    grid_manager.load_spam_data()
    
    print("3. Testing USA Mapping...")
    # This uses the proper lookup logic
    prod_ids, harv_ids = mapper.map_country_to_grid_cells(240.0, code_system='GDAM ')
    
    print(f"   Mapped {len(prod_ids)} production cells for USA (Code 240)")
    
    if len(prod_ids) == 0:
        print("   FAILURE: USA mapping returned 0 cells")
        return
        
    print("4. Testing Dust Bowl State Mapping...")
    # Get the actual dataframe for inspection
    prod_df, _ = mapper.get_country_grid_cells_dataframe(240.0, code_system='GDAM ')
    
    unique_states = sorted(prod_df['ADM1_NAME'].dropna().unique().astype(str).tolist())
    print(f"   Found {len(unique_states)} states in USA data")
    print(f"   Sample states: {unique_states[:5]}")
    
    dust_bowl_states = ['Colorado', 'Kansas', 'Nebraska', 'New Mexico', 'North Dakota', 'Oklahoma', 'South Dakota', 'Texas']
    missing_states = []
    for state in dust_bowl_states:
        if state not in unique_states:
            missing_states.append(state)
            
    if missing_states:
        print(f"   MISSING STATES: {missing_states}")
        # Try case insensitive check
        print("   Checking case-insensitive matches...")
        unique_states_lower = [s.lower() for s in unique_states]
        for state in missing_states:
            if state.lower() in unique_states_lower:
                 print(f"   Found '{state}' as case-insensitive match")
            else:
                 print(f"   Still missing '{state}'")
    else:
        print("   ALL Dust Bowl states found!")

if __name__ == "__main__":
    verify_usa_mapping()


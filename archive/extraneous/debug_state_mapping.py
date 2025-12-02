# Test state mapping
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.data.spatial_mapper import SpatialMapper

config = Config(crop_type='allgrain', root_dir='.')
config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')

print("Loading Spatial Mapper...")
grid_manager = GridDataManager(config)
mapper = SpatialMapper(config, grid_manager)
# We need to load grid data to check state names
mapper.grid_manager.load_spam_data()

print("\nChecking available state names in SPAM data for USA (ISO3='USA')...")
prod_df, _ = mapper.grid_manager.get_grid_cells_by_iso3('USA')

unique_states = sorted(prod_df['ADM1_NAME'].dropna().unique().tolist())
print(f"Found {len(unique_states)} states in SPAM data:")
print(unique_states)

dust_bowl_states = ['Colorado', 'Kansas', 'Nebraska', 'New Mexico', 'North Dakota', 'Oklahoma', 'South Dakota', 'Texas']
print("\nVerifying Dust Bowl states:")
for state in dust_bowl_states:
    if state in unique_states:
        print(f"  ✓ {state} found")
    else:
        print(f"  ✗ {state} NOT found")
        # Fuzzy match check?
        matches = [s for s in unique_states if state in s]
        if matches:
            print(f"    Did you mean: {matches}?")


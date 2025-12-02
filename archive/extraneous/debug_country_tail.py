
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

sys.path.append(str(Path('.').absolute()))

from agririchter.core.config import Config
from agririchter.data.grid_manager import GridDataManager
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_country_tails():
    config = Config(crop_type='allgrain', root_dir='.')
    config.data_files['production'] = Path('spam2020V2r0_global_production/spam2020V2r0_global_production/spam2020V2r0_global_P_TA.csv')
    config.data_files['harvest_area'] = Path('spam2020V2r0_global_harvested_area/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv')
    
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    
    countries = [
        {'name': 'China', 'spam_name': 'China'},
        {'name': 'Brazil', 'spam_name': 'Brazil'}
    ]
    
    calculator = HPEnvelopeCalculatorV2(config)
    
    for country in countries:
        name = country['name']
        print(f"\n=== ANALYZING {name} ===")
        
        # Filter data
        mask = prod_df['ADM0_NAME'] == country['spam_name']
        c_prod = prod_df[mask]
        c_harv = harv_df[mask]
        
        print(f"Grid cells: {len(c_prod)}")
        
        # Run simplified preparation logic to see raw values
        P_arr, H_arr = calculator._prepare_cell_data(c_prod, c_harv)
        Y_arr = P_arr / H_arr
        
        print(f"Valid cells (P>0, H>0): {len(P_arr)}")
        
        # Sort by yield ascending (Lower Bound tail / Upper Bound start)
        # We want to check the TAIL of the UPPER BOUND, which corresponds to LOWEST yields.
        # Upper bound sorts descending. The tail is the low yield cells.
        
        indices = np.argsort(Y_arr) # Ascending
        # Lowest yields are at the beginning of indices
        
        k = 50 # Check lowest 50
        low_yield_idx = indices[:k]
        
        print(f"\nLowest {k} Yields (kcal/ha):")
        print(Y_arr[low_yield_idx])
        print(f"Corresponding H (ha):")
        print(H_arr[low_yield_idx])
        print(f"Corresponding P (kcal):")
        print(P_arr[low_yield_idx])
        
        # Check if P is extremely small but H is large?
        # Slope = P/H = Y
        
        # Calculate envelope
        envelope = calculator.calculate_hp_envelope(c_prod, c_harv)
        
        upper_H = envelope['upper_bound_harvest']
        upper_P = envelope['upper_bound_production']
        
        # Check slope at the end of Upper Bound
        # dP/dH should be decreasing and approaching min_yield
        
        print("\nUpper Bound Tail (Last 10 points):")
        for i in range(1, 11):
            idx = -1 * i
            if abs(idx) > len(upper_H): break
            h = upper_H[idx]
            p = upper_P[idx]
            if i > 1:
                prev_h = upper_H[idx+1]
                prev_p = upper_P[idx+1]
                slope = (prev_p - p) / (prev_h - h) if (prev_h - h) > 0 else 0
                print(f"  pt{idx}: H={h:.2f}, P={p:.2e}, dP/dH={slope:.2e}")
            else:
                print(f"  pt{idx}: H={h:.2f}, P={p:.2e}")

if __name__ == "__main__":
    analyze_country_tails()


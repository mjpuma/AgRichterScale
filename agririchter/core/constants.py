"""Constants and parameters for AgriRichter analysis."""

from typing import Dict, List, Tuple

# SPAM 2020 Crop Indices (0-based positions in crop array)
# Key grain crops: WHEA_A(56), RICE_A(41), MAIZ_A(25), BARL_A(13), MILL_A(26), SORG_A(44), OCER_A(27)
CROP_INDICES: Dict[str, List[int]] = {
    'allgrain': [56, 41, 25, 13, 26, 44, 27],  # WHEA_A, RICE_A, MAIZ_A, BARL_A, MILL_A, SORG_A, OCER_A
    'wheat': [56],                              # WHEA_A only  
    'rice': [41],                              # RICE_A only
    'maize': [25],                             # MAIZ_A only
}

# Crop names corresponding to SPAM indices
CROP_NAMES: List[str] = [
    'wheat', 'rice', 'maize', 'barley', 'millet_pearl',
    'millet_small', 'sorghum', 'cereals_other', 'potato', 'sweet_potato',
    'yam', 'cassava', 'roots_and_tubers_other', 'bean', 'chickpea', 'cowpea',
    'pigeonpea', 'lentil', 'pulses_other', 'soybean', 'groundnut',
    'coconut', 'oil_palm', 'sunflower', 'rapeseed', 'sesame_seed',
    'oil_crops_other', 'sugar_cane', 'sugar_beet', 'cotton',
    'fibers_other', 'coffee_arabica', 'coffee_robusta', 'cocoa',
    'tea', 'tobacco', 'banana', 'plantain', 'fruit_tropical',
    'fruit_temperate', 'vegetable', 'rest_of_crops'
]

# Caloric content for each crop type (kcal/g)
CALORIC_CONTENT: Dict[str, float] = {
    'Allgrain': 3.45,
    'Barley': 3.32,
    'Corn': 3.56,
    'Millet': 3.40,
    'MixedGrain': 3.40,
    'Oats': 3.85,
    'Rice': 3.60,
    'Rye': 3.19,
    'Sorghum': 3.43,
    'Wheat': 3.34
}

# AgriRichter Scale Thresholds (in kcal)
THRESHOLDS: Dict[str, Dict[str, float]] = {
    'allgrain': {
        'T1': 1.16E+14,
        'T2': 1.56E+15,
        'T3': 5.86E+15,
        'T4': 9.86E+15
    },
    'wheat': {
        'T1': 1.00E+14,
        'T2': 5.75E+14,
        'T3': 1.86E+15,
        'T4': 3.00E+15
    },
    'rice': {
        'T1': 1.31E+14,
        'T2': 3.16E+14,
        'T3': 1.29E+15,
        'T4': 2.13E+15
    },
    'maize': {
        'T1': 1.25E+14,
        'T2': 6.50E+14,
        'T3': 2.10E+15,
        'T4': 3.50E+15
    }
}

# Historical disruption events
HISTORICAL_EVENTS: List[str] = [
    'GreatFamine', 'Laki1783', 'NoSummer', 'PotatoFamine', 'Drought18761878',
    'SovietFamine1921', 'ChineseFamine1960',
    'DustBowl', 'SahelDrought2010',
    'MillenniumDrought', 'NorthKorea1990s', 'ENSO2015_2016',
    'Solomon', 'Vanuatu',
    'EastTimor', 'Haiti',
    'SierraLeone', 'Liberia',
    'Yemen', 'Ethiopia',
    'Laos', 'Bangladesh',
    'Syria'
]

# Unit conversion constants
GRAMS_PER_METRIC_TON: float = 1_000_000.0
HECTARES_TO_KM2: float = 0.01

# Grid parameters (5-minute resolution)
GRID_PARAMS: Dict[str, float] = {
    'ncols': 4320,
    'nrows': 2160,
    'xllcorner': -180.0,
    'yllcorner': -90.0,
    'cellsize': 0.0833333333333333
}

# Disruption area ranges for envelope calculation (km²) - MATLAB exact
DISRUPTION_RANGES: Dict[str, List[float]] = {
    'allgrain': [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000] + 
                list(range(10000, 7000400, 2000)),  # MATLAB: 1-7M km²
    'wheat': [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000] + 
             list(range(10000, 2205000, 2000)),     # MATLAB: 1-2.2M km²
    'rice': [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000] + 
            list(range(10000, 1400000, 2000)),      # MATLAB: 1-1.4M km² (corrected from your code)
    'maize': [1, 10, 50, 100, 200, 400, 600, 800, 1000, 4000, 6000, 8000] + 
             list(range(10000, 2000000, 2000)),     # Estimated range for maize: 1-2M km²
}

# Production range for AgriRichter scale plotting
PRODUCTION_RANGES: Dict[str, Tuple[float, float]] = {
    'allgrain': (10, 15.9),
    'wheat': (10, 15.2),
    'rice': (10, 15.0),
    'maize': (10, 15.5)
}

# Event visualization settings
EVENT_COLORS: Dict[str, Dict[str, str]] = {
    'wheat': {
        'GreatFamine': 'yellow', 'Laki1783': 'orange', 'NoSummer': 'orange',
        'Drought18761878': 'orange', 'SovietFamine1921': 'yellow', 'ChineseFamine1960': 'yellow',
        'DustBowl': 'yellow', 'SahelDrought2010': 'green', 'MillenniumDrought': 'green',
        'NorthKorea1990s': 'green', 'Solomon': 'green', 'Vanuatu': 'green',
        'EastTimor': 'green', 'Haiti': 'green', 'SierraLeone': 'green',
        'Liberia': 'green', 'Yemen': 'green', 'Ethiopia': 'green',
        'Laos': 'green', 'Bangladesh': 'green', 'Syria': 'green'
    },
    'rice': {
        'GreatFamine': 'green', 'Laki1783': 'orange', 'NoSummer': 'green',
        'Drought18761878': 'red', 'SovietFamine1921': 'green', 'ChineseFamine1960': 'orange',
        'DustBowl': 'green', 'SahelDrought2010': 'green', 'MillenniumDrought': 'green',
        'NorthKorea1990s': 'green', 'Solomon': 'green', 'Vanuatu': 'green',
        'EastTimor': 'green', 'Haiti': 'green', 'SierraLeone': 'green',
        'Liberia': 'green', 'Yemen': 'green', 'Ethiopia': 'green',
        'Laos': 'green', 'Bangladesh': 'yellow', 'Syria': 'green'
    },
    'allgrain': {
        'GreatFamine': 'yellow', 'Laki1783': 'orange', 'NoSummer': 'orange',
        'Drought18761878': 'red', 'SovietFamine1921': 'yellow', 'ChineseFamine1960': 'orange',
        'DustBowl': 'orange', 'SahelDrought2010': 'yellow', 'MillenniumDrought': 'yellow',
        'NorthKorea1990s': 'green', 'Solomon': 'green', 'Vanuatu': 'green',
        'EastTimor': 'green', 'Haiti': 'green', 'SierraLeone': 'green',
        'Liberia': 'green', 'Yemen': 'green', 'Ethiopia': 'green',
        'Laos': 'green', 'Bangladesh': 'yellow', 'Syria': 'green'
    },
    'maize': {
        'GreatFamine': 'green', 'Laki1783': 'orange', 'NoSummer': 'orange',
        'Drought18761878': 'red', 'SovietFamine1921': 'yellow', 'ChineseFamine1960': 'orange',
        'DustBowl': 'orange', 'SahelDrought2010': 'yellow', 'MillenniumDrought': 'green',
        'NorthKorea1990s': 'green', 'Solomon': 'green', 'Vanuatu': 'green',
        'EastTimor': 'green', 'Haiti': 'green', 'SierraLeone': 'green',
        'Liberia': 'green', 'Yemen': 'green', 'Ethiopia': 'green',
        'Laos': 'green', 'Bangladesh': 'green', 'Syria': 'green'
    }
}

EVENT_MARKERS: Dict[str, str] = {
    'green': 'd',    # diamond
    'yellow': 's',   # square
    'orange': 'o',   # circle
    'red': '^'       # triangle
}
# MATLAB Reference Output Generation Guide

## Task 12.1: Generate MATLAB Reference Outputs

This document provides instructions for generating MATLAB reference outputs to validate the Python implementation.

## Prerequisites

- MATLAB R2020a or later (recommended)
- Access to the original `AgriRichter_Events.m` file
- All required data files in the `ancillary/` directory
- SPAM 2010 data files (as used in original MATLAB code)

## MATLAB Version and Settings

**Document the following when running MATLAB:**
- MATLAB Version: _______________
- Operating System: _______________
- Date of execution: _______________
- SPAM data version used: SPAM 2010 v2r0

## Step-by-Step Instructions

### 1. Prepare MATLAB Environment

```matlab
% Set paths
ancillaryfold = './ancillary/';
inputfold = './';  % Adjust to your SPAM 2010 data location
outputfold = './matlab_outputs/';

% Create output directory
if ~exist(outputfold, 'dir')
    mkdir(outputfold);
end

% Set analysis parameters
year_start = 1961;
year_end = 2021;
```

### 2. Run MATLAB Code for Each Crop

#### Wheat
```matlab
crop = 'wheat';
[TotalEvent_wheat] = AgriRichter_Events(crop, ancillaryfold, inputfold, outputfold, year_start, year_end);

% Save results
event_names = {'GreatFamine', 'Laki1783', 'NoSummer', 'Drought18761878', ...
    'SovietFamine1921', 'ChineseFamine1960', 'DustBowl', 'SahelDrought2010', ...
    'MillenniumDrought', 'NorthKorea1990s', 'Solomon', 'Vanuatu', ...
    'EastTimor', 'Haiti', 'SierraLeone', 'Liberia', ...
    'Yemen', 'Ethiopia', 'Laos', 'Bangladesh', 'Syria'};

% Calculate magnitudes
magnitudes_wheat = log10(TotalEvent_wheat(:,1) * 0.01);  % ha to km^2, then log10

% Create results table
results_wheat = table(event_names', TotalEvent_wheat(:,1), TotalEvent_wheat(:,2), magnitudes_wheat, ...
    'VariableNames', {'event_name', 'harvest_area_loss_ha', 'production_loss_kcal', 'magnitude'});

% Save to CSV
writetable(results_wheat, [outputfold 'matlab_events_wheat_spam2010.csv']);
```

#### Rice
```matlab
crop = 'rice';
[TotalEvent_rice] = AgriRichter_Events(crop, ancillaryfold, inputfold, outputfold, year_start, year_end);

% Calculate magnitudes
magnitudes_rice = log10(TotalEvent_rice(:,1) * 0.01);

% Create results table
results_rice = table(event_names', TotalEvent_rice(:,1), TotalEvent_rice(:,2), magnitudes_rice, ...
    'VariableNames', {'event_name', 'harvest_area_loss_ha', 'production_loss_kcal', 'magnitude'});

% Save to CSV
writetable(results_rice, [outputfold 'matlab_events_rice_spam2010.csv']);
```

#### Allgrain
```matlab
crop = 'allgrain';
[TotalEvent_allgrain] = AgriRichter_Events(crop, ancillaryfold, inputfold, outputfold, year_start, year_end);

% Calculate magnitudes
magnitudes_allgrain = log10(TotalEvent_allgrain(:,1) * 0.01);

% Create results table
results_allgrain = table(event_names', TotalEvent_allgrain(:,1), TotalEvent_allgrain(:,2), magnitudes_allgrain, ...
    'VariableNames', {'event_name', 'harvest_area_loss_ha', 'production_loss_kcal', 'magnitude'});

% Save to CSV
writetable(results_allgrain, [outputfold 'matlab_events_allgrain_spam2010.csv']);
```

### 3. Save Generated Figures

The MATLAB code should generate figures. Save them with descriptive names:

```matlab
% Save figures in multiple formats
saveas(gcf, [outputfold 'matlab_hp_envelope_wheat.fig']);
saveas(gcf, [outputfold 'matlab_hp_envelope_wheat.png']);
saveas(gcf, [outputfold 'matlab_hp_envelope_wheat.eps']);
```

### 4. Document MATLAB Settings

Create a file `matlab_outputs/matlab_execution_info.txt` with:

```
MATLAB Execution Information
============================

MATLAB Version: [version output from 'ver']
Execution Date: [date]
Operating System: [OS info]
SPAM Data Version: SPAM 2010 v2r0

Data File Locations:
- Production: spam2010v2r0_global_P_TA.csv
- Harvest Area: spam2010v2r0_global_H_TA.csv
- Ancillary: ./ancillary/

Crops Processed:
- wheat
- rice  
- allgrain

Output Files Generated:
- matlab_events_wheat_spam2010.csv
- matlab_events_rice_spam2010.csv
- matlab_events_allgrain_spam2010.csv
- [list all figure files]

Notes:
[Any special considerations or issues encountered]
```

## Expected Output Files

After completing these steps, you should have:

```
matlab_outputs/
├── matlab_execution_info.txt
├── matlab_events_wheat_spam2010.csv
├── matlab_events_rice_spam2010.csv
├── matlab_events_allgrain_spam2010.csv
├── matlab_hp_envelope_wheat.png
├── matlab_hp_envelope_wheat.eps
├── matlab_hp_envelope_rice.png
├── matlab_hp_envelope_rice.eps
├── matlab_hp_envelope_allgrain.png
└── matlab_hp_envelope_allgrain.eps
```

## CSV File Format

Each CSV file should contain columns:
- `event_name`: Name of the historical event
- `harvest_area_loss_ha`: Disrupted harvest area in hectares
- `production_loss_kcal`: Production loss in kilocalories
- `magnitude`: AgriRichter magnitude (log10 of area in km²)

## Troubleshooting

### Common Issues

1. **Missing SPAM 2010 data**: The original MATLAB code uses SPAM 2010. Ensure you have the correct version.

2. **Path issues**: Adjust `inputfold` to point to your SPAM data location.

3. **Memory errors**: MATLAB may require significant memory for large datasets. Close other applications.

4. **Missing ancillary files**: Ensure all files in `ancillary/` directory are present.

## Next Steps

Once MATLAB reference outputs are generated:
1. Place CSV files in `matlab_outputs/` directory
2. Run Python comparison script: `python validate_matlab_comparison.py`
3. Review comparison report
4. Investigate any differences > 5%

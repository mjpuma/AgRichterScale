% =========================================================================
% AgRichter_main_v2024.m 
% Creates a Richter Scale for historic food production disruptions.
%
% User Options:
%   - Crop: Crop type options include 'allgrain', 'wheat', and 'rice'. 
%   - loss_factor: Fraction [0-1] indicating production loss in grid cells.
%
% Output:
%   Graphics saved in 'media' folder.
% =========================================================================

clear all; close all; clc;

% Add necessary paths (Update these paths according to your directory structure)
addpath('/Users/mjp38/GitHub/FSC-WorldModelers/cbrewer2/');
addpath('/Users/mjp38/GitHub/FSC-WorldModelers/colorspace/');

%% User inputs
year_start = 2010; 
year_end = 2010;
crop = 'rice';        % Options: 'allgrain', 'wheat', 'rice', 'corn'
loss_factor = 1;       % Fractional loss (0-1)

% Define the path
rootdir = '/Users/mjp38/Dropbox (Personal)/GitHub/AgriRichterScale/';  % Root directory

% Set up directories
ancillaryfold = [rootdir 'ancillary/'];
inputfold = [rootdir 'inputs/'];
outputfold = [rootdir 'outputs/'];
mediafold = [rootdir 'media/'];

% Define the range of years for USDA PSD data
start_year = '2001/2002';
end_year = '2021/2022';

% Extract the start and end years
start = str2double(start_year(1:4));
finish = str2double(end_year(6:9));

% Generate the range of years
years = cellstr(strcat(num2str((start:finish-1)'), '/', num2str((start+1:finish)')));

%% Load USDA PSD data and calculate thresholds
% Load data from csv files
opts = detectImportOptions([rootdir 'USDAdata/grains_world_usdapsd_endingstocks_jul142023_kcal.csv']);
opts.VariableNamingRule = 'preserve';
stocks_df = readtable([rootdir 'USDAdata/grains_world_usdapsd_endingstocks_jul142023_kcal.csv'], opts);

opts = detectImportOptions([rootdir 'USDAdata/grains_world_usdapsd_production_jul142023_kcal.csv']);
opts.VariableNamingRule = 'preserve';
production_df = readtable([rootdir 'USDAdata/grains_world_usdapsd_production_jul142023_kcal.csv'], opts);

opts = detectImportOptions([rootdir 'USDAdata/grains_world_usdapsd_consumption_jul142023_kcal.csv']);
opts.VariableNamingRule = 'preserve';
consumption_df = readtable([rootdir 'USDAdata/grains_world_usdapsd_consumption_jul142023_kcal.csv'], opts);

% Filter dataframes for the chosen crop
stocks = filter_dataframe(stocks_df, crop);
production = filter_dataframe(production_df, crop);
consumption = filter_dataframe(consumption_df, crop);

% Get the column names that correspond to years
year_columns = stocks.Properties.VariableNames(cellfun(@(x) ~isnan(str2double(x(1:4))), stocks.Properties.VariableNames));

% Select only the relevant years
stocks_numeric = table2array(stocks(:, year_columns));
production_numeric = table2array(production(:, year_columns));
consumption_numeric = table2array(consumption(:, year_columns));

% Compute the stock-to-use ratio
stocks_to_consumption_ratio = stocks_numeric ./ consumption_numeric;

% Calculate percentiles and medians
SUR_15percentile = calculate_percentile(stocks_to_consumption_ratio, 15);
SUR_median = calculate_percentile(stocks_to_consumption_ratio, 50);
stocks_median = calculate_percentile(stocks_numeric, 50);
production_median = calculate_percentile(production_numeric, 50);
consumption_median = calculate_percentile(consumption_numeric, 50);

% Compute the thresholds
alphaC = 0.5;
Threshold_AgriPhase2 = stocks_median - SUR_15percentile * consumption_median;
Threshold_AgriPhase3 = SUR_15percentile * consumption_median;
Threshold_AgriPhase4 = SUR_median * consumption_median + alphaC * production_median;
Threshold_AgriPhase5 = SUR_median * consumption_median + production_median;

%% Disruption Event list
EventList = {
    'GreatFamine', 'Laki1783', 'NoSummer', 'Drought18761878',...
    'SovietFamine1921', 'ChineseFamine1960', 'DustBowl', 'SahelDrought2010',...
    'MillenniumDrought', 'NorthKorea1990s', 'Solomon', 'Vanuatu',...
    'EastTimor', 'Haiti', 'SierraLeone', 'Liberia',...
    'Yemen', 'Ethiopia', 'Laos', 'Bangladesh', 'Syria'};


% Caloric content per gram for each crop (kcal/g)
calories_cropprod_dict = struct(...
    'Allgrain', 3.45, ...  % Average for grains
    'Barley', 3.32, ...
    'Corn', 3.56, ...
    'Millet', 3.40, ...
    'Mixed_Grain', 3.40, ...
    'Oats', 3.85, ...
    'Rice', 3.60, ...
    'Rye', 3.19, ...
    'Sorghum', 3.43, ...
    'Wheat', 3.34 ...
);

%% Crop options and thresholds
if strcmp(crop, 'allgrain')
    i_crop = 1:8;  % Indices for grain crops in SPAM dataset
    title_map = 'Grains (log_{10} kcal)';
    crop_choice = 'allgrain';
    calories_cropprod = calories_cropprod_dict.Allgrain;
elseif strcmp(crop, 'wheat')
    i_crop = 1;    % Index for wheat in SPAM dataset
    title_map = 'Wheat (log_{10} kcal)';
    crop_choice = 'wheat';
    calories_cropprod = calories_cropprod_dict.Wheat;
elseif strcmp(crop, 'rice')
    i_crop = 2;    % Index for rice in SPAM dataset
    title_map = 'Rice (log_{10} kcal)';
    crop_choice = 'rice';
    calories_cropprod = calories_cropprod_dict.Rice;
elseif strcmp(crop, 'corn')
    i_crop = 3;    % Index for corn in SPAM dataset (verify this is correct)
    title_map = 'Corn (log_{10} kcal)';
    crop_choice = 'corn';
    calories_cropprod = calories_cropprod_dict.Corn;
elseif strcmp(crop, 'barley')
    i_crop = 4;    % Index for barley in SPAM dataset (verify this is correct)
    title_map = 'Barley (log_{10} kcal)';
    crop_choice = 'barley';
    calories_cropprod = calories_cropprod_dict.Barley;
elseif strcmp(crop, 'millet')
    i_crop = 5;    % Index for millet in SPAM dataset (verify this is correct)
    title_map = 'Millet (log_{10} kcal)';
    crop_choice = 'millet';
    calories_cropprod = calories_cropprod_dict.Millet;
elseif strcmp(crop, 'sorghum')
    i_crop = 7;    % Index for sorghum in SPAM dataset (verify this is correct)
    title_map = 'Sorghum (log_{10} kcal)';
    crop_choice = 'sorghum';
    calories_cropprod = calories_cropprod_dict.Sorghum;
else
    error('Crop not supported. Please choose from ''allgrain'', ''wheat'', ''rice'', ''corn'', ''barley'', ''millet'', or ''sorghum''.');
end

% Remove crops not selected
removelist = setdiff(1:42, i_crop);

%% Plot Balance Time Series
figure('Position', [100, 100, 1200, 1000]);

% Convert year strings to datetime for better x-axis labeling
year_datetimes = datetime(year_columns, 'InputFormat', 'yyyy/yyyy');

% Plot Stocks to Consumption Ratio
subplot(2,2,1)
plot(year_datetimes, stocks_to_consumption_ratio)
hold on
yline(SUR_median, 'k--', 'LineWidth', 0.7)
yline(SUR_15percentile, 'r--', 'LineWidth', 0.7)
title(['Stocks to Use Ratio (SUR) - ' crop])
xticks(year_datetimes(1:2:end))
xtickangle(45)
ylabel('Ratio')
grid on

% Plot Stocks
subplot(2,2,2)
plot(year_datetimes, stocks_numeric)
hold on
yline(stocks_median, 'k--', 'LineWidth', 0.7)
title(['Stocks - ' crop])
xticks(year_datetimes(1:2:end))
xtickangle(45)
ylabel('kcal')
grid on

% Plot Production
subplot(2,2,3)
plot(year_datetimes, production_numeric)
hold on
yline(production_median, 'k--', 'LineWidth', 0.7)
title(['Production - ' crop])
xticks(year_datetimes(1:2:end))
xtickangle(45)
ylabel('kcal')
grid on

% Plot Consumption
subplot(2,2,4)
plot(year_datetimes, consumption_numeric)
hold on
yline(consumption_median, 'k--', 'LineWidth', 0.7)
title(['Consumption - ' crop])
xticks(year_datetimes(1:2:end))
xtickangle(45)
ylabel('kcal')
grid on

% Adjust layout
sgtitle(['Balance Time Series - ' crop], 'FontSize', 16)

% Adjust subplot spacing
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
p = get(gcf, 'Position');
p(3) = p(3) * 1.1;
p(4) = p(4) * 1.1;
set(gcf, 'Position', p)
set(gcf, 'PaperPositionMode', 'auto')
spacing = 0.09;
subplotPositions = {
    [spacing, 0.55, 0.45-spacing, 0.4],
    [0.55, 0.55, 0.45-spacing, 0.4],
    [spacing, 0.08, 0.45-spacing, 0.4],
    [0.55, 0.08, 0.45-spacing, 0.4]
};
for i = 1:4
    set(gcf, 'CurrentAxes', subplot(2,2,i))
    set(gca, 'Position', subplotPositions{i})
end

% Save and close figure
saveas(gcf, [mediafold 'BalanceTimeSeries_' crop '.png'])
close(gcf)


%% Set up directories
ancillaryfold = [rootdir 'ancillary/'];
inputfold = [rootdir 'inputs/'];
outputfold = [rootdir 'outputs/'];
mediafold = [rootdir 'media/'];

% Conversion constants
gramsperMetricTon = 1e6;  % 1 metric ton = 1,000,000 grams

%% Load crop nutritional information
% Load caloric content per crop from Excel file
[~, ~, raw] = xlsread([ancillaryfold 'Nutrition_SPAMcrops.xls'], 'Sheet1');
raw = raw(2:end, 2:end);
NutritionperCrop = reshape([raw{:}], size(raw));
NutritionperCrop(NutritionperCrop == -9999) = NaN;
clearvars raw;

% Crop titles corresponding to SPAM dataset indices
crop_filetitle = {'wheat','rice','maize','barley','millet_pearl',...
    'millet_small','sorghum','cereals_other','potato','sweet_potato',...
    'yam','cassava','roots_and_tubers_other','bean','chickpea','cowpea',...
    'pigeonpea','lentil','pulses_other','soybean','groundnut',...
    'coconut','oil_palm','sunflower','rapeseed','sesame_seed',...
    'oil_crops_other','sugar_cane','sugar_beet','cotton',...
    'fibers_other','coffee_arabica','coffee_robusta','cocoa',...
    'tea','tobacco','banana','plantain','fruit_tropical',...
    'fruit_temperate','vegetable','rest_of_crops'};

% Remove crops not selected
crop_filetitle(removelist) = [];
NutritionperCrop(removelist, :) = [];

% Convert kcal per 100g to kcal per gram
Crop_kcalpergram = NutritionperCrop(:, 1) / 100;

%% Load SPAM data (Production and Harvest Area)
% Load production data with 'x' and 'y' coordinates
[spamPTA, crop_columns] = loadSpamDataWithCoords([inputfold 'spam2010v2r0_global_P_TA.csv'], i_crop);
[spamHTA, ~] = loadSpamDataWithCoords([inputfold 'spam2010v2r0_global_H_TA.csv'], i_crop);

%% Create grid for mapping
% Grid parameters
ncols = 4320;
nrows = 2160;
xllcorner = -180;
yllcorner = -90;
cellsize = 0.0833333333333333;

% Create coordinate vectors
x = xllcorner + cellsize * (0:(ncols - 1));
y = yllcorner + cellsize * ((nrows - 1):-1:0);

% Create grid of coordinates
[lonGrid, latGrid] = meshgrid(x, y);

%% Process and map production data
% Convert production from metric tons to grams and then to kcal
CropSubset_Prod = spamPTA{:, crop_columns} * gramsperMetricTon;
CropSubset_Prod = CropSubset_Prod .* Crop_kcalpergram';

% Convert harvested areas from hectares to km^2
CropSubset_HA = spamHTA{:, crop_columns} * 0.01;

% Sum to get total values for selected group of crops
TotalProduction = sum(CropSubset_Prod, 2);
TotalHarvest = sum(CropSubset_HA, 2);

% Initialize production grid
prod_grid = nan(nrows, ncols);

% Apply log10 transformation to production data
log_prod_grid = log10(TotalProduction + 1);  % Add 1 to avoid log(0)

% Assign values to prod_grid using the closest lat/lon
for i = 1:height(spamPTA)
    lat_idx = findClosest(y, spamPTA.y(i));
    lon_idx = findClosest(x, spamPTA.x(i));
    prod_grid(lat_idx, lon_idx) = log_prod_grid(i);
end

%% Plot Global Production Map using Mapping Toolbox
figure;

% Define a special value for NaNs
nan_value = min(prod_grid(~isinf(prod_grid) & ~isnan(prod_grid))) - 1;

% Set NaN and Inf values to the special value
prod_grid(~isfinite(prod_grid)) = nan_value;

% Set colormap (yellow to green) using cbrewer2
if exist('cbrewer2', 'file')
    cmap = cbrewer2('seq', 'YlGn', 256);
else
    warning('cbrewer2 not found. Using default colormap.');
    cmap = parula(256);
end

% Add an extra color for NaN values at the beginning of the colormap (white)
cmap = [[1 1 1]; cmap];

% Set up a map axes for a Robinson projection and remove the grid
axesm('robinson', 'Grid', 'off', 'Frame', 'on', 'MeridianLabel', 'off', 'ParallelLabel', 'off');

% Set the color of the ocean
setm(gca, 'FFaceColor', [0.7 0.9 1]);

% Display the data
geoshow(latGrid, lonGrid, prod_grid, 'DisplayType', 'texturemap');

% Set colormap
colormap(gca, cmap);

% Get color limits, excluding the nan_value
cmin = min(prod_grid(:));
cmax = max(prod_grid(:));
caxis([cmin + 1, cmax]);

% Add colorbar
h = colorbar;
h.FontSize = 16;

% Add title
title(title_map, 'FontSize', 20);

% Add coastlines
load coastlines
geoshow(coastlat, coastlon, 'Color', 'black', 'LineWidth', 1);

% Adjust axes properties
set(gca, 'LineWidth', 1, 'Box', 'off');
axis off;

% Remove the outer box around the plot
ax = gca;
ax.OuterPosition = [0 0 1 1];

% Save figure as SVG and other formats
saveas(gcf, [mediafold 'Global_Production_Map_' crop '.svg'], 'svg');
saveas(gcf, [mediafold 'Global_Production_Map_' crop '.png']);
saveas(gcf, [mediafold 'Global_Production_Map_' crop '.eps']);
close(gcf);  % Close figure to free memory

disp('Global Production Map plotted and saved using Mapping Toolbox.');

%% Prepare data for disrupted H-P Envelope
% Create matrix with Harvest Area, Production, and Yield
HPmatrix = [TotalHarvest(:), TotalProduction(:), TotalProduction(:) ./ TotalHarvest(:)];

% Remove NaNs and zeros
HPmatrix = HPmatrix(~any(isnan(HPmatrix) | HPmatrix(:, 2) == 0, 2), :);

% Sort by yield
HPmatrix_sorted = sortrows(HPmatrix, 3);

% Compute cumulative sums for lower and upper bounds
HPmatrix_cumsum_SmallLarge = cumsum(HPmatrix_sorted);
HPmatrix_cumsum_LargeSmall = cumsum(flipud(HPmatrix_sorted));

%% Compute Historical Losses
% Note: The function 'AgriRichter_Events' is assumed to exist.
% Ensure that this function returns a matrix 'TotalEvent' with columns:
% [Disrupted Harvest Area (km^2), Lost Production (kcal)]
TotalEvent = AgriRichter_Events(crop, ancillaryfold, inputfold, outputfold, year_start, year_end);

% Convert Harvest Area from hectares to km^2
TotalEvent(:, 1) = TotalEvent(:, 1) * 0.01;

%% Define IPC Phase Colors
IPC_Phase_Colors = struct(...
    'Phase2', [1, 1, 0], ...          % Yellow
    'Phase3', [1, 0.6471, 0], ...     % Orange
    'Phase4', [1, 0, 0], ...          % Red
    'Phase5', [0.5, 0, 0]);           % Dark Red

%% First Figure: AgRichter Scale Plot (Uniform Production)
% Calculate average production per unit area (kcal per km^2)
TotalGlobalProduction = sum(TotalProduction); % Total global production (kcal)
TotalGlobalHarvestArea = sum(TotalHarvest);   % Total global harvest area (km^2)
AverageProductionPerArea = TotalGlobalProduction / TotalGlobalHarvestArea; % kcal per km^2

% Define a range of disrupted harvest areas (A_H) in km^2
A_H_range = logspace(2, 7, 10000); % From 10^2 to 10^7 km^2
M_D = log10(A_H_range);            % Magnitude

% Compute theoretical lost production (L_F) for each A_H assuming uniform production
L_F_theoretical = A_H_range * AverageProductionPerArea;

% Proceed to plotting
figure;
hold on;

% Plot the theoretical line
plot(M_D, L_F_theoretical, '-k', 'LineWidth', 2, 'DisplayName', 'Theoretical Line');

% Plot historical events
M_D_event = log10(TotalEvent(:, 1)); % Magnitude of events
L_F_event = loss_factor * TotalEvent(:, 2); % Lost production of events

scatter(M_D_event, L_F_event, 100, 'r', 'filled', 'DisplayName', 'Historical Events');

% Label the markers with the event names from EventList
for i = 1:length(M_D_event)
    text(M_D_event(i), L_F_event(i), EventList{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

% Plot horizontal threshold lines with AgriPhase naming
yline(Threshold_AgriPhase2, '--', 'Color', IPC_Phase_Colors.Phase2, 'LineWidth', 2, 'DisplayName', 'AgriPhase 2');
yline(Threshold_AgriPhase3, '--', 'Color', IPC_Phase_Colors.Phase3, 'LineWidth', 2, 'DisplayName', 'AgriPhase 3');
yline(Threshold_AgriPhase4, '--', 'Color', IPC_Phase_Colors.Phase4, 'LineWidth', 2, 'DisplayName', 'AgriPhase 4');
yline(Threshold_AgriPhase5, '--', 'Color', IPC_Phase_Colors.Phase5, 'LineWidth', 2, 'DisplayName', 'AgriPhase 5');

% Add legend with smaller font size and positioned in the southeast
legend('Theoretical Line', 'Historical Events', 'AgriPhase 2', ...
       'AgriPhase 3', 'AgriPhase 4', 'AgriPhase 5', ...
       'Location', 'southeast', 'FontSize', 12);

% Set axes labels and title
set(gca, 'FontSize', 20);
xlabel('Magnitude ($$M_D = \log_{10} A_H$$)', 'FontSize', 20, 'Interpreter', 'latex');
ylabel('Lost Production (kcal)', 'FontSize', 20);
title(['AgRichter Scale - ' crop], 'FontSize', 24);
box on;

% Set Y-axis to logarithmic scale
set(gca, 'YScale', 'log');

% Set consistent y-axis limits
ylim([1e10, 1.62e16]); % Adjusted to accommodate maximum log10(kcal) ~16.2

% Save figure with crop name included
print(gcf, '-depsc', '-vector', [mediafold 'RichterScale_' crop '.eps']);
print(gcf, '-djpeg', [mediafold 'RichterScale_' crop '.jpg']);
close(gcf);  % Close figure to free memory

%% Second Figure: Disrupted H-P Envelope
disp('Preparing data for envelope plotting...');

% Calculate cumulative production from least and most productive areas
A_H_cum_lower = HPmatrix_cumsum_SmallLarge(:, 1);
L_F_cum_lower = HPmatrix_cumsum_SmallLarge(:, 2);

A_H_cum_upper = HPmatrix_cumsum_LargeSmall(:, 1);
L_F_cum_upper = HPmatrix_cumsum_LargeSmall(:, 2);

% Remove NaN or zero values
valid_idx_lower = isfinite(A_H_cum_lower) & isfinite(L_F_cum_lower) & (A_H_cum_lower > 0) & (L_F_cum_lower > 0);
A_H_cum_lower = A_H_cum_lower(valid_idx_lower);
L_F_cum_lower = L_F_cum_lower(valid_idx_lower);

valid_idx_upper = isfinite(A_H_cum_upper) & isfinite(L_F_cum_upper) & (A_H_cum_upper > 0) & (L_F_cum_upper > 0);
A_H_cum_upper = A_H_cum_upper(valid_idx_upper);
L_F_cum_upper = L_F_cum_upper(valid_idx_upper);

% Downsample data to reduce array sizes
downsample_factor = 100; % Adjust as needed
A_H_cum_lower = A_H_cum_lower(1:downsample_factor:end);
L_F_cum_lower = L_F_cum_lower(1:downsample_factor:end);
A_H_cum_upper = A_H_cum_upper(1:downsample_factor:end);
L_F_cum_upper = L_F_cum_upper(1:downsample_factor:end);

% Compute M_D values
epsilon = 1e-10; % Small value to avoid log of zero
M_D_lower = log10(A_H_cum_lower + epsilon);
M_D_upper = log10(A_H_cum_upper + epsilon);

% Ensure M_D_lower and M_D_upper have the same length
numPoints = min(length(M_D_lower), length(M_D_upper));
M_D_lower = M_D_lower(1:numPoints);
L_F_cum_lower = L_F_cum_lower(1:numPoints);
M_D_upper = M_D_upper(1:numPoints);
L_F_cum_upper = L_F_cum_upper(1:numPoints);

% Prepare data for envelope
X_env = [M_D_lower; flipud(M_D_upper)];
Y_env = [L_F_cum_lower; flipud(L_F_cum_upper)];

% Remove any remaining NaNs or Infs
valid_idx = isfinite(X_env) & isfinite(Y_env);
X_env = X_env(valid_idx);
Y_env = Y_env(valid_idx);

% Plot the envelope
figure;
hold on;

% Fill the area between the lower and upper cumulative production curves
fill(X_env, Y_env, [0.8 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Envelope');

% Plot the upper and lower bounds
plot(M_D_upper, L_F_cum_upper, '-k', 'LineWidth', 2, 'DisplayName', 'Upper Bound');
plot(M_D_lower, L_F_cum_lower, '-b', 'LineWidth', 2, 'DisplayName', 'Lower Bound');

% Plot historical events
scatter(M_D_event, L_F_event, 100, 'r', 'filled', 'DisplayName', 'Historical Events');

% Label the markers with the event names from EventList
for i = 1:length(M_D_event)
    text(M_D_event(i), L_F_event(i), EventList{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

% Plot horizontal threshold lines with AgriPhase naming
yline(Threshold_AgriPhase2, '--', 'Color', IPC_Phase_Colors.Phase2, 'LineWidth', 2, 'DisplayName', 'AgriPhase 2');
yline(Threshold_AgriPhase3, '--', 'Color', IPC_Phase_Colors.Phase3, 'LineWidth', 2, 'DisplayName', 'AgriPhase 3');
yline(Threshold_AgriPhase4, '--', 'Color', IPC_Phase_Colors.Phase4, 'LineWidth', 2, 'DisplayName', 'AgriPhase 4');
yline(Threshold_AgriPhase5, '--', 'Color', IPC_Phase_Colors.Phase5, 'LineWidth', 2, 'DisplayName', 'AgriPhase 5');

% Add legend with smaller font size and positioned in the southeast
legend('Envelope', 'Upper Bound', 'Lower Bound', 'Historical Events', ...
       'AgriPhase 2', 'AgriPhase 3', ...
       'AgriPhase 4', 'AgriPhase 5', ...
       'Location', 'southeast', 'FontSize', 12);

% Set axes labels and title
set(gca, 'FontSize', 20);
xlabel('Magnitude ($$M_D = \log_{10} A_H$$)', 'FontSize', 20, 'Interpreter', 'latex');
ylabel('Lost Production (kcal)', 'FontSize', 20);
title(['Disrupted H-P Envelope - ' crop], 'FontSize', 24);
box on;

% Set Y-axis to logarithmic scale
set(gca, 'YScale', 'log');

% Set consistent y-axis limits
ylim([1e10, 1.62e16]); % Adjusted to accommodate maximum log10(kcal) ~16.2

% Adjust x-axis limits if necessary
xlim([2, 7]);  % As per your requirement

% Save figure with crop name included
print(gcf, '-depsc', '-vector', [mediafold 'Production_vs_HarvestArea_' crop '.eps']);
print(gcf, '-djpeg', [mediafold 'Production_vs_HarvestArea_' crop '.jpg']);
close(gcf);  % Close figure to free memory

disp('Disrupted H-P Envelope plotted.');

%% Save Event Data
EventListSave = [EventList'; {'Total Global Production'}];
Y2save = [TotalEvent(:, 2); sum(TotalProduction)];
output_table = table(EventListSave, Y2save, 'VariableNames', {'Event', 'Production_Lost_kcal'});
filename = [rootdir 'outputs/LostProd_Events_spam2010V2r0_' crop '.csv'];
writetable(output_table, filename);

%% Supporting Functions

function value = fetchValue(thresholdID, crop_choice, thresholds, allgrain, wheat, rice)
    % Fetch threshold value based on crop choice
    idx = strcmp(thresholds, thresholdID);
    if ~any(idx)
        error(['Threshold ID ' thresholdID ' not found.']);
    end
    switch crop_choice
        case 'allgrain'
            value = allgrain(idx);
        case 'wheat'
            value = wheat(idx);
        case 'rice'
            value = rice(idx);
        otherwise
            error('Invalid crop choice.');
    end
    if isempty(value)
        error(['Threshold value for ' thresholdID ' and crop ' crop_choice ' is empty.']);
    end
end

function idx = findClosest(vector, value)
    % Find the index of the closest value in vector to the given value
    [~, idx] = min(abs(vector - value));
end

function [dataTable, crop_columns] = loadSpamDataWithCoords(filename, i_crop)
    % Load SPAM data including 'x' and 'y' coordinates and extract relevant crop columns

    % Create import options
    opts = delimitedTextImportOptions("NumVariables", 57);

    % Specify range and delimiter
    opts.DataLines = [2, Inf];
    opts.Delimiter = ",";

    % Specify column names
    opts.VariableNames = {'iso3', 'prod_level', 'alloc_key', 'cell5m', 'x', 'y', ...
        'rec_type', 'tech_type', 'unit', 'whea_a', 'rice_a', 'maiz_a', 'barl_a', ...
        'pmil_a', 'smil_a', 'sorg_a', 'ocer_a', 'pota_a', 'swpo_a', 'yams_a', ...
        'cass_a', 'orts_a', 'bean_a', 'chic_a', 'cowp_a', 'pige_a', 'lent_a', ...
        'opul_a', 'soyb_a', 'grou_a', 'cnut_a', 'oilp_a', 'sunf_a', 'rape_a', ...
        'sesa_a', 'ooil_a', 'sugc_a', 'sugb_a', 'cott_a', 'ofib_a', 'acof_a', ...
        'rcof_a', 'coco_a', 'teas_a', 'toba_a', 'bana_a', 'plnt_a', 'trof_a', ...
        'temf_a', 'vege_a', 'rest_a', 'crea_date', 'year_data', 'source', ...
        'name_cntr', 'name_adm1', 'name_adm2'};

    % Specify variable types
    opts.VariableTypes = [{'categorical'}, repmat({'double'}, 1, 5), repmat({'categorical'}, 1, 3), ...
        repmat({'double'}, 1, 42), 'datetime', 'double', repmat({'categorical'}, 1, 4)];

    % Set variable options
    % Set InputFormat for datetime variable
    opts = setvaropts(opts, 'crea_date', 'InputFormat', 'MM/dd/yy hh:mm:ss aa');

    % For categorical variables, set EmptyFieldRule
    categoricalVars = {'iso3', 'rec_type', 'tech_type', 'unit', 'source', 'name_cntr', 'name_adm1', 'name_adm2'};
    opts = setvaropts(opts, categoricalVars, 'EmptyFieldRule', 'auto');

    % Import the data
    dataTable = readtable(filename, opts);

    % Adjust column indices for crops
    crop_columns = i_crop + 9;  % Adjust column indices for crops (crop data starts at column 10)
end

function df_filtered = filter_dataframe(df, crop_type)
    % Special handling for rice
    if strcmpi(crop_type, 'rice')
        crop_type = 'Rice, Milled';
    end
    
    % Filter the dataframe for rows where 'Commodity' contains crop_type
    df_filtered = df(contains(df.Commodity, crop_type, 'IgnoreCase', true), :);
end

function percentile_value = calculate_percentile(data, percentile)
    percentile_value = prctile(data, percentile);
end
function [TotalEvent] = AgriRichter_Events(crop, ancillaryfold, inputfold, outputfold, year_start, year_end)
% AgriRichter_Events.m
% Function to compute the contemporary losses from historic events.    
%
% Inputs:
%   crop - string: crop type. 'allgrain', 'wheat', 'rice', or 'corn' currently supported
%   ancillaryfold - string: path to ancillary folder
%   inputfold - string: path to input folder
%   outputfold - string: path to output folder
%   year_start - integer: start year for analysis
%   year_end - integer: end year for analysis
%
% Output: 
%   TotalEvent - matrix: losses per event
%     Column 1: disrupted harvest area (hectares)
%     Column 2: production losses (kcal)

% Constants
GRAMS_PER_METRIC_TON = 1e6;

%% Crop options
switch crop
    case 'allgrain'
        i_crop = [1:8, 37:42];
    case 'wheat'
        i_crop = 1;
    case 'rice'
        i_crop = 2;
    case 'corn'
        i_crop = 3;
    otherwise
        error('Crop not supported. Choose ''allgrain'', ''wheat'', ''rice'', or ''corn''.');
end

% Specify crops that should be removed (from full list)
removelist = setdiff(1:42, i_crop);

%% Disruption list
EventList = {'GreatFamine', 'Laki1783', 'NoSummer', 'Drought18761878', ...
    'SovietFamine1921', 'ChineseFamine1960', 'DustBowl', 'SahelDrought2010', ...
    'MillenniumDrought', 'NorthKorea1990s', 'Solomon', 'Vanuatu', ...
    'EastTimor', 'Haiti', 'SierraLeone', 'Liberia', ...
    'Yemen', 'Ethiopia', 'Laos', 'Bangladesh', 'Syria'};

% Number of disturbance events and countries/provinces
num_disturb = length(EventList);
longestlist = length(EventList) + 14;  % Update if a longer country list needed

%% Crop-related parameters
% Metric ton: unit of weight equal to 1,000 kilograms, or approximately 2,204.6 pounds.
gramsperMetricTon = 1000000;

%% File VARIABLE and crop titles
field = {'global_P_TA','global_H_TA','physical-area','yield'};

crop_filetitle ={'wheat','rice','maize','barley','millet_pearl',...
    'millet_small','sorghum','cereals_other','potato','sweet_potato',...
    'yam','cassava','roots_and_tubers_other','bean','chickpea','cowpea',...
    'pigeonpea','lentil','pulses_other','soybean','groundnut',...
    'coconut','oil_palm','sunflower','rapeseed','sesame_seed',...
    'oil_crops_other','sugar_cane','sugar_beet','cotton',...
    'fibers_other','coffee_arabica','coffee_robusta','cocoa',...
    'tea','tobacco','banana','plantain','fruit_tropical',...
    'fruit_temperate','vegetable','rest_of_crops'};

%% Food Codes from SPAM and corresponding FAOSTAT codes
[~, ~, FoodcodesSPAMtoFAOSTAT] = xlsread([ancillaryfold 'Foodcodes_SPAMtoFAOSTAT.xls'],'Sheet1');
FoodcodesSPAMtoFAOSTAT = FoodcodesSPAMtoFAOSTAT(:,2:end);
FoodcodesSPAMtoFAOSTAT(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),FoodcodesSPAMtoFAOSTAT)) = {''};


%% Calories for each of 41 crop categories (count 2 millet categories as 1)
[~, ~, raw] = xlsread([ancillaryfold 'Nutrition_SPAMcrops.xls'],'Sheet1');
raw = raw(2:end,2:end);
NutritionperCrop = reshape([raw{:}],size(raw));
NutritionperCrop(NutritionperCrop==-9999)=NaN;
clearvars raw;

%% Keep information on selected crops only
crop_filetitle(removelist) = [];
NutritionperCrop(removelist,:)=[];

% Adjust units: "kilocalorie" (aka "Calorie" or "food calorie") per 100g to
% kcal/g
Crop_kcalpergram = NutritionperCrop(:,1)/100;
Crop_mgproteinpergram = NutritionperCrop(:,2)/100;

%% DATASET: Conversion table for FAOSTAT-based full list
opts = spreadsheetImportOptions("NumVariables", 11);
opts.Sheet = "Sheet1";
opts.DataRange = "A2:K274";
opts.VariableNames = ["Country", "PumaIndex", "FAOSTAT", "Siebert", "ISOCode", "FAO_perhaps", "GAUL", "GDAM", "WorldBank", "ISO3Alpha", "USDAPSD", "ISO3custom"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "string", "string", "string", "string"];
opts = setvaropts(opts, ["Country", "WorldBank", "ISO3Alpha", "USDAPSD", "ISO3custom"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Country", "WorldBank", "ISO3Alpha", "USDAPSD", "ISO3custom"], "EmptyFieldRule", "auto");
tbl = readtable([ancillaryfold 'CountryCode_Convert.xls'], opts, "UseExcel", false);

Country = tbl.Country;
PumaIndex = tbl.PumaIndex;
FAOSTAT = tbl.FAOSTAT;
Siebert = tbl.Siebert;
ISOCode = tbl.ISOCode;
FAO_perhaps = tbl.FAO_perhaps;
GAUL = tbl.GAUL;
GDAM = tbl.GDAM;
WorldBank = tbl.WorldBank;
ISO3Alpha = tbl.ISO3Alpha;
USDAPSD = tbl.USDAPSD;
ISO3custom = tbl.ISO3custom;
clear opts tbl


%% Open needed data files
% Read gridded GDAM country and state code data
filename=[ancillaryfold 'gdam_v2_country.txt'];
country_matrix = rot90(arcgridread(filename),-1);

filename=[ancillaryfold 'gdam_v2_state.asc'];
state_matrix = rot90(arcgridread(filename),-1);

%% Load production
% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 57);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["iso3", "prod_level", "alloc_key", "cell5m", "x", "y", "rec_type", "tech_type", "unit", "whea_a", "rice_a", "maiz_a", "barl_a", "pmil_a", "smil_a", "sorg_a", "ocer_a", "pota_a", "swpo_a", "yams_a", "cass_a", "orts_a", "bean_a", "chic_a", "cowp_a", "pige_a", "lent_a", "opul_a", "soyb_a", "grou_a", "cnut_a", "oilp_a", "sunf_a", "rape_a", "sesa_a", "ooil_a", "sugc_a", "sugb_a", "cott_a", "ofib_a", "acof_a", "rcof_a", "coco_a", "teas_a", "toba_a", "bana_a", "plnt_a", "trof_a", "temf_a", "vege_a", "rest_a", "crea_date", "year_data", "source", "name_cntr", "name_adm1", "name_adm2"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "categorical", "categorical", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "datetime", "categorical", "double", "categorical", "categorical", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["iso3", "rec_type", "tech_type", "unit", "year_data", "name_cntr", "name_adm1", "name_adm2"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "crea_date", "InputFormat", "MM/dd/yy hh:mm:ss aa");
opts = setvaropts(opts, ["prod_level", "source"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["prod_level", "source"], "ThousandsSeparator", ",");

% Import the data
spam2010V2r0globalPTA = readtable([inputfold 'spam2010v2r0_' field{1} '.csv'], opts);
spam2010V2r0globalPTA(1,1) = {'CHN'};

% Clear temporary variables
clear opts


%% Load harvest area
% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 57);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["iso3", "prod_level", "alloc_key", "cell5m", "x", "y", "rec_type", "tech_type", "unit", "whea_a", "rice_a", "maiz_a", "barl_a", "pmil_a", "smil_a", "sorg_a", "ocer_a", "pota_a", "swpo_a", "yams_a", "cass_a", "orts_a", "bean_a", "chic_a", "cowp_a", "pige_a", "lent_a", "opul_a", "soyb_a", "grou_a", "cnut_a", "oilp_a", "sunf_a", "rape_a", "sesa_a", "ooil_a", "sugc_a", "sugb_a", "cott_a", "ofib_a", "acof_a", "rcof_a", "coco_a", "teas_a", "toba_a", "bana_a", "plnt_a", "trof_a", "temf_a", "vege_a", "rest_a", "crea_date", "year_data", "source", "name_cntr", "name_adm1", "name_adm2"];
opts.VariableTypes = ["categorical", "double", "double", "double", "double", "double", "categorical", "categorical", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "datetime", "categorical", "double", "categorical", "categorical", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["iso3", "rec_type", "tech_type", "unit", "year_data", "name_cntr", "name_adm1", "name_adm2"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "crea_date", "InputFormat", "MM/dd/yy hh:mm:ss aa");
opts = setvaropts(opts, ["prod_level", "source"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["prod_level", "source"], "ThousandsSeparator", ",");

% Import the data
spam2010V2r0globalHTA = readtable([inputfold 'spam2010v2r0_' field{2} '.csv'], opts);

% Clear temporary variables
clear opts

%% Sum harvest area and production for cells (and convert units)
% Subset harvested areas
CropSubset_HA = spam2010V2r0globalHTA(:,[1,i_crop+9,56,57]);
%%convert harvested areas from hectares to km^2
%%CropSubset_HA{:,2:size(CropSubset_HA,2)-2} = CropSubset_HA{:,2:size(CropSubset_HA,2)-2}*0.01;

% COde to pull out province names
% [C,ia] = unique(spam2010V2r0globalHTA(:,56),'rows');
% countryUnique = spam2010V2r0globalHTA(ia,[1,55,56,57]);

% Subset and convert production from metric tons to grams
CropSubset_Prod = spam2010V2r0globalPTA(:,[1,i_crop+9,56,57]);
CropSubset_Prod{:,2:size(CropSubset_HA,2)-2} = CropSubset_Prod{:,2:size(CropSubset_HA,2)-2}*gramsperMetricTon;
CropSubset_Prod{:,2:size(CropSubset_HA,2)-2} = CropSubset_Prod{:,2:size(CropSubset_HA,2)-2}.*Crop_kcalpergram';

%% Initialize matrices for combined event lists
TotalEvent  = zeros(num_disturb,2);
disturbance_all = nan(longestlist,num_disturb);
stateflag_all = nan(longestlist,num_disturb);
%subdivname_all = cell(longestlist,num_disturb);
countrystatecode_all = nan(longestlist,num_disturb);


%% Great Famine of 1315 to 1317
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{1});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
GreatFamine_countryname = name(2:end,1);
GreatFamine_countrycode = reshape([raw1{:}],size(raw1));
GreatFamine_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

% State level losses
[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{1});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
GreatFamine_statename = name(2:end,1);
GreatFamine_countrystatecode = reshape([raw1{:}],size(raw1));
%GreatFamine_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(GreatFamine_stateflag),1)=GreatFamine_stateflag; 
%subdivname_all(1:length(GreatFamine_subdivname),1)=GreatFamine_subdivname; 
countrystatecode_all(1:length(GreatFamine_countrystatecode),1)=GreatFamine_countrystatecode; 


%% Laki eruption of 1783
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{2});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Laki1783_countryname = name(2:end,1);
Laki1783_countrycode = reshape([raw1{:}],size(raw1));
Laki1783_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{2});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Laki1783_statename = name(2:end,1);
Laki1783_countrystatecode = reshape([raw1{:}],size(raw1));
%Laki1783_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Laki1783_stateflag),2)=Laki1783_stateflag;
%subdivname_all(1:length(Laki1783_subdivname),2)=Laki1783_subdivname;
countrystatecode_all(1:length(Laki1783_countrystatecode),2)=Laki1783_countrystatecode;


%% "Year Without a Summer" of 1816 (Tambora eruption of 1815)
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{3});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
NoSummer1816_countryname = name(2:end,1);
NoSummer1816_countrycode = reshape([raw1{:}],size(raw1));
NoSummer1816_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{3});
raw1 = raw(2:end,2); raw2 = raw(2:end,3); 
NoSummer1816_statename = name(2:end,1);
NoSummer1816_countrystatecode = reshape([raw1{:}],size(raw1));
NoSummer1816_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(NoSummer1816_stateflag),3)=NoSummer1816_stateflag;  
subdivname_all(1:length(NoSummer1816_subdivname),3)=NoSummer1816_subdivname; 
countrystatecode_all(1:length(NoSummer1816_countrystatecode),3)=NoSummer1816_countrystatecode; 


%% Great Drought of 1876 to 1878
% Countries
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{4});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Drought18761878_countryname = name(2:end,1);
Drought18761878_countrycode = reshape([raw1{:}],size(raw1));
Drought18761878_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

% States
[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{4});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Drought18761878_statename = name(2:end,1);
Drought18761878_countrystatecode = reshape([raw1{:}],size(raw1));
Drought18761878_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Drought18761878_stateflag),4)=Drought18761878_stateflag; 
subdivname_all(1:length(Drought18761878_subdivname),4)=Drought18761878_subdivname; 
countrystatecode_all(1:length(Drought18761878_countrystatecode),4)=Drought18761878_countrystatecode; 


%% Soviet Famine of 1921 to 1922
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{5});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Soviet1921to1922_countryname = name(2:end,1);
Soviet1921to1922_countrycode = reshape([raw1{:}],size(raw1));
Soviet1921to1922_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{5});
raw1 = raw(2:end,2); raw2 = raw(2:end,3); 
Soviet1921to1922_statename = name(2:end,1);
Soviet1921to1922_countrystatecode = reshape([raw1{:}],size(raw1));
Soviet1921to1922_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Soviet1921to1922_stateflag),5)=Soviet1921to1922_stateflag;
subdivname_all(1:length(Soviet1921to1922_subdivname),5)=Soviet1921to1922_subdivname;
countrystatecode_all(1:length(Soviet1921to1922_countrystatecode),5)=Soviet1921to1922_countrystatecode;


%% Chinese Famine of 1959 to 1961
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{6});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Chinese1960_countryname = name(2:end,1);
Chinese1960_countrycode = reshape([raw1{:}],size(raw1));
Chinese1960_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{6});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Chinese1960_statename = name(2:end,1);
Chinese1960_countrystatecode = reshape([raw1{:}],size(raw1));
Chinese1960_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Chinese1960_stateflag),6)=Chinese1960_stateflag;
subdivname_all(1:length(Chinese1960_subdivname),6)=Chinese1960_subdivname;
countrystatecode_all(1:length(Chinese1960_countrystatecode),6)=Chinese1960_countrystatecode;


%% Dust Bowl Drought
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{7});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
DustBowl_countryname = name(2:end,1);
DustBowl_countrycode = reshape([raw1{:}],size(raw1));
DustBowl_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{7});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
DustBowl_statename = name(2:end,1);
DustBowl_countrystatecode = reshape([raw1{:}],size(raw1));
DustBowl_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(DustBowl_stateflag),7)=DustBowl_stateflag;
subdivname_all(1:length(DustBowl_subdivname),7)=DustBowl_subdivname;
countrystatecode_all(1:length(DustBowl_countrystatecode),7)=DustBowl_countrystatecode;


%% 2010 Sahel famine
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{8});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
SahelDrought2010_countryname = name(2:end,1);
SahelDrought2010_countrycode = reshape([raw1{:}],size(raw1));
SahelDrought2010_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{8});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
SahelDrought2010_statename = name(2:end,1);
SahelDrought2010_countrystatecode = reshape([raw1{:}],size(raw1));
SahelDrought2010_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(SahelDrought2010_stateflag),8)=SahelDrought2010_stateflag;
subdivname_all(1:length(SahelDrought2010_subdivname),8)=SahelDrought2010_subdivname;
countrystatecode_all(1:length(SahelDrought2010_countrystatecode),8)=SahelDrought2010_countrystatecode;


%% Australian Drought of 2000s
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{9});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
MillenniumDrought_countryname = name(2:end,1);
MillenniumDrought_countrycode = reshape([raw1{:}],size(raw1));
MillenniumDrought_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{9});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
MillenniumDrought_statename = name(2:end,1);
MillenniumDrought_countrystatecode = reshape([raw1{:}],size(raw1));
MillenniumDrought_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(MillenniumDrought_stateflag),9)=MillenniumDrought_stateflag;
subdivname_all(1:length(MillenniumDrought_subdivname),9)=MillenniumDrought_subdivname;
countrystatecode_all(1:length(MillenniumDrought_countrystatecode),9)=MillenniumDrought_countrystatecode;


%% North Korean Famine of 1990s
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{10});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
NorthKorea1990s_countryname = name(2:end,1);
NorthKorea1990s_countrycode = reshape([raw1{:}],size(raw1));
NorthKorea1990s_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{10});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
NorthKorea1990s_statename = name(2:end,1);
NorthKorea1990s_countrystatecode = reshape([raw1{:}],size(raw1));
NorthKorea1990s_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(NorthKorea1990s_stateflag),10)=NorthKorea1990s_stateflag;
subdivname_all(1:length(NorthKorea1990s_subdivname),10)=NorthKorea1990s_subdivname;
countrystatecode_all(1:length(NorthKorea1990s_countrystatecode),10)=NorthKorea1990s_countrystatecode;


%% Solomon
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{11});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Solomon_countryname = name(2:end,1);
Solomon_countrycode = reshape([raw1{:}],size(raw1));
Solomon_stateflag =   reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{11});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Solomon_statename = name(2:end,1);
Solomon_countrystatecode = reshape([raw1{:}],size(raw1));
Solomon_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Solomon_stateflag),11)=Solomon_stateflag;
subdivname_all(1:length(Solomon_subdivname),11)=Solomon_subdivname;
countrystatecode_all(1:length(Solomon_countrystatecode),11)=Solomon_countrystatecode;


%% Vanuatu
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{12});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Vanuatu_countryname = name(2:end,1);
Vanuatu_countrycode = reshape([raw1{:}],size(raw1));
Vanuatu_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{12});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Vanuatu_statename = name(2:end,1);
Vanuatu_countrystatecode = reshape([raw1{:}],size(raw1));
Vanuatu_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Vanuatu_stateflag),12)=Vanuatu_stateflag;
subdivname_all(1:length(Vanuatu_subdivname),12)=Vanuatu_subdivname;
countrystatecode_all(1:length(Vanuatu_countrystatecode),12)=Vanuatu_countrystatecode;


%% East Timor
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{13});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
EastTimor_countryname = name(2:end,1);
EastTimor_countrycode = reshape([raw1{:}],size(raw1));
EastTimor_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{13});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
EastTimor_statename = name(2:end,1);
EastTimor_countrystatecode = reshape([raw1{:}],size(raw1));
EastTimor_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(EastTimor_stateflag),13)=EastTimor_stateflag;
subdivname_all(1:length(EastTimor_subdivname),13)=EastTimor_subdivname;
countrystatecode_all(1:length(EastTimor_countrystatecode),13)=EastTimor_countrystatecode;


%% Haiti
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{14});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Haiti_countryname = name(2:end,1);
Haiti_countrycode = reshape([raw1{:}],size(raw1));
Haiti_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{14});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Haiti_statename = name(2:end,1);
Haiti_countrystatecode = reshape([raw1{:}],size(raw1));
Haiti_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Haiti_stateflag),14)=Haiti_stateflag;
subdivname_all(1:length(Haiti_subdivname),14)=Haiti_subdivname;
countrystatecode_all(1:length(Haiti_countrystatecode),14)=Haiti_countrystatecode;


%% Sierra Leone
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{15});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
SierraLeone_countryname = name(2:end,1);
SierraLeone_countrycode = reshape([raw1{:}],size(raw1));
SierraLeone_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{15});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
SierraLeone_statename = name(2:end,1);
SierraLeone_countrystatecode = reshape([raw1{:}],size(raw1));
SierraLeone_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(SierraLeone_stateflag),15)=SierraLeone_stateflag;
subdivname_all(1:length(SierraLeone_subdivname),15)=SierraLeone_subdivname;
countrystatecode_all(1:length(SierraLeone_countrystatecode),15)=SierraLeone_countrystatecode;


%% Liberia
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{16});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Liberia_countryname = name(2:end,1);
Liberia_countrycode = reshape([raw1{:}],size(raw1));
Liberia_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{16});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Liberia_statename = name(2:end,1);
Liberia_countrystatecode = reshape([raw1{:}],size(raw1));
Liberia_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Liberia_stateflag),16)=Liberia_stateflag;
subdivname_all(1:length(Liberia_subdivname),16)=Liberia_subdivname;
countrystatecode_all(1:length(Liberia_countrystatecode),16)=Liberia_countrystatecode;


%% Yemen
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{17});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Yemen_countryname = name(2:end,1);
Yemen_countrycode = reshape([raw1{:}],size(raw1));
Yemen_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{17});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Yemen_statename = name(2:end,1);
Yemen_countrystatecode = reshape([raw1{:}],size(raw1));
Yemen_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Yemen_stateflag),17)=Yemen_stateflag;
subdivname_all(1:length(Yemen_subdivname),17)=Yemen_subdivname;
countrystatecode_all(1:length(Yemen_countrystatecode),17)=Yemen_countrystatecode;


%% Ethiopia
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{18});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Ethiopia_countryname = name(2:end,1);
Ethiopia_countrycode = reshape([raw1{:}],size(raw1));
Ethiopia_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{18});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Ethiopia_statename = name(2:end,1);
Ethiopia_countrystatecode = reshape([raw1{:}],size(raw1));
Ethiopia_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Ethiopia_stateflag),18)=Ethiopia_stateflag;
subdivname_all(1:length(Ethiopia_subdivname),18)=Ethiopia_subdivname;
countrystatecode_all(1:length(Ethiopia_countrystatecode),18)=Ethiopia_countrystatecode;


%% Laos
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{19});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Laos_countryname = name(2:end,1);
Laos_countrycode = reshape([raw1{:}],size(raw1));
Laos_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{19});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Laos_statename = name(2:end,1);
Laos_countrystatecode = reshape([raw1{:}],size(raw1));
Laos_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Laos_stateflag),19)=Laos_stateflag;
subdivname_all(1:length(Laos_subdivname),19)=Laos_subdivname;
countrystatecode_all(1:length(Laos_countrystatecode),19)=Laos_countrystatecode;


%% Bangladesh
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{20});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Bangladesh_countryname = name(2:end,1);
Bangladesh_countrycode = reshape([raw1{:}],size(raw1));
Bangladesh_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{20});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Bangladesh_statename = name(2:end,1);
Bangladesh_countrystatecode = reshape([raw1{:}],size(raw1));
Bangladesh_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Bangladesh_stateflag),20)=Bangladesh_stateflag;
subdivname_all(1:length(Bangladesh_subdivname),20)=Bangladesh_subdivname;
countrystatecode_all(1:length(Bangladesh_countrystatecode),20)=Bangladesh_countrystatecode;


%% Syria
[~, name, raw] = xlsread([ancillaryfold 'DisruptionCountry.xls'],EventList{21});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Syria_countryname = name(2:end,1);
Syria_countrycode = reshape([raw1{:}],size(raw1));
Syria_stateflag = reshape([raw2{:}],size(raw2));
clearvars raw raw1 raw2 name;

[~, name, raw] = xlsread([ancillaryfold 'DisruptionStateProvince.xls'],EventList{21});
raw1 = raw(2:end,2); raw2 = raw(2:end,3);
Syria_statename = name(2:end,1);
Syria_countrystatecode = reshape([raw1{:}],size(raw1));
Syria_subdivname = name(2:end,3);
clearvars raw raw1 raw2 name;

stateflag_all(1:length(Syria_stateflag),21)=Syria_stateflag;
subdivname_all(1:length(Syria_subdivname),21)=Syria_subdivname;
countrystatecode_all(1:length(Syria_countrystatecode),21)=Syria_countrystatecode;


% Combine disruptions
disturbance_all(1:length(GreatFamine_countrycode),1)=GreatFamine_countrycode; 
disturbance_all(1:length(Laki1783_countrycode),2)=Laki1783_countrycode;
disturbance_all(1:length(NoSummer1816_countrycode),3)=NoSummer1816_countrycode;  
disturbance_all(1:length(Drought18761878_countrycode),4)=Drought18761878_countrycode; 
disturbance_all(1:length(Soviet1921to1922_countrycode),5)=Soviet1921to1922_countrycode;
disturbance_all(1:length(Chinese1960_countrycode),6)=Chinese1960_countrycode;
disturbance_all(1:length(DustBowl_countrycode),7)=DustBowl_countrycode;
disturbance_all(1:length(SahelDrought2010_countrycode),8)=SahelDrought2010_countrycode;
disturbance_all(1:length(MillenniumDrought_countrycode),9)=MillenniumDrought_countrycode;
disturbance_all(1:length(NorthKorea1990s_countrycode),10)=NorthKorea1990s_countrycode;
disturbance_all(1:length(Solomon_countrycode),11)=Solomon_countrycode;
disturbance_all(1:length(Vanuatu_countrycode),12)=Vanuatu_countrycode;
disturbance_all(1:length(EastTimor_countrycode),13)=EastTimor_countrycode;
disturbance_all(1:length(Haiti_countrycode),14)=Haiti_countrycode;
disturbance_all(1:length(SierraLeone_countrycode),15)=SierraLeone_countrycode;
disturbance_all(1:length(Liberia_countrycode),16)=Liberia_countrycode;
disturbance_all(1:length(Yemen_countrycode),17)=Yemen_countrycode;
disturbance_all(1:length(Ethiopia_countrycode),18)=Ethiopia_countrycode;
disturbance_all(1:length(Laos_countrycode),19)=Laos_countrycode;
disturbance_all(1:length(Bangladesh_countrycode),20)=Bangladesh_countrycode;
disturbance_all(1:length(Syria_countrycode),21)=Syria_countrycode;

stateflag_all = nan(longestlist,num_disturb);
stateflag_all(1:length(GreatFamine_stateflag),1)=GreatFamine_stateflag; 
stateflag_all(1:length(Laki1783_stateflag),2)=Laki1783_stateflag;
stateflag_all(1:length(NoSummer1816_stateflag),3)=NoSummer1816_stateflag;  
stateflag_all(1:length(Drought18761878_stateflag),4)=Drought18761878_stateflag; 
stateflag_all(1:length(Soviet1921to1922_stateflag),5)=Soviet1921to1922_stateflag;
stateflag_all(1:length(Chinese1960_stateflag),6)=Chinese1960_stateflag;
stateflag_all(1:length(DustBowl_stateflag),7)=DustBowl_stateflag;
stateflag_all(1:length(SahelDrought2010_stateflag),8)=SahelDrought2010_stateflag;
stateflag_all(1:length(MillenniumDrought_stateflag),9)=MillenniumDrought_stateflag;
stateflag_all(1:length(NorthKorea1990s_stateflag),10)=NorthKorea1990s_stateflag;
stateflag_all(1:length(Solomon_stateflag),11)=Solomon_stateflag;
stateflag_all(1:length(Vanuatu_stateflag),12)=Vanuatu_stateflag;
stateflag_all(1:length(EastTimor_stateflag),13)=EastTimor_stateflag;
stateflag_all(1:length(Haiti_stateflag),14)=Haiti_stateflag;
stateflag_all(1:length(SierraLeone_stateflag),15)=SierraLeone_stateflag;
stateflag_all(1:length(Liberia_stateflag),16)=Liberia_stateflag;
stateflag_all(1:length(Yemen_stateflag),17)=Yemen_stateflag;
stateflag_all(1:length(Ethiopia_stateflag),18)=Ethiopia_stateflag;
stateflag_all(1:length(Laos_stateflag),19)=Laos_stateflag;
stateflag_all(1:length(Bangladesh_stateflag),20)=Bangladesh_stateflag;
stateflag_all(1:length(Syria_stateflag),21)=Syria_stateflag;


%% Initialize lat-lon gridded variables
% Compute losses by country for each disturbance event
EventProduction_loss  = zeros(longestlist,num_disturb);
EventHarvestArea_loss = zeros(longestlist,num_disturb);

% Loop over events and countries
for i_disturb = 1:num_disturb

    % Get country list and flag for state/province data for individual events
    event_countries = squeeze(disturbance_all(:,i_disturb));
    event_stateflag = squeeze(stateflag_all(:,i_disturb));

    event_countries(isnan(event_countries)) = [];
    event_stateflag(isnan(event_stateflag)) = [];

    for i_country = 1:length(event_countries)
        
        % Vectors for countries with state/province level data
        event_countrystatecode = countrystatecode_all(:,i_disturb);
        %%event_subdivname = subdivname_all(:,i_disturb);

        % Check to see if state/province level data needed; otherwise,
        % use country level data
        if event_countrystatecode(i_country,1) == 1 % STATE-LEVEL DATA

            % Find states corresponding to the current country
            %%event_subdivname(event_countrystatecode~=event_countries(i_country,1)) = [];
            event_countrystatecode(event_countrystatecode~=event_countries(i_country,1)) = [];

            for i_state  = 1:size(event_subdivname,1)
                % Extract production and harvest area for state/province
                Pvalues  =CropSubset_Prod([ismember(CropSubset_Prod.name_adm1,event_countrystatecode(i_state,1))],2);
                HAvalues =  CropSubset_HA([ismember(CropSubset_HA.name_adm1,event_countrystatecode(i_state,1))],2);
                
                % Sum production (metric tons) and harvest area (hectares) 
                % for each country and crop affected by the event
                EventProduction_loss(i_country,i_disturb) = ...
                    EventProduction_loss(i_country,i_disturb)+sum(Pvalues{:,1}); 
                EventHarvestArea_loss(i_country,i_disturb) = ...
                    EventHarvestArea_loss(i_country,i_disturb)+sum(HAvalues{:,1}); 
            end

        else % COUNTRY-LEVEL DATA
            % ISO3 code for given country
            index_event = find(GDAM==event_countries(i_country,1));

            TF = ( spam2010V2r0globalPTA.iso3 == ISO3custom(index_event) );
            eventPTA = spam2010V2r0globalPTA(TF,[1 10:51]);
            eventPTA(:,removelist+1)=[];

            TF = ( spam2010V2r0globalHTA.iso3 == ISO3custom(index_event) );
            eventHTA = spam2010V2r0globalHTA(TF,[1 10:51]);
            eventHTA(:,removelist+1)=[];

            % Sum production (after converting to kcal) for each
            % country and crop affected by the event
            % Subset and convert production from metric tons to grams
            EventProduction_loss(i_country,i_disturb) = sum(sum(eventPTA{:,2:end}).*Crop_kcalpergram'*gramsperMetricTon);

            % Subset harvested areas 
            EventHarvestArea_loss(i_country,i_disturb)= sum(sum(eventHTA{:,2:end})');
               %%% convert harvested areas from hectares to km^2
               %EventHarvest_loss(i_country,i_disturb)= EventHarvest_loss(i_country,i_disturb)*0.01;

        end

    end

end
   
%% Separate losses by event for saving
% Crop production losses by country by crop (metric tons)
GreatFamine_ProductionLoss = squeeze(EventProduction_loss(:,1)); 
Laki1783_ProductionLoss = squeeze(EventProduction_loss(:,2));
NoSummer1816_ProductionLoss = squeeze(EventProduction_loss(:,3)); 
Drought18761878_ProductionLoss = squeeze(EventProduction_loss(:,4)); 
Soviet1921to1922_ProductionLoss = squeeze(EventProduction_loss(:,5));
Chinese1960_ProductionLoss = squeeze(EventProduction_loss(:,6));
DustBowl_ProductionLoss = squeeze(EventProduction_loss(:,7));
SahelDrought2010_ProductionLoss = squeeze(EventProduction_loss(:,8));
MillenniumDrought_ProductionLoss = squeeze(EventProduction_loss(:,9));
NorthKorea1990s_ProductionLoss = squeeze(EventProduction_loss(:,10));
Solomon_ProductionLoss = squeeze(EventProduction_loss(:,11));
Vanuatu_ProductionLoss = squeeze(EventProduction_loss(:,12));
EastTimor_ProductionLoss = squeeze(EventProduction_loss(:,13));
Haiti_ProductionLoss = squeeze(EventProduction_loss(:,14));
SierraLeone_ProductionLoss = squeeze(EventProduction_loss(:,15));
Liberia_ProductionLoss = squeeze(EventProduction_loss(:,16));
Yemen_ProductionLoss = squeeze(EventProduction_loss(:,17));
Ethiopia_ProductionLoss = squeeze(EventProduction_loss(:,18));
Laos_ProductionLoss = squeeze(EventProduction_loss(:,19));
Bangladesh_ProductionLoss = squeeze(EventProduction_loss(:,20));
Syria_ProductionLoss = squeeze(EventProduction_loss(:,21));


% Harvest area losses by country by crop (hectares)
GreatFamine_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,1)); 
Laki1783_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,2));
NoSummer1816_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,3));  
Drought18761878_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,4)); 
Soviet1921to1922_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,5));
Chinese1960_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,6));
DustBowl_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,7));
SahelDrought2010_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,8));
MillenniumDrought_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,9));
NorthKorea1990s_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,10));
Solomon_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,11));
Vanuatu_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,12));
EastTimor_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,13));
Haiti_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,14));
SierraLeone_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,15));
Liberia_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,16));
Yemen_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,17));
Ethiopia_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,18));
Laos_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,19));
Bangladesh_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,20));
Syria_HarvestAreaLoss = squeeze(EventHarvestArea_loss(:,21));


%% Transfer total losses to matrix
TotalEvent(1,1) = nansum(GreatFamine_HarvestAreaLoss(:));
TotalEvent(2,1) = nansum(Laki1783_HarvestAreaLoss(:));
TotalEvent(3,1) = nansum(NoSummer1816_HarvestAreaLoss(:));
TotalEvent(4,1) = nansum(Drought18761878_HarvestAreaLoss(:));
TotalEvent(5,1) = nansum(Soviet1921to1922_HarvestAreaLoss(:));
TotalEvent(6,1) = nansum(Chinese1960_HarvestAreaLoss(:));
TotalEvent(7,1) = nansum(DustBowl_HarvestAreaLoss(:));
TotalEvent(8,1) = nansum(SahelDrought2010_HarvestAreaLoss(:));
TotalEvent(9,1) = nansum(MillenniumDrought_HarvestAreaLoss(:));
TotalEvent(10,1) = nansum(NorthKorea1990s_HarvestAreaLoss(:));
TotalEvent(11,1) = nansum(Solomon_HarvestAreaLoss(:));
TotalEvent(12,1) = nansum(Vanuatu_HarvestAreaLoss(:));
TotalEvent(13,1) = nansum(EastTimor_HarvestAreaLoss(:));
TotalEvent(14,1) = nansum(Haiti_HarvestAreaLoss(:));
TotalEvent(15,1) = nansum(SierraLeone_HarvestAreaLoss(:));
TotalEvent(16,1) = nansum(Liberia_HarvestAreaLoss(:));
TotalEvent(17,1) = nansum(Yemen_HarvestAreaLoss(:));
TotalEvent(18,1) = nansum(Ethiopia_HarvestAreaLoss(:));
TotalEvent(19,1) = nansum(Laos_HarvestAreaLoss(:));
TotalEvent(20,1) = nansum(Bangladesh_HarvestAreaLoss(:));
TotalEvent(21,1) = nansum(Syria_HarvestAreaLoss(:));

TotalEvent(1,2) = nansum(GreatFamine_ProductionLoss(:));
TotalEvent(2,2) = nansum(Laki1783_ProductionLoss(:));
TotalEvent(3,2) = nansum(NoSummer1816_ProductionLoss(:));
TotalEvent(4,2) = nansum(Drought18761878_ProductionLoss(:));
TotalEvent(5,2) = nansum(Soviet1921to1922_ProductionLoss(:));
TotalEvent(6,2) = nansum(Chinese1960_ProductionLoss(:));
TotalEvent(7,2) = nansum(DustBowl_ProductionLoss(:));
TotalEvent(8,2) = nansum(SahelDrought2010_ProductionLoss(:));
TotalEvent(9,2) = nansum(MillenniumDrought_ProductionLoss(:));
TotalEvent(10,2) = nansum(NorthKorea1990s_ProductionLoss(:));
TotalEvent(11,2) = nansum(Solomon_ProductionLoss(:));
TotalEvent(12,2) = nansum(Vanuatu_ProductionLoss(:));
TotalEvent(13,2) = nansum(EastTimor_ProductionLoss(:));
TotalEvent(14,2) = nansum(Haiti_ProductionLoss(:));
TotalEvent(15,2) = nansum(SierraLeone_ProductionLoss(:));
TotalEvent(16,2) = nansum(Liberia_ProductionLoss(:));
TotalEvent(17,2) = nansum(Yemen_ProductionLoss(:));
TotalEvent(18,2) = nansum(Ethiopia_ProductionLoss(:));
TotalEvent(19,2) = nansum(Laos_ProductionLoss(:));
TotalEvent(20,2) = nansum(Bangladesh_ProductionLoss(:));
TotalEvent(21,2) = nansum(Syria_ProductionLoss(:));
%  Save 
save([outputfold 'TotalEventLosses_' num2str(year_start) '_' num2str(year_end) '.mat'],'TotalEvent');
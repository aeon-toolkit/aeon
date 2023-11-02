"""Datasets in the Monash tser data archives."""


tser_all = {
    "AppliancesEnergy": 3902637,
    "HouseholdPowerConsumption1": 3902704,
    "HouseholdPowerConsumption2": 3902706,
    "BenzeneConcentration": 3902673,
    "BeijingPM25Quality": 3902671,
    "BeijingPM10Quality": 3902667,
    "LiveFuelMoistureContent": 4632439,
    "FloodModeling1": 3902694,
    "FloodModeling2": 3902696,
    "FloodModeling3": 3902698,
    "AustraliaRainfall": 3902654,
    #    "PPGDalia": 3902728, #this dataset has unequal length within a case, and this
    #    format is not supported by aeon
    "IEEEPPG": 3902710,
    "BIDMC32RR": 4001463,
    "BIDMC32HR": 4001456,
    "BIDMC32SpO2": 4001464,
    "NewsHeadlineSentiment": 3902718,
    "NewsTitleSentiment": 3902726,
    "Covid3Month": 3902690,
}
tser_all_tsc = {
    "AppliancesEnergy" "HouseholdPowerConsumption1-no-missing",
    "HouseholdPowerConsumption2-no-missing",
    "BenzeneConcentration-no-missing",
    "BeijingPM25Quality-no-missing",
    "BeijingPM10Quality-no-missing",
    "LiveFuelMoistureContent",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "AustraliaRainfall",
    "PPGDalia-equal-length",
    "IEEEPPG",
    #   "BIDMCRR": 4001463, Not found
    #    "BIDMCHR": 4001456, Not found
    #    "BIDMCSpO2": 4001464, Not found
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "Covid3Month",
}

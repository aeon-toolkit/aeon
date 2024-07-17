"""Dictionaries of expected outputs of regressor predict runs."""

import numpy as np

# predict results on Covid3Month data
covid_3month_preds = dict()

# predict results on CardanoSentiment data
cardano_sentiment_preds = dict()


covid_3month_preds["FreshPRINCERegressor"] = np.array(
    [
        0.0545,
        0.0767,
        0.0347,
        0.0505,
        0.0683,
        0.0434,
        0.0674,
        0.1111,
        0.0946,
        0.0366,
    ]
)

cardano_sentiment_preds["FreshPRINCERegressor"] = np.array(
    [
        0.3484,
        0.1438,
        0.3705,
        0.0842,
        0.3892,
        0.3705,
        0.1342,
        0.3476,
        0.0959,
        0.409,
    ]
)

covid_3month_preds["Catch22Regressor"] = np.array(
    [
        0.0302,
        0.0354,
        0.0352,
        0.0345,
        0.0259,
        0.0484,
        0.0369,
        0.0827,
        0.0737,
        0.0526,
    ]
)

cardano_sentiment_preds["Catch22Regressor"] = np.array(
    [
        0.2174,
        0.1394,
        0.3623,
        0.1496,
        0.3502,
        0.2719,
        0.1378,
        0.076,
        0.0587,
        0.3773,
    ]
)

covid_3month_preds["RandomForestRegressor"] = np.array(
    [
        0.0319,
        0.0505,
        0.0082,
        0.0291,
        0.028,
        0.0266,
        0.0239,
        0.0946,
        0.0946,
        0.0251,
    ]
)

cardano_sentiment_preds["SummaryRegressor"] = np.array(
    [
        0.3172,
        0.5002,
        0.3072,
        0.4486,
        0.1765,
        0.4664,
        0.0828,
        0.251,
        0.0402,
        0.2641,
    ]
)

covid_3month_preds["TSFreshRegressor"] = np.array(
    [
        0.0106,
        0.0587,
        0.0082,
        0.0291,
        0.0453,
        0.0185,
        0.036,
        0.0946,
        0.0946,
        0.0251,
    ]
)

cardano_sentiment_preds["TSFreshRegressor"] = np.array(
    [
        0.2997,
        0.2633,
        0.3408,
        0.1045,
        0.2517,
        0.2001,
        0.1546,
        0.1751,
        0.0936,
        0.2973,
    ]
)

covid_3month_preds["HydraRegressor"] = np.array(
    [
        -0.0073,
        0.0516,
        0.0378,
        0.0439,
        0.0247,
        0.0426,
        0.0272,
        0.054,
        0.0484,
        0.044,
    ]
)

cardano_sentiment_preds["HydraRegressor"] = np.array(
    [
        0.5925,
        0.2068,
        0.5268,
        0.2383,
        0.4586,
        0.1701,
        0.2336,
        0.1333,
        0.0025,
        0.4788,
    ]
)

covid_3month_preds["MultiRocketHydraRegressor"] = np.array(
    [
        -0.0751,
        0.0604,
        0.0315,
        0.0376,
        0.022,
        0.0337,
        0.0249,
        0.0835,
        0.1012,
        0.029,
    ]
)

covid_3month_preds["RocketRegressor"] = np.array(
    [
        0.0381,
        0.0379,
        0.0379,
        0.0368,
        0.04,
        0.0375,
        0.0387,
        0.0419,
        0.0371,
        0.0371,
    ]
)

cardano_sentiment_preds["RocketRegressor"] = np.array(
    [
        0.1841,
        0.1884,
        0.1882,
        0.1879,
        0.1862,
        0.1817,
        0.1858,
        0.1894,
        0.1845,
        0.1844,
    ]
)

covid_3month_preds["KNeighborsTimeSeriesRegressor"] = np.array(
    [
        0.0081,
        0.1111,
        0.0408,
        0.0212,
        0.0557,
        0.0408,
        0.0818,
        0.1111,
        0.1111,
        0.0,
    ]
)

cardano_sentiment_preds["KNeighborsTimeSeriesRegressor"] = np.array(
    [
        0.3503,
        -0.101,
        0.3847,
        0.0,
        0.3847,
        0.0,
        0.178,
        0.0,
        0.0,
        0.3847,
    ]
)

cardano_sentiment_preds["RISTRegressor"] = np.array(
    [
        0.3002,
        0.3174,
        0.718,
        0.089,
        0.4002,
        0.0825,
        0.5342,
        0.0,
        0.3503,
        0.448,
    ]
)

covid_3month_preds["CanonicalIntervalForestRegressor"] = np.array(
    [
        0.049,
        0.04,
        0.0299,
        0.0352,
        0.0423,
        0.0315,
        0.0519,
        0.0605,
        0.0647,
        0.037,
    ]
)

cardano_sentiment_preds["CanonicalIntervalForestRegressor"] = np.array(
    [
        0.276,
        0.1466,
        0.282,
        0.205,
        0.125,
        0.0111,
        0.3672,
        0.0677,
        0.1773,
        0.2586,
    ]
)

covid_3month_preds["DrCIFRegressor"] = np.array(
    [
        0.0302,
        0.0778,
        0.0272,
        0.03,
        0.0405,
        0.0388,
        0.0351,
        0.093,
        0.1041,
        0.0263,
    ]
)

cardano_sentiment_preds["DrCIFRegressor"] = np.array(
    [
        0.2361,
        0.2222,
        0.2046,
        0.1709,
        0.2462,
        0.2369,
        0.1916,
        0.1995,
        0.0407,
        0.1428,
    ]
)

covid_3month_preds["IntervalForestRegressor"] = np.array(
    [
        0.0358,
        0.0511,
        0.024,
        0.023,
        0.0475,
        0.0367,
        0.0308,
        0.0555,
        0.0606,
        0.026,
    ]
)

cardano_sentiment_preds["IntervalForestRegressor"] = np.array(
    [
        0.173,
        0.083,
        0.3016,
        0.1179,
        0.1651,
        0.1383,
        -0.0245,
        0.0961,
        0.049,
        0.1718,
    ]
)

covid_3month_preds["RandomIntervalRegressor"] = np.array(
    [
        0.0405,
        0.062,
        0.0069,
        0.032,
        0.0258,
        0.0478,
        0.032,
        0.062,
        0.062,
        0.0505,
    ]
)

cardano_sentiment_preds["RandomIntervalRegressor"] = np.array(
    [
        0.2546,
        0.1749,
        0.5911,
        0.1562,
        0.3726,
        0.228,
        0.211,
        0.2102,
        0.176,
        0.3056,
    ]
)

covid_3month_preds["RandomIntervalSpectralEnsembleRegressor"] = np.array(
    [
        0.0432,
        0.0516,
        0.0291,
        0.0423,
        0.0259,
        0.0247,
        0.0397,
        0.0536,
        0.0406,
        0.0468,
    ]
)

cardano_sentiment_preds["RandomIntervalSpectralEnsembleRegressor"] = np.array(
    [
        0.4,
        -0.0056,
        0.2023,
        0.0986,
        0.068,
        0.2063,
        0.1309,
        0.1811,
        0.1295,
        0.2968,
    ]
)

covid_3month_preds["TimeSeriesForestRegressor"] = np.array(
    [
        0.0319,
        0.0556,
        0.0249,
        0.0212,
        0.0385,
        0.0249,
        0.0105,
        0.0556,
        0.076,
        0.0143,
    ]
)

cardano_sentiment_preds["TimeSeriesForestRegressor"] = np.array(
    [
        0.2336,
        -0.0505,
        0.5514,
        0.089,
        0.3174,
        0.125,
        0.089,
        0.0,
        0.0412,
        0.1924,
    ]
)

covid_3month_preds["RDSTRegressor"] = np.array(
    [
        0.0368,
        0.0368,
        0.0368,
        0.0369,
        0.0369,
        0.0368,
        0.0368,
        0.0369,
        0.0369,
        0.0368,
    ]
)

cardano_sentiment_preds["RDSTRegressor"] = np.array(
    [
        0.1816,
        0.1706,
        0.1993,
        0.2251,
        0.1606,
        0.1682,
        0.1815,
        0.1829,
        0.1578,
        0.2048,
    ]
)
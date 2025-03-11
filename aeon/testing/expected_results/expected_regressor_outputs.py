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

covid_3month_preds["Catch22Regressor"] = np.array(
    [0.0310, 0.0555, 0.0193, 0.0359, 0.0261, 0.0361, 0.0387, 0.0835, 0.0827, 0.0414]
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

covid_3month_preds["KNeighborsTimeSeriesRegressor"] = np.array(
    [
        0.0081,
        0.1111,
        0.0,
        0.0,
        0.0557,
        0.0408,
        0.0212,
        0.1111,
        0.1111,
        0.0,
    ]
)

covid_3month_preds["CanonicalIntervalForestRegressor"] = np.array(
    [0.0412, 0.0420, 0.0292, 0.0202, 0.0432, 0.0192, 0.0155, 0.0543, 0.0412, 0.0399]
)

covid_3month_preds["DrCIFRegressor"] = np.array(
    [0.0376, 0.0317, 0.0274, 0.0143, 0.0332, 0.0397, 0.0386, 0.0721, 0.0632, 0.0211]
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

covid_3month_preds["RDSTRegressor"] = np.array(
    [
        0.0687,
        0.071,
        0.0227,
        0.0425,
        0.0834,
        0.0251,
        0.0064,
        0.1398,
        0.0667,
        0.031,
    ]
)

cardano_sentiment_preds["FreshPRINCERegressor"] = np.array(
    [0.36, 0.14, 0.36, 0.08, 0.45, 0.35, 0.15, 0.28, 0.09, 0.37]
)

cardano_sentiment_preds["Catch22Regressor"] = np.array(
    [
        0.2715,
        0.175,
        0.3152,
        0.1324,
        0.3341,
        0.1919,
        0.179,
        0.1295,
        0.1744,
        0.3658,
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

cardano_sentiment_preds["KNeighborsTimeSeriesRegressor"] = np.array(
    [
        0.178,
        0.0,
        0.3503,
        0.0,
        -0.101,
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
    ]
)

cardano_sentiment_preds["RISTRegressor"] = np.array(
    [
        0.0745,
        0.0745,
        0.448,
        0.0413,
        0.484,
        0.0825,
        0.0413,
        0.1419,
        -0.101,
        0.2814,
    ]
)

cardano_sentiment_preds["CanonicalIntervalForestRegressor"] = np.array(
    [0.28, 0.15, 0.33, 0.14, 0.19, 0.22, 0.15, 0.23, 0.12, 0.2]
)

cardano_sentiment_preds["DrCIFRegressor"] = np.array(
    [
        0.252,
        0.21,
        0.2664,
        0.1791,
        0.1999,
        0.1513,
        0.1448,
        0.0956,
        0.1547,
        0.111,
    ]
)

cardano_sentiment_preds["IntervalForestRegressor"] = np.array(
    [
        0.173,
        0.0907,
        0.2631,
        0.153,
        0.1266,
        0.1245,
        0.0296,
        0.1578,
        0.0389,
        0.1636,
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

cardano_sentiment_preds["RandomIntervalSpectralEnsembleRegressor"] = np.array(
    [
        0.4563,
        0.004,
        0.2486,
        0.101,
        0.1233,
        0.188,
        0.1242,
        0.238,
        0.1252,
        0.3264,
    ]
)

cardano_sentiment_preds["TimeSeriesForestRegressor"] = np.array(
    [
        0.2336,
        0.0385,
        0.5514,
        0.089,
        0.3174,
        -0.0505,
        -0.0505,
        -0.0505,
        -0.0093,
        0.1924,
    ]
)

cardano_sentiment_preds["RDSTRegressor"] = np.array(
    [
        0.1851,
        0.1864,
        0.1882,
        0.1875,
        0.1866,
        0.1862,
        0.1859,
        0.1857,
        0.1851,
        0.1878,
    ]
)

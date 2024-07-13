"""Dictionaries of expected outputs of regressor predict runs."""

import numpy as np

# predict results on Covid3Month data
covid_3month_preds = dict()

# predict results on CardanoSentiment data
cardano_sentiment_preds = dict()


covid_3month_preds["FreshPRINCERegressor"] = np.array(
    [
        0.0491,
        0.0781,
        0.0409,
        0.0412,
        0.0612,
        0.0337,
        0.0492,
        0.1111,
        0.1029,
        0.0344,
    ]
)

cardano_sentiment_preds["FreshPRINCERegressor"] = np.array(
    [
        0.3672,
        0.108,
        0.3813,
        0.0617,
        0.5077,
        0.3122,
        0.1335,
        0.3857,
        0.0954,
        0.4744,
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

cardano_sentiment_preds["MultiRocketHydraRegressor"] = np.array(
    [
        0.423,
        0.2261,
        0.5539,
        0.1887,
        0.4114,
        0.0997,
        0.1863,
        0.0313,
        -0.1093,
        0.4305,
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


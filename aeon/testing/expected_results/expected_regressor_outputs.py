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
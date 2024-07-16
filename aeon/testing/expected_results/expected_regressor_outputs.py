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

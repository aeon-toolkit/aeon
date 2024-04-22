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

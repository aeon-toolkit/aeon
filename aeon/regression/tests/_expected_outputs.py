# -*- coding: utf-8 -*-
"""Dictionaries of expected outputs of regressor predict runs."""

import numpy as np

# predict results on Covid3Month data
covid_3month_preds = dict()

# predict results on CardanoSentiment data
cardano_sentiment_preds = dict()


covid_3month_preds["FreshPRINCERegressor"] = np.array(
    [
        0.02293673,
        0.05257829,
        0.02664756,
        0.05216405,
        0.05694562,
        0.04583259,
        0.0628256,
        0.11111111,
        0.09460317,
        0.03036391,
    ]
)
cardano_sentiment_preds["FreshPRINCERegressor"] = np.array(
    [
        0.39238,
        0.35529,
        0.14529,
        0.06912,
        -0.29213,
        0.20573,
        -0.05512,
        -0.355,
        0.33625,
        0.21085,
    ]
)

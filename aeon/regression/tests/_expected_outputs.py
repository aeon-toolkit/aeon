# -*- coding: utf-8 -*-
"""Dictionaries of expected outputs of regressor predict runs."""

import numpy as np

# predict results on Covid3Month data
covid_3month_preds = dict()

# predict results on CardanoSentiment data
cardano_sentiment_preds = dict()


covid_3month_preds["FreshPRINCERegressor"] = np.array(
    [
        0.07631480,
        0.03413321,
        0.04036498,
        0.02149011,
        0.03624045,
        0.02760669,
        0.03473498,
        0.07489107,
        0.0,
        0.05722531,
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

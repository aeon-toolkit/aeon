# -*- coding: utf-8 -*-
"""Dictionaries of expected outputs of regressor predict runs."""

import numpy as np

# predict results on Covid3Month data
covid_3month_preds = dict()

# predict results on CardanoSentiment data
cardano_sentiment_proba = dict()


covid_3month_preds["FreshPRINCERegressor"] = np.array(
    [
        0.02294,
        0.05258,
        0.02665,
        0.05216,
        0.05695,
        0.04583,
        0.06283,
        0.11111,
        0.0946,
        0.03036,
    ]
)
cardano_sentiment_proba["FreshPRINCERegressor"] = np.array(
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

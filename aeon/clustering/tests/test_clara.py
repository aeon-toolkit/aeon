# -*- coding: utf-8 -*-
"""Tests for time series k-medoids."""
import numpy as np
from sklearn import metrics
from sklearn.utils import check_random_state

from aeon.clustering.clara import TimeSeriesCLARA
from aeon.datasets import load_basic_motions, load_gunpoint
from aeon.distances import euclidean_distance


def test_clara_uni():
    """Test implementation of Kmedoids."""
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")

    num_points = 10

    X_train = X_test[:num_points]
    y_test = y_test[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]

    kmedoids = TimeSeriesCLARA(
        random_state=1,
        n_init=2,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
    )
    train_medoids_result = kmedoids.fit_predict(X_train)
    train_score = metrics.rand_score(y_train, train_medoids_result)
    test_medoids_result = kmedoids.predict(X_test)
    test_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)
    joe = ""

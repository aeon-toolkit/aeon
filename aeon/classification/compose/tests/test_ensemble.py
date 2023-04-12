# -*- coding: utf-8 -*-
"""Test the ComposableTimeSeriesForestClassifier."""

__author__ = ["mloning"]

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from aeon.classification.compose import ComposableTimeSeriesForestClassifier
from aeon.datasets import load_unit_test
from aeon.transformations.panel.summarize import RandomIntervalFeatureExtractor
from aeon.transformations.series.adapt import TabularToSeriesAdaptor
from aeon.utils.slope_and_trend import _slope

rng = np.random.RandomState(42)
X = rng.rand(10, 1, 20)
y = rng.randint(0, 2, 10)
n_classes = len(np.unique(y))

mean_transformer = TabularToSeriesAdaptor(
    FunctionTransformer(func=np.mean, validate=False, kw_args={"axis": 0})
)
std_transformer = TabularToSeriesAdaptor(
    FunctionTransformer(func=np.std, validate=False, kw_args={"axis": 0})
)


# Check simple cases.
def test_tsf_predict_proba():
    """Test composable TSF predict proba."""
    clf = ComposableTimeSeriesForestClassifier(n_estimators=2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], n_classes)
    np.testing.assert_array_equal(np.ones(X.shape[0]), np.sum(proba, axis=1))

    # test single row input
    y_proba = clf.predict_proba(X[[0], :])
    assert y_proba.shape == (1, n_classes)

    y_pred = clf.predict(X[[0], :])
    assert y_pred.shape == (1,)


# Compare TimeSeriesForest ensemble predictions using pipeline as estimator
@pytest.mark.parametrize("n_intervals", ["sqrt", 1])
@pytest.mark.parametrize("n_estimators", [1, 3])
def test_tsf_predictions(n_estimators, n_intervals):
    """Test TSF predictions."""
    random_state = 1234
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    features = [np.mean, np.std, _slope]
    steps = [
        (
            "transform",
            RandomIntervalFeatureExtractor(
                random_state=random_state, features=features
            ),
        ),
        ("clf", DecisionTreeClassifier(random_state=random_state, max_depth=2)),
    ]
    estimator = Pipeline(steps)

    clf1 = ComposableTimeSeriesForestClassifier(
        estimator=estimator, random_state=random_state, n_estimators=n_estimators
    )
    clf1.fit(X_train, y_train)
    a = clf1.predict_proba(X_test)

    # default, semi-modular implementation using
    # RandomIntervalFeatureExtractor internally
    clf2 = ComposableTimeSeriesForestClassifier(
        random_state=random_state, n_estimators=n_estimators
    )
    clf2.fit(X_train, y_train)
    b = clf2.predict_proba(X_test)

    np.testing.assert_array_equal(a, b)

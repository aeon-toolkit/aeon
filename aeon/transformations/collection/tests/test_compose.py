# -*- coding: utf-8 -*-
"""Tests for collection composers."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from aeon.transformations.collection.compose import ColumnTransformer
from aeon.transformations.collection.reduce import Tabularizer
from aeon.utils._testing.collection import make_nested_dataframe_data


def test_ColumnTransformer_pipeline():
    """Test pipeline with ColumnTransformer."""
    X_train, y_train = make_nested_dataframe_data(n_channels=2)
    X_test, y_test = make_nested_dataframe_data(n_channels=2)
    X_train.columns = ["dim_0", "dim_1"]
    X_test.columns = ["dim_0", "dim_1"]

    # using Identity function transformations (transform series to series)
    def id_func(X):
        return X

    column_transformer = ColumnTransformer(
        [
            ("id0", FunctionTransformer(func=id_func, validate=False), ["dim_0"]),
            ("id1", FunctionTransformer(func=id_func, validate=False), ["dim_1"]),
        ]
    )
    steps = [
        ("extract", column_transformer),
        ("tabularise", Tabularizer()),
        ("classify", RandomForestClassifier(n_estimators=2, random_state=1)),
    ]
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == y_test.shape[0]
    np.testing.assert_array_equal(np.unique(y_pred), np.unique(y_test))

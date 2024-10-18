"""Tests for the RIST estimators."""

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import FunctionTransformer

from aeon.classification.hybrid import RISTClassifier
from aeon.regression.hybrid import RISTRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["statsmodels", "pycatch22"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rist_soft_dependencies():
    """Test the RIST class with different soft dependencies."""
    rist = RISTClassifier()
    assert rist.get_tag("python_dependencies") == "statsmodels"

    rist = RISTClassifier(use_pycatch22=True)
    assert rist.get_tag("python_dependencies") == ["statsmodels", "pycatch22"]

    X, y = make_example_3d_numpy()
    rist.fit(X, y)
    preds = rist.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 10


@pytest.mark.skipif(
    not _check_soft_dependencies(["statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rist_estimator_input():
    """Test the RIST class different with estimator input."""
    X, y = make_example_3d_numpy()

    rist = RISTClassifier(n_intervals=3, n_shapelets=3, series_transformers=None)
    rist.fit(X, y)
    assert isinstance(rist._estimator, ExtraTreesClassifier)

    rist = RISTRegressor(n_intervals=3, n_shapelets=3, series_transformers=None)
    rist.fit(X, y)
    assert isinstance(rist._estimator, ExtraTreesRegressor)

    with pytest.raises(
        ValueError, match="base_estimator must be a scikit-learn BaseEstimator"
    ):
        rist = RISTClassifier(estimator="invalid")
        rist.fit(X, y)

    rist = RISTClassifier(
        estimator=RidgeClassifierCV(),
        n_intervals=3,
        n_shapelets=3,
        series_transformers=None,
    )
    rist.fit(X, y)
    proba = rist.predict_proba(X)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (10, 2)


@pytest.mark.skipif(
    not _check_soft_dependencies(["statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rist_series_transform_input():
    """Test the RIST class series transformer input."""
    X, y = make_example_3d_numpy()

    rist = RISTClassifier(
        n_intervals=3, n_shapelets=3, estimator=ExtraTreesClassifier(n_estimators=10)
    )
    rist.fit(X, y)

    assert len(rist._series_transformers) == 4
    assert rist._series_transformers[0] is None
    assert isinstance(rist._series_transformers[1], FunctionTransformer)
    assert isinstance(rist._series_transformers[2], PeriodogramTransformer)
    assert isinstance(rist._series_transformers[3], ARCoefficientTransformer)

    rist = RISTClassifier(
        series_transformers=[
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
        ],
        n_intervals=3,
        n_shapelets=3,
        estimator=ExtraTreesClassifier(n_estimators=10),
    )
    rist.fit(X, y)

    assert len(rist._series_transformers) == 2
    assert rist._series_transformers[0] is None
    assert isinstance(rist._series_transformers[1], FunctionTransformer)

    rist = RISTClassifier(
        series_transformers=None,
        n_intervals=3,
        n_shapelets=3,
        estimator=ExtraTreesClassifier(n_estimators=10),
    )
    rist.fit(X, y)

    assert rist._series_transformers == [None]


@pytest.mark.skipif(
    not _check_soft_dependencies(["statsmodels"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rist_n_interval_and_n_shapelet_lambda():
    """Test the RIST lambda input for n_intervals and n_shapelets."""
    X, y = make_example_3d_numpy()

    rist = RISTClassifier(
        n_intervals=lambda X: len(X),
        n_shapelets=lambda X: len(X),
        series_transformers=[
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
        ],
        estimator=ExtraTreesClassifier(n_estimators=10),
    )
    rist.fit(X, y)
    preds = rist.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 10

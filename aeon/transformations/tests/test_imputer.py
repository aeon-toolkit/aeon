"""Unit tests of Imputer functionality."""

__maintainer__ = []
__all__ = []

import numpy as np
import pandas as pd
import pytest

from aeon.forecasting.model_selection import temporal_train_test_split
from aeon.forecasting.naive import NaiveForecaster
from aeon.testing.utils.data_gen import make_forecasting_problem
from aeon.transformations.impute import Imputer

y, X = make_forecasting_problem(make_X=True)

X.iloc[3, 0] = np.nan
X.iloc[3, 1] = np.nan
X.iloc[0, 1] = np.nan
X.iloc[-1, 1] = np.nan

y.iloc[3] = np.nan
y.iloc[0] = np.nan
y.iloc[-1] = np.nan


@pytest.mark.parametrize("forecaster", [None, NaiveForecaster()])
@pytest.mark.parametrize("X", [y, X])
@pytest.mark.parametrize(
    "method",
    [
        "drift",
        "linear",
        "nearest",
        "constant",
        "mean",
        "median",
        "backfill",
        "pad",
        "random",
        "forecaster",
    ],
)
def test_imputer(method, X, forecaster):
    """Test univariate and multivariate Imputer with all methods."""
    forecaster = NaiveForecaster() if method == "forecaster" else forecaster

    t = Imputer(method=method, forecaster=forecaster)
    y_hat = t.fit_transform(X)
    assert not y_hat.isnull().to_numpy().any()

    # test train and transform data is different
    X_train, X_test = temporal_train_test_split(X, test_size=5)

    t = Imputer(method=method, forecaster=forecaster)
    t = t.fit(X_train)
    y_hat = t.transform(X_test)
    assert not y_hat.isnull().to_numpy().any()

    # test some columns only contain NaN, either in fit or transform
    t = Imputer(method=method, forecaster=forecaster)
    # one column only contains NaN
    if isinstance(X, pd.Series):
        X = X.to_frame()
        X.iloc[:, 0] = np.nan
        X = pd.Series(X.iloc[:, 0])

    y_hat = t.fit_transform(X)
    assert not y_hat.isnull().to_numpy().any()

    if isinstance(X, pd.DataFrame):
        X_train.iloc[:, 0] = np.nan
        X_test.iloc[:, 1] = np.nan

        t = Imputer(method=method, forecaster=NaiveForecaster())
        t = t.fit(X_train)
        y_hat = t.transform(X_test)
        assert not y_hat.isnull().to_numpy().any()

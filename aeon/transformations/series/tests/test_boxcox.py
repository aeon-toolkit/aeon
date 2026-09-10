"""Tests for BoxCoxTransformer."""

import numpy as np
import pytest
from scipy.stats import boxcox

from aeon.datasets import load_airline
from aeon.transformations.series._boxcox import BoxCoxTransformer


def test_boxcox_against_scipy():
    """Test BoxCoxTransformer against scipy implementation."""
    y = load_airline()

    t = BoxCoxTransformer()
    actual = t.fit_transform(y)

    excepted, expected_lambda = boxcox(y)

    np.testing.assert_array_equal(actual, excepted)
    assert t.lambda_ == expected_lambda


@pytest.mark.parametrize("bounds", [(0, 1), (-1, 0), (-1, 2)])
@pytest.mark.parametrize(
    "method, sp", [("mle", None), ("pearsonr", None), ("guerrero", 5)]
)
def test_lambda_bounds(bounds, method, sp):
    """Test lambda bounds for BoxCox."""
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method=method, sp=sp)
    t.fit(y)
    assert bounds[0] < t.lambda_ < bounds[1]


@pytest.mark.parametrize(
    "bounds, r_lambda",
    [
        ((0, 1), 6.61069613518961e-05),
        ((-1, 0), -0.156975034727656),
        ((-1, 2), -0.156981228426408),
    ],
)
def test_guerrero_against_r_implementation(bounds, r_lambda):
    """Test BoxCoxTransformer against forecast guerrero method.

    Testing lambda values estimated by the R implementation of the Guerrero method
    https://github.com/robjhyndman/forecast/blob/master/R/guerrero.R
    against the guerrero method in _BoxCoxTransformer.
    R code to generate the hardcoded value for bounds=(-1, 2) used in the test
    ('Airline.csv' contains the data from 'load_airline()'):
        airline_file <- read.csv(file = 'Airline.csv')[,c('Passengers')]
        airline.ts <- ts(airline_file)
        guerrero(airline.ts, lower=-1, upper=2, nonseasonal.length = 20)
    Output:
        -0.156981228426408
    """
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method="guerrero", sp=20)
    t.fit(y)
    np.testing.assert_almost_equal(t.lambda_, r_lambda, decimal=4)


def test_boxcox_inverse_transform_roundtrip():
    """Test fitted Box-Cox transform is numerically invertible on positive data."""
    y = load_airline()
    t = BoxCoxTransformer()
    Xt = t.fit_transform(y)
    Xinv = t.inverse_transform(Xt)
    np.testing.assert_allclose(
        np.asarray(Xinv).squeeze(), np.asarray(y).squeeze(), rtol=1e-8
    )


def test_boxcox_all_method():
    """``method='all'`` returns the Pearson and MLE estimates in that order."""
    y = load_airline()
    expected = np.array(
        [
            BoxCoxTransformer(method="pearsonr").fit(y).lambda_,
            BoxCoxTransformer(method="mle").fit(y).lambda_,
        ]
    )
    actual = BoxCoxTransformer(method="all").fit(y).lambda_
    np.testing.assert_allclose(actual, expected)


def test_boxcox_invalid_method_raises():
    """Test fitting rejects unknown lambda-optimization methods."""
    y = load_airline()
    t = BoxCoxTransformer(method="bogus")
    with pytest.raises(ValueError, match="not recognized"):
        t.fit(y)


def test_boxcox_guerrero_invalid_sp_raises():
    """Test Guerrero lambda estimation requires seasonal periodicity >= 2."""
    y = load_airline()
    t = BoxCoxTransformer(method="guerrero", sp=1)
    with pytest.raises(ValueError, match="seasonal periodicity"):
        t.fit(y)


def test_boxcox_invalid_bounds_raises():
    """Fitting rejects bounds that are not a lower-upper pair."""
    with pytest.raises(ValueError, match="tuple of length 2"):
        BoxCoxTransformer(bounds=(1, 2, 3)).fit(load_airline())

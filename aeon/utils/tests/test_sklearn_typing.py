"""Tests for sklearn typing utilities in utils.aeon."""

__author__ = ["fkiraly"]


import pytest
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from aeon.classification.feature_based import SummaryClassifier
from aeon.forecasting.naive import NaiveForecaster
from aeon.utils.sklearn import is_sklearn_estimator, sklearn_estimator_identifier

CORRECT_IDENTIFIERS = {
    KMeans: "clusterer",
    KNeighborsClassifier: "classifier",
    KNeighborsRegressor: "regressor",
    StandardScaler: "transformer",
}

sklearn_estimators = list(CORRECT_IDENTIFIERS.keys())
aeon_estimators = [SummaryClassifier, NaiveForecaster]


@pytest.mark.parametrize("estimator", sklearn_estimators)
def test_is_sklearn_estimator_positive(estimator):
    """Test that is_sklearn_estimator recognizes positive examples correctly."""
    msg = (
        f"is_sklearn_estimator incorrectly considers {estimator.__name__} "
        f"as not an sklearn estimator (output False), but output should be True"
    )
    assert is_sklearn_estimator(estimator), msg


@pytest.mark.parametrize("estimator", aeon_estimators)
def test_is_sklearn_estimator_negative(estimator):
    """Test that is_sklearn_estimator recognizes negative examples correctly."""
    msg = (
        f"is_sklearn_estimator incorrectly considers {estimator.__name__} "
        f"as an sklearn estimator (output True), but output should be False"
    )
    assert not is_sklearn_estimator(estimator), msg


@pytest.mark.parametrize("estimator", sklearn_estimators)
def test_sklearn_identifiers(estimator):
    """Test that sklearn_estimator_identifier returns the correct identifier string."""
    identifier = sklearn_estimator_identifier(estimator)
    expected_identifier = CORRECT_IDENTIFIERS[estimator]
    msg = (
        f"is_sklearn_estimator returns the incorrect identifier string. Should be"
        f" {expected_identifier}, but {identifier} was returned."
    )
    assert identifier == expected_identifier, msg

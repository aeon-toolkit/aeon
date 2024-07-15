"""Unit tests for sklearn regressors."""

__maintainer__ = ["MatthewMiddlehurst"]

from sklearn.utils.estimator_checks import parametrize_with_checks

from aeon.regression.sklearn import RotationForestRegressor


@parametrize_with_checks([RotationForestRegressor(n_estimators=3)])
def test_sklearn_compatible_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    check(estimator)

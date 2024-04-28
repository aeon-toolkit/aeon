"""Test YtoX."""

__maintainer__ = []

from numpy.testing import assert_array_equal

from aeon.datasets import load_longley
from aeon.forecasting.model_selection import temporal_train_test_split
from aeon.transformations.compose import YtoX
from aeon.transformations.exponent import ExponentTransformer
from aeon.transformations.lag import Lag

y, X = load_longley()
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)


def test_featurized_values():
    """Test against plain transformation.

    Test to check that the featurized values are same as if transformation
    is done without YtoX.
    """
    lags = len(y_test)
    featurizer = YtoX() * ExponentTransformer() * Lag(lags)
    featurizer.fit(X_train, y_train)
    X_hat = featurizer.transform(X_test, y_test)

    exp_transformer = ExponentTransformer()
    expected_len = lags + len(y_test)
    y_hat = exp_transformer.fit_transform(y[-expected_len:])
    assert_array_equal(X_hat[f"lag_{lags}__TOTEMP"].values, y_hat.values)

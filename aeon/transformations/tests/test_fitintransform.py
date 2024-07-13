"""Unit tests of FitInTransform functionality."""

__maintainer__ = []
__all__ = []

import numpy as np

from aeon.transformations.compose import FitInTransform
from aeon.transformations.series._boxcox import BoxCoxTransformer


def test_transform_fitintransform():
    """Test fit/transform against _BoxCoxTransformer."""
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X_test = np.array([5, 4, 5, 4, 5, 4, 5, 4, 5, 4])
    fitintransform = FitInTransform(BoxCoxTransformer())
    fitintransform.fit(X=X_train)
    y_hat = fitintransform.transform(X=X_test)

    y_hat_expected = BoxCoxTransformer().fit_transform(X_test)
    np.testing.assert_array_equal(y_hat, y_hat_expected)

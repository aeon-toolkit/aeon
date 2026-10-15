"""Test TSFresh regressor."""

import numpy as np
import pytest

from aeon.regression.feature_based import TSFreshRegressor
from aeon.testing.data_generation import make_example_3d_numpy_list
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_tsfresh_regressor_no_features_unequal_length_list():
    """Predict the training mean for list input when no features are kept."""
    X, y = make_example_3d_numpy_list(
        n_cases=4,
        min_n_timepoints=3,
        max_n_timepoints=5,
        regression_target=True,
        random_state=0,
    )
    reg = TSFreshRegressor(relevant_feature_extractor=True)
    with pytest.warns(UserWarning, match="mean of the training targets"):
        reg.fit(X, y)
    assert reg._return_mean is True
    p = reg.predict(X)
    np.testing.assert_allclose(p, np.full(len(X), np.mean(y)))

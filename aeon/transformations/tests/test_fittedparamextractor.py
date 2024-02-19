"""Tests for FittedParamExtractor."""

__maintainer__ = []
__all__ = []

import pytest

from aeon.datasets import load_gunpoint
from aeon.forecasting.exp_smoothing import ExponentialSmoothing
from aeon.transformations.summarize import FittedParamExtractor
from aeon.utils.validation._dependencies import _check_estimator_deps

X_train, y_train = load_gunpoint("train", return_type="nested_univ")


@pytest.mark.skipif(
    not _check_estimator_deps(ExponentialSmoothing, severity="none"),
    reason="skip test if required soft dependency for hmmlearn not available",
)
@pytest.mark.parametrize("param_names", ["initial_level"])
def test_fitted_param_extractor(param_names):
    forecaster = ExponentialSmoothing()
    t = FittedParamExtractor(forecaster=forecaster, param_names=param_names)
    Xt = t.fit_transform(X_train)
    assert Xt.shape == (X_train.shape[0], len(t._check_param_names(param_names)))

    # check specific value
    forecaster.fit(X_train.iloc[47, 0])
    fitted_param = forecaster.get_fitted_params()[param_names]
    assert Xt.iloc[47, 0] == fitted_param

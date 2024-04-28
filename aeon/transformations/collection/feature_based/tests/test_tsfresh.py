"""Tests for TSFreshFeatureExtractor."""

__maintainer__ = []

import numpy as np
import pytest

from aeon.datasets import load_unit_test
from aeon.transformations.collection.feature_based import TSFreshFeatureExtractor
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
@pytest.mark.parametrize("default_fc_parameters", ["minimal"])
def test_tsfresh_extractor(default_fc_parameters):
    """Test that mean feature of TSFreshFeatureExtract is identical with sample mean."""
    X = np.random.rand(10, 1, 30)

    transformer = TSFreshFeatureExtractor(
        default_fc_parameters=default_fc_parameters, disable_progressbar=True
    )

    Xt = transformer.fit_transform(X)
    actual = Xt.filter(like="__mean", axis=1).values.ravel()
    expected = np.mean(X, axis=2).flatten()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(
    #    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_kind_tsfresh_extractor():
    """Test extractor returns an array of expected num of cols."""
    X, y = load_unit_test()
    features_to_calc = [
        "dim_0__quantile__q_0.6",
        "dim_0__longest_strike_above_mean",
        "dim_0__variance",
    ]
    ts_custom = TSFreshFeatureExtractor(
        kind_to_fc_parameters=features_to_calc, disable_progressbar=True
    )
    Xts_custom = ts_custom.fit_transform(X)
    assert Xts_custom.shape[1] == len(features_to_calc)

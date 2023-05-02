# -*- coding: utf-8 -*-
"""Tests for TSFreshFeatureExtractor."""
__author__ = ["AyushmannSeth", "mloning"]

import pytest

from aeon.datasets import load_unit_test
from aeon.transformations.panel.tsfresh import TSFreshFeatureExtractor
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_kind_tsfresh_extractor():
    """Test extractor returns an array of expected num of cols."""
    X, y = load_unit_test(split="TRAIN")
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

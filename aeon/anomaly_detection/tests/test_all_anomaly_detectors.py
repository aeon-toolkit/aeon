"""Tests for all anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest

from aeon.base._base_series import VALID_INNER_TYPES
from aeon.registry import all_estimators
from aeon.testing.data_generation._legacy import make_series

ALL_ANOMALY_DETECTORS = all_estimators(
    estimator_types="anomaly-detector",
    return_names=False,
)

labels = np.zeros(15, dtype=np.int_)
labels[np.random.choice(15, 5)] = 1
uv_series = make_series(n_timepoints=15, return_numpy=True, random_state=0)
uv_series[labels == 1] += 1
mv_series = make_series(
    n_timepoints=15, n_columns=2, return_numpy=True, random_state=0
).T
mv_series[:, labels == 1] += 1


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_anomaly_detector_univariate(anomaly_detector):
    """Test the anomaly detector on univariate data."""
    try:
        ad = anomaly_detector.create_test_instance()
    except ModuleNotFoundError:
        return None

    if anomaly_detector.get_class_tag(tag_name="capability:univariate"):
        pred = ad.fit_predict(uv_series, labels)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (15,)
        assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))
    else:
        with pytest.raises(ValueError, match="Univariate data not supported"):
            ad.fit_predict(uv_series, labels)


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_anomaly_detector_multivariate(anomaly_detector):
    """Test the anomaly detector on multivariate data."""
    try:
        ad = anomaly_detector.create_test_instance()
    except ModuleNotFoundError:
        return None

    if anomaly_detector.get_class_tag(tag_name="capability:multivariate"):
        pred = ad.fit_predict(mv_series, labels)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (15,)
        assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            ad.fit_predict(mv_series, labels)


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_anomaly_detector_overrides_and_tags(anomaly_detector):
    """Test compliance with the anomaly detector base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    assert "fit" not in anomaly_detector.__dict__
    assert "predict" not in anomaly_detector.__dict__
    assert "fit_predict" not in anomaly_detector.__dict__

    # Test that all anomaly detectors implement abstract predict.
    assert "_predict" in anomaly_detector.__dict__

    # axis class parameter is for internal use only
    assert "axis" not in anomaly_detector.__dict__

    # Test that fit_is_empty is correctly set
    fit_is_empty = anomaly_detector.get_class_tag(tag_name="fit_is_empty")
    assert not fit_is_empty == "_fit" not in anomaly_detector.__dict__

    # Test valid tag for X_inner_type
    X_inner_type = anomaly_detector.get_class_tag(tag_name="X_inner_type")
    assert X_inner_type in VALID_INNER_TYPES

    # Must have at least one set to True
    multi = anomaly_detector.get_class_tag(tag_name="capability:multivariate")
    uni = anomaly_detector.get_class_tag(tag_name="capability:univariate")
    assert multi or uni

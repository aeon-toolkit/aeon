__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest

from aeon.base._base_series import VALID_INNER_TYPES
from aeon.registry import all_estimators
from aeon.testing.utils.data_gen import make_series

ALL_ANOMALY_DETECTORS = all_estimators(
    estimator_types="anomaly-detector",
    return_names=False,
)


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_anomaly_detector_univariate(anomaly_detector):
    ad = anomaly_detector.create_test_instance()
    series = make_series(n_timepoints=10, return_numpy=True)
    y = np.random.randint(0, 2, 10)

    if anomaly_detector.get_class_tag(tag_name="capability:univariate"):
        pred = ad.fit_predict(series, y)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (10,)
        assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))
    else:
        with pytest.raises(ValueError, match="Univariate data not supported"):
            ad.fit_predict(series, y)


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_anomaly_detector_multivariate(anomaly_detector):
    ad = anomaly_detector.create_test_instance()
    series = make_series(n_timepoints=10, n_columns=2, return_numpy=True).T
    y = np.random.randint(0, 2, 10)

    if anomaly_detector.get_class_tag(tag_name="capability:multivariate"):
        pred = ad.fit_predict(series, y)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (10,)
        assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            ad.fit_predict(series, y)


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

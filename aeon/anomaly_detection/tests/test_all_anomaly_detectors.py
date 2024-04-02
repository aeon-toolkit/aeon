import numpy as np
import pytest

from aeon.base._base_series import VALID_INNER_TYPES
from aeon.registry import all_estimators

ALL_ANOMALY_DETECTORS = all_estimators(
    estimator_types="anomaly-detector",
    return_names=False,
)


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_anomaly_detector_base_functionality(anomaly_detector):
    """Test compliance with the base class contract."""
    # Test they dont override final methods, because python does not enforce this
    assert "fit" not in anomaly_detector.__dict__
    assert "predict" not in anomaly_detector.__dict__
    assert "fit_predict" not in anomaly_detector.__dict__

    # Test that all segmenters implement abstract predict.
    assert "_predict" in anomaly_detector.__dict__

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


def _assert_output(output, dense, length):
    """Assert the properties of the anomaly detector output."""
    assert isinstance(output, np.ndarray)
    if dense:  # Change points returned
        assert len(output) < length
        assert max(output) < length
        assert min(output) >= 0
        # Test in ascending order
        assert all(output[i] <= output[i + 1] for i in range(len(output) - 1))
    else:  # Segment labels returned, must be same length sas series
        assert len(output) == length


@pytest.mark.parametrize("anomaly_detector", ALL_ANOMALY_DETECTORS)
def test_segmenter_instance(anomaly_detector):
    """Test anomaly detectors."""
    instance = anomaly_detector.create_test_instance()

    multivariate = anomaly_detector.get_class_tag(tag_name="capability:multivariate")
    X = np.random.random(size=(5, 20))
    # Also tests does not fail if y is passed
    y = np.array([0, 0, 0, 1, 1])

    # Test that capability:multivariate is correctly set
    dense = anomaly_detector.get_class_tag(tag_name="returns_dense")
    if multivariate:
        output = instance.fit_predict(X, y, axis=1)
        _assert_output(output, dense, X.shape[1])
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            instance.fit_predict(X, y, axis=1)

    # Test that output is correct type
    X = np.random.random(size=(20))
    uni = anomaly_detector.get_class_tag(tag_name="capability:univariate")
    if uni:
        output = instance.fit_predict(X, y=X)
        _assert_output(output, dense, len(X))
    else:
        with pytest.raises(ValueError, match="Univariate data not supported"):
            instance.fit_predict(X)

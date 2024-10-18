"""Test base class identifier."""

import pytest

from aeon.testing.mock_estimators import (
    MockClassifier,
    MockRegressor,
    MockSeriesTransformer,
)
from aeon.testing.mock_estimators._mock_anomaly_detectors import MockAnomalyDetector
from aeon.utils.base import BASE_CLASS_REGISTER, get_identifier


@pytest.mark.parametrize("item", BASE_CLASS_REGISTER.items())
def test_base_type_inference(item):
    """Check that get_identifier returns the correct key for base classes."""
    identifier = get_identifier(item[1])

    assert (
        identifier == item[0]
    ), f"get_identifier returned {identifier} for {item[1]}, expected {item[0]}"


def test_get_identifier():
    """Check that get_identifier returns correct output for selected classes."""
    assert (
        get_identifier(MockClassifier)
        == get_identifier(MockClassifier())
        == "classifier"
    )
    assert (
        get_identifier(MockRegressor) == get_identifier(MockRegressor()) == "regressor"
    )
    assert (
        get_identifier(MockAnomalyDetector)
        == get_identifier(MockAnomalyDetector())
        == "anomaly-detector"
    )
    assert (
        get_identifier(MockSeriesTransformer)
        == get_identifier(MockSeriesTransformer())
        == "series-transformer"
    )

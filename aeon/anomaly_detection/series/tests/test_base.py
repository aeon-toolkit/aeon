"""Test series anomaly detection base class."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)
from aeon.testing.mock_estimators._mock_anomaly_detectors import (
    MockAnomalyDetector,
    MockAnomalyDetectorRequiresFit,
    MockAnomalyDetectorRequiresY,
)

test_series = make_example_1d_numpy(n_timepoints=10)
test_series_2d = make_example_2d_numpy_series(n_timepoints=10, n_channels=2)
test_series_pd = pd.Series(test_series)
test_series_pd_2d = pd.DataFrame(test_series_2d)
test_y = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])


@pytest.mark.parametrize(
    "series", [test_series, test_series_2d, test_series_pd, test_series_pd_2d]
)
def test_fit(series):
    """Test the anomaly detection fit method."""
    ad = MockAnomalyDetector()
    ad_fit = MockAnomalyDetectorRequiresFit()
    ad_y = MockAnomalyDetectorRequiresY()

    # test fit is empty
    ad.fit(series)
    assert ad.is_fitted

    ad.fit(series, test_y)
    assert ad.is_fitted

    # test requires fit
    ad_fit.fit(series)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X.squeeze(), series)

    ad_fit.fit(series, test_y)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X.squeeze(), series)

    # test requires y
    with pytest.raises(ValueError, match="Tag requires_y is true, but"):
        ad_y.fit(series)

    ad_y.fit(series, test_y)
    assert ad_y.is_fitted
    assert_almost_equal(ad_y._X.squeeze(), series)
    assert_almost_equal(ad_y._y, test_y)


@pytest.mark.parametrize("series", [test_series_2d, test_series_pd_2d])
def test_fit_axis(series):
    """Test the anomaly detection fit method with axis."""
    ad_fit = MockAnomalyDetectorRequiresFit()
    assert ad_fit.axis == 1
    invert_series = series.T

    ad_fit.fit(series)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X, series)

    ad_fit.fit(invert_series, axis=0)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X, series)


@pytest.mark.parametrize(
    "series", [test_series, test_series_2d, test_series_pd, test_series_pd_2d]
)
def test_predict(series):
    """Test the anomaly detection predict method."""
    ad = MockAnomalyDetector()
    ad_fit = MockAnomalyDetectorRequiresFit()

    # test fit is empty
    pred = ad.predict(series)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))

    ad.fit(series)
    pred = ad.predict(series)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))

    # test requires fit
    with pytest.raises(ValueError, match="has not been fitted yet"):
        ad_fit.predict(series)
    ad_fit.fit(series)

    pred = ad_fit.predict(series)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))


@pytest.mark.parametrize("series", [test_series_2d, test_series_pd_2d])
def test_predict_axis(series):
    """Test the anomaly detection predict method with axis."""
    ad = MockAnomalyDetector()
    assert ad.axis == 1
    invert_series = series.T

    pred = ad.predict(series)
    assert len(pred) == series.shape[1]

    pred = ad.predict(invert_series, axis=0)
    assert len(pred) == series.shape[1]


@pytest.mark.parametrize(
    "series", [test_series, test_series_2d, test_series_pd, test_series_pd_2d]
)
def test_fit_predict(series):
    """Test the anomaly detection fit_predict method."""
    ad = MockAnomalyDetector()
    ad_fit = MockAnomalyDetectorRequiresFit()
    ad_y = MockAnomalyDetectorRequiresY()

    # test fit is empty
    pred = ad.fit_predict(series)
    assert ad.is_fitted
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))

    pred = ad.fit_predict(series, test_y)
    assert ad.is_fitted
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))

    # test requires fit
    pred = ad_fit.fit_predict(series)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X.squeeze(), series)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))

    pred = ad_fit.fit_predict(series, test_y)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X.squeeze(), series)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))

    # test requires y
    with pytest.raises(ValueError, match="Tag requires_y is true, but"):
        ad_y.fit_predict(series)

    pred = ad_y.fit_predict(series, test_y)
    assert ad_y.is_fitted
    assert_almost_equal(ad_y._X.squeeze(), series)
    assert_almost_equal(ad_y._y, test_y)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (10,)
    assert issubclass(pred.dtype.type, (np.integer, np.floating, np.bool_))


@pytest.mark.parametrize("series", [test_series_2d, test_series_pd_2d])
def test_fit_predict_axis(series):
    """Test the anomaly detection fit_predict method with axis."""
    ad_fit = MockAnomalyDetectorRequiresFit()
    assert ad_fit.axis == 1
    invert_series = series.T

    pred = ad_fit.fit_predict(series)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X, series)
    assert len(pred) == series.shape[1]

    pred = ad_fit.fit_predict(invert_series, axis=0)
    assert ad_fit.is_fitted
    assert_almost_equal(ad_fit._X, series)
    assert len(pred) == series.shape[1]

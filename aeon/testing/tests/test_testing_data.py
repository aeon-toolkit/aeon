"""Tests for testing data dictionaries."""

import numpy as np

from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_MULTIVARIATE_REGRESSION,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_REGRESSION,
    FULL_TEST_DATA_DICT,
    MISSING_VALUES_CLASSIFICATION,
    MISSING_VALUES_REGRESSION,
    MISSING_VALUES_SERIES,
    MISSING_VALUES_SERIES_ANOMALY,
    MULTIVARIATE_SERIES,
    MULTIVARIATE_SERIES_ANOMALY,
    UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION,
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_REGRESSION,
    UNIVARIATE_SERIES,
    UNIVARIATE_SERIES_ANOMALY,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES, SERIES_DATA_TYPES
from aeon.utils.validation.collection import has_missing as has_missing_collection
from aeon.utils.validation.collection import (
    is_collection,
    is_equal_length,
)
from aeon.utils.validation.collection import is_univariate as is_univariate_collection
from aeon.utils.validation.labels import (
    check_anomaly_detection_y,
    check_classification_y,
    check_regression_y,
)
from aeon.utils.validation.series import has_missing as has_missing_series
from aeon.utils.validation.series import (
    is_series,
)
from aeon.utils.validation.series import is_univariate as is_univariate_series


def test_datatype_exists():
    """Check that the basic testing data case has all data types."""
    for data in COLLECTIONS_DATA_TYPES:
        assert data in EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
        assert data in EQUAL_LENGTH_UNIVARIATE_REGRESSION

    for data in SERIES_DATA_TYPES:
        assert data in UNIVARIATE_SERIES
        assert data in UNIVARIATE_SERIES_ANOMALY


def test_testing_data_dict():
    """Test the contents of the test data dictionary."""
    for key in FULL_TEST_DATA_DICT:
        # format
        assert isinstance(FULL_TEST_DATA_DICT[key], dict)
        assert len(FULL_TEST_DATA_DICT[key]) == 2
        assert "train" in FULL_TEST_DATA_DICT[key]
        assert "test" in FULL_TEST_DATA_DICT[key]
        # data
        assert is_collection(FULL_TEST_DATA_DICT[key]["train"][0]) or is_series(
            FULL_TEST_DATA_DICT[key]["train"][0], include_2d=True
        )
        assert is_collection(FULL_TEST_DATA_DICT[key]["test"][0]) or is_series(
            FULL_TEST_DATA_DICT[key]["test"][0], include_2d=True
        )
        # label
        if FULL_TEST_DATA_DICT[key]["train"][1] is not None:
            assert isinstance(FULL_TEST_DATA_DICT[key]["train"][1], np.ndarray)
            assert isinstance(FULL_TEST_DATA_DICT[key]["test"][1], np.ndarray)
            assert FULL_TEST_DATA_DICT[key]["train"][1].ndim == 1
            assert FULL_TEST_DATA_DICT[key]["test"][1].ndim == 1


def test_equal_length_univariate_collection():
    """Test the contents of the equal length univariate data dictionaries."""
    for key in EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0], include_2d=True
        )
        assert is_univariate_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_y(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][1])

        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0], include_2d=True
        )
        assert is_univariate_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        check_classification_y(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][1])

    for key in EQUAL_LENGTH_UNIVARIATE_REGRESSION:
        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0], include_2d=True
        )
        assert is_univariate_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0]
        )
        check_regression_y(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][1])

        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0], include_2d=True
        )
        assert is_univariate_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0]
        )
        check_regression_y(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][1])


def test_unequal_length_univariate_collection():
    """Test the contents of the unequal length univariate data dictionary."""
    for key in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        assert is_univariate_collection(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_y(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        assert is_univariate_collection(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        check_classification_y(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][1])

    for key in UNEQUAL_LENGTH_UNIVARIATE_REGRESSION:
        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert is_univariate_collection(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0]
        )
        check_regression_y(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][1])

        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert is_univariate_collection(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0]
        )
        assert not is_equal_length(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert not has_missing_collection(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0]
        )
        check_regression_y(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][1])


def test_equal_length_multivariate_collection():
    """Test the contents of the equal length multivariate data dictionary."""
    for key in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0])
        assert not is_univariate_collection(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert is_equal_length(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not has_missing_collection(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_y(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not is_univariate_collection(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        check_classification_y(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][1])

    for key in EQUAL_LENGTH_MULTIVARIATE_REGRESSION:
        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert not is_univariate_collection(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        check_regression_y(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][1])

        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not is_univariate_collection(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not has_missing_collection(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0]
        )
        check_regression_y(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][1])


def test_unequal_length_multivariate_collection():
    """Test the contents of the unequal length multivariate data dictionary."""
    for key in UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        assert is_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not is_univariate_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_y(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not is_univariate_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        check_classification_y(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][1]
        )

    for key in UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION:
        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert not is_univariate_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        check_regression_y(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][1])

        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not is_univariate_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0]
        )
        assert not has_missing_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0]
        )
        check_regression_y(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][1])


def test_missing_values_collection():
    """Test the contents of the collection missing value data dictionary."""
    for key in MISSING_VALUES_CLASSIFICATION:
        assert is_collection(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        assert is_univariate_collection(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        assert is_equal_length(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        assert has_missing_collection(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        check_classification_y(MISSING_VALUES_CLASSIFICATION[key]["train"][1])

        assert is_collection(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        assert is_univariate_collection(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        assert is_equal_length(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        assert has_missing_collection(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        check_classification_y(MISSING_VALUES_CLASSIFICATION[key]["test"][1])

    for key in MISSING_VALUES_REGRESSION:
        assert is_collection(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert is_univariate_collection(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert is_equal_length(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert has_missing_collection(MISSING_VALUES_REGRESSION[key]["train"][0])
        check_regression_y(MISSING_VALUES_REGRESSION[key]["train"][1])

        assert is_collection(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert is_univariate_collection(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert is_equal_length(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert has_missing_collection(MISSING_VALUES_REGRESSION[key]["test"][0])
        check_regression_y(MISSING_VALUES_REGRESSION[key]["test"][1])


def test_univariate_series():
    """Test the contents of the univariate series data dictionary."""
    for key in UNIVARIATE_SERIES:
        assert is_series(UNIVARIATE_SERIES[key]["train"][0], include_2d=True)
        assert is_univariate_series(UNIVARIATE_SERIES[key]["train"][0], axis=1)
        assert not has_missing_series(UNIVARIATE_SERIES[key]["train"][0])
        assert UNIVARIATE_SERIES[key]["train"][1] is None

        assert is_series(UNIVARIATE_SERIES[key]["test"][0], include_2d=True)
        assert is_univariate_series(UNIVARIATE_SERIES[key]["test"][0], axis=1)
        assert not has_missing_series(UNIVARIATE_SERIES[key]["test"][0])
        assert UNIVARIATE_SERIES[key]["test"][1] is None

    for key in UNIVARIATE_SERIES_ANOMALY:
        assert is_series(UNIVARIATE_SERIES_ANOMALY[key]["train"][0], include_2d=True)
        assert is_univariate_series(UNIVARIATE_SERIES_ANOMALY[key]["train"][0], axis=1)
        assert not has_missing_series(UNIVARIATE_SERIES_ANOMALY[key]["train"][0])
        check_anomaly_detection_y(UNIVARIATE_SERIES_ANOMALY[key]["train"][1])

        assert is_series(UNIVARIATE_SERIES_ANOMALY[key]["test"][0], include_2d=True)
        assert is_univariate_series(UNIVARIATE_SERIES_ANOMALY[key]["test"][0], axis=1)
        assert not has_missing_series(UNIVARIATE_SERIES_ANOMALY[key]["test"][0])
        check_anomaly_detection_y(UNIVARIATE_SERIES_ANOMALY[key]["test"][1])


def test_multivariate_series():
    """Test the contents of the multivariate series data dictionary."""
    for key in MULTIVARIATE_SERIES:
        assert is_series(MULTIVARIATE_SERIES[key]["train"][0], include_2d=True)
        assert not is_univariate_series(MULTIVARIATE_SERIES[key]["train"][0], axis=1)
        assert not has_missing_series(MULTIVARIATE_SERIES[key]["train"][0])
        assert MULTIVARIATE_SERIES[key]["train"][1] is None

        assert is_series(MULTIVARIATE_SERIES[key]["test"][0], include_2d=True)
        assert not is_univariate_series(MULTIVARIATE_SERIES[key]["test"][0], axis=1)
        assert not has_missing_series(MULTIVARIATE_SERIES[key]["test"][0])
        assert MULTIVARIATE_SERIES[key]["test"][1] is None

    for key in MULTIVARIATE_SERIES_ANOMALY:
        assert is_series(MULTIVARIATE_SERIES_ANOMALY[key]["train"][0], include_2d=True)
        assert not is_univariate_series(
            MULTIVARIATE_SERIES_ANOMALY[key]["train"][0], axis=1
        )
        assert not has_missing_series(MULTIVARIATE_SERIES_ANOMALY[key]["train"][0])
        check_anomaly_detection_y(MULTIVARIATE_SERIES_ANOMALY[key]["train"][1])

        assert is_series(MULTIVARIATE_SERIES_ANOMALY[key]["test"][0], include_2d=True)
        assert not is_univariate_series(
            MULTIVARIATE_SERIES_ANOMALY[key]["test"][0], axis=1
        )
        assert not has_missing_series(MULTIVARIATE_SERIES_ANOMALY[key]["test"][0])
        check_anomaly_detection_y(MULTIVARIATE_SERIES_ANOMALY[key]["test"][1])


def test_missing_series():
    """Test the contents of the series missing value data dictionary."""
    for key in MISSING_VALUES_SERIES:
        assert is_series(MISSING_VALUES_SERIES[key]["train"][0], include_2d=True)
        assert is_univariate_series(MISSING_VALUES_SERIES[key]["train"][0], axis=1)
        assert has_missing_series(MISSING_VALUES_SERIES[key]["train"][0])
        assert MISSING_VALUES_SERIES[key]["train"][1] is None

        assert is_series(MISSING_VALUES_SERIES[key]["test"][0], include_2d=True)
        assert is_univariate_series(MISSING_VALUES_SERIES[key]["test"][0], axis=1)
        assert has_missing_series(MISSING_VALUES_SERIES[key]["test"][0])
        assert MISSING_VALUES_SERIES[key]["test"][1] is None

    for key in MISSING_VALUES_SERIES_ANOMALY:
        assert is_series(
            MISSING_VALUES_SERIES_ANOMALY[key]["train"][0], include_2d=True
        )
        assert is_univariate_series(
            MISSING_VALUES_SERIES_ANOMALY[key]["train"][0], axis=1
        )
        assert has_missing_series(MISSING_VALUES_SERIES_ANOMALY[key]["train"][0])
        check_anomaly_detection_y(MISSING_VALUES_SERIES_ANOMALY[key]["train"][1])

        assert is_series(MISSING_VALUES_SERIES_ANOMALY[key]["test"][0], include_2d=True)
        assert is_univariate_series(
            MISSING_VALUES_SERIES_ANOMALY[key]["test"][0], axis=1
        )
        assert has_missing_series(MISSING_VALUES_SERIES_ANOMALY[key]["test"][0])
        check_anomaly_detection_y(MISSING_VALUES_SERIES_ANOMALY[key]["test"][1])

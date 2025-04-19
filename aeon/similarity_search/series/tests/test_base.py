"""Test for series similarity search base class."""

__maintainer__ = ["baraline"]

from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockSeriesMotifSearch,
    MockSeriesNeighborsSearch,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT, _get_datatypes_for_estimator


def test_input_shape_fit_predict_collection_motifs():
    """Test input shapes."""
    estimator = MockSeriesMotifSearch()
    datatypes = _get_datatypes_for_estimator(estimator)
    # dummy data to pass to fit when testing predict/predict_proba
    for datatype in datatypes:
        X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
        X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
        estimator.fit(X_train, y_train).predict(X_test)


def test_input_shape_fit_predict_collection_neighbors():
    """Test input shapes."""
    estimator = MockSeriesNeighborsSearch()
    datatypes = _get_datatypes_for_estimator(estimator)
    # dummy data to pass to fit when testing predict/predict_proba
    for datatype in datatypes:
        X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
        X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
        estimator.fit(X_train, y_train).predict(X_test)

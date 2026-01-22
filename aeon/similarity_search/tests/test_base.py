"""Test for similarity search base class."""

__maintainer__ = ["baraline"]

from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockSubsequenceSearch,
    MockWholeSeriesSearch,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT, _get_datatypes_for_estimator


def test_input_shape_fit_predict():
    """Test input shapes for similarity search.

    Fit takes a collection (3D), predict takes a single series (2D).
    """
    estimator_ss = MockSubsequenceSearch()
    estimator_ws = MockWholeSeriesSearch()
    datatypes_ss = _get_datatypes_for_estimator(estimator_ss)
    datatypes_ws = _get_datatypes_for_estimator(estimator_ws)

    # dummy data to pass to fit when testing predict/predict_proba
    for datatype in datatypes_ss:
        X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
        X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
        # fit takes a collection, predict takes a single series
        estimator_ss.fit(X_train, y_train).predict(X_test[0])

    for datatype in datatypes_ws:
        X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
        X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
        # fit takes a collection, predict takes a single series
        estimator_ws.fit(X_train, y_train).predict(X_test[0])

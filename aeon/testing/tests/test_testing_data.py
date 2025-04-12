"""Tests for testing data dictionaries."""

import numpy as np
from sklearn.utils.multiclass import check_classification_targets

from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_MULTIVARIATE_REGRESSION,
    EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_REGRESSION,
    EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH,
    FULL_TEST_DATA_DICT,
    MISSING_VALUES_CLASSIFICATION,
    MISSING_VALUES_REGRESSION,
    UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION,
    UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH,
    UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_REGRESSION,
    UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation import (
    has_missing,
    is_collection,
    is_equal_length,
    is_single_series,
    is_univariate,
)


def test_datatype_exists():
    """Check that the basic testing data case has all data types."""
    for data in COLLECTIONS_DATA_TYPES:
        assert data in EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
        assert data in EQUAL_LENGTH_UNIVARIATE_REGRESSION


def test_testing_data_dict():
    """Test the contents of the test data dictionary."""
    for key in FULL_TEST_DATA_DICT:
        # format
        assert isinstance(FULL_TEST_DATA_DICT[key], dict)
        assert len(FULL_TEST_DATA_DICT[key]) == 2
        assert "train" in FULL_TEST_DATA_DICT[key]
        assert "test" in FULL_TEST_DATA_DICT[key]
        # data
        assert is_collection(FULL_TEST_DATA_DICT[key]["train"][0]) or is_single_series(
            FULL_TEST_DATA_DICT[key]["train"][0]
        )
        assert is_collection(FULL_TEST_DATA_DICT[key]["test"][0]) or is_single_series(
            FULL_TEST_DATA_DICT[key]["test"][0]
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
        assert is_univariate(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        assert not has_missing(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        check_classification_targets(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0], include_2d=True
        )
        assert is_univariate(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not has_missing(EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        check_classification_targets(
            EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][1]
        )

    for key in EQUAL_LENGTH_UNIVARIATE_REGRESSION:
        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0], include_2d=True
        )
        assert is_univariate(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert not has_missing(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert np.issubdtype(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][1].dtype, np.integer
        ) or np.issubdtype(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][1].dtype, np.floating
        )

        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0], include_2d=True
        )
        assert is_univariate(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert not has_missing(EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert np.issubdtype(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][1].dtype, np.integer
        ) or np.issubdtype(
            EQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][1].dtype, np.floating
        )

    for key in EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH:
        assert is_collection(
            EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0], include_2d=True
        )
        assert is_univariate(EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0])
        assert is_equal_length(
            EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not has_missing(
            EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not is_collection(
            EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        # assert is_univariate(
        #     EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0],
        # )
        assert is_equal_length(
            EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        assert not has_missing(
            EQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )


def test_unequal_length_univariate_collection():
    """Test the contents of the unequal length univariate data dictionary."""
    for key in UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION:
        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        assert is_univariate(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0])
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_targets(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        assert is_univariate(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not has_missing(UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][0])
        check_classification_targets(
            UNEQUAL_LENGTH_UNIVARIATE_CLASSIFICATION[key]["test"][1]
        )

    for key in UNEQUAL_LENGTH_UNIVARIATE_REGRESSION:
        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert is_univariate(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not has_missing(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][0])
        assert np.issubdtype(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][1].dtype, np.integer
        ) or np.issubdtype(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["train"][1].dtype, np.floating
        )

        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert is_univariate(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert not is_equal_length(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert not has_missing(UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][0])
        assert np.issubdtype(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][1].dtype, np.integer
        ) or np.issubdtype(
            UNEQUAL_LENGTH_UNIVARIATE_REGRESSION[key]["test"][1].dtype, np.floating
        )

    for key in UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH:
        assert is_collection(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0],
            include_2d=True,
        )
        assert is_univariate(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not is_collection(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        # assert is_univariate(
        #     UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0],
        # )
        assert is_equal_length(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_UNIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )


def test_equal_length_multivariate_collection():
    """Test the contents of the equal length multivariate data dictionary."""
    for key in EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0])
        assert not is_univariate(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert is_equal_length(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not has_missing(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_targets(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not is_univariate(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not has_missing(EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        check_classification_targets(
            EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][1]
        )

    for key in EQUAL_LENGTH_MULTIVARIATE_REGRESSION:
        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert not is_univariate(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert not has_missing(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert np.issubdtype(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][1].dtype, np.integer
        ) or np.issubdtype(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][1].dtype, np.floating
        )

        assert is_collection(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not is_univariate(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not has_missing(EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert np.issubdtype(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][1].dtype, np.integer
        ) or np.issubdtype(
            EQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][1].dtype, np.floating
        )

    for key in EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH:
        assert is_collection(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0],
            include_2d=True,
        )
        assert not is_univariate(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert is_equal_length(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not has_missing(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not is_collection(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        # assert not is_univariate(
        #     EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0],
        # )
        assert is_equal_length(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        assert not has_missing(
            EQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )


def test_unequal_length_multivariate_collection():
    """Test the contents of the unequal length multivariate data dictionary."""
    for key in UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION:
        assert is_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not is_univariate(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][0]
        )
        check_classification_targets(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["train"][1]
        )

        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0])
        assert not is_univariate(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][0]
        )
        check_classification_targets(
            UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION[key]["test"][1]
        )

    for key in UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION:
        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert not is_univariate(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0]
        )
        assert not has_missing(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][0])
        assert np.issubdtype(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][1].dtype, np.integer
        ) or np.issubdtype(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["train"][1].dtype, np.floating
        )

        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not is_univariate(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0]
        )
        assert not has_missing(UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][0])
        assert np.issubdtype(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][1].dtype, np.integer
        ) or np.issubdtype(
            UNEQUAL_LENGTH_MULTIVARIATE_REGRESSION[key]["test"][1].dtype, np.floating
        )

    for key in UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH:
        assert is_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0],
            include_2d=True,
        )
        assert not is_univariate(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["train"][0]
        )
        assert not is_collection(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        # assert not is_univariate(
        #     UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0],
        # )
        assert is_equal_length(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )
        assert not has_missing(
            UNEQUAL_LENGTH_MULTIVARIATE_SIMILARITY_SEARCH[key]["test"][0]
        )


def test_missing_values_collection():
    """Test the contents of the missing value data dictionary."""
    for key in MISSING_VALUES_CLASSIFICATION:
        assert is_collection(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        assert is_univariate(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        assert is_equal_length(MISSING_VALUES_CLASSIFICATION[key]["train"][0])
        assert has_missing(MISSING_VALUES_CLASSIFICATION[key]["train"][0])

        check_classification_targets(MISSING_VALUES_CLASSIFICATION[key]["train"][1])

        assert is_collection(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        assert is_univariate(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        assert is_equal_length(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        assert has_missing(MISSING_VALUES_CLASSIFICATION[key]["test"][0])
        check_classification_targets(MISSING_VALUES_CLASSIFICATION[key]["test"][1])

    for key in MISSING_VALUES_REGRESSION:
        assert is_collection(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert is_univariate(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert is_equal_length(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert has_missing(MISSING_VALUES_REGRESSION[key]["train"][0])
        assert np.issubdtype(
            MISSING_VALUES_REGRESSION[key]["train"][1].dtype, np.integer
        ) or np.issubdtype(
            MISSING_VALUES_REGRESSION[key]["train"][1].dtype, np.floating
        )

        assert is_collection(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert is_univariate(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert is_equal_length(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert has_missing(MISSING_VALUES_REGRESSION[key]["test"][0])
        assert np.issubdtype(
            MISSING_VALUES_REGRESSION[key]["test"][1].dtype, np.integer
        ) or np.issubdtype(MISSING_VALUES_REGRESSION[key]["test"][1].dtype, np.floating)


# todo series testing data

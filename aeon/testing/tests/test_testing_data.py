"""Tests for testing data dictionaries."""

import numpy as np

from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE,
    EQUAL_LENGTH_UNIVARIATE,
    TEST_DATA_DICT,
    TEST_LABEL_DICT,
    UNEQUAL_LENGTH_MULTIVARIATE,
    UNEQUAL_LENGTH_UNIVARIATE,
)
from aeon.utils.validation import (
    is_collection,
    is_equal_length,
    is_single_series,
    is_univariate,
)


def test_test_data_dict():
    """Test the contents of the test data dictionary."""
    for key in TEST_DATA_DICT:
        assert isinstance(TEST_DATA_DICT[key], dict)
        assert len(TEST_DATA_DICT[key]) == 2
        assert "train" in TEST_DATA_DICT[key]
        assert "test" in TEST_DATA_DICT[key]
        assert is_collection(TEST_DATA_DICT[key]["train"]) or is_single_series(
            TEST_DATA_DICT[key]["train"]
        )
        assert is_collection(TEST_DATA_DICT[key]["test"]) or is_single_series(
            TEST_DATA_DICT[key]["train"]
        )


def test_test_label_dict():
    """Test the contents of the test label dictionary."""
    for key in TEST_LABEL_DICT:
        assert isinstance(TEST_LABEL_DICT[key], dict)
        assert len(TEST_LABEL_DICT[key]) == 2
        assert "train" in TEST_LABEL_DICT[key]
        assert "test" in TEST_LABEL_DICT[key]
        if TEST_LABEL_DICT[key]["train"] is not None:
            assert isinstance(TEST_LABEL_DICT[key]["train"], np.ndarray)
            assert isinstance(TEST_LABEL_DICT[key]["test"], np.ndarray)
            assert TEST_LABEL_DICT[key]["train"].ndim == 1
            assert TEST_LABEL_DICT[key]["test"].ndim == 1


def test_equal_length_univariate():
    """Test the contents of the equal length univariate data dictionary."""
    for key in EQUAL_LENGTH_UNIVARIATE:
        assert is_collection(EQUAL_LENGTH_UNIVARIATE[key], include_2d=True)
        assert is_univariate(EQUAL_LENGTH_UNIVARIATE[key])
        assert is_equal_length(EQUAL_LENGTH_UNIVARIATE[key])


def test_unequal_length_univariate():
    """Test the contents of the unequal length univariate data dictionary."""
    for key in UNEQUAL_LENGTH_UNIVARIATE:
        assert is_collection(UNEQUAL_LENGTH_UNIVARIATE[key])
        assert is_univariate(UNEQUAL_LENGTH_UNIVARIATE[key])
        assert not is_equal_length(UNEQUAL_LENGTH_UNIVARIATE[key])


def test_equal_length_multivariate():
    """Test the contents of the equal length multivariate data dictionary."""
    for key in EQUAL_LENGTH_MULTIVARIATE:
        assert is_collection(EQUAL_LENGTH_MULTIVARIATE[key])
        assert not is_univariate(EQUAL_LENGTH_MULTIVARIATE[key])
        assert is_equal_length(EQUAL_LENGTH_MULTIVARIATE[key])


def test_unequal_length_multivariate():
    """Test the contents of the unequal length multivariate data dictionary."""
    for key in UNEQUAL_LENGTH_MULTIVARIATE:
        assert is_collection(UNEQUAL_LENGTH_MULTIVARIATE[key])
        assert not is_univariate(UNEQUAL_LENGTH_MULTIVARIATE[key])
        assert not is_equal_length(UNEQUAL_LENGTH_MULTIVARIATE[key])

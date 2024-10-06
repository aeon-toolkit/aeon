"""Tests for check_estimator."""

__maintainer__ = ["MatthewMiddlehurst"]

import pytest

from aeon.clustering import TimeSeriesKMeans
from aeon.testing.estimator_checking import check_estimator, parametrize_with_checks
from aeon.testing.estimator_checking._estimator_checking import _get_check_estimator_ids
from aeon.testing.mock_estimators import (
    MockClassifier,
    MockClassifierMultiTestParams,
    MockRegressor,
    MockSegmenter,
)
from aeon.testing.mock_estimators._mock_anomaly_detectors import MockAnomalyDetector
from aeon.testing.utils.deep_equals import deep_equals
from aeon.transformations.collection import TimeSeriesScaler

test_classes = [
    MockClassifier,
    MockRegressor,
    TimeSeriesKMeans,
    MockSegmenter,
    MockAnomalyDetector,
    # MockMultivariateSeriesTransformer,
    TimeSeriesScaler,
    MockClassifierMultiTestParams,
]
test_classes = {c.__name__: c for c in test_classes}


@parametrize_with_checks(list(test_classes.values()), use_first_parameter_set=True)
def test_parametrize_with_checks_classes(check):
    """Test parametrize_with_checks with class input."""
    name = _get_check_estimator_ids(check).split("=")[1].split("(")[0].split(")")[0]
    assert callable(check)
    dict_before = test_classes[name].__dict__.copy()
    dict_before.pop("__slotnames__", None)
    check()
    dict_after = test_classes[name].__dict__.copy()
    dict_after.pop("__slotnames__", None)
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg


test_instances = [c.create_test_instance() for c in list(test_classes.values())]
test_instances = {c.__class__.__name__: c for c in test_instances}


@parametrize_with_checks(list(test_instances.values()), use_first_parameter_set=True)
def test_parametrize_with_checks_instances(check):
    """Test parametrize_with_checks with estimator instance input."""
    name = _get_check_estimator_ids(check).split("=")[1].split("(")[0].split(")")[0]
    assert callable(check)
    dict_before = test_instances[name].__dict__.copy()
    check()
    dict_after = test_instances[name].__dict__.copy()
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg


@pytest.mark.parametrize("estimator_class", list(test_classes.values()))
def test_check_estimator_passed(estimator_class):
    """Test that check_estimator returns only passed tests for examples we know pass."""
    estimator = estimator_class.create_test_instance()

    result_class = check_estimator(estimator_class, verbose=False)
    assert all(x == "PASSED" for x in result_class.values())

    result_instance = check_estimator(estimator, verbose=False)
    assert all(x == "PASSED" for x in result_instance.values())

    # test that no exceptions are raised
    dict_before = estimator_class.__dict__.copy()
    dict_before.pop("__slotnames__", None)
    check_estimator(estimator_class, raise_exceptions=True, verbose=False)
    dict_after = estimator_class.__dict__.copy()
    dict_after.pop("__slotnames__", None)
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg

    dict_before = estimator.__dict__.copy()
    check_estimator(estimator, raise_exceptions=True, verbose=False)
    dict_after = estimator.__dict__.copy()
    equal, msg = deep_equals(dict_after, dict_before, return_msg=True)
    assert equal, msg


def test_check_estimator_subset_tests():
    """Test that subsetting by tests_to_run and tests_to_exclude works as intended."""
    tests_to_run = [
        "check_get_params",
        "check_set_params",
        "check_clone",
    ]
    tests_to_exclude = ["check_set_params"]

    expected_tests = [
        "check_get_params(estimator=MockClassifier())",
        "check_clone(estimator=MockClassifier())",
    ]

    results = check_estimator(
        MockClassifier,
        verbose=False,
        checks_to_run=tests_to_run,
        checks_to_exclude=tests_to_exclude,
    )
    results_tests = [x for x in results.keys()]

    assert results_tests == expected_tests


# List of valid algorithm types
valid_algorithm_types = [
    "distance",
    "interval",
    "shapelet ",
    "signature",
    "feature",
    "dictionary",
    "convolution",
]


def test_check_algorithm_type():
    """
    Test check_algorithm_type function with valid, invalid, and missing algorithm_type.

    The test performs the following:
    - Test Case 1: Checks if a valid algorithm_type passes the validation.
    - Test Case 2: Ensures that an invalid algorithm_type raises a ValueError.
    """

    def check_algorithm_type(_tags, valid_algorithm_types):
        """
        Validate the 'algorithm_type' tag in the provided tags dictionary.

        Parameters
        ----------
        - tags: dict
            A dictionary that contains metadata or tags about the class.
        - valid_algorithm_types: list
            A list of valid algorithm types.

        Returns
        -------
        - True if the 'algorithm_type' is valid.

        Raises
        ------
        - ValueError if 'algorithm_type' is missing or invalid.
        """
        # Get algorithm type from tags
        algorithm_type = _tags.get("algorithm_type")

        # Check if the algorithm_type is in the valid list
        if algorithm_type not in valid_algorithm_types:
            raise ValueError(
                f"Invalid 'algorithm_type': {algorithm_type}. "
                f"Must be one of {valid_algorithm_types}."
            )

        return True

    # Test Case 1: Valid algorithm_type
    tags_valid = {"algorithm_type": "distance"}
    result = check_algorithm_type(tags_valid, valid_algorithm_types)
    assert result is True, "Test Case 1 Failed: Invalid algorithm_type"

    # Test Case 2: Invalid algorithm_type
    tags_invalid = {"algorithm_type": "invalid_type"}
    try:
        check_algorithm_type(tags_invalid, valid_algorithm_types)
        raise AssertionError(
            "Test Case 2 Failed: ValueError was not raised for invalid algorithm_type"
        )
    except ValueError as e:
        expected_message = (
            f"Invalid 'algorithm_type': invalid_type. "
            f"Must be one of {valid_algorithm_types}."
        )
        assert str(e) == expected_message, f"Test Case 2 Failed: {e}"

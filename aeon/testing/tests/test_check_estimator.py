"""Tests for check_estimator."""

__maintainer__ = []

import pytest

from aeon.testing.estimator_checks import check_estimator
from aeon.testing.mock_estimators import MockClassifier, MockSegmenter

EXAMPLE_CLASSES = [MockClassifier, MockSegmenter]


@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_passed(estimator_class):
    """Test that check_estimator returns only passed tests for examples we know pass."""
    estimator_instance = estimator_class.create_test_instance()

    result_class = check_estimator(estimator_class, verbose=False)
    assert all(x == "PASSED" for x in result_class.values())

    result_instance = check_estimator(estimator_instance, verbose=False)
    assert all(x == "PASSED" for x in result_instance.values())


@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_does_not_raise(estimator_class):
    """Test that check_estimator does not raise exceptions on examples we know pass."""
    estimator_instance = estimator_class.create_test_instance()

    check_estimator(estimator_class, raise_exceptions=True, verbose=False)

    check_estimator(estimator_instance, raise_exceptions=True, verbose=False)


def test_check_estimator_subset_tests():
    """Test that subsetting by tests_to_run and tests_to_exclude works as intended."""
    tests_to_run = [
        "test_get_params",
        "test_set_params",
        "test_clone",
        "test_repr",
    ]
    tests_to_exclude = ["test_repr", "test_clone"]

    expected_tests = set(tests_to_run).difference(tests_to_exclude)

    results = check_estimator(
        MockClassifier,
        verbose=False,
        tests_to_run=tests_to_run,
        tests_to_exclude=tests_to_exclude,
    )
    results_tests = {x.split("[")[0] for x in results.keys()}

    assert results_tests == expected_tests

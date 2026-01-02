"""Test threading util decorator."""

import os
from unittest.mock import MagicMock, patch

import pytest

from aeon.utils.numba._threading import threaded


def check_n_jobs(n_jobs):
    """Mock implementation of check_n_jobs."""
    return n_jobs if n_jobs is not None else 1


def set_num_threads(n_threads):
    """Mock implementation of set_num_threads."""
    pass


@pytest.fixture
def clean_env():
    """Save and restore environment variables between tests."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_basic_functionality():
    """Test that the decorator correctly sets and restores thread count."""
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(n_jobs=None):
                return "executed"

            result = sample_func(n_jobs=4)

            assert result == "executed"
            check_jobs_mock.assert_called_once_with(4)
            assert set_threads_mock.call_count == 2


def test_numba_env_variable(clean_env):
    """Test that the decorator respects NUMBA_NUM_THREADS environment variable."""
    os.environ["NUMBA_NUM_THREADS"] = "8"

    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(n_jobs=None):
                return "executed"

            sample_func(n_jobs=4)

            assert set_threads_mock.call_args_list[0][0][0] == 4
            assert set_threads_mock.call_args_list[1][0][0] == 8


def test_fallback_to_threading_count(clean_env):
    """Test the fallback mechanism to CPU count/affinity."""
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    # Mock the new fallback mechanism
    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):
            with patch(
                "aeon.utils.numba._threading._num_threads_default", return_value=3
            ):

                @threaded
                def sample_func(n_jobs=None):
                    return "executed"

                sample_func(n_jobs=4)

                assert set_threads_mock.call_args_list[1][0][0] == 3


def test_positional_argument():
    """
    Test the extraction of n_jobs when passed as a positional argument.

    The threaded decorator needs to correctly identify the n_jobs parameter
    regardless of how it's passed to the function. This test verifies that
    when n_jobs is passed as a positional argument, the decorator correctly
    extracts its value and uses it to configure the thread count.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(data, n_jobs=None):
                return data

            sample_func("test_data", 4)

            check_jobs_mock.assert_called_once_with(4)


def test_keyword_argument():
    """
    Test the extraction of n_jobs when passed as a keyword argument.

    Functions decorated with the threaded decorator can receive the n_jobs
    parameter as a keyword argument. This test ensures that the decorator
    correctly identifies and extracts the n_jobs value when passed this way,
    demonstrating the decorator's flexibility in handling different calling styles.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(data, n_jobs=None):
                return data

            sample_func(data="test_data", n_jobs=4)

            check_jobs_mock.assert_called_once_with(4)


def test_default_value():
    """
    Test the use of default n_jobs value when not explicitly provided.

    When a function has a default value for the n_jobs parameter and is called
    without specifying this parameter, the threaded decorator should use the
    function's default value. This test verifies this behavior, ensuring that
    default function parameters are properly respected by the decorator.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(data, n_jobs=2):
                return data

            sample_func("test_data")

            check_jobs_mock.assert_called_once_with(2)


def test_exception_handling():
    """
    Test resource cleanup when exceptions occur in the decorated function.

    A robust decorator must ensure resources are properly managed even when
    the decorated function raises an exception. This test verifies that the
    threaded decorator correctly restores the original thread count even when
    the function execution fails with an exception, preventing resource leaks.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(n_jobs=None):
                raise ValueError("Test exception")

            with pytest.raises(ValueError, match="Test exception"):
                sample_func(n_jobs=4)

            assert set_threads_mock.call_count == 2


def test_class_attribute():
    """
    Test the extraction of n_jobs from a class attribute.

    The threaded decorator should be able to extract the n_jobs value from
    the first argument (typically 'self' in class methods) when it has an
    n_jobs attribute. This test verifies that the decorator correctly identifies
    and uses this attribute when the n_jobs parameter is not explicitly passed.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            class TestClass:
                def __init__(self, n_jobs):
                    self.n_jobs = n_jobs

                @threaded
                def process_data(self, data):
                    return data

            test_instance = TestClass(n_jobs=5)

            test_instance.process_data("test_data")

            check_jobs_mock.assert_called_once_with(5)
            assert set_threads_mock.call_count == 2


def test_parameter_precedence_over_attribute():
    """
    Test that n_jobs parameter takes precedence over class attribute.

    When both a class attribute and a method parameter for n_jobs exist,
    the parameter value should take precedence. This test verifies this
    precedence rule, ensuring that explicit parameter values override
    attribute values.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            class TestClass:
                def __init__(self, n_jobs):
                    self.n_jobs = n_jobs

                @threaded
                def process_data(self, data, n_jobs=None):
                    return data

            test_instance = TestClass(n_jobs=5)

            test_instance.process_data("test_data", n_jobs=7)

            check_jobs_mock.assert_called_once_with(7)
            assert set_threads_mock.call_count == 2


def test_fallback_when_no_attribute():
    """
    Test fallback behavior when neither parameter nor attribute is available.

    When a class doesn't have an n_jobs attribute and the method doesn't
    have an n_jobs parameter, the decorator should fall back to using None,
    which will be converted to 1 by check_n_jobs. This test verifies this
    fallback behavior.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()

    with patch("aeon.utils.numba._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils.numba._threading.set_num_threads", set_threads_mock):

            class TestClass:
                # No n_jobs attribute
                pass

                @threaded
                def process_data(self, data):
                    return data

            test_instance = TestClass()

            test_instance.process_data("test_data")

            check_jobs_mock.assert_called_once_with(None)
            assert set_threads_mock.call_count == 2

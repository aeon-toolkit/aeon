"""Test threading decorator."""

import os
from unittest.mock import MagicMock, patch

import pytest

from aeon.utils._threading import threaded


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

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):

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

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(n_jobs=None):
                return "executed"

            sample_func(n_jobs=4)

            assert set_threads_mock.call_args_list[0][0][0] == 4
            assert set_threads_mock.call_args_list[1][0][0] == 8


def test_fallback_to_threading_count(clean_env):
    """
    Test the fallback mechanism to the system's active thread count.

    When the NUMBA_NUM_THREADS environment variable is not set or is invalid,
    the decorator should use the system's active thread count as the baseline.
    This ensures proper thread management even when no explicit configuration is
    provided.
    """
    check_jobs_mock = MagicMock(side_effect=lambda x: x if x is not None else 1)
    set_threads_mock = MagicMock()
    thread_count_mock = MagicMock(return_value=3)

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):
            with patch("threading.active_count", thread_count_mock):

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

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):

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

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):

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

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):

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

    with patch("aeon.utils._threading.check_n_jobs", check_jobs_mock):
        with patch("aeon.utils._threading.set_num_threads", set_threads_mock):

            @threaded
            def sample_func(n_jobs=None):
                raise ValueError("Test exception")

            with pytest.raises(ValueError, match="Test exception"):
                sample_func(n_jobs=4)

            assert set_threads_mock.call_count == 2

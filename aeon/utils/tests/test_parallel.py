"""Tests for internal parallel helpers."""

from joblib import delayed
from numba import config, get_num_threads

from aeon.utils._parallel import _numba_threads, _run_jobs


def _add(left, right):
    return left + right


def test_run_jobs_skips_joblib_for_single_job(monkeypatch):
    """Single-job execution should call delayed tasks directly."""

    def fail_parallel(*args, **kwargs):
        raise AssertionError("Parallel should not be constructed for n_jobs=1")

    monkeypatch.setattr("aeon.utils._parallel.Parallel", fail_parallel)
    tasks = (delayed(_add)(value, 1) for value in range(3))

    assert _run_jobs(tasks, n_jobs=1, prefer="threads") == [1, 2, 3]


def test_numba_threads_caps_and_restores():
    """n_jobs above numba's thread pool is capped and the old count restored."""
    prev_threads = get_num_threads()

    with _numba_threads(config.NUMBA_NUM_THREADS + 1):
        assert get_num_threads() == config.NUMBA_NUM_THREADS

    assert get_num_threads() == prev_threads

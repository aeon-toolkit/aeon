"""Tests for internal parallel helpers."""

from joblib import delayed

from aeon.utils._parallel import _run_jobs


def _add(left, right):
    return left + right


def test_run_jobs_skips_joblib_for_single_job(monkeypatch):
    """Single-job execution should call delayed tasks directly."""

    def fail_parallel(*args, **kwargs):
        raise AssertionError("Parallel should not be constructed for n_jobs=1")

    monkeypatch.setattr("aeon.utils._parallel.Parallel", fail_parallel)
    tasks = (delayed(_add)(value, 1) for value in range(3))

    assert _run_jobs(tasks, n_jobs=1, prefer="threads") == [1, 2, 3]

"""Test all estimators in aeon."""

import platform
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.utils import check_random_state

from aeon.testing.estimator_checking import parametrize_with_checks
from aeon.testing.testing_config import PR_TESTING
from aeon.utils.discovery import all_estimators

ALL_TEST_ESTIMATORS = all_estimators(return_names=False, include_sklearn=False)


def _get_pr_subsample_index(python_minor, os_str):
    """Get the index of the estimator subsample to test in a PR run.

    Map the Python versions used in the PR pytest matrix to distinct indices. Other
    versions keep their minor version, i.e. 3.11, which only runs with PR testing in
    the soft dependency skip job and is not part of the matrix rotation.
    """
    i = python_minor
    if i == 12:
        i = 0
    elif i == 13:
        i = 1
    elif i == 14:
        i = 2

    if os_str == "Linux":
        i = i + 1
    elif os_str == "Darwin":
        i = i + 2

    return i % 3


# subsample estimators by OS & python version
# this ensures that only 1/3 of estimators are tested for a given combination
# but across the PR pytest matrix all are tested on every OS and python version once
if PR_TESTING:
    i = _get_pr_subsample_index(sys.version_info.minor, platform.system())

    rng = check_random_state(42)
    idx = np.arange(len(ALL_TEST_ESTIMATORS))
    rng.shuffle(idx)

    ALL_TEST_ESTIMATORS = [ALL_TEST_ESTIMATORS[n] for n in idx[i::3]]


@parametrize_with_checks(ALL_TEST_ESTIMATORS)
def test_all_estimators(check):
    """Run general estimator checks on all aeon estimators."""
    check()


def test_pr_subsample_covers_pr_pytest_matrix():
    """Test that PR runs test all estimators on each OS and Python version.

    Reads the pytest job matrix from the PR workflow, so this fails if the workflow
    Python versions or OS change without updating _get_pr_subsample_index.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    workflow = repo_root / ".github" / "workflows" / "pr_pytest.yml"

    assert workflow.exists(), f"PR pytest workflow not found at {workflow}."

    with open(workflow, encoding="utf-8") as f:
        workflow_config = yaml.safe_load(f)

    try:
        matrix = workflow_config["jobs"]["pytest"]["strategy"]["matrix"]
        runners = matrix["os"]
        versions = matrix["python-version"]
    except (KeyError, TypeError) as e:
        raise AssertionError(
            f"Could not find the pytest job OS and Python version matrix in "
            f"{workflow.name}."
        ) from e

    # matrix entries which are removed when running with PR testing
    pr_excludes = [e for e in matrix.get("exclude", []) if e.get("pr-testing") is True]

    runner_systems = {"ubuntu": "Linux", "macos": "Darwin", "windows": "Windows"}
    os_indices = {}
    version_indices = {}
    for runner in runners:
        os_str = [s for r, s in runner_systems.items() if runner.lower().startswith(r)]
        assert len(os_str) == 1, f"Unknown OS for runner {runner} in {workflow.name}."

        for version in versions:
            if any(
                e.get("os", runner) == runner
                and e.get("python-version", version) == version
                for e in pr_excludes
            ):
                continue

            i = _get_pr_subsample_index(int(str(version).split(".")[1]), os_str[0])
            os_indices.setdefault(runner, set()).add(i)
            version_indices.setdefault(version, set()).add(i)

    for name, indices in list(os_indices.items()) + list(version_indices.items()):
        assert indices == {0, 1, 2}, (
            f"PR runs for {name} in {workflow.name} only test estimator subsamples "
            f"{sorted(indices)}, update _get_pr_subsample_index so that every "
            f"subsample is tested on each OS and Python version."
        )

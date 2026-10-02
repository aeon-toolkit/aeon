"""Tests for aeon CI workflow configuration."""

from pathlib import Path

import pytest
import yaml

from aeon.testing.testing_config import _get_pr_subsample_index

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_pr_subsample_covers_pr_pytest_matrix():
    """Test that PR runs test all estimators on each OS and Python version.

    Reads the pytest job matrix from the PR workflow, so this fails if the workflow
    OS or Python versions change without updating _get_pr_subsample_index.
    """
    # workflow files are not shipped with the package, only test a repository checkout
    if not (REPO_ROOT / ".github").exists():
        pytest.skip("Tests are not being run from a repository checkout.")

    workflow = REPO_ROOT / ".github" / "workflows" / "pr_pytest.yml"
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

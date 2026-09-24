"""Test all estimators in aeon."""

import platform
import sys

import numpy as np
from sklearn.utils import check_random_state

from aeon.testing.estimator_checking import parametrize_with_checks
from aeon.testing.testing_config import PR_TESTING, _get_pr_subsample_index
from aeon.utils.discovery import all_estimators

ALL_TEST_ESTIMATORS = all_estimators(return_names=False, include_sklearn=False)

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

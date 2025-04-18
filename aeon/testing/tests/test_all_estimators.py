"""Test all estimators in aeon."""

import platform
import sys

import numpy as np
from sklearn.utils import check_random_state

from aeon.testing.estimator_checking import parametrize_with_checks
from aeon.testing.testing_config import PR_TESTING
from aeon.utils.discovery import all_estimators

ALL_TEST_ESTIMATORS = all_estimators(return_names=False, include_sklearn=False)

# subsample estimators by OS & python version
# this ensures that only a 1/3 of estimators are tested for a given combination
# but all are tested on every OS at least once, and on every python version once
if PR_TESTING:
    # only use 3 Python versions in PR
    i = sys.version_info.minor
    if i == 9:
        i = 0
    elif i == 11:
        i = 1
    elif i == 13:
        i = 2

    os_str = platform.system()
    if os_str == "Windows":
        i = i
    elif os_str == "Linux":
        i = i + 1
    elif os_str == "Darwin":
        i = i + 2

    i = i % 3

    rng = check_random_state(42)
    idx = np.arange(len(ALL_TEST_ESTIMATORS))
    rng.shuffle(idx)

    ALL_TEST_ESTIMATORS = [ALL_TEST_ESTIMATORS[n] for n in idx[i::3]]


@parametrize_with_checks(ALL_TEST_ESTIMATORS)
def test_all_estimators(check):
    """Run general estimator checks on all aeon estimators."""
    check()

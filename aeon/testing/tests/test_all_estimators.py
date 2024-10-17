"""Test all estimators in aeon."""

import platform
import sys

from aeon.testing.estimator_checking import parametrize_with_checks
from aeon.testing.testing_config import PR_TESTING
from aeon.utils.discovery import all_estimators
from aeon.utils.sampling import random_partition

ALL_TEST_ESTIMATORS = all_estimators(return_names=False, include_sklearn=False)

# subsample estimators by OS & python version
# this ensures that only a 1/3 of estimators are tested for a given combination
# but all are tested on every OS at least once, and on every python version once
if PR_TESTING:
    # only use 3 Python versions in PR
    ix = sys.version_info.minor
    if ix == 9:
        ix = 0
    elif ix == 11:
        ix = 1
    elif ix == 12:
        ix = 2

    os_str = platform.system()
    if os_str == "Windows":
        ix = ix
    elif os_str == "Linux":
        ix = ix + 1
    elif os_str == "Darwin":
        ix = ix + 2

    ix = ix % 3

    ALL_TEST_ESTIMATORS = [
        ALL_TEST_ESTIMATORS[i]
        for i in random_partition(len(ALL_TEST_ESTIMATORS), 3)[ix]
    ]


@parametrize_with_checks(ALL_TEST_ESTIMATORS)
def test_all_estimators(check):
    """Run general estimator checks on all aeon estimators."""
    check()

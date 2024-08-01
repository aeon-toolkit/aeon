"""Test all estimators in aeon."""

import platform
import sys

from aeon.registry import all_estimators
from aeon.testing.estimator_checking import parametrize_with_checks
from aeon.testing.test_config import PR_TESTING
from aeon.utils.sampling import random_partition

ALL_ESTIMATORS = all_estimators(
    estimator_types=["classifier", "regressor"],
    return_names=False,
)

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

    ALL_ESTIMATORS = [
        ALL_ESTIMATORS[i] for i in random_partition(len(ALL_ESTIMATORS), 3)[ix]
    ]


@parametrize_with_checks(ALL_ESTIMATORS)
def test_all_estimators(check):
    """Run general estimator checks on all aeon estimators."""
    check()

"""Test all estimators in aeon."""

from aeon.registry import all_estimators
from aeon.testing.estimator_checking import parametrize_with_checks

ALL_ESTIMATORS = all_estimators(
    estimator_types=["classifier"],
    return_names=False,
)


@parametrize_with_checks(ALL_ESTIMATORS)
def test_all_estimators(estimator, check):
    """Run general estimator checks on all aeon estimators."""
    check(estimator)

"""Estimator checks."""

__all__ = [
    "check_estimator",
    "parametrize_with_checks",
    "check_estimator_legacy",
]

from aeon.testing.estimator_checking._estimator_checks import (
    check_estimator,
    parametrize_with_checks,
)
from aeon.testing.estimator_checking._legacy._legacy_estimator_checks import (
    check_estimator_legacy,
)

"""Unit tests for sklearn classifiers."""

__maintainer__ = []

from sklearn.utils.estimator_checks import parametrize_with_checks

from aeon.classification.sklearn import ContinuousIntervalTree, RotationForestClassifier


@parametrize_with_checks(
    [RotationForestClassifier(n_estimators=3), ContinuousIntervalTree()]
)
def test_sklearn_compatible_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    try:
        check(estimator)
    except AssertionError as error:
        # ContinuousIntervalTree can handle NaN values
        if not isinstance(
            estimator, ContinuousIntervalTree
        ) or "check for NaN and inf" not in str(error):
            raise error

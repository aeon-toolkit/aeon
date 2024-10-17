"""Test the RISE classifier."""

from aeon.classification.interval_based import RandomIntervalSpectralEnsembleClassifier
from aeon.classification.sklearn import ContinuousIntervalTree


def test_with_nan():
    """Test nans correct with ContinuousIntervalTree."""
    r = RandomIntervalSpectralEnsembleClassifier(
        base_estimator=ContinuousIntervalTree()
    )
    assert r.replace_nan == "nan"

"""Test the CIF classifier."""

from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.sklearn import ContinuousIntervalTree


def test_cif():
    """Test nans correct with ContinuousIntervalTree."""
    tsf = TimeSeriesForestClassifier(base_estimator=ContinuousIntervalTree())
    assert tsf.replace_nan == "nan"

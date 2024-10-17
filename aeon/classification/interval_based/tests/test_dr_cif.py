"""Test the DrCIF classifier."""

from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.sklearn import ContinuousIntervalTree


def test_dr_cif():
    """Test nans correct with ContinuousIntervalTree."""
    cif = DrCIFClassifier(base_estimator=ContinuousIntervalTree(), use_pycatch22=True)
    assert cif.replace_nan == "nan"
    d = cif.get_tag("python_dependencies")
    assert d == ["pycatch22"]

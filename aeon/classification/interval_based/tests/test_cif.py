"""Test the CIF classifier."""

from aeon.classification.interval_based import CanonicalIntervalForestClassifier
from aeon.classification.sklearn import ContinuousIntervalTree

#todo fix these interval tests/classes
asda def test_cif():
    """Test nans correct with ContinuousIntervalTree."""
    cif = CanonicalIntervalForestClassifier(
        base_estimator=ContinuousIntervalTree(), use_pycatch22=True
    )
    assert cif.replace_nan == "nan"
    d = cif.get_tag("python_dependencies")
    assert d == "pycatch22"

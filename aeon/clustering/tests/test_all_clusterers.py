"""Test all clusterers comply to interface."""
import numpy as np
import pytest

from aeon.registry import all_estimators
from aeon.tests.test_softdeps import soft_deps_installed

ALL_CLUSTERERS = all_estimators("clusterer", return_names=False)


@pytest.mark.parametrize("clst", ALL_CLUSTERERS)
def test_interface_compliance(clst):
    """Test does not override final methods."""
    assert "fit" not in clst.__dict__
    assert "predict" not in clst.__dict__


@pytest.mark.parametrize("clst", ALL_CLUSTERERS)
def test_capability_tags(clst):
    """Test all estimators capability tags reflect their capabilities."""
    if not soft_deps_installed(clst):
        return
    # Test the tag X_inner_type is consistent with capability:unequal_length
    unequal_length = clst.get_class_tag("capability:unequal_length")
    valid_types = {"np-list", "df-list", "pd-multivariate", "nested_univ"}
    if unequal_length:  # one of X_inner_types must be capable of storing unequal length
        internal_types = clst.get_class_tag("X_inner_type")
        if isinstance(internal_types, str):
            assert internal_types in valid_types
        else:  # must be a list
            assert bool(set(internal_types) & valid_types)

    # Test can actually fit/predict with multivariate if tag is set
    multivariate = clst.get_class_tag("capability:multivariate")
    X = np.random.random((10, 2, 20))
    inst = clst.create_test_instance(parameter_set="default")

    if multivariate:
        inst.fit(X)
        inst.predict(X)
        inst.predict_proba(X)

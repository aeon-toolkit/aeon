"""Tests for _HeterogenousMetaEstimator."""

import pytest

from aeon.base._meta import _HeterogenousMetaEstimator
from aeon.classification import DummyClassifier
from aeon.classification.compose._channel_ensemble import ChannelEnsembleClassifier


def test_hetero_meta():
    """Test _HeterogenousMetaEstimator."""
    h = _HeterogenousMetaEstimator()
    assert h.is_composite()
    with pytest.raises(ValueError, match="Names provided are not unique"):
        h._check_names(["FOO", "FOO"])
    bce = ChannelEnsembleClassifier(estimators=[("Dummy", DummyClassifier(), 0)])
    with pytest.raises(ValueError, match="Estimator names must not contain"):
        bce._check_names(["__FOO"])
    names = ["FOO", "estimators"]
    with pytest.raises(ValueError, match="Estimator names conflict with constructor"):
        bce._check_names(names)
    names = ["DummyClassifier"]
    bce._check_names(names)
    assert not h._is_name_and_est("Single")
    assert not h._is_name_and_est(("Single", "Tuple"))
    with pytest.raises(TypeError, match="must be of type BaseEstimator"):
        h._check_estimators(estimators="FOO")
        h._check_estimators(estimators=None)
    with pytest.raises(TypeError, match="cls_type must be a class"):
        h._check_estimators(estimators="FOO", cls_type="BAR")
    x = h._coerce_estimator_tuple(obj=bce, clone_est=True)
    assert isinstance(x, tuple)
    assert isinstance(x[0], str)
    assert isinstance(x[1], ChannelEnsembleClassifier)
    list = h._make_strings_unique([("49", "49")])
    assert list[0][0] != list[0][1]
    list = h._make_strings_unique(("49", "49"))
    assert list[0][0] != list[0][1]
    with pytest.raises(TypeError, match="concat_order must be str"):
        h._dunder_concat(
            other=None, base_class=None, composite_class=None, concat_order=49
        )
    with pytest.raises(ValueError, match="concat_order must be one of"):
        h._dunder_concat(
            other=None, base_class=None, composite_class=None, concat_order="up"
        )
    with pytest.raises(TypeError, match="attr_name must be str"):
        h._dunder_concat(
            other=None, base_class=None, composite_class=None, attr_name=49
        )
    with pytest.raises(TypeError, match="composite_class must be a class"):
        h._dunder_concat(other=None, base_class=None, composite_class=None)
    with pytest.raises(TypeError, match="base_class must be a class"):
        h._dunder_concat(
            other=None, base_class=None, composite_class=ChannelEnsembleClassifier
        )
    with pytest.raises(TypeError, match="self must be an instance of composite_class"):
        _HeterogenousMetaEstimator._dunder_concat(
            str,
            other=None,
            base_class=_HeterogenousMetaEstimator,
            composite_class=ChannelEnsembleClassifier,
        )

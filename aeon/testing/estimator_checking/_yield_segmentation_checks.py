"""Tests for all segmenters."""

from functools import partial

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.segmentation import SegmenterScoresMixin
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES


def _yield_segmentation_checks(estimator_class, estimator_instances, datatypes):
    """Yield all segmentation checks for an aeon segmenter."""
    # only class required
    yield partial(check_segmenter_base_functionality, estimator_class=estimator_class)

    # test class instances
    for _, estimator in enumerate(estimator_instances):
        # no data needed
        yield partial(check_segmenter_instance, estimator=estimator)


def check_segmenter_base_functionality(estimator_class):
    """Test compliance with the base class contract."""
    # Test they dont override final methods, because python does not enforce this
    assert "fit" not in estimator_class.__dict__
    assert "predict" not in estimator_class.__dict__
    assert "fit_predict" not in estimator_class.__dict__
    # Test that all segmenters implement abstract predict.
    assert "_predict" in estimator_class.__dict__
    # Test that fit_is_empty is correctly set
    fit_is_empty = estimator_class.get_class_tag(tag_name="fit_is_empty")
    assert not fit_is_empty == "_fit" not in estimator_class.__dict__
    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    assert X_inner_type in VALID_SERIES_INNER_TYPES
    # Must have at least one set to True
    multi = estimator_class.get_class_tag(tag_name="capability:multivariate")
    uni = estimator_class.get_class_tag(tag_name="capability:univariate")
    assert multi or uni
    # Test that segmenters with predict_scores implement the private method and do
    # not override the final public one. Gated on the mixin until ClaSPSegmenter and
    # FLUSSSegmenter are migrated (#3852, #3853), after which the predict_scores
    # override assertion becomes unconditional.
    if issubclass(estimator_class, SegmenterScoresMixin):
        assert "predict_scores" not in estimator_class.__dict__
        assert "_predict_scores" in estimator_class.__dict__


def check_segmenter_instance(estimator):
    """Test segmenters."""
    import pytest

    estimator = _clone_estimator(estimator)

    def _assert_output(output, dense, length):
        """Assert the properties of the segmenter output."""
        assert isinstance(output, np.ndarray)
        if dense:  # Change points returned
            assert len(output) < length
            assert max(output) < length
            assert min(output) >= 0
            # Test in ascending order
            assert all(output[i] <= output[i + 1] for i in range(len(output) - 1))
        else:  # Segment labels returned, must be same length sas series
            assert len(output) == length

    def _assert_predict_scores(X, length, axis=1):
        """Assert predict_scores returns one float score per time point."""
        if isinstance(estimator, SegmenterScoresMixin):
            scores = estimator.predict_scores(X, axis=axis)
            assert isinstance(scores, np.ndarray)
            assert scores.shape == (length,)
            assert np.issubdtype(scores.dtype, np.floating)

    multivariate = estimator.get_tag(tag_name="capability:multivariate")
    X = np.random.random(size=(5, 20))
    # Also tests does not fail if y is passed
    y = np.array([0, 0, 0, 1, 1])
    # Test that capability:multivariate is correctly set
    dense = estimator.get_tag(tag_name="returns_dense")
    if multivariate:
        output = estimator.fit_predict(X, y, axis=1)
        _assert_output(output, dense, X.shape[1])
        _assert_predict_scores(X, X.shape[1], axis=1)
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            estimator.fit_predict(X, y, axis=1)
    # Test that output is correct type
    X = np.random.random(size=(20))
    uni = estimator.get_tag(tag_name="capability:univariate")
    if uni:
        output = estimator.fit_predict(X, y=X)
        _assert_output(output, dense, len(X))
        _assert_predict_scores(X, len(X))
    else:
        with pytest.raises(ValueError, match="Univariate data not supported"):
            estimator.fit_predict(X)

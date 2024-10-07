"""Tests for all collection transformers."""

from functools import partial

import numpy as np

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector


def _yield_collection_transformation_checks(
    estimator_class, estimator_instances, datatypes
):
    """Yield all collection transformer checks for an aeon estimator."""
    # only class required
    yield partial(
        check_does_not_override_final_methods, estimator_class=estimator_class
    )

    if issubclass(estimator_class, BaseChannelSelector):
        yield partial(check_channel_selectors, estimator_class=estimator_class)


def check_does_not_override_final_methods(estimator_class):
    """Test does not override final methods."""
    assert "fit" not in estimator_class.__dict__
    assert "transform" not in estimator_class.__dict__
    assert "fit_transform" not in estimator_class.__dict__


def check_channel_selectors(estimator_class):
    """Test channel selectors.

    Needs fit and must select at least one channel
    """
    X, _ = make_example_3d_numpy(n_cases=20, n_channels=6, n_timepoints=30)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    cs = estimator_class.create_test_instance(return_first=True)
    assert not cs.get_tag("fit_is_empty")
    cs.fit(X, y)
    assert cs.channels_selected_ is not None
    assert len(cs.channels_selected_) > 0
    X2 = cs.transform(X)
    assert isinstance(X2, np.ndarray)
    assert X2.ndim == 3

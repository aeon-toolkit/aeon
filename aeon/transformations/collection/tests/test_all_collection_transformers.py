"""Test all BaseCollection transformers comply to interface."""

import numpy as np
import pytest

from aeon.registry import all_estimators
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

ALL_COLL_TRANS = all_estimators("collection-transformer", return_names=False)


@pytest.mark.parametrize("trans", ALL_COLL_TRANS)
def test_does_not_override_final_methods(trans):
    """Test does not override final methods."""
    assert "fit" not in trans.__dict__
    assert "transform" not in trans.__dict__
    assert "fit_transform" not in trans.__dict__


@pytest.mark.parametrize("trans", ALL_COLL_TRANS)
def test_channel_selectors(trans):
    """Test channel selectors."""
    if issubclass(trans, BaseChannelSelector):
        # Need fit for channel selection
        # Must select at least one channel
        X, _ = make_example_3d_numpy(n_cases=20, n_channels=6, n_timepoints=30)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        cs = trans()
        assert not cs.get_tag("fit_is_empty")
        cs.fit(X, y)
        assert cs.channels_selected_ is not None
        assert len(cs.channels_selected_) > 0
        X2 = cs.transform(X)
        assert isinstance(X2, np.ndarray)
        assert X2.ndim == 3

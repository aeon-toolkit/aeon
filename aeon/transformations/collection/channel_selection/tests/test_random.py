"""Test RandomChannelSelector."""

import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.channel_selection._random import (
    RandomChannelSelector,
)


def test_random_channel_selector():
    """Test random channel selection."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=10, n_timepoints=20, return_y=False
    )
    # Standard case, select 4
    r = RandomChannelSelector()
    X2 = r.fit_transform(X)
    assert X2.shape == (10, 4, 20)
    # Should round up to number of channels
    r = RandomChannelSelector(p=0.55)
    X2 = r.fit_transform(X)
    assert X2.shape == (10, 6, 20)
    r = RandomChannelSelector(p=0.91)
    X2 = r.fit_transform(X)
    assert X2.shape == X.shape
    r = RandomChannelSelector(p=1.0)
    X2 = r.fit_transform(X)
    assert X2.shape == X.shape
    with pytest.raises(
        ValueError, match="Proportion of channels to select should be in the range."
    ):
        RandomChannelSelector(p=0)

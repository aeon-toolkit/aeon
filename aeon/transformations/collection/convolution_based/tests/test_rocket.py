"""Test the Rocket transformer."""

import pytest

from aeon.testing.data_generation import make_example_3d_numpy_list
from aeon.transformations.collection.convolution_based import Rocket


def test_unequal_length():
    """Test length shorter in transform."""
    X = make_example_3d_numpy_list(
        return_y=False, min_n_timepoints=50, max_n_timepoints=100
    )
    X2 = make_example_3d_numpy_list(
        return_y=False, min_n_timepoints=10, max_n_timepoints=50
    )
    r = Rocket(n_kernels=100)
    r.fit(X)
    with pytest.raises(ValueError, match="Min length in transform = "):
        r.transform(X2)

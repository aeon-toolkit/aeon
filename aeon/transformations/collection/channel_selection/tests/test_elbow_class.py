"""Channel selection test code."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.channel_selection._elbow_class import (
    ElbowClassPairwise,
    _ClassPrototype,
    _create_distance_matrix,
)


def test_elbow_class():
    """Test channel selection on random nested data frame."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=4, n_timepoints=20, n_labels=3)
    ecp = ElbowClassPairwise()
    ecp.fit(X, y)
    Xt = ecp.transform(X, y)
    # test shape of transformed data should be
    # (n_samples, n_channels_selected, n_timepoints)
    assert Xt.shape == (X.shape[0], len(ecp.channels_selected_), X.shape[2])


def test_create_distance_matrix():
    """Test distance matrix creation."""
    with pytest.raises(ValueError, match="must be of same length"):
        _create_distance_matrix(np.array([1, 2, 3]), np.array([0, 1]))
    X = np.array([[[1, 2, 3]], [[1, 2, 3]]])
    res = _create_distance_matrix(X, np.array([0, 1]), distance="squared")
    assert res.shape == (1, 1)


def test_prototype():
    """Test function in _ClassPrototype."""
    p = _ClassPrototype()
    X, y = make_example_3d_numpy(n_cases=3, n_channels=3, n_timepoints=20, n_labels=3)
    r = p._create_median_prototype(X, y)
    assert r.shape == (X.shape[0] * X.shape[1], X.shape[2])

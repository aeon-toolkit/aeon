"""Channel selection test code."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.channel_selection._elbow_class import (
    ElbowClassPairwise,
    ElbowClassSum,
    _ClassPrototype,
    _create_distance_matrix,
    _detect_knee_point,
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


def test_create_distance_matrix_uses_requested_distance():
    """Test distance matrix uses the requested non-Euclidean distance."""
    X = np.array(
        [
            [[0, 0, 0], [1, 2, 3]],
            [[0, 1, 2], [1, 3, 5]],
        ]
    )

    dtw_res = _create_distance_matrix(X, np.array([0, 1]), distance="dtw")
    manhattan_res = _create_distance_matrix(X, np.array([0, 1]), distance="manhattan")

    assert not np.array_equal(dtw_res.to_numpy(), manhattan_res.to_numpy())


def test_create_distance_matrix_rejects_invalid_distance():
    """Test invalid distance strings are rejected."""
    X = np.array([[[1, 2, 3]], [[1, 2, 3]]])

    with pytest.raises(ValueError):
        _create_distance_matrix(X, np.array([0, 1]), distance="not_a_distance")


def test_prototype():
    """Test function in _ClassPrototype."""
    p = _ClassPrototype()
    X, y = make_example_3d_numpy(n_cases=3, n_channels=3, n_timepoints=20, n_labels=3)
    r = p._create_median_prototype(X, y)
    assert r.shape == (X.shape[0] * X.shape[1], X.shape[2])


def test_prototype_invalid_type_raises():
    """Test an unsupported prototype type raises a ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        _ClassPrototype(prototype_type="bogus")


def test_prototype_mad():
    """Test mad class prototype creation."""
    X, y = make_example_3d_numpy(n_cases=12, n_channels=2, n_timepoints=10, n_labels=3)
    p = _ClassPrototype(prototype_type="mad")
    r = p._create_mad_prototype(X[:, 0, :], y)
    assert r.shape == (3, X.shape[2])


def test_prototype_mean_centering():
    """Test mean centering is applied to the class prototype."""
    X, y = make_example_3d_numpy(n_cases=12, n_channels=2, n_timepoints=10, n_labels=3)
    p = _ClassPrototype(prototype_type="mean", mean_centering=True)
    prototype, labels = p._create_prototype(X, y)
    assert prototype.shape == (3, X.shape[1], X.shape[2])
    assert len(labels) == 3


def test_elbow_class_sum():
    """Test ElbowClassSum fit and transform."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=4, n_timepoints=20, n_labels=3)
    ecs = ElbowClassSum()
    ecs.fit(X, y)
    Xt = ecs.transform(X)
    assert Xt.shape == (X.shape[0], len(ecs.channels_selected_), X.shape[2])


def test_detect_knee_point_no_elbow_found():
    """Test all indices are returned when no elbow point is found."""
    result = _detect_knee_point([1.0, 1.0, 1.0], [0, 1, 2])
    assert result == [0, 1, 2]

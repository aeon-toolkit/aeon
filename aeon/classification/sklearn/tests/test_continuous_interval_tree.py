"""ContinuousIntervalTree test code."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.classification.sklearn._continuous_interval_tree import _TreeNode
from aeon.datasets import load_unit_test
from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)


def test_cit_output():
    """Test of RotF contracting and train estimate on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    cit = ContinuousIntervalTree(
        random_state=0,
    )
    cit.fit(X_train, y_train)

    expected = [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    np.testing.assert_array_almost_equal(
        expected, cit.predict_proba(X_test[:15]), decimal=4
    )


def test_cit_edge_cases():
    """Test ContinuousIntervalTree edge cases."""
    X, y = make_example_2d_numpy_collection(n_cases=5, n_labels=1)
    cit = ContinuousIntervalTree(max_depth=1)
    with pytest.raises(NotFittedError, match="please call `fit` first"):
        cit.predict_proba(X)
    model = cit.fit(X, y)
    assert isinstance(model, ContinuousIntervalTree)
    preds = cit.predict(X)
    # Should predict all 0
    assert np.all(preds == 0)

    X, y = make_example_2d_numpy_collection(n_cases=5)
    cit = ContinuousIntervalTree(max_depth=1)
    cit.fit(X, y)
    node = cit._root
    assert isinstance(node, _TreeNode)
    assert len(node.children[0].children) == 0

    X, y = make_example_3d_numpy(n_channels=3)
    with pytest.raises(
        ValueError, match="ContinuousIntervalTree is not a time series classifier"
    ):
        cit.fit(X, y)


def test_cit_nan_values():
    """Test that ContinuousIntervalTree can handle NaN values."""
    X, y = make_example_2d_numpy_collection()
    X[0:3, 0] = np.nan

    clf = ContinuousIntervalTree()
    clf.fit(X, y)
    clf.predict(X)

    # check inf values still raise an error
    X[0:3, 0] = np.inf
    with pytest.raises(ValueError):
        clf.fit(X, y)

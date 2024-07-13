"""ContinuousIntervalTree test code."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.classification.sklearn._continuous_interval_tree import _TreeNode
from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)


def test_predict_proba():
    """Test ContinuousIntervalTree predict_proba."""
    X, y = make_example_2d_numpy_collection(n_cases=5, n_labels=1)
    cit = ContinuousIntervalTree(max_depth=1)
    with pytest.raises(NotFittedError, match="please call `fit` first"):
        cit.predict_proba(X)
    model = cit.fit(X, y)
    assert isinstance(model, ContinuousIntervalTree)
    preds = cit.predict(X)
    assert np.all(preds == 0)
    # Should predict all 0
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


def test_nan_values():
    """Test that ContinuousIntervalTree can handle NaN values."""
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(10, 3))
    X[0:3, 0] = np.nan
    y = np.zeros(10)
    y[:5] = 1

    clf = ContinuousIntervalTree()
    clf.fit(X, y)
    clf.predict(X)

    # check inf values still raise an error
    X[0:3, 0] = np.inf
    with pytest.raises(ValueError):
        clf.fit(X, y)

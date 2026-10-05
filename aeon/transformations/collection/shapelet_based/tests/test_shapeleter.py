"""Tests for Shapeleter transformer."""

import numpy as np
from aeon.transformations.collection.shapelet_based._shapeleter import (
    ShapeleterTransformer,
)


def test_shapeleter_transformer_features():
    """Test fit and transform output dimensions with positional embeddings."""
    X = np.random.RandomState(42).normal(size=(5, 1, 30))
    y = np.array([0, 1, 0, 1, 0])

    st = ShapeleterTransformer(max_shapelets=20, ka=1.5, ko=1.5, random_state=42)
    st.fit(X, y)
    Xt = st.transform(X)

    assert Xt.ndim == 2
    assert Xt.shape[0] == 5
    # Feature count should exceed base RDST features due to pe_abs and pe_ord
    assert Xt.shape[1] > 20

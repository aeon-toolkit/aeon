# -*- coding: utf-8 -*-
"""Tests for PCATransformer."""

__author__ = ["aiwalter"]

from sktime.transformations.series.pca import PCATransformer

from sktime.utils._testing.series import _make_series


def test_pca():
    """Test PCA transformer."""
    X = _make_series(n_columns=3, return_numpy=False)
    transformer = PCATransformer(n_components=2)
    Xt = transformer.fit_transform(X)
    # test that the shape is correct
    assert Xt.shape == (X.shape[0], 2)
    # test that the column names are correct
    assert "PC_0" in Xt.columns
    assert "PC_1" in Xt.columns

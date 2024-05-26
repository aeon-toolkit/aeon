"""Tests for PCATransformer."""

__maintainer__ = ["TonyBagnall"]

from aeon.testing.utils.data_gen import make_series
from aeon.transformations.series._pca import PCASeriesTransformer


def test_pca():
    """Test PCA transformer."""
    X = make_series(n_columns=3, return_numpy=False)
    transformer = PCASeriesTransformer(n_components=2)
    Xt = transformer.fit_transform(X, axis=0)
    # test that the shape is correct
    assert Xt.shape == (X.shape[0], 2)
    # test that the column names are correct
    assert "PC_0" in Xt.columns
    assert "PC_1" in Xt.columns

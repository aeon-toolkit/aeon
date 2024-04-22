"""Tests for PCATransformer."""

__maintainer__ = []

from aeon.testing.utils.data_gen import make_series
from aeon.transformations.pca import PCATransformer
from aeon.utils.validation._dependencies import _check_python_version


def test_pca():
    """Test PCA transformer."""
    if _check_python_version(PCATransformer, severity="none"):
        X = make_series(n_columns=3, return_numpy=False)
        transformer = PCATransformer(n_components=2)
        Xt = transformer.fit_transform(X)
        # test that the shape is correct
        assert Xt.shape == (X.shape[0], 2)
        # test that the column names are correct
        assert "PC_0" in Xt.columns
        assert "PC_1" in Xt.columns

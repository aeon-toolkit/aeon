"""TS2vec tests."""

import numpy as np
import pytest

from aeon.transformations.collection.self_supervised import TS2Vec
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
@pytest.mark.parametrize("expected_feature_size", [3, 5, 10])
@pytest.mark.parametrize("n_series", [1, 2, 5])
@pytest.mark.parametrize("n_channels", [1, 2, 3])
@pytest.mark.parametrize("series_length", [3, 10, 20])
def test_ts2vec_output_shapes(
    expected_feature_size, n_series, n_channels, series_length
):
    """Test the output shapes of the TS2Vec transformer."""
    X = np.random.random(size=(n_series, n_channels, series_length))
    transformer = TS2Vec(output_dim=expected_feature_size, device="cpu", n_epochs=2)
    X_t = transformer.fit_transform(X)
    assert X_t.shape == (n_series, expected_feature_size)

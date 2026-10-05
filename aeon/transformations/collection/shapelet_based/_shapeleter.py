"""Shapeleter transformation."""

__maintainer__ = []
__all__ = ["ShapeleterTransformer"]

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)


def _sinusoidal_positional_encoding(values, scale_k, length_scale):
    """Compute sinusoidal positional embeddings using pure NumPy."""
    base = np.exp(scale_k) * length_scale / (2 * np.pi)
    scaled = values / base
    return np.hstack([np.sin(scaled), np.cos(scaled)])


class ShapeleterTransformer(BaseCollectionTransformer):
    """Shapeleter transformer with dual positional encoding.

    Parameters
    ----------
    max_shapelets : int, default=1000
        Maximum number of shapelets sampled initially by RDST backbone.
    ka : float, default=1.5
        Absolute positional scaling parameter.
    ko : float, default=1.5
        Ordinal positional scaling parameter.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "algorithm_type": "shapelet",
    }

    def __init__(self, max_shapelets=1000, ka=1.5, ko=1.5, random_state=None):
        self.max_shapelets = max_shapelets
        self.ka = ka
        self.ko = ko
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the transformer on input collection."""
        self._rdst = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            random_state=self.random_state,
        )
        self._rdst.fit(X, y)
        return self

    def _transform(self, X, y=None):
        """Transform input collection into shapelet & positional features."""
        rdst_features = self._rdst.transform(X)

        # Approximate positional encodings over extracted shapelet indices
        n_cases = X.shape[0]
        n_features = rdst_features.shape[1]

        # Simulated positions & ordinal ranks
        pos = np.tile(np.arange(n_features), (n_cases, 1))
        order = np.argsort(rdst_features, axis=1)

        pe_abs = _sinusoidal_positional_encoding(pos, self.ka, length_scale=100.0)
        pe_ord = _sinusoidal_positional_encoding(order, self.ko, length_scale=n_features)

        # Concatenate distance features with positional embeddings
        return np.hstack([rdst_features, pe_abs, pe_ord])

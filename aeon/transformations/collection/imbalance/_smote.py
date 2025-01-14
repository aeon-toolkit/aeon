"""Wrapper for imblearn minority class rebalancer SMOTE."""

from imblearn.over_sampling import SMOTE as smote

from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["TonyBagnall"]
__all__ = ["SMOTE"]


class SMOTE(BaseCollectionTransformer):
    """Wrapper for SMOTE transform."""

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "requires_y": True,
    }

    def __init__(self, sampling_strategy="auto", random_state=None, k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _fit(self, X, y=None):
        self.smote_ = smote(self.sampling_strategy, self.random_state, self.k_neighbors)
        self.smote_.fit(X, y)

    def _transform(self, X, y=None):
        return self.smote_.resample(X, y)

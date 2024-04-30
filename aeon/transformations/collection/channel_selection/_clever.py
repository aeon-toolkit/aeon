"""Common principal component Loading based Variable subset selection method (CleVer).

Channel selection based on clustered principal components.
"""

from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["TonyBagnall"]


class ElbowClassSum(BaseCollectionTransformer):
    """Common principal component Loading based Variable subset selection method [1].

    Assumes input of collectionm shape (n_cases, n_channels, n_timepoints). Works as
    follows
    1. Take the PCA of each series independently. Form linear combinations of channels
    2. Form descriptive components for each class.
    3. Select a subset of variables using K-meansx clustering on "CDPC loadings"

    Parameters
    ----------
    delta: float
        predefined threshold

    References
    ----------
    ..[1]:
    """

    _tags = {
        "capability:multivariate": True,
        "requires_y": True,
    }

    def __init__(
        self,
    ):
        super().__init__()

    def _fit(self, X, y):
        """Fit selects channels to retain."""
        return self

    def _transform(self, X, y=None):
        """Select channels based on fit."""

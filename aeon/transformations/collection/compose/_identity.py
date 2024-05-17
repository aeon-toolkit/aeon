"""Identity transformer."""

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils import COLLECTIONS_DATA_TYPES


class CollectionId(BaseCollectionTransformer):
    """Identity transformer, returns data unchanged in transform/inverse_transform."""

    _tags = {
        "X_inner_type": COLLECTIONS_DATA_TYPES,
        "y_inner_type": "None",
        "fit_is_empty": True,
        "capability:inverse_transform": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
    }

    def _transform(self, X, y=None):
        return X

    def _inverse_transform(self, X, y=None):
        return X

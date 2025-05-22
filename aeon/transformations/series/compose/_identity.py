"""Identity transformer."""

from aeon.transformations.series import BaseSeriesTransformer
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES


class SeriesId(BaseSeriesTransformer):
    """Identity transformer, returns data unchanged in transform/inverse_transform."""

    _tags = {
        "X_inner_type": VALID_SERIES_INNER_TYPES,
        "fit_is_empty": True,
        "capability:inverse_transform": True,
        "capability:multivariate": True,
        "capability:missing_values": True,
    }

    def __init__(self):
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        return X

    def _inverse_transform(self, X, y=None):
        return X

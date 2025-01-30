"""Base channel selection transformer.

Extends BaseCollectionTransformer and implements _transform to return
selected indexes.
"""

__maintainer__ = ["TonyBagnall"]

from abc import abstractmethod

from aeon.transformations.collection.base import BaseCollectionTransformer


class BaseChannelSelector(BaseCollectionTransformer):
    """Abstract base class for channel selection transformers.

    Extends BaseCollectionTransformer by implementing``_transform`` to return
    channels selected in fit.

    Attributes
    ----------
    channels_selected_ : list[int]
        List of channels selected in fit.
    """

    @abstractmethod
    def __init__(self):
        self.channels_selected_ = []
        super().__init__()

    def _transform(self, X, y=None):
        """
        Transform X and return a transformed version.

        Parameters
        ----------
        X : np.ndarray
            Time series collection of shape ``(n_cases,n_channels,n_timepoints)``.

        Returns
        -------
        np.ndarray
            Time series collection with a subset of channels
        """
        return X[:, self.channels_selected_]

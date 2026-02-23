"""Base class for similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSimilaritySearch",
]


from abc import abstractmethod

import numpy as np
from numba.typed import List

from aeon.base import BaseAeonEstimator


class BaseSimilaritySearch(BaseAeonEstimator):
    """Base class for similarity search applications."""

    _tags = {
        "requires_y": False,
        "fit_is_empty": False,
    }

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | List,
        y=None,
    ):
        """
        Fit estimator to X.

        State change:
            Changes state to "fitted".

        Writes to self:
            _is_fitted : flag is set to True.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit transform to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: np.ndarray | None = None,
    ):
        """
        Predict method.

        Can either work with new series or with None (for case when predict can be made
        using the data given in fit against itself) depending on the estimator.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit transform to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i
                None : If None type is accepted, it means that the predict function will
                work only with the data given in fit. (e.g. self matrix profile instead)
        """
        ...

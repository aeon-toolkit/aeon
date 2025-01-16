"""Base similiarity search for collections."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseCollectionSimilaritySearch",
]

from abc import abstractmethod
from typing import Union, final

import numpy as np
from numba import get_num_threads, set_num_threads

from aeon.base import BaseCollectionEstimator
from aeon.similarity_search._base import BaseSimilaritySearch


class BaseCollectionSimilaritySearch(BaseCollectionEstimator, BaseSimilaritySearch):
    """Similarity search base class for collections."""

    # tag values specific to CollectionTransformers
    _tags = {
        "input_data_type": "Collection",
    }

    @abstractmethod
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        super().__init__()

    @final
    def fit(
        self,
        X: np.ndarray,
        y=None,
    ):
        """
        Fit method: data preprocessing and storage.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Input array to be used as database for the similarity search. If it is an
            unequal length collection, it should be a list of 2d numpy arrays.
        y : optional
            Not used.

        Raises
        ------
        TypeError
            If the input X array is not 3D raise an error.

        Returns
        -------
        self
        """
        self.reset()
        X = self._preprocess_collection(X)
        # Store minimum number of n_timepoints for unequal length collections
        self.n_channels_ = X[0].shape[1]
        self.n_cases_ = len(X)
        self.X_ = X

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        self._fit(X, y=y)
        set_num_threads(prev_threads)

        self.is_fitted = True
        return self

    @abstractmethod
    def _fit(
        self,
        X: np.ndarray,
        y=None,
    ): ...

    def _pre_predict(
        self,
        X: Union[np.ndarray, None] = None,
    ):
        """
        Predict method.

        Parameters
        ----------
        X : Union[np.ndarray, None], optional
            Optional data to use for predict.. The default is None.

        """
        self._check_is_fitted()
        if X is not None:
            # Could we call somehow _preprocess_series from a BaseCollectionEstimator ?
            self._check_predict_format(X)
        return X

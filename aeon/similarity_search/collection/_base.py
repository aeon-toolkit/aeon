"""Base similarity search for collections."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseCollectionSimilaritySearch",
]

from abc import abstractmethod
from typing import final

import numpy as np

from aeon.base import BaseCollectionEstimator
from aeon.similarity_search._base import BaseSimilaritySearch


class BaseCollectionSimilaritySearch(BaseCollectionEstimator, BaseSimilaritySearch):
    """
    Similarity search base class for collections.

    Such estimators include nearest neighbors on whole series or subsequences with
    indexing or concenssus motifs search over a collection.
    """

    # tag values specific to CollectionTransformers
    _tags = {
        "input_data_type": "Collection",
        "capability:multivariate": True,
        "X_inner_type": ["numpy3D"],
    }

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
        self.n_channels_ = self.metadata_["n_channels"]
        self.n_cases_ = self.metadata_["n_cases"]
        self._fit(X, y=y)
        self.is_fitted = True
        return self

    @abstractmethod
    def _fit(self, X: np.ndarray, y=None): ...

    @final
    def predict(self, X, **kwargs):
        """
        Predict function.

        Parameters
        ----------
        X : np.ndarray, 3D array of shape = (n_cases, n_channels, n_timepoints)
            Collections of series to predict on.
        kwargs : dict, optional
            Additional keyword arguments to be passed to the _predict function of the
            estimator.

        Returns
        -------
        indexes : np.ndarray, shape = (n_cases, k)
            Indexes of series in the that are similar to X.
        distances : np.ndarray, shape = (n_cases, k)
            Distance of the matches to each series

        """
        self._check_is_fitted()
        X = self._preprocess_collection(X, store_metadata=False)
        self._check_predict_series_format(X)
        indexes, distances = self._predict(X, **kwargs)
        return indexes, distances

    def _check_predict_series_format(self, X):
        """
        Check whether a series X in predict is correctly formatted.

        Parameters
        ----------
        X : np.ndarray, shape = (n_channels, n_timepoints)
            A series to be used in predict.
        """
        if self.n_channels_ != X[0].shape[0]:
            raise ValueError(
                f"Expected X to have {self.n_channels_} channels but"
                f" got {X[0].shape[0]} channels."
            )

    @abstractmethod
    def _predict(self, X, **kwargs): ...

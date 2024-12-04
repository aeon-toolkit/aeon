"""Base class for similarity search."""

__maintainer__ = ["baraline"]

from abc import abstractmethod
from typing import Optional, Union, final

import numpy as np
from numba import get_num_threads, set_num_threads
from numba.typed import List

from aeon.base import BaseCollectionEstimator


class BaseSimilaritySearch(BaseCollectionEstimator):
    """
    Base class for similarity search applications.

    Parameters
    ----------
    normalize : bool, optional
        Whether the inputs should be z-normalized. The default is False.
    n_jobs : int, optional
        Number of parallel jobs to use. The default is 1.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "fit_is_empty": False,
        "X_inner_type": ["np-list", "numpy3D"],
    }

    @abstractmethod
    def __init__(
        self,
        normalize: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        self.n_jobs = n_jobs
        self.normalize = normalize
        super().__init__()

    @final
    def fit(
        self,
        X: Union[np.ndarray, List],
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
        prev_threads = get_num_threads()
        X = self._preprocess_collection(X)
        # Store minimum number of n_timepoints for unequal length collections
        self.min_timepoints_ = min([X[i].shape[-1] for i in range(len(X))])
        self.n_channels_ = X[0].shape[0]
        self.n_cases_ = len(X)
        if self.metadata_["unequal_length"]:
            X = List(X)
        set_num_threads(self._n_jobs)
        self._fit(X, y)
        set_num_threads(prev_threads)
        self.is_fitted = True
        return self

    @abstractmethod
    def find_motifs(
        self,
        k: int,
        threshold: float,
        X: Optional[np.ndarray] = None,
        allow_overlap: Optional[bool] = True,
    ):
        """
        Find the top-k motifs in the training data.

        Given ``k`` and ``threshold`` parameters, this methods returns the top-k motif
        sets. We define a motif set as a set of candidates which all are at a distance
        of at most ``threshold`` from each other. The top-k motifs sets are the
        motif sets with the most candidates.

        Parameters
        ----------
        X : np.ndarray, optional
            The query in which we want to indentify motifs. If provided, the motifs
            extracted should appear in X and in the database given in fit. If not
            provided, the motifs will be extracted only from the database given in fit.
        k : int, optional
            Number of motifs to return
        threshold : int, optional
            A threshold on the similarity measure to determine which candidates will be
            part of a motif set.
        allow_overlap: bool, optional
            Wheter a candidate can be part of multiple motif sets (True), or if motif
            sets should be mutually exclusive (False).

        Returns
        -------
        list of ndarray, shape=(k,)
            A list of at most ``k`` numpy arrays containing the indexes of the
            candidates in each motif.

        """
        ...

    @abstractmethod
    def find_neighbors(
        self,
        X: np.ndarray,
        k: Optional[int] = 1,
        threshold: Optional[float] = np.inf,
    ):
        """
        Find the top-k neighbors of X in the database.

        Given ``k`` and ``threshold`` parameters, this methods returns the top-k
        neighbors of X, such as each of the ``k`` neighbors as a distance inferior or
        equal to ``threshold``. By default, ``threshold`` is set to infinity. It is
        possible for this method to return less than ``k`` neighbors, either if there
        is less than ``k`` admissible candidate in the database, or if in the top-k
        candidates, some do not meet the ``threshold`` condition.

        Parameters
        ----------
        X: np.ndarray
            The query for which we want to identify nearest neighbors in the database.
        k : int, optional
            Number of neighbors to return.
        threshold : int, optional
            A threshold on the distance to determine which candidates will be returned.

        Returns
        -------
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the indexes of the
            candidates in each motif.

        """
        ...

    @abstractmethod
    def _fit(self, X, y=None): ...

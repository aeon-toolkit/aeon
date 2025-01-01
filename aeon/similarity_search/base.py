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
    normalise : bool, optional
        Whether the inputs should be z-normalised. The default is False.
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
        normalise: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        self.n_jobs = n_jobs
        self.normalise = normalise
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
        self.reset()
        prev_threads = get_num_threads()
        self._check_fit_format(X)
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
        X: np.ndarray,
        k: int,
        threshold: float,
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
        X : np.ndarray,
            A series in which we want to indentify motifs.
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
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the indexes of the
            motifs.
        ndarray, shape=(k,)
            A numpy array of at most ``k`` elements containing the distances of the
            motifs to .

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

    def _check_fit_format(self, X):
        if isinstance(X, np.ndarray):  # "numpy3D" or numpy2D
            if X.ndim != 3:
                raise TypeError(
                    f"A np.ndarray given in fit must be 3D but found {X.ndim}D"
                )
        elif isinstance(X, list):  # np-list or df-list
            if isinstance(X[0], np.ndarray):  # if one a numpy they must all be 2D numpy
                for a in X:
                    if not (isinstance(a, np.ndarray) and a.ndim == 2):
                        raise TypeError(
                            "A np-list given in fit must contain 2D np.ndarray but"
                            f" found {a.ndim}D"
                        )

    def _check_find_neighbors_motif_format(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise TypeError(
                    "A np.ndarray given in find_neighbors must be 2D"
                    f"(n_channels, n_timepoints) but found {X.ndim}D."
                )
        else:
            raise TypeError(
                "Expected a 2D np.ndarray in find_neighbors but found" f" {type(X)}."
            )
        if self.n_channels_ != X.shape[0]:
            raise ValueError(
                f"Expected X to have {self.n_channels_} channels but"
                f" got {X.shape[0]} channels."
            )

    @abstractmethod
    def _fit(self, X, y=None): ...

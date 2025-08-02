"""
Base class for estimators that fit collections of time series.

    class name: BaseCollectionEstimator

Defining methods:
    preprocessing         - _preprocess_collection(self, X, store_metadata=True)
    input checking        - _check_X(self, X)
    input conversion      - _convert_X(self, X)
    shape checking        - _check_shape(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

"""

from abc import abstractmethod

from aeon.base._base import BaseAeonEstimator
from aeon.utils.preprocessing import preprocess_collection
from aeon.utils.validation.collection import (
    get_n_channels,
    get_n_timepoints,
)


class BaseCollectionEstimator(BaseAeonEstimator):
    """
    Base class for estimators that use collections of time series for ``fit``.

    Provides functions that are common to estimators which use collections such as
    ``BaseClassifier``, ``BaseRegressor``, ``BaseClusterer``, ``BaseSimilaritySearch``
    and ``BaseCollectionTransformer``. Functionality includes checking and
    conversion of input in ``fit``, ``predict`` and ``predict_proba``, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
    }

    @abstractmethod
    def __init__(self):
        self.metadata_ = {}  # metadata/properties of data seen in fit
        self._n_jobs = 1

        super().__init__()

    def _preprocess_collection(self, X, store_metadata=True):
        """
        Preprocess input X prior to calling fit.

        1. Checks the characteristics of X and that self can handle the data
        2. Stores metadata about X in self.metadata_ if store_metadata is True
        3. Converts X to X_inner_type if necessary

        Parameters
        ----------
        X : collection
            See aeon.utils.COLLECTIONS_DATA_TYPES for details on aeon supported
            data structures.
        store_metadata : bool, default=True
            Whether to store metadata about X in self.metadata_.

        Returns
        -------
        X : collection
            Processed X. A data structure of type self.get_tag("X_inner_type").

        Raises
        ------
        ValueError
            If X is an invalid type or has characteristics that the estimator cannot
            handle.

        See Also
        --------
        _check_X :
            Function that checks X is valid before conversion.
        _convert_X :
            Function that converts to inner type.

        Examples
        --------
        >>> from aeon.testing.mock_estimators import MockClassifier
        >>> from aeon.testing.data_generation import make_example_2d_numpy_collection
        >>> clf = MockClassifier()
        >>> X, _ = make_example_2d_numpy_collection(n_cases=10, n_timepoints=20)
        >>> X2 = clf._preprocess_collection(X)
        >>> X2.shape
        (10, 1, 20)
        """
        result = preprocess_collection(
            X,
            self.get_tags(),
            return_metadata=store_metadata,
        )
        if store_metadata:
            X, meta = result
            self.metadata_ = meta
        else:
            X = result
        return X

    def _check_shape(self, X):
        """
        Check that the shape of X is consistent with the data seen in fit.

        Parameters
        ----------
        X : data structure
            Must be of type aeon.registry.COLLECTIONS_DATA_TYPES.

        Raises
        ------
        ValueError
            If the shape of X is not consistent with the data seen in fit.
        """
        # if metadata is empty, then we have not seen any data in fit. If the estimator
        # has not been fitted, then _is_fitted should catch this.
        # there are valid cases where metadata is empty and the estimator has been
        # fitted, i.e. deep learner loading.
        if len(self.metadata_) != 0:
            if not self.get_tag("capability:unequal_length"):
                nt = get_n_timepoints(X)
                if nt != self.metadata_["n_timepoints"]:
                    raise ValueError(
                        "X has different length to the data seen in fit but "
                        "this classifier cannot handle unequal length series."
                        f"length of train set was {self.metadata_['n_timepoints']}",
                        f" length in predict is {nt}.",
                    )
            if self.get_tag("capability:multivariate"):
                nc = get_n_channels(X)
                if nc != self.metadata_["n_channels"]:
                    raise ValueError(
                        "X has different number of channels to the data seen in fit "
                        "number of channels in train set was ",
                        f"{self.metadata_['n_channels']} but in predict it is {nc}.",
                    )

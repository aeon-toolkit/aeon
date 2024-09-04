"""Base class for estimators that fit collections of time series."""

from aeon.base._base import BaseEstimator
from aeon.utils.conversion import (
    convert_collection,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation._dependencies import _check_estimator_deps
from aeon.utils.validation.collection import (
    get_n_cases,
    get_n_channels,
    get_n_timepoints,
    get_type,
    has_missing,
    is_equal_length,
    is_univariate,
)


class BaseCollectionEstimator(BaseEstimator):
    """Base class for estimators that use collections of time series for method fit.

    Provides functions that are common to BaseClassifier, BaseRegressor,
    BaseClusterer and BaseCollectionTransformer for the checking and
    conversion of input to fit, predict and predict_proba, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
        "X_inner_type": "numpy3D",
        "python_version": None,
    }

    def __init__(self):
        self.metadata_ = {}  # metadata/properties of data seen in fit
        self.fit_time_ = 0  # time elapsed in last fit call
        self._n_jobs = 1
        super().__init__()
        _check_estimator_deps(self)

    def _preprocess_collection(self, X, store_metadata=True):
        """Preprocess input X prior to call to fit.

        1. Checks the characteristics of X, store metadata, checks self can handle
            the data
        2. convert X to X_inner_type
        3. Check multi-threading against capabilities

        Parameters
        ----------
        X : collection
            See aeon.utils.registry.COLLECTIONS_DATA_TYPES for details
            on aeon supported data structures.
        store_metadata : bool, default=True
            Whether to store metadata about X in self.metadata_.

        Returns
        -------
        Data structure of type self.tags["X_inner_type"]

        See Also
        --------
        _check_X : function that checks X is valid before conversion.
        _convert_X : function that converts to inner type.

        Examples
        --------
        >>> from aeon.base import BaseCollectionEstimator
        >>> import numpy as np
        >>> bce = BaseCollectionEstimator()
        >>> X = np.random.random(size=(10,20))
        >>> X2 = bce._preprocess_collection(X)
        >>> X2.shape
        (10, 1, 20)
        """
        meta = self._check_X(X)
        if len(self.metadata_) == 0 and store_metadata:
            self.metadata_ = meta

        X = self._convert_X(X)
        # This usage of n_jobs is legacy, see issue #102
        multithread = self.get_tag("capability:multithreading")
        if multithread:
            if hasattr(self, "n_jobs"):
                self._n_jobs = check_n_jobs(self.n_jobs)
            else:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )
        return X

    def _check_X(self, X):
        """Check classifier input X is valid.

        Check if the input data is a compatible type, and that this estimator is
        able to handle the data characteristics. This is done by matching the
        capabilities of the estimator against the metadata for X for
        univariate/multivariate, equal length/unequal length and no missing
        values/missing values.

        Parameters
        ----------
        X : data structure
           See aeon.utils.registry.COLLECTIONS_DATA_TYPES for details
           on aeon supported data structures.

        Returns
        -------
        dict
            Meta data about X, with flags:
            metadata["multivariate"] : whether X has more than one channel or not
            metadata["missing_values"] : whether X has missing values or not
            metadata["unequal_length"] : whether X contains unequal length series.
            metadata["n_cases"] : number of cases in X
            metadata["n_channels"] : number of channels in X
            metadata["n_timepoints"] : number of timepoints in X if equal length, else
                None

        See Also
        --------
        _convert_X : function that converts X after it has been checked.

        Examples
        --------
        >>> from aeon.classification.hybrid import HIVECOTEV2
        >>> import numpy as np
        >>> X = np.random.random(size=(5,3,10))  # X is equal length, multivariate
        >>> hc = HIVECOTEV2()
        >>> meta = hc._check_X(X)  # HC2 can handle this
        """
        metadata = self._get_X_metadata(X)
        # Check classifier capabilities for X
        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")

        # Check capabilities vs input
        problems = []
        if metadata["missing_values"] and not allow_missing:
            problems += ["missing values"]
        if metadata["multivariate"] and not allow_multivariate:
            problems += ["multivariate series"]
        if metadata["unequal_length"] and not allow_unequal:
            problems += ["unequal length series"]

        if problems:
            # construct error message
            problems_and = " and ".join(problems)
            problems_or = " or ".join(problems)
            msg = (
                f"Data seen by instance of {type(self).__name__} has {problems_and}, "
                f"but {type(self).__name__} cannot handle {problems_or}. "
            )
            raise ValueError(msg)

        return metadata

    def _convert_X(self, X):
        """Convert X to type defined by tag X_inner_type.

        If the input data is already an allowed type, it is returned unchanged.

        If multiple types are allowed by self, then the best one for the type of input
        data is selected. So, for example, if X_inner_tag is `["np-list", "numpy3D"]`
        and an `df-list` is passed containing equal length series, will be converted
        to numpy3D.

        Parameters
        ----------
        X : data structure
            Must be of type aeon.utils.registry.COLLECTIONS_DATA_TYPES.

        Returns
        -------
        data structure of type one of self.get_tag("X_inner_type").

        See Also
        --------
        _check_X : function that checks X is valid and finds metadata.

        Examples
        --------
        >>> from aeon.classification.hybrid import HIVECOTEV2
        >>> import numpy as np
        >>> from aeon.utils.validation import get_type
        >>> X = [np.random.random(size=(5,10)), np.random.random(size=(5,10))]
        >>> get_type(X)
        'np-list'
        >>> hc = HIVECOTEV2()
        >>> hc.get_tag("X_inner_type")
        'numpy3D'
        >>> X = hc._convert_X(X)
        >>> get_type(X)
        'numpy3D'
        """
        inner_type = self.get_tag("X_inner_type")
        if not isinstance(inner_type, list):
            inner_type = [inner_type]
        input_type = get_type(X)

        # Check if we need to convert X, return if not
        if input_type in inner_type:
            return X

        if len(self.metadata_) == 0:
            metadata = self._get_X_metadata(X)
        else:
            metadata = self.metadata_

        # Convert X to X_inner_type if possible
        # If self can handle more than one internal type, resolve correct conversion
        # If unequal, choose data structure that can hold unequal
        if metadata["unequal_length"]:
            inner_type = resolve_unequal_length_inner_type(inner_type)
        else:
            inner_type = resolve_equal_length_inner_type(inner_type)

        return convert_collection(X, inner_type)

    def _check_shape(self, X):
        """Check that the shape of X is consistent with the data seen in fit.

        Parameters
        ----------
        X : data structure
            Must be of type aeon.registry.COLLECTIONS_DATA_TYPES.
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

    @staticmethod
    def _get_X_metadata(X):
        # Get and store X meta data.
        metadata = {}
        metadata["multivariate"] = not is_univariate(X)
        metadata["missing_values"] = has_missing(X)
        metadata["unequal_length"] = not is_equal_length(X)
        metadata["n_cases"] = get_n_cases(X)
        metadata["n_channels"] = get_n_channels(X)
        metadata["n_timepoints"] = (
            None if metadata["unequal_length"] else get_n_timepoints(X)
        )
        return metadata

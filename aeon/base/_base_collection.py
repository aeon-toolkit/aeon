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
        "X_inner_type": "numpy3D",
        "capability:multithreading": False,
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    def __init__(self):
        self.metadata_ = {}  # metadata/properties of data seen in fit
        self.fit_time_ = 0  # time elapsed in last fit call
        self._n_jobs = 1
        super().__init__()
        _check_estimator_deps(self)

    def _preprocess_collection(self, X):
        """Preprocess input X prior to call to fit.

        1. Checks the characteristics of X, store metadata, checks self can handle
        the data
        2. convert X to X_inner_type
        3. Check multi-threading against capabilities

        Parameters
        ----------
        X : collection
            See aeon.registry.COLLECTIONS_DATA_TYPES for details
            on aeon supported data structures.

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
        if len(self.metadata_) == 0:
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
           See aeon.registry.COLLECTIONS_DATA_TYPES for details
           on aeon supported data structures.

        Returns
        -------
        dict
            Meta data about X, with flags:
            metadata["missing_values"] : whether X has missing values or not
            metadata["multivariate"] : whether X has more than one channel or not
            metadata["unequal_length"] : whether X contains unequal length series.

        See Also
        --------
        _convert_X : function that converts X after it has been checked.

        Examples
        --------
        >>> from aeon.classification.hybrid import HIVECOTEV2
        >>> import numpy as np
        >>> X = np.random.random(size=(5,3,10)) # X is equal length, multivariate
        >>> hc = HIVECOTEV2()
        >>> meta=hc._check_X(X)    # HC2 can handle this
        """
        metadata = self._get_metadata(X)
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

        if self.metadata_ has not been set, it is set here from X, because we need to
        know if the data is unequal length in order to choose between different
        allowed input types. If multiple types are allowed by self, then the best
        one for the data is selected. So, for example, if X_inner_tag is `["np-list",
        "numpy3D"]` and an `np-list` is passed containing equal length series,
        X will be converted to numpy3D.

        Parameters
        ----------
        X : data structure
        must be of type aeon.registry.COLLECTIONS_DATA_TYPES.

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
        if len(self.metadata_) == 0:
            metadata = self._get_metadata(X)
        else:
            metadata = self.metadata_
        # Convert X to X_inner_type if possible
        inner_type = self.get_tag("X_inner_type")
        if isinstance(inner_type, list):
            # If self can handle more than one internal type, resolve correct conversion
            # If unequal, choose data structure that can hold unequal
            if metadata["unequal_length"]:
                inner_type = resolve_unequal_length_inner_type(inner_type)
            else:
                inner_type = resolve_equal_length_inner_type(inner_type)
        X = convert_collection(X, inner_type)
        return X

    @staticmethod
    def _get_metadata(X):
        # Get and store X meta data.
        metadata = {}
        metadata["multivariate"] = not is_univariate(X)
        metadata["missing_values"] = has_missing(X)
        metadata["unequal_length"] = not is_equal_length(X)
        metadata["n_cases"] = get_n_cases(X)

        return metadata

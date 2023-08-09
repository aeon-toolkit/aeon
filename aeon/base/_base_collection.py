# -*- coding: utf-8 -*-
"""Base class for estimators that fit collections of time series."""
from aeon.base._base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps
from aeon.utils.validation.collection import (
    convert_collection,
    get_n_cases,
    has_missing,
    is_equal_length,
    is_univariate,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)


class BaseCollectionEstimator(BaseEstimator):
    """Base class for estimators that fit collections of time series.

    Groups common functions that are used by BaseClassifier, BaseRegressor,
    BaseClusterer and BaseCollectionTransformer.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "X_inner_mtype": "numpy3D",
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    def __init__(self):
        self.metadata_ = {}  # metadata/properties of data seen in fit
        self.fit_time_ = 0  # time elapsed in last fit call
        self._n_jobs = 1
        super(BaseCollectionEstimator, self).__init__()
        _check_estimator_deps(self)

    def checkX(self, X):
        """Check classifier input X is valid.

        Check if the input data is a compatible type, and that this classifier is
        able to handle the data characteristics. This is done by matching the
        capabilities of the classifier against the metadata for X for
        univariate/multivariate, equal length/unequal length and no missing
        values/missing values.

        Parameters
        ----------
        X : object.

        Returns
        -------
        bool : True if classifier can deal with X.

        See Also
        --------
        convertX : function that converts X after it has been checked.

        Examples
        --------
        >>> from aeon.classification.hybrid import HIVECOTEV2
        >>> import numpy as np
        >>> X = np.random.random(size=(5,3,10))
        >>> X[0][1][3] = np.NAN # X is equal length, multivariate, with missing
        >>> hc = HIVECOTEV2()
        >>> hc.checkX(X)    # HC2 can handle this
        True
        """
        self.metadata_ = _get_metadata(X)
        # Check classifier capabilities for X
        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")

        # Check capabilities vs input
        problems = []
        if self.metadata_["missing_values"] and not allow_missing:
            problems += ["missing values"]
        if self.metadata_["multivariate"] and not allow_multivariate:
            problems += ["multivariate series"]
        if self.metadata_["unequal_length"] and not allow_unequal:
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
        return self.metadata_

    def convertX(self, X):
        """Convert X to type defined by tag X_inner_mtype.

        if self.metadata_ has not been set, it is set here from X, because we need to
        know if the data is unequal length in order to choose between different
        allowed input types. If multiple types are allowed by self, then the best
        one for the data is selected. So, for example, if X_inner_tag is `["np-list",
        "numpy3D"]` and an `np-list` is passed containing equal length series,
        X will be converted to numpy3D.

        Parameters
        ----------
        X : data structure
        must be of type aeon.utils.validation.collection.COLLECTIONS_DATA_TYPES.

        Returns
        -------
        data structure of type one of self.get_tag("X_inner_mtype").


        See Also
        --------
        checkX : function that checks X is valid and finds metadata.

        Examples
        --------
        >>> from aeon.classification.hybrid import HIVECOTEV2
        >>> import numpy as np
        >>> from aeon.utils.validation.collection import get_type
        >>> X = [np.random.random(size=(5,10), np.random.random(size=(5,10)]
        >>> get_type(X)
        np-list
        >>> hc = HIVECOTEV2()
        >>> hc.get_tag("X_inner_mtype")
        ["np-list", "numpy3D"]
        >>> X = hc.convertX(X)
        >>> get_type(X)
        numpy3D
        """
        if len(self.metadata_) == 0:
            metadata = _get_metadata(X)
        else:
            metadata = self.metadata_
        # Convert X to X_inner_mtype if possible
        inner_type = self.get_tag("X_inner_mtype")
        if type(inner_type) == list:
            # If self can handle more than one internal type, resolve correct conversion
            # If unequal, choose data structure that can hold unequal
            if metadata["unequal_length"]:
                inner_type = resolve_unequal_length_inner_type(inner_type)
            else:
                inner_type = resolve_equal_length_inner_type(inner_type)
        X = convert_collection(X, inner_type)
        return X


def _get_metadata(X):
    # Get and store X meta data.
    metadata = {}
    metadata["multivariate"] = not is_univariate(X)
    metadata["missing_values"] = has_missing(X)
    metadata["unequal_length"] = not is_equal_length(X)
    metadata["n_cases"] = get_n_cases(X)

    return metadata

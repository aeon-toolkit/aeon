"""
Base class template for Collection transformers.

    class name: BaseCollectionTransformer

Defining methods:
    fitting         - fit(self, X, y=None)
    transform       - transform(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)
    updating        - update(self, X, y=None)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__maintainer__ = []
__all__ = [
    "BaseCollectionTransformer",
]

from abc import ABCMeta, abstractmethod
from typing import final

import numpy as np
import pandas as pd

from aeon.base import BaseCollectionEstimator
from aeon.transformations.base import BaseTransformer


class BaseCollectionTransformer(
    BaseCollectionEstimator, BaseTransformer, metaclass=ABCMeta
):
    """Transformer base class for collections."""

    # tag values specific to CollectionTransformers
    _tags = {
        "input_data_type": "Collection",
        "output_data_type": "Collection",
        "fit_is_empty": False,
        "requires_y": False,
        "capability:inverse_transform": False,
    }

    def __init__(self):
        super().__init__()

    @final
    def fit(self, X, y=None):
        """Fit transformer to X, optionally using y if supervised.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : a fitted instance of the estimator
        """
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")
        # skip the rest if fit_is_empty is True
        if self.get_tag("fit_is_empty"):
            self._is_fitted = True
            return self
        self.reset()

        # input checks and datatype conversion
        X_inner = self._preprocess_collection(X)
        y_inner = y
        self._fit(X=X_inner, y=y_inner)

        self._is_fitted = True

        return self

    @final
    def transform(self, X, y=None):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True
        _X : optionally accessed, only available if remember_data tag is True
        fitted model attributes (ending in "_") : must be set, accessed by _transform

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner = self._preprocess_collection(X)
        y_inner = y

        Xt = self._transform(X=X_inner, y=y_inner)

        return Xt

    @final
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
            possibly coerced to inner type or update_data compatible type
            by reference, when possible
        model attributes (ending in "_") : dependent on estimator.

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # input checks and datatype conversion
        self.reset()
        X_inner = self._preprocess_collection(X)
        y_inner = y

        Xt = self._fit_transform(X=X_inner, y=y_inner)

        self._is_fitted = True

        return Xt

    @final
    def inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        Currently it is assumed that only transformers with tags
             "input_data_type"="Series", "output_data_type"="Series",
        can have an inverse_transform.

        State required:
             Requires state to be "fitted".

        Accesses in self:
         _is_fitted : must be True
         _X : optionally accessed, only available if remember_data tag is True
         fitted model attributes (ending in "_") : accessed by _inverse_transform

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
             Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X
        """
        if self.get_tag("skip-inverse-transform"):
            return X

        if not self.get_tag("capability:inverse_transform"):
            raise NotImplementedError(
                f"{type(self)} does not implement inverse_transform"
            )

        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner = self._preprocess_collection(X)
        y_inner = y

        Xt = self._inverse_transform(X=X_inner, y=y_inner)

        return Xt

    @final
    def update(self, X, y=None, update_params=True):
        """Update transformer with X, optionally y.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True
        fitted model attributes (ending in "_") : must be set, accessed by _update

        Writes to self:
        _X : set to be X, if remember_data tag is True, potentially used in _update
        fitted model attributes (ending in "_") : only if update_params=True
            type and nature of update are dependent on estimator

        Parameters
        ----------
        X : data to update of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
        update_params : bool, default=True
            whether the model is updated. Yes if true, if false, simply skips call.
            argument exists for compatibility with forecasting module.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # check whether is fitted
        self.check_is_fitted()

        # if requires_y is set, y is required in fit and update
        if self.get_tag("requires_y") and y is None:
            raise ValueError(f"{self.__class__.__name__} requires `y` in `update`.")

        # check and convert X/y
        X_inner = self._preprocess_collection(X)
        y_inner = y

        # update memory of X, if remember_data exists and is set to True
        if self.get_tag("remember_data", tag_value_default=False):
            self._X = X_inner

        # skip everything if update_params is False or fit_is_empty is present and True
        if not update_params or self.get_tag("fit_is_empty", tag_value_default=False):
            return self

        self._update(X=X_inner, y=y_inner)

        return self

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator

        See extension_templates/transformer.py for implementation details.
        """
        # default fit is "no fitting happens"
        return self

    @abstractmethod
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        private _fit_transform containing the core logic, called from fit_transform.

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation.

        Returns
        -------
        transformed version of X.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        self._fit(X, y)
        return self._transform(X, y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform.

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def _update(self, X, y=None):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator.
        """
        # standard behaviour: no update takes place, new data is ignored
        return self

    def _check_y(self, y, n_cases):
        if y is None:
            return None
        # Check y valid input for collection transformations
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        if isinstance(y, np.ndarray) and y.ndim > 1:
            raise TypeError(f"y must be 1-dimensional, found {y.ndim} dimensions")
        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                f"{n_labels}"
            )

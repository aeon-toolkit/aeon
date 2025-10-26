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

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "BaseCollectionTransformer",
]

from abc import abstractmethod
from typing import final

from aeon.base import BaseCollectionEstimator
from aeon.transformations.base import BaseTransformer
from aeon.utils.validation.collection import get_n_cases


class BaseCollectionTransformer(BaseCollectionEstimator, BaseTransformer):
    """Transformer base class for collections."""

    # default tag values for collection transformers
    _tags = {
        "input_data_type": "Collection",
        "output_data_type": "Collection",
        "removes_unequal_length": False,
    }

    @abstractmethod
    def __init__(self):
        super().__init__()

    @final
    def fit(self, X, y=None):
        """Fit transformer to X, optionally using y if supervised.

        Writes to self:
        - is_fitted : flag is set to True.
        - model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : np.ndarray or list
            Data to fit transform to, of valid collection type. Input data,
            any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)`` or list of numpy arrays (number
            of channels, series length) of shape ``[n_cases]``, 2D np.array
            ``(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is length of
            series ``i``. Other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability to handle.
        y : np.ndarray, default=None
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.
            If None, no labels are used in fitting.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self

        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        # input checks and datatype conversion
        X = self._preprocess_collection(X, store_metadata=True)
        if y is not None:
            self._check_y(y, n_cases=self.metadata_["n_cases"])

        self._fit(X=X, y=y)

        self.is_fitted = True
        return self

    @final
    def transform(self, X, y=None):
        """Transform X and return a transformed version.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True
        fitted model attributes (ending in "_") : must be set, accessed by _transform

        Parameters
        ----------
        X : np.ndarray or list
            Data to fit transform to, of valid collection type. Input data,
            any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)`` or list of numpy arrays (number
            of channels, series length) of shape ``[n_cases]``, 2D np.array
            ``(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is length of
            series ``i``. Other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability to handle.

        y : np.ndarray, default=None
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.
            If None, no labels are used in fitting.

        Returns
        -------
        transformed version of X
        """
        fit_empty = self.get_tag("fit_is_empty")
        if not fit_empty:
            self._check_is_fitted()

        # input checks and datatype conversion
        X = self._preprocess_collection(X, store_metadata=False)
        if y is not None:
            self._check_y(y, n_cases=get_n_cases(X))

        if not fit_empty:
            self._check_shape(X)

        Xt = self._transform(X, y)
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
        model attributes (ending in "_") : dependent on estimator.

        Parameters
        ----------
        X : np.ndarray or list
            Data to fit transform to, of valid collection type. Input data,
            any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)`` or list of numpy arrays (number
            of channels, series length) of shape ``[n_cases]``, 2D np.array
            ``(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is length of
            series ``i``. Other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability to handle.
        y : np.ndarray, default=None
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.
            If None, no labels are used in fitting.

        Returns
        -------
        transformed version of X
        """
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        # input checks and datatype conversion
        X = self._preprocess_collection(X, store_metadata=True)
        if y is not None:
            self._check_y(y, n_cases=self.metadata_["n_cases"])

        Xt = self._fit_transform(X=X, y=y)

        self.is_fitted = True
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
         fitted model attributes (ending in "_") : accessed by _inverse_transform

        Parameters
        ----------
        X : np.ndarray or list
            Data to fit transform to, of valid collection type. Input data,
            any number of channels, equal length series of shape ``(
            n_cases, n_channels, n_timepoints)`` or list of numpy arrays (number
            of channels, series length) of shape ``[n_cases]``, 2D np.array
            ``(n_channels, n_timepoints_i)``, where ``n_timepoints_i`` is length of
            series ``i``. Other types are allowed and converted into one of the above.

            Different estimators have different capabilities to handle different
            types of input. If ``self.get_tag("capability:multivariate")`` is False,
            they cannot handle multivariate series. If ``self.get_tag(
            "capability:unequal_length")`` is False, they cannot handle unequal
            length input. In both situations, a ``ValueError`` is raised if X has a
            characteristic that the estimator does not have the capability to handle.
        y : np.ndarray, default=None
            1D np.array of float or str, of shape ``(n_cases)`` - class labels
            (ground truth) for fitting indices corresponding to instance indices in X.
            If None, no labels are used in fitting.

        Returns
        -------
        inverse transformed version of X
            of the same type as X
        """
        if not self.get_tag("capability:inverse_transform"):
            raise NotImplementedError(
                f"{type(self)} does not implement inverse_transform"
            )

        # check whether is fitted
        self._check_is_fitted()

        # input check and conversion for X/y
        X_inner = self._preprocess_collection(X, store_metadata=False)
        y_inner = y

        Xt = self._inverse_transform(X=X_inner, y=y_inner)

        return Xt

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Input data
            Data to fit transform to, of valid collection type.
        y : Target variable, default=None
            Additional data, e.g., labels for transformation
        """
        pass

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
        ...

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        private _fit_transform containing the core logic, called from fit_transform.

        Non-optimized default implementation; override when a better
        method is possible for a given algorithm.

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
        if not self.get_tag("fit_is_empty"):
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

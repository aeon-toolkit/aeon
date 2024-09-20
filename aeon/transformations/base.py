"""
Base class template for transformers.

    class name: BaseTransformer

Covers all types of transformers.
Type and behaviour of transformer is determined by the following tags:
    "input_data_type" tag with values "Primitives" or "Series"
        this determines expected type of input of transform
        if "Primitives", expected inputs X are pd.DataFrame
        if "Series", expected inputs X are Series or Panel
        Note: placeholder tag for upwards compatibility currently only "Series" is
        supported
    "output_data_type" tag with values "Primitives", or "Series"
        this determines type of output of transform
        if "Primitives", output is pd.DataFrame with as many rows as X has instances
        i-th instance of X is transformed into i-th row of output
        if "Series", output is a Series or Panel, with as many instances as X i-th
        instance of X is transformed into i-th instance of output
        Series are treated as one-instance-Panels
        if Series is input, output is a 1-row pd.DataFrame or a Series
    "instancewise" tag which is boolean
        if True, fit/transform is statistically independent by instance

Class defining methods:
    fitting         - fit(self, X, y=None)
    transform       - transform(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)
    updating        - update(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__maintainer__ = []
__all__ = [
    "BaseTransformer",
]

from itertools import product
from typing import Union

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps

# single/multiple primitives
Primitive = Union[np.integer, int, float, str]
Primitives = np.ndarray

# tabular/cross-sectional data
Tabular = Union[pd.DataFrame, np.ndarray]  # 2d arrays

# univariate/multivariate series
UnivariateSeries = Union[pd.Series, np.ndarray]
MultivariateSeries = Union[pd.DataFrame, np.ndarray]
Series = Union[UnivariateSeries, MultivariateSeries]

# panel/longitudinal/series-as-features data
Panel = Union[pd.DataFrame, np.ndarray]  # 3d or nested array


def _coerce_to_list(obj):
    """Return [obj] if obj is not a list, otherwise obj."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class BaseTransformer(BaseEstimator):
    """Transformer base class."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "transform_labels": "None",
        "instancewise": True,
        "capability:multivariate": True,  # can the transformer handle multivariate X?
        "X_inner_type": "pd.DataFrame",
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_type": "None",
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "remember_data": False,  # whether all data seen is remembered as self._X
    }

    # allowed types for transformers - Series and Panel
    ALLOWED_INPUT_TYPES = [
        "pd.Series",
        "pd.DataFrame",
        "np.ndarray",
        "nested_univ",
        "numpy3D",
        # "numpy2D",
        "pd-multiindex",
        # "pd-wide",
        # "pd-long",
        "df-list",
        "np-list",
        "pd_multiindex_hier",
    ]

    def __init__(self, _output_convert="auto"):
        self._converter_store_X = dict()  # storage dictionary for in/output conversion
        self._output_convert = _output_convert

        super().__init__()
        _check_estimator_deps(self)

    def fit(self, X, y=None):
        """Fit transformer to X, optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
        possibly coerced to inner type or update_data compatible type
        by reference, when possible
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported type
            Data to fit transform to, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                nested pd.DataFrame, or pd.DataFrame in long/wide format.
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # input checks and datatype conversion
        self._fit(X=X, y=y)
        # this should happen last: fitted state is set to True
        self._is_fitted = True

        return self

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
        X : Series or Panel, any supported type
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and output_data_type tag:
            |          | `transform`  |                        |
            |   `X`    |  `-output`   |     type of return     |
            |__________|______________|________________________|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same type
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
        """
        # check whether is fitted
        self.check_is_fitted()

        Xt = self._transform(X=X, y=y)
        return Xt

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change: changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
            possibly coerced to inner type or update_data compatible type
            by reference, when possible
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported type
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and output_data_type tag:
            |   `X`    | `tf-output`  |     type of return     |
            |__________|______________|________________________|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same mtype
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
        """
        Xt = self._fit_transform(X=X, y=y)
        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        Currently it is assumed that only transformers with tags
            "input_data_type"="Series", "output_data_type"="Series",
        have an inverse_transform.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True
        _X : optionally accessed, only available if remember_data tag is True
        fitted model attributes (ending in "_") : accessed by _inverse_transform

        Parameters
        ----------
        X : Series or Panel, any supported type
            Data to be inverse transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X, and conforming to mtype format specifications
        """
        if self.get_tag("skip-inverse-transform"):
            return X

        if not self.get_tag("capability:inverse_transform"):
            raise NotImplementedError(
                f"{type(self)} does not implement inverse_transform"
            )

        # check whether is fitted
        self.check_is_fitted()
        Xt = self._inverse_transform(X=X, y=y)
        return Xt

    def update(self, X, y=None, update_params=True):
        """Update transformer with X, optionally y.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True
        _X : accessed by _update and by update_data, if remember_data tag is True
        fitted model attributes (ending in "_") : must be set, accessed by _update

        Writes to self:
        _X : updated by values in X, via update_data, if remember_data tag is True
        fitted model attributes (ending in "_") : only if update_params=True
            type and nature of update are dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported type
            Data to fit transform to, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
        y : Series or Panel, default=None
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

        self._update(X=X, y=y)

        return self

    def get_fitted_params(self, deep=True):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        deep : bool, default=True
            Whether to return fitted parameters of components.

            * If True, will return a dict of parameter name : value for this object,
              including fitted parameters of fittable components
              (= BaseEstimator-valued parameters).
            * If False, will return a dict of parameter name : value for this object,
              but not include fitted parameters of components.

        Returns
        -------
        fitted_params : dict with str-valued keys
            Dictionary of fitted parameters, paramname : paramvalue
            keys-value pairs include:

            * always: all fitted parameters of this object, as via `get_param_names`
              values are fitted parameter value for that key, of this object
            * if `deep=True`, also contains keys/value pairs of component parameters
              parameters of components are indexed as `[componentname]__[paramname]`
              all parameters of `componentname` appear as `paramname` with its value
            * if `deep=True`, also contains arbitrary levels of component recursion,
              e.g., `[componentname]__[componentcomponentname]__[paramname]`, etc
        """
        # if self is not vectorized, run the default get_fitted_params
        if not getattr(self, "_is_vectorized", False):
            return super().get_fitted_params(deep=deep)

        # otherwise, we delegate to the instances' get_fitted_params
        # instances' parameters are returned at dataframe-slice-like keys
        fitted_params = {}

        # transformers contains a pd.DataFrame with the individual transformers
        transformers = self.transformers_

        # return transformers in the "transformers" param
        fitted_params["transformers"] = transformers

        def _to_str(x):
            if isinstance(x, str):
                x = f"'{x}'"
            return str(x)

        # populate fitted_params with transformers and their parameters
        for ix, col in product(transformers.index, transformers.columns):
            trafo = transformers.loc[ix, col]
            trafo_key = f"transformers.loc[{_to_str(ix)},{_to_str(col)}]"
            fitted_params[trafo_key] = trafo
            trafo_params = trafo.get_fitted_params(deep=deep)
            for key, val in trafo_params.items():
                fitted_params[f"{trafo_key}__{key}"] = val

        return fitted_params

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X: data structure of type X_inner_type
            if X_inner_type is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of type y_inner_type, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator

        See extension_templates/transformer.py for implementation details.
        """
        # default fit is "no fitting happens"
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X: data structure of type X_inner_type
            if X_inner_type is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and output_data_type tag:
            |          | `transform`  |                        |
            |   `X`    |  `-output`   |     type of return     |
            |__________|______________|________________________|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        See extension_templates/transformer.py for implementation details.
        """
        raise NotImplementedError("abstract method")

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        private _fit_transform containing the core logic, called from fit_transform

        Parameters
        ----------
        X: data structure of type X_inner_type
            if X_inner_type is list, _fit_transform must support all types in it
            Data to fit transform to
        y : Series or Panel of type y_inner_type, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator

        See extension_templates/transformer.py for implementation details.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        self._fit(X, y)
        return self._transform(X, y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X: data structure of type X_inner_type
            if X_inner_type is list, _inverse_transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X, and conforming to type format specifications

        See extension_templates/transformer.py for implementation details.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, y=None):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X: data structure of type X_inner_type
            if X_inner_type is list, _update must support all types in it
            Data to update transformer with
        y : Series or Panel of type y_inner_type, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator

        See extension_templates/transformer.py for implementation details.
        """
        # standard behaviour: no update takes place, new data is ignored
        return self

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

from typing import Union

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator
from aeon.utils.conversion import convert_collection, convert_series
from aeon.utils.index_functions import update_data
from aeon.utils.sklearn import (
    is_sklearn_classifier,
    is_sklearn_regressor,
    is_sklearn_transformer,
)
from aeon.utils.validation import validate_input
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

    def __mul__(self, other):
        """Magic * method, return (right) concatenated TransformerPipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-TransformerPipeline `aeon` transformers
        """
        from aeon.transformations.compose import TransformerPipeline

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformerPipeline does the rest, e.g., case distinctions on other
        if (
            isinstance(other, BaseTransformer)
            or is_sklearn_classifier(other)
            or is_sklearn_regressor(other)
            or is_sklearn_transformer(other)
        ):
            self_as_pipeline = TransformerPipeline(steps=[self])
            return self_as_pipeline * other
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated TransformerPipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `other` (first) with `self` (last).
            not nested, contains only non-TransformerPipeline `aeon` transformers
        """
        from aeon.transformations.compose import TransformerPipeline

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformerPipeline does the rest, e.g., case distinctions on other
        if isinstance(other, BaseTransformer) or is_sklearn_transformer(other):
            self_as_pipeline = TransformerPipeline(steps=[self])
            return other * self_as_pipeline
        else:
            return NotImplemented

    def __or__(self, other):
        """Magic | method, return MultiplexTranformer.

        Implemented for `other` being either a MultiplexTransformer or a transformer.

        Parameters
        ----------
        other: `aeon` transformer or aeon MultiplexTransformer

        Returns
        -------
        MultiplexTransformer object
        """
        from aeon.transformations.compose import MultiplexTransformer

        if isinstance(other, BaseTransformer):
            multiplex_self = MultiplexTransformer([self])
            return multiplex_self | other
        else:
            return NotImplemented

    def __add__(self, other):
        """Magic + method, return (right) concatenated FeatureUnion.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        FeatureUnion object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-TransformerPipeline `aeon` transformers
        """
        from aeon.transformations.compose import FeatureUnion

        # we wrap self in a pipeline, and concatenate with the other
        #   the FeatureUnion does the rest, e.g., case distinctions on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = FeatureUnion(transformer_list=[self])
            return self_as_pipeline + other
        else:
            return NotImplemented

    def __radd__(self, other):
        """Magic + method, return (left) concatenated FeatureUnion.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        FeatureUnion object, concatenation of `other` (first) with `self` (last).
            not nested, contains only non-FeatureUnion `aeon` transformers
        """
        from aeon.transformations.compose import FeatureUnion

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformerPipeline does the rest, e.g., case distinctions on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = FeatureUnion(transformer_list=[self])
            return other + self_as_pipeline
        else:
            return NotImplemented

    def __invert__(self):
        """Magic unary ~ (inversion) method, return InvertTransform of self.

        Returns
        -------
        `InvertTransform` object, containing `self`.
        """
        from aeon.transformations.compose import InvertTransform

        return InvertTransform(self)

    def __neg__(self):
        """Magic unary - (negation) method, return OptionalPassthrough of self.

        Intuition: `OptionalPassthrough` is "not having transformer", as an option.

        Returns
        -------
        `OptionalPassthrough` object, containing `self`, with `passthrough=False`.
            The `passthrough` parameter can be set via `set_params`.
        """
        from aeon.transformations.compose import OptionalPassthrough

        return OptionalPassthrough(self, passthrough=False)

    def __getitem__(self, key):
        """Magic [...] method, return column subsetted transformer.

        First index does intput subsetting, second index does output subsetting.

        Keys must be valid inputs for `columns` in `ColumnSubset`.

        Parameters
        ----------
        key: valid input for `columns` in `ColumnSubset`, or pair thereof
            keys can also be a :-slice, in which case it is considered as not passed

        Returns
        -------
        the following TransformerPipeline object:
            ColumnSubset(columns1) * self * ColumnSubset(columns2)
            where `columns1` is first or only item in `key`, and `columns2` is the last
            if only one item is passed in `key`, only `columns1` is applied to input
        """
        from aeon.transformations.subset import ColumnSelect

        def is_noneslice(obj):
            res = isinstance(obj, slice)
            res = res and obj.start is None and obj.stop is None and obj.step is None
            return res

        if isinstance(key, tuple):
            if not len(key) == 2:
                raise ValueError(
                    "there should be one or two keys when calling [] or getitem, "
                    "e.g., mytrafo[key], or mytrafo[key1, key2]"
                )
            columns1 = key[0]
            columns2 = key[1]
            if is_noneslice(columns1) and is_noneslice(columns2):
                return self
            elif is_noneslice(columns2):
                return ColumnSelect(columns1) * self
            elif is_noneslice(columns1):
                return self * ColumnSelect(columns2)
            else:
                return ColumnSelect(columns1) * self * ColumnSelect(columns2)
        else:
            return ColumnSelect(key) * self

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
        y : Series or Collection, default=None
            Additional data, e.g., labels for transformation.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # input checks and datatype conversion
        X_inner, y_inner = self._fit_checks(X, y)

        # skip the rest if fit_is_empty is True
        if self.get_tag("fit_is_empty"):
            self._is_fitted = True
            return self

        self._fit(X=X_inner, y=y_inner)
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
        """
        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        Xt = self._transform(X=X_inner, y=y_inner)

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
        """
        # input checks and datatype conversion
        X_inner, y_inner, metadata = self._fit_checks(X, y, False, True)

        # checks and conversions complete, pass to inner fit_transform
        ####################################################
        Xt = self._fit_transform(X=X_inner, y=y_inner)
        self._is_fitted = True

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

        # input check and conversion for X/y
        X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)
        Xt = self._inverse_transform(X=X_inner, y=y_inner)

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

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update memory of X, if remember_data exists and is set to True
        if self.get_tag("remember_data", tag_value_default=False):
            self._X = update_data(None, X_new=X_inner)

        # skip everything if update_params is False
        # skip everything if fit_is_empty is True
        if not update_params or self.get_tag("fit_is_empty", False):
            return self

        # checks and conversions complete, pass to inner fit
        #####################################################
        self._update(X=X_inner, y=y_inner)

        return self

    def _check_X_y(self, X=None, y=None, return_metadata=False):
        """Check and coerce X/y for fit/transform functions.

        Parameters
        ----------
        X : object of aeon compatible time series type
            can be Series, Panel, Hierarchical
        y : None (default), or object of aeon compatible time series type
            can be Series, Panel, Hierarchical
        return_metadata : bool, optional, default=False
            whether to return the metadata return object

        Returns
        -------
        X_inner : Series, Collection, or Hierarchical object
                compatible with self.get_tag("X_inner_type") format
        y_inner : Series, Collection, or Hierarchical object
                compatible with self.get_tag("y_inner_type") format

        metadata : dict, returned only if return_metadata=True

        Raises
        ------
        TypeError if X is None
        TypeError if X or y is not one of the permissible Series types
        TypeError if X is incompatible with self.get_tag("capability:multivariate")
            if tag value is "False", X must be univariate
        ValueError if self.get_tag("requires_y")=True but y is None
        """
        if X is None:
            raise TypeError("X cannot be None, but found None")

        # retrieve supported mtypes
        X_inner_type = _coerce_to_list(self.get_tag("X_inner_type"))
        y_inner_type = _coerce_to_list(self.get_tag("y_inner_type"))

        valid, X_metadata = validate_input(X)
        if not valid:
            raise TypeError(
                "must be in an aeon compatible format for storing series, hierarchical "
                "series or collections of series."
            )

        X_inner_type = X_metadata["mtype"]
        if X_inner_type not in self.ALLOWED_INPUT_TYPES:
            raise TypeError("X an invalid internal type")

        # checking X vs tags
        if not X_metadata["is_univariate"]:
            if not self.get_tag("capability:multivariate"):
                raise TypeError("X is multivariate, estimator cannot handle it")
        if X_metadata["scitype"] == "Series":
            X_inner = convert_series(X, output_type=X_inner_type)
        elif X_metadata["scitype"] == "Collection":
            X_inner = convert_collection(X, output_type=X_inner_type)
        y_inner = None
        if y_inner_type != ["None"] and y is not None:
            valid, y_metadata = validate_input(y)
            if not valid:
                raise TypeError("Error, y is not a valid type for X type.")
            if y_metadata["scitype"] == "Series":
                y_inner = convert_series(y, output_type=y_inner_type)
            elif y_metadata["scitype"] == "Collection":
                y_inner = convert_collection(X, output_type=y_inner_type)

        if return_metadata:
            return X_inner, y_inner, X_metadata
        else:
            return X_inner, y_inner

    def _check_X(self, X=None):
        """Shorthand for _check_X_y with one argument X, see _check_X_y."""
        return self._check_X_y(X=X)[0]

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
        y : Series or Collection of type y_inner_type, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator
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
        y : Series or Collection, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X, and conforming to type format specifications
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
        y : Series or Collection of type y_inner_type, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        # standard behaviour: no update takes place, new data is ignored
        return self

    def _fit_checks(self, X, y, early_abandon=True, return_metadata=False):
        """Input checks and conversions for fit and fit_transform."""
        self.reset()

        X_inner = None
        y_inner = None
        metadata = None

        # skip everything if fit_is_empty is True and we do not need to remember data
        if (
            not early_abandon
            or not self.get_tag("fit_is_empty")
            or self.get_tag("remember_data", False)
        ):
            # if requires_y is set, y is required in fit and update
            if self.get_tag("requires_y") and y is None:
                raise ValueError(f"{self.__class__.__name__} requires `y` in `fit`.")

            # check and convert X/y
            X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

            # memorize X as self._X, if remember_data tag is set to True
            if self.get_tag("remember_data", False):
                self._X = update_data(None, X_new=X_inner)

        if return_metadata:
            return X_inner, y_inner, metadata
        else:
            return X_inner, y_inner

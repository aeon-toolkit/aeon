# -*- coding: utf-8 -*-
"""
Base class template for panel transformers.

    class name: BasePanelTransformer

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

__author__ = ["mloning", "fkiraly", "miraep8", "MatthewMiddlehurst"]
__all__ = [
    "BasePanelTransformer",
]

from abc import ABCMeta, abstractmethod

import pandas as pd

from aeon.datatypes import (
    check_is_mtype,
    check_is_scitype,
    convert_to,
    mtype_to_scitype,
    update_data,
)
from aeon.transformations.base import BaseTransformer, _coerce_to_list
from aeon.utils.validation._dependencies import _check_estimator_deps


class BasePanelTransformer(BaseTransformer, metaclass=ABCMeta):
    """Transformer base class."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "scitype:transform-input": "Panel",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Panel",
        # what scitype is returned: Primitives, Series, Panel
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
    }

    # allowed mtypes for transformers - Series and Panel
    ALLOWED_INPUT_MTYPES = [
        "nested_univ",
        "numpy3D",
        "numpyflat",
        "pd-multiindex",
        "df-list",
    ]

    def __init__(self, _output_convert="auto"):
        self._output_convert = _output_convert

        super(BasePanelTransformer, self).__init__()
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
        X : Series or Panel, any supported mtype
            Data to fit transform to, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to aeon mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

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
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to aeon mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |          | `transform`  |                        |
            |   `X`    |  `-output`   |     type of return     |
            |----------|--------------|------------------------|
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
        # check whether is fitted
        self.check_is_fitted()

        # input check and conversion for X/y
        X_inner, y_inner, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        Xt = self._transform(X=X_inner, y=y_inner)

        # convert to output mtype
        if self._output_convert == "auto":
            Xt = self._convert_output(Xt, metadata=metadata)

        return Xt

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

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
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to aeon mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
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
        # input checks and datatype conversion
        X_inner, y_inner, metadata = self._fit_checks(X, y, False, True)

        Xt = self._fit_transform(X=X_inner, y=y_inner)

        self._is_fitted = True

        # convert to output mtype
        if self._output_convert == "auto":
            Xt = self._convert_output(Xt, metadata=metadata)

        return Xt

    def inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        Currently it is assumed that only transformers with tags
            "scitype:transform-input"="Series", "scitype:transform-output"="Series",
        have an inverse_transform.

        State required:
            Requires state to be "fitted".

        Accesses in self:
        _is_fitted : must be True
        _X : optionally accessed, only available if remember_data tag is True
        fitted model attributes (ending in "_") : accessed by _inverse_transform

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be inverse transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to aeon mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
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

        # convert to output mtype
        if self._output_convert == "auto":
            Xt = self._convert_output(Xt, metadata=metadata, inverse=True)

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
        X : Series or Panel, any supported mtype
            Data to fit transform to, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to aeon mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
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

        # update memory of X, if remember_data tag is set to True
        if self.get_tag("remember_data", False):
            self._X = update_data(None, X_new=X_inner)

        # skip everything if update_params is False
        # skip everything if fit_is_empty is True
        if not update_params or self.get_tag("fit_is_empty", False):
            return self

        self._update(X=X_inner, y=y_inner)

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
        X_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("X_inner_mtype") format
            Case 1: self.get_tag("X_inner_mtype") supports scitype of X, then
                converted/coerced version of X, mtype determined by "X_inner_mtype" tag
            Case 2: self.get_tag("X_inner_mtype") supports *higher* scitype than X
                then X converted to "one-Series" or "one-Panel" sub-case of that scitype
                always pd-multiindex (Panel) or pd_multiindex_hier (Hierarchical)
            Case 3: self.get_tag("X_inner_mtype") supports only *simpler* scitype than X
                then VectorizedDF of X, iterated as the most complex supported scitype
        y_inner : Series, Panel, or Hierarchical object, or VectorizedDF
                compatible with self.get_tag("y_inner_mtype") format
            Case 1: self.get_tag("y_inner_mtype") supports scitype of y, then
                converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            Case 2: self.get_tag("y_inner_mtype") supports *higher* scitype than y
                then X converted to "one-Series" or "one-Panel" sub-case of that scitype
                always pd-multiindex (Panel) or pd_multiindex_hier (Hierarchical)
            Case 3: self.get_tag("y_inner_mtype") supports only *simpler* scitype than y
                then VectorizedDF of X, iterated as the most complex supported scitype
            Case 4: None if y was None, or self.get_tag("y_inner_mtype") is "None"

            Complexity order above: Hierarchical > Panel > Series

        metadata : dict, returned only if return_metadata=True
            dictionary with str keys, contents as follows
            _converter_store_X : dict, metadata from X conversion, for back-conversion
            _X_mtype_last_seen : str, mtype of X seen last
            _X_input_scitype : str, scitype of X seen last
            _convert_case : str, coversion case (see above), one of
                "case 1: scitype supported"
                "case 2: higher scitype supported"
                "case 3: requires vectorization"

        Raises
        ------
        TypeError if X is None
        TypeError if X or y is not one of the permissible Series mtypes
        TypeError if X is not compatible with self.get_tag("univariate_only")
            if tag value is "True", X must be univariate
        ValueError if self.get_tag("requires_y")=True but y is None
        """
        if X is None:
            raise TypeError("X cannot be None, but found None")

        metadata = dict()
        metadata["_converter_store_X"] = dict()

        # retrieve supported mtypes
        X_inner_mtype = _coerce_to_list(self.get_tag("X_inner_mtype"))
        y_inner_mtype = _coerce_to_list(self.get_tag("y_inner_mtype"))
        y_inner_scitype = mtype_to_scitype(y_inner_mtype, return_unique=True)

        # checking X
        X_valid, msg, X_metadata = check_is_scitype(
            X,
            scitype="Panel",
            return_metadata=True,
            var_name="X",
        )

        if not X_valid:
            raise TypeError("invalid input type for X")

        X_scitype = X_metadata["scitype"]
        X_mtype = X_metadata["mtype"]
        # remember these for potential back-conversion (in transform etc)
        metadata["_X_mtype_last_seen"] = X_mtype
        metadata["_X_input_scitype"] = X_scitype

        if X_mtype not in self.ALLOWED_INPUT_MTYPES:
            raise TypeError("invalid input mtype for X")

        # check if univariate-only
        if self.get_tag("univariate-only") and not X_metadata["is_univariate"]:
            raise TypeError("X must be univariate, but found multivariate")

        # checking y
        if y_inner_mtype != ["None"] and y is not None:
            if "Table" in y_inner_scitype:
                y_possible_scitypes = "Table"
            elif X_scitype == "Series":
                y_possible_scitypes = "Series"
            elif X_scitype == "Panel":
                y_possible_scitypes = "Panel"
            elif X_scitype == "Hierarchical":
                y_possible_scitypes = ["Panel", "Hierarchical"]

            y_valid, _, y_metadata = check_is_scitype(
                y, scitype=y_possible_scitypes, return_metadata=True, var_name="y"
            )
            if not y_valid:
                raise TypeError("invalid input mtype for y")

            y_scitype = y_metadata["scitype"]
        else:
            # y_scitype is used below - set to None if y is None
            y_scitype = None

        X_inner = convert_to(
            X,
            to_type=X_inner_mtype,
            store=metadata["_converter_store_X"],
            store_behaviour="reset",
        )

        # converts y, returns None if y is None
        if y_inner_mtype != ["None"] and y is not None:
            y_inner = convert_to(
                y,
                to_type=y_inner_mtype,
                as_scitype=y_scitype,
            )
        else:
            y_inner = None

        if return_metadata:
            return X_inner, y_inner, metadata
        else:
            return X_inner, y_inner

    def _check_X(self, X=None):
        """Shorthand for _check_X_y with one argument X, see _check_X_y."""
        return self._check_X_y(X=X)[0]

    def _convert_output(self, X, metadata, inverse=False):
        """Convert transform or inverse_transform output to expected format.

        Parameters
        ----------
        X : output of _transform or _vectorize("transform"), or inverse variants
        metadata : dict, output of _check_X_y
        inverse : bool, optional, default = False
            whether conversion is for transform (False) or inverse_transform (True)

        Returns
        -------
        Xt : final output of transform or inverse_transform
        """
        Xt = X
        X_input_mtype = metadata["_X_mtype_last_seen"]
        X_input_scitype = metadata["_X_input_scitype"]
        _converter_store_X = metadata["_converter_store_X"]

        if inverse:
            # the output of inverse transform is equal to input of transform
            output_scitype = self.get_tag("scitype:transform-input")
        else:
            output_scitype = self.get_tag("scitype:transform-output")

        # now, in all cases, Xt is in the right scitype,
        #   but not necessarily in the right mtype.
        # additionally, Primitives may have an extra column

        #   "case 1: scitype supported"
        #   "case 2: higher scitype supported"
        #   "case 3: requires vectorization"

        if output_scitype == "Series":
            # output mtype is input mtype
            X_output_mtype = X_input_mtype

            # exception to this: if the transformer outputs multivariate series,
            #   we cannot convert back to pd.Series, do pd.DataFrame instead then
            #   this happens only for Series, not Panel
            if X_input_scitype == "Series":
                valid, msg, metadata = check_is_mtype(
                    Xt,
                    ["pd.DataFrame", "pd.Series", "np.ndarray"],
                    return_metadata=True,
                )
                if not valid:
                    raise TypeError(
                        f"_transform output of {type(self)} does not comply "
                        "with aeon mtype specifications. See datatypes.MTYPE_REGISTER"
                        " for mtype specifications. Returned error message:"
                        f" {msg}. Returned object: {Xt}"
                    )
                if not metadata["is_univariate"] and X_input_mtype == "pd.Series":
                    X_output_mtype = "pd.DataFrame"

            Xt = convert_to(
                Xt,
                to_type=X_output_mtype,
                as_scitype=X_input_scitype,
                store=_converter_store_X,
                store_behaviour="freeze",
            )
        elif output_scitype == "Primitives":
            # we ensure the output is pd_DataFrame_Table
            # & ensure the returned index is sensible
            # for return index, we need to deal with last level, constant 0
            if isinstance(Xt, (pd.DataFrame, pd.Series)):
                # if index is multiindex, last level is constant 0
                # and other levels are hierarchy
                if isinstance(Xt.index, pd.MultiIndex):
                    Xt.index = Xt.index.droplevel(-1)
                # else this is only zeros and should be reset to RangeIndex
                else:
                    Xt = Xt.reset_index(drop=True)
            Xt = convert_to(
                Xt,
                to_type="pd_DataFrame_Table",
                as_scitype="Table",
                # no converter store since this is not a "1:1 back-conversion"
            )
        # else output_scitype is "Panel" and no need for conversion

        return Xt

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

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
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |          | `transform`  |                        |
            |   `X`    |  `-output`   |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        See extension_templates/transformer.py for implementation details.
        """

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        private _fit_transform containing the core logic, called from fit_transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit_transform must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
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
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
            of the same type as X, and conforming to mtype format specifications

        See extension_templates/transformer.py for implementation details.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def _update(self, X, y=None):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _update must support all types in it
            Data to update transformer with
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator

        See extension_templates/transformer.py for implementation details.
        """
        # standard behaviour: no update takes place, new data is ignored
        return self

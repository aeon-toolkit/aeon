"""Transformers for index and column subsetting."""

__maintainer__ = []

import pandas as pd
from pandas.api.types import is_integer_dtype

from aeon.transformations.base import BaseTransformer


class _ColumnSelect(BaseTransformer):
    r"""Column selection transformer.

    In transform, subsets `X` to `columns` provided as hyper-parameters.

    Sequence of columns in `Xt=transform(X)` is as in `columns` hyper-parameter.
    Caveat: this means that `transform` may change sequence of columns,
        even if no columns are removed from `X` in `transform(X)`.

    Parameters
    ----------
    columns : pandas compatible index or index coercible, optional, default = None
        columns to which X in transform is to be subset
    integer_treatment : str, optional, one of "col" (default) and "coerce"
        determines how integer index columns are treated
        "col" = subsets by column iloc index, even if columns is not in X.columns
        "coerce" = coerces to integer pandas.Index and attempts to subset
    index_treatment : str, optional, one of "remove" (default) or "keep"
        determines which column are kept in `Xt = transform(X, y)`
        "remove" = only indices that appear in both X and columns are present in Xt.
        "keep" = all indices in columns appear in Xt. If not present in X, NA is filled.
    """

    _tags = {
        "input_data_type": "Series",
        # what is the abstract type of X: Series, or Panel
        "output_data_type": "Series",
        # what abstract type is returned: Primitives, Series, Panel
        "instancewise": True,  # is this an instance-wise transform?
        "X_inner_type": "pd.DataFrame",
        "y_inner_type": "None",
        "transform-returns-same-time-index": True,
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:inverse_transform": False,
    }

    def __init__(self, columns=None, integer_treatment="col", index_treatment="remove"):
        self.columns = columns
        self.integer_treatment = integer_treatment
        self.index_treatment = index_treatment
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : Ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.DataFrame
            transformed version of X
        """
        columns = self.columns
        integer_treatment = self.integer_treatment
        index_treatment = self.index_treatment

        if columns is None:
            return X
        if pd.api.types.is_scalar(columns):
            columns = [columns]

        columns = pd.Index(columns)

        if integer_treatment == "col" and is_integer_dtype(columns):
            columns = [x for x in columns if x < len(X.columns)]
            col_idx = X.columns[columns]
            return X[col_idx]

        in_cols = columns.isin(X.columns)
        col_X_and_cols = columns[in_cols]

        if index_treatment == "remove":
            Xt = X[col_X_and_cols]
        elif index_treatment == "keep":
            Xt = X[col_X_and_cols]
            X_idx_frame = type(X)(columns=columns)
            Xt = Xt.combine_first(X_idx_frame)
            Xt = Xt[columns]
        else:
            raise ValueError(
                f'index_treatment must be one of "remove", "keep", but found'
                f' "{index_treatment}"'
            )
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"columns": None}
        params2 = {"columns": [0, 2, 3]}
        params3 = {"columns": ["a", "foo", "bar"], "index_treatment": "keep"}
        params4 = {"columns": "a", "index_treatment": "keep"}

        return [params1, params2, params3, params4]

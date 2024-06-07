"""Implements feature selection algorithms."""

__maintainer__ = []
__all__ = ["FeatureSelection"]

import math

from sklearn.ensemble import GradientBoostingRegressor

from aeon.transformations.series.base import BaseSeriesTransformer


class FeatureSelection(BaseSeriesTransformer):
    """
    Select exogenous features.

    Transformer to enable tuneable feauture selection of exogenous data. The
    FeatureSelection implements multiple methods to select features (columns).
    In case X is a pd.Series, then it is just passed through, unless method="none",
    then None is returned in transform().

    Parameters
    ----------
    method : str
        The method of how to select the features. Implemeted methods are:
        * "feature-importances": Use feature_importances_ of the regressor (meta-model)
          to select n_columns with highest importance values.
          Requires parameter n_columns.
        * "random": Randomly select n_columns features. Requires parameter n_columns.
        * "columns": Select features by given names.
        * "none": Remove all columns by setting Z to None.
        * "all": Select all given features.
    n_columns : int, default = None
        Number of feautres (columns) to select. n_columns must be <=
        number of X columns. Some methods require n_columns to be given.
    regressor : sklearn-like regressor, default=None
        Used as meta-model for the method "feature-importances". The given
        regressor must have an attribute "feature_importances_". If None,
        then a GradientBoostingRegressor(max_depth=5) is used.
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor and to
        set random.seed() if method="random".
    columns : list of str
        A list of columns to select. If columns is given.

    Attributes
    ----------
    columns_ : list of str
        List of columns that have been selected as features.
    regressor_ : sklearn-like regressor
        Fitted regressor (meta-model).
    n_columns_: int
        Derived from number of features if n_columns is None, then
        n_columns_ is calculated as int(math.ceil(Z.shape[1] / 2)). So taking
        half of given features only as default.
    feature_importances_ : dict or None
        A dictionary with column name as key and feature imporatnce value as value.
        The dict is sorted descending on value. This attribute is a dict if
        method="feature-importances", else None.

    Examples
    --------
    >>> from aeon.transformations.series._feature_selection import FeatureSelection
    >>> from aeon.datasets import load_longley
    >>> y, X = load_longley()
    >>> transformer = FeatureSelection(method="feature-importances", n_columns=3)
    >>> X_hat = transformer.fit_transform(X, y)
    """

    _tags = {
        "X_inner_type": "pd.DataFrame",
        "fit_is_empty": False,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        method="feature-importances",
        n_columns=None,
        regressor=None,
        random_state=None,
        columns=None,
    ):
        self.n_columns = n_columns
        self.method = method
        self.regressor = regressor
        if regressor is not None:
            if not getattr(regressor, "_estimator_type", None) == "regressor":
                raise ValueError(
                    f"`regressor` should be a sklearn-like regressor, "
                    f"but found: {regressor}"
                )

        self.random_state = random_state
        self.columns = columns

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit transform to
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        if self.n_columns is None:
            self.n_columns_ = int(math.ceil(X.shape[1] / 2))
        else:
            self.n_columns_ = self.n_columns
        self.feature_importances_ = None

        if self.method == "none":
            self.set_tags(**{"output_data_type": "Primitives"})

        # Only do this if multivariate X
        if len(X.columns) > 1:
            if self.method == "feature-importances":
                if y is None:
                    raise ValueError(
                        "y must be passed if method is 'feature-importances'."
                    )
                if self.regressor is None:
                    self.regressor_ = GradientBoostingRegressor(max_depth=5)
                else:
                    self.regressor_ = self.regressor  # CHANGE
                # fit regressor with X as exog data and y as endog data (target)
                self.regressor_.fit(X=X, y=y)
                if not hasattr(self.regressor_, "feature_importances_"):
                    raise ValueError(
                        """The given regressor must have an
                        attribute feature_importances_ after fitting."""
                    )
                # create dict with columns name (key) and feauter importance (value)
                d = dict(zip(X.columns, self.regressor_.feature_importances_))
                # sort d descending
                d = {k: d[k] for k in sorted(d, key=d.get, reverse=True)}
                self.feature_importances_ = d
                self.columns_ = list(d.keys())[: self.n_columns_]
            elif self.method == "random":
                self._check_n_columns(X)
                self.columns_ = list(
                    X.sample(
                        n=self.n_columns_, random_state=self.random_state, axis=1
                    ).columns
                )
            elif self.method == "columns":
                if self.columns is None:
                    raise AttributeError("Parameter columns must be given.")
                self.columns_ = self.columns
            elif self.method == "none":
                self.columns_ = None
            elif self.method == "all":
                self.columns_ = list(X.columns)
            else:
                raise ValueError("Incorrect method given. Try another method.")
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        # multivariate case
        if len(X.columns) > 1:
            if self.method == "none":
                Xt = None
            else:
                Xt = X[self.columns_]
        # univariate case
        else:
            if self.method == "none":
                Xt = None
            else:
                Xt = X
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"method": "all"}

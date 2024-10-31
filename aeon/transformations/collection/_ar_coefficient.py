"""AR coefficient feature transformer."""

__maintainer__ = []
__all__ = ["ARCoefficientTransformer"]


import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class ARCoefficientTransformer(BaseCollectionTransformer):
    """Autoreggression coefficient feature transformer.

    Coefficients of an autoregressive model using Burg's method. The Burg method
    fits a forward-backward autoregressive model to the data using least squares
    regression.

    Parameters
    ----------
    order : int or callable, default=100
        The order of the autoregression. If callable, the function should take a 3D
        numpy array of shape (n_cases, n_channels, n_timepoints) and return an
        integer.
    min_values : int, default=0
        Always transform at least this many values unless the series length is too
        short. This will reduce order if needed.
    replace_nan : bool, default=False
        If True, replace NaNs in output with 0s.

    Examples
    --------
    >>> from aeon.transformations.collection import ARCoefficientTransformer
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X = make_example_3d_numpy(n_cases=4, n_channels=2, n_timepoints=20,
    ...                           random_state=0, return_y=False)
    >>> tnf = ARCoefficientTransformer(order=5)  # doctest: +SKIP
    >>> tnf.fit(X)  # doctest: +SKIP
    ARCoefficientTransformer(...)
    >>> print(tnf.transform(X)[0])  # doctest: +SKIP
    [[ 0.05445952 -0.02106654 -0.24989205 -0.19153596  0.08833235]
     [-0.13034384  0.16255828 -0.27993791 -0.06842601 -0.01382752]]
    """

    _tags = {
        "capability:multivariate": True,
        "python_dependencies": "statsmodels",
        "fit_is_empty": True,
    }

    def __init__(
        self,
        order=100,
        min_values=0,
        replace_nan=False,
    ):
        self.order = order
        self.min_values = min_values
        self.replace_nan = replace_nan

        super().__init__()

    def _transform(self, X, y=None):
        from statsmodels.regression.linear_model import burg

        n_cases, n_channels, n_timepoints = X.shape

        order = self.order(X) if callable(self.order) else self.order
        if order > n_timepoints - self.min_values:
            order = n_timepoints - self.min_values
        if order <= 0:
            order = 1

        if order > n_timepoints - 1:
            raise ValueError(
                f"order ({order}) must be smaller than n_timepoints - 1 "
                f"({n_timepoints - 1})."
            )

        Xt = np.zeros((n_cases, n_channels, order))
        for i in range(n_cases):
            for n in range(n_channels):
                coefs, _ = burg(X[i, n], order=order)
                Xt[i, n] = coefs

        if self.replace_nan:
            Xt[np.isnan(Xt)] = 0

        return Xt

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
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
        """
        return {
            "order": 4,
        }

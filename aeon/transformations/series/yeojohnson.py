"""Implemenents Yeo-Johnson Transformation."""

__maintainer__ = []
__all__ = ["YeoJohnsonTransformer"]

import numpy as np
from numba import njit
from scipy.stats import yeojohnson_normmax

from aeon.transformations.series.base import BaseSeriesTransformer


class YeoJohnsonTransformer(BaseSeriesTransformer):
    r"""Yeo-Johnson power transform.

    Yeo-Johnson transformation is related to the BoxCox transformation, and is
    a power transformation that is used to make data more normally distributed
    and stabilize its variance based on the hyperparameter lambda. [1]_

    The YeoJohnsonTransformer solves for the lambda parameter used in the
    Yeo-Johnson transformation using maximum likelihood estimation,
    on input data provided to `fit`.

    Parameters
    ----------
    bounds : tuple
        Lower and upper bounds used to restrict the feasible range
        when solving for the value of lambda.
    lambda_ : float
        The Yeo-Johnson lambda parameter. If not supplied, it is solved for based
        on the data provided in `fit`.

    Attributes
    ----------
    bounds : tuple
        Lower and upper bounds used to restrict the feasible range when
        solving for lambda.
    lambda_ : float
        The Yeo-Johnson lambda parameter that was solved for based on the supplied
        `method` and data provided in `fit`.

    See Also
    --------
    aeon.transformations.boxcox.BoxCoxTransformer :
        Transform input data by using the Box-Cox power transform. Used to
        make data more normally distributed and stabilize its variance based
        on the hyperparameter lambda.
    aeon.transformations.boxcox.LogTransformer :
        Transform input data using natural log. Can help normalize data and
        compress variance of the series.
    aeon.transformations.exponent.ExponentTransformer :
        Transform input data by raising it to an exponent. Can help compress
        variance of series if a fractional exponent is supplied.
    aeon.transformations.exponent.SqrtTransformer :
        Transform input data by taking its square root. Can help compress
        variance of input series.

    Notes
    -----
    The Yeo-Johnson transformation is defined as :math:`\[
    \phi(\lambda,y) =
    \begin{cases}
    log(y+1) & \text{if $\lambda=0, y\geq0$} \\
    \frac{(y+1)^\lambda - 1}{\lambda} & \text{if $\lambda\neq0, y\geq0$} \\
    -log(1-y) & \text{if $\lambda=2, y<0$} \\
    -\frac{(1-y)^{2-\lambda}-1)}{2-\lambda} & \text{if $\lambda\neq2, y<0$}
    \end{cases}
    \]`.

    References
    ----------
    .. [1] Yeo and R.A. Johnson, “A New Family of Power Transformations to
        Improve Normality or Symmetry”, Biometrika 87.4 (2000).

    Examples
    --------
    >>> from aeon.transformations.series.yeojohnson import YeoJohnsonTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = YeoJohnsonTransformer()
    >>> y_hat = transformer.fit_torm(y)
    """

    # tag values specific to SeriesTransformers
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "instancewise": True,  # is this an instance-wise transform?
        "X_inner_type": "ndarray",
        "y_inner_type": "None",
        "transform-returns-same-time-index": True,
        "fit_is_empty": False,
        "univariate-only": True,
        "requires_y": False,
        "capability:inverse_transform": False,
    }

    def __init__(self, lambda_=None, bounds=None):
        self.bounds = bounds
        self.lambda_ = lambda_
        super().__init__(axis=1)

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : 2D np.ndarray (n x 1)
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        if self.lambda_ is None:
            X = X.flatten()
            self.lambda_ = yeojohnson_normmax(X, brack=self.bounds)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray (n x 1)
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        return _yeo_johnson_transform(X, self.lambda_)


@njit
def _yeo_johnson_transform(X, lambda_):
    X_shape = X.shape
    Xt = X.flatten()
    X_gte_0 = Xt >= 0
    X_lt_0 = Xt < 0
    if lambda_ != 0:
        Xt[X_gte_0] = (np.power(Xt[X_gte_0] + 1, lambda_) - 1) / lambda_
    elif lambda_ == 0:
        Xt[X_gte_0] = np.log(Xt[X_gte_0] + 1)
    if lambda_ != 2:
        Xt[X_lt_0] = -(np.power(-Xt[X_lt_0] + 1, 2 - lambda_) - 1) / (2 - lambda_)
    elif lambda_ == 2:
        Xt[X_lt_0] = -np.log(-Xt[X_lt_0] + 1)
    Xt = Xt.reshape(X_shape)
    return Xt

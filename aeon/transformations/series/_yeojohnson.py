"""Implemenents Yeo-Johnson Transformation."""

__maintainer__ = []
__all__ = ["YeoJohnsonTransformer"]

from scipy.stats import yeojohnson, yeojohnson_normmax

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
    >>> from aeon.transformations.series._yeojohnson import YeoJohnsonTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = YeoJohnsonTransformer()
    >>> y_hat = transformer.fit_transform(y)
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

    def __init__(self, bounds=None):
        self.bounds = bounds
        super().__init__()

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
        X_shape = X.shape
        Xt = yeojohnson(X.flatten(), self.lambda_)
        Xt = Xt.reshape(X_shape)
        return Xt

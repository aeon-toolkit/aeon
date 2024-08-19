"""QUANT: A Minimalist Interval Method for Time Series Regression."""

from sklearn.ensemble import ExtraTreesRegressor

from aeon.base._base import _clone_estimator
from aeon.regression import BaseRegressor
from aeon.transformations.collection.interval_based import QUANTTransformer


class QUANTRegressor(BaseRegressor):
    """QUANT interval regressor.

    The regressor computes quantiles over a fixed set of dyadic intervals of
    the input series and three transformations of the input time series. For each set of
    intervals extracted, the window is shifted by half the interval length to extract
    more intervals.

    The feature extraction is performed on the first order differences, second order
    differences, and a Fourier transform of the input series along with the original
    series.

    The transform output is then used to train an extra trees regressor by default.

    Parameters
    ----------
    interval_depth : int, default=6
        The depth to stop extracting intervals at. Starting with the full series, the
        number of intervals extracted is ``2 ** depth`` (starting at 0) for each level.
        The features from all intervals extracted at each level are concatenated
        together for the transform output.
    quantile_divisor : int, default=4
        The divisor to find the number of quantiles to extract from intervals. The
        number of quantiles per interval is
        ``1 + (interval_length - 1) // quantile_divisor``.
    estimator : sklearn estimator, default=None
        The estimator to use for regression. If None, an ExtraTreesRegressor
        with 200 estimators is used.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    QUANTTransformer

    Notes
    -----
    Original code: https://github.com/angus924/quant

    References
    ----------
    .. [1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. QUANT: A Minimalist
        Interval Method for Time Series Classification. arXiv preprint arXiv:2308.00928.

    Examples
    --------
    >>> from aeon.regression.interval_based import QUANTRegressor
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              random_state=0, regression_target=True)
    >>> reg = QUANTRegressor()  # doctest: +SKIP
    >>> reg.fit(X, y)  # doctest: +SKIP
    QUANTRegressor()
    >>> reg.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "interval",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        interval_depth=6,
        quantile_divisor=4,
        estimator=None,
        random_state=None,
    ):
        self.interval_depth = interval_depth
        self.quantile_divisor = quantile_divisor
        self.estimator = estimator
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._transformer = QUANTTransformer(
            interval_depth=self.interval_depth,
            quantile_divisor=self.quantile_divisor,
        )

        self._estimator = _clone_estimator(
            (
                ExtraTreesRegressor(
                    n_estimators=200,
                    max_features=0.1,
                    random_state=self.random_state,
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X):
        return self._estimator.predict(self._transformer.transform(X))

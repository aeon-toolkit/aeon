"""sklearn PCA applied as transformation."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["PCASeriesTransformer"]

import pandas as pd
from sklearn.decomposition import PCA

from aeon.transformations.series.base import BaseSeriesTransformer


class PCASeriesTransformer(BaseSeriesTransformer):
    """Principal Components Analysis applied as transformer.

    Provides a simple wrapper around ``sklearn.decomposition.PCA``.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)
        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.
        Hence, the None case results in::
            n_components == min(n_samples, n_features) - 1
    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.
    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.
    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).
    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).
    n_oversamples : int, default=10
        This parameter is only relevant when `svd_solver="randomized"`.
        It corresponds to the additional number of random vectors to sample the
        range of `X` so as to ensure proper conditioning. See
        :func:`~sklearn.utils.extmath.randomized_svd` for more details.
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
        for more details.
    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.

    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        The fitted PCA object

    Examples
    --------
    >>> # skip DOCTEST if Python < 3.8
    >>> import sys, pytest
    >>> if sys.version_info < (3, 8):
    ...     pytest.skip("PCATransformer requires Python >= 3.8")
    >>>
    >>> from aeon.transformations.series._pca import PCASeriesTransformer
    >>> from aeon.datasets import load_longley
    >>> data = load_longley(return_array=False)
    >>> transformer = PCASeriesTransformer(n_components=2)
    >>> X_hat = transformer.fit_transform(data)

    References
    ----------
    # noqa: E501
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    _tags = {
        "X_inner_type": "pd.DataFrame",
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        iterated_power="auto",
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.iterated_power = iterated_power
        self.random_state = random_state
        super().__init__(axis=0)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X: pd.DataFrame
        y : Ignored

        Returns
        -------
        self: reference to self
        """
        self.pca_ = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )
        self.pca_.fit(X=X)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X: pd.DataFrame
        y : Ignored

        Returns
        -------
        transformed version of X
        """
        Xt = self.pca_.transform(X=X)
        columns = [f"PC_{i}" for i in range(Xt.shape[1])]
        Xt = pd.DataFrame(Xt, index=X.index, columns=columns)

        return Xt

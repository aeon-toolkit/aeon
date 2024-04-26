"""A Learning Shapelet classifier (LSC).

Learning shapelet classifier that simply wraps the LearningShapelet class from tslearn.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["LearningShapeletClassifier"]


import numpy as np

from aeon.classification.base import BaseClassifier


def _X_transformed_tslearn(X):
    if X.ndim == 3:
        X_transformed = np.transpose(X, (0, 2, 1))
    elif X.ndim == 2:
        X_transformed = np.transpose(X)
    return X_transformed


class LearningShapeletClassifier(BaseClassifier):
    """
    Learning Shapelet classifier.

    This is a wrapper for the `LearningShapelet` class of `tslearn`.
    Learning Shapelet classifier, presented in [1]_, operates by
    identifying discriminative subsequences, called shapelets, within
    the input time series data. These shapelets are representative patterns
    that capture essential characteristics of different classes or categories
    within the data.

    Parameters
    ----------
    n_shapelets_per_size: dict, default=None
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value).
        If None, `grabocka_params_to_shapelet_size_dict` is used and the
        size used to compute is that of the shortest time series passed at fit
        time.
    max_iter: int, default=10000
        Number of training epochs.
    batch_size: int, default=256
        Batch size to be used.
    verbose: {0, 1, 2}, default=0
        `keras` verbose level.
    optimizer: str or keras.optimizers.Optimizer, default="sgd"
        `keras` optimizer to use for training.
    weight_regularizer: float or None, default=0.0
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.
    shapelet_length: float, default=0.15
        The length of the shapelets, expressed as a fraction of the time
        series length.
        Used only if `n_shapelets_per_size` is None.
    total_lengths: int, default=3
        The number of different shapelet lengths. Will extract shapelets of
        length i * shapelet_length for i in [1, total_lengths]
        Used only if `n_shapelets_per_size` is None.
    max_size: int or None, default=None
        Maximum size for time series to be fed to the model. If None, it is
        set to the size (number of timestamps) of the training time series.
    scale: bool, default=False
        Whether input data should be scaled for each feature of each time
        series to lie in the [0-1] interval. Default for this parameter is set to
        `False`.
    random_state : int or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    References
    ----------
    .. Grabocka, J., Schilling, N., Wistuba, M. and Schmidt-Thieme, L., 2014, August.
       Learning time-series shapelets. In Proceedings of the 20th ACM SIGKDD
       international conference on Knowledge discovery and data mining (pp. 392-401).

    Examples
    --------
    >>> from aeon.classification.shapelet_based import LearningShapeletClassifier
    >>> from aeon.testing.utils.data_gen import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(random_state=0)
    >>> clf = LearningShapeletClassifier(max_iter=50, random_state=0) # doctest: +SKIP
    >>> clf.fit(X, y) # doctest: +SKIP
    MrSQMClassifier(...)
    >>> clf.predict(X) # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "shapelet",
        "cant-pickle": True,
        "python_dependencies": ["tslearn", "tensorflow"],
    }

    def __init__(
        self,
        n_shapelets_per_size=None,
        max_iter=10000,
        batch_size=256,
        verbose=0,
        optimizer="sgd",
        weight_regularizer=0.0,
        shapelet_length=0.15,
        total_lengths=3,
        max_size=None,
        scale=False,
        random_state=None,
    ):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = optimizer
        self.weight_regularizer = weight_regularizer
        self.shapelet_length = shapelet_length
        self.total_lengths = total_lengths
        self.max_size = max_size
        self.scale = scale
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        from tslearn.shapelets import LearningShapelets

        self.clf_ = LearningShapelets(
            n_shapelets_per_size=self.n_shapelets_per_size,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            verbose=self.verbose,
            optimizer=self.optimizer,
            weight_regularizer=self.weight_regularizer,
            shapelet_length=self.shapelet_length,
            total_lengths=self.total_lengths,
            max_size=self.max_size,
            scale=self.scale,
            random_state=self.random_state,
        )
        _X_transformed = _X_transformed_tslearn(X)
        self.clf_.fit(_X_transformed, y)
        return self

    def _predict(self, X) -> np.ndarray:
        _X_transformed = _X_transformed_tslearn(X)
        return self.clf_.predict(_X_transformed)

    def _predict_proba(self, X) -> np.ndarray:
        _X_transformed = _X_transformed_tslearn(X)
        return self.clf_.predict_proba(_X_transformed)

    # def transform(self, X):
    #     """Generate shapelet transform for a set of time series.
    #
    #     Parameters
    #     ----------
    #     X : array-like of shape=(n_ts, sz, d)
    #         Time series dataset.
    #
    #     Returns
    #     -------
    #     array of shape=(n_ts, n_shapelets)
    #         Shapelet-Transform of the provided time series.
    #     """
    #     _X_transformed = _X_transformed_tslearn(X)
    #     return self.clf_.transform(_X_transformed)
    #
    # def locate(self, X):
    #     """Compute shapelet match location for a set of time series.
    #
    #     Parameters
    #     ----------
    #     X : array-like of shape=(n_ts, sz, d)
    #         Time series dataset.
    #
    #     Returns
    #     -------
    #     array of shape=(n_ts, n_shapelets)
    #         Location of the shapelet matches for the provided time series.
    #     """
    #     _X_transformed = _X_transformed_tslearn(X)
    #     return self.clf_.locate(_X_transformed)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {"max_iter": 50, "batch_size": 10}

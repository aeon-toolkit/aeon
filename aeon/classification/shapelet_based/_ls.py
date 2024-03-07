"""A Learning Shapelet classifier (LSC).

Learning shapelet classifier that simply wraps the LearningShapelet class from tslearn.
"""

import numpy as np

from aeon.classification.base import BaseClassifier


class LearningShapeletClassifier(BaseClassifier):
    """
    Learning Shapelet.

    Parameters
    ----------
    n_shapelets_per_size: dict (default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value).
        If None, `grabocka_params_to_shapelet_size_dict` is used and the
        size used to compute is that of the shortest time series passed at fit
        time.a

    max_iter: int (default: 10,000)
        Number of training epochs.

    batch_size: int (default: 256)
        Batch size to be used.

    verbose: {0, 1, 2} (default: 0)
        `keras` verbose level.

    optimizer: str or keras.optimizers.Optimizer (default: "sgd")
        `keras` optimizer to use for training.

    weight_regularizer: float or None (default: 0.)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.

    shapelet_length: float (default: 0.15)
        The length of the shapelets, expressed as a fraction of the time
        series length.
        Used only if `n_shapelets_per_size` is None.

    total_lengths: int (default: 3)
        The number of different shapelet lengths. Will extract shapelets of
        length i * shapelet_length for i in [1, total_lengths]
        Used only if `n_shapelets_per_size` is None.

    max_size: int or None (default: None)
        Maximum size for time series to be fed to the model. If None, it is
        set to the size (number of timestamps) of the training time series.

    scale: bool (default: False)
        Whether input data should be scaled for each feature of each time
        series to lie in the [0-1] interval.
        Default for this parameter is set to `False` in version 0.4 to ensure
        backward compatibility, but is likely to change in a future version.

    random_state : int or None, optional (default: None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "shapelet",
        "python_dependencies": "tslearn",
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
        super().__init__()
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

    def _X_trnasformed_tslearn(self, X):

        if X.ndim == 3:
            X_transformed = np.transpose(X, (0, 2, 1))
        elif X.ndim == 2:
            X_transformed = np.transpose(X)
        return X_transformed

    def _fit(self, X_transformed, y):
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
        self.clf_.fit(X_transformed, y)
        return self

    def _predict(self, X_transformed) -> np.ndarray:
        return self.clf_.predict(X_transformed)

    def _predict_proba(self, X_transformed) -> np.ndarray:
        return self.clf_.predict_proba(X_transformed)

    def transform(self, X_transformed):
        """Generate shapelet transform for a set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_shapelets)
            Shapelet-Transform of the provided time series.
        """
        return self.clf_.transform(X_transformed)

    def locate(self, X_transformed):
        """Compute shapelet match location for a set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_shapelets)
            Location of the shapelet matches for the provided time series.
        """
        return self.clf_.locate(X_transformed)

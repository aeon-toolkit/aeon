"""Base class for deep clustering."""

__maintainer__ = []

from abc import ABC, abstractmethod

from aeon.clustering._k_means import TimeSeriesKMeans
from aeon.clustering.base import BaseClusterer


class BaseDeepClusterer(BaseClusterer, ABC):
    """Abstract base class for deep learning time series clusterers.

    Parameters
    ----------
    n_clusters : int, default=None
        Please use 'estimator' parameter.
    estimator : aeon clusterer, default=None
        An aeon estimator to be built using the transformed data.
        Defaults to aeon TimeSeriesKMeans() with euclidean distance
        and mean averaging method and n_clusters set to 2.
    clustering_algorithm : str, default="deprecated"
        Please use 'estimator' parameter.
    clustering_params : dict, default=None
        Please use 'estimator' parameter.
    batch_size : int, default = 40
        training batch size for the model
    last_file_name : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used

    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non-deterministic": True,
        "cant-pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        n_clusters=None,
        estimator=None,
        clustering_algorithm="deprecated",
        clustering_params=None,
        batch_size=32,
        last_file_name="last_file",
    ):
        self.estimator = estimator
        self.n_clusters = n_clusters
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params
        self.batch_size = batch_size
        self.last_file_name = last_file_name

        self.model_ = None

        super().__init__(n_clusters=n_clusters)

    @abstractmethod
    def build_model(self, input_shape):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def summary(self):
        """
        Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history : dict or None,
            Dictionary containing model's train/validation losses and metrics

        """
        return self.history.history if self.history is not None else None

    def save_last_model_to_file(self, file_path="./"):
        """Save the last epoch of the trained deep learning model.

        Parameters
        ----------
        file_path : str, default = "./"
            The directory where the model will be saved

        Returns
        -------
        None
        """
        self.model_.save(file_path + self.last_file_name + ".keras")

    def _fit_clustering(self, X):
        """Train the clustering algorithm in the latent space.

        Parameters
        ----------
        X : np.ndarray, shape=(n_cases, n_timepoints, n_channels)
            The input time series.
        """
        import warnings

        self._estimator = (
            TimeSeriesKMeans(
                n_clusters=2, distance="euclidean", averaging_method="mean"
            )
            if self.estimator is None
            else self.estimator
        )

        # to be removed in 1.0.0
        if (
            self.clustering_algorithm != "deprecated"
            or self.clustering_params is not None
            or self.n_clusters is not None
        ):
            warnings.warn(
                "The 'n_clusters' 'clustering_algorithm' and "
                "'clustering_params' parameters "
                "will be removed in v1.0.0. "
                "Their usage will not have an effect, "
                "please use the new 'estimator' parameter to directly "
                "give an aeon clusterer as input.",
                FutureWarning,
                stacklevel=2,
            )

        latent_space = self.model_.layers[1].predict(X)
        self._estimator.fit(X=latent_space)

        return self

    def _predict(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        clusters = self._estimator.predict(latent_space)

        return clusters

    def _predict_proba(self, X):
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)
        latent_space = self.model_.layers[1].predict(X)
        clusters_proba = self._estimator.predict_proba(latent_space)

        return clusters_proba

    def _get_model_checkpoint_callback(self, callbacks, file_path, file_name):
        import tensorflow as tf

        model_checkpoint_ = tf.keras.callbacks.ModelCheckpoint(
            filepath=file_path + file_name + ".keras",
            monitor="loss",
            save_best_only=True,
        )

        if isinstance(callbacks, list):
            return callbacks + [model_checkpoint_]
        else:
            return [callbacks] + [model_checkpoint_]

# -*- coding: utf-8 -*-
"""Base class for deep clustering."""
__author__ = ["hadifawaz1999"]

from abc import ABC, abstractmethod

from aeon.clustering.base import BaseClusterer


class BaseDeepClusterer(BaseClusterer, ABC):
    """Abstract base class for deep learning time series clusterers.

    Parameters
    ----------
    n_clusters: int, default=None
        Number of clusters for the deep learnign model.
    batch_size : int, default = 40
        training batch size for the model
    last_file_name      : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used

    Arguments
    ---------
    self.model_ = None

    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non-deterministic": True,
        "cant-pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(self, n_clusters, batch_size=32, last_file_name="last_file"):
        super(BaseDeepClusterer, self).__init__(n_clusters)

        self.batch_size = batch_size
        self.last_file_name = last_file_name
        self.model_ = None

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
        history: dict or None,
            Dictionary containing model's train/validation losses and metrics

        """
        return self.history.history if self.history is not None else None

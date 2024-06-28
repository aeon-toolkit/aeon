import math

import numpy as np
from sklearn.metrics import accuracy_score

from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based._rocket_classifier import RocketClassifier
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector


class ChannelScorer(BaseChannelSelector):
    """Channel scorer performs channel selection using a single channel classifier.

    ChannelScorer uses a time series classifier to score each channel using an
    estimate of accuracy on the training data, then selects a proportion of the top
    channels to keep. Can be configured through the constructor to use any time
    series classifier and could easily be adapted to use forward selection or elbow
    class methods. Approximately as described in [1]_.

    Parameters
    ----------
    classifier, BaseClassifier, default = MiniROCKET
    proportion : float, default = 0.2
        Proportion of channels to keep, rounded up to nearest integer.

    References
    ----------
    ..[1]: Alejandro Pasos Ruiz and Anthony Bagnall. "Dimension selection strategies
    for multivariate time series classification with HIVE-COTEv2.0." AALTD,
    ECML-PKDD, 2022
    """

    _tags = {
        "requires_y": True,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        classifier: BaseClassifier = None,
        proportion: float = 0.4,
    ):
        self.proportion = proportion
        self.classifier = classifier
        super().__init__()

    def _fit(self, X, y):
        """
        Fit ECP to a specified X and y.

        Parameters
        ----------
        X: np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : reference to self.
        """
        if self.proportion <= 0 or self.proportion > 1:
            raise ValueError("proportion must be in the range 0-1")

        if self.classifier is None:
            self.classifier_ = RocketClassifier(
                rocket_transform="minirocket", num_kernels=5000
            )
        elif not isinstance(self.classifier, BaseClassifier):
            raise ValueError(
                "parameter classifier must be None or an instance of a  "
                "BaseClassifier."
            )
        else:
            self.classifier_ = self.classifier.clone()
        n_channels = X.shape[1]
        scores = np.zeros(n_channels)
        # Evaluate each channel with the classifier
        for i in range(n_channels):
            preds = self.classifier_.fit_predict(X[:, i, :], y)
            scores[i] = accuracy_score(y, preds)
        # Select the top n_keep channels
        sorted_indices = np.argsort(-scores)
        n_keep = math.ceil(n_channels * self.proportion)
        self.channels_selected_ = sorted_indices[:n_keep]
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
        set with that name is available, the default set is returned.

        Returns
        -------
        dict
            Dictionary of testing parameters.
        """
        from aeon.classification import DummyClassifier

        return {"classifier": DummyClassifier(), "proportion": 0.4}

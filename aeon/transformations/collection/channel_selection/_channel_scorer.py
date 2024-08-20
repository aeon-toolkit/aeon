import math
from typing import Dict
from typing import List as TypingList
from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from aeon.base import BaseEstimator
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based._rocket_classifier import RocketClassifier
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

__maintainer__ = ["TonyBagnall"]
__all__ = ["ChannelScorer"]


class ChannelScorer(BaseChannelSelector):
    """Performs channel selection using a single channel classifier or regressor.

    ChannelScorer uses a time series classifier or a regressor to score each channel
    using an estimate of accuracy on the training data fro classifier or mean
    squared error for regressor, then selects a proportion of the top
    channels to keep. Can be configured through the constructor to use any time
    series estimator and could easily be adapted to use forward selection or elbow
    class methods. Approximately as described in [1]_.

    Parameters
    ----------
    estimator: BaseEstimator, default: None
        The classifier or regressor to be used for scoring channels.
    classifier: BaseClassifier, default = MiniROCKET
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
        estimator: BaseEstimator = None,
        proportion: float = 0.4,
    ):
        self.proportion = proportion
        self.estimator = estimator
        super().__init__()

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, TypingList]):
        """
        Fit to a specified X and y.

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

        if self.estimator is None:
            # Default to a classifier if no estimator is provided
            self.estimator_ = RocketClassifier(
                rocket_transform="minirocket", num_kernels=5000
            )
            scoring_function = accuracy_score
            score_sign = 1
        elif isinstance(self.estimator, BaseClassifier):
            self.estimator_ = self.estimator.clone()
            scoring_function = accuracy_score
            score_sign = 1  # Higher accuracy is better, hence positive sign
        elif isinstance(self.estimator, BaseRegressor):
            self.estimator_ = self.estimator.clone()
            scoring_function = mean_squared_error
            score_sign = -1  # Lower MSE is better, hence negative sign
        else:
            raise ValueError(
                "parameter estimator must be an instance of BaseClassifier, "
                "BaseRegressor or None."
            )

        n_channels = X.shape[1]
        scores = np.zeros(n_channels)
        # Evaluate each channel with the classifier
        for i in range(n_channels):
            preds = self.estimator_.fit_predict(X[:, i, :], y)
            scores[i] = score_sign * scoring_function(
                y, preds
            )  # Applying the proper scoring function
        # Select the top n_keep channels
        sorted_indices = np.argsort(-scores)
        n_keep = math.ceil(n_channels * self.proportion)
        self.channels_selected_ = sorted_indices[:n_keep]
        return self

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> Dict[str, any]:
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
        from aeon.regression import DummyRegressor

        if parameter_set == "default":
            return {"estimator": DummyClassifier(), "proportion": 0.4}
        elif parameter_set == "alternative":
            return {"estimator": DummyRegressor(), "proportion": 0.4}
        else:
            raise ValueError(f"Unknown parameter_set: {parameter_set}")

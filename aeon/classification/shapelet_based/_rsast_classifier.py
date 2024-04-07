"""Random Scalable and Accurate Subsequence Transform (RSAST).

Pipeline classifier using the RSAST transformer and an sklearn classifier.
"""

__maintainer__ = []
__all__ = ["RSASTClassifier"]


from operator import itemgetter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.shapelet_based import RSAST
from aeon.utils.numba.general import z_normalise_series
import matplotlib.pyplot as plt


class RSASTClassifier(BaseClassifier):
    """Classification pipeline using RSAST [1]_ transformer and an sklean classifier.

    Parameters
    ----------
    n_random_points: int default = 10 the number of initial random points to extract
    len_method:  string default="both" the type of statistical tool used to get the length of shapelets. "both"=ACF&PACF, "ACF"=ACF, "PACF"=PACF, "None"=Extract randomly any length from the TS
    nb_inst_per_class : int default = 10
        the number of reference time series to select per class
    seed : int, default = None
        the seed of the random generator
    classifier : sklearn compatible classifier, default = None
        if None, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used.
    n_jobs : int, default -1
        Number of threads to use for the transform.


    Reference
    ---------
    .. [1] Varela, N. R., Mbouopda, M. F., & Nguifo, E. M. (2023). RSAST: Sampling Shapelets for Time Series Classification.
    https://hal.science/hal-04311309/
    
    Examples
    --------
    >>> from aeon.classification.shapelet_based import RSASTClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = RSASTClassifier()
    >>> clf.fit(X_train, y_train)
    RSASTClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "capability:multivariate": False,
        "algorithm_type": "subsequence",
    }

    def __init__(
        self,
        n_random_points=10,
        len_method="both",
        nb_inst_per_class=10,
        seed=None,
        classifier=None,
        n_jobs=-1,
    ):
        super().__init__()
        self.n_random_points = n_random_points
        self.len_method = len_method
        self.nb_inst_per_class = nb_inst_per_class
        self.n_jobs = n_jobs
        self.seed = seed
        self.classifier = classifier

    def _fit(self, X, y):
        """Fit RSASTClassifier to the training data.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class values for X.

        Return
        ------
        self : RSASTClassifier
            This pipeline classifier

        """
        self._transformer = RSAST(
            self.n_random_points,
	        self.len_method,
            self.nb_inst_per_class,
            self.seed,
            self.n_jobs,
        )

        self._classifier = _clone_estimator(
            (
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                if self.classifier is None
                else self.classifier
            ),
            self.seed,
        )

        self._pipeline = make_pipeline(self._transformer, self._classifier)

        self._pipeline.fit(X, y)

        return self

    def _predict(self, X):
        """Predict labels for the input.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.

        Return
        ------
        array-like or list
            Predicted class labels.
        """
        return self._pipeline.predict(X)

    def _predict_proba(self, X):
        """Predict labels probabilities for the input.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.

        Return
        ------
        dists : np.ndarray shape (n_cases, n_timepoints)
            Predicted class probabilities.
        """
        m = getattr(self._classifier, "predict_proba", None)
        if callable(m):
            dists = self._pipeline.predict_proba(X)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._pipeline.predict(X)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists

    def plot_most_important_feature_on_ts(self, ts,feature_importance, limit=5):
        
        """Plot the most important features on ts.

        Parameters
        ----------
        ts : float[:]
            The time series
        feature_importance : float[:]
            The importance of each feature in the transformed data
        limit : int, default = 5
            The maximum number of features to plot

        Returns
        -------
        fig : plt.figure
            The figure
        """
        features = zip(self._transformer._kernel_orig, feature_importance)
        sorted_features = sorted(features, key=itemgetter(1), reverse=True)

        max_ = min(limit, len(sorted_features))

        fig, axes = plt.subplots(
            1, max_, sharey=True, figsize=(3 * max_, 3), tight_layout=True
        )

        for f in range(max_):
            kernel, _ = sorted_features[f]
            znorm_kernel = z_normalise_series(kernel)
            d_best = np.inf
            for i in range(ts.size - kernel.size):
                s = ts[i : i + kernel.size]
                s = z_normalise_series(s)
                d = np.sum((s - znorm_kernel) ** 2)
                if d < d_best:
                    d_best = d
                    start_pos = i
            axes[f].plot(range(start_pos, start_pos + kernel.size), kernel, linewidth=5)
            axes[f].plot(range(ts.size), ts, linewidth=2)
            axes[f].set_title(f"feature: {f+1}")

        #return fig


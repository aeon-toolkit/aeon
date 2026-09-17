"""KGMTP classifier.

Pipeline classifier using the KGMTP transformer and RidgeClassifierCV classifier.
"""

__maintainer__ = ["johannfaouzi"]
__all__ = ["KGMTPClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import KGMTP
from aeon.utils.validation import check_n_jobs


class KGMTPClassifier(BaseClassifier):
    """KG-MTP classifier.

    This classifier transforms the input data using the `KGMTP` [1]_ transformer,
    extracting PPV-pooling and Hydra-style features from three representations of each
    series (raw, its Hilbert transform, and its first difference) and scaling them
    internally, then fits a sklearn classifier on the transformed features (default
    classifier is `RidgeClassifierCV`).

    Parameters
    ----------
    n_kernels : int, default=50_000
        Total PPV-pooling feature budget for the `KGMTP` transform, split evenly across
        its three internal representations (raw, Hilbert, first difference).
    max_dilations_per_kernel : int, default=32
        The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=5
        The number of PPV-pooling statistics per kernel.
    estimator : sklearn compatible classifier or None, default=None
        The estimator used. If None, a RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        is used.
    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Only applies if estimator is None and the default is used. From sklearn
        documentation: If not given, all classes are supposed to have weight one.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`. ``-1`` means
        using all processors.
    random_state : int, ``numpy.random.Generator``, or None, default=None
        If ``int``, seed a new ``Generator``. If a ``Generator`` instance, use it
        directly -- several ``KGMTP`` instances can then share one
        continuously-advancing stream across repeated `fit` calls. If ``None``, use a
        fresh, unseeded ``Generator``. Unlike most aeon estimators, a legacy
        ``numpy.random.RandomState`` is not accepted.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.
    estimator_ : sklearn classifier
        The fitted estimator.

    See Also
    --------
    RocketClassifier, MiniRocketClassifier, MultiRocketClassifier, HydraClassifier

    References
    ----------
    .. [1] Wang, Wu, Wei, Li. "KG-MTP: Kernel Grouping for Time Series Classification
           with Multiple Transformations and Pooling Operators." Expert Systems with
           Applications, 2025.
           https://doi.org/10.1016/j.eswa.2025.128693

    Examples
    --------
    >>> from aeon.classification.convolution_based import KGMTPClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = KGMTPClassifier(n_kernels=1200)
    >>> clf.fit(X_train, y_train)
    KGMTPClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=50_000,
        max_dilations_per_kernel=32,
        n_features_per_kernel=5,
        estimator=None,
        class_weight=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.estimator = estimator

        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        """Fit KGMTP transform + estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray
            The training data of shape = (n_cases, n_channels, n_timepoints).
        y : np.ndarray
            The class labels, shape = (n_cases,).

        Returns
        -------
        self : Reference to self.
        """
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = KGMTP(
            n_kernels=self.n_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_features_per_kernel=self.n_features_per_kernel,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        self.estimator_ = _clone_estimator(
            (
                RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
                )
                if self.estimator is None
                else self.estimator
            ),
            self.random_state,
        )

        self.pipeline_ = make_pipeline(self._transformer, self.estimator_)
        self.pipeline_.fit(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X."""
        return self.pipeline_.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X."""
        m = getattr(self.estimator_, "predict_proba", None)
        if callable(m):
            return self.pipeline_.predict_proba(X)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self.pipeline_.predict(X)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "n_kernels": 1200,
            "max_dilations_per_kernel": 4,
            "n_features_per_kernel": 5,
        }

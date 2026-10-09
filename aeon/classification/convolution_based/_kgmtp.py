"""KGMTP classifier.

Pipeline classifier using the KGMTP transformer and a scikit-learn estimator: by
default, a StandardScaler followed by RidgeClassifierCV, or a user-supplied
estimator fit directly on the transform's output.
"""

__maintainer__ = ["johannfaouzi"]
__all__ = ["KGMTPClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based import KGMTP
from aeon.utils.validation import check_n_jobs


class KGMTPClassifier(BaseClassifier):
    """KG-MTP classifier.

    This classifier transforms the input data using the `KGMTP` [1]_ transformer,
    extracting pooling and Hydra-style features from three representations of each
    series (raw, its Hilbert transform, and its first difference), with the
    Hydra-style block scaled according to `scale_hydra`. If `estimator` is left as
    ``None`` (the default), a `StandardScaler` -> `RidgeClassifierCV` pipeline is
    then fit on the full concatenated output (pooling + Hydra features) --
    together with `scale_hydra`'s default of ``True``, reproducing the original
    paper's pipeline exactly. A custom `estimator` is instead fit directly on the
    concatenated output, with no `StandardScaler` applied.

    Multivariate series are supported: `KGMTP` processes each channel independently
    and concatenates every channel's output (see its docstring), so the per-channel
    feature budget below scales with the number of channels.

    Parameters
    ----------
    n_kernels : int, default=50_000
        Total pooling feature budget *per channel* for the `KGMTP` transform,
        split evenly across its three internal representations (raw, Hilbert, first
        difference).
    max_dilations_per_kernel : int, default=32
        The maximum number of dilations per kernel.
    scale_hydra : bool, default=True
        Whether to scale the pooled Hydra features with `KGMTP`'s masked,
        epsilon-regularized scaler, passed straight through to the `KGMTP`
        transform. The default, ``True``, is what the original paper's pipeline
        uses.
    estimator : sklearn compatible classifier or None, default=None
        The estimator used. If None, a pipeline consisting of StandardScaler() followed
        by RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)) is used. If not None, it is
        fit directly on KGMTP's raw output instead.
    class_weight : {None, "balanced"}, dict or list of dicts, default=None
        Only applies if estimator is None and the default is used.
        From sklearn documentation:
        If None, all classes are assigned equal weights.
        The “balanced” mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))
        For multi-output, the weights of each column of y will be multiplied.
        A dictionary can also be provided to specify weights for each class manually.
        Note that these weights will be multiplied with sample_weight (passed through
        the fit method) if sample_weight is specified.
        Note: "balanced_subsample" is not supported as RidgeClassifierCV is not an
        ensemble model.
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
        The fitted estimator: a `StandardScaler` -> `RidgeClassifierCV` pipeline when
        `estimator` is None, otherwise the fitted clone of the given `estimator`.

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
        "capability:multivariate": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels=50_000,
        max_dilations_per_kernel=32,
        scale_hydra=True,
        estimator=None,
        class_weight=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.scale_hydra = scale_hydra
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
            n_jobs=self._n_jobs,
            scale_hydra=self.scale_hydra,
            random_state=self.random_state,
        )

        estimator = (
            make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight
                ),
            )
            if self.estimator is None
            else self.estimator
        )
        self.estimator_ = _clone_estimator(estimator, self.random_state)

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
        }

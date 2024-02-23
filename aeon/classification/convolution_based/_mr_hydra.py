import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from aeon.classification import BaseClassifier
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import (
    MultiRocket,
    MultiRocketMultivariate,
)
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer


class MultiRocketHydraClassifier(BaseClassifier):
    """MultiRocket-Hydra Classifier.

    A combination of the Hydra and MultiRocket algorithms. The algorithm concatenates
    the output of both algorithms and trains a linear classifier on the combined
    features.

    See both individual classifier/transformation for more details.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group for the Hydra transform.
    n_groups : int, default=64
        Number of groups per dilation for the Hydra transform.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.

    See Also
    --------
    HydraClassifier
    RocketClassifier

    References
    ----------
    .. [1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing
        convolutional kernels for fast and accurate time series classification.
        Data Mining and Knowledge Discovery, pp.1-27.

    Examples
    --------
    >>> from aeon.classification.convolution_based import MultiRocketHydraClassifier
    >>> from aeon.testing.utils.data_gen import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              random_state=0)
    >>> clf = MultiRocketHydraClassifier(random_state=0)  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    HydraClassifier(random_state=0)
    >>> clf.predict(X)  # doctest: +SKIP
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(self, n_kernels=8, n_groups=64, n_jobs=1, random_state=None):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._transform_hydra = HydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        Xt_hydra = self._transform_hydra.fit_transform(X)

        self._scale_hydra = _SparseScaler()
        Xt_hydra = self._scale_hydra.fit_transform(Xt_hydra)

        self._transform_multirocket = (
            MultiRocket(
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            if X.shape[1] == 1
            else MultiRocketMultivariate(
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        )
        Xt_multirocket = self._transform_multirocket.fit_transform(X)

        self._scale_multirocket = StandardScaler()
        Xt_multirocket = self._scale_multirocket.fit_transform(Xt_multirocket)

        Xt = np.concatenate((Xt_hydra, Xt_multirocket), axis=1)

        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(Xt, y)

        return self

    def _predict(self, X) -> np.ndarray:
        Xt_hydra = self._transform_hydra.transform(X)
        Xt_hydra = self._scale_hydra.transform(Xt_hydra)

        Xt_multirocket = self._transform_multirocket.transform(X)
        Xt_multirocket = self._scale_multirocket.transform(Xt_multirocket)

        Xt = np.concatenate((Xt_hydra, Xt_multirocket), axis=1)

        return self.classifier.predict(Xt)

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.classification import BaseClassifier
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer


class HYDRA(BaseClassifier):
    """Hydra Classifier."""

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "classifier_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(self, k=8, g=64, n_jobs=1, random_state=None):
        self.k = k
        self.g = g
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        transform = HydraTransformer(
            k=self.k,
            g=self.g,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self._clf = make_pipeline(
            transform,
            _SparseScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        self._clf.fit(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        return self._clf.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)


class _SparseScaler:
    """Sparse Scaler for hydra transform."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None):
        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

    def transform(self, X, y=None):
        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

"""Isolation Forest Adapter for Anomaly Detection."""

__all__ = ["IsolationForest"]

from typing import Optional, Union

import numpy as np

from aeon.anomaly_detection._pyodadapter import PyODAdapter
from aeon.utils.validation._dependencies import _check_soft_dependencies


class IsolationForest(PyODAdapter):
    """IForest adapter for anomaly detection.

    This class implements the Isolation Forest algorithm for anomaly detection
    using PyODAdadpter to be used in the Aeon framework.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    behaviour : str, default='old'
        Behaviour of the ``decision_function`` which can be either 'old' or
        'new'. Passing ``behaviour='new'`` makes the ``decision_function``
        change to match other anomaly detection algorithm API which will be
        the default behaviour in the future. As explained in details in the
        ``offset_`` attribute documentation, the ``decision_function`` becomes
        dependent on the contamination parameter, in such a way that 0 becomes
        its natural threshold to detect outliers.

        .. versionadded:: 0.7.0
           ``behaviour`` is added in 0.7.0 for back-compatibility purpose.

        .. deprecated:: 0.20
           ``behaviour='old'`` is deprecated in sklearn 0.20 and will not be
           possible in 0.22.

        .. deprecated:: 0.22
           ``behaviour`` parameter will be deprecated in sklearn 0.22 and
           removed in 0.24.

        .. warning::
            Only applicable for sklearn 0.20 above.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    window_size : int, default=10
        Size of the sliding window.

    stride : int, default=1
        Stride of the sliding window.

    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float] = "auto",
        contamination: float = 0.1,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: int = 1,
        behaviour: str = "old",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,
        window_size: int = 10,
        stride: int = 1,
    ):
        if not _check_soft_dependencies("pyod", severity="none"):
            raise ModuleNotFoundError(
                "pyod is a soft dependency and not included"
                " in the base aeon installation."
            )
        from pyod.models.iforest import IForest

        model = IForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            behaviour=behaviour,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose
        super().__init__(model, window_size, stride)

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        super()._fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return super()._predict(X)

    def _fit_predict(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        return super()._fit_predict(X, y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `IsolationForest(**params)` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.1,
            "max_features": 1.0,
            "bootstrap": False,
            "n_jobs": 1,
            "behaviour": "old",
            "random_state": None,
            "verbose": 0,
            "window_size": 10,
            "stride": 1,
        }

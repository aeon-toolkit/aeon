"""Local Outlier Factor (LOF) algorithm for anomaly detection."""

__maintainer__ = []
__all__ = ["LOF"]

from typing import Optional, Union

import numpy as np
from sklearn.exceptions import NotFittedError

from aeon.anomaly_detection._pyodadapter import PyODAdapter
from aeon.utils.validation._dependencies import _check_soft_dependencies


class LOF(PyODAdapter):
    """Local Outlier Factor (LOF) algorithm for anomaly detection.

    This class implement metrics-based outlier detection algorithms using the
    Local Outlier Factor (LOF) algorithm from PyOD.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for `kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default=30)
        Leaf size passed to `BallTree` or `KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : string or callable, default 'minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.
        If 'precomputed', the training input X is expected to be a distance
        matrix.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
    p : integer, optional (default = 2)
        Parameter for the Minkowski metric
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the decision function.
    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.
    novelty : bool (default=False)
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "python_dependencies": ["pyod"],
    }

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: Optional[str] = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: Optional[dict] = None,
        contamination: float = 0.1,
        n_jobs: int = 1,
        novelty: bool = True,
        window_size: int = 10,
        stride: int = 1,
    ):
        _check_soft_dependencies(*self._tags["python_dependencies"])
        from pyod.models.lof import LOF

        # Validate that stride is not greater than winow_size
        if stride > window_size:
            raise ValueError(
                f"Stride ({stride}) cannot be greater than window size ({window_size})."
            )

        model = LOF(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            n_jobs=n_jobs,
            novelty=novelty,
        )
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.novelty = novelty
        super().__init__(model, window_size, stride)

    def fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        super()._fit(X, y)
        self.is_fitted = True  # Fitting completed
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_pyod_model_"):
            raise NotFittedError(
                "This instance of LOF has not been fitted yet; please call `fit` first."
            )
        if not hasattr(self, "fitted_pyod_model_"):
            raise NotFittedError(
                "This instance of LOF has not been fitted yet; please call `fit` first."
            )
        return super()._predict(X)

    def fit_predict(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        return super()._fit_predict(X, y)

    def _check_params(self, X: np.ndarray) -> None:
        if self.window_size < 1 or self.window_size > X.shape[0]:
            self.window_size = min(max(1, self.window_size), X.shape[0])
            raise ValueError(
                "The window size must be at least 1 and "
                "at most the length of the time series."
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
            Each dict corresponds to parameters that will create an "interesting"
            test instance.
        """
        # Define a test parameter set with different combinations of parameters
        return {
            "n_neighbors": 5,
            "leaf_size": 10,
            "p": 2,
            "contamination": 0.05,
            "window_size": 10,
            "stride": 2,
        }

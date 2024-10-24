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

    .. list-table:: Capabilities
       :stub-columns: 1

       * - Input data format
         - univariate or multivariate
       * - Output data format
         - binary classification
       * - missing_values
         - False
       * - Learning Type
         - unsupervised or semi-supervised
       * - python_dependencies
         - ["pyod"]

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
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
    p : integer, optional (default = 2)
        Parameter for the Minkowski metric
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
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
        n_jobs: int = 1,
        window_size: int = 10,
        stride: int = 1,
        pyod_model=None,
    ):
        _check_soft_dependencies(*self._tags["python_dependencies"])
        from pyod.models.lof import LOF

        # Using the pyod_model if already provided else creating a new instance
        model = pyod_model or LOF(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            novelty=False,
        )
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.window_size = window_size
        self.stride = stride
        super().__init__(model, window_size, stride)

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, None] = None) -> None:
        # Set novelty to True for supervised learning
        self.pyod_model.novelty = True
        super()._fit(X, y)
        self.is_fitted = True  # Fitting completed
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_pyod_model_"):
            raise NotFittedError(
                "This instance of LOF has not been fitted yet; please call `fit` first."
            )
        # Set novelty to True for prediction on unseen data
        self.pyod_model.novelty = True
        return super()._predict(X)

    def _fit_predict(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        # Set novelty to False for unsupervised learning
        self.pyod_model.novelty = False
        return super()._fit_predict(X, y)

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
            "window_size": 10,
            "stride": 2,
        }


from aeon.classification.base import BaseClassifier
import numpy as np

__all__ = ["TDMVDCClassifier"]

class TDMVDCClassifier(BaseClassifier):
    """
    Tracking Differentiator-based Multiview Dilated Characteristics 
    for Time Series Classification.

    Parameters
    ----------
    default_fc_parameters : str, default="efficient"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    k1 : floot, default=2
        Filter parameter of the Tracking Differentiator1 with generating first-order 
        differential series
    k2 : floot, default=2
        Filter parameter of the Tracking Differentiator2 with generating second-order 
        differential series
    feature_store_ratios : list, default=[0.1, 0.2, 0.3, 0.4, 0.5]
        List of feature saving ratios for different feature selectors
    
    
    References
    ----------
    .. [1] Changchun He, and Xin Huo. "Tracking Differentiator-based Multiview Dilated 
        Characteristics for Time Series Classification." in The 22nd IEEE International 
        Conference on Industrial Informatics (INDIN2024) (2024).
    """

    _tags = { # needs to be changed later
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:train_estimate": True,
        "algorithm_type": "feature",
        "python_dependencies": "tsfresh",
    }
    def __init__(self):
        pass

    def _fit(self, X, y):
        pass

    def _predict(self, X) -> np.ndarray:
        pass

    def _predict_proba(self, X): #optional
        return super()._predict_proba(X)
    def _fit_predict(self, X, y) -> np.ndarray:
        pass
    
    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        pass
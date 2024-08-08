""" Moving average transformation """

__maintainer__ = ['Datadote']
__all__ = 'MovingAverageTransformer'

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer

class MovingAverageTransformer(BaseSeriesTransformer):
    """ Filter a time series using a simple moving average.
    """
    
    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True, # TODO: what bool to set?
    }

    def __init__(
            self,
            window_size: int = 5,
    ) -> None:
        super().__init__(axis=0)
        self.window_size = window_size

    def _transform(self, X, y=None):
        """ Transform X and return a transformed version.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        csum = np.cumsum(X, axis=0)
        csum[self.window_size:, :] = csum[self.window_size:, :] - csum[:-self.window_size, :]    
        return csum[self.window_size - 1:, :] / self.window_size

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """ Return testing parameter settings for the estimator.
        """
        params = {"window_size": 5}
        return params

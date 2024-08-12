""" Moving average transformation """

__maintainer__ = ["Datadote"]
__all__ = "MovingAverageTransformer"

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer

class MovingAverageTransformer(BaseSeriesTransformer):
    """ Filter a time series using a simple moving average. 
    
    Parameters
    ----------
    window_size: int, default=5
        Number of values to average for each window

    References
    ----------
    James Large, Paul Southam, Anthony Bagnall
        "Can automated smoothing significantly improve benchmark time series classification algorithms?"
        https://arxiv.org/abs/1811.00894

    Examples
    --------
    import numpy as np
    from aeon.transformations.series._moving_average import MovingAverageTransformer
    X = np.array([-3, -2, -1,  0,  1,  2,  3])
    transformer = MovingAverageTransformer(2)
    Xt = transformer.fit_transform(X)
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
        super().__init__(axis=0) # TODO: init first or last?
        self.window_size = window_size

    def _transform(self, X, y=None):
        """ Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt: 2D np.ndarray
            transformed version of X
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        csum = np.cumsum(X, axis=0)
        csum[self.window_size:, :] = csum[self.window_size:, :] - csum[:-self.window_size, :]    
        Xt = csum[self.window_size - 1:, :] / self.window_size
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """ Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
        """
        params = {"window_size": 5}
        return params

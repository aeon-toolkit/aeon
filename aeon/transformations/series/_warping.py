"""Warping Transformer using elastic measures for warping path."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["WarpingSeriesTransformer"]

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class WarpingSeriesTransformer(BaseSeriesTransformer):
    """Warping Path Transformer.

    This transformer produces a longer version of the input series
    following the warping path produced by an elastic measure.
    The transformer assumes the path is pre-computed between the input
    series and another one.

    Parameters
    ----------
    series_index : int, default = 0
        The index of the series, either 0 or 1 to choose from the warping
        path. Given the path is generated using two series, the user
        should choose which one is being transformed.
    warping_path : List[Tuple[int,int]], default = None
        The warping path used to transform the series.
        If None, the output series is returned as is.

    Examples
    --------
    >>> from aeon.transformations.series import WarpingSeriesTransformer
    >>> from aeon.distances import dtw_alignment_path
    >>> import numpy as np
    >>> x = np.random.normal((2, 100))
    >>> y = np.random.normal((2, 100))
    >>> dtw_path, _ = dtw_alignment_path(x, y)
    >>> x_transformed = WarpingSeriesTransformer(
    ... series_index=0, warping_path=dtw_path).fit_transform(x)
    >>> y_transformed = WarpingSeriesTransformer(
    ... series_index=1, warping_path=dtw_path).fit_transform(y)
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        series_index: int = 0,
        warping_path: list[tuple[int, int]] = None,
    ) -> None:

        self.series_index = series_index
        self.warping_path = warping_path

        super().__init__(axis=1)

    def _transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : np.ndarray
            time series of shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility

        Returns
        -------
        aligned_series : np.ndarray of shape n_channels, len(warping_path)
        """
        if self.warping_path is None:
            return X

        indices_0, indices_1 = zip(*self.warping_path)

        if self.series_index == 0:
            indices_ = np.array(indices_0)
        elif self.series_index == 1:
            indices_ = np.array(indices_1)
        else:
            raise ValueError("The parameter series_index can only be 0 or 1.")

        aligned_series = X[:, indices_]

        return aligned_series

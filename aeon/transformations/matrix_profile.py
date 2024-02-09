"""Implements matrix profile transformation."""

__author__ = ["mloning"]
__all__ = ["MatrixProfileTransformer"]

from deprecated.sphinx import deprecated

from aeon.transformations.base import BaseTransformer


# TODO: remove in v0.8.0
@deprecated(
    version="0.7.0",
    reason="MatrixProfileTransformer will be removed from the base directory of "
    "transformations in v0.8.0, it has been replaced by MatrixProfileSeriesTransformer"
    "in the transformations.series module.",
    category=FutureWarning,
)
class MatrixProfileTransformer(BaseTransformer):
    """Calculate the matrix profile of a time series.

    Takes as input a single time series dataset and returns the matrix profile
    for that time series dataset. The matrix profile is a vector that stores the
    z-normalized Euclidean distance between any subsequence within a
    time series and its nearest neighbor.

    For more information on the matrix profile, see `stumpy's tutorial
    <https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html>`_

    Parameters
    ----------
    window_length : int

    Notes
    -----
    Provides wrapper around functionality in `stumpy.stump
    <https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stump>`_

    Examples
    --------
    >>> from aeon.transformations.matrix_profile import \
    MatrixProfileTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MatrixProfileTransformer()  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "input_data_type": "Series",
        # what is the abstract type of X: Series, or Panel
        "output_data_type": "Series",
        # what abstract type is returned: Primitives, Series, Panel
        "instancewise": True,  # is this an instance-wise transform?
        "X_inner_type": ["np.ndarray"],
        "y_inner_type": "None",
        "univariate-only": True,
        "fit_is_empty": True,  # for unit test cases
        "python_dependencies": "stumpy",
    }

    def __init__(self, window_length=3):
        self.window_length = window_length
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 1D np.ndarray
            transformed version of X
            Matrix Profile of time series as output with length as
            (n_timepoints-window_length+1)
        """
        import stumpy

        X = X.flatten()
        Xt = stumpy.stump(X, self.window_length)
        Xt = Xt[:, 0].astype("float")
        return Xt

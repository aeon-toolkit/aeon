"""Sliding Window transformation."""

__maintainer__ = []
__all__ = ["TrainTestTransformer"]

import math

from aeon.transformations.format.base import BaseFormatTransformer


class TrainTestTransformer(BaseFormatTransformer):
    """
    Convert a single time series into train/test sets.

    This function assumes that the input DataFrame contains only one time series.
    It splits the series into training and testing sets based on
    the specified proportion.

    Parameters
    ----------
    train_proportion : float, optional (default=0.7)
        The proportion of the time series to use for training,
        with the remaining used for test.
    max_series_length : int, optional (default=10000)
        The maximum length of the series to consider. If the series is longer
        than this value, it will be truncated.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.format import TrainTestTransformer
    >>> X = np.array([-3, -2, -1,  0,  1,  2,  3, 4])
    >>> transformer = TrainTestTransformer(0.75)
    >>> Xt = transformer.fit_transform(X)
    >>> print(Xt)
    (array([-3, -2, -1,  0,  1,  2]), array([3, 4]))

    Returns
    -------
    None
        A tuple containing the training and testing sets.

    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
        "output_data_type": "Tuple",
    }

    def __init__(
        self, train_proportion: float = 0.7, max_series_length: int = 10000
    ) -> None:
        super().__init__(axis=1)
        if train_proportion <= 0 or train_proportion >= 1:
            raise ValueError(
                f"train_proportion must be between 0 and 1, got {train_proportion}"
            )
        self.train_proportion = train_proportion
        self.max_series_length = max_series_length

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

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
        X = X[0]
        # Compute split index
        if len(X) < self.max_series_length or self.max_series_length == -1:
            end_location = len(X)
        else:
            end_location = self.max_series_length
        train_test_split_location = math.ceil(end_location * self.train_proportion)

        # Split into train and test sets
        train_series = X[:train_test_split_location]
        test_series = X[train_test_split_location:end_location]

        # Generate windowed versions of train and test sets
        return train_series, test_series

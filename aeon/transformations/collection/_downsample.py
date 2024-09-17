"""Downsample transformer."""

__maintainer__ = []
__all__ = ["DownsampleTransformer"]

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class DownsampleTransformer(BaseCollectionTransformer):
    """Downsample the time dimension of a collection of time series.

    Parameters
    ----------
    downsample_by : str, optional
        The method to downsample by, either "frequency" or "proportion",
        by default "frequency".
    source_sfreq : int or float, optional
        The source sampling frequency in Hz.
        Required if `downsample_by = "frequency"`, by default 2.0.
    target_sfreq : int or float, optional
        The target sampling frequency in Hz.
        Required if `downsample_by = "frequency"`, by default 1.0.
    proportion : float, optional
        The proportion between 0-1 to downsample by.
        Required if `downsample_by = "proportion"`, by default None.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection import DownsampleTransformer
    >>> # unequal length collection of time series
    >>> X = [np.array([[1, 2, 3, 4, 5, 6]]), np.array([[7, 8, 9, 10]])]
    >>> transformer = DownsampleTransformer(
    ...     downsample_by="frequency",
    ...     source_sfreq=2.0,
    ...     target_sfreq=1.0,
    ... )
    >>> transformer.fit_transform(X)
    [array([[1, 3, 5]]), array([[7, 9]])]
    >>> # same downsampling, but by proportion instead of frequency
    >>> transformer.set_params(downsample_by="proportion", proportion=0.5)
    DownsampleTransformer(downsample_by='proportion', proportion=0.5)
    >>> transformer.fit_transform(X)
    [array([[1, 3, 5]]), array([[7, 9]])]
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        downsample_by="frequency",
        source_sfreq=2.0,
        target_sfreq=1.0,
        proportion=None,
    ):
        self.downsample_by = downsample_by
        self.source_sfreq = source_sfreq
        self.target_sfreq = target_sfreq
        self.proportion = proportion
        super().__init__()

    def _transform(self, X, y=None):
        """Transform the input collection to downsampled collection.

        Parameters
        ----------
        X : list or np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Input time series collection where n_timepoints can vary over cases.
        y : None
            Ignored for interface compatibility, by default None.

        Returns
        -------
        list of 2D numpy arrays of shape [(n_channels, n_timepoints_downsampled), ...]
        or np.ndarray of shape (n_cases, n_channels, n_timepoints_downsampled)
            Downsampled time series collection.
        """
        self._check_parameters()

        if self.downsample_by == "frequency":
            step = int(self.source_sfreq / self.target_sfreq)
        elif self.downsample_by == "proportion":
            step = int(1 / (1 - self.proportion))

        X_downsampled = []
        for x in X:
            n_timepoints = x.shape[-1]
            indices = np.arange(0, n_timepoints, step)
            X_downsampled.append(x[:, indices])

        if isinstance(X, np.ndarray):
            return np.asarray(X_downsampled)
        else:
            return X_downsampled

    def _check_parameters(self):
        """Check the values of parameters passed to DownsampleTransformer.

        Raises
        ------
        ValueError
            If `downsample_by` is not "frequency" or "proportion".
            If `source_sfreq` < `target_sfreq` when `downsample_by = "frequency"`.
            If `proportion` is not between 0-1 when `downsample_by = "proportion"`.
        """
        if self.downsample_by not in ["frequency", "proportion"]:
            raise ValueError('downsample_by must be either "frequency" or "proportion"')

        if self.downsample_by == "frequency":
            if self.source_sfreq is None or self.target_sfreq is None:
                raise ValueError("source_sfreq and target_sfreq must be provided")
            if self.source_sfreq < self.target_sfreq:
                raise ValueError("source_sfreq must be > target_sfreq")

        if self.downsample_by == "proportion":
            if self.proportion is None or not (0 < self.proportion < 1):
                raise ValueError("proportion must be provided and between 0-1.")

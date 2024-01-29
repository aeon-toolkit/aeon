"""Transformations for segmenting time series."""

import numpy as np
import pandas as pd

from aeon.transformations.base import BaseTransformer


class PlateauFinder(BaseTransformer):
    """
    Plateau finder transformer.

    Transformer that finds segments of the same given value, plateau in
    the time series, and returns the starting indices and lengths.

    Parameters
    ----------
    value : {int, float, np.nan, np.inf}
        Value for which to find segments
    min_length : int
        Minimum lengths of segments with same value to include.
        If min_length is set to 1, the transformer can be used as a value
        finder.
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": True,
        "output_data_type": "Series",
        "instancewise": False,
        "X_inner_type": "numpy3D",
        "y_inner_type": "None",
    }

    def __init__(self, value=np.nan, min_length=2):
        self.value = value
        self.min_length = min_length
        super().__init__(_output_convert=False)

    def _transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : numpy3D array shape (n_cases, 1, series_length)

        Returns
        -------
        X : pandas data frame
        """
        _starts = []
        _lengths = []

        # find plateaus (segments of the same value)
        for x in X[:, 0]:
            # find indices of transition
            if np.isnan(self.value):
                i = np.where(np.isnan(x), 1, 0)

            elif np.isinf(self.value):
                i = np.where(np.isinf(x), 1, 0)

            else:
                i = np.where(x == self.value, 1, 0)

            # pad and find where segments transition
            transitions = np.diff(np.hstack([0, i, 0]))

            # compute starts, ends and lengths of the segments
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            lengths = ends - starts

            # filter out single points
            starts = starts[lengths >= self.min_length]
            lengths = lengths[lengths >= self.min_length]

            _starts.append(starts)
            _lengths.append(lengths)

        # put into dataframe
        Xt = pd.DataFrame()
        column_prefix = "{}_{}".format(
            "channel_",
            "nan" if np.isnan(self.value) else str(self.value),
        )
        Xt["%s_starts" % column_prefix] = pd.Series(_starts)
        Xt["%s_lengths" % column_prefix] = pd.Series(_lengths)

        Xt = Xt.applymap(lambda x: pd.Series(x))
        return Xt

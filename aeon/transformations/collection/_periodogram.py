"""Periodogram transformer."""

__maintainer__ = []
__all__ = ["PeriodogramTransformer"]

import math

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


class PeriodogramTransformer(BaseCollectionTransformer):
    """Periodogram transformer.

    This transformer converts a collection of time series into its periodogram
    representation.

    Parameters
    ----------
    pad_series : bool, default=True
        Whether to pad the series to the next power of 2. If False, the series
        will be used as is.
    pad_with : str, default="constant"
        The type of padding to use. see the numpy.pad documentation mode parameter for
        options. By default, the series will be padded with zeros.
    constant_value : int, default=0
        The value to use when padding with a constant value.

    Examples
    --------
    >>> from aeon.transformations.collection import PeriodogramTransformer
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X = make_example_3d_numpy(n_cases=4, n_channels=2, n_timepoints=20,
    ...                           random_state=0, return_y=False)
    >>> tnf = PeriodogramTransformer()
    >>> tnf.fit(X)
    PeriodogramTransformer(...)
    >>> print(tnf.transform(X)[0])
    [[22.16456597 11.08122685  3.69018936  2.17665255  5.27387039  3.10598557
       6.311107    1.70468284  1.8269671   0.88838033  1.56747869  3.42037058
       1.67988661  1.71142437  3.49821716  1.25120108]
     [22.71382067  8.64933688  6.36412194  0.9298486   5.70358068  2.70669743
       4.33906385  0.36544821  2.28769936  3.67702091  1.45018642  1.26838712
       3.36395549  2.69146494  2.27041859  3.9023142 ]]
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        pad_series=True,
        pad_with="constant",
        constant_value=0,
    ):
        self.pad_series = pad_series
        self.pad_with = pad_with
        self.constant_value = constant_value

        super().__init__()

    def _transform(self, X, y=None):
        if self.pad_series:
            kwargs = {"mode": self.pad_with}
            if self.pad_with == "constant":
                kwargs["constant_values"] = self.constant_value
            len = int(math.pow(2, math.ceil(math.log(X.shape[2], 2))) - X.shape[2])
            X = np.pad(
                X,
                (
                    (0, 0),
                    (0, 0),
                    (
                        0,
                        len,
                    ),
                ),
                **kwargs,
            )
        Xt = np.abs(np.fft.fft(X)[:, :, : int(X.shape[2] / 2)])

        return Xt

"""QUANT, a minimalistic interval transform using quantile features."""

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


class QUANTTransformer(BaseCollectionTransformer):
    """QUANT interval transform.

    The transform involves computing quantiles over a fixed set of dyadic intervals of
    the input series and three transformations of the input time series. For each set of
    intervals extracted, the window is shifted by half the interval length to extract
    more intervals.

    The feature extraction is performed on the first order differences, second order
    differences, and a Fourier transform of the input series along with the original
    series.

    Parameters
    ----------
    interval_depth : int, default=6
        The depth to stop extracting intervals at. Starting with the full series, the
        number of intervals extracted is 2 ** depth (starting at 0) for each level.
        The features from all intervals extracted at each level are concatenated
        together for the transform output.
    quantile_divisor : int, default=4
        The divisor to find the number of quantiles to extract from intervals. The
        number of quantiles per interval is
        ``1 + (interval_length - 1) // quantile_divisor``.

    Attributes
    ----------
    intervals_ : list of np.ndarray
        The intervals extracted for each representation function.

    See Also
    --------
    QUANTClassifier

    Notes
    -----
    Original code: https://github.com/angus924/quant

    References
    ----------
    .. [1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. QUANT: A Minimalist
        Interval Method for Time Series Classification. arXiv preprint arXiv:2308.00928.

    Examples
    --------
    >>> from aeon.transformations.collection.interval_based import QUANTTransformer
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, _ = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              random_state=0)
    >>> q = QUANTTransformer(interval_depth=2, quantile_divisor=8)  # doctest: +SKIP
    >>> q.fit(X)  # doctest: +SKIP
    QUANTTransformer(interval_depth=2, quantile_divisor=8)
    >>> q.transform(X)[0]  # doctest: +SKIP
    tensor([ 0.0000,  0.7724,  1.1476,  1.3206,  1.1908, -0.3842,  0.6883,  0.2584,
             0.0102,  0.0583, -1.6552,  2.1726, -0.1267, -0.7646, -0.7646,  1.6744,
             1.7010,  1.2805,  1.6744])
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "interval",
        "python_dependencies": "torch",
    }

    def __init__(self, interval_depth=6, quantile_divisor=4):
        self.interval_depth = interval_depth
        self.quantile_divisor = quantile_divisor

        super().__init__()

    def _fit(self, X, y=None):
        import torch
        import torch.nn.functional as F

        X = torch.tensor(X).float()

        if self.quantile_divisor < 1:
            raise ValueError("quantile_divisor must be >= 1")
        if self.interval_depth < 1:
            raise ValueError("interval_depth must be >= 1")

        representation_functions = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

        self.intervals_ = []
        for function in representation_functions:
            Z = function(X)
            self.intervals_.append(self._make_intervals(input_length=Z.shape[-1]))

        return self

    def _transform(self, X, y=None):
        import torch
        import torch.nn.functional as F

        X = torch.tensor(X).float()

        representation_functions = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

        Xt = []
        for index, function in enumerate(representation_functions):
            Z = function(X)
            features = []
            for a, b in self.intervals_[index]:
                features.append(self._find_quantiles(Z[..., a:b]).squeeze(1))
            Xt.append(torch.cat(features, -1))

        Xt = torch.cat(Xt, -1)

        if len(Xt.shape) == 2:
            return Xt.numpy()
        else:
            return np.reshape(Xt.numpy(), (Xt.shape[0], Xt.shape[1] * Xt.shape[2]))

    def _make_intervals(self, input_length):
        import torch

        exponent = min(self.interval_depth, int(np.log2(input_length)) + 1)
        intervals = []
        for n in 2 ** torch.arange(exponent):
            indices = torch.linspace(0, input_length, n + 1).long()
            intervals_n = torch.stack((indices[:-1], indices[1:]), 1)
            intervals.append(intervals_n)
            if n > 1 and intervals_n.diff().median() > 1:
                shift = int(np.ceil(input_length / n / 2))
                intervals.append(intervals_n[:-1] + shift)
        return torch.cat(intervals)

    def _find_quantiles(self, X):
        import torch

        n = X.shape[-1]

        if n == 1:
            return X
        else:
            num_quantiles = 1 + (n - 1) // self.quantile_divisor
            if num_quantiles == 1:
                return X.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
            else:
                quantiles = X.quantile(
                    torch.linspace(0, 1, num_quantiles), dim=-1
                ).permute(1, 2, 0)
                quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims=True)
                return quantiles

"""QUANT: A Minimalist Interval Method for Time Series Classification.

Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
https://arxiv.org/abs/2308.00928

Original code: https://github.com/angus924/quant
"""

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


class QUANTTransformer(BaseCollectionTransformer):
    """QUANT transform."""

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "interval",
        "python_dependencies": "torch",
    }

    def __init__(self, depth=6, div=4):
        self.depth = depth
        self.div = div

        super().__init__()

    def _fit(self, X, y=None):
        import torch
        import torch.nn.functional as F

        X = torch.tensor(X).float()

        if self.div < 1 or self.depth < 1:
            raise ValueError("depth and div must be >= 1")

        self.representation_functions = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

        self.intervals = []
        for function in self.representation_functions:
            Z = function(X)
            self.intervals.append(
                self._make_intervals(
                    input_length=Z.shape[-1],
                    depth=self.depth,
                )
            )

        return self

    def _transform(self, X):
        import torch

        X = torch.tensor(X).float()

        Xt = []
        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            features = []
            for a, b in self.intervals[index]:
                features.append(self._f_quantile(Z[..., a:b], div=self.div).squeeze(1))
            Xt.append(torch.cat(features, -1))

        return torch.cat(Xt, -1)

    @staticmethod
    def _make_intervals(input_length, depth):
        import torch

        exponent = min(depth, int(np.log2(input_length)) + 1)
        intervals = []
        for n in 2 ** torch.arange(exponent):
            indices = torch.linspace(0, input_length, n + 1).long()
            intervals_n = torch.stack((indices[:-1], indices[1:]), 1)
            intervals.append(intervals_n)
            if n > 1 and intervals_n.diff().median() > 1:
                shift = int(np.ceil(input_length / n / 2))
                intervals.append(intervals_n[:-1] + shift)
        return torch.cat(intervals)

    @staticmethod
    def _f_quantile(X, div=4):
        import torch

        n = X.shape[-1]

        if n == 1:
            return X
        else:
            num_quantiles = 1 + (n - 1) // div
            if num_quantiles == 1:
                return X.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
            else:
                quantiles = X.quantile(
                    torch.linspace(0, 1, num_quantiles), dim=-1
                ).permute(1, 2, 0)
                quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims=True)
                return quantiles

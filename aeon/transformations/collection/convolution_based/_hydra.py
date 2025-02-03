"""Hydra Transformer."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["HydraTransformer"]

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation._dependencies import _check_soft_dependencies


class HydraTransformer(BaseCollectionTransformer):
    """Hydra Transformer.

    The algorithm utilises convolutional kernels grouped into ``g`` groups per dilation
    with ``k`` kernels per group. It transforms input time series using these kernels
    and counts the kernels representing the closest match to the input at each time
    point. This counts for each group are then concatenated and returned.

    The algorithm combines aspects of both Rocket (convolutional approach)
    and traditional dictionary methods (pattern counting), It extracts features from
    both the base series and first-order differences of the series.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group.
    n_groups : int, default=64
        Number of groups per dilation.
    max_num_channels : int, default=8
        Maximum number of channels to use for each dilation.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    HydraClassifier
    MultiRocketHydraClassifier

    Notes
    -----
    Original code: https://github.com/angus924/hydra

    References
    ----------
    .. [1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing
        convolutional kernels for fast and accurate time series classification.
        Data Mining and Knowledge Discovery, pp.1-27.

    Examples
    --------
    >>> from aeon.transformations.collection.convolution_based import HydraTransformer
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, _ = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              random_state=0)
    >>> clf = HydraTransformer(random_state=0)  # doctest: +SKIP
    >>> clf.fit(X)  # doctest: +SKIP
    HydraTransformer(random_state=0)
    >>> clf.transform(X)[0]  # doctest: +SKIP
    tensor([0.6077, 1.3868, 0.2571,  ..., 1.0000, 1.0000, 2.0000])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(
        self, n_kernels=8, n_groups=64, max_num_channels=8, n_jobs=1, random_state=None
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.max_num_channels = max_num_channels
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y=None):
        import torch

        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        n_jobs = check_n_jobs(self.n_jobs)
        torch.set_num_threads(n_jobs)

        self._hydra = _HydraInternal(
            X.shape[2],
            X.shape[1],
            k=self.n_kernels,
            g=self.n_groups,
            max_num_channels=self.max_num_channels,
        )

    def _transform(self, X, y=None):
        return self._hydra(torch.tensor(X).float())


if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class _HydraInternal(nn.Module):
        """HYDRA torch internals."""

        def __init__(self, n_timepoints, n_channels, k=8, g=64, max_num_channels=8):
            super().__init__()

            self.k = k  # num kernels per group
            self.g = g  # num groups

            max_exponent = np.log2((n_timepoints - 1) / (9 - 1))  # kernel length = 9

            self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
            self.num_dilations = len(self.dilations)

            self.paddings = torch.div(
                (9 - 1) * self.dilations, 2, rounding_mode="floor"
            ).int()

            self.divisor = min(2, self.g)
            self.h = self.g // self.divisor

            self.W = torch.randn(
                self.num_dilations, self.divisor, self.k * self.h, 1, 9
            )
            self.W = self.W - self.W.mean(-1, keepdims=True)
            self.W = self.W / self.W.abs().sum(-1, keepdims=True)

            num_channels_per = np.clip(n_channels // 2, 2, max_num_channels)
            self.idx = [
                torch.randint(0, n_channels, (self.divisor, self.h, num_channels_per))
                for _ in range(self.num_dilations)
            ]

        # transform in batches of *batch_size*
        def batch(self, X, batch_size=256):
            num_examples = X.shape[0]
            if num_examples <= batch_size:
                return self(X)
            else:
                Z = []
                batches = torch.arange(num_examples).split(batch_size)
                for batch in batches:
                    Z.append(self(X[batch]))
                return torch.cat(Z)

        def forward(self, X):
            n_examples, n_channels, _ = X.shape

            if self.divisor > 1:
                diff_X = torch.diff(X)

            Z = []

            for dilation_index in range(self.num_dilations):
                d = self.dilations[dilation_index].item()
                p = self.paddings[dilation_index].item()

                for diff_index in range(self.divisor):
                    if n_channels > 1:  # Multivariate
                        _Z = F.conv1d(
                            (
                                X[:, self.idx[dilation_index][diff_index]].sum(2)
                                if diff_index == 0
                                else diff_X[
                                    :, self.idx[dilation_index][diff_index]
                                ].sum(2)
                            ),
                            self.W[dilation_index][diff_index],
                            dilation=d,
                            padding=p,
                            groups=self.h,
                        ).view(n_examples, self.h, self.k, -1)
                    else:  # Univariate
                        _Z = F.conv1d(
                            X if diff_index == 0 else diff_X,
                            self.W[dilation_index, diff_index],
                            dilation=d,
                            padding=p,
                        ).view(n_examples, self.h, self.k, -1)

                    max_values, max_indices = _Z.max(2)
                    count_max = torch.zeros(n_examples, self.h, self.k)

                    min_values, min_indices = _Z.min(2)
                    count_min = torch.zeros(n_examples, self.h, self.k)

                    count_max.scatter_add_(-1, max_indices, max_values)
                    count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                    Z.append(count_max)
                    Z.append(count_min)

            Z = torch.cat(Z, 1).view(n_examples, -1)

            return Z

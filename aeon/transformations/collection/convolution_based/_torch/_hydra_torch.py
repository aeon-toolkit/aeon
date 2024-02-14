"""HYDRA internals."""

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class _HydraInternal(nn.Module):
        """HYDRA torch internals."""

        def __init__(self, input_length, k=8, g=64):
            super().__init__()

            self.k = k  # num kernels per group
            self.g = g  # num groups

            max_exponent = np.log2((input_length - 1) / (9 - 1))  # kernel length = 9

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
            num_examples = X.shape[0]

            if self.divisor > 1:
                diff_X = torch.diff(X)

            Z = []

            for dilation_index in range(self.num_dilations):
                d = self.dilations[dilation_index].item()
                p = self.paddings[dilation_index].item()

                for diff_index in range(self.divisor):
                    _Z = F.conv1d(
                        X if diff_index == 0 else diff_X,
                        self.W[dilation_index, diff_index],
                        dilation=d,
                        padding=p,
                    ).view(num_examples, self.h, self.k, -1)

                    max_values, max_indices = _Z.max(2)
                    count_max = torch.zeros(num_examples, self.h, self.k)

                    min_values, min_indices = _Z.min(2)
                    count_min = torch.zeros(num_examples, self.h, self.k)

                    count_max.scatter_add_(-1, max_indices, max_values)
                    count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                    Z.append(count_max)
                    Z.append(count_min)

            Z = torch.cat(Z, 1).view(num_examples, -1)

            return Z

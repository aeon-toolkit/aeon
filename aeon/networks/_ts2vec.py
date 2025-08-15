"""Fully Convolutional Network (FCNNetwork)."""

__maintainer__ = ["GasperPetelin"]


class TS2VecNetwork:
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims,
        depth,
    ):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        super().__init__()

    def build_network(self, input_shape, **kwargs):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch import nn

        class ConvBlock(nn.Module):
            def __init__(
                self, in_channels, out_channels, kernel_size, dilation, final=False
            ):
                super().__init__()
                self.conv1 = SamePadConv(
                    in_channels, out_channels, kernel_size, dilation=dilation
                )
                self.conv2 = SamePadConv(
                    out_channels, out_channels, kernel_size, dilation=dilation
                )
                self.projector = (
                    nn.Conv1d(in_channels, out_channels, 1)
                    if in_channels != out_channels or final
                    else None
                )

            def forward(self, x):
                residual = x if self.projector is None else self.projector(x)
                x = F.gelu(x)
                x = self.conv1(x)
                x = F.gelu(x)
                x = self.conv2(x)
                return x + residual

        class DilatedConvEncoder(nn.Module):
            def __init__(self, in_channels, channels, kernel_size):
                super().__init__()
                self.net = nn.Sequential(
                    *[
                        ConvBlock(
                            channels[i - 1] if i > 0 else in_channels,
                            channels[i],
                            kernel_size=kernel_size,
                            dilation=2**i,
                            final=(i == len(channels) - 1),
                        )
                        for i in range(len(channels))
                    ]
                )

            def forward(self, x):
                return self.net(x)

        class SamePadConv(nn.Module):
            def __init__(
                self, in_channels, out_channels, kernel_size, dilation=1, groups=1
            ):
                super().__init__()
                self.receptive_field = (kernel_size - 1) * dilation + 1
                padding = self.receptive_field // 2
                self.conv = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                self.remove = 1 if self.receptive_field % 2 == 0 else 0

            def forward(self, x):
                out = self.conv(x)
                if self.remove > 0:
                    out = out[:, :, : -self.remove]
                return out

        class TSEncoder(nn.Module):
            def __init__(
                self,
                input_dims,
                output_dims,
                hidden_dims=64,
                depth=10,
                mask_mode="binomial",
            ):
                super().__init__()
                self.input_dims = input_dims
                self.output_dims = output_dims
                self.hidden_dims = hidden_dims
                self.mask_mode = mask_mode
                self.input_fc = nn.Linear(input_dims, hidden_dims)
                self.feature_extractor = DilatedConvEncoder(
                    hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
                )
                self.repr_dropout = nn.Dropout(p=0.1)

            def forward(self, x, mask=None):  # x: B x T x input_dims
                nan_mask = ~x.isnan().any(axis=-1)
                x[~nan_mask] = 0
                x = self.input_fc(x)  # B x T x Ch

                # generate & apply mask
                if mask is None:
                    if self.training:
                        mask = self.mask_mode
                    else:
                        mask = "all_true"

                if mask == "binomial":
                    mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
                elif mask == "continuous":
                    mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
                elif mask == "all_true":
                    mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
                elif mask == "all_false":
                    mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
                elif mask == "mask_last":
                    mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
                    mask[:, -1] = False

                mask &= nan_mask
                x[~mask] = 0

                # conv encoder
                x = x.transpose(1, 2)  # B x Ch x T
                x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
                x = x.transpose(1, 2)  # B x T x Co

                return x

        def generate_binomial_mask(B, T, p=0.5):
            return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(
                torch.bool
            )

        def generate_continuous_mask(B, T, n=5, mask_length=0.1):
            res = torch.full((B, T), True, dtype=torch.bool)
            if isinstance(n, float):
                n = int(n * T)
            n = max(min(n, T // 2), 1)

            if isinstance(mask_length, float):
                mask_length = int(mask_length * T)
            mask_length = max(mask_length, 1)

            for i in range(B):
                for _ in range(n):
                    t = np.random.randint(T - mask_length + 1)
                    res[i, t : t + mask_length] = False
            return res

        return TSEncoder(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
        )

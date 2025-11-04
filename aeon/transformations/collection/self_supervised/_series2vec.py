"""Series2Vec transformer for aeon, with selectable SSL losses.

Implements the 'Series2Vec' objective from the reference repo/paper, and adds
two lighter alternatives. Expected input shape after aeon preprocessing:
(n_cases, n_channels, n_timepoints). transform() returns (n_cases, 2*rep_size).

This version uses the PyTorch Soft-DTW implementation:
    aeon.distances.elastic.soft._torch_soft_dtw.soft_dtw_distance_torch
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from aeon.distances.elastic.soft._torch_soft_dtw import soft_dtw_distance_torch
from aeon.transformations.collection import BaseCollectionTransformer


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _lower_tri_mask(mat: torch.Tensor) -> torch.Tensor:
    """Boolean mask for strictly lower triangle of a square matrix (B,B)."""
    return torch.tril(torch.ones_like(mat, dtype=torch.bool), diagonal=-1)


def _normalize_vector(v: torch.Tensor) -> torch.Tensor:
    if v.numel() <= 1:
        return v
    vmin = torch.min(v)
    vmax = torch.max(v)
    return (v - vmin) / (vmax - vmin + 1e-8)


def _pair_indices(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return i,j indices for the lower-triangular pairs (strict) of an n×n matrix."""
    return torch.tril_indices(n, n, offset=-1)


def _euclidean_pairs(x: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Euclidean distances between pairs of flattened samples x[i], x[j]."""
    a = x[i].reshape(len(i), -1)
    b = x[j].reshape(len(j), -1)
    return torch.norm(a - b, dim=1)


def _fft_filter_default(x: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
    """Simple low-pass keep_ratio filter in frequency domain (B,C,T) -> (B,C,T)."""
    B, C, T = x.shape
    Xf = torch.fft.fft(x, dim=-1)
    k = int(math.ceil(keep_ratio * (T // 2)))  # keep DC..k and mirror
    mask = torch.zeros(T, device=x.device, dtype=torch.bool)
    mask[: k + 1] = True
    if k > 0:
        mask[-k:] = True
    Xf_filtered = torch.where(mask, Xf, torch.zeros_like(Xf))
    return torch.fft.ifft(Xf_filtered, dim=-1).real


# ---------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------
class SamePadConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.receptive_field: int = (kernel_size - 1) * dilation + 1
        padding: int = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove: int = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.projector: nn.Conv1d | None = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int) -> None:
        super().__init__()
        self.ConvEncoder = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvEncoder(x)


class DisjoinEncoder(nn.Module):
    def __init__(
        self, channel_size: int, emb_size: int, rep_size: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.temporal_CNN = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=[1, kernel_size], padding="valid"),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )
        self.spatial_CNN = nn.Sequential(
            nn.Conv2d(
                emb_size, emb_size, kernel_size=[channel_size, 1], padding="valid"
            ),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )
        self.rep_CNN = nn.Sequential(
            nn.Conv1d(emb_size, rep_size, kernel_size=3),
            nn.BatchNorm1d(rep_size),
            nn.GELU(),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal_CNN(x)  # (B, emb, C, T')
        x = self.spatial_CNN(x)  # (B, emb, 1, T')
        x = self.rep_CNN(x.squeeze(2))  # (B, rep, T'')
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)


# ---------------------------------------------------------------------
# Torch model
# ---------------------------------------------------------------------
class _Series2VecTorch(nn.Module):
    """Torch backbone + heads."""

    def __init__(
        self,
        in_channels: int,
        rep_size: int = 128,
        emb_size: int = 128,
        num_heads: int = 8,
        dim_ff: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 8,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_layer = DisjoinEncoder(
            in_channels, emb_size, rep_size, kernel_size=kernel_size
        )
        self.embed_layer_f = DisjoinEncoder(
            in_channels, emb_size, rep_size, kernel_size=kernel_size
        )

        self.layernorm1 = nn.LayerNorm(rep_size, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(rep_size, eps=1e-5)
        self.attn = nn.MultiheadAttention(
            rep_size, num_heads, dropout=dropout, batch_first=False
        )
        self.ff = nn.Sequential(
            nn.Linear(rep_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, rep_size),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gap_f = nn.AdaptiveAvgPool1d(1)

        self.head: nn.Linear | None = (
            nn.Linear(2 * rep_size, num_classes) if num_classes is not None else None
        )

    @torch.no_grad()
    def linear_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled embeddings (B, 2*rep)."""
        out_t = self.gap(self.embed_layer(x)).squeeze(-1)  # (B, rep)
        out_f = self.gap_f(self.embed_layer_f(torch.fft.fft(x).float())).squeeze(
            -1
        )  # (B, rep)
        return torch.cat([out_t, out_f], dim=1)  # (B, 2*rep)

    def pretrain_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B×B) pairwise Euclidean distance matrices between pooled reps."""
        # time stream
        xt = self.embed_layer(x)  # (B, rep, T)
        xt = self.gap(xt).permute(2, 0, 1)  # (1, B, rep)
        att, _ = self.attn(xt, xt, xt)  # (1, B, rep)
        att = self.layernorm1(att + xt)
        out_t = self.layernorm2(att + self.ff(att)).squeeze(0)  # (B, rep)

        # freq stream
        xf = self.embed_layer_f(torch.fft.fft(x).float())  # (B, rep, T)
        xf = self.gap_f(xf).squeeze(-1)  # (B, rep)

        dt = torch.cdist(out_t, out_t)  # (B, B)
        df = torch.cdist(xf, xf)  # (B, B)
        return dt, df

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear_prob(x)
        return z if self.head is None else self.head(z)


# ---------------------------------------------------------------------
# Aeon transformer
# ---------------------------------------------------------------------
class Series2Vec(BaseCollectionTransformer, nn.Module):
    """Series2Vec transformer with selectable SSL objective.

    Parameters
    ----------
    rep_size : int, default=128
    emb_size : int, default=128
    num_heads : int, default=8
    dim_ff : int, default=256
    dropout : float, default=0.1
    kernel_size : int, default=8
    epochs : int, default=20
    batch_size : int, default=64
    lr : float, default=1e-3
    weight_decay : float, default=0.0
    pretrain_weight : float, default=1.0
        If y is provided to fit(), the total loss is CE + pretrain_weight * SSL.
    device : Optional[str], default=None
    seed : Optional[int], default=None
    verbose : bool, default=False
    ssl_loss : Literal["series2vec","embed_euclidean","embed_softdtw"], default="series2vec"
        - "series2vec": align to SoftDTW(raw X) for time + Euclidean(filtered X) for freq.
        - "embed_euclidean": align Euclidean distances between pooled embeddings.
        - "embed_softdtw": align SoftDTW distances between sequence embeddings.
    sdtw_gamma : float, default=0.1
        Temperature for Soft-DTW.
    freq_filter_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], default=None
        Optional callable to prefilter X for the spectral target. If None, uses a
        simple low-pass FFT filter keeping 50% of low frequencies.
    """

    _tags = {
        "input_data_type": "Collection",
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "python_dependencies": "torch",
        "fit_is_empty": False,
        "requires_y": False,
    }

    def __init__(
        self,
        *,
        rep_size: int = 128,
        emb_size: int = 128,
        num_heads: int = 8,
        dim_ff: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 8,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pretrain_weight: float = 1.0,
        device: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
        ssl_loss: Literal[
            "series2vec", "embed_euclidean", "embed_softdtw"
        ] = "series2vec",
        sdtw_gamma: float = 0.1,
        freq_filter_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        BaseCollectionTransformer.__init__(self)
        nn.Module.__init__(self)

        # hparams
        self.rep_size = rep_size
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.kernel_size = kernel_size

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_weight = pretrain_weight

        self.device_str = device
        self.seed = seed
        self.verbose = verbose

        self.ssl_loss = ssl_loss
        self.sdtw_gamma = float(sdtw_gamma)
        self.freq_filter_fn = freq_filter_fn

        # fitted attrs
        self.model_: _Series2VecTorch | None = None
        self.device_: torch.device | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_idx_: dict[Any, int] | None = None

    # ----------------- utils ----------------- #
    def _select_device(self) -> torch.device:
        if self.device_str:
            return torch.device(self.device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _set_seed(self) -> None:
        if self.seed is None:
            return
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _build_model(
        self, in_channels: int, num_classes: int | None
    ) -> _Series2VecTorch:
        return _Series2VecTorch(
            in_channels=in_channels,
            rep_size=self.rep_size,
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            kernel_size=self.kernel_size,
            num_classes=num_classes,
        )

    @staticmethod
    def _np_to_tensor(X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(X, dtype=np.float32))

    @staticmethod
    def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[Any, int]]:
        classes = np.unique(y)
        mapping = {c: i for i, c in enumerate(classes)}
        return classes, np.vectorize(mapping.get)(y).astype(np.int64), mapping

    @staticmethod
    def _ssl_align_loss(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 on normalized distance vectors."""
        v1 = _normalize_vector(d1)
        v2 = _normalize_vector(d2)
        return F.smooth_l1_loss(v1, v2)

    def _freq_filter(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.freq_filter_fn(x)
            if self.freq_filter_fn is not None
            else _fft_filter_default(x, keep_ratio=0.5)
        )

    def _pairwise_softdtw_raw(
        self, xb: torch.Tensor, i: torch.Tensor, j: torch.Tensor
    ) -> torch.Tensor:
        """Soft-DTW over raw series for selected pairs. xb: (B,C,T). Returns (M,) distances."""
        vals: list[torch.Tensor] = []
        with torch.no_grad():  # targets don't require grad
            for ii, jj in zip(i.tolist(), j.tolist()):
                d = soft_dtw_distance_torch(
                    xb[ii], xb[jj], gamma=self.sdtw_gamma
                )  # xb[ii]: (C,T)
                vals.append(d)
        return torch.stack(vals).to(device=xb.device, dtype=xb.dtype)

    # ----------------- Aeon hooks ----------------- #
    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Series2Vec:
        """Train the model with the selected SSL objective. If y is provided, trains CE + λ·SSL."""
        self._set_seed()
        self.device_ = self._select_device()
        n_channels: int = self.metadata_["n_channels"]

        num_classes: int | None = None
        y_idx_np: np.ndarray | None = None
        if y is not None:
            self.classes_, y_idx_np, self.class_to_idx_ = self._encode_labels(
                np.asarray(y)
            )
            num_classes = len(self.classes_)

        self.model_ = self._build_model(n_channels, num_classes).to(self.device_)
        self.model_.train()

        X_t = self._np_to_tensor(X)
        if y_idx_np is None:
            dataset: torch.utils.data.Dataset[tuple[torch.Tensor]] = (
                torch.utils.data.TensorDataset(X_t)
            )
        else:
            dataset = torch.utils.data.TensorDataset(X_t, torch.from_numpy(y_idx_np))

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            total_loss: float = 0.0

            for batch in loader:
                if y_idx_np is None:
                    (xb,) = batch
                    yb_t: torch.Tensor | None = None
                else:
                    xb, yb_t = batch
                    yb_t = yb_t.to(self.device_, dtype=torch.long)

                xb = xb.to(self.device_)
                opt.zero_grad()

                # ---------- SSL term ----------
                if self.ssl_loss == "series2vec":
                    # predicted distances from reps
                    dt_pred, df_pred = self.model_.pretrain_forward(xb)  # (B,B), (B,B)
                    mask = _lower_tri_mask(dt_pred)
                    dt_pred_v = dt_pred[mask]
                    df_pred_v = df_pred[mask]

                    # targets: SoftDTW on raw, Euclidean on filtered-freq
                    i_idx, j_idx = _pair_indices(xb.size(0))
                    dt_target = self._pairwise_softdtw_raw(xb, i_idx, j_idx)  # (M,)
                    xfilt = self._freq_filter(xb)
                    df_target = _euclidean_pairs(xfilt, i_idx, j_idx)  # (M,)

                    loss_ssl = self._ssl_align_loss(
                        dt_pred_v, dt_target
                    ) + self._ssl_align_loss(df_pred_v, df_target)

                elif self.ssl_loss == "embed_euclidean":
                    # Euclidean between pooled embeddings (time/freq)
                    dt_pred, df_pred = self.model_.pretrain_forward(xb)
                    mask = _lower_tri_mask(dt_pred)
                    dt_pred_v = dt_pred[mask]
                    df_pred_v = df_pred[mask]
                    with torch.no_grad():
                        dt_tgt = dt_pred.detach()
                        df_tgt = df_pred.detach()
                    loss_ssl = F.mse_loss(dt_pred_v, dt_tgt[mask]) + F.mse_loss(
                        df_pred_v, df_tgt[mask]
                    )

                elif self.ssl_loss == "embed_softdtw":
                    # SoftDTW between sequence embeddings per stream
                    # sequences as (T, C=rep)
                    xt_seq = (
                        self.model_.embed_layer(xb).transpose(1, 2).contiguous()
                    )  # (B, T, rep)
                    xf_seq = (
                        self.model_.embed_layer_f(torch.fft.fft(xb).float())
                        .transpose(1, 2)
                        .contiguous()
                    )  # (B, T, rep)
                    i_idx, j_idx = _pair_indices(xb.size(0))

                    def sdtw_pairs(seq: torch.Tensor) -> torch.Tensor:
                        vals: list[torch.Tensor] = []
                        with torch.no_grad():
                            for ii, jj in zip(i_idx.tolist(), j_idx.tolist()):
                                aa = seq[ii].transpose(0, 1)  # (rep, T)
                                bb = seq[jj].transpose(0, 1)  # (rep, T)
                                vals.append(
                                    soft_dtw_distance_torch(
                                        aa, bb, gamma=self.sdtw_gamma
                                    )
                                )
                        return torch.stack(vals).to(device=seq.device, dtype=seq.dtype)

                    dt_pred, df_pred = self.model_.pretrain_forward(xb)
                    loss_ssl = self._ssl_align_loss(
                        dt_pred[_lower_tri_mask(dt_pred)], sdtw_pairs(xt_seq)
                    ) + self._ssl_align_loss(
                        df_pred[_lower_tri_mask(df_pred)], sdtw_pairs(xf_seq)
                    )

                else:
                    raise ValueError(f"Unknown ssl_loss={self.ssl_loss!r}")

                # ---------- Supervised term ----------
                if yb_t is not None and self.model_.head is not None:
                    logits = self.model_(xb)
                    loss_ce = F.cross_entropy(logits, yb_t)
                    loss = loss_ce + (
                        self.pretrain_weight * loss_ssl
                        if self.pretrain_weight > 0
                        else 0.0
                    )
                else:
                    loss = loss_ssl

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=4.0)
                opt.step()

                total_loss += float(loss.detach().cpu())

            if self.verbose:
                denom = max(1, len(loader))
                print(
                    f"[Series2Vec] epoch {epoch+1}/{self.epochs} - loss: {total_loss/denom:.4f}"
                )

        self.model_.eval()
        return self

    def _transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return embeddings (n_cases, 2*rep_size)."""
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model not initialised. Call fit first.")
        self.model_.eval()

        X_t = self._np_to_tensor(X)
        loader = torch.utils.data.DataLoader(
            X_t, batch_size=self.batch_size, shuffle=False
        )
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device_)
                z = self.model_.linear_prob(xb)
                outs.append(z.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    def get_fitted_params(self) -> dict[str, Any]:
        return {
            "classes_": None if self.classes_ is None else self.classes_.copy(),
            "class_to_idx_": (
                None if self.class_to_idx_ is None else dict(self.class_to_idx_)
            ),
            "device_": None if self.device_ is None else str(self.device_),
            "rep_size": self.rep_size,
            "emb_size": self.emb_size,
            "ssl_loss": self.ssl_loss,
        }

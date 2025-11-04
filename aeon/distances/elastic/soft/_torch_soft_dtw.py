from __future__ import annotations

import torch


def _as_channels_last(x: torch.Tensor) -> torch.Tensor:
    """Accept (T,) or (C, T); return (C, T) without extra validation."""
    return x.unsqueeze(0) if x.dim() == 1 else x  # assume 1D -> univariate


def _softmin3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Soft minimum over three values (differentiable)."""
    return -gamma * torch.logsumexp(
        torch.stack((-a / gamma, -b / gamma, -c / gamma), dim=0), dim=0
    )


def soft_dtw_cost_matrix_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Soft-DTW accumulated cost matrix (no constraints).
    x: (T,) or (C, T); y: (U,) or (C, U). Returns (T, U).
    """
    X = _as_channels_last(x)  # (C, T)
    Y = _as_channels_last(y)  # (C, U)
    _, T = X.shape
    _, U = Y.shape

    R = torch.full((T + 1, U + 1), float("inf"), device=X.device, dtype=X.dtype)
    R[0, 0] = 0.0

    for i in range(1, T + 1):
        xi = X[:, i - 1]  # (C,)
        for j in range(1, U + 1):
            yj = Y[:, j - 1]  # (C,)
            d = torch.sum((xi - yj) ** 2)
            R[i, j] = d + _softmin3(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma)

    return R[1:, 1:]  # (T, U)


def soft_dtw_distance_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Soft-DTW value s_γ(x, y) (scalar tensor). Differentiable w.r.t. x and y.
    """
    C = soft_dtw_cost_matrix_torch(x, y, gamma=gamma)
    return C[-1, -1]

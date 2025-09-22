# _torch_msm.py
import torch


# --- keep existing helpers (unchanged API) ------------------------------------
def _softmin3(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float
) -> torch.Tensor:
    """softmin(a,b,c) = -γ logsumexp([-a/γ, -b/γ, -c/γ])"""
    stack = torch.stack([-a / gamma, -b / gamma, -c / gamma], dim=0)
    return -gamma * torch.logsumexp(stack, dim=0)


def _softmin2_t(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """softmin(a,b) in torch"""
    stack = torch.stack([-a / gamma, -b / gamma], dim=0)
    return -gamma * torch.logsumexp(stack, dim=0)


def _trans_cost_t(
    x_val: torch.Tensor,
    y_prev: torch.Tensor,
    z_other: torch.Tensor,
    c: float,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """
    Differentiable relaxation of MSM 'independent' transition:
      if x between y and z -> cost = c
      else                 -> cost = c + min((x-y)^2, (x-z)^2)
    Using: g = sigmoid(alpha * (-(x-y)*(x-z))) and softmin with temperature gamma.
    """
    a = x_val - y_prev
    b = x_val - z_other
    g = torch.sigmoid(alpha * (-(a * b)))  # ~1 when between, ~0 otherwise
    base = _softmin2_t(a * a, b * b, gamma)  # ≈ min((x-y)^2, (x-z)^2)
    return c + (1.0 - g) * base


@torch.no_grad()
def _check_inputs(x: torch.Tensor, y: torch.Tensor, gamma: float):
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for a differentiable soft minimum.")
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("x and y must be 1D tensors of shape (T,).")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("x and y must be floating dtype tensors.")


# --- new internal helpers (private) -------------------------------------------
def _softmin3_scalar(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float):
    """Scalar (0-d tensor) softmin3 with fewer tiny ops than stack+logsumexp."""
    s1 = -a / gamma
    s2 = -b / gamma
    s3 = -c / gamma
    m = torch.maximum(s1, torch.maximum(s2, s3))
    z = torch.exp(s1 - m) + torch.exp(s2 - m) + torch.exp(s3 - m)
    return -gamma * (torch.log(z) + m)


def _softmin2_vec_scalar_first(
    t1_scalar: torch.Tensor, t2_vec: torch.Tensor, gamma: float
):
    """Vectorized softmin2 when first arg is scalar and second is vector."""
    s1 = -t1_scalar / gamma
    s2 = -t2_vec / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    return -gamma * (torch.log(z) + m)


def _trans_cost_row_up(xi, xim1, y_slice, c: float, alpha: float, gamma: float):
    # x_val=xi (scalar), y_prev=xim1 (scalar), z_other=yj (vector)
    a = xi - xim1  # scalar
    b = xi - y_slice  # vector
    s = -(a * b)  # vector
    g = torch.sigmoid(alpha * s)  # vector
    d_same = a * a  # scalar
    d_cross = b * b  # vector
    base = _softmin2_vec_scalar_first(d_same, d_cross, gamma)  # vector
    return c + (1.0 - g) * base


def _trans_cost_row_left(
    y_slice, y_prev_slice, xi, c: float, alpha: float, gamma: float
):
    # x_val=yj (vector), y_prev=y_{j-1} (vector), z_other=xi (scalar)
    a = y_slice - y_prev_slice  # vector
    b = y_slice - xi  # vector
    s = -(a * b)  # vector
    g = torch.sigmoid(alpha * s)  # vector
    d_same = a * a  # vector
    d_cross = b * b  # vector
    # vectorized softmin2
    s1 = -d_same / gamma
    s2 = -d_cross / gamma
    m = torch.maximum(s1, s2)
    z = torch.exp(s1 - m) + torch.exp(s2 - m)
    base = -gamma * (torch.log(z) + m)
    return c + (1.0 - g) * base


# --- public API (same name/signature) -----------------------------------------
def soft_msm_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    c: float = 1.0,
    gamma: float = 1.0,  # > 0
    alpha: float = 25.0,  # gate sharpness
    window: (
        int | None
    ) = None,  # Sakoe–Chiba half-width (cells with |i-j|>window are forbidden)
) -> torch.Tensor:
    """
    Differentiable soft-MSM distance between 1D series x (len n) and y (len m),
    matching the Numba formulation (gate × softmin, squared distances, same boundaries).
    Returns a scalar tensor suitable for .backward().
    """
    _check_inputs(x, y, gamma)
    device, dtype = x.device, x.dtype
    n, m = x.numel(), y.numel()

    # DP table like before: C[0,0] pays the (x0 - y0)^2 match
    C = torch.full((n, m), float("inf"), device=device, dtype=dtype)
    C[0, 0] = (x[0] - y[0]) ** 2

    def in_band(i: int, j: int) -> bool:
        return True if window is None else (abs(i - j) <= window)

    # First column (vertical)
    for i in range(1, n):
        if in_band(i, 0):
            a = x[i] - x[i - 1]
            b = x[i] - y[0]
            g = torch.sigmoid(alpha * (-(a * b)))
            s1 = -(a * a) / gamma
            s2 = -(b * b) / gamma
            mm = torch.maximum(s1, s2)
            z = torch.exp(s1 - mm) + torch.exp(s2 - mm)
            base = -gamma * (torch.log(z) + mm)
            trans = c + (1.0 - g) * base
            C[i, 0] = C[i - 1, 0] + trans

    # First row (horizontal)
    for j in range(1, m):
        if in_band(0, j):
            a = y[j] - y[j - 1]
            b = y[j] - x[0]
            g = torch.sigmoid(alpha * (-(a * b)))
            s1 = -(a * a) / gamma
            s2 = -(b * b) / gamma
            mm = torch.maximum(s1, s2)
            z = torch.exp(s1 - mm) + torch.exp(s2 - mm)
            base = -gamma * (torch.log(z) + mm)
            trans = c + (1.0 - g) * base
            C[0, j] = C[0, j - 1] + trans

    # Main DP (row-wise vectorized costs, scalar recurrence)
    for i in range(1, n):
        j_lo = 1 if window is None else max(1, i - window)
        j_hi = m - 1 if window is None else min(m - 1, i + window)
        if j_lo > j_hi:
            continue

        xi, xim1 = x[i], x[i - 1]
        y_cur = y[j_lo : j_hi + 1]  # [L]
        y_prev = y[j_lo - 1 : j_hi]  # [L]

        up_cost = _trans_cost_row_up(xi, xim1, y_cur, c, alpha, gamma)  # [L]
        left_cost = _trans_cost_row_left(y_cur, y_prev, xi, c, alpha, gamma)  # [L]
        match = (xi - y_cur).pow(2)  # [L]

        Cijm1 = C[i, j_lo - 1]
        for t in range(y_cur.numel()):
            j = j_lo + t
            d_diag = C[i - 1, j - 1] + match[t]
            d_up = C[i - 1, j] + up_cost[t]
            d_left = Cijm1 + left_cost[t]
            Cij = _softmin3_scalar(d_diag, d_up, d_left, gamma)
            C[i, j] = Cij
            Cijm1 = Cij

    return C[n - 1, m - 1]


if __name__ == "__main__":
    import time

    import numpy as np

    from aeon.distances.elastic.soft._soft_msm import (
        soft_msm_grad_x as soft_msm_grad_x_old,
    )
    from aeon.testing.data_generation import make_example_2d_numpy_series

    shared = dict(c=0.5, gamma=0.2, alpha=1.0, window=None)

    n_timepoint = 1000

    x = make_example_2d_numpy_series(n_timepoint, 1, random_state=1)
    y = make_example_2d_numpy_series(n_timepoint, 1, random_state=2)
    x_np, y_np = x[0], y[0]

    # ---------- Helper(s) ----------
    def _sync(dev: torch.device):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    def run_fb(x_arr, y_arr, device, dtype, warmup=False):
        # optional warmup (build kernels, etc.) not timed
        if warmup:
            xw = torch.tensor(x_arr, dtype=dtype, device=device, requires_grad=True)
            yw = torch.tensor(y_arr, dtype=dtype, device=device, requires_grad=False)
            Dw = soft_msm_torch(xw, yw, **shared)
            Dw.backward()
            _sync(torch.device(device))
            del xw, yw, Dw

        x_t = torch.tensor(x_arr, dtype=dtype, device=device, requires_grad=True)
        y_t = torch.tensor(y_arr, dtype=dtype, device=device, requires_grad=False)

        _sync(torch.device(device))
        t0 = time.perf_counter()
        D = soft_msm_torch(x_t, y_t, **shared)
        D.backward()
        _sync(torch.device(device))
        t1 = time.perf_counter()

        D_val = D.detach().cpu().item()
        grad = x_t.grad.detach().cpu().numpy()
        return D_val, grad, (t1 - t0)

    # ---------- CPU (fp64) vs Numba (ground truth) ----------
    x_cpu64 = torch.tensor(x_np, dtype=torch.float64, device="cpu", requires_grad=True)
    y_cpu64 = torch.tensor(y_np, dtype=torch.float64, device="cpu", requires_grad=False)
    D_cpu64 = soft_msm_torch(x_cpu64, y_cpu64, **shared)
    D_cpu64.backward()
    grad_cpu64 = x_cpu64.grad.detach().cpu().numpy()

    grad_numba, D_numba = soft_msm_grad_x_old(x, y, **shared)

    print(
        "CPU vs Numba -- D match:",
        np.allclose(D_cpu64.item(), D_numba, rtol=1e-8, atol=1e-8),
    )
    print(
        "CPU vs Numba -- grad allclose:",
        np.allclose(grad_cpu64, grad_numba, rtol=1e-8, atol=1e-8),
    )

    # ---------- CPU (fp32) vs MPS (fp32) with timings ----------
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    # CPU fp32
    D_cpu32, g_cpu32, t_cpu32 = run_fb(x_np, y_np, device="cpu", dtype=torch.float32)

    if has_mps:
        # Warmup once on MPS to avoid first-time JIT overhead in timing
        _ = run_fb(x_np, y_np, device="mps", dtype=torch.float32, warmup=True)
        D_mps32, g_mps32, t_mps32 = run_fb(
            x_np, y_np, device="mps", dtype=torch.float32
        )

        rtol, atol = 1e-5, 1e-6
        print(
            f"\nCPU32 vs MPS32 -- D match:  {np.allclose(D_cpu32, D_mps32, rtol=rtol, atol=atol)}"
        )
        print(
            f"CPU32 vs MPS32 -- grad allclose: {np.allclose(g_cpu32, g_mps32, rtol=rtol, atol=atol)}"
        )
        print(
            f"CPU32 vs MPS32 -- grad max abs diff: {float(np.max(np.abs(g_cpu32 - g_mps32)))}"
        )

        print(f"\nTiming (forward + backward):")
        print(f"  CPU  fp32: {t_cpu32*1000:.3f} ms   D={D_cpu32:.8f}")
        print(f"  MPS  fp32: {t_mps32*1000:.3f} ms   D={D_mps32:.8f}")
    else:
        print("\nMPS not available; showing CPU fp32 only:")
        print(f"  CPU fp32 time: {t_cpu32*1000:.3f} ms   D={D_cpu32:.8f}")

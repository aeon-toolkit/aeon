"""EvoForest-TS-WM: a frozen, interpretable, closed-form time-series feature transform.

No learned weights; discovered by EvoForest under a world-model objective. numba-only;
the seeded banks are embedded (no torch, no data file). See the class docstring.
"""

__maintainer__ = []
__all__ = ["EvoForestTSWM"]

import base64
import io
import math
import zlib
from functools import lru_cache

import numpy as np
from numba import njit, prange

from aeon.transformations.collection.base import BaseCollectionTransformer

# ===================== frozen seeded banks (embedded; ~8 KB) =====================
_B64 = (
    "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"
)


def load_banks():
    """Return the 6 banks as a dict of float64 numpy arrays."""
    raw = zlib.decompress(base64.b64decode(_B64))
    with np.load(io.BytesIO(raw)) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


# ===================== njit phi kernels (MiniRocket-style) =====================
L = 64
_EPS = 1e-8


# ---------- constant banks (numpy, built once) ----------
def build_consts(banks=None):
    if banks is None:
        banks = load_banks()
    trf_mu = np.asarray(banks["trf_mu"], np.float64)
    trf_sig = np.asarray(banks["trf_sig"], np.float64)
    t = np.arange(L, dtype=np.float64) / L
    win = np.exp(
        -((t[None, :] - trf_mu[:, None]) ** 2) / (2 * trf_sig[:, None] ** 2 + _EPS)
    )
    GW = win / (win.sum(1, keepdims=True) + _EPS)  # (12, L) normalised windows

    nn = np.arange(L, dtype=np.float64)
    kr = np.arange(L // 2 + 1, dtype=np.float64)
    a = 2 * np.pi * np.outer(kr, nn) / L
    Cr, Ci = np.cos(a), np.sin(a)  # rfft cos/sin (33, L)
    kf = np.arange(L, dtype=np.float64)
    af = 2 * np.pi * np.outer(kf, nn) / L
    Fc, Fs = np.cos(af), np.sin(af)  # full DFT cos/sin (L, L)
    hfir = np.where(
        kf == 0, 1.0, np.where(kf < L / 2, 2.0, np.where(kf == L // 2, 1.0, 0.0))
    )

    crf = np.ascontiguousarray(
        np.asarray(banks["crf_w"], np.float64)[:, 0, :]
    )  # (16, 9)
    x = np.linspace(-7.0, 7.0, 15)
    Rk = np.empty((4, 15), np.float64)
    for i, s in enumerate((1.5, 2.5, 4.0, 6.0)):
        r = (1.0 - (x / s) ** 2) * np.exp(-(x**2) / (2 * s * s))
        Rk[i] = (r - r.mean()) / (np.abs(r).sum() + _EPS)
    srfW = np.ascontiguousarray(np.asarray(banks["srf_W"], np.float64))  # (12,6,12)
    srfb = np.ascontiguousarray(np.asarray(banks["srf_b"], np.float64))  # (12,6)
    srfu = np.ascontiguousarray(np.asarray(banks["srf_u"], np.float64))  # (12,6)
    HYW = np.ascontiguousarray(np.asarray(banks["hydra_w"], np.float64))  # (2,4,9)
    return (GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW)


# ---------- njit helpers ----------
@njit(cache=True)
def _quantile_lin(sorted_x, q):
    n = sorted_x.shape[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_x[n - 1]
    return sorted_x[lo] + frac * (sorted_x[lo + 1] - sorted_x[lo])


@njit(cache=True)
def _conv_same(w, ker, dilation):
    """length-L cross-correlation, 'same' padding = dilation*(k//2). Returns (L,)."""
    Lw = w.shape[0]
    k = ker.shape[0]
    pad = dilation * (k // 2)
    out = np.zeros(Lw, np.float64)
    for t in range(Lw):  # out_len == L for these configs
        acc = 0.0
        for j in range(k):
            idx = t + j * dilation - pad
            if 0 <= idx < Lw:
                acc += ker[j] * w[idx]
        out[t] = acc
    return out


@njit(cache=True)
def _phi_one(w, GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW):
    """One length-L patch -> 173 features, filled in family order."""
    n = w.shape[0]
    out = np.empty(173, np.float64)
    o = 0

    mean = 0.0
    for i in range(n):
        mean += w[i]
    mean /= n
    c = w - mean
    ss = 0.0
    for i in range(n):
        ss += c[i] * c[i]
    var1 = ss / (n - 1)  # unbiased (ddof=1), matches torch
    std1 = math.sqrt(var1)
    sden = std1 if std1 > _EPS else _EPS

    sw = np.sort(w)
    q25 = _quantile_lin(sw, 0.25)
    q50 = _quantile_lin(sw, 0.50)
    q75 = _quantile_lin(sw, 0.75)
    dw = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        dw[i] = w[i + 1] - w[i]

    # ===== family 0: stats (12) =====
    skew = 0.0
    kurt = 0.0
    for i in range(n):
        skew += c[i] ** 3
        kurt += c[i] ** 4
    skew = (skew / n) / (sden**3 + _EPS)
    kurt = (kurt / n) / (sden**4 + _EPS) - 3.0
    a1n = 0.0
    d0 = 0.0
    d1 = 0.0
    for i in range(n - 1):
        a1n += c[i] * c[i + 1]
        d0 += c[i] * c[i]
        d1 += c[i + 1] * c[i + 1]
    ac1 = a1n / (math.sqrt(d0) * math.sqrt(d1) + _EPS)
    absdiff = 0.0
    for i in range(n - 1):
        absdiff += abs(dw[i])
    absdiff /= n - 1
    st = np.empty(12, np.float64)
    st[0] = mean
    st[1] = sden
    st[2] = skew
    st[3] = kurt
    st[4] = q25
    st[5] = q50
    st[6] = q75
    st[7] = q75 - q25
    st[8] = ac1
    st[9] = absdiff
    st[10] = sw[0]
    st[11] = sw[n - 1]
    for i in range(12):
        out[o] = st[i]
        o += 1

    # ===== family 1: srf_mlp (12) =====
    for kk in range(12):
        acc = 0.0
        for dd in range(6):
            h = srfb[kk, dd]
            for mm in range(12):
                h += st[mm] * srfW[kk, dd, mm]
            if h < 0.0:
                h = 0.0
            acc += h * srfu[kk, dd]
        out[o] = acc
        o += 1

    # ===== family 2: autocorr lags 1,2 (2) =====
    a2n = 0.0
    a2d = 0.0
    for i in range(n - 2):
        a2n += c[i] * c[i + 2]
        a2d += c[i] * c[i]
    out[o] = a1n / (d0 + _EPS)
    o += 1
    out[o] = a2n / (a2d + _EPS)
    o += 1

    # ----- rfft magnitude (shared by spectral & fftbands) -----
    nb = Cr.shape[0]  # 33
    mag = np.empty(nb, np.float64)
    for kb in range(nb):
        re = 0.0
        im = 0.0
        for i in range(n):
            re += w[i] * Cr[kb, i]
            im -= w[i] * Ci[kb, i]
        mag[kb] = math.sqrt(re * re + im * im)

    # ===== family 3: spectral centroid + entropy (2) =====
    psum = 0.0
    for kb in range(1, nb):
        psum += mag[kb]
    psum += _EPS
    cent = 0.0
    ent = 0.0
    for j in range(nb - 1):
        pj = mag[j + 1] / psum
        cent += j * pj
        ent -= pj * math.log(pj + 1e-12)
    out[o] = cent
    o += 1
    out[o] = ent
    o += 1

    # ===== family 4: turning (2) =====
    tp = 0.0
    for i in range(n - 2):
        if dw[i + 1] * dw[i] < 0.0:
            tp += 1.0
    fpos = 0.0
    for i in range(n - 1):
        if dw[i] > 0.0:
            fpos += 1.0
    out[o] = tp / (n - 2)
    o += 1
    out[o] = fpos / (n - 1)
    o += 1

    # ===== family 5: trf_gausswin (12) = w @ GW.T =====
    for kk in range(12):
        acc = 0.0
        for i in range(n):
            acc += w[i] * GW[kk, i]
        out[o] = acc
        o += 1

    # conv outputs at dilations 2 and 4 (cached for crf_ppv / crf_max / conv_position)
    conv2 = np.empty((16, n), np.float64)
    conv4 = np.empty((16, n), np.float64)
    for ci in range(16):
        conv2[ci] = _conv_same(w, crf[ci], 2)
        conv4[ci] = _conv_same(w, crf[ci], 4)

    # ===== family 6: crf_ppv (32) = ppv@dil2 (16) then ppv@dil4 (16) =====
    for ci in range(16):
        p = 0.0
        for t in range(n):
            if conv2[ci, t] > 0.0:
                p += 1.0
        out[o] = p / n
        o += 1
    for ci in range(16):
        p = 0.0
        for t in range(n):
            if conv4[ci, t] > 0.0:
                p += 1.0
        out[o] = p / n
        o += 1

    # ===== family 7: hilbert_env (2) =====
    yre = np.empty(L, np.float64)
    yim = np.empty(L, np.float64)
    for kb in range(L):
        re = 0.0
        im = 0.0
        for i in range(n):
            re += w[i] * Fc[kb, i]
            im -= w[i] * Fs[kb, i]
        yre[kb] = re * hfir[kb]
        yim[kb] = im * hfir[kb]
    emean = 0.0
    env = np.empty(L, np.float64)
    for ni in range(L):
        ir = 0.0
        ii = 0.0
        for kb in range(L):
            ir += yre[kb] * Fc[kb, ni] - yim[kb] * Fs[kb, ni]
            ii += yre[kb] * Fs[kb, ni] + yim[kb] * Fc[kb, ni]
        ir /= L
        ii /= L
        env[ni] = math.sqrt(ir * ir + ii * ii)
        emean += env[ni]
    emean /= L
    evar = 0.0
    eabs = 0.0
    for i in range(L):
        evar += (env[i] - emean) ** 2
    for i in range(L - 1):
        eabs += abs(env[i + 1] - env[i])
    out[o] = math.sqrt(evar / (L - 1)) / (emean + _EPS)
    o += 1
    out[o] = (eabs / (L - 1)) / (emean + _EPS)
    o += 1

    # ===== family 8: crf_max (32) = max@dil2 (16) then max@dil4 (16) =====
    for ci in range(16):
        m = conv2[ci, 0]
        for t in range(1, n):
            if conv2[ci, t] > m:
                m = conv2[ci, t]
        out[o] = m
        o += 1
    for ci in range(16):
        m = conv4[ci, 0]
        for t in range(1, n):
            if conv4[ci, t] > m:
                m = conv4[ci, t]
        out[o] = m
        o += 1

    # ===== family 9: morphology_updown (4) =====
    pmax = 0.0
    nmax = 0.0
    pe = 0.0
    ne = 0.0
    for i in range(n - 1):
        d = dw[i]
        if d > 0.0:
            if d > pmax:
                pmax = d
            pe += d * d
        elif d < 0.0:
            if -d > nmax:
                nmax = -d
            ne += d * d
    out[o] = pmax
    o += 1
    out[o] = nmax
    o += 1
    out[o] = math.log((pmax + 1e-6) / (nmax + 1e-6))
    o += 1
    out[o] = math.log((pe + 1e-6) / (ne + 1e-6))
    o += 1

    # ===== family 10: fftbands (6) =====
    s2 = 0.0
    pw = np.empty(nb - 1, np.float64)
    for j in range(nb - 1):
        pw[j] = mag[j + 1] * mag[j + 1]
        s2 += pw[j]
    s2 += _EPS
    for b0 in range(0, nb - 1, 6):
        b1 = b0 + 6
        if b1 > nb - 1:
            b1 = nb - 1
        acc = 0.0
        for j in range(b0, b1):
            acc += pw[j] / s2
        if acc < 1e-8:
            acc = 1e-8
        out[o] = math.log(acc)
        o += 1

    # ===== family 11: perm_entropy + Hjorth mobility/complexity (3) =====
    H = np.zeros(8, np.float64)
    for i in range(n - 2):
        a = w[i]
        b = w[i + 1]
        cc = w[i + 2]
        code = 0
        if a < b:
            code += 4
        if b < cc:
            code += 2
        if a < cc:
            code += 1
        H[code] += 1.0
    pent = 0.0
    for k in range(8):
        Hk = H[k] / (n - 2)
        pent -= Hk * math.log(Hk + 1e-12)
    pent /= math.log(6.0)
    # var1(d1) and var1(d2), ddof=1
    md = 0.0
    for i in range(n - 1):
        md += dw[i]
    md /= n - 1
    vd1 = 0.0
    for i in range(n - 1):
        vd1 += (dw[i] - md) ** 2
    vd1 /= n - 2
    m2d = 0.0
    d2 = np.empty(n - 2, np.float64)
    for i in range(n - 2):
        d2[i] = w[i + 2] - 2.0 * w[i + 1] + w[i]
        m2d += d2[i]
    m2d /= n - 2
    vd2 = 0.0
    for i in range(n - 2):
        vd2 += (d2[i] - m2d) ** 2
    vd2 /= n - 3
    mob = math.sqrt((vd1 + _EPS) / (var1 + _EPS))
    comp = math.sqrt((vd2 + _EPS) / (vd1 + _EPS)) / (mob + _EPS)
    out[o] = pent
    o += 1
    out[o] = mob
    o += 1
    out[o] = comp
    o += 1

    # ===== family 12: curvature (3) =====
    cabsm = 0.0
    cabsx = 0.0
    cpe = 0.0
    cne = 0.0
    for i in range(n - 2):
        cc = d2[i]
        ac = abs(cc)
        cabsm += ac
        if ac > cabsx:
            cabsx = ac
        if cc > 0.0:
            cpe += cc * cc
        elif cc < 0.0:
            cne += cc * cc
    out[o] = cabsm / (n - 2)
    o += 1
    out[o] = cabsx
    o += 1
    out[o] = math.log((cpe + 1e-6) / (cne + 1e-6))
    o += 1

    # ===== family 13: conv_position (3) from dil2 argmax over 16 kernels =====
    pmean = 0.0
    posv = np.empty(16, np.float64)
    for ci in range(16):
        m = conv2[ci, 0]
        am = 0
        for t in range(1, n):
            if conv2[ci, t] > m:
                m = conv2[ci, t]
                am = t
        posv[ci] = am / n
        pmean += posv[ci]
    pmean /= 16
    pvar = 0.0
    pmn = posv[0]
    pmx = posv[0]
    for ci in range(16):
        pvar += (posv[ci] - pmean) ** 2
        if posv[ci] < pmn:
            pmn = posv[ci]
        if posv[ci] > pmx:
            pmx = posv[ci]
    out[o] = pmean
    o += 1
    out[o] = math.sqrt(pvar / 15)
    o += 1  # std ddof=1 over 16 positions
    out[o] = pmx - pmn
    o += 1

    # ===== family 14: ar_residual AR(2)+AR(3) (4) =====
    out[o] = 0.0
    out[o + 1] = 0.0
    out[o + 2] = 0.0
    out[o + 3] = 0.0
    _ar_fit(c, 2, out, o)
    _ar_fit(c, 3, out, o + 2)
    o += 4

    # ===== family 15: acf_first_min (3) =====
    cc2 = 0.0
    for i in range(n):
        cc2 += c[i] * c[i]
    acf = np.empty(32, np.float64)
    for k in range(1, 33):
        s = 0.0
        for i in range(n - k):
            s += c[i] * c[i + k]
        acf[k - 1] = s / (cc2 + _EPS)
    has = False
    fm = 0
    for j in range(1, 31):
        if acf[j] < acf[j - 1] and acf[j] <= acf[j + 1]:
            has = True
            fm = j - 1  # argmax of ismin (first True), index into length-30
            break
    first_min = ((fm + 2.0) if has else 32.0) / n
    fz = 0
    for j in range(32):
        if acf[j] < 0.0:
            fz = j
            break
    first_zero = fz / n
    idx_va = fm + 1
    if idx_va > 31:
        idx_va = 31
    out[o] = first_min
    o += 1
    out[o] = first_zero
    o += 1
    out[o] = acf[idx_va]
    o += 1

    # ===== family 16: histogram_mode (3) =====
    zden = std1 if std1 > 1e-6 else 1e-6
    z = np.empty(n, np.float64)
    for i in range(n):
        z[i] = (w[i] - mean) / zden
    inv = 1.0 / (5.0 / 9.0)
    e = np.empty(10, np.float64)
    maxl = -1e30
    for j in range(10):
        ctr = -2.5 + 5.0 * j / 9.0
        s = 0.0
        for i in range(n):
            u = (ctr - z[i]) * inv
            s += math.exp(-0.5 * u * u)
        e[j] = 4.0 * s
        if e[j] > maxl:
            maxl = e[j]
    den = 0.0
    for j in range(10):
        e[j] = math.exp(e[j] - maxl)
        den += e[j]
    m10 = 0.0
    for j in range(10):
        ctr = -2.5 + 5.0 * j / 9.0
        m10 += (e[j] / den) * ctr
    zs = np.sort(z)
    zmed = zs[(n - 1) // 2]  # torch lower median
    cmass = 0.0
    for i in range(n):
        if abs(z[i]) < 0.5:
            cmass += 1.0
    out[o] = m10
    o += 1
    out[o] = m10 - zmed
    o += 1
    out[o] = cmass / n
    o += 1

    # ===== family 17: ricker_wavelet (4) =====
    for ri in range(4):
        oc = _conv_same(w, Rk[ri], 1)
        p = 0.0
        for t in range(n):
            if oc[t] > 0.0:
                p += 1.0
        out[o] = p / n
        o += 1

    # ===== family 18: hydra_compete (32) — competing-kernel soft win-counts, sqrt-compressed =====
    xd = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        xd[i] = w[i + 1] - w[i]
    for chan in range(2):
        for d in (2, 4):
            for g in range(HYW.shape[0]):
                if chan == 0:
                    r0 = _conv_same(w, HYW[g, 0], d); r1 = _conv_same(w, HYW[g, 1], d)
                    r2 = _conv_same(w, HYW[g, 2], d); r3 = _conv_same(w, HYW[g, 3], d)
                else:
                    r0 = _conv_same(xd, HYW[g, 0], d); r1 = _conv_same(xd, HYW[g, 1], d)
                    r2 = _conv_same(xd, HYW[g, 2], d); r3 = _conv_same(xd, HYW[g, 3], d)
                s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0
                for t in range(r0.shape[0]):
                    v0 = r0[t]; v1 = r1[t]; v2 = r2[t]; v3 = r3[t]
                    am = 0; vm = v0
                    if v1 > vm: am = 1; vm = v1
                    if v2 > vm: am = 2; vm = v2
                    if v3 > vm: am = 3; vm = v3
                    if vm > 0.0:
                        if am == 0: s0 += vm
                        elif am == 1: s1 += vm
                        elif am == 2: s2 += vm
                        else: s3 += vm
                tot = s0 + s1 + s2 + s3 + 1e-8
                out[o] = math.sqrt(s0 / tot); o += 1
                out[o] = math.sqrt(s1 / tot); o += 1
                out[o] = math.sqrt(s2 / tot); o += 1
                out[o] = math.sqrt(s3 / tot); o += 1

    return out


@njit(cache=True)
def _ar_fit(c, p, out, off):
    """AR(p) ridge fit on centered c; writes [log resid_var, log ||beta||^2] to out."""
    n = c.shape[0]
    m = n - p  # rows
    A = np.zeros((p, p), np.float64)
    b = np.zeros(p, np.float64)
    for r in range(m):
        t = r + p  # target index c[t]; predictors c[t-1..t-p]
        for a in range(p):
            xa = c[t - 1 - a]
            b[a] += xa * c[t]
            for d in range(p):
                A[a, d] += xa * c[t - 1 - d]
    for a in range(p):
        A[a, a] += 1e-3
    beta = np.linalg.solve(A, b)
    # residual variance (ddof=1) and coeff norm
    rm = 0.0
    resid = np.empty(m, np.float64)
    for r in range(m):
        t = r + p
        pred = 0.0
        for a in range(p):
            pred += beta[a] * c[t - 1 - a]
        resid[r] = c[t] - pred
        rm += resid[r]
    rm /= m
    rv = 0.0
    for r in range(m):
        rv += (resid[r] - rm) ** 2
    rv /= m - 1
    bn = 0.0
    for a in range(p):
        bn += beta[a] * beta[a]
    out[off] = math.log(rv + _EPS)
    out[off + 1] = math.log(bn if bn > _EPS else _EPS)


@njit(parallel=True, cache=True)
def _phi_batch(W, GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW):
    B = W.shape[0]
    out = np.empty((B, 173), np.float64)
    for b in prange(B):
        out[b] = _phi_one(W[b], GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu, HYW)
    return out


# ===================== patchify + pooling =====================
_FAMILIES = [
    ("stats", 12),
    ("srf_mlp", 12),
    ("autocorr", 2),
    ("spectral", 2),
    ("turning", 2),
    ("trf_gausswin", 12),
    ("crf_ppv", 32),
    ("hilbert_env", 2),
    ("crf_max", 32),
    ("morphology_updown", 4),
    ("fftbands", 6),
    ("perm_entropy", 3),
    ("curvature", 3),
    ("conv_position", 3),
    ("ar_residual", 4),
    ("acf_first_min", 3),
    ("histogram_mode", 3),
    ("ricker_wavelet", 4),
    ("hydra_compete", 32),
]
POOL_MAP = {
    "srf_mlp": ["max"], "spectral": ["max"], "trf_gausswin": ["mean"],
    "crf_ppv": ["mean", "std"], "crf_max": ["mean"], "morphology_updown": ["std"],
    "perm_entropy": ["max"], "curvature": ["mean"], "conv_position": ["max"],
    "ar_residual": ["mean", "max"], "ricker_wavelet": ["mean"], "hydra_compete": ["mean", "std"],
}
_OPS = ("mean", "std", "max")


def _patchify(v, stride=16, resample_short=True):
    v = np.asarray(v, float)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v)
    if len(v) < L:
        if resample_short and len(v) >= 2:
            v = np.interp(np.linspace(0, len(v) - 1, L), np.arange(len(v)), v)
        else:
            v = np.pad(v, (L - len(v), 0), mode="edge")
    st = list(range(0, len(v) - L + 1, stride))
    if st[-1] != len(v) - L:
        st.append(len(v) - L)
    return np.stack([v[s : s + L] for s in st])


@lru_cache(maxsize=None)
def _consts():
    return build_consts()  # embedded banks, built once


def _phi(W):
    return _phi_batch(np.ascontiguousarray(np.asarray(W, np.float64)), *_consts())


def _pool(p, pooling):
    if pooling == "full":
        return np.concatenate([p.mean(0), p.std(0), p.max(0)])
    cols, idx = [], 0
    for nm, w in _FAMILIES:
        seg = p[:, idx : idx + w]
        idx += w
        keep = POOL_MAP.get(nm, [])
        if "mean" in keep:
            cols.append(seg.mean(0))
        if "std" in keep:
            cols.append(seg.std(0))
        if "max" in keep:
            cols.append(seg.max(0))
    return np.concatenate(cols)


def _encode(instances, pooling):
    """Encode instances (each a list of 1-D channels) -> (n, D); one batched phi."""
    all_pats, bounds = [], []
    for inst in instances:
        cnt = 0
        for ch in inst:
            ch = np.asarray(ch, float)
            z = (ch - ch.mean()) / (ch.std() + 1e-8)
            P = _patchify(z)
            all_pats.append(P)
            cnt += len(P)
        bounds.append(cnt)
    feats = _phi(np.concatenate(all_pats, 0))
    out, i = [], 0
    for cnt in bounds:
        out.append(_pool(feats[i : i + cnt], pooling))
        i += cnt
    return np.asarray(out)


class EvoForestTSWM(BaseCollectionTransformer):
    """EvoForest Time-Series World-Model encoder (TS-WM).

    A frozen, closed-form feature transform: 19 interpretable feature families
    (173 formula columns over a length-64 patch) pooled by ``mean || std || max``
    over all patches (and, for multivariate input, all channels) of a series. The
    encoder has **no learned weights** (a fixed function of seeded random-projection
    banks), so ``fit`` is empty and the same transform applies to every dataset.
    Discovered by EvoForest under a world-model objective; a peer of :class:`Catch22`
    and ``MiniRocket`` at far fewer features.

    Parameters
    ----------
    pooling : {"full", "pruned"}, default="full"
        ``"full"`` -> 519 features; ``"pruned"`` -> 211 (a discovered subset).

    References
    ----------
    .. [1] "A Foundational Neuro-Symbolic World Model for Multivariate Time Series:
           Evolve Once, Freeze, Transfer." (TS-WM / EvoForest), 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.feature_based import EvoForestTSWM
    >>> X = np.random.RandomState(0).normal(size=(8, 1, 100))
    >>> Xt = EvoForestTSWM().fit_transform(X)  # doctest: +SKIP
    """

    _tags = {
        "output_data_type": "Tabular",
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:unequal_length": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "algorithm_type": "feature",
        "python_dependencies": "numba",
    }

    def __init__(self, pooling="full"):
        self.pooling = pooling
        super().__init__()

    def _transform(self, X, y=None):
        if self.pooling not in ("full", "pruned"):
            raise ValueError(
                f"pooling must be 'full' or 'pruned', got {self.pooling!r}"
            )
        return _encode([list(np.asarray(x, dtype=float)) for x in X], self.pooling)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"pooling": "full"}, {"pooling": "pruned"}]

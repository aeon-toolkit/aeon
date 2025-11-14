import math
import numpy as np
import multiprocessing
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils import check_random_state  # CHANGED: new import


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)

    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd

    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)

    if shift == 0:
        return a

    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False

    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)

    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def _ncc_c_3dim(data):
    x, y = data[0], data[1]
    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    fft_size = 1 << (2 * x_len - 1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den


def _sbd(x, y):
    ncc = _ncc_c_3dim([x, y])
    idx = np.argmax(ncc)
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return yshift


def collect_shift(data):
    x, cur_center = data[0], data[1]
    if np.all(cur_center == 0):
        return x
    else:
        return _sbd(cur_center, x)


def _extract_shape(idx, x, j, cur_center, rng):  # CHANGED: added rng argument
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            _a.append(collect_shift([x[i], cur_center]))

    a = np.array(_a)

    if len(a) == 0:
        indices = rng.choice(x.shape[0], 1)  # CHANGED: use rng instead of np.random
        return np.squeeze(x[indices].copy())
        # return np.zeros((x.shape[1]))

    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)

    s = np.dot(y[:, :, 0].transpose(), y[:, :, 0])
    p = np.empty((columns, columns))
    p.fill(1.0 / columns)
    p = np.eye(columns) - p
    m = np.dot(np.dot(p, s), p)

    _, vec = eigh(m)
    centroid = vec[:, -1]

    finddistance1 = np.sum(
        np.linalg.norm(a - centroid.reshape((x.shape[1], 1)), axis=(1, 2)))
    finddistance2 = np.sum(
        np.linalg.norm(a + centroid.reshape((x.shape[1], 1)), axis=(1, 2)))

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)


def _kshape(x, k, centroid_init='zero', max_iter=100, n_jobs=1,
            random_state=None):  # CHANGED: added random_state param
    rng = check_random_state(random_state)  # CHANGED: create RNG from random_state

    m = x.shape[0]
    idx = rng.randint(0, k, size=m)  # CHANGED: use rng instead of global randint

    if isinstance(centroid_init, np.ndarray):
        centroids = centroid_init
    elif centroid_init == 'zero':
        centroids = np.zeros((k, x.shape[1], x.shape[2]))
    elif centroid_init == 'random':
        indices = rng.choice(x.shape[0], k)  # CHANGED: use rng instead of np.random
        centroids = x[indices].copy()
    distances = np.empty((m, k))

    for it in range(max_iter):
        old_idx = idx

        for j in range(k):
            for d in range(x.shape[2]):
                centroids[j, :, d] = _extract_shape(
                    idx,
                    np.expand_dims(x[:, :, d], axis=2),
                    j,
                    np.expand_dims(centroids[j, :, d], axis=1),
                    rng,  # CHANGED: pass rng through
                )
                # centroids[j] = np.expand_dims(_extract_shape(idx, x, j, centroids[j]), axis=1)

        pool = multiprocessing.Pool(n_jobs)
        args = []
        for p in range(m):
            for q in range(k):
                args.append([x[p, :], centroids[q, :]])
        result = pool.map(_ncc_c_3dim, args)
        pool.close()
        r = 0
        for p in range(m):
            for q in range(k):
                distances[p, q] = 1 - result[r].max()
                r = r + 1

        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids


def kshape(x, k, centroid_init='zero', max_iter=100,
           random_state=None):  # CHANGED: added random_state param
    idx, centroids = _kshape(
        np.array(x),
        k,
        centroid_init=centroid_init,
        max_iter=max_iter,
        n_jobs=1,
        random_state=random_state,  # CHANGED: pass random_state through
    )
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))

    return clusters


class KShapeClusteringCPU(ClusterMixin, BaseEstimator):
    labels_ = None
    centroids_ = None

    def __init__(self, n_clusters, centroid_init='zero', max_iter=100,
                 n_jobs=None, random_state=None):  # CHANGED: added random_state param
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        self.random_state = random_state  # CHANGED: store random_state
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit(self, X, y=None):
        clusters = self._fit(
            X,
            self.n_clusters,
            self.centroid_init,
            self.max_iter,
            self.n_jobs,
        )
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ = np.zeros((self.n_clusters, X.shape[1], X.shape[2]))
        for i in range(self.n_clusters):
            self.labels_[clusters[i][1]] = i
            self.centroids_[i] = clusters[i][0]
        return self

    def predict(self, X):
        labels, _ = self._predict(X, self.centroids_)
        return labels

    def _predict(self, x, centroids):
        m = x.shape[0]
        rng = check_random_state(self.random_state)  # CHANGED: create RNG here
        idx = rng.randint(0, self.n_clusters, size=m)  # CHANGED: use rng instead of randint
        distances = np.empty((m, self.n_clusters))

        pool = multiprocessing.Pool(self.n_jobs)
        args = []
        for p in range(m):
            for q in range(self.n_clusters):
                args.append([x[p, :], centroids[q, :]])
        result = pool.map(_ncc_c_3dim, args)
        pool.close()
        r = 0
        for p in range(m):
            for q in range(self.n_clusters):
                distances[p, q] = 1 - result[r].max()
                r = r + 1

        idx = distances.argmin(1)

        return idx, centroids

    def _fit(self, x, k, centroid_init='zero', max_iter=100, n_jobs=1):
        idx, centroids = _kshape(
            np.array(x),
            k,
            centroid_init=centroid_init,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=self.random_state,  # CHANGED: pass random_state through
        )
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))

        return clusters


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod()[0])

"""E-Agglo: agglomerative clustering algorithm that preserves observation order."""

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

from aeon.segmentation.base import BaseSegmenter

__maintainer__ = []
__all__ = ["EAggloSegmenter"]


class EAggloSegmenter(BaseSegmenter):
    """
    Hierarchical agglomerative estimation of multiple change points.

    E-Agglo is a non-parametric clustering approach for multivariate timeseries[1]_,
    where neighboring segments are sequentially merged_ to maximize a goodness-of-fit
    statistic. Unlike most general purpose agglomerative clustering algorithms, this
    procedure preserves the time ordering of the observations.

    This method can detect distributional change within an independent sequence,
    and does not make any distributional assumptions (beyond the existence of an
    alpha-th moment). Estimation is performed in a manner that simultaneously
    identifies both the number and locations of change points.

    Parameters
    ----------
    member : array_like (default=None)
        Assigns points to the initial cluster membership, therefore the first
        dimension should be the same as for data. If `None` it will be initialized
        to dummy vector where each point is assigned to separate cluster.
    alpha : float (default=1.0)
        Fixed constant alpha in (0, 2] used in the divergence measure, as the
        alpha-th absolute moment, see equation (4) in [1]_.
    penalty : str or callable or None (default=None)
        Function that defines a penalization of the sequence of goodness-of-fit
        statistic, when overfitting is a concern. If `None` not penalty is applied.
        Could also be an existing penalty name, either `len_penalty` or
        `mean_diff_penalty`.

    Attributes
    ----------
    merged_ : array_like
        2D `array_like` outlining which clusters were merged_ at each step.
    gof_ : float
        goodness-of-fit statistic for current clsutering.
    cluster_ : array_like
        1D `array_like` specifying which cluster each row of input data
        X belongs to.

    Notes
    -----
    Based on the work from [1]_.

    - source code based on: https://github.com/cran/ecp/blob/master/R/e_agglomerative.R
    - paper available at: https://www.tandfonline.com/doi/full/10.1080/01621459.\
        2013.849605

    References
    ----------
    .. [1] Matteson, David S., and Nicholas A. James. "A nonparametric approach for
    multiple change point analysis of multivariate data." Journal of the American
    Statistical Association 109.505 (2014): 334-345.

    .. [2] James, Nicholas A., and David S. Matteson. "ecp: An R package for
    nonparametric multiple change point analysis of multivariate data." arXiv preprint
    arXiv:1309.3295 (2013).

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_dataframe_series
    >>> from aeon.segmentation import EAggloSegmenter
    >>> X = make_example_dataframe_series(n_channels=2, random_state=10)
    >>> model = EAggloSegmenter()
    >>> y = model.fit_predict(X, axis=0)
    """

    _tags = {
        "X_inner_type": "pd.DataFrame",
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
    }

    def __init__(
        self,
        member=None,
        alpha=1.0,
        penalty=None,
    ):
        self.member = member
        self.alpha = alpha
        self.penalty = penalty
        super().__init__(axis=0, n_segments=None)

    def _fit(self, X, y=None):
        """Find optimally clustered segments.

        First, by determining which pairs of adjacent clusters will be merged_. Then,
        this process is repeated, recording the goodness-of-fit statistic at each step,
        until all observations belong to a single cluster. Finally, the estimated number
        of change points is estimated by the clustering that maximizes the goodness-of-
        fit statistic over the entire merging sequence.
        """
        self._X = X

        if self.alpha <= 0 or self.alpha > 2:
            raise ValueError(
                f"allowed values for 'alpha' are (0, 2], " f"got: {self.alpha}"
            )

        self._initialize_params(X)

        # find which clusters optimize the gof_ and then update the distances
        for K in range(self.n_cluster - 1, 2 * self.n_cluster - 2):
            i, j = self._find_closest(K)

            _update_distances(
                i,
                j,
                K,
                self.merged_,
                self.n_cluster,
                self.distances,
                self.left,
                self.right,
                self.open,
                self.sizes,
                self.progression,
                self.lm,
            )

        def filter_na(i):
            return list(
                filter(
                    lambda v: v == v,
                    self.progression[i,],
                )
            )

        # penalize the gof_ statistic
        if self.penalty is not None:
            penalty_func = self._get_penalty_func()
            cps = [filter_na(i) for i in range(len(self.progression))]
            self.gof_ += list(map(penalty_func, cps))

        # get the set of change points for the "best" clustering
        idx = np.argmax(self.gof_)
        self._estimates = np.sort(filter_na(idx))

        # remove change point N+1 if a cyclic merger was performed
        self._estimates = (
            self._estimates[:-1] if self._estimates[0] != 0 else self._estimates
        )

        # create final membership vector
        def get_cluster(estimates):
            return np.repeat(
                range(len(np.diff(estimates))), np.diff(estimates).astype(int)
            )

        if self._estimates[0] == 0:
            self.cluster_ = get_cluster(self._estimates)
        else:
            tmp = get_cluster(np.append([0], self._estimates))
            self.cluster_ = np.append(tmp, np.zeros(X.shape[0] - len(tmp)))

        return self

    def _predict(self, X: pd.DataFrame, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.Dataframe
            Data to be transformed
        y : default=None
            Ignored, for interface compatibility

        Returns
        -------
        cluster
            numeric representation of cluster membership for each row of X.
        """
        # fit again if indices not seen, but don't store anything
        if not X.index.equals(self._X.index):
            X_full = X.combine_first(self._X)
            new_eagglo = EAggloSegmenter(
                member=self.member,
                alpha=self.alpha,
                penalty=self.penalty,
            ).fit(X_full)
            warnings.warn(
                "Warning: Input data X differs from that given to fit(). "
                "Refitting with both the data in fit and new input data, not storing "
                "updated public class attributes. For this, explicitly use fit(X) or "
                "fit_predict(X).",
                stacklevel=1,
            )
            return new_eagglo.cluster_

        return self.cluster_

    def _initialize_params(self, X: pd.DataFrame) -> None:
        """Initialize parameters and store to self."""
        self._member = np.array(
            self.member if self.member is not None else range(X.shape[0])
        )

        unique_labels = np.sort(np.unique(self._member))
        self.n_cluster = len(unique_labels)

        # relabel clusters to be consecutive numbers (when user specified)
        for i in range(self.n_cluster):
            self._member[np.where(self._member == unique_labels[i])[0]] = i

        # check if sorted
        if not all(sorted(self._member) == self._member):
            raise ValueError("'_member' should be sorted")

        self.sizes = np.zeros(2 * self.n_cluster)
        self.sizes[: self.n_cluster] = [
            sum(self._member == i) for i in range(self.n_cluster)
        ]  # calculate initial cluster sizes

        # array of between-within distances
        self.distances = np.empty((2 * self.n_cluster, 2 * self.n_cluster))

        # if there is an initial grouping...
        if self.member is not None:
            grouped = X.copy().set_index(self._member).groupby(level=0)
            within = grouped.apply(
                lambda x: get_distance_matrix(x.to_numpy(), x.to_numpy(), self.alpha)
            )

            for i, xi in grouped:
                self.distances[: self.n_cluster, i] = (
                    2
                    * grouped.apply(
                        lambda xj: get_distance_matrix(
                            xi.to_numpy(), xj.to_numpy(), self.alpha  # noqa
                        )
                    )
                    - within[i]
                    - within
                )
        # else (no initial groupings)...
        else:
            X_num = X.to_numpy()
            for i in range(len(X)):
                for j in range(len(X)):
                    self.distances[i, j] = 2 * get_distance_single(
                        X_num[i], X_num[j], self.alpha
                    )

        np.fill_diagonal(self.distances, 0)

        # set up left and right neighbors
        # special case for clusters 0 and n_cluster-1 to allow for cyclic merging
        self.left = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.left[: self.n_cluster] = [
            i - 1 if i >= 1 else self.n_cluster - 1 for i in range(self.n_cluster)
        ]

        self.right = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.right[: self.n_cluster] = [
            i + 1 if i + 1 < self.n_cluster else 0 for i in range(self.n_cluster)
        ]

        # True means that a cluster has not been merged_
        self.open = np.ones(2 * self.n_cluster - 1, dtype=bool)

        # which clusters were merged_ at each step
        self.merged_ = np.empty((self.n_cluster - 1, 2))

        # set initial gof_ value
        self.gof_ = np.array(
            [
                sum(
                    self.distances[i, self.left[i]] + self.distances[i, self.right[i]]
                    for i in range(self.n_cluster)
                )
            ]
        )

        # change point progression
        self.progression = np.empty((self.n_cluster, self.n_cluster + 1))
        self.progression[0, :] = [
            sum(self.sizes[:i]) if i > 0 else 0 for i in range(self.n_cluster + 1)
        ]  # N + 1 for cyclic mergers

        # array to specify the starting point of a cluster
        self.lm = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.lm[: self.n_cluster] = range(self.n_cluster)

    def _find_closest(self, K: int) -> tuple[int, int]:
        """Determine which clusters will be merged_, for K clusters.

        Greedily optimize the goodness-of-fit statistic by merging the pair of adjacent
        clusters that results in the largest increase of the statistic's value.

        Parameters
        ----------
        K: int
            Number of clusters

        Returns
        -------
        result : Tuple[int, int]
            Tuple of left cluster and right cluster index values
        """
        best_fit = -1e10
        result = (0, 0)

        # iterate through each cluster to see how the gof_ value changes if merged_
        for i in range(K + 1):
            if self.open[i]:
                gof_ = _gof_update(
                    i, self.gof_, self.left, self.right, self.distances, self.sizes
                )

                if gof_ > best_fit:
                    best_fit = gof_
                    result = (i, self.right[i])

        self.gof_ = np.append(self.gof_, best_fit)
        return result

    def _get_penalty_func(self) -> Callable:  # sourcery skip: raise-specific-error
        """Define penalty function given (possibly string) input."""
        PENALTIES = {"len_penalty": len_penalty, "mean_diff_penalty": mean_diff_penalty}

        if callable(self.penalty):
            return self.penalty

        elif isinstance(self.penalty, str):
            if self.penalty in PENALTIES:
                return PENALTIES[self.penalty]

        raise Exception(
            f"'penalty' must be callable or {PENALTIES.keys()}, got {self.penalty}"
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> list[dict]:
        """Test parameters."""
        return [
            {"alpha": 1.0, "penalty": None},
            {"alpha": 2.0, "penalty": "len_penalty"},
        ]


@njit(fastmath=True, cache=True)
def _update_distances(
    i, j, K, merged_, n_cluster, distances, left, right, open, sizes, progression, lm
) -> None:
    """Update distance from new cluster to other clusters, store to self."""
    # which clusters were merged_, info only
    merged_[K - n_cluster + 1, 0] = -i if i <= n_cluster else i - n_cluster
    merged_[K - n_cluster + 1, 1] = -j if j <= n_cluster else j - n_cluster

    # update left and right neighbors
    ll = left[i]
    rr = right[j]
    left[K + 1] = ll
    right[K + 1] = rr
    right[ll] = K + 1
    left[rr] = K + 1

    # update information about which clusters have been merged_
    open[i] = False
    open[j] = False

    # assign size to newly created cluster
    n1 = sizes[i]
    n2 = sizes[j]
    sizes[K + 1] = n1 + n2

    # update set of change points
    progression[K - n_cluster + 2, :] = progression[K - n_cluster + 1,]
    progression[K - n_cluster + 2, lm[j]] = np.nan
    lm[K + 1] = lm[i]

    # update distances
    for k in range(K + 1):
        if open[k]:
            n3 = sizes[k]
            n = n1 + n2 + n3
            val = (
                (n - n2) * distances[i, k]
                + (n - n1) * distances[j, k]
                - n3 * distances[i, j]
            ) / n
            distances[K + 1, k] = val
            distances[k, K + 1] = val


@njit(fastmath=True, cache=True)
def _gof_update(i, gof_, left, right, distances, sizes):
    """Compute the updated goodness-of-fit statistic, left cluster given by i."""
    fit = gof_[-1]
    j = right[i]

    # get new left and right clusters
    rr = right[j]
    ll = left[i]

    # remove unneeded values in the gof_
    fit -= 2 * (distances[i, j] + distances[i, ll] + distances[j, rr])

    # get cluster sizes
    n1 = sizes[i]
    n2 = sizes[j]

    # add distance to new left cluster
    n3 = sizes[ll]
    k = (
        (n1 + n3) * distances[i, ll]
        + (n2 + n3) * distances[j, ll]
        - n3 * distances[i, j]
    ) / (n1 + n2 + n3)
    fit += 2 * k

    # add distance to new right
    n3 = sizes[rr]
    k = (
        (n1 + n3) * distances[i, rr]
        + (n2 + n3) * distances[j, rr]
        - n3 * distances[i, j]
    ) / (n1 + n2 + n3)
    fit += 2 * k

    return fit


@njit(fastmath=True, cache=True)
def len_penalty(x: pd.DataFrame) -> int:
    """Penalize goodness-of-fit statistic for number of change points."""
    return -len(x)


@njit(fastmath=True, cache=True)
def mean_diff_penalty(x: pd.DataFrame) -> float:
    """Penalize goodness-of-fit statistic.

    Favors segmentations with larger sizes, while taking into consideration
    the size of the new segments.
    """
    return np.mean(np.diff(np.sort(x)))


@njit(fastmath=True, cache=True)
def get_distance_matrix(X, Y, alpha) -> float:
    """Calculate cluster distance."""
    dist = euclidean_matrix_to_matrix(X, Y)
    return np.power(dist, alpha).mean()


@njit(fastmath=True, cache=True)
def get_distance_single(X, Y, alpha) -> float:
    dist = euclidean(X, Y)
    return np.power(dist, alpha)  # .mean()


@njit(fastmath=True, cache=True)
def euclidean_matrix_to_matrix(a, b):
    """Compute the Euclidean distances between the rows of two matrices."""
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = euclidean(a[i], b[j])
    return out


@njit(fastmath=True, cache=True)
def euclidean(u, v):
    buf = u - v
    return np.sqrt(buf @ buf)

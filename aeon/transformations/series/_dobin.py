"""Preprocessing algorithm DOBIN (Distance based Outlier BasIs using Neighbors)."""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.linalg import null_space
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from aeon.transformations.series.base import BaseSeriesTransformer

__maintainer__ = []
__all__ = ["Dobin"]


class Dobin(BaseSeriesTransformer):
    """Distance based Outlier BasIs using Neighbors (DOBIN).

    DOBIN is a pre-processing algorithm that constructs a set of basis
    vectors tailored for outlier detection as described by _[1]. DOBIN
    has a simple mathematical foundation and can be used as a dimension
    reduction tool for outlier detection tasks.

    Method assumes normalized data, the original R code implementation uses:
    ``from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler``
    This prevents variables with large variances having disproportional
    influence on Euclidean distances. The original implelemtation _[1] uses
    ``MinMaxScaler`` normalization, and removes NA values before normalization.

    We emphasize that DOBIN is not an outlier detection method; rather it is
    a pre-processing step that can be used by any outlier detection method.

    Parameters
    ----------
    frac : float (default=0.95)
        The cut-off quantile for Y space
        (parameter q in _[1]).
    k : int (default=None)
        Number of nearest neighbours considered
        (parameter k_2 on page 9 in _[1])

    Attributes
    ----------
    _basis : pd.DataFrame
        The basis vectors suitable for outlier detection
        (denoted as Theta in _[1]).
    _coords : pd.DataFrame
        The transformed coordinates of the data
        (denoted as tilde{X}, see equation 8 in _[1])

    References
    ----------
    .. [1] Kandanaarachchi, Sevvandi, and Rob J. Hyndman. "Dimension reduction
    for outlier detection using DOBIN." Journal of Computational and Graphical
    Statistics 30.1 (2021): 204-219.

    Examples
    --------
    >>> from aeon.transformations.series._dobin import Dobin
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> import numpy as np
    >>> import pandas as pd
    >>> from aeon.datasets import load_uschange
    >>> X = load_uschange()
    >>> minmax = MinMaxScaler()
    >>> Xt = minmax.fit_transform(X.T)
    >>> model = Dobin()
    >>> X_outlier = model.fit_transform(X)
    >>> X_outlier.T.head()
            DB0       DB1       DB2       DB3       DB4
    0  4.786838 -1.332530 -1.891908  1.566322  0.753280
    1  7.290015  0.149297 -1.242303  0.558777  0.474924
    2  7.297553  0.419074 -1.688429  0.282187  0.573991
    3  0.954141 -1.639316 -0.423461  1.552961  0.434186
    4  3.702288  2.066720 -1.807646 -1.777854  0.422556
    """

    _tags = {
        "X_inner_type": "pd.DataFrame",
        "capability:multivariate": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        frac=0.95,
        k=None,
    ):
        self.frac = frac
        self.k = k
        super().__init__(axis=0)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series of type X_inner_type
            Data to be transformed
        y : Series, default=None
            Not required for this unsupervised transform.

        Returns
        -------
        self: reference to self
        """
        self._X = X

        assert all(X.apply(is_numeric_dtype, axis=0))

        n_obs, n_dim = X.shape

        if n_dim == 1:
            warnings.warn(
                "Warning: Input data X is univariate. For dimensionality reduction, "
                "please provide multivariate input.",
                stacklevel=2,
            )
            self._coords = X
            return self

        # if more dimensions than observations, change of basis to subspace
        if n_obs < n_dim:
            pca = PCA(n_components=n_obs)
            X = pca.fit_transform(X)
            self._X_pca = X
            _, n_dim = X.shape

        self.k_ = min(20, max(n_obs // 20, 2)) if self.k is None else self.k

        X_copy = X.copy()
        B = np.identity(n_dim)
        basis = pd.DataFrame()

        for _ in range(n_dim):
            # Compute Y space
            y_space = close_distance_matrix(X_copy, self.k_, self.frac)

            # Find eta
            w = y_space.apply(sum, axis=0)
            eta = np.array([w / np.sqrt(sum(w**2))])

            # If issues finding Y space (e.g. no variance in column)
            # get null space of basis
            if np.isnan(eta).any():
                basis_col = pd.DataFrame(null_space(basis.T))
                basis = pd.concat([basis, basis_col], axis=1)
                break

            # Update basis
            basis_col = pd.DataFrame(np.dot(B, eta.T))
            basis = pd.concat([basis, basis_col], axis=1)

            # Find xperp
            xperp = X_copy - np.dot(np.dot(np.array(X_copy), eta.T), eta)

            # Find a basis B for xperp
            B1 = null_space(eta)

            # Change xperp coordinates to B basis
            X_copy = np.dot(xperp, B1)

            # Update B with B1, each time 1 dimension is reduced
            B = np.dot(B, B1)

        # new coordinates
        coords = pd.DataFrame(
            X.dot(np.array(basis))
        )  # convert np.array, error if both pd.DataFrames and rownames != colnames

        basis.columns = ["".join(["DB", str(i)]) for i in range(len(basis.columns))]
        self._basis = basis
        coords.columns = ["".join(["DB", str(i)]) for i in range(len(coords.columns))]
        self._coords = coords

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series of type X_inner_type
            Data to be transformed
        y : Series, default=None
            Not required for this unsupervised transform.

        Returns
        -------
        transformed version of X, representing the original data on a new set of
        coordinates, obtained by multiplying input data by the basis vectors.
        """
        # fit again if indices not seen, but don't store anything
        if not X.index.equals(self._X.index):
            X_full = X.combine_first(self._X)
            new_dobin = Dobin(
                frac=self.frac,
                k=self.k,
            ).fit(X_full)
            warnings.warn(
                "Warning: Input data X differs from that given to fit(). "
                "Refitting with new input data, not storing updated public class "
                "attributes. For this, explicitly use fit(X) or fit_transform(X).",
                stacklevel=2,
            )
            return new_dobin._coords

        return self._coords


def close_distance_matrix(X: npt.ArrayLike, k: int, frac: float):
    """Calculate distance between close pairs.

    Parameters
    ----------
    X : np.ArrayLike
        Data to be transformed
    k : int
        Number of nearest neighbours considered. If k = None, it is empirically
        derived as ``min(0.05 * number of observations, 20)``

    Returns
    -------
    pd.DataFrame of pairs of close neighbour indices
    """
    X = pd.DataFrame(X)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)

    dist = pd.DataFrame(
        [
            ((X.iloc[i,] - X.iloc[j,]) ** 2).tolist()
            for (i, j) in zip(
                np.repeat(indices[:, 0], repeats=k), indices[:, 1:].flatten()
            )
        ]
    )

    row_sums = dist.apply(sum, axis=1)
    q_frac = np.quantile(row_sums, q=frac)

    mask = row_sums > q_frac

    return dist.loc[mask,]

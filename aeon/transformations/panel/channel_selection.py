# -*- coding: utf-8 -*-
"""Channel Selection techniques for Multivariate Time Series Classification.

A transformer that selects a subset of channels/dimensions for time series
classification using a scoring system with an elbow point method.
"""

__author__ = ["haskarb"]
__all__ = ["ElbowClassSum", "ElbowClassPairwise"]


import itertools
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import LabelEncoder

from aeon.distances import distance as aeon_distance
from aeon.transformations.base import BaseTransformer


def _detect_knee_point(values: List[float], indices: List[int]) -> List[int]:
    """Find elbow point."""
    n_points = len(values)
    all_coords = np.vstack((range(n_points), values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    knee_idx = np.argmax(dist_to_line)
    knee = values[knee_idx]
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem > knee]
    if len(best_dims) == 0:
        # return all dimensions if no elbow point is found
        return indices
    return best_dims


def create_distance_matrix(
    prototype: Union[pd.DataFrame, np.ndarray],
    class_vals: np.array,
    distance: str = "euclidean",
) -> pd.DataFrame:
    """Create a distance matrix between class prototypes.

    Parameters
    ----------
    prototype : pd.DataFrame or np.ndarray
        A multivatiate time series representation for entire dataset.
    class_vals : np.array
        Class values.
    distance : str, default="euclidean"
        Distance metric to be used for calculating distance between class prototypes.

    Returns
    -------
    distance_frame : pd.DataFrame
        Distance matrix between class prototypes.

    """
    if prototype.shape[0] != len(class_vals):
        raise ValueError(
            f"Prototype {prototype.shape[0]} and \
            class values {len(class_vals)} must be of same length."
        )

    distance_pair = list(itertools.combinations(range(0, class_vals.shape[0]), 2))
    # create a dictionary of class values and their indexes
    idx_class = {i: class_vals[i] for i in range(0, len(class_vals))}

    distance_frame = pd.DataFrame()
    for cls_ in distance_pair:
        # calculate the distance of centroid here
        for _, (cls1_ch, cls2_ch) in enumerate(
            zip(
                prototype[class_vals == idx_class[cls_[0]]],
                prototype[class_vals == idx_class[cls_[1]]],
            )
        ):
            if distance == "euclidean":
                dis = np.linalg.norm(cls1_ch - cls2_ch, axis=1)
            else:
                dis = np.apply_along_axis(
                    lambda row: aeon_distance(
                        row[: row.shape[0] // 2],
                        row[row.shape[0] // 2 :],
                        metric="dtw",
                    ),
                    axis=1,
                    arr=np.concatenate((cls1_ch, cls2_ch), axis=1),
                )
            dict_ = {f"Centroid_{idx_class[cls_[0]]}_{idx_class[cls_[1]]}": dis}
        distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)
    return distance_frame


class ClassPrototype:
    """
    Representation for each class from the dataset.

    Parameters
    ----------
    prototype_type : str, default="mean"
        Class prototype to be used for class prototype creation.
        Available options are "mean", "median", "mad".
    mean_centering : bool, default=False
        If True, mean centering is applied to the class prototype.

    Attributes
    ----------
    prototype : str
        Class prototype to be used for class prototype creation.

    Notes
    -----
    For more information on the prototype types and class prototype, see [1] and [2].

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021
    ..[2]: Bhaskar Dhariyal et al. “Scalable Classifier-Agnostic Channel Selection
    for Multivariate Time Series Classification", DAMI, ECML, Springer, 2023

    """

    def __init__(
        self,
        prototype_type: str = "mean",
        mean_centering: bool = False,
    ):
        self.prototype_type = prototype_type
        self.mean_centering = mean_centering

        if self.prototype_type not in ["mean", "median", "mad"]:
            raise ValueError(
                f"Prototype type {self.prototype_type} not supported. "
                "Available options are 'mean', 'median', 'mad'."
            )

    def _mad_median(self, class_X, median=None):
        """Calculate upper and lower bounds for median absolute deviation."""
        _mad = median_abs_deviation(class_X, axis=0)

        low_value = median - _mad * 0.50
        high_value = median + _mad * 0.50
        # clip = lambda x: np.clip(x, low_value, high_value)
        class_X = np.apply_along_axis(
            lambda x: np.clip(x, a_min=low_value, a_max=high_value),
            axis=1,
            arr=class_X,
        )

        return np.mean(class_X, axis=0)

    def create_mad_prototype(self, X: np.ndarray, y: np.array) -> np.array:
        """Create mad class prototype for each class."""
        classes_ = np.unique(y)

        channel_median = []
        for class_ in classes_:
            class_idx = np.where(
                y == class_
            )  # find the indexes of data point where particular class is located

            class_median = np.median(X[class_idx], axis=0)
            class_median = self._mad_median(X[class_idx], class_median)
            channel_median.append(class_median)

        return np.vstack(channel_median)

    def create_mean_prototype(self, X: np.ndarray, y: np.array):
        """Create mean class prototype for each class."""
        classes_ = np.unique(y)
        channel_mean = [np.mean(X[y == class_], axis=0) for class_ in classes_]
        return np.vstack(channel_mean)

    def create_median_prototype(self, X: np.ndarray, y: np.array):
        """Create mean class prototype for each class."""
        classes_ = np.unique(y)
        channel_median = [np.median(X[y == class_], axis=0) for class_ in classes_]
        return np.vstack(channel_median)

    def create_median_prototype1(self, X: pd.DataFrame, y: pd.Series):
        """Create median class prototype for each class."""
        classes_ = np.unique(y)

        channel_median = []
        for class_ in classes_:
            class_idx = np.where(
                y == class_
            )  # find the indexes of data point where particular class is located
            class_median = np.median(X[class_idx], axis=0)
            channel_median.append(class_median)
        return np.vstack(channel_median)

    def create_prototype(
        self, X: np.ndarray, y: np.array
    ) -> Union[Tuple[pd.DataFrame, np.array], Tuple[np.ndarray, np.array]]:
        """Create the class prototype for each class."""
        le = LabelEncoder()
        y_ind = le.fit_transform(y)

        prototype_funcs = {
            "mean": self.create_mean_prototype,
            "median": self.create_median_prototype,
            "mad": self.create_mad_prototype,
        }
        prototypes = []
        for channel in range(X.shape[1]):  # iterating over channels
            train = X[:, channel, :]
            _prototype = prototype_funcs[self.prototype_type](train, y_ind)
            prototypes.append(_prototype)

        prototypes = np.stack(prototypes, axis=1)

        if self.mean_centering:
            prototypes -= np.mean(prototypes, axis=2, keepdims=True)

        return (prototypes, le.classes_)


class ElbowClassSum(BaseTransformer):
    """Elbow Class Sum (ECS) transformer to select a subset of channels/variables.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1, 2] by calculating the distance between each class prototype. The
    ECS selects the subset of channels using the elbow method, which maximizes the
    distance between the class centroids by aggregating the distance for every
    class pair across each channel.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature. E.g., channel selection = variable selection.

    Parameters
    ----------
    distance : str
        Distance metric to use for creating the class prototype.
        Default: 'euclidean'
    prototype_type : str
        Type of class prototype to use for representing a class.
        Default: 'mean'
    mean_center : bool
        If True, mean centering is applied to the class prototype.
        Default: False

    Attributes
    ----------
    prototype : DataFrame
        A multivariate time series representation for entire dataset.
    distance_frame : DataFrame
        Distance matrix for each class pair.
        ``shape = [n_channels, n_class_pairs]``
    channels_selected_idx : list
        List of selected channels.
    rank: list
        Rank of channels based on the distance between class prototypes.

    Notes
    -----
    More details on class prototype can be found in [1] and [2].
    Original repository:
    1. https://github.com/mlgig/Channel-Selection-MTSC
    2. https://github.com/mlgig/ChannelSelectionMTSC

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021
    ..[2]: Bhaskar Dhariyal et al. “Scalable Classifier-Agnostic Channel Selection
    for Multivariate Time Series Classification", DAMI, ECML, Springer, 2023

    Examples
    --------
    >>> from aeon.transformations.panel.channel_selection import ElbowClassSum
    >>> from aeon.utils._testing.panel import make_classification_problem
    >>> X, y = make_classification_problem(n_columns=3, n_classes=3, random_state=42)
    >>> cs = ElbowClassSum()
    >>> cs.fit(X, y)
    ElbowClassSum(...)
    >>> Xt = cs.transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        # "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # which mtypes do _fit/_predict support for y?
        "requires_y": True,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
    }

    def __init__(
        self,
        distance: str = "euclidean",
        prototype_type: str = "mean",
        mean_center: bool = False,
    ):
        self.distance = distance
        self.mean_center = mean_center
        self.prototype_type = prototype_type
        self._is_fitted = False

        super(ElbowClassSum, self).__init__()

    def _fit(self, X, y):
        """Fit ECS to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : reference to self.
        """
        cp = ClassPrototype(
            prototype_type=self.prototype_type,
            mean_centering=self.mean_center,
        )
        self.prototype, labels = cp.create_prototype(X.copy(), y)

        self.distance_frame = create_distance_matrix(
            self.prototype.copy(), labels, distance=self.distance
        )
        self.channels_selected_idx = []
        distance = self.distance_frame.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame.sum(axis=1).sort_values(ascending=False).index

        self.channels_selected_idx.extend(_detect_knee_point(distance, indices))
        self.rank = self.channels_selected_idx
        self._is_fitted = True

        return self

    def _transform(self, X, y=None):
        """
        Transform X and return a transformed version.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            X with a subset of channels
        """
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before transform()")
        return X[:, self.channels_selected_idx]


class ElbowClassPairwise(BaseTransformer):
    """Elbow Class Pairwise (ECP) transformer to select a subset of channels.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The ECP
    selects the subset of channels using the elbow method that maximizes the
    distance between each class centroids pair across all channels.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature.

    Parameters
    ----------
    distance : str
        Distance metric to use for creating the class prototype.
        Default: 'euclidean'
    prototype_type : str
        Type of class prototype to use for representing a class.
        Default: 'mean', Options: ['mean', 'median', 'mad']
    mean_center : bool
        If True, mean centering is applied to the class prototype.
        Default: False, Options: [True, False]


    Attributes
    ----------
    distance_frame : DataFrame
        Distance matrix between class prototypes.
    channels_selected_idx : list
        List of selected channels.
    rank: list
        Rank of channels based on the distance between class prototypes.
    prototype : DataFrame
        A multivariate time series representation for entire dataset.

    Notes
    -----
    More details on class prototype can be found in [1] and [2].

    Original repository:
    1. https://github.com/mlgig/Channel-Selection-MTSC
    2. https://github.com/mlgig/ChannelSelectionMTSC

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021
    ..[2]: Bhaskar Dhariyal et al. “Scalable Classifier-Agnostic Channel Selection
    for Multivariate Time Series Classification", DAMI, ECML, Springer, 2023

    Examples
    --------
    >>> from aeon.transformations.panel.channel_selection import ElbowClassPairwise
    >>> from aeon.utils._testing.panel import make_classification_problem
    >>> X, y = make_classification_problem(n_columns=3, n_classes=3, random_state=42)
    >>> cs = ElbowClassPairwise()
    >>> cs.fit(X, y)
    ElbowClassPairwise(...)
    >>> Xt = cs.transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        # "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # which mtypes do _fit/_predict support for y?
        "requires_y": True,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
    }

    def __init__(
        self,
        distance: str = "euclidean",
        prototype_type: str = "mad",
        mean_center: bool = False,
    ):
        self.distance = distance
        self.prototype_type = prototype_type
        self.mean_center = mean_center
        self._is_fitted = False

        super(ElbowClassPairwise, self).__init__()

    def _fit(self, X, y):
        """
        Fit ECP to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : reference to self.
        """
        cp = ClassPrototype(
            prototype_type=self.prototype_type, mean_centering=self.mean_center
        )
        self.prototype, labels = cp.create_prototype(
            X.copy(), y
        )  # Centroid created here
        self.distance_frame = create_distance_matrix(
            self.prototype.copy(), labels, self.distance
        )  # Distance matrix created here

        self.channels_selected_idx = []
        for pairdistance in self.distance_frame.items():
            distances = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            chs_dis = _detect_knee_point(distances, indices)
            self.channels_selected_idx.extend(chs_dis)

        self.rank = self._rank()
        self.channels_selected_idx = list(set(self.channels_selected_idx))
        self._is_fitted = True
        return self

    def _transform(self, X, y=None):
        """
        Transform X and return a transformed version.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            X with a subset of channels
        """
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before transform()")
        return X[:, self.channels_selected_idx]

    def _rank(self) -> List[int]:
        """Return the rank of channels for ECP."""
        all_index = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        series = self.distance_frame.sum(axis=1)
        series.drop(
            index=list(set(all_index) - set(self.channels_selected_idx)), inplace=True
        )
        return series.sort_values(ascending=False).index.tolist()

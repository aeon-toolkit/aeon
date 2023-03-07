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

from sktime.distances import distance
from sktime.transformations.base import BaseTransformer


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
    distance_: str = "euclidean",
) -> pd.DataFrame:
    """Create a distance matrix between class_prototypes."""
    assert prototype.shape[0] == len(
        class_vals
    ), "Prototype and class values must be of same length."

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
            if distance_ == "euclidean":
                dis = np.linalg.norm(cls1_ch - cls2_ch, axis=1)
            else:
                dis = np.apply_along_axis(
                    lambda row: distance(
                        cls1_ch[: cls1_ch.shape[1]],
                        cls2_ch[: cls2_ch.shape[1]],
                        metric="dtw",
                    ),
                    axis=1,
                    arr=np.concatenate((cls1_ch, cls2_ch), axis=1),
                )
            dict_ = {f"Centroid_{idx_class[cls_[0]]}_{idx_class[cls_[1]]}": dis}
        distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)
    return distance_frame


def _clip(x, low_value, high_value):
    return np.clip(x, low_value, high_value)


class ClassPrototype:
    """
    Class prototype for each class.

    Parameters
    ----------
    prototype : str, default="mean"
        Class prototype to be used for class prototype creation.
        Available options are "mean", "median", "mad".
    mean_centering : bool, default=False
        If True, mean centering is applied to the class prototype.
    return_df : bool, default=False
        If True, returns a dataframe of class prototypes.

    Attributes
    ----------
    prototype : str
        Class prototype to be used for class prototype creation.

    """

    def __init__(
        self,
        prototype: str = "mean",
        mean_centering: bool = False,
        return_df: bool = False,
    ):
        self.prototype = prototype
        self.mean_centering = mean_centering
        self.return_df = return_df

        assert self.prototype in [
            "mean",
            "median",
            "mad",
        ], "Class prototype not supported."

    def _mad_median(self, class_X, median=None):
        """Calculate upper and lower bounds for median absolute deviation."""
        _mad = median_abs_deviation(class_X, axis=0)

        low_value = median - _mad * 0.50
        high_value = median + _mad * 0.50
        # clip = lambda x: np.clip(x, low_value, high_value)
        class_X = np.apply_along_axis(
            _clip, axis=1, arr=class_X, low_value=low_value, high_value=high_value
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
            _prototype = prototype_funcs[self.prototype](train, y_ind)
            prototypes.append(_prototype)

        prototypes = np.stack(prototypes, axis=1)

        if self.mean_centering:
            prototypes -= np.mean(prototypes, axis=2, keepdims=True)

        return (prototypes, le.classes_)


class ElbowClassSum(BaseTransformer):
    """Elbow Class Sum (ECS) transformer to select a subset of channels/variables.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class prototype. The
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
    class_prototype : str
        Type of class prototype to use for representing a class.
        Default: 'mean'
    mean_centering : bool
        If True, mean centering is applied to the class prototype.
        Default: False


    Attributes
    ----------
    class_prototype_ : DataFrame
        Class prototype for each class.
    distance_frame_ : DataFrame
        Distance matrix for each class pair.
        ``shape = [n_channels, n_class_prototype_pairs]``
    channels_selected_idx : list
        List of selected channels.
    rank: list
        Rank of channels based on the distance between class prototypes.
    class_prototype_ : DataFrame
        Class prototype for each class.


    Notes
    -----
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
    >>> from sktime.transformations.panel.channel_selection import ElbowClassSum
    >>> from sktime.utils._testing.panel import make_classification_problem
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
        distance_: str = "euclidean",
        prototype: str = "mean",
        mean_centering: bool = False,
    ):
        self.distance_ = distance_
        self.mean_centering = mean_centering
        self.prototype = prototype
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
        centroid_obj = ClassPrototype(
            prototype=self.prototype,
            mean_centering=self.mean_centering,
        )
        self.prototype, labels = centroid_obj.create_prototype(X.copy(), y)

        # obj = DistanceMatrix(self.distance_)

        self.distance_frame = create_distance_matrix(
            self.prototype.copy(), labels, distance_=self.distance_
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
        assert self._is_fitted, "fit() must be called before transform()"
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
    class_prototype : str
        Type of class prototype to use for representing a class.
        Default: 'mean'
    mean_centering : bool
        If True, mean centering is applied to the class prototype.
        Default: False


    Attributes
    ----------
    class_prototype_ : DataFrame
        Class prototype for each class.
    distance_frame_ : DataFrame
        Distance matrix for each class pair.
        ``shape = [n_channels, n_class_prototype_pairs]``
    channels_selected_idx : list
        List of selected channels.
    rank: list
        Rank of channels based on the distance between class prototypes.
    class_prototype_ : DataFrame
        Class prototype for each class.

    Notes
    -----
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
    >>> from sktime.transformations.panel.channel_selection import ElbowClassPairwise
    >>> from sktime.utils._testing.panel import make_classification_problem
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
        class_prototype: str = "mad",
        mean_centering: bool = False,
    ):
        self.distance = distance
        self.class_prototype = class_prototype
        self.mean_centering = mean_centering
        self._is_fitted = False

        super(ElbowClassPairwise, self).__init__()

    def _rank(self) -> List[int]:
        all_index = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        series = self.distance_frame.sum(axis=1)
        series.drop(
            index=list(set(all_index) - set(self.channels_selected_idx)), inplace=True
        )
        return series.sort_values(ascending=False).index.tolist()

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
        centroid_obj = ClassPrototype(
            prototype=self.class_prototype, mean_centering=self.mean_centering
        )
        self.class_prototype_, labels = centroid_obj.create_prototype(
            X.copy(), y
        )  # Centroid created here
        # obj = DistanceMatrix(distance=self.distance)
        self.distance_frame = create_distance_matrix(
            self.class_prototype_.copy(), labels, self.distance
        )  # Distance matrix created here

        self.channels_selected_idx = []
        for pairdistance in self.distance_frame.items():
            distance_ = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            chs_dis = _detect_knee_point(distance_, indices)
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
        assert self._is_fitted, "Transformer must be fitted before calling transform"
        return X[:, self.channels_selected_idx]

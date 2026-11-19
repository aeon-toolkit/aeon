"""PULSAR interval classifier."""

__maintainer__ = []
__all__ = ["PULSARClassifier"]

from time import perf_counter
from typing import NamedTuple

import numpy as np
from numba import njit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.utils.validation import check_n_jobs

_DEFAULT_REPRESENTATIONS = (
    "original",
    "periodogram",
    "derivative",
    "autoregressive",
)
_DEFAULT_LOCAL_STATISTICS = (
    "mean",
    "stdev",
    "slope",
    "min",
    "max",
    "iqr",
    "median",
)
_DEFAULT_POOLING_OPERATORS = (
    "max",
    "mean",
    "min",
    "median",
    "iqr",
    "stdev",
    "mean_crossing_count",
    "values_above_mean",
    "slope",
)
_VALID_REPRESENTATIONS = set(_DEFAULT_REPRESENTATIONS)
_VALID_LOCAL_STATISTICS = set(_DEFAULT_LOCAL_STATISTICS)
_VALID_POOLING_OPERATORS = set(_DEFAULT_POOLING_OPERATORS)


class _FeatureMetadata(NamedTuple):
    """Compact description of one unscaled feature column."""

    representation: str
    interval_length: int
    dilation: int
    local_statistic: str
    pooling_operator: str
    partition_start: int
    partition_end: int
    level: int


class _PartitionState(NamedTuple):
    """A half-open response-sequence partition and its selected operators."""

    start: int
    end: int
    level: int
    operators: tuple


class _IntervalState(NamedTuple):
    """A fitted interval configuration and its hierarchy selections."""

    length: int
    dilation: int
    partitions: tuple


class _RepresentationState(NamedTuple):
    """A fitted representation and all valid interval configurations."""

    name: str
    intervals: tuple


@njit(cache=True)
def _histogram_median_iqr(values, bins=64):
    """Compute the reference histogram approximations for median and IQR."""
    n_rows, n_values = values.shape
    medians = np.empty(n_rows, dtype=np.float32)
    iqrs = np.empty(n_rows, dtype=np.float32)

    for row_index in range(n_rows):
        row = values[row_index]
        row_min = row[0]
        row_max = row[0]
        for value_index in range(1, n_values):
            value = row[value_index]
            if value < row_min:
                row_min = value
            if value > row_max:
                row_max = value

        if row_min == row_max:
            medians[row_index] = row_min
            iqrs[row_index] = 0.0
            continue

        width = (row_max - row_min) / bins
        counts = np.zeros(bins, dtype=np.int32)
        for value_index in range(n_values):
            bin_index = int((row[value_index] - row_min) / width)
            if bin_index >= bins:
                bin_index = bins - 1
            counts[bin_index] += 1

        cumulative = 0
        q1_bin = -1
        median_bin = -1
        q3_bin = -1
        for bin_index in range(bins):
            cumulative += counts[bin_index]
            if q1_bin < 0 and cumulative >= n_values * 0.25:
                q1_bin = bin_index
            if median_bin < 0 and cumulative >= n_values * 0.50:
                median_bin = bin_index
            if q3_bin < 0 and cumulative >= n_values * 0.75:
                q3_bin = bin_index

        q1 = row_min + (q1_bin + 0.5) * width
        median = row_min + (median_bin + 0.5) * width
        q3 = row_min + (q3_bin + 0.5) * width
        medians[row_index] = median
        iqrs[row_index] = q3 - q1

    return medians, iqrs


def _sliding_intervals(X, length, dilation):
    """Return a read-only view of all valid dilated intervals."""
    n_cases, n_timepoints = X.shape
    n_positions = n_timepoints - (length - 1) * dilation
    if n_positions <= 0:
        raise ValueError("The interval configuration is not valid for the series")

    return np.lib.stride_tricks.as_strided(
        X,
        shape=(n_cases, n_positions, length),
        strides=(X.strides[0], X.strides[1], dilation * X.strides[1]),
        writeable=False,
    )


def _generate_interval_configurations(input_length, interval_lengths, max_dilation):
    """Enumerate valid base lengths and power-of-two dilations."""
    configurations = []
    for length in interval_lengths:
        if length > input_length:
            continue
        maximum = (input_length - 1) // (length - 1)
        for exponent in range(int(np.floor(np.log2(min(maximum, max_dilation)))) + 1):
            configurations.append((int(length), 2**exponent))
    return tuple(configurations)


def _make_partitions(n_positions, depth):
    """Build contiguous, half-open partitions for each hierarchy level."""
    partitions = []
    for level in range(depth):
        n_parts = 2**level
        if n_parts > n_positions:
            break
        base, remainder = divmod(n_positions, n_parts)
        start = 0
        for part in range(n_parts):
            end = start + base + (part < remainder)
            partitions.append((start, end, level))
            start = end
    return partitions


def _local_statistic_matrix(intervals, statistics):
    """Calculate local statistics for flattened interval rows."""
    values = np.asarray(intervals, dtype=np.float32)
    n_rows, length = values.shape
    accumulated = values.astype(np.float64)
    result = []

    coordinate = np.arange(length, dtype=np.float32)
    sum_coordinate = coordinate.sum()
    sum_coordinate_squared = np.square(coordinate).sum()
    slope_denominator = length * sum_coordinate_squared - sum_coordinate**2

    for statistic in statistics:
        if statistic == "mean":
            column = np.mean(accumulated, axis=1)
        elif statistic == "stdev":
            mean = np.mean(accumulated, axis=1)
            variance = np.mean(np.square(accumulated), axis=1) - np.square(mean)
            column = np.where(variance > 1e-14, np.sqrt(variance), 0.0)
        elif statistic == "slope":
            weighted_sum = accumulated @ coordinate
            numerator = length * weighted_sum - sum_coordinate * accumulated.sum(axis=1)
            column = (
                numerator / slope_denominator
                if slope_denominator != 0
                else np.zeros(n_rows)
            )
        elif statistic == "min":
            column = np.min(values, axis=1)
        elif statistic == "max":
            column = np.max(values, axis=1)
        elif statistic in ("median", "iqr"):
            median, iqr = _histogram_median_iqr(values)
            column = median if statistic == "median" else iqr
        else:  # pragma: no cover - validated before this function is called
            raise ValueError(f"Unknown local statistic: {statistic}")
        result.append(np.asarray(column, dtype=np.float32))

    return np.column_stack(result).astype(np.float32, copy=False)


def _response_map(X, length, dilation, statistics):
    """Calculate a statistic-by-position response map for one interval config."""
    intervals = _sliding_intervals(X, length, dilation)
    n_cases, n_positions, _ = intervals.shape
    local = _local_statistic_matrix(intervals.reshape(-1, length), statistics)
    return local.reshape(n_cases, n_positions, len(statistics)).transpose(0, 2, 1)


def _pool_rows(values, operator):
    """Apply one pooling operator to rows of a response sequence."""
    values = np.asarray(values, dtype=np.float32)
    accumulated = values.astype(np.float64)
    n_values = values.shape[1]
    if operator == "max":
        return np.max(values, axis=1)
    if operator == "mean":
        return np.mean(accumulated, axis=1).astype(np.float32)
    if operator == "min":
        return np.min(values, axis=1)
    if operator == "median":
        return _histogram_median_iqr(values)[0]
    if operator == "iqr":
        return _histogram_median_iqr(values)[1]
    if operator == "stdev":
        mean = np.mean(accumulated, axis=1)
        variance = np.mean(np.square(accumulated), axis=1) - np.square(mean)
        result = np.where(variance > 1e-6, np.sqrt(variance), 0.0)
        return np.where(result > 1e-6, result, 0).astype(np.float32)
    if operator == "slope":
        coordinate = np.arange(n_values, dtype=np.float32)
        sum_coordinate = coordinate.sum()
        denominator = n_values * np.square(coordinate).sum() - sum_coordinate**2
        if denominator == 0:
            return np.zeros(values.shape[0], dtype=np.float32)
        numerator = n_values * (accumulated @ coordinate) - sum_coordinate * (
            accumulated.sum(axis=1)
        )
        return (numerator / denominator).astype(np.float32)
    if operator in ("mean_crossing_count", "values_above_mean"):
        means = np.mean(accumulated, axis=1)
        if operator == "values_above_mean":
            return np.mean(values > means[:, None], axis=1).astype(np.float32)
        if n_values <= 1:
            return np.zeros(values.shape[0], dtype=np.float32)
        previous = values[:, :-1]
        current = values[:, 1:]
        crossings = ((previous <= means[:, None]) & (current > means[:, None])) | (
            (previous >= means[:, None]) & (current < means[:, None])
        )
        return (crossings.sum(axis=1) / (n_values - 1)).astype(np.float32)
    raise ValueError(f"Unknown pooling operator: {operator}")


def _fisher_scores(X, y):
    """Calculate the class-separation score used for candidate selection."""
    scores = np.zeros(X.shape[1], dtype=np.float64)
    denominator = np.zeros(X.shape[1], dtype=np.float64)
    overall_mean = np.mean(X, axis=0)
    for label in np.unique(y):
        values = X[y == label]
        class_mean = np.mean(values, axis=0)
        if values.shape[0] > 1:
            class_std = np.std(values, axis=0, ddof=1)
        else:
            class_std = np.full(X.shape[1], 1e-5)
        class_std = np.maximum(class_std, 1e-4)
        scores += values.shape[0] * np.square(class_mean - overall_mean)
        # The reference uses sample variance for class-wise Fisher dispersion.
        denominator += values.shape[0] * np.square(class_std)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = scores / denominator
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _ar_order(X):
    """Return the truncated AR order used by the published configuration."""
    return int(12 * (X.shape[-1] / 100.0) ** 0.25)


def _decision_probabilities(estimator, X):
    """Convert an uncalibrated classifier's decision output to probabilities."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    decision = np.asarray(estimator.decision_function(X))
    if decision.ndim == 1:
        decision = np.column_stack((-decision, decision))
    decision -= np.max(decision, axis=1, keepdims=True)
    probabilities = np.exp(decision)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class PULSARClassifier(BaseClassifier):
    """PULSAR interval classifier.

    PULSAR (Pooled multi-scale summaries from randomized intervals) creates four
    univariate representations, applies local statistics to every valid sliding
    dilated interval, and pools each response sequence over a hierarchy of
    contiguous partitions. All coarsest-level features are retained, while finer
    pooled features and the raw representation values are ranked by a Fisher score.
    The selected features are standardized and passed to calibrated Ridge and
    Extra-Trees classifiers whose probabilities are averaged.

    Parameters
    ----------
    representations : tuple of str or None, default=None
        Representations to use. ``None`` selects ``("original", "periodogram",
        "derivative", "autoregressive")``.
    interval_lengths : tuple of int, default=(7, 9, 11)
        Base interval lengths.
    max_dilation : int, default=16
        Maximum power-of-two dilation.
    local_statistics : tuple of str or None, default=None
        Local statistics. ``None`` selects the seven published statistics.
    pooling_operators : tuple of str or None, default=None
        Pooling operators. ``None`` selects the nine published operators.
    hierarchical_depth : int, default=4
        Number of hierarchy levels, including the global level.
    n_random_pooling_operators : int, default=6
        Number of pooling operators randomly retained per finer partition.
    feature_selection_percentage : float, default=40
        Percentage of finer pooled and raw candidate features retained by Fisher
        score. At least one candidate is retained when candidates exist.
    classifiers : tuple of str, default=("ridge", "extra_trees")
        Classifier heads to average. Valid values are ``"ridge"`` and
        ``"extra_trees"``.
    n_estimators : int, default=50
        Number of trees in the Extra-Trees head.
    n_jobs : int, default=1
        Number of jobs used by supported classifier heads.
    random_state : int, RandomState instance or None, default=None
        Controls pooling-operator selection and randomized classifier heads.

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels observed during fitting.
    n_features_in_ : int
        Number of scaled features passed to the classifier heads.
    selected_feature_indices_ : np.ndarray
        Candidate-column indices retained by Fisher selection.
    selected_feature_metadata_ : tuple
        Descriptions of retained candidate columns.
    fit_stage_times_ : dict
        Wall-clock timings for representation, response-map, pooling, selection,
        scaling, and classifier fitting stages.
    predict_stage_times_ : dict
        Timings for the corresponding prediction stages.

    See Also
    --------
    DrCIFClassifier
        A randomized catch22 interval forest using multiple representations.
    QUANTClassifier
        An interval classifier using hierarchical quantile features.

    Notes
    -----
    The autoregressive representation uses :class:`ARCoefficientTransformer`, so
    the optional ``statsmodels`` dependency is required for the default settings.
    The published implementation uses ``pyfftw`` and a deprecated prefit calibration
    API; this implementation uses aeon's FFT-compatible transformer and the current
    scikit-learn calibration API. The official implementation is available at
    https://github.com/stevcabello/PULSAR.

    References
    ----------
    .. [1] Cabello, N. and Kulik, L. "PULSAR: Advancing Interval-Based Time Series
       Classification to State-of-the-Art Performance." ICDM, 2025.

    Examples
    --------
    >>> from aeon.classification.interval_based import PULSARClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(
    ...     n_cases=10, n_channels=1, n_timepoints=20, random_state=0
    ... )
    >>> clf = PULSARClassifier(
    ...     representations=("original",), interval_lengths=(7,),
    ...     max_dilation=2, classifiers=("ridge",), random_state=0
    ... )
    >>> clf.fit(X, y)  # doctest: +SKIP
    PULSARClassifier(...)
    >>> clf.predict(X)  # doctest: +SKIP
    array([...])
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "X_inner_type": "numpy3D",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        representations=None,
        interval_lengths=(7, 9, 11),
        max_dilation=16,
        local_statistics=None,
        pooling_operators=None,
        hierarchical_depth=4,
        n_random_pooling_operators=6,
        feature_selection_percentage=40,
        classifiers=("ridge", "extra_trees"),
        n_estimators=50,
        n_jobs=1,
        random_state=None,
    ):
        self.representations = representations
        self.interval_lengths = interval_lengths
        self.max_dilation = max_dilation
        self.local_statistics = local_statistics
        self.pooling_operators = pooling_operators
        self.hierarchical_depth = hierarchical_depth
        self.n_random_pooling_operators = n_random_pooling_operators
        self.feature_selection_percentage = feature_selection_percentage
        self.classifiers = classifiers
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self):
        """Validate constructor values used by feature generation."""
        representations = (
            _DEFAULT_REPRESENTATIONS
            if self.representations is None
            else self.representations
        )
        local_statistics = (
            _DEFAULT_LOCAL_STATISTICS
            if self.local_statistics is None
            else self.local_statistics
        )
        pooling_operators = (
            _DEFAULT_POOLING_OPERATORS
            if self.pooling_operators is None
            else self.pooling_operators
        )
        for name, values, valid in (
            ("representations", representations, _VALID_REPRESENTATIONS),
            ("local_statistics", local_statistics, _VALID_LOCAL_STATISTICS),
            ("pooling_operators", pooling_operators, _VALID_POOLING_OPERATORS),
        ):
            if (
                isinstance(values, str)
                or not values
                or any(value not in valid for value in values)
            ):
                raise ValueError(f"Invalid {name}")
        if len(set(representations)) != len(representations):
            raise ValueError("representations must not contain duplicates")
        if (
            isinstance(self.interval_lengths, (int, np.integer))
            or not self.interval_lengths
        ):
            raise ValueError("interval_lengths must be a non-empty sequence")
        if any(
            not isinstance(length, (int, np.integer)) or length < 2
            for length in self.interval_lengths
        ):
            raise ValueError("interval_lengths must contain integers >= 2")
        if (
            not isinstance(self.max_dilation, (int, np.integer))
            or self.max_dilation < 1
        ):
            raise ValueError("max_dilation must be >= 1")
        if (
            not isinstance(self.hierarchical_depth, (int, np.integer))
            or self.hierarchical_depth < 1
        ):
            raise ValueError("hierarchical_depth must be >= 1")
        if (
            not isinstance(self.n_random_pooling_operators, (int, np.integer))
            or self.n_random_pooling_operators < 1
        ):
            raise ValueError("n_random_pooling_operators must be >= 1")
        if not 0 <= self.feature_selection_percentage <= 100:
            raise ValueError("feature_selection_percentage must be in [0, 100]")
        if (
            isinstance(self.classifiers, str)
            or not self.classifiers
            or any(
                classifier not in ("ridge", "extra_trees")
                for classifier in self.classifiers
            )
        ):
            raise ValueError("classifiers must contain only 'ridge' and 'extra_trees'")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._representations = tuple(representations)
        self._local_statistics = tuple(local_statistics)
        self._pooling_operators = tuple(pooling_operators)

    def _get_representation(self, X, name):
        """Generate one univariate representation."""
        series = X[:, 0, :]
        if name == "original":
            return np.asarray(series, dtype=np.float64)
        if name == "derivative":
            return np.diff(series, axis=1).astype(np.float64, copy=False)
        if name == "periodogram":
            transformed = PeriodogramTransformer(pad_series=False).fit_transform(X)
            return np.asarray(transformed[:, 0, :], dtype=np.float64)
        if name == "autoregressive":
            # Burg cannot estimate an order for the very shortest series. Skipping
            # this optional representation leaves the other representations usable.
            if series.shape[1] < 6:
                return None
            transformed = ARCoefficientTransformer(
                order=_ar_order, replace_nan=True
            ).fit_transform(X)
            return np.asarray(transformed[:, 0, :], dtype=np.float64)
        raise ValueError(f"Unknown representation: {name}")

    def _new_interval_states(self, input_length, rng):
        """Create interval and random pooling state for one representation."""
        states = []
        for length, dilation in _generate_interval_configurations(
            input_length, self.interval_lengths, self.max_dilation
        ):
            n_positions = input_length - (length - 1) * dilation
            selected_partitions = []
            for start, end, level in _make_partitions(
                n_positions, self.hierarchical_depth
            ):
                if level == 0:
                    operators = tuple(range(len(self._pooling_operators)))
                else:
                    n_selected = min(
                        self.n_random_pooling_operators,
                        len(self._pooling_operators),
                    )
                    operators = tuple(
                        rng.choice(
                            len(self._pooling_operators),
                            size=n_selected,
                            replace=False,
                        ).tolist()
                    )
                selected_partitions.append(
                    _PartitionState(start, end, level, operators)
                )
            states.append(_IntervalState(length, dilation, tuple(selected_partitions)))
        return tuple(states)

    def _pool_response_map(self, response, representation, interval, fitting):
        """Pool one response map and return global/local columns and metadata."""
        global_columns = []
        local_columns = []
        global_metadata = []
        local_metadata = []
        for statistic_index, statistic in enumerate(self._local_statistics):
            for operator_index, operator in enumerate(self._pooling_operators):
                for partition in interval.partitions:
                    if operator_index not in partition.operators:
                        continue
                    if partition.end - partition.start <= 1:
                        continue
                    values = response[
                        :, statistic_index, partition.start : partition.end
                    ]
                    pooled = _pool_rows(values, operator)
                    metadata = _FeatureMetadata(
                        representation,
                        interval.length,
                        interval.dilation,
                        statistic,
                        operator,
                        partition.start,
                        partition.end,
                        partition.level,
                    )
                    if partition.level == 0:
                        global_columns.append(pooled)
                        global_metadata.append(metadata)
                    else:
                        local_columns.append(pooled)
                        local_metadata.append(metadata)
        return (
            local_columns,
            global_columns,
            local_metadata,
            global_metadata,
        )

    def _feature_transform(self, X, fitting):
        """Generate unscaled global and candidate features."""
        stage_times = {
            "representation_generation": 0.0,
            "response_map_generation": 0.0,
            "hierarchical_pooling": 0.0,
            "fisher_scoring_and_selection": 0.0,
            "scaling": 0.0,
        }
        rng = check_random_state(self.random_state) if fitting else None
        global_columns = []
        candidate_columns = []
        global_metadata = []
        candidate_metadata = []
        states = []

        for representation in self._representations:
            start = perf_counter()
            transformed = self._get_representation(X, representation)
            stage_times["representation_generation"] += perf_counter() - start
            if transformed is None:
                continue

            intervals = (
                self._new_interval_states(transformed.shape[1], rng)
                if fitting
                else self._states_by_representation[representation].intervals
            )
            if fitting:
                states.append(_RepresentationState(representation, intervals))

            for interval in intervals:
                start = perf_counter()
                response = _response_map(
                    transformed,
                    interval.length,
                    interval.dilation,
                    self._local_statistics,
                )
                stage_times["response_map_generation"] += perf_counter() - start
                start = perf_counter()
                local, global_, local_meta, global_meta = self._pool_response_map(
                    response, representation, interval, fitting
                )
                stage_times["hierarchical_pooling"] += perf_counter() - start
                candidate_columns.extend(local)
                global_columns.extend(global_)
                candidate_metadata.extend(local_meta)
                global_metadata.extend(global_meta)

            # The raw representation values are deliberately part of the candidate
            # pool, after its finer pooled features, as in the published pipeline.
            candidate_columns.extend(transformed.T)
            candidate_metadata.extend(
                _FeatureMetadata(representation, 0, 0, "raw", "", index, index + 1, -1)
                for index in range(transformed.shape[1])
            )

        if fitting:
            self._representation_states_ = tuple(states)
            self._states_by_representation = {
                state.name: state for state in self._representation_states_
            }
            self.global_feature_metadata_ = tuple(global_metadata)
            self.candidate_feature_metadata_ = tuple(candidate_metadata)

        global_features = (
            np.column_stack(global_columns).astype(np.float64, copy=False)
            if global_columns
            else np.empty((X.shape[0], 0), dtype=np.float64)
        )
        candidates = (
            np.column_stack(candidate_columns).astype(np.float64, copy=False)
            if candidate_columns
            else np.empty((X.shape[0], 0), dtype=np.float64)
        )
        global_features = np.nan_to_num(
            global_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        candidates = np.nan_to_num(candidates, nan=0.0, posinf=0.0, neginf=0.0)

        if fitting:
            self._fit_candidate_features_ = candidates
            self._fit_global_features_ = global_features
        return global_features, candidates, stage_times

    def _fit_classifier(self, name, X, y):
        """Fit and calibrate one published classifier head."""
        if name == "ridge":
            estimator = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        else:
            estimator = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion="entropy",
                max_features=0.10,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        counts = np.unique(y, return_counts=True)[1]
        minimum_count = int(np.min(counts))
        if minimum_count >= 5:
            calibrated = CalibratedClassifierCV(
                estimator=estimator,
                method="sigmoid",
                cv=5,
                n_jobs=self.n_jobs,
            )
            calibrated.fit(X, y)
            return calibrated

        estimator.fit(X, y)
        if minimum_count >= 2:
            try:
                from sklearn.frozen import FrozenEstimator

                fitted = FrozenEstimator(estimator)
                calibrated = CalibratedClassifierCV(
                    estimator=fitted,
                    method="sigmoid",
                    cv=minimum_count,
                )
            except ImportError:  # pragma: no cover - for older sklearn versions
                calibrated = CalibratedClassifierCV(
                    estimator=estimator,
                    method="sigmoid",
                    cv="prefit",
                )
            calibrated.fit(X, y)
            return calibrated
        return estimator

    def _fit(self, X, y):
        """Fit feature generation, selection, scaling, and classifier heads."""
        self._validate_parameters()
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        if self.n_channels_ != 1:
            raise ValueError("PULSARClassifier only supports univariate series")

        start = perf_counter()
        global_features, candidates, stage_times = self._feature_transform(X, True)
        if candidates.shape[1] == 0:
            raise ValueError("PULSAR generated no candidate features")
        selection_start = perf_counter()
        scores = _fisher_scores(candidates, y)
        n_selected = max(
            1,
            min(
                candidates.shape[1],
                int(candidates.shape[1] * self.feature_selection_percentage / 100),
            ),
        )
        selected = np.argsort(scores, kind="stable")[-n_selected:]
        stage_times["fisher_scoring_and_selection"] = perf_counter() - selection_start
        self.selected_feature_indices_ = selected.astype(np.intp, copy=False)
        self.feature_selection_scores_ = scores
        self.selected_feature_metadata_ = tuple(
            self.candidate_feature_metadata_[index] for index in selected
        )

        combined = np.hstack((global_features, candidates[:, selected]))
        scaling_start = perf_counter()
        self.scaler_ = StandardScaler()
        combined = self.scaler_.fit_transform(combined)
        stage_times["scaling"] = perf_counter() - scaling_start
        self.n_features_in_ = combined.shape[1]
        self.n_global_features_ = global_features.shape[1]
        self.n_candidate_features_ = candidates.shape[1]
        self.n_selected_features_ = selected.shape[0]

        classifier_start = perf_counter()
        self.classifiers_ = {
            name: self._fit_classifier(name, combined, y) for name in self.classifiers
        }
        stage_times["classifier_fitting"] = perf_counter() - classifier_start
        stage_times["total"] = perf_counter() - start
        self.fit_stage_times_ = stage_times
        return self

    def _transform_for_prediction(self, X):
        """Generate, select, and scale features for new series."""
        global_features, candidates, stage_times = self._feature_transform(X, False)
        start = perf_counter()
        combined = np.hstack(
            (global_features, candidates[:, self.selected_feature_indices_])
        )
        combined = self.scaler_.transform(combined)
        stage_times["scaling"] = perf_counter() - start
        stage_times["total"] = sum(stage_times.values())
        self.predict_stage_times_ = stage_times
        return combined

    def _predict_proba(self, X):
        """Predict class probabilities from the averaged calibrated heads."""
        features = self._transform_for_prediction(X)
        probabilities = []
        start = perf_counter()
        for estimator in self.classifiers_.values():
            predicted = _decision_probabilities(estimator, features)
            aligned = np.zeros((len(X), self.n_classes_), dtype=np.float64)
            for index, label in enumerate(estimator.classes_):
                aligned[:, self._class_dictionary[label]] = predicted[:, index]
            probabilities.append(aligned)
        result = np.mean(probabilities, axis=0)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = result.sum(axis=1, keepdims=True)
        result = np.divide(
            result,
            row_sums,
            out=np.full_like(result, 1 / self.n_classes_),
            where=row_sums != 0,
        )
        self.predict_stage_times_["classifier_prediction"] = perf_counter() - start
        self.predict_stage_times_["total"] += self.predict_stage_times_[
            "classifier_prediction"
        ]
        return result

    def _predict(self, X):
        """Predict class labels using the highest averaged probability."""
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a small parameter set for estimator checks."""
        return {
            "representations": ("original", "derivative"),
            "interval_lengths": (3,),
            "max_dilation": 2,
            "local_statistics": ("mean", "stdev"),
            "pooling_operators": ("max", "mean"),
            "hierarchical_depth": 2,
            "n_random_pooling_operators": 1,
            "feature_selection_percentage": 40,
            "classifiers": ("ridge",),
            "n_jobs": 1,
            "random_state": 0,
        }

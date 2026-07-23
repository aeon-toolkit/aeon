"""Shapelet transform.

A transformer from the time domain into the shapelet domain.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RandomShapeletTransform"]

import heapq
import math
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numba.typed.typedlist import List
from sklearn import preprocessing
from sklearn.utils._random import check_random_state

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD, z_normalise_series
from aeon.utils.validation import check_n_jobs

# Column layout of a shapelet record, constant across its whole lifecycle. The
# working records built during fitting are a numba ``List`` of floats populating
# indices 0-6; the fitted records in ``self.shapelets`` extend this with the
# z-normalised subsequence at index 7 (``_VALUES``). Every index means the same
# thing in both forms.
(
    _QUALITY,  # information-gain score separating the classes (higher is better)
    _LENGTH,  # number of time points in the shapelet
    _POSITION,  # start index of the subsequence within its source series
    _CHANNEL,  # channel the subsequence was taken from
    _CASE,  # index of the source train case the shapelet was extracted from
    _CLASS,  # encoded class index; the label is ``classes_[_CLASS]``
    _DIST,  # row in the cached distance-vectors list, or -1 when not caching
    _VALUES,  # z-normalised subsequence values; present in fitted records only
) = range(8)


class RandomShapeletTransform(BaseCollectionTransformer):
    """Random Shapelet Transform.

    Implementation of the binary shapelet transform along the lines of [1]_, [2]_, with
    randomly extracted shapelets. A shapelet is a subsequence from the training set. The
    transform finds a set of shapelets that are good at separating the classes based on
    the distances between shapelets and whole series. The distance between a shapelet
    and a series (called sDist in the literature) is the minimum mean squared distance
    between the z-normalised shapelet and every z-normalised window of the same length
    in the series.

    Given ``n`` series with ``d`` channels, candidate shapelets are extracted and
    filtered in batches.

    For each candidate shapelet:

    - Extract a shapelet from an instance with random length, position and channel.
    - Z-normalise the shapelet.
    - Find the distance from the shapelet to all training cases.
    - Derive a binary orderline and score the shapelet by information gain.
    - Retain only the best shapelets per class.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        Number of candidate shapelets to assess. Ignored when
        ``time_limit_in_minutes > 0``.
    max_shapelets : int or None, default=None
        Maximum number of shapelets to keep. The shapelet budget is divided equally
        among the classes. If None, set to ``min(10 * n_cases, 1000)`` during fit.
    min_shapelet_length : int, default=3
        Lower bound on candidate shapelet lengths.
    max_shapelet_length : int or None, default=None
        Upper bound on candidate shapelet lengths. If None, the length of the longest
        input series is used. A candidate is also limited by the length of its source
        series.
    remove_self_similar : bool, default=True
        Remove overlapping "self-similar" shapelets when merging candidate shapelets.
    batch_size : int, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets.
    verbose : bool, default=False
        Whether to print progress messages during fitting.
    time_limit_in_minutes : float, default=0.0
        Time contract to limit build time in minutes, overriding n_shapelet_samples.
        The default of 0 means ``n_shapelet_samples`` is used.
    contract_max_n_shapelet_samples : float, default=np.inf
        Maximum number of shapelets to extract when ``time_limit_in_minutes`` is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``transform``.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Joblib parallelisation backend. If None, ``prefer="threads"`` is used. Valid
        options include ``"loky"``, ``"multiprocessing"``, ``"threading"``, or a
        custom backend. See the Joblib ``Parallel`` documentation for details.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_cases_ : int
        The number of training cases.
    n_channels_ : int
        The number of channels per case.
    max_n_timepoints_ : int
        The maximum series length in the training data.
    classes_ : np.ndarray
        The class labels.
    shapelets : list of tuple
        The stored shapelets after fitting. Each tuple has a fixed layout
        ``(quality, length, position, channel, case_index, class_index,
        distance_index, values)``: ``class_index`` indexes ``classes_``,
        ``distance_index`` is reserved for fitting and reset to ``-1`` in fitted
        records, and ``values`` is the z-normalised subsequence from the source case.
        The layout matches the numeric working records used during fitting, which
        populate every field except ``values``.

    See Also
    --------
    ShapeletTransformClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/time-series-machine-learning/tsml-java/src/java/tsml/>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851-881, 2014.

    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge-Centered
       Systems, 32, 2017.
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "requires_y": True,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        n_shapelet_samples: int = 10000,
        max_shapelets: int | None = None,
        min_shapelet_length: int = 3,
        max_shapelet_length: int | None = None,
        remove_self_similar: bool = True,
        batch_size: int = 100,
        verbose: bool = False,
        time_limit_in_minutes: float = 0.0,
        contract_max_n_shapelet_samples: float = np.inf,
        n_jobs: int = 1,
        parallel_backend=None,
        random_state: int | None = None,
    ) -> None:
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar
        self.batch_size = batch_size
        self.verbose = verbose
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.random_state = random_state

        # Set in fit
        self.n_classes_ = 0
        self.n_cases_ = 0
        self.n_channels_ = 0
        self.max_n_timepoints_ = 0
        self.classes_ = []
        self.shapelets = []

        # Protected attributes
        self._max_shapelets = max_shapelets
        self._max_shapelet_length = max_shapelet_length
        self._n_jobs = n_jobs
        self._batch_size = batch_size
        self._class_counts = []
        self._sorted_indices = []

        super().__init__()

    def _fit(self, X, y=None):
        self._fit_shapelets(X, y, cache_distance_vectors=False)

    def _fit_transform(self, X, y=None):
        """Fit while caching the transformed training data."""
        return self._fit_shapelets(X, y, cache_distance_vectors=True)

    def _fit_shapelets(self, X, y, cache_distance_vectors):
        """Fit shapelets and optionally return transformed training data.

        Scoring a candidate already computes its distance to every training case
        (the orderline used for information gain). When ``cache_distance_vectors``
        is set by ``fit_transform``, those per-shapelet vectors of length ``n_cases``
        are kept in ``distance_vectors`` and stacked into ``Xt`` at the end, so the
        training transform is not recomputed. ``_fit`` leaves it off.
        """
        y = self._set_fit_attributes(X, y)
        X_values, X_offsets = self._pack_collection(X)

        max_shapelets_per_class = int(self._max_shapelets / self.n_classes_)
        if max_shapelets_per_class < 1:
            max_shapelets_per_class = 1
        # Cache of distance vectors for the currently kept shapelets, indexed by
        # their _DIST field; stays empty unless cache_distance_vectors is set.
        distance_vectors = []
        rng = check_random_state(self.random_state)

        # Candidate store indexed by encoded class: the outer Numba typed list
        # holds one min-heap of mutable shapelet records per class. Records use
        # the _QUALITY.._DIST field layout. A negative-quality sentinel seeds each
        # heap to establish the nested Numba type and is filtered during finalisation.
        shapelets_by_class = List(
            [
                List([List([-1.0, -1, -1, -1, -1, -1, -1])])
                for _ in range(self.n_classes_)
            ]
        )
        # Find and score shapelets by time limit or by number of shapelets
        shapelets_by_class, distance_vectors = self._run_fit_batches(
            X_values=X_values,
            X_offsets=X_offsets,
            y=y,
            shapelets_by_class=shapelets_by_class,
            distance_vectors=distance_vectors,
            max_shapelets_per_class=max_shapelets_per_class,
            cache_distance_vectors=cache_distance_vectors,
            rng=rng,
        )
        # Extract all shapelet parameters and normalised shapelets
        self.shapelets = []
        for class_shapelets in shapelets_by_class:
            for shapelet in class_shapelets:
                if shapelet[_QUALITY] <= 0:
                    continue

                values = z_normalise_series(
                    X[int(shapelet[_CASE])][int(shapelet[_CHANNEL])][
                        int(shapelet[_POSITION]) : int(shapelet[_POSITION])
                        + int(shapelet[_LENGTH])
                    ]
                )
                self.shapelets.append(
                    (
                        shapelet[_QUALITY],
                        int(shapelet[_LENGTH]),
                        int(shapelet[_POSITION]),
                        int(shapelet[_CHANNEL]),
                        int(shapelet[_CASE]),
                        int(shapelet[_CLASS]),
                        int(shapelet[_DIST]),
                        values,
                    )
                )
        # Sort by quality
        self.shapelets.sort(
            reverse=True,
            key=lambda s: (
                s[_QUALITY],
                -s[_LENGTH],
                s[_POSITION],
                s[_CHANNEL],
                s[_CASE],
            ),
        )
        if self.shapelets:
            to_keep = self._remove_identical_shapelets(List(self.shapelets))
            self.shapelets = [n for (n, b) in zip(self.shapelets, to_keep) if b]

        if self.verbose:
            print(f"Final shapelet count: {len(self.shapelets)}")  # noqa: T201

        self._sorted_indices = []
        for s in self.shapelets:
            sabs = np.abs(s[_VALUES])
            self._sorted_indices.append(
                np.array(
                    sorted(
                        range(s[_LENGTH]),
                        reverse=True,
                        key=lambda j, sabs=sabs: sabs[j],
                    ),
                    dtype=np.int32,
                )
            )

        self._build_transform_inputs()

        if cache_distance_vectors:
            Xt = np.array(
                [distance_vectors[s[_DIST]] for s in self.shapelets]
            ).transpose()
            self.shapelets = [
                s[:_DIST] + (-1,) + s[_DIST + 1 :] for s in self.shapelets
            ]
            return Xt

        return None

    def _run_fit_batches(
        self,
        X_values,
        X_offsets,
        y,
        shapelets_by_class,
        distance_vectors,
        max_shapelets_per_class,
        cache_distance_vectors,
        rng,
    ):
        """Fit candidate batches using fixed-count or time-contracted stopping."""
        contracted = self.time_limit_in_minutes > 0
        mode = "contract" if contracted else "fixed"
        if contracted:
            time_limit = self.time_limit_in_minutes * 60
            sample_limit = self.contract_max_n_shapelet_samples
        else:
            time_limit = np.inf
            sample_limit = self.n_shapelet_samples
        total_start = perf_counter()

        n_shapelets_extracted = 0
        timed_shapelets_extracted = 0
        timed_elapsed = 0.0
        avg_batch_time = None
        n_batches = 0

        while (
            perf_counter() - total_start < time_limit
            and n_shapelets_extracted < sample_limit
        ):
            batch_size = self._batch_size
            if not contracted:
                batch_size = min(
                    batch_size,
                    sample_limit - n_shapelets_extracted,
                )

            batch_elapsed, current_kept = self._process_fit_batch(
                X_values=X_values,
                X_offsets=X_offsets,
                y=y,
                shapelets_by_class=shapelets_by_class,
                distance_vectors=distance_vectors,
                max_shapelets_per_class=max_shapelets_per_class,
                cache_distance_vectors=cache_distance_vectors,
                rng=rng,
                start_idx=n_shapelets_extracted,
                batch_size=batch_size,
            )

            n_shapelets_extracted += batch_size
            n_batches += 1
            total_elapsed = perf_counter() - total_start

            # Ignore first batch for averages and rate estimates, it is dominated
            # by numba/joblib start-up.
            if n_batches > 1:
                timed_shapelets_extracted += batch_size
                timed_elapsed += batch_elapsed
                avg_batch_time = self._update_average_batch_time(
                    avg_batch_time, batch_elapsed
                )

                if self.verbose:
                    self._log_fit_progress(
                        mode=mode,
                        n_shapelets_extracted=n_shapelets_extracted,
                        timed_shapelets_extracted=timed_shapelets_extracted,
                        current_kept=current_kept,
                        timed_elapsed=timed_elapsed,
                        total_elapsed=total_elapsed,
                        avg_batch_time=avg_batch_time,
                        time_limit=time_limit,
                        n_shapelet_samples=self.n_shapelet_samples,
                    )

        return shapelets_by_class, distance_vectors

    def _process_fit_batch(
        self,
        X_values,
        X_offsets,
        y,
        shapelets_by_class,
        distance_vectors,
        max_shapelets_per_class,
        cache_distance_vectors,
        rng,
        start_idx,
        batch_size,
    ):
        """Extract, merge, and post-process one batch of candidate shapelets."""
        batch_start = perf_counter()
        # Draw all sampling parameters up front on the main thread. A single
        # RandomState is reused, reseeded per candidate from the parent stream:
        # this is bit-identical to constructing a fresh RandomState each time but
        # ~16x cheaper, and it keeps the parallelised scoring free of RNG state.
        child_rng = np.random.RandomState()
        params = []
        for i in range(batch_size):
            child_rng.seed(rng.randint(np.iinfo(np.int32).max))
            params.append(self._sample_shapelet(X_offsets, child_rng, start_idx + i))

        if self._n_jobs == 1:
            results = [
                self._score_shapelet(X_values, X_offsets, y, cache_distance_vectors, *p)
                for p in params
            ]
        else:
            results = Parallel(
                n_jobs=self._n_jobs,
                backend=self.parallel_backend,
                prefer="threads",
            )(
                delayed(self._score_shapelet)(
                    X_values, X_offsets, y, cache_distance_vectors, *p
                )
                for p in params
            )
        candidate_shapelets, candidate_distance_vectors = zip(*results)

        if cache_distance_vectors:
            # Append every candidate's distance vector and point its _DIST at it.
            for i, shapelet in enumerate(candidate_shapelets):
                shapelet[_DIST] = len(distance_vectors) + i
            distance_vectors.extend(candidate_distance_vectors)

        candidate_shapelets = List([List(shapelet) for shapelet in candidate_shapelets])

        for class_idx, heap in enumerate(shapelets_by_class):
            self._merge_shapelets(
                heap,
                candidate_shapelets,
                max_shapelets_per_class,
                class_idx,
            )

        if self.remove_self_similar:
            for class_idx, heap in enumerate(shapelets_by_class):
                to_keep = self._remove_self_similar_shapelets(heap)
                shapelets_by_class[class_idx] = List(
                    [shapelet for shapelet, keep in zip(heap, to_keep) if keep]
                )

        if cache_distance_vectors:
            # Drop vectors for candidates pruned by the merge / self-similar
            # steps and renumber _DIST so it stays dense over the kept shapelets.
            new_distance_vectors = []
            dist_idx = 0
            for heap in shapelets_by_class:
                for shapelet in heap:
                    new_distance_vectors.append(distance_vectors[int(shapelet[_DIST])])
                    shapelet[_DIST] = dist_idx
                    dist_idx += 1
            distance_vectors[:] = new_distance_vectors

        batch_elapsed = perf_counter() - batch_start
        current_kept = sum(len(heap) for heap in shapelets_by_class)

        return batch_elapsed, current_kept

    def _set_fit_attributes(self, X, y):
        """Set fitted collection metadata and resolve shapelet parameters."""
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.classes_, self._class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = self.classes_.shape[0]

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        self.n_cases_ = len(X)
        self.n_channels_ = X[0].shape[0]
        self.max_n_timepoints_ = max(x.shape[1] for x in X)

        # Lookup table k*log2(k) (with 0 at k=0) so the information-gain scan
        # over split points uses table entries instead of repeated logarithms.
        ks = np.arange(self.n_cases_ + 1, dtype=np.float64)
        self._klog = np.zeros(self.n_cases_ + 1, dtype=np.float64)
        self._klog[1:] = ks[1:] * np.log2(ks[1:])

        if self.max_shapelets is None:
            self._max_shapelets = min(10 * self.n_cases_, 1000)
        else:
            self._max_shapelets = self.max_shapelets
        if self._max_shapelets < self.n_classes_:
            self._max_shapelets = self.n_classes_

        self._max_shapelet_length = self.max_shapelet_length
        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.max_n_timepoints_

        minl = min(x.shape[1] for x in X)
        self._min_shapelet_length = self.min_shapelet_length
        if minl < self.min_shapelet_length:
            self._min_shapelet_length = minl

        return y

    @staticmethod
    def _update_average_batch_time(avg_batch_time, batch_elapsed, alpha=0.5):
        """Update exponential moving average of batch time."""
        if avg_batch_time is None:
            return batch_elapsed
        return alpha * batch_elapsed + (1 - alpha) * avg_batch_time

    @staticmethod
    def _format_seconds(seconds):
        """Format seconds as h:mm:ss or m:ss."""
        if not np.isfinite(seconds):
            return "unknown"

        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _log_fit_progress(
        self,
        mode,
        n_shapelets_extracted,
        timed_shapelets_extracted,
        current_kept,
        timed_elapsed,
        total_elapsed,
        avg_batch_time,
        time_limit=None,
        n_shapelet_samples=None,
    ):
        """Log progress during shapelet extraction using post-warm-up estimates."""
        rate = (
            timed_shapelets_extracted / timed_elapsed
            if timed_elapsed > 0
            else float("nan")
        )

        if mode == "fixed":
            remaining = max(0, n_shapelet_samples - n_shapelets_extracted)
            eta_seconds = (
                remaining / rate if rate > 0 and np.isfinite(rate) else float("nan")
            )

            print(  # noqa: T201
                "[RST] "
                f"extracted={n_shapelets_extracted}/{n_shapelet_samples}, "
                f"kept={current_kept}, "
                f"avg_batch={avg_batch_time:.2f}s, "
                f"rate={rate:.1f}/s, "
                f"elapsed (h:m:s/m:s)={self._format_seconds(total_elapsed)}, "
                f"remaining={self._format_seconds(eta_seconds)}",
                flush=True,
            )
        else:
            remaining_time = max(0.0, time_limit - total_elapsed)
            projected_total = (
                n_shapelets_extracted + int(rate * remaining_time)
                if rate > 0 and np.isfinite(rate)
                else n_shapelets_extracted
            )

            print(  # noqa: T201
                "[RST] "
                f"extracted={n_shapelets_extracted}, "
                f"kept={current_kept}, "
                f"avg_batch={avg_batch_time:.2f}s, "
                f"elapsed={self._format_seconds(total_elapsed)}/"
                f"{self._format_seconds(time_limit)}, "
                f"remaining={self._format_seconds(remaining_time)}, "
                f"projected_total~{projected_total}",
                flush=True,
            )

    def _build_transform_inputs(self):
        """Pack the fitted shapelets into Numba-friendly arrays for transform.

        ``self.shapelets`` stays a list of tuples for external use. The ragged
        shapelet values and sorted indices are packed into single flat arrays
        with a CSR-style ``offsets`` array (shapelet ``n`` spans
        ``offsets[n]:offsets[n + 1]``). Flat arrays are plain NumPy, so the
        fitted estimator pickles cleanly (a Numba typed-list attribute does
        not) and each transform block still runs in one ``njit`` call.
        """
        self._transform_lengths = np.array(
            [s[_LENGTH] for s in self.shapelets], dtype=np.int32
        )
        self._transform_positions = np.array(
            [s[_POSITION] for s in self.shapelets], dtype=np.int32
        )
        self._transform_channels = np.array(
            [s[_CHANNEL] for s in self.shapelets], dtype=np.int32
        )
        # A shapelet's values and its sorted indices share the same length, so
        # one offsets array indexes both flat buffers.
        self._transform_offsets = np.zeros(len(self.shapelets) + 1, dtype=np.int64)
        self._transform_offsets[1:] = np.cumsum(self._transform_lengths)
        if len(self.shapelets) == 0:
            self._transform_values = np.empty(0, dtype=np.float64)
            self._transform_sorted_indices = np.empty(0, dtype=np.int32)
            return

        self._transform_values = np.concatenate(
            [np.ascontiguousarray(s[_VALUES], dtype=np.float64) for s in self.shapelets]
        )
        self._transform_sorted_indices = np.concatenate(
            [np.ascontiguousarray(si, dtype=np.int32) for si in self._sorted_indices]
        )

    @staticmethod
    def _pack_collection(X):
        """Pack a collection into a picklable flat buffer and case offsets."""
        n_cases = len(X)
        if isinstance(X, np.ndarray):
            case_size = X.shape[1] * X.shape[2]
            offsets = np.arange(n_cases + 1, dtype=np.int64) * case_size
            values = np.ascontiguousarray(X, dtype=np.float64).reshape(-1)
            return values, offsets

        offsets = np.zeros(n_cases + 1, dtype=np.int64)
        flat_cases = []
        for i, x in enumerate(X):
            flat_case = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
            flat_cases.append(flat_case)
            offsets[i + 1] = offsets[i] + flat_case.size

        return np.concatenate(flat_cases), offsets

    def _transform_block(self, X_values, X_offsets, start, stop):
        """Transform a contiguous block of cases using the Numba kernel."""
        return _transform_block_numba(
            X_values,
            X_offsets,
            self.n_channels_,
            self._transform_lengths,
            self._transform_positions,
            self._transform_channels,
            self._transform_values,
            self._transform_sorted_indices,
            self._transform_offsets,
            start,
            stop,
        )

    def _transform(self, X, y=None):
        """Transform ``X`` using the extracted shapelets.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Collection of time series.

        Returns
        -------
        output : np.ndarray of shape (n_cases, n_shapelets)
            The transformed data.
        """
        n_cases = len(X)
        n_shapelets = len(self.shapelets)
        output = np.empty((n_cases, n_shapelets))

        if n_cases == 0 or n_shapelets == 0:
            return output

        # Plain NumPy buffers remain picklable for process-based Joblib backends.
        X_values, X_offsets = self._pack_collection(X)

        if self._n_jobs == 1 or n_cases == 1:
            return self._transform_block(X_values, X_offsets, 0, n_cases)

        n_blocks = min(n_cases, self._n_jobs * 4)
        block_size = (n_cases + n_blocks - 1) // n_blocks

        blocks = [
            (start, min(start + block_size, n_cases))
            for start in range(0, n_cases, block_size)
        ]

        results = Parallel(
            n_jobs=self._n_jobs,
            backend=self.parallel_backend,
            prefer="threads",
        )(
            delayed(self._transform_block)(X_values, X_offsets, start, stop)
            for start, stop in blocks
        )

        for (start, stop), block in zip(blocks, results):
            output[start:stop] = block

        return output

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test-parameter set to return. If no special parameters are
            defined for ``parameter_set``, return the ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters used to create test instances of the class. Each dictionary
            constructs a valid test instance using ``MyClass(**params)`` or
            ``MyClass(**params[i])``.
        """
        if parameter_set == "results_comparison":
            return {"max_shapelets": 10, "n_shapelet_samples": 500}
        else:
            return {"max_shapelets": 5, "n_shapelet_samples": 50, "batch_size": 20}

    def _sample_shapelet(self, X_offsets, rng, candidate_index):
        """Draw the source case, length, position and channel of one shapelet.

        The source case is chosen by round-robin over the training set
        (``candidate_index % n_cases_``); the length, start position and channel
        are drawn from ``rng``. Sampling is separated from scoring so it can run
        serially on the main thread with a single reused generator, leaving the
        (parallelised) scoring free of any RNG state.

        Parameters
        ----------
        X_offsets : np.ndarray
            Start offsets of the cases in the packed training collection.
        rng : numpy.random.RandomState
            Generator for the random length, position and channel.
        candidate_index : int
            Running index of this candidate across the whole fit; only its value
            modulo ``n_cases_`` is used, so candidates cycle through the cases.

        Returns
        -------
        tuple of int
            ``(inst_idx, length, position, channel)``.
        """
        inst_idx = candidate_index % self.n_cases_
        case_size = X_offsets[inst_idx + 1] - X_offsets[inst_idx]
        case_length = case_size // self.n_channels_
        minl = min(case_length, self._max_shapelet_length)
        length = (
            rng.randint(0, minl - self._min_shapelet_length) + self._min_shapelet_length
            if minl - self._min_shapelet_length > 0
            else minl
        )
        position = (
            rng.randint(0, case_length - length) if case_length - length > 0 else 0
        )
        channel = rng.randint(0, self.n_channels_)
        return inst_idx, length, position, channel

    def _score_shapelet(
        self,
        X_values,
        X_offsets,
        y,
        cache_distance_vector,
        inst_idx,
        length,
        position,
        channel,
    ):
        """Score one sampled shapelet against every training case.

        The subsequence is z-normalised, and its indices are sorted by descending
        magnitude for early abandonment. It is then scored by the information gain
        of its distances to every training case. This happens inside a single Numba
        call, so each candidate crosses into Numba once.

        Parameters
        ----------
        X_values, X_offsets : np.ndarray
            Flat training collection and the start offset of each case.
        y : np.ndarray
            Label-encoded class index of each training case.
        cache_distance_vector : bool
            Whether to retain the candidate's distances for ``fit_transform``.
        inst_idx, length, position, channel : int
            The sampled shapelet parameters from ``_sample_shapelet``.

        Returns
        -------
        record : list
            The working shapelet record with the ``_QUALITY.._DIST`` fields;
            ``_DIST`` is a ``-1`` placeholder set later only when distance
            vectors are cached.
        distance_vector : np.ndarray or None
            Distance from this shapelet to every training case when caching, otherwise
            ``None``.
        """
        cls_idx = int(y[inst_idx])
        distance_vector = np.empty(len(y)) if cache_distance_vector else None
        quality = self._find_shapelet_quality(
            X_values,
            X_offsets,
            self.n_channels_,
            y,
            position,
            length,
            channel,
            inst_idx,
            self._class_counts[cls_idx],
            self.n_cases_ - self._class_counts[cls_idx],
            self._klog,
            distance_vector,
        )

        return (
            [np.round(quality, 8), length, position, channel, inst_idx, cls_idx, -1],
            distance_vector,
        )

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _find_shapelet_quality(
        X_values,
        X_offsets,
        n_channels,
        y,
        position,
        length,
        dim,
        inst_idx,
        this_cls_count,
        other_cls_count,
        klog,
        distance_vector,
    ):
        # Extract, z-normalise and order the shapelet inside numba so the whole
        # per-candidate pipeline is a single Python -> numba crossing. Indices
        # are ordered by descending magnitude (stable, matching a reverse Python
        # sort) so _online_shapelet_distance abandons early on the key points.
        source_start = X_offsets[inst_idx]
        source_length = (X_offsets[inst_idx + 1] - source_start) // n_channels
        channel_start = source_start + dim * source_length
        shapelet = z_normalise_series(
            X_values[channel_start + position : channel_start + position + length]
        )
        sorted_indicies = np.argsort(-np.abs(shapelet), kind="mergesort").astype(
            np.int32
        )

        orderline = []
        this_cls_traversed = 0
        other_cls_traversed = 0

        for i in range(len(y)):
            if i != inst_idx:
                case_start = X_offsets[i]
                case_length = (X_offsets[i + 1] - case_start) // n_channels
                channel_start = case_start + dim * case_length
                distance = _online_shapelet_distance(
                    X_values[channel_start : channel_start + case_length],
                    shapelet,
                    sorted_indicies,
                    position,
                    length,
                )
            else:
                distance = 0

            if y[i] == y[inst_idx]:
                cls = 1
                this_cls_traversed += 1
            else:
                cls = -1
                other_cls_traversed += 1

            orderline.append((distance, cls))
            if distance_vector is not None:
                distance_vector[i] = distance

        orderline.sort()
        return _calc_binary_ig(orderline, this_cls_count, other_cls_count, klog)

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _merge_shapelets(
        shapelet_heap, candidate_shapelets, max_shapelets_per_class, cls_idx
    ):
        for shapelet in candidate_shapelets:
            if shapelet[_CLASS] == cls_idx and shapelet[_QUALITY] > 0:
                if (
                    len(shapelet_heap) == max_shapelets_per_class
                    and shapelet[_QUALITY] < shapelet_heap[0][_QUALITY]
                ):
                    continue

                heapq.heappush(shapelet_heap, shapelet)

                if len(shapelet_heap) > max_shapelets_per_class:
                    heapq.heappop(shapelet_heap)

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _remove_self_similar_shapelets(shapelet_heap):
        to_keep = [True] * len(shapelet_heap)

        for i in range(len(shapelet_heap)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelet_heap)):
                if to_keep[n] and _is_self_similar(shapelet_heap[i], shapelet_heap[n]):
                    if (
                        shapelet_heap[i][_QUALITY],
                        -shapelet_heap[i][_LENGTH],
                    ) >= (
                        shapelet_heap[n][_QUALITY],
                        -shapelet_heap[n][_LENGTH],
                    ):
                        to_keep[n] = False
                    else:
                        to_keep[i] = False
                        break

        return to_keep

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _remove_identical_shapelets(shapelets):
        to_keep = [True] * len(shapelets)

        for i in range(len(shapelets)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelets)):
                if (
                    to_keep[n]
                    and shapelets[i][_LENGTH] == shapelets[n][_LENGTH]
                    and np.array_equal(shapelets[i][_VALUES], shapelets[n][_VALUES])
                ):
                    to_keep[n] = False

        return to_keep


@njit(fastmath=True, cache=True, nogil=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    if len(series) < len(shapelet):
        t = series
        series = shapelet
        shapelet = t
        length = len(shapelet)
        sorted_indicies = np.arange(length, dtype=np.int32)
    if position + length > len(series):
        position = int((len(series) - length) / 2)

    sum = 0.0
    sum2 = 0.0
    shapelet_sq_sum = 0.0
    for i in range(length):
        val = series[position + i]
        sum += val
        sum2 += val * val
        shapelet_sq_sum += shapelet[i] * shapelet[i]

    mean = sum / length
    std = math.sqrt((sum2 - mean * mean * length) / length)

    if std > AEON_NUMBA_STD_THRESHOLD:
        inv_std = 1.0 / std
        best_dist = 0.0
        for i in range(length):
            temp = shapelet[i] - (series[position + i] - mean) * inv_std
            best_dist += temp * temp
    else:
        # flat subsequence normalises to zeros, so the distance is ||shapelet||^2
        best_dist = shapelet_sq_sum

    i = 1
    traverse_left = True
    traverse_right = True
    left_sum = sum
    right_sum = sum
    left_sum2 = sum2
    right_sum2 = sum2

    while traverse_left or traverse_right:
        pos = position - i
        traverse_left = pos >= 0
        if traverse_left:
            start = series[pos]
            end = series[pos + length]

            left_sum += start - end
            left_sum2 += start * start - end * end

            mean = left_sum / length
            std = math.sqrt((left_sum2 - mean * mean * length) / length)

            if std > AEON_NUMBA_STD_THRESHOLD:
                inv_std = 1.0 / std
                dist = 0.0
                for j in range(length):
                    idx = sorted_indicies[j]
                    temp = shapelet[idx] - (series[pos + idx] - mean) * inv_std
                    dist += temp * temp

                    if dist > best_dist:
                        break
            else:
                dist = shapelet_sq_sum

            if dist < best_dist:
                best_dist = dist

        pos = position + i
        traverse_right = pos <= len(series) - length
        if traverse_right:
            start = series[pos - 1]
            end = series[pos - 1 + length]

            right_sum += end - start
            right_sum2 += end * end - start * start

            mean = right_sum / length
            std = math.sqrt((right_sum2 - mean * mean * length) / length)

            if std > AEON_NUMBA_STD_THRESHOLD:
                inv_std = 1.0 / std
                dist = 0.0
                for j in range(length):
                    idx = sorted_indicies[j]
                    temp = shapelet[idx] - (series[pos + idx] - mean) * inv_std
                    dist += temp * temp

                    if dist > best_dist:
                        break
            else:
                dist = shapelet_sq_sum

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True, nogil=True)
def _transform_block_numba(
    X_values,
    X_offsets,
    n_channels,
    lengths,
    positions,
    channels,
    values,
    sorted_indices,
    offsets,
    start,
    stop,
):
    """Distance from every shapelet to each packed case in ``start:stop``.

    Cases are flattened channel-major in ``X_values`` and indexed by
    ``X_offsets``. ``values`` and ``sorted_indices`` are flat shapelet buffers;
    shapelet ``n`` occupies ``offsets[n]:offsets[n + 1]`` in both.
    """
    n_shapelets = len(lengths)
    out = np.empty((stop - start, n_shapelets))
    for i in range(stop - start):
        case_idx = start + i
        case_start = X_offsets[case_idx]
        case_size = X_offsets[case_idx + 1] - case_start
        case_length = case_size // n_channels
        for n in range(n_shapelets):
            lo = offsets[n]
            hi = offsets[n + 1]
            channel_start = case_start + channels[n] * case_length
            out[i, n] = _online_shapelet_distance(
                X_values[channel_start : channel_start + case_length],
                values[lo:hi],
                sorted_indices[lo:hi],
                positions[n],
                lengths[n],
            )
    return out


@njit(fastmath=True, cache=True, nogil=True)
def _calc_binary_ig(orderline, c1, c2, klog):
    # Binary entropy H(a, b) = (klog[a+b] - klog[a] - klog[b]) / (a+b), where
    # klog[k] = k*log2(k). The left partition at a split has (split+1) elements,
    # so both partition sizes and class counts are integer indices into klog and
    # no logarithm is evaluated in the loop. Maximising the information gain
    #     ig = initial_ent - (left + right) / n
    # is equivalent to minimising (left + right), so we track that minimum.
    n = c1 + c2
    initial_ent = (klog[n] - klog[c1] - klog[c2]) / n

    min_lr = np.inf
    c1_count = 0
    c2_count = 0
    for split in range(len(orderline)):
        if orderline[split][1] > 0:  # +1 this class, -1 other class
            c1_count += 1
        else:
            c2_count += 1

        n_left = split + 1
        n_right = n - n_left
        lr = (
            klog[n_left]
            + klog[n_right]
            - klog[c1_count]
            - klog[c2_count]
            - klog[c1 - c1_count]
            - klog[c2 - c2_count]
        )
        if lr < min_lr:
            min_lr = lr

    ig = initial_ent - min_lr / n
    return ig if ig > 0 else 0.0


@njit(fastmath=True, cache=True, nogil=True)
def _is_self_similar(s1, s2):
    # not self similar if from different series or channel
    if s1[_CASE] == s2[_CASE] and s1[_CHANNEL] == s2[_CHANNEL]:
        if s2[_POSITION] <= s1[_POSITION] <= s2[_POSITION] + s2[_LENGTH]:
            return True
        if s1[_POSITION] <= s2[_POSITION] <= s1[_POSITION] + s1[_LENGTH]:
            return True
    return False

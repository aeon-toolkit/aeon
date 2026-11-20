"""Torsk anomaly detector."""

__maintainer__ = ["lazizbekravshanov"]
__all__ = ["Torsk"]

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.special import erf
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector


class Torsk(BaseSeriesAnomalyDetector):
    """Torsk anomaly detector.

    Torsk [1]_ is a prediction-based anomaly detector built on an echo state
    network (ESN). The series is cut into consecutive frames of ``window_size``
    time points. A fixed random sparse reservoir is driven by the frames, and at
    every sliding position a linear readout is fitted by least squares to
    predict the next frame. The readout then runs freely for
    ``prediction_window_size`` frames, and the mean absolute error between the
    free-running prediction and the observed frames gives one error per
    position. Errors are converted to anomaly scores by comparing the mean of a
    small leading window of errors against the distribution of a large trailing
    window through a Gaussian tail probability, and scores are mapped back to
    time points over the span each prediction covers.

    Only the readout is ever trained. The reservoir and the input map are drawn
    once from ``random_state`` and remain fixed, which makes the detector
    strongly dependent on the seed: across six seeds on the same series, ROC AUC
    ranged from 0.55 to 0.89 in our experiments. Set ``random_state`` for any
    reproducible use.

    Parameters
    ----------
    window_size : int, default=10
        Length of the tumbling window that turns the series into frames. Each
        frame of shape ``(window_size, n_channels)`` is flattened into a single
        input vector.
    reservoir_size : int, default=100
        Dimension of the reservoir state.
    input_scale : float, default=0.125
        Gain applied to the random input map, including its bias. The default is
        tuned for input scaled to the range minus one to one, which this
        implementation applies internally.
    spectral_radius : float, default=2.0
        Largest eigenvalue modulus of the reservoir matrix after rescaling.
    density : float, default=0.01
        Fraction controlling reservoir connectivity. Every reservoir row gets
        ``max(1, int(reservoir_size * density))`` nonzero entries.
    train_window_size : int, default=50
        Number of frames driven through the reservoir at each sliding position
        before the readout fit.
    prediction_window_size : int, default=20
        Number of frames predicted autoregressively at each position. Also sets
        the span of time points each position's score is assigned to.
    transient_window_size : int, default=10
        Number of leading reservoir states discarded before the readout fit.
    normality_small_window : int, default=10
        Number of leading errors whose mean is tested against the baseline.
    normality_large_window : int, default=100
        Number of trailing errors forming the baseline distribution.
    rcond : float, default=1e-4
        Relative singular value cutoff for the truncated SVD readout solve.
        Singular values below ``rcond`` times the largest are discarded. The
        readout system is underdetermined at the default window sizes, so this
        truncation is load bearing rather than cosmetic.
    random_state : int, np.random.RandomState instance or None, default=None
        Seed for the reservoir and input map draw.

    Attributes
    ----------
    reservoir_ : scipy.sparse.csr_matrix of shape (reservoir_size, reservoir_size)
        The fixed random reservoir matrix, rescaled to ``spectral_radius``.
    input_weights_ : np.ndarray of shape (reservoir_size, window_size * n_channels)
        The fixed random input map.
    input_bias_ : np.ndarray of shape (reservoir_size,)
        The fixed random input bias.

    Notes
    -----
    This implementation follows the TimeEval configuration of Torsk [2]_ rather
    than the full method of the paper. We make the following changes:

    - The paper composes up to eleven spatially aware input maps (resampling,
      convolutions, DCT, gradients) for two dimensional fields. We use the
      single dense random input map that TimeEval uses for one dimensional
      series.
    - Equation 3 of the paper writes the normality score with the difference
      reversed, which would flag drops in prediction error instead of spikes.
      We follow both reference implementations, which use the opposite sign.
    - The reference implementations offer a Tikhonov solver but implement it
      incorrectly, adding the scalar regularisation constant to every element
      of the Gram matrix rather than to its diagonal. We solve the readout by
      truncated SVD of the design matrix, matching the reference's default
      ``pinv_svd`` path. Solving the normal equations instead squares the
      condition number of an already ill conditioned system, which measurably
      degrades both accuracy and runtime for series with many channels.
    - The paper whitens readout targets with the inverse metric of the IMED at
      fit time only. We omit IMED entirely.
    - Like TimeEval, the series is min max scaled to the range minus one to one
      before the reservoir sees it, using the global minimum and maximum of the
      series being scored. Note this uses statistics of the full series,
      including any anomaly.
    - Scores for the first ``normality_large_window // 10`` positions are set to
      zero, since no meaningful baseline exists there. Time points never covered
      by a prediction, including the first ``train_window_size * window_size``
      points, also score zero.

    References
    ----------
    .. [1] Heim, N., & Avery, J. E. (2019). Adaptive Anomaly Detection in
           Chaotic Time Series with a Spatially Aware Echo State Network.
           arXiv:1909.01709.
    .. [2] Schmidl, S., Wenig, P., & Papenbrock, T. (2022). Anomaly Detection in
           Time Series: A Comprehensive Evaluation. PVLDB, 15(9), 1779-1797.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.deep_learning import Torsk
    >>> rng = np.random.RandomState(0)
    >>> series = np.sin(np.linspace(0, 20 * np.pi, 400)) + rng.normal(0, 0.05, 400)
    >>> series[300:320] += 3.0
    >>> detector = Torsk(
    ...     window_size=5, train_window_size=20, prediction_window_size=5,
    ...     transient_window_size=4, normality_small_window=4,
    ...     normality_large_window=20, random_state=1,
    ... )
    >>> scores = detector.fit_predict(series)
    >>> scores.shape
    (400,)
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
    }

    def __init__(
        self,
        window_size: int = 10,
        reservoir_size: int = 100,
        input_scale: float = 0.125,
        spectral_radius: float = 2.0,
        density: float = 0.01,
        train_window_size: int = 50,
        prediction_window_size: int = 20,
        transient_window_size: int = 10,
        normality_small_window: int = 10,
        normality_large_window: int = 100,
        rcond: float = 1e-4,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.window_size = window_size
        self.reservoir_size = reservoir_size
        self.input_scale = input_scale
        self.spectral_radius = spectral_radius
        self.density = density
        self.train_window_size = train_window_size
        self.prediction_window_size = prediction_window_size
        self.transient_window_size = transient_window_size
        self.normality_small_window = normality_small_window
        self.normality_large_window = normality_large_window
        self.rcond = rcond
        self.random_state = random_state

        super().__init__(axis=0)

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Torsk":
        self._check_params()
        rng = check_random_state(self.random_state)
        frame_dim = self.window_size * X.shape[1]

        self.reservoir_ = self._make_reservoir(rng)
        self.input_weights_ = rng.uniform(
            -1.0, 1.0, size=(self.reservoir_size, frame_dim)
        )
        self.input_bias_ = rng.uniform(-1.0, 1.0, size=self.reservoir_size)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        n_timepoints, n_channels = X.shape
        frame_dim = self.window_size * n_channels
        if self.input_weights_.shape[1] != frame_dim:
            raise ValueError(
                f"X has {n_channels} channels but the detector was fitted for "
                f"{self.input_weights_.shape[1] // self.window_size} channels"
            )

        X = self._scale(X)
        n_frames = n_timepoints // self.window_size
        span = self.train_window_size + self.prediction_window_size
        if n_frames <= span:
            raise ValueError(
                f"series produces {n_frames} frames of window_size="
                f"{self.window_size}, but train_window_size + "
                f"prediction_window_size = {span} requires at least {span + 1}"
            )

        frames = X[: n_frames * self.window_size].reshape(n_frames, frame_dim)
        n_positions = n_frames - span
        errors = np.empty(n_positions)
        for i in range(n_positions):
            errors[i] = self._position_error(frames, i)

        window_scores = self._normality_scores(errors)
        return self._scores_to_timepoints(window_scores, n_timepoints)

    def _check_params(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.reservoir_size < 1:
            raise ValueError("reservoir_size must be at least 1")
        if not 0.0 < self.density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        if self.spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be positive")
        if self.train_window_size < 2:
            raise ValueError("train_window_size must be at least 2")
        if self.prediction_window_size < 1:
            raise ValueError("prediction_window_size must be at least 1")
        if not 0 <= self.transient_window_size <= self.train_window_size - 2:
            raise ValueError(
                "transient_window_size must be in [0, train_window_size - 2] so "
                "that at least one row remains for the readout fit"
            )
        if self.normality_small_window < 1 or self.normality_large_window < 1:
            raise ValueError("normality window sizes must be at least 1")
        if not 0.0 <= self.rcond < 1.0:
            raise ValueError("rcond must be in [0, 1)")

    def _make_reservoir(self, rng) -> csr_matrix:
        h = self.reservoir_size
        per_row = max(1, int(h * self.density))
        rows = np.repeat(np.arange(h), per_row)
        cols = rng.randint(0, h, size=h * per_row)
        vals = rng.uniform(-1.0, 1.0, size=h * per_row)
        reservoir = csr_matrix((vals, (rows, cols)), shape=(h, h))

        if h > 3:
            try:
                # v0 must be seeded: without it ARPACK draws its starting vector
                # from the global RNG and the rescaled reservoir is not
                # reproducible even with a fixed random_state.
                ev = eigs(
                    reservoir,
                    k=1,
                    which="LM",
                    return_eigenvectors=False,
                    maxiter=5000,
                    v0=rng.uniform(-1.0, 1.0, size=h),
                )
                radius = abs(ev[0])
            except Exception:
                radius = abs(np.linalg.eigvals(reservoir.toarray())).max()
        else:
            radius = abs(np.linalg.eigvals(reservoir.toarray())).max()

        if radius > 1e-12:
            reservoir = reservoir * (self.spectral_radius / radius)
        return reservoir.tocsr()

    @staticmethod
    def _scale(X: np.ndarray) -> np.ndarray:
        lo, hi = X.min(), X.max()
        if hi - lo > 1e-12:
            return ((X - lo) / (hi - lo)) * 2.0 - 1.0
        return X - lo

    def _drive(self, frames: np.ndarray) -> np.ndarray:
        states = np.empty((frames.shape[0], self.reservoir_size))
        x = np.zeros(self.reservoir_size)
        driven = self.input_scale * (frames @ self.input_weights_.T + self.input_bias_)
        for t in range(frames.shape[0]):
            x = np.tanh(driven[t] + self.reservoir_.dot(x))
            states[t] = x
        return states

    def _position_error(self, frames: np.ndarray, i: int) -> float:
        train = frames[i : i + self.train_window_size]
        states = self._drive(train)

        tr = self.transient_window_size
        design = np.hstack(
            [np.ones((states[tr:-1].shape[0], 1)), train[tr:-1], states[tr:-1]]
        )
        targets = train[tr + 1 :]
        # Truncated SVD of the design matrix. Solving the normal equations
        # instead squares the condition number, which visibly degrades accuracy
        # and runtime once window_size * n_channels grows large.
        u_svd, s_svd, vt_svd = np.linalg.svd(design, full_matrices=False)
        keep = s_svd > self.rcond * s_svd[0]
        readout = (vt_svd[keep].T * (1.0 / s_svd[keep])) @ (u_svd[:, keep].T @ targets)

        u, x = train[-1].copy(), states[-1].copy()
        pred_start = i + self.train_window_size
        truth = frames[pred_start : pred_start + self.prediction_window_size]
        error = 0.0
        for k in range(self.prediction_window_size):
            x = np.tanh(
                self.input_scale * (self.input_weights_ @ u + self.input_bias_)
                + self.reservoir_.dot(x)
            )
            u = np.concatenate(([1.0], u, x)) @ readout
            error += np.abs(u - truth[k]).mean()
        return error / self.prediction_window_size

    def _normality_scores(self, errors: np.ndarray) -> np.ndarray:
        n = errors.shape[0]
        scores = np.zeros(n)
        for i in range(1, n):
            large = errors[max(0, i - self.normality_large_window) : i]
            small = errors[i : i + self.normality_small_window]
            if large.size < 2 or small.size == 0:
                continue
            sd = large.std()
            if sd < 1e-10:
                continue
            scores[i] = erf(max(0.0, small.mean() - large.mean()) / (np.sqrt(2.0) * sd))
        scores[: self.normality_large_window // 10] = 0.0
        return scores

    def _scores_to_timepoints(
        self, window_scores: np.ndarray, n_timepoints: int
    ) -> np.ndarray:
        """Assign position scores to the time points their predictions cover.

        Position ``i`` trains on frames ``[i, i + train_window_size)`` and
        predicts frames ``[i + train_window_size, i + train_window_size +
        prediction_window_size)``, so its score belongs to the time points of
        the predicted frames, averaged where spans overlap.
        """
        acc = np.zeros(n_timepoints)
        counts = np.zeros(n_timepoints)
        for i, value in enumerate(window_scores):
            start = (i + self.train_window_size) * self.window_size
            if start >= n_timepoints:
                break
            end = min(
                start + self.prediction_window_size * self.window_size, n_timepoints
            )
            acc[start:end] += value
            counts[start:end] += 1
        covered = counts > 0
        acc[covered] /= counts[covered]
        return acc

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Sized for the 20 time point series used by the general test suite:
        window_size 2 gives 10 frames, which exceeds the required
        train_window_size + prediction_window_size + 1 = 7.
        """
        return {
            "window_size": 2,
            "reservoir_size": 8,
            "train_window_size": 4,
            "prediction_window_size": 2,
            "transient_window_size": 1,
            "normality_small_window": 2,
            "normality_large_window": 4,
            "random_state": 0,
        }

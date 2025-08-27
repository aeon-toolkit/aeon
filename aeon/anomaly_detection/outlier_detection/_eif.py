"""EIF(Extended Isolation Forest) anomaly detector."""

__maintainer__ = ["Akhil-Jasson"]
__all__ = ["EIF"]

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.windowing import reverse_windowing, sliding_windows


class EIF(BaseAnomalyDetector):
    """Extended Isolation Forest (EIF) for anomaly detection.

    Implementation of the Extended Isolation Forest algorithm that uses
    random hyperplanes for splitting instead of axis-parallel splits.
    This allows for better handling of high-dimensional data and complex distributions.

    The implementation supports both unsupervised and semi-supervised learning:
    - Unsupervised: No labels provided (y=None)
    - Semi-supervised: Labels provided (y=0 for normal data)

    Parameters
    ----------
    n_estimators : int, default=100
        The number of isolation trees in the ensemble.
    sample_size : float or int, default='auto'
        The number of samples to draw from X to train each base estimator.
        - If float, should be between 0.0 and 1.0 and represents the proportion
          of the dataset to draw for training each base estimator.
        - If int, represents the absolute number of samples.
        - If 'auto', sample_size is set to min(256, n_samples).
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples. Only used in unsupervised mode.
    extension_level : int, default=None
        The extension level of the isolation forest. If None, an appropriate value
        will be determined based on the dimensionality of the data.
    random_state : int, RandomState instance, default=None
        Controls the pseudo-randomization process.
    window_size : int, default=1
        Size of the sliding window. If 1, the original point-wise behavior is used.
    stride : int, default=1
        Stride of the sliding window.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "requires_y": False,  # Can work with or without y
        "X_inner_type": "np.ndarray",
    }

    def __init__(
        self,
        n_estimators=100,
        sample_size="auto",
        contamination=0.1,
        extension_level=None,
        random_state=None,
        window_size=1,
        stride=1,
    ):
        super().__init__()

        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.contamination = contamination
        self.extension_level = extension_level
        self.random_state = random_state
        self.window_size = window_size
        self.stride = stride

    def _fit(self, X, y=None):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_timepoints,) or (n_timepoints, n_channels)
        y : np.ndarray, optional
            Labels for semi-supervised learning. 0 for normal data is supported.
            If None, unsupervised learning is used.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if y is not None:
            y = np.asarray(y)
            if not np.all(y == 0):
                raise ValueError(
                    "Semi-supervised mode only supports training on normal data (y==0)."
                )

        # Apply sliding window if window_size > 1
        if self.window_size > 1:
            X_windowed, _ = sliding_windows(
                X, window_size=self.window_size, stride=self.stride
            )
        else:
            X_windowed = X

        # Set random state
        rng = np.random.RandomState(self.random_state)

        # Determine sample size
        n_samples = X_windowed.shape[0]
        if isinstance(self.sample_size, str):
            if self.sample_size != "auto":
                raise ValueError("sample_size must be 'auto', int or float")
            self.sample_size_ = min(256, n_samples)
        elif isinstance(self.sample_size, float):
            self.sample_size_ = int(self.sample_size * n_samples)
        else:
            self.sample_size_ = min(self.sample_size, n_samples)

        # Store extension level as fitted parameter
        self.extension_level_ = (
            min(X_windowed.shape[1], 8)
            if self.extension_level is None
            else self.extension_level
        )

        # Build the forest
        self.forest_ = []
        for _ in range(self.n_estimators):
            # Sample data
            if n_samples > self.sample_size_:
                sample_indices = rng.choice(
                    n_samples, size=self.sample_size_, replace=False
                )
                X_sample = X_windowed[sample_indices]
            else:
                X_sample = X_windowed

            # Build tree
            tree = self._build_tree(X_sample, rng)
            self.forest_.append(tree)

        return self
    
    def _set_threshold_if_needed(self, scores):
        """Set threshold based on contamination if not already set.
        
        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores to use for threshold calculation
        """
        if not hasattr(self, 'threshold_'):
            self.threshold_ = float(np.percentile(scores, 100 * (1 - self.contamination)))

    def _build_tree(self, X, rng, depth=0):
        """Build an isolation tree recursively.

        Parameters
        ----------
        X : np.ndarray
           The data to build the tree on
        rng : RandomState
           Random number generator
        max_depth : int, optional
           Maximum depth of the tree. If None, it will be set to
           log2(len(X))

        Returns
        -------
        dict
           The tree structure
        """
        n_samples = X.shape[0]

        if n_samples <= 1 or depth >= int(np.ceil(np.log2(self.sample_size_))):
            return {"size": n_samples}

        # Generate random hyperplane
        n_features = X.shape[1]
        normal = rng.normal(size=n_features)
        normal /= np.linalg.norm(normal)
        intercept = rng.uniform(X.min(), X.max())

        # Split data
        projections = X @ normal
        left_mask = projections < intercept
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return {"size": n_samples}

        # Build subtrees
        left_tree = self._build_tree(X[left_mask], rng, depth + 1)
        right_tree = self._build_tree(X[right_mask], rng, depth + 1)

        return {
            "normal": normal,
            "intercept": intercept,
            "left": left_tree,
            "right": right_tree,
        }

    def _path_length(self, X, tree, current_length=0):
        """Calculate the path length for a single point.

        Parameters
        ----------
        x : np.ndarray
            The point to calculate path length for
        tree : dict
            The tree structure
        current_length : int
            Current path length

        Returns
        -------
        float
            The path length
        """
        if "size" in tree:
            return current_length + self._c(tree["size"])

        projections = X @ tree["normal"]
        left_mask = projections < tree["intercept"]

        lengths = np.zeros(len(X))
        if np.any(left_mask):
            lengths[left_mask] = self._path_length(
                X[left_mask], tree["left"], current_length + 1
            )
        if np.any(~left_mask):
            lengths[~left_mask] = self._path_length(
                X[~left_mask], tree["right"], current_length + 1
            )
        return lengths

    def _c(self, n):
        """Average path length of unsuccessful search in BST.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        float
            Average path length
        """
        if n <= 1:
            return 0
        elif n == 2:
            return 1
        
        # Formula: 2 * (H(n-1)) - 2*(n-1)/n
        # where H(n-1) is the harmonic number ≈ ln(n-1) + γ
        # where γ is the Euler-Mascheroni constant
        euler_mascheroni_constant = 0.5772156649015329
        harmonic_number = np.log(n - 1) + euler_mascheroni_constant
    
        return 2 * harmonic_number - 2 * (n - 1) / n

    def _predict(self, X) -> np.ndarray:
        """Predict anomaly scores for X.

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_timepoints,) or (n_timepoints, n_channels)

        Returns
        -------
        np.ndarray
            The anomaly scores of the input samples.
            The higher, the more abnormal.
        """
        # Store original shape for reconstruction
        original_n_timepoints = X.shape[0]

        # Apply sliding window if window_size > 1
        if self.window_size > 1:
            X_windowed, _ = sliding_windows(
                X, window_size=self.window_size, stride=self.stride
            )
            X_for_scoring = X_windowed
        else:
            X_for_scoring = X

        # Calculate anomaly scores
        scores = np.zeros(len(X_for_scoring))
        for tree in self.forest_:
            scores += self._path_length(X_for_scoring, tree)
        scores = scores / len(self.forest_)

        # Convert to anomaly scores
        scores = 2 ** (-scores / self._c(self.sample_size_))

        # If windowing was applied, we need to map scores back to original samples
        if self.window_size > 1:
            scores = reverse_windowing(
                scores,
                window_size=self.window_size,
                stride=self.stride,
                reduction=np.mean 
            )

            if len(scores) > original_n_timepoints:
                scores = scores[:original_n_timepoints]
            elif len(scores) < original_n_timepoints:
                # Pad with edge values if needed
                pad_length = original_n_timepoints - len(scores)
                scores = np.pad(scores, (0, pad_length), mode='edge')

        return scores

    def predict_labels(self, X) -> np.ndarray:
        """Predict if points are anomalies or not.

        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            The time series to predict for.

        Returns
        -------
        np.ndarray
            Returns 1 for anomalies/outliers and 0 for inliers.
        """
        self._check_is_fitted()
        X = self._preprocess_series(X, False)

        # Get anomaly scores
        scores = self._predict(X)

        # Set threshold based on current data if not already set
        self._set_threshold_if_needed(scores)

        # Use threshold to determine outliers
        predictions = np.zeros(len(X), dtype=int)
        predictions[scores > self.threshold_] = 1

        return predictions

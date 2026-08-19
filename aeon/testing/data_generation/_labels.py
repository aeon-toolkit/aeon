"""Functions for generating learning task labels."""

import numpy as np


def make_anomaly_detection_labels(
    n_labels: int,
    anomaly_rate: float = 0.01,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate anomaly detection labels.

    Creates a boolean array of length ``n_labels`` with ``ceil(anomaly_rate*n_labels)``
    anomalies at random positions.

    Parameters
    ----------
    n_labels : int
        The number of labels to generate.
    anomaly_rate : float, default=0.01
        The proportion of anomalies in the generated labels, between 0 and 1.
        For example, if `anomaly_rate` is 0.01, then approximately 1% of the labels
        will be anomalies (1s).
    random_state : int or None, default=None
        Seed for random number generation.

    Returns
    -------
    y : np.ndarray
        An array of labels indicating anomalies.
    """
    rng = np.random.RandomState(random_state)
    n_anomalies = np.ceil(n_labels * anomaly_rate)
    anomaly_indices = rng.choice(n_labels, size=int(n_anomalies), replace=False)
    y = np.zeros(n_labels, dtype=bool)
    y[anomaly_indices] = 1
    return y

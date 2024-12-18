"""Test For RCluster."""

import numpy as np

from aeon.clustering.feature_based._r_cluster import RCluster
from aeon.datasets import load_gunpoint


def test_r_cluster():
    """Test implementation of RCluster."""
    X_train, y_train = load_gunpoint(split="train")

    num_points = 20

    X_train = X_train[:num_points]

    rcluster = RCluster(random_state=1, n_clusters=2)
    rcluster.fit(X_train)
    train_result = rcluster.predict(X_train)
    labs = rcluster.labels_
    assert np.array_equal(labs, train_result)

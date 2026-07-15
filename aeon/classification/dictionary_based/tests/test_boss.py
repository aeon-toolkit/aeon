"""BOSS test code."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from aeon.classification.dictionary_based import BOSSEnsemble
from aeon.classification.dictionary_based._boss import boss_distance, pairwise_distances
from aeon.testing.data_generation import (
    make_example_3d_numpy,
)


def test_boss_min_window():
    """Test BOSS throws error when min window too big."""
    boss = BOSSEnsemble(min_window=20)
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=10)
    with pytest.raises(ValueError, match="Error in BOSSEnsemble, min_window"):
        boss._fit(X, y)


def test_boss_pairwise_distances_matches_row_wise_distance():
    """Test vectorised BOSS distances match the row-wise sparse calculation."""
    X = csr_matrix(
        (
            [2, 0, 1, 3, 1, 1, 2],
            ([0, 0, 0, 1, 1, 2, 2], [0, 1, 2, 1, 4, 0, 3]),
        ),
        shape=(3, 5),
    )
    Y = csr_matrix(
        [
            [1, 0, 4, 0, 0],
            [0, 1, 0, 5, 0],
            [3, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
        ]
    )

    expected = np.vstack([boss_distance(X, Y, i) for i in range(X.shape[0])])
    actual = pairwise_distances(X, Y, use_boss_distance=True)

    assert_allclose(actual, expected)

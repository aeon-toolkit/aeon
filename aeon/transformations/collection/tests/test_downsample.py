"""Tests for DownsampleTransformer."""

import numpy as np
import pytest

from aeon.transformations.collection import DownsampleTransformer

# list of 2D numpy arrays, unequal lengths
X = [
    np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]),
    np.array([[10, 20, 30, 40, 50, 60, 70, 80], [100, 90, 80, 70, 60, 50, 40, 30]]),
]

test_cases = [
    (
        10,  # source_sfreq
        5,  # target_sfreq
        0.5,  # proportion
        X,  # X
        [
            np.array([[1, 3, 5, 7, 9], [10, 8, 6, 4, 2]]),
            np.array([[10, 30, 50, 70], [100, 80, 60, 40]]),
        ],  # X_expected
    ),
    (
        10,
        3,
        0.7,
        X,
        [
            np.array([[1, 4, 7, 10], [10, 7, 4, 1]]),
            np.array([[10, 40, 70], [100, 70, 40]]),
        ],
    ),
    (
        10,
        2,
        0.8,
        X,
        [
            np.array([[1, 6], [10, 5]]),
            np.array([[10, 60], [100, 50]]),
        ],
    ),
]


@pytest.mark.parametrize("downsample_by", ["frequency", "proportion"])
@pytest.mark.parametrize(
    "source_sfreq, target_sfreq, proportion, X, X_expected", test_cases
)
def test_downsample_series(
    downsample_by, source_sfreq, target_sfreq, proportion, X, X_expected
):
    """Test the downsampling of a time series."""
    if downsample_by == "frequency":
        transformer = DownsampleTransformer(
            downsample_by="frequency",
            source_sfreq=source_sfreq,
            target_sfreq=target_sfreq,
        )
    elif downsample_by == "proportion":
        transformer = DownsampleTransformer(
            downsample_by="proportion", proportion=proportion
        )

    X_downsampled = transformer.fit_transform(X)

    assert isinstance(X_downsampled, list)
    for idx in range(len(X_downsampled)):
        np.testing.assert_array_equal(X_expected[idx], X_downsampled[idx])


def test_value_errors():
    """Test DowmsampleTransformer ValueErrors."""
    # default initialization
    rng = np.random.default_rng(seed=0)
    X = rng.random((32, 1000))
    transformer = DownsampleTransformer()
    transformer.fit_transform(X)

    # invalid initializations
    with pytest.raises(
        ValueError, match='downsample_by must be either "frequency" or "proportion"'
    ):
        transformer = DownsampleTransformer(downsample_by="invalid")
        transformer.fit_transform(X)

    with pytest.raises(ValueError, match="source_sfreq must be > target_sfreq"):
        transformer = DownsampleTransformer(
            downsample_by="frequency", source_sfreq=1.0, target_sfreq=2.0
        )
        transformer.fit_transform(X)

    with pytest.raises(
        ValueError, match="proportion must be provided and between 0-1."
    ):
        transformer = DownsampleTransformer(downsample_by="proportion", proportion=1)
        transformer.fit_transform(X)

    with pytest.raises(
        ValueError, match="proportion must be provided and between 0-1."
    ):
        transformer = DownsampleTransformer(downsample_by="proportion", proportion=0)
        transformer.fit_transform(X)

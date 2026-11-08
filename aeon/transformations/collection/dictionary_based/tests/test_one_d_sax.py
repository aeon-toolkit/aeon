"""Tests for OneDSAX (1d-SAX) transformation."""

import numpy as np
import pytest

from aeon.testing.estimator_checking import check_estimator
from aeon.transformations.collection.dictionary_based import SAX, OneDSAX


def test_one_d_sax_perfect_line():
    """Test 1d-SAX on a strictly increasing linear series with known symbols."""
    # Series increasing from 1.0 to 8.0
    X = np.arange(1.0, 9.0).reshape(1, 1, 8)
    od_sax = OneDSAX(
        n_segments=2,
        alphabet_size_avg=2,
        alphabet_size_slope=2,
        znormalized=False,
    )
    X_trans = od_sax.fit_transform(X)

    # Segment 1 ([1, 2, 3, 4]): mean < 0 (avg=0), slope > 0 (slope=1) -> 0*2 + 1 = 1
    # Segment 2 ([5, 6, 7, 8]): mean > 0 (avg=1), slope > 0 (slope=1) -> 1*2 + 1 = 3
    assert X_trans[0, 0, 0] == 1
    assert X_trans[0, 0, 1] == 3


def test_one_d_sax_robust_v_shape_distinguishes_slopes():
    """Test 1d-SAX on a V-shaped series with means and slopes far from breakpoints."""
    # Whole-series mean is 6.0, std is ~4.2426.
    # Segment 1 ([14, 8, 6, 4]): mean=8.0 (z +0.4714 > 0, avg=1), slope < 0 (slope=0).
    # Segment 2 ([0, 2, 4, 10]): mean=4.0 (z -0.4714 < 0, avg=0), slope > 0 (slope=1).
    X = np.array([[[14.0, 8.0, 6.0, 4.0, 0.0, 2.0, 4.0, 10.0]]])
    od_sax = OneDSAX(
        n_segments=2,
        alphabet_size_avg=2,
        alphabet_size_slope=2,
        znormalized=False,
    )
    X_trans = od_sax.fit_transform(X)

    # Segment 1: avg=1, slope=0 -> 1*2 + 0 = 2
    # Segment 2: avg=0, slope=1 -> 0*2 + 1 = 1
    assert X_trans[0, 0, 0] == 2
    assert X_trans[0, 0, 1] == 1

    # Standard SAX only looks at mean, producing [1, 0]
    sax = SAX(n_segments=2, alphabet_size=2, znormalized=False)
    X_sax = sax.fit_transform(X)
    assert X_sax[0, 0, 0] == 1
    assert X_sax[0, 0, 1] == 0


def test_one_d_sax_scaled_slope_breakpoints():
    """Test 1d-SAX with alphabet_size_slope=3 where sqrt(0.03/l) scaling matters."""
    # Series of length 12 with 3 segments of length 4.
    # Scaled slope breakpoints (sigma_s = sqrt(0.03/4) = 0.0866): [-0.0373, 0.0373].
    # Unscaled slope breakpoints (sigma = 1.0): [-0.4307, 0.4307].
    # Segment 0 ([8, 6, 4, 2]): mean=+1.1169 (avg=1), slope=-0.8123 < -0.0373 -> 3
    # Segment 1 ([0, 0, 0, 0]): mean=-0.9138 (avg=0), slope=0.0 in range -> 1
    # Segment 2 ([1, 1.5, 2, 2.5]): mean=-0.2031 (avg=0), slope=+0.2031 > 0.0373 -> 2
    X = np.array([[[8.0, 6.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.5]]])
    od_sax = OneDSAX(
        n_segments=3,
        alphabet_size_avg=2,
        alphabet_size_slope=3,
        znormalized=False,
    )
    X_trans = od_sax.fit_transform(X)

    assert X_trans[0, 0, 0] == 3
    assert X_trans[0, 0, 1] == 1
    assert X_trans[0, 0, 2] == 2


@pytest.mark.parametrize(
    "n_cases,n_channels,n_timepoints,n_segments,avg_size,slope_size",
    [
        (5, 1, 32, 4, 3, 3),
        (3, 2, 40, 5, 4, 2),
        (2, 3, 35, 4, 2, 5),  # Non-divisible length: 35 % 4 != 0
        (1, 1, 16, 2, 5, 10),
    ],
)
def test_one_d_sax_shapes_and_symbol_range(
    n_cases, n_channels, n_timepoints, n_segments, avg_size, slope_size
):
    """Test output shape and symbol range across diverse configurations."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(n_cases, n_channels, n_timepoints))

    od_sax = OneDSAX(
        n_segments=n_segments,
        alphabet_size_avg=avg_size,
        alphabet_size_slope=slope_size,
    )
    X_trans = od_sax.fit_transform(X)

    assert X_trans.shape == (n_cases, n_channels, n_segments)
    assert X_trans.dtype == np.int64
    max_symbol = avg_size * slope_size - 1
    assert np.all(X_trans >= 0)
    assert np.all(X_trans <= max_symbol)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_one_d_sax_input_dtypes(dtype):
    """Test that integer and floating dtypes are supported and yield int64 symbols."""
    X = np.arange(24, dtype=dtype).reshape(2, 1, 12)
    od_sax = OneDSAX(n_segments=3, alphabet_size_avg=3, alphabet_size_slope=3)
    X_trans = od_sax.fit_transform(X)

    assert X_trans.shape == (2, 1, 3)
    assert X_trans.dtype == np.int64


def test_one_d_sax_matches_sax_mean_symbols():
    """Verify that decomposing 1d-SAX symbols gives the exact standard SAX symbols."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(4, 2, 30))

    avg_size = 4
    slope_size = 3
    n_segments = 5

    od_sax = OneDSAX(
        n_segments=n_segments,
        alphabet_size_avg=avg_size,
        alphabet_size_slope=slope_size,
        znormalized=True,
    )
    sax = SAX(
        n_segments=n_segments,
        alphabet_size=avg_size,
        znormalized=True,
    )

    X_od = od_sax.fit_transform(X)
    X_sax = sax.fit_transform(X)

    # avg_symbol = composite_symbol // alphabet_size_slope
    recovered_avg_symbols = X_od // slope_size
    np.testing.assert_array_equal(recovered_avg_symbols, X_sax)


def test_one_d_sax_constant_series():
    """Test constant series with zero variance does not produce NaN or error."""
    X = np.full((3, 2, 20), 42.0)
    od_sax = OneDSAX(n_segments=4, znormalized=False)
    X_trans = od_sax.fit_transform(X)

    assert X_trans.shape == (3, 2, 4)
    assert np.all(np.isfinite(X_trans))
    assert np.all(X_trans >= 0)


def test_one_d_sax_single_point_segments():
    """Test edge case where n_segments == n_timepoints (segment_length == 1)."""
    X = np.array([[[1.0, 5.0, 2.0, 8.0]]])
    od_sax = OneDSAX(
        n_segments=4,
        alphabet_size_avg=2,
        alphabet_size_slope=2,
        znormalized=True,
    )
    X_trans = od_sax.fit_transform(X)

    assert X_trans.shape == (1, 1, 4)
    # Symbols must be valid integers in [0, 3]
    assert np.all((X_trans >= 0) & (X_trans <= 3))


def test_one_d_sax_extreme_scale_values():
    """Test robustness on very large, small, and negative numbers."""
    X_large = np.full((2, 1, 16), 1e8)
    X_large[:, :, 8:] = -1e8
    od_sax = OneDSAX(n_segments=2, alphabet_size_avg=3, alphabet_size_slope=3)
    res_large = od_sax.fit_transform(X_large)
    assert np.all(np.isfinite(res_large))

    X_small = np.linspace(-1e-6, 1e-6, 16).reshape(1, 1, 16)
    res_small = od_sax.fit_transform(X_small)
    assert np.all(np.isfinite(res_small))


def test_one_d_sax_n_segments_greater_than_timepoints():
    """Test error when n_segments > n_timepoints."""
    X = np.zeros((1, 1, 5))
    od_sax = OneDSAX(n_segments=10)
    with pytest.raises(ValueError, match="cannot be greater than the series length"):
        od_sax.fit_transform(X)


@pytest.mark.parametrize(
    "invalid_params,match",
    [
        ({"n_segments": 0}, "n_segments must be an integer >= 1"),
        ({"n_segments": "four"}, "n_segments must be an integer >= 1"),
        ({"alphabet_size_avg": 1}, "alphabet_size_avg must be an integer >= 2"),
        ({"alphabet_size_slope": 1}, "alphabet_size_slope must be an integer >= 2"),
        ({"znormalized": "yes"}, "znormalized must be a boolean"),
    ],
)
def test_one_d_sax_invalid_parameters(invalid_params, match):
    """Test parameter validation during transform / fit_transform."""
    X = np.zeros((1, 1, 8))
    od_sax = OneDSAX(**invalid_params)
    with pytest.raises(ValueError, match=match):
        od_sax.fit_transform(X)


def test_one_d_sax_set_params_validation():
    """Test that set_params cannot bypass parameter validation."""
    X = np.zeros((1, 1, 8))
    od_sax = OneDSAX(n_segments=2)
    od_sax.set_params(n_segments=-1)
    with pytest.raises(ValueError, match="n_segments must be an integer >= 1"):
        od_sax.fit_transform(X)

    # Verify set_params validation via separate fit() and transform() calls
    od_sax.set_params(n_segments=2)
    od_sax.fit(X)
    od_sax.set_params(n_segments=-2)
    with pytest.raises(ValueError, match="n_segments must be an integer >= 1"):
        od_sax.transform(X)

    od_sax.set_params(n_segments=2, alphabet_size_avg=1)
    with pytest.raises(ValueError, match="alphabet_size_avg must be an integer >= 2"):
        od_sax.fit_transform(X)

    od_sax.set_params(alphabet_size_avg=4, alphabet_size_slope=1)
    with pytest.raises(ValueError, match="alphabet_size_slope must be an integer >= 2"):
        od_sax.fit_transform(X)


def test_one_d_sax_decoding_symbols():
    """Test decoding composite symbols back into avg and slope components."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(3, 2, 16))
    slope_size = 4
    avg_size = 3
    od_sax = OneDSAX(
        n_segments=4,
        alphabet_size_avg=avg_size,
        alphabet_size_slope=slope_size,
    )
    X_trans = od_sax.fit_transform(X)

    avg_symbols = X_trans // slope_size
    slope_symbols = X_trans % slope_size

    assert np.all((avg_symbols >= 0) & (avg_symbols < avg_size))
    assert np.all((slope_symbols >= 0) & (slope_symbols < slope_size))
    recomposed = avg_symbols * slope_size + slope_symbols
    np.testing.assert_array_equal(recomposed, X_trans)


def test_one_d_sax_nan_input():
    """Test NaN input raises ValueError consistent with SAX."""
    od_sax = OneDSAX(n_segments=2)
    X_nan = np.array([[[1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]])
    with pytest.raises(ValueError, match="missing values"):
        od_sax.fit_transform(X_nan)


def test_one_d_sax_single_timepoint():
    """Test series of length 1 with n_segments=1."""
    od_sax = OneDSAX(n_segments=1, alphabet_size_avg=2, alphabet_size_slope=2)
    X_single = np.array([[[4.0]]])
    res = od_sax.fit_transform(X_single)
    assert res.shape == (1, 1, 1)
    assert 0 <= res[0, 0, 0] <= 3


def test_one_d_sax_single_segment_multiple_timepoints():
    """Test single segment (n_segments=1) on a series with multiple timepoints."""
    X = np.arange(1.0, 9.0).reshape(1, 1, 8)
    od_sax = OneDSAX(
        n_segments=1,
        alphabet_size_avg=3,
        alphabet_size_slope=3,
        znormalized=False,
    )
    res = od_sax.fit_transform(X)
    assert res.shape == (1, 1, 1)
    assert 0 <= res[0, 0, 0] <= 8


def test_one_d_sax_divisible_vs_segmented_consistency():
    """Verify that _quantize_segment returns identical symbols for single vs batch."""
    from aeon.transformations.collection.dictionary_based._one_d_sax import (
        _quantize_segment,
    )

    rng = np.random.RandomState(42)
    X = rng.normal(size=(2, 2, 12))  # 12 points, 3 segments of length 4
    n_segments = 3
    alphabet_size_avg = 4
    alphabet_size_slope = 3

    import scipy.stats

    bp_avg = scipy.stats.norm.ppf(
        np.arange(1, alphabet_size_avg, dtype=np.float64) / alphabet_size_avg
    )

    # Vectorized call across all segments
    X_reshaped = X.reshape(2, 2, n_segments, 4)
    avg_vec, slope_vec = _quantize_segment(X_reshaped, bp_avg, alphabet_size_slope)

    # Segment-by-segment call
    for s in range(n_segments):
        avg_s, slope_s = _quantize_segment(
            X[:, :, s * 4 : (s + 1) * 4], bp_avg, alphabet_size_slope
        )
        np.testing.assert_array_equal(avg_vec[:, :, s], avg_s)
        np.testing.assert_array_equal(slope_vec[:, :, s], slope_s)


def test_one_d_sax_check_estimator():
    """Run OneDSAX through aeon's standard check_estimator test suite."""
    results = check_estimator(OneDSAX)
    for check_name, status in results.items():
        assert status == "PASSED", f"check_estimator failed on {check_name}: {status}"

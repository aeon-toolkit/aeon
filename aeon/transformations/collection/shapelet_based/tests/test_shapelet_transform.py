"""Tests for the random shapelet transform."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from aeon.transformations.collection.shapelet_based._shapelet_transform import (
    _DIST,
    _VALUES,
)

_N_CASES = 12
_N_CHANNELS = 2
_N_CLASSES = 2
_N_TIMEPOINTS = 22
_MIN_UNEQUAL_TIMEPOINTS = 20
_N_UNEQUAL_LENGTHS = 3
_N_JOBS = 2
_RST_TEST_PARAMS = {
    "n_shapelet_samples": 20,
    "max_shapelets": 6,
    "batch_size": 10,
    "random_state": 0,
}


def _make_test_data(unequal_length=False):
    """Create deterministic equal- or unequal-length test data."""
    rng = np.random.RandomState(0)
    if unequal_length:
        X = [
            rng.normal(
                size=(_N_CHANNELS, _MIN_UNEQUAL_TIMEPOINTS + i % _N_UNEQUAL_LENGTHS)
            )
            for i in range(_N_CASES)
        ]
    else:
        X = rng.normal(size=(_N_CASES, _N_CHANNELS, _N_TIMEPOINTS))
    y = np.arange(_N_CASES) % _N_CLASSES
    return X, y


def _assert_same_shapelets(first, second):
    """Assert that two fitted transforms retained identical shapelets."""
    assert [shapelet[:_VALUES] for shapelet in first.shapelets_] == [
        shapelet[:_VALUES] for shapelet in second.shapelets_
    ]
    for first_shapelet, second_shapelet in zip(first.shapelets_, second.shapelets_):
        assert_array_equal(first_shapelet[_VALUES], second_shapelet[_VALUES])


def test_fit_transform_matches_fit_without_stale_distance_indices():
    """Fit-transform caching should not change fitted state or retain cache indices."""
    X, y = _make_test_data()
    cached_transformer = RandomShapeletTransform(**_RST_TEST_PARAMS)
    fitted_transformer = RandomShapeletTransform(**_RST_TEST_PARAMS).fit(X, y)

    fit_transformed = cached_transformer.fit_transform(X, y)

    assert all(shapelet[_DIST] == -1 for shapelet in cached_transformer.shapelets_)
    _assert_same_shapelets(cached_transformer, fitted_transformer)
    assert_allclose(
        fit_transformed,
        cached_transformer.transform(X),
        rtol=1e-12,
        atol=1e-12,
    )


def test_unequal_length_fit_with_process_backend():
    """Process and thread fitting should match on unequal-length input."""
    X, y = _make_test_data(unequal_length=True)

    threaded_transformer = RandomShapeletTransform(
        n_jobs=_N_JOBS, parallel_backend="threading", **_RST_TEST_PARAMS
    )
    threaded = threaded_transformer.fit_transform(X, y)
    process_transformer = RandomShapeletTransform(
        n_jobs=_N_JOBS, parallel_backend="loky", **_RST_TEST_PARAMS
    )
    process = process_transformer.fit_transform(X, y)

    assert_array_equal(process, threaded)
    _assert_same_shapelets(process_transformer, threaded_transformer)


@pytest.mark.parametrize("unequal_length", [False, True])
def test_transform_with_process_backend(unequal_length):
    """Process and thread backends should produce the same transform."""
    X, y = _make_test_data(unequal_length=unequal_length)

    transformer = RandomShapeletTransform(
        n_jobs=_N_JOBS,
        parallel_backend="threading",
        **_RST_TEST_PARAMS,
    ).fit(X, y)

    threaded = transformer.transform(X)
    transformer.parallel_backend = "loky"
    process = transformer.transform(X)

    assert_array_equal(process, threaded)


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_fit_verbosity_levels(verbose, capsys):
    """RST verbosity controls summary and per-batch fit progress."""
    X, y = _make_test_data()
    transformer = RandomShapeletTransform(verbose=verbose, **_RST_TEST_PARAMS)

    transformer.fit(X, y)
    output = capsys.readouterr().out

    if verbose == 0:
        assert output == ""
    else:
        assert "[RST] Starting fit: mode=fixed" in output
        assert "[RST] Finished fit: extracted=20/20" in output
        if verbose == 1:
            assert "[RST] Progress: extracted=" in output
            assert "[RST] Batch " not in output
        else:
            assert "[RST] Batch 1: extracted=10/20" in output
            assert "estimated_remaining=" in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "formatted_limit"),
    [(1, "1:00"), (2, "2:00"), (120, "2:00:00")],
)
def test_contract_verbosity_reports_remaining_time(
    time_limit_in_minutes, formatted_limit, capsys
):
    """RST level-two output reports short and long fit contracts."""
    X, y = _make_test_data()
    transformer = RandomShapeletTransform(
        verbose=2,
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_shapelet_samples=10,
        **_RST_TEST_PARAMS,
    )

    transformer.fit(X, y)
    output = capsys.readouterr().out

    assert f"time_limit={formatted_limit}" in output
    assert "[RST] Batch 1: extracted=10/10" in output
    assert "contract_remaining=" in output
    assert "projected_total~" in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "reports_progress"),
    [(0.0005, True), (120, False)],
)
def test_contract_level_one_progress_is_rate_limited(
    time_limit_in_minutes, reports_progress, capsys
):
    """RST level one reports contract progress only after a budget interval."""
    X, y = _make_test_data()
    transformer = RandomShapeletTransform(
        n_shapelet_samples=100,
        max_shapelets=6,
        batch_size=100,
        random_state=0,
        verbose=1,
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_shapelet_samples=100,
    )

    transformer.fit(X, y)
    output = capsys.readouterr().out

    assert ("[RST] Progress: extracted=100/100" in output) is reports_progress
    assert "[RST] Batch " not in output

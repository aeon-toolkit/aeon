"""Tests for the random shapelet transform."""

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
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
_N_JOBS = 2
_DATA_PARAMS = {
    "n_cases": _N_CASES,
    "n_channels": _N_CHANNELS,
    "n_labels": _N_CLASSES,
    "random_state": 0,
}
_RST_TEST_PARAMS = {
    "n_shapelet_samples": 20,
    "max_shapelets": 6,
    "batch_size": 10,
    "random_state": 0,
}


def _assert_same_shapelets(first, second):
    """Assert that two fitted transforms retained identical shapelets."""
    assert [shapelet[:_VALUES] for shapelet in first.shapelets] == [
        shapelet[:_VALUES] for shapelet in second.shapelets
    ]
    for first_shapelet, second_shapelet in zip(first.shapelets, second.shapelets):
        assert_array_equal(first_shapelet[_VALUES], second_shapelet[_VALUES])


def test_fit_transform_matches_fit_without_stale_distance_indices():
    """Fit-transform caching should not change fitted state or retain cache indices."""
    X, y = make_example_3d_numpy(n_timepoints=_N_TIMEPOINTS, **_DATA_PARAMS)
    cached_transformer = RandomShapeletTransform(**_RST_TEST_PARAMS)
    fitted_transformer = RandomShapeletTransform(**_RST_TEST_PARAMS).fit(X, y)

    fit_transformed = cached_transformer.fit_transform(X, y)

    assert all(shapelet[_DIST] == -1 for shapelet in cached_transformer.shapelets)
    _assert_same_shapelets(cached_transformer, fitted_transformer)
    assert_allclose(
        fit_transformed,
        cached_transformer.transform(X),
        rtol=1e-12,
        atol=1e-12,
    )


def test_refit_with_no_shapelets_discards_previous_shapelets():
    """Refitting with no candidates should discard previously retained shapelets."""
    X, y = make_example_3d_numpy(n_timepoints=_N_TIMEPOINTS, **_DATA_PARAMS)
    transformer = RandomShapeletTransform(**_RST_TEST_PARAMS).fit(X, y)
    assert len(transformer.shapelets) > 0

    transformer.set_params(n_shapelet_samples=0).fit(X, y)

    assert transformer.shapelets == []
    assert transformer.transform(X).shape == (_N_CASES, 0)


def test_unequal_length_fit_with_process_backend():
    """Process and thread fitting should match on unequal-length input."""
    X, y = make_example_3d_numpy_list(
        min_n_timepoints=_MIN_UNEQUAL_TIMEPOINTS,
        max_n_timepoints=_N_TIMEPOINTS,
        **_DATA_PARAMS,
    )

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


@pytest.mark.parametrize(
    ("data_generator", "timepoint_params"),
    [
        (make_example_3d_numpy, {"n_timepoints": _N_TIMEPOINTS}),
        (
            make_example_3d_numpy_list,
            {
                "min_n_timepoints": _MIN_UNEQUAL_TIMEPOINTS,
                "max_n_timepoints": _N_TIMEPOINTS,
            },
        ),
    ],
)
def test_transform_with_process_backend(data_generator, timepoint_params):
    """Process and thread backends should produce the same transform."""
    X, y = data_generator(**timepoint_params, **_DATA_PARAMS)

    transformer = RandomShapeletTransform(
        n_jobs=_N_JOBS,
        parallel_backend="threading",
        **_RST_TEST_PARAMS,
    ).fit(X, y)

    threaded = transformer.transform(X)
    transformer.parallel_backend = "loky"
    process = transformer.transform(X)

    assert_array_equal(process, threaded)

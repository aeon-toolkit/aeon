"""Unit tests for series transform pipeline."""

import pytest
from numpy.testing import assert_array_almost_equal

from aeon.testing.data_generation import make_example_2d_numpy_series
from aeon.transformations.series import AutoCorrelationSeriesTransformer, LogTransformer
from aeon.transformations.series.compose import SeriesId, SeriesTransformerPipeline


@pytest.mark.parametrize(
    "transformers",
    [
        LogTransformer(),
        [LogTransformer(), AutoCorrelationSeriesTransformer()],
    ],
)
def test_series_transform_pipeline(transformers):
    """Test the collection transform pipeline."""
    X = make_example_2d_numpy_series(n_timepoints=12)

    pipeline = SeriesTransformerPipeline(transformers=transformers)
    pipeline.fit(X)
    Xt = pipeline.transform(X)

    pipeline2 = SeriesTransformerPipeline(transformers=transformers)
    Xt2 = pipeline2.fit_transform(X)

    if not isinstance(transformers, list):
        transformers = [transformers]
    for t in transformers:
        X = t.fit_transform(X)

    assert_array_almost_equal(Xt, X)
    assert_array_almost_equal(Xt2, X)


def test_single_transformer_pipeline_matches_transformer():
    """A one-step pipeline has the same result as its transformer."""
    X = make_example_2d_numpy_series(n_timepoints=12)

    pipeline = SeriesTransformerPipeline(transformers=[LogTransformer()])
    Xt = pipeline.fit_transform(X)

    expected = LogTransformer().fit_transform(X)
    assert_array_almost_equal(Xt, expected)


def test_series_id_transform_and_inverse_transform_are_identity():
    """Test ``SeriesId`` is an exact identity in both transform directions."""
    X = make_example_2d_numpy_series(n_timepoints=12)

    transformer = SeriesId()
    transformer.fit(X)
    Xt = transformer.transform(X)
    Xinv = transformer.inverse_transform(Xt)

    assert_array_almost_equal(Xt, X)
    assert_array_almost_equal(Xinv, X)

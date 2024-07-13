"""Unit tests for clustering pipeline."""

__maintainer__ = ["MatthewMiddlehurst"]

import pytest
from sklearn.preprocessing import StandardScaler

from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.mock_estimators import MockCollectionTransformer
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal
from aeon.transformations.adapt import TabularToSeriesAdaptor
from aeon.transformations.collection import (
    AutocorrelationFunctionTransformer,
    HOG1DTransformer,
    PaddingTransformer,
    Tabularizer,
    TimeSeriesScaler,
)
from aeon.transformations.collection.compose import CollectionTransformerPipeline
from aeon.transformations.collection.feature_based import SevenNumberSummaryTransformer


@pytest.mark.parametrize(
    "transformers",
    [
        PaddingTransformer(pad_length=15),
        SevenNumberSummaryTransformer(),
        [TabularToSeriesAdaptor(StandardScaler())],
        [PaddingTransformer(pad_length=15), Tabularizer(), StandardScaler()],
        [PaddingTransformer(pad_length=15), SevenNumberSummaryTransformer()],
        [Tabularizer(), StandardScaler(), SevenNumberSummaryTransformer()],
        [
            TabularToSeriesAdaptor(StandardScaler()),
            PaddingTransformer(pad_length=15),
            SevenNumberSummaryTransformer(),
        ],
    ],
)
def test_collection_transform_pipeline(transformers):
    """Test the collection transform pipeline."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    pipeline = CollectionTransformerPipeline(transformers=transformers)
    pipeline.fit(X, y)
    Xt = pipeline.transform(X)

    if not isinstance(transformers, list):
        transformers = [transformers]

    for t in transformers:
        X = t.fit_transform(X, y)

    _assert_array_almost_equal(Xt, X)


def test_unequal_tag_inference():
    """Test that CollectionTransformerPipeline infers unequal length tag correctly."""
    X, y = make_example_3d_numpy_list(
        n_cases=10, min_n_timepoints=8, max_n_timepoints=12
    )

    t1 = SevenNumberSummaryTransformer()
    t2 = PaddingTransformer()
    t3 = TimeSeriesScaler()
    t4 = AutocorrelationFunctionTransformer(n_lags=5)
    t5 = StandardScaler()
    t6 = Tabularizer()

    assert t1.get_tag("capability:unequal_length")
    assert t1.get_tag("output_data_type") == "Tabular"
    assert t2.get_tag("capability:unequal_length")
    assert t2.get_tag("capability:unequal_length:removes")
    assert not t2.get_tag("output_data_type") == "Tabular"
    assert t3.get_tag("capability:unequal_length")
    assert not t3.get_tag("capability:unequal_length:removes")
    assert not t3.get_tag("output_data_type") == "Tabular"
    assert not t4.get_tag("capability:unequal_length")

    # all handle unequal length
    p1 = CollectionTransformerPipeline(transformers=t3)
    assert p1.get_tag("capability:unequal_length")
    p1.fit(X, y)

    # transformer chain removes unequal length
    p2 = CollectionTransformerPipeline(transformers=[t3, t2])
    assert p2.get_tag("capability:unequal_length")
    p2.fit(X, y)

    # transformer chain removes unequal length (sklearn)
    p3 = CollectionTransformerPipeline(transformers=[t3, t2, t6, t5])
    assert p3.get_tag("capability:unequal_length")
    p3.fit(X, y)

    # transformers handle unequal length and output is tabular
    p4 = CollectionTransformerPipeline(transformers=[t3, t1])
    assert p4.get_tag("capability:unequal_length")
    p4.fit(X, y)

    # test they fit even if they cannot handle unequal length
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    #  transformer does not unequal length
    p5 = CollectionTransformerPipeline(transformers=t4)
    assert not p5.get_tag("capability:unequal_length")
    p5.fit(X, y)

    # transformer removes unequal length but prior cannot handle
    p6 = CollectionTransformerPipeline(transformers=[t4, t2])
    assert not p6.get_tag("capability:unequal_length")
    p6.fit(X, y)


def test_missing_tag_inference():
    """Test that CollectionTransformerPipeline infers missing data tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    t1 = MockCollectionTransformer()
    t1.set_tags(
        **{"capability:missing_values": True, "capability:missing_values:removes": True}
    )
    t2 = TimeSeriesScaler()
    t3 = StandardScaler()
    t4 = Tabularizer()

    assert t1.get_tag("capability:missing_values")
    assert t1.get_tag("capability:missing_values:removes")
    assert not t2.get_tag("capability:missing_values")

    # transformer chain removes missing values
    p1 = CollectionTransformerPipeline(transformers=t1)
    assert p1.get_tag("capability:missing_values")
    p1.fit(X, y)

    # transformer removes missing values(sklearn)
    p2 = CollectionTransformerPipeline(transformers=[t1, t4, t3])
    assert p2.get_tag("capability:missing_values")
    p2.fit(X, y)

    # test they fit even if they cannot handle missing data
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # transformers cannot handle missing data
    p3 = CollectionTransformerPipeline(transformers=t2)
    assert not p3.get_tag("capability:missing_values")
    p3.fit(X, y)

    # transformer removes missing values but prior cannot handle
    p5 = CollectionTransformerPipeline(transformers=[t2, t1])
    assert not p5.get_tag("capability:missing_values")
    p5.fit(X, y)


def test_multivariate_tag_inference():
    """Test that CollectionTransformerPipeline infers multivariate tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=12)

    t1 = SevenNumberSummaryTransformer()
    t2 = TimeSeriesScaler()
    t3 = HOG1DTransformer()
    t4 = StandardScaler()

    assert t1.get_tag("capability:multivariate")
    assert t1.get_tag("output_data_type") == "Tabular"
    assert t2.get_tag("capability:multivariate")
    assert not t2.get_tag("output_data_type") == "Tabular"
    assert not t3.get_tag("capability:multivariate")

    # all handle multivariate
    p1 = CollectionTransformerPipeline(transformers=t2)
    assert p1.get_tag("capability:multivariate")
    p1.fit(X, y)

    # transformers handle multivariate and output is tabular
    p2 = CollectionTransformerPipeline(transformers=[t1, t4])
    assert p2.get_tag("capability:multivariate")
    p2.fit(X, y)

    # test they fit even if they cannot handle multivariate
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    # transformer does not handle multivariate
    p3 = CollectionTransformerPipeline(transformers=t3)
    assert not p3.get_tag("capability:multivariate")
    p3.fit(X, y)

    # transformer converts multivariate to tabular but prior cannot handle
    p4 = CollectionTransformerPipeline(transformers=[t3, t1])
    assert not p4.get_tag("capability:multivariate")
    p4.fit(X, y)

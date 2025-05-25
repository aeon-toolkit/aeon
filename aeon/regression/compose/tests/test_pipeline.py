"""Unit tests for regression pipeline."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from aeon.regression import DummyRegressor
from aeon.regression.compose import RegressorPipeline
from aeon.regression.convolution_based import RocketRegressor
from aeon.regression.tests.test_base import _TestRegressor
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.testing.mock_estimators import MockCollectionTransformer, MockRegressor
from aeon.transformations.collection import (
    AutocorrelationFunctionTransformer,
    HOG1DTransformer,
    Normalizer,
    Padder,
    Tabularizer,
)
from aeon.transformations.collection.feature_based import SevenNumberSummary


@pytest.mark.parametrize(
    "transformers",
    [
        Padder(pad_length=15),
        SevenNumberSummary(),
        [Padder(pad_length=15), Tabularizer(), StandardScaler()],
        [Padder(pad_length=15), SevenNumberSummary()],
        [Tabularizer(), StandardScaler(), SevenNumberSummary()],
        [
            Padder(pad_length=15),
            SevenNumberSummary(),
        ],
    ],
)
def test_regressor_pipeline(transformers):
    """Test the regressor pipeline."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    r = DummyRegressor()
    pipeline = RegressorPipeline(transformers=transformers, regressor=r)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    assert isinstance(y_pred, np.ndarray)

    if not isinstance(transformers, list):
        transformers = [transformers]

    for t in transformers:
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

    r.fit(X_train, y_train)
    assert_array_almost_equal(y_pred, r.predict(X_test))


@pytest.mark.parametrize(
    "transformers",
    [
        [Padder(pad_length=15), Tabularizer()],
        SevenNumberSummary(),
        [Tabularizer(), StandardScaler()],
        [Padder(pad_length=15), Tabularizer(), StandardScaler()],
        [Padder(pad_length=15), SevenNumberSummary()],
        [Tabularizer(), StandardScaler(), SevenNumberSummary()],
        [
            Padder(pad_length=15),
            SevenNumberSummary(),
        ],
    ],
)
def test_sklearn_regressor_pipeline(transformers):
    """Test regressor pipeline with sklearn estimator."""
    X_train, y_train = make_example_3d_numpy(n_cases=10, n_timepoints=12)
    X_test, _ = make_example_3d_numpy(n_cases=10, n_timepoints=12)

    r = RandomForestRegressor(n_estimators=2, random_state=0)
    pipeline = RegressorPipeline(transformers=transformers, regressor=r)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    assert isinstance(y_pred, np.ndarray)

    if not isinstance(transformers, list):
        transformers = [transformers]

    for t in transformers:
        X_train = t.fit_transform(X_train)
        X_test = t.transform(X_test)

    r.fit(X_train, y_train)
    assert_array_almost_equal(y_pred, r.predict(X_test))


def test_unequal_tag_inference():
    """Test that RegressorPipeline infers unequal length tag correctly."""
    X, y = make_example_3d_numpy_list(
        n_cases=10, min_n_timepoints=8, max_n_timepoints=12, regression_target=True
    )

    t1 = SevenNumberSummary()
    t2 = Padder()
    t3 = Normalizer()
    t4 = AutocorrelationFunctionTransformer(n_lags=5)
    t5 = StandardScaler()
    t6 = Tabularizer()

    assert t1.get_tag("capability:unequal_length")
    assert t1.get_tag("output_data_type") == "Tabular"
    assert t2.get_tag("capability:unequal_length")
    assert t2.get_tag("removes_unequal_length")
    assert not t2.get_tag("output_data_type") == "Tabular"
    assert t3.get_tag("capability:unequal_length")
    assert not t3.get_tag("removes_unequal_length")
    assert not t3.get_tag("output_data_type") == "Tabular"
    assert not t4.get_tag("capability:unequal_length")

    c1 = DummyRegressor()
    c2 = MockRegressor()
    c3 = RandomForestRegressor(n_estimators=2)

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")

    # all handle unequal length
    p1 = RegressorPipeline(transformers=t3, regressor=c1)
    assert p1.get_tag("capability:unequal_length")
    p1.fit(X, y)

    # regressor does not handle unequal length but transformer chain removes
    p2 = RegressorPipeline(transformers=[t3, t2], regressor=c2)
    assert p2.get_tag("capability:unequal_length")
    p2.fit(X, y)

    # regressor does not handle unequal length but transformer chain removes (sklearn)
    p3 = RegressorPipeline(transformers=[t3, t2, t6, t5], regressor=c3)
    assert p3.get_tag("capability:unequal_length")
    p3.fit(X, y)

    # transformers handle unequal length and output is tabular
    p4 = RegressorPipeline(transformers=[t3, t1], regressor=c3)
    assert p4.get_tag("capability:unequal_length")
    p4.fit(X, y)

    # test they fit even if they cannot handle unequal length
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)

    # transformers handle unequal length but regressor does not
    p5 = RegressorPipeline(transformers=t3, regressor=c2)
    assert not p5.get_tag("capability:unequal_length")
    p5.fit(X, y)

    # regressor handles unequal length but transformer does not
    p6 = RegressorPipeline(transformers=t4, regressor=c1)
    assert not p6.get_tag("capability:unequal_length")
    p6.fit(X, y)

    # transformer removes unequal length but prior cannot handle
    p7 = RegressorPipeline(transformers=[t4, t2], regressor=c1)
    assert not p7.get_tag("capability:unequal_length")
    p7.fit(X, y)


def test_missing_tag_inference():
    """Test that RegressorPipeline infers missing data tag correctly."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)
    # tags are reset so this causes a crash due to t1
    # X[5, 0, 4] = np.nan

    t1 = MockCollectionTransformer()
    t1.set_tags(**{"capability:missing_values": True, "removes_missing_values": True})
    t2 = Normalizer()
    t3 = StandardScaler()
    t4 = Tabularizer()

    assert t1.get_tag("capability:missing_values")
    assert t1.get_tag("removes_missing_values")
    assert not t2.get_tag("capability:missing_values")

    c1 = DummyRegressor()
    c2 = RocketRegressor(n_kernels=5)
    c3 = RandomForestRegressor(n_estimators=2)

    assert c1.get_tag("capability:missing_values")
    assert not c2.get_tag("capability:missing_values")

    # regressor does not handle missing values but transformer chain removes
    p1 = RegressorPipeline(transformers=t1, regressor=c2)
    assert p1.get_tag("capability:missing_values")
    p1.fit(X, y)

    # regressor does not handle missing values but transformer chain removes (sklearn)
    p2 = RegressorPipeline(transformers=[t1, t4, t3], regressor=c3)
    assert p2.get_tag("capability:missing_values")
    p2.fit(X, y)

    # test they fit even if they cannot handle missing data
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)

    # transformers cannot handle missing data but regressor does
    p3 = RegressorPipeline(transformers=t2, regressor=c1)
    assert not p3.get_tag("capability:missing_values")
    p3.fit(X, y)

    # transformers and regressor cannot handle missing data
    p4 = RegressorPipeline(transformers=t2, regressor=c2)
    assert not p4.get_tag("capability:missing_values")
    p4.fit(X, y)

    # transformer removes missing values but prior cannot handle
    p5 = RegressorPipeline(transformers=[t2, t1], regressor=c1)
    assert not p5.get_tag("capability:missing_values")
    p5.fit(X, y)


def test_multivariate_tag_inference():
    """Test that RegressorPipeline infers multivariate tag correctly."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=12, regression_target=True
    )

    t1 = SevenNumberSummary()
    t2 = Normalizer()
    t3 = HOG1DTransformer()
    t4 = StandardScaler()

    assert t1.get_tag("capability:multivariate")
    assert t1.get_tag("output_data_type") == "Tabular"
    assert t2.get_tag("capability:multivariate")
    assert not t2.get_tag("output_data_type") == "Tabular"
    assert not t3.get_tag("capability:multivariate")

    c1 = DummyRegressor()
    c2 = _TestRegressor()
    c3 = RandomForestRegressor(n_estimators=2)

    assert c1.get_tag("capability:multivariate")
    assert not c2.get_tag("capability:multivariate")

    # all handle multivariate
    p1 = RegressorPipeline(transformers=t2, regressor=c1)
    assert p1.get_tag("capability:multivariate")
    p1.fit(X, y)

    # transformers handle multivariate and output is tabular
    p2 = RegressorPipeline(transformers=[t1, t4], regressor=c3)
    assert p2.get_tag("capability:multivariate")
    p2.fit(X, y)

    # test they fit even if they cannot handle multivariate
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=12, regression_target=True)

    # transformers handle multivariate but regressor does not
    p3 = RegressorPipeline(transformers=t2, regressor=c2)
    assert not p3.get_tag("capability:multivariate")
    p3.fit(X, y)

    # regressor handles multivariate but transformer does not
    p4 = RegressorPipeline(transformers=t3, regressor=c1)
    assert not p4.get_tag("capability:multivariate")
    p4.fit(X, y)

    # transformer converts multivariate to tabular but prior cannot handle
    p5 = RegressorPipeline(transformers=[t3, t1], regressor=c3)
    assert not p5.get_tag("capability:multivariate")
    p5.fit(X, y)

"""ShapeDTW test code."""

import pytest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer

from aeon.classification.distance_based import ShapeDTW
from aeon.datasets import load_unit_test
from aeon.transformations.collection.dictionary_based._paa import PAA
from aeon.transformations.collection.dwt import DWTTransformer

# Transformers
from aeon.transformations.collection.hog1d import HOG1DTransformer
from aeon.transformations.collection.slope import SlopeTransformer


def test_shape_dtw_compound():
    """Test of ShapeDTW compound."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train BOSS
    dtw = ShapeDTW(
        shape_descriptor_function="compound",
    )
    dtw.fit(X_train, y_train)

    # test train estimate
    preds = dtw.predict(X_train)
    assert accuracy_score(y_train, preds) >= 0.6


def test__get_transformer():
    """Test get_transformer.

    _get_transformer requires self._metric_params, so only call after fit or in
    fit after these
        lines of code
        if self.metric_params is None:
            self._metric_params = {}
        else:
            self._metric_params = self.metric_params

    """
    s = ShapeDTW()
    s._metric_params = {}
    assert s._get_transformer("raw") is None
    assert isinstance(s._get_transformer("paa"), PAA)
    assert isinstance(s._get_transformer("dwt"), DWTTransformer)
    assert isinstance(s._get_transformer("SLOPE"), SlopeTransformer)
    assert isinstance(s._get_transformer("derivative"), FunctionTransformer)
    assert isinstance(s._get_transformer("hog1d"), HOG1DTransformer)
    with pytest.raises(ValueError, match="Invalid shape descriptor function"):
        s._get_transformer("FOO")


def test__get_hog_transformer():
    """Test _get_hog_transformer."""
    s = ShapeDTW()
    params = {}
    hog = s._get_hog_transformer(params)
    # None set
    # Default values for hog transformer in v0.4.0, if defaults change this will fail.
    assert hog.n_intervals == 2 and hog.n_bins == 8 and hog.scaling_factor == 0.1
    # One set. This parameter name is not consistent with others
    params = {"scaling_factor_hog1d": 0.5}
    hog = s._get_hog_transformer(params)
    assert hog.scaling_factor == 0.5
    params = {"num_bins_hog1d": 5}
    hog = s._get_hog_transformer(params)
    assert hog.n_bins == 5
    params = {"num_intervals_hog1d": 4}
    hog = s._get_hog_transformer(params)
    assert hog.n_intervals == 4
    # Two set
    params = {"scaling_factor_hog1d": 0.25, "num_bins_hog1d": 3}
    hog = s._get_hog_transformer(params)
    assert hog.scaling_factor == 0.25 and hog.n_bins == 3
    params = {"scaling_factor_hog1d": 0.35, "num_intervals_hog1d": 3}
    hog = s._get_hog_transformer(params)
    assert hog.scaling_factor == 0.35 and hog.n_intervals == 3
    params = {"num_bins_hog1d": 4, "num_intervals_hog1d": 3}
    hog = s._get_hog_transformer(params)
    assert hog.n_bins == 4 and hog.n_intervals == 3
    params = {
        "scaling_factor_hog1d": 0.6,
        "num_bins_hog1d": 4,
        "num_intervals_hog1d": 3,
    }
    hog = s._get_hog_transformer(params)
    assert hog.n_bins == 4 and hog.n_intervals == 3 and hog.scaling_factor == 0.6

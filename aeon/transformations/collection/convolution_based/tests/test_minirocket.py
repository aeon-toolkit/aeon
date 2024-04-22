"""MiniRocketMultivariateVariable test code."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.datasets import load_japanese_vowels
from aeon.transformations.collection.convolution_based import (
    MiniRocketMultivariateVariable,
)
from aeon.transformations.collection.convolution_based._minirocket import (
    _PPV,
    MiniRocket,
    _fit_dilations,
)
from aeon.transformations.collection.convolution_based._minirocket_mv import (
    _np_list_transposed2D_array_and_len_list,
)


def test_minirocket_multivariate_variable_on_japanese_vowels():
    """Test of MiniRocketMultivariate on japanese vowels."""
    # load training data
    X_training, Y_training = load_japanese_vowels(split="train")

    # 'fit' MINIROCKET -> infer data dimensions, generate random kernels
    num_kernels = 10_000
    minirocket_mv_var = MiniRocketMultivariateVariable(
        num_kernels=num_kernels,
        pad_value_short_series=0,
        reference_length="max",
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state=42,
    )
    minirocket_mv_var.fit(X_training)

    # transform training data
    X_training_transform = minirocket_mv_var.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 1000)
    np.testing.assert_equal(
        X_training_transform.shape, (len(X_training), 84 * (num_kernels // 84))
    )

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_japanese_vowels(split="test")

    # transform test data
    X_test_transform = minirocket_mv_var.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(
        X_test_transform.shape, (len(X_test), 84 * (num_kernels // 84))
    )

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test accuracy, mean usually .987, and minimum .983
    assert accuracy > 0.97, "Test accuracy should be greater than 0.97"


x = np.random.random(size=(2, 9))
arr2 = [x, x, x]
arr3 = [np.random.random(size=(2, 6)), x, x]
arr4 = [
    np.random.random(size=(2, 7)),
    np.random.random(size=(2, 8)),
    np.random.random(size=(2, 6)),
]
TEST_DATA = [np.random.random(size=(3, 2, 9)), arr2, arr3, arr4]


@pytest.mark.parametrize("data", TEST_DATA)
def test__np_list_transposed2D_array_and_len_list(data):
    """Test the concatenation by channel works correctly."""
    trans, lengths = _np_list_transposed2D_array_and_len_list(data)
    assert isinstance(trans, np.ndarray)
    assert trans.shape == (2, 27)
    assert len(lengths) == 3 and sum(lengths) == 27


def test__fit_dilations():
    """Test for fitting the dilations."""
    dilations, features_per_dilation = _fit_dilations(32, 168, 6)
    assert np.array_equal(dilations, np.array([1, 3]))
    assert np.array_equal(features_per_dilation, np.array([1, 1]))
    dilations, features_per_dilation = _fit_dilations(32, 1680, 6)
    assert np.array_equal(dilations, np.array([1, 2, 3]))
    assert np.array_equal(features_per_dilation, np.array([11, 6, 3]))
    assert _PPV(np.float32(10.0), np.float32(0.0)) == 1
    assert _PPV(np.float32(-110.0), np.float32(0.0)) == 0


def test_wrong_input():
    """Test for parsing a wrong input."""
    arr = np.random.random(size=(10, 1, 8))
    mini = MiniRocket()
    with pytest.raises(ValueError, match="n_timepoints must be >= 9"):
        mini.fit(arr)

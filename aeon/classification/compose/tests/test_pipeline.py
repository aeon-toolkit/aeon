"""Unit tests for (dunder) composition functionality attached to the base class."""

__author__ = ["fkiraly", "TonyBagnall"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from aeon.classification import DummyClassifier
from aeon.classification.compose import ClassifierPipeline, SklearnClassifierPipeline
from aeon.classification.convolution_based import RocketClassifier
from aeon.testing.utils.collection import make_3d_test_data
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal
from aeon.transformations.collection import PaddingTransformer
from aeon.transformations.exponent import ExponentTransformer
from aeon.transformations.impute import Imputer


def test_classifier_pipeline():
    """Test the classifier pipeline."""
    RAND_SEED = 42
    X_train, y_train = make_3d_test_data(
        n_cases=10, n_timepoints=12, random_state=RAND_SEED
    )
    X_test, _ = make_3d_test_data(n_cases=10, n_timepoints=12, random_state=RAND_SEED)

    t1 = ExponentTransformer(power=4)
    t2 = ExponentTransformer(power=0.25)
    c = DummyClassifier()
    pipeline = ClassifierPipeline(transformers=[t1, t2], classifier=c)

    y_pred = pipeline.fit(X_train, y_train).predict(X_test)
    assert isinstance(y_pred, np.ndarray)


def test_sklearn_classifier_pipeline():
    """Test auto-adapter for sklearn in mul."""
    RAND_SEED = 42
    X_train, y_train = make_3d_test_data(
        n_cases=10, n_timepoints=12, random_state=RAND_SEED
    )
    X_test, _ = make_3d_test_data(n_cases=10, n_timepoints=12, random_state=RAND_SEED)

    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    c = RandomForestClassifier(n_estimators=2)
    pipeline = SklearnClassifierPipeline(transformers=[t1, t2], classifier=c)

    y_pred = pipeline.fit(X_train, y_train).predict(X_test)
    assert isinstance(y_pred, np.ndarray)


def test_missing_unequal_tag_inference():
    """Test that ClassifierPipeline infers missing/unequal tags correctly."""
    t1 = ExponentTransformer(power=4)
    t2 = PaddingTransformer()
    t3 = Imputer()
    c1 = RocketClassifier(num_kernels=100)
    c2 = DummyClassifier()

    p1 = ClassifierPipeline(transformers=[t1, t2], classifier=c1)
    p2 = ClassifierPipeline(transformers=[t1, t1], classifier=c1)
    p3 = ClassifierPipeline(transformers=[t3, t1], classifier=c1)
    p4 = ClassifierPipeline(transformers=[t1, t3], classifier=c1)

    assert c1.get_tag("capability:unequal_length")
    assert not c2.get_tag("capability:unequal_length")
    assert c3.get_tag("capability:missing_values")
    assert not c4.get_tag("capability:missing_values")

    # assert t1 t1 with dummy has unequal
    # assert t1 t3 with dummy has missing


def test_operator_overload():
    """Test auto-adapter for sklearn in mul."""
    RAND_SEED = 42
    X_train, y_train = make_3d_test_data(
        n_cases=10, n_timepoints=12, random_state=RAND_SEED
    )
    X_test, _ = make_3d_test_data(n_cases=10, n_timepoints=12, random_state=RAND_SEED)

    t1 = ExponentTransformer(power=2)
    t2 = StandardScaler()
    c = DummyClassifier()

    t1 = t1 * c
    t2 = t1 * (t2 * c)
    t3 = (t1 * t2) * c
    t4 = t1 * t2 * c

    assert isinstance(t1, ClassifierPipeline)
    assert isinstance(t2, ClassifierPipeline)
    assert isinstance(t3, ClassifierPipeline)
    assert isinstance(t4, ClassifierPipeline)

    # assert single transformer works
    # assert regular is the same as these three

    y_pred = t12c_1.fit(X, y).predict(X_test)

    _assert_array_almost_equal(y_pred, t12c_2.fit(X_train, y_train).predict(X_test))
    _assert_array_almost_equal(y_pred, t12c_3.fit(X_train, y_train).predict(X_test))


# series-to-primitive transformer sklearn
# collection transformer
# collection transformer sklearn

# t1 = ExponentTransformer(power=2)
# t2 = SummaryTransformer()
# c = KNeighborsClassifier()
# params2 = {"transformers": [t1, t2], "classifier": c}
